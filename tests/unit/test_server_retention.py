"""Bounded job/upload retention, restart scavenging, and disk-pressure errors."""

from __future__ import annotations

import json
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord
from hawavoclean.server.retention import (
    UPLOAD_MARKER,
    DiskUsage,
    StoragePressureError,
    UploadStore,
)

pytestmark = pytest.mark.unit
TOKEN = "retention-token"
HEADERS = {"X-Hawa-Token": TOKEN}


def _success(record: JobRecord) -> list[str]:
    report = {"schema_version": 1, "summary": {"units_total": 1}}
    script = f"""
        import json, pathlib, sys
        pathlib.Path({str(record.output_path)!r}).write_bytes(b'RIFF')
        pathlib.Path({str(record.report_path)!r}).write_text({json.dumps(json.dumps(report))})
        print(json.dumps({{"event":"done","report_path":{str(record.report_path)!r}}}))
    """
    return [sys.executable, "-u", "-c", textwrap.dedent(script)]


def _client(work: Path, manager: JobManager, **kwargs: Any) -> Iterator[TestClient]:
    assert work.is_dir()
    app = create_app(
        TOKEN,
        None,
        job_manager=manager,
        on_shutdown=lambda: None,
        min_free_bytes=0,
        **kwargs,
    )
    with TestClient(app) as client:
        yield client


def _wait_api_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}", headers=HEADERS)
        assert response.status_code == 200
        body: dict[str, object] = response.json()
        if body["state"] in TERMINAL_STATES:
            return body
        time.sleep(0.02)
    raise AssertionError("job did not become terminal")


def test_cleanup_removes_only_the_marked_input_not_sibling_output(tmp_path: Path) -> None:
    store = UploadStore(tmp_path / "uploads", min_free_bytes=0)
    input_path = store.stage("source.wav")
    input_path.write_bytes(b"input")
    output = input_path.parent / "source_clean.wav"
    output.write_bytes(b"committed output")

    assert store.cleanup_input(input_path) is True
    assert not input_path.exists()
    assert not (input_path.parent / UPLOAD_MARKER).exists()
    assert output.read_bytes() == b"committed output"
    assert store.cleanup_input(input_path) is False


def test_fake_clock_scavenging_protects_active_and_unknown_files(tmp_path: Path) -> None:
    now = [0.0]
    store = UploadStore(tmp_path / "uploads", ttl_s=10, min_free_bytes=0, clock=lambda: now[0])
    active = store.stage("active.wav")
    expired = store.stage("expired.wav")
    active.write_bytes(b"a")
    expired.write_bytes(b"b")
    unknown = store.root / ("f" * 32)
    unknown.mkdir()
    (unknown / "keep.wav").write_bytes(b"user-owned")

    now[0] = 10.0
    assert store.scavenge([active]) == 1
    assert active.exists()
    assert not expired.exists()
    assert (unknown / "keep.wav").read_bytes() == b"user-owned"


def test_marker_commit_failure_leaves_no_unowned_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = UploadStore(tmp_path / "uploads", min_free_bytes=0)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated marker commit failure")

    monkeypatch.setattr("hawavoclean.server.retention.os.replace", fail_replace)
    with pytest.raises(OSError, match="marker commit failure"):
        store.stage("source.wav")
    assert list(store.root.iterdir()) == []


@pytest.mark.parametrize(
    "payload",
    [
        "{not-json",
        "[]",
        '{"schema_version":1}',
        '{"schema_version":2,"created_epoch":0,"input_name":"x"}',
    ],
)
def test_malformed_or_unknown_markers_are_never_followed(tmp_path: Path, payload: str) -> None:
    store = UploadStore(tmp_path / "uploads", min_free_bytes=0)
    directory = store.root / ("a" * 32)
    directory.mkdir()
    (directory / UPLOAD_MARKER).write_text(payload, encoding="utf-8")

    assert store._read_marker(directory) is None
    assert store.scavenge() == 0
    assert directory.exists()


def test_progress_checks_and_names_fail_closed(tmp_path: Path) -> None:
    usage = DiskUsage(total=100, used=96, free=4)
    store = UploadStore(
        tmp_path / "uploads",
        max_total_bytes=10,
        min_free_bytes=5,
        disk_usage=lambda _path: usage,
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        store.ensure_capacity(-1, 0)
    with pytest.raises(StoragePressureError, match="total quota"):
        store.ensure_progress(5, 6)
    with pytest.raises(StoragePressureError, match="below"):
        store.ensure_progress(0, 1)
    for invalid in ("", "../escape.wav", "nested/input.wav"):
        with pytest.raises(ValueError, match="safe basename"):
            store.stage(invalid)


def test_total_quota_and_free_space_reserve_fail_closed(tmp_path: Path) -> None:
    usage = DiskUsage(total=1000, used=900, free=100)
    store = UploadStore(
        tmp_path / "uploads",
        max_total_bytes=100,
        min_free_bytes=50,
        disk_usage=lambda _path: usage,
    )

    store.ensure_capacity(20, 30)
    with pytest.raises(StoragePressureError, match="total quota"):
        store.ensure_capacity(90, 11)
    with pytest.raises(StoragePressureError, match="free-space reserve"):
        store.ensure_capacity(0, 51)


@pytest.mark.parametrize(
    ("ttl", "total", "reserve"),
    [(0.0, 1, 0), (-1.0, 1, 0), (1.0, 0, 0), (1.0, 1, -1)],
)
def test_upload_retention_cannot_be_configured_unbounded(
    tmp_path: Path, ttl: float, total: int, reserve: int
) -> None:
    with pytest.raises(ValueError):
        UploadStore(tmp_path / "uploads", ttl_s=ttl, max_total_bytes=total, min_free_bytes=reserve)


def test_api_reports_total_quota_pressure_without_leaving_a_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    manager = JobManager()
    try:
        for client in _client(tmp_path, manager, max_upload_total_bytes=16):
            response = client.post(
                "/api/upload",
                headers=HEADERS,
                files={"file": ("too-big.wav", b"x" * 32)},
            )
            assert response.status_code == 507
            assert response.json()["error"] == "insufficient_storage"
            assert list((tmp_path / "uploads").iterdir()) == []
    finally:
        manager.shutdown()


def test_api_reports_a_full_job_queue_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    manager = JobManager(
        command_factory=lambda _record: [sys.executable, "-c", "import time; time.sleep(60)"],
        max_active_jobs=1,
    )
    try:
        for client in _client(tmp_path, manager):
            first = client.post(
                "/api/jobs",
                headers=HEADERS,
                json={"input_path": str(source), "profile": "production"},
            )
            assert first.status_code == 202
            second = client.post(
                "/api/jobs",
                headers=HEADERS,
                json={"input_path": str(source), "profile": "production"},
            )
            assert second.status_code == 503
            assert second.json()["error"] == "queue_full"
    finally:
        manager.shutdown(grace_s=0.2)


def test_terminal_job_cleans_upload_input_but_keeps_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    manager = JobManager(command_factory=_success)
    try:
        for client in _client(tmp_path, manager):
            uploaded = client.post(
                "/api/upload", headers=HEADERS, files={"file": ("voice.wav", b"audio")}
            )
            assert uploaded.status_code == 200
            input_path = Path(uploaded.json()["path"])
            submitted = client.post(
                "/api/jobs",
                headers=HEADERS,
                json={"input_path": str(input_path), "profile": "production"},
            )
            assert submitted.status_code == 202
            body = _wait_api_terminal(client, submitted.json()["job_id"])
            assert body["state"] == "done"
            assert not input_path.exists()
            assert Path(str(body["output_path"])).read_bytes() == b"RIFF"
            assert Path(str(body["report_path"])).is_file()
            assert not (input_path.parent / UPLOAD_MARKER).exists()
    finally:
        manager.shutdown()


def test_restart_scavenges_expired_input_and_preserves_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    old = UploadStore(tmp_path / "uploads", ttl_s=10, min_free_bytes=0, clock=lambda: 0.0)
    input_path = old.stage("old.wav")
    input_path.write_bytes(b"old input")
    output = input_path.parent / "old_clean.wav"
    output.write_bytes(b"published")

    manager = JobManager()
    try:
        app = create_app(
            TOKEN,
            None,
            job_manager=manager,
            on_shutdown=lambda: None,
            upload_ttl_s=10,
            min_free_bytes=0,
            retention_clock=lambda: 11.0,
        )
        assert not input_path.exists()
        assert output.read_bytes() == b"published"
        assert not (input_path.parent / UPLOAD_MARKER).exists()
        assert app.state.upload_store.usage_bytes() == 0
    finally:
        manager.shutdown()
