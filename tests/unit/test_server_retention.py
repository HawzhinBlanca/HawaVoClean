"""Bounded job/upload retention, restart scavenging, and disk-pressure errors."""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
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


def _absent(path: Path) -> Callable[[], bool]:
    """A predicate for :func:`_wait_until`, bound now rather than at call time."""

    def gone() -> bool:
        return not path.exists()

    return gone


def _wait_until(predicate: Callable[[], bool], message: str, timeout: float = 10.0) -> None:
    """Wait for something a terminal callback does after the job lock is released."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"{message} within {timeout:.0f}s")


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
    with TestClient(app, base_url="http://127.0.0.1") as client:
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


def test_source_ids_are_opaque_marker_bound_and_fail_closed(tmp_path: Path) -> None:
    store = UploadStore(tmp_path / "uploads", min_free_bytes=0)
    input_path = store.stage("source.wav")
    input_path.write_bytes(b"input")
    source_id = store.source_id(input_path)

    assert len(source_id) == 32
    assert store.resolve_source(source_id) == input_path
    assert store.resolve_source("../" + source_id) is None
    assert store.resolve_source("not-an-id") is None

    input_path.unlink()
    input_path.symlink_to("outside.wav")
    assert store.resolve_source(source_id) is None
    with pytest.raises(ValueError, match="not owned"):
        store.source_id(tmp_path / "unmanaged.wav")


def test_source_lease_closes_resolve_then_cleanup_race(tmp_path: Path) -> None:
    now = [0.0]
    store = UploadStore(
        tmp_path / "uploads",
        ttl_s=1.0,
        min_free_bytes=0,
        clock=lambda: now[0],
    )
    input_path = store.stage("source.wav")
    input_path.write_bytes(b"input")
    source_id = store.source_id(input_path)
    now[0] = 2.0

    with store.lease_source(source_id) as leased:
        assert leased == input_path.resolve()
        assert store.scavenge() == 0
        assert store.cleanup_input(input_path) is False
        assert input_path.read_bytes() == b"input"

    assert store.scavenge() == 1
    assert not input_path.exists()


def test_unknown_source_lease_is_a_noop(tmp_path: Path) -> None:
    store = UploadStore(tmp_path / "uploads", min_free_bytes=0)
    with store.lease_source("not-an-id") as leased:
        assert leased is None


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
            # Cleanup is deliberately NOT synchronous with the job reaching
            # terminal: 2143062 moved terminal callbacks outside the job lock,
            # because asking the manager whether another job still needs an
            # upload deadlocked inside it. So a client can see "done" a moment
            # before the upload is gone, and this waits for it instead of
            # assuming the old ordering. It raced once in a full-suite run and
            # passed three times alone.
            _wait_until(_absent(input_path), "the upload was never cleaned up")
            assert Path(str(body["output_path"])).read_bytes() == b"RIFF"
            assert Path(str(body["report_path"])).is_file()
            _wait_until(
                _absent(input_path.parent / UPLOAD_MARKER),
                "the upload marker was never removed",
            )
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


@pytest.mark.unit
def test_a_finished_job_does_not_delete_an_upload_another_job_still_needs(
    tmp_path: Path,
) -> None:
    """One upload can back several jobs — the same file under two profiles, or
    natural and restore. Deleting it when the FIRST finishes destroys the input
    the others have yet to decode, so the user's upload vanishes and their
    second job fails preflight on a file they never removed.

    This reproduces the production wiring in ``app.py``: a terminal callback
    that hands the record's input to ``UploadStore.cleanup_input``.
    """
    store = UploadStore(root=tmp_path / "uploads")
    shared = store.stage("shared.wav")
    shared.write_bytes(b"RIFF" + b"\x00" * 2048)

    def factory(record: Any) -> list[str]:
        # The first job finishes promptly, the second lingers.
        delay = 0.1 if record.output_path.name == "a.wav" else 3.0
        return [
            sys.executable,
            "-u",
            "-c",
            textwrap.dedent(f"""
                import json, sys, time
                def emit(o):
                    sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
                emit({{"event":"progress","stage":"preflight","progress":0.1,"message":"p"}})
                time.sleep({delay})
                open({str(record.report_path)!r}, "w").write('{{"schema_version": 1}}')
                open({str(record.output_path)!r}, "wb").write(b"RIFF")
                emit({{"event":"done","progress":1.0,
                       "output_path":{str(record.output_path)!r},
                       "report_path":{str(record.report_path)!r}}})
            """),
        ]

    manager = JobManager(command_factory=factory, max_active_jobs=4)

    def cleanup_terminal_input(record: Any) -> None:
        # Mirrors the server's terminal hook: skip the delete while another
        # live job still names this upload as its input.
        if record.input_path.resolve() in manager.active_input_paths():
            return
        store.cleanup_input(record.input_path)

    manager.add_terminal_callback(cleanup_terminal_input)
    try:
        first = manager.submit(
            input_path=shared, output_path=tmp_path / "a.wav", profile="studio", overwrite=False
        )["job_id"]
        second = manager.submit(
            input_path=shared, output_path=tmp_path / "b.wav", profile="production", overwrite=False
        )["job_id"]

        deadline = time.monotonic() + 30.0
        checked = False
        while time.monotonic() < deadline:
            one = manager.get_status(first)
            two = manager.get_status(second)
            assert one is not None and two is not None
            if one["state"] in TERMINAL_STATES and two["state"] not in TERMINAL_STATES:
                assert shared.exists(), "the upload was deleted while a second job still needed it"
                checked = True
                break
            if two["state"] in TERMINAL_STATES:
                break
            time.sleep(0.02)
        assert checked, "the first job never finished while the second was still active"

        # Once BOTH are terminal, the upload is nobody's and must be reclaimed.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            two = manager.get_status(second)
            assert two is not None
            if two["state"] in TERMINAL_STATES:
                break
            time.sleep(0.02)
        record = manager._jobs[second]
        if record.input_path.resolve() not in manager.active_input_paths():
            store.cleanup_input(record.input_path)
        assert not shared.exists(), "the upload leaked once every job was done with it"
    finally:
        manager.shutdown()


def test_a_terminal_callback_may_ask_the_manager_a_question(tmp_path: Path) -> None:
    """Cleanup callbacks run with the job lock released.

    Terminal callbacks exist to clean up, and cleanup has to look before it
    deletes -- retention asks ``active_input_paths()`` whether another live
    job still needs an upload. That question was once asked from inside the
    manager's own non-reentrant lock, which deadlocked the ``hawavoclean-jobs``
    thread outright: no exception, no failed test, just a suite that stopped.

    The polling runs on its own thread on purpose. Once that lock is held for
    good, every public method blocks on it, so an in-line deadline loop never
    gets a turn to notice the time -- the regression would hang the run
    instead of reporting. Joining a watcher with a timeout turns the wedge
    into an ordinary failure.
    """
    manager = JobManager(command_factory=_success, max_active_jobs=2)
    seen: list[int] = []

    def ask_the_manager(record: JobRecord) -> None:
        # Terminal callbacks must run with the manager lock released so other threads can query/submit.
        acquired_from_other_thread = threading.Event()

        def probe() -> None:
            if manager._lock.acquire(timeout=0.5):
                try:
                    acquired_from_other_thread.set()
                finally:
                    manager._lock.release()

        probe_thread = threading.Thread(target=probe)
        probe_thread.start()
        probe_thread.join(timeout=1.0)
        assert acquired_from_other_thread.is_set(), (
            "manager lock was held during terminal callback execution"
        )
        # Every one of these re-enters the lock the callback used to hold.
        manager.active_input_paths()
        manager.list_jobs()
        manager.get_status(record.job_id)
        seen.append(len(manager.list_jobs()))

    manager.add_terminal_callback(ask_the_manager)
    source = tmp_path / "in.wav"
    source.write_bytes(b"RIFF")
    job_id = manager.submit(
        input_path=source,
        output_path=tmp_path / "out.wav",
        profile="studio",
        overwrite=False,
    )["job_id"]

    reached: list[str] = []

    def poll() -> None:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            status = manager.get_status(job_id)
            if status is not None and status["state"] in TERMINAL_STATES:
                reached.append(str(status["state"]))
                return
            time.sleep(0.02)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    watcher.join(timeout=25.0)

    # Deliberately no ``finally: shutdown()`` -- shutdown takes the same lock,
    # so on a regression it would hang the very run this assert is rescuing.
    assert not watcher.is_alive(), (
        "the jobs thread is wedged: a terminal callback re-entered the job lock"
    )
    assert reached and reached[0] in TERMINAL_STATES
    assert seen, "the terminal callback never completed"
    manager.shutdown()
