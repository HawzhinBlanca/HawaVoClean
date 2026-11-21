"""Durable Full Processing Record job integration and failure semantics."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean.hashing import hash_file
from hawavoclean.publication import (
    publication_paths,
    publish_output_generation,
    resolve_committed_publication,
)
from hawavoclean.record_bundle import create_processing_record
from hawavoclean.report.schema import (
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)
from hawavoclean.report.writer import serialize_json_report
from hawavoclean.server.app import create_app
from hawavoclean.server.job_store import (
    DurableJobStore,
    IdempotencyConflictError,
    JobStoreError,
    OutputConflictError,
    canonical_request_hash,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord, default_command
from tests.support.report_provenance import build, core, environment, guard

pytestmark = pytest.mark.unit


def _record_evidence(record: Any) -> dict[str, Any]:
    return {
        "path": str(record.path),
        "archive_sha256": record.archive_sha256,
        "content_sha256": record.content_sha256,
        "master_sha256": record.master_sha256,
        "report_sha256": record.report_sha256,
        "summary_sha256": record.summary_sha256,
        "total_uncompressed_bytes": record.total_uncompressed_bytes,
        "internal_hashes_verified": True,
        "authenticated_publisher": record.authenticated_publisher,
    }


def _wait_terminal(manager: JobManager, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        snapshot = manager.get_status(job_id)
        assert snapshot is not None
        if snapshot["state"] in TERMINAL_STATES:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job did not finish: {manager.get_status(job_id)}")


def _fixture_export(root: Path, *, amplitude: float = 0.0) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    master = root / "master.wav"
    report_path = root / "report.json"
    summary = root / "summary.txt"
    sf.write(
        master,
        np.full(4_800, amplitude, dtype=np.float32),
        48_000,
        subtype="PCM_24",
    )
    output = MediaStats(
        path="master.wav",
        sha256=hash_file(master),
        sample_rate=48_000,
        channels=1,
        samples=4_800,
        duration_s=0.1,
    )
    report = HawaVoCleanReport(
        schema_version=2,
        release=current_release_metadata(),
        build=build(),
        job_id="bundle-job-test",
        config_hash="a" * 64,
        input=output.model_copy(update={"path": "source.wav", "sha256": "b" * 64}),
        output=output,
        core=core("core", "algorithm", "c" * 64),
        guard=guard("guard", "d" * 64, "e" * 64),
        environment=environment(),
        summary=UnitSummary(),
    )
    report_path.write_text(serialize_json_report(report), encoding="utf-8")
    summary.write_text("Verified HawaVoClean summary\n", encoding="utf-8")
    return master, report_path, summary


def _bundle_success_factory(
    fixtures: tuple[Path, Path, Path],
) -> Callable[[JobRecord], list[str]]:
    master, report, summary = fixtures

    def factory(record: JobRecord) -> list[str]:
        assert record.bundle_path is not None
        script = f"""
            import json, shutil
            from hawavoclean.record_bundle import create_processing_record
            shutil.copyfile({str(master)!r}, {str(record.output_path)!r})
            shutil.copyfile({str(report)!r}, {str(record.report_path)!r})
            summary = {str(record.output_path.parent / f"{record.output_path.stem}.hawavoclean.txt")!r}
            shutil.copyfile({str(summary)!r}, summary)
            verified = create_processing_record(
                master_path={str(record.output_path)!r},
                report_path={str(record.report_path)!r},
                summary_path=summary,
                destination={str(record.bundle_path)!r},
                overwrite={record.overwrite!r},
            )
            print(json.dumps({{
                "event": "done",
                "report_path": {str(record.report_path)!r},
                "bundle_path": str(verified.path),
                "bundle_sha256": verified.archive_sha256,
            }}), flush=True)
        """
        return [sys.executable, "-c", textwrap.dedent(script)]

    return factory


def test_default_child_command_requests_exact_reserved_bundle_path(tmp_path: Path) -> None:
    record = JobRecord(
        job_id="j_bundle",
        input_path=tmp_path / "input.wav",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.hawavoclean.json",
        profile="production",
        overwrite=False,
        record_bundle=True,
        bundle_path=tmp_path / "out.hawavoclean.zip",
    )
    assert default_command(record)[-3:] == [
        "--record-bundle",
        str(record.bundle_path),
        "--progress-json",
    ]


def test_bundle_job_completes_only_with_independently_verified_evidence_and_recovers(
    tmp_path: Path,
) -> None:
    fixtures = _fixture_export(tmp_path / "fixtures")
    database = tmp_path / "jobs.sqlite3"
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    manager = JobManager(command_factory=_bundle_success_factory(fixtures), store_path=database)
    try:
        submitted = manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            conflict_policy="fail",
            idempotency_key="record-job-1",
            record_bundle=True,
        )
        completed = _wait_terminal(manager, submitted["job_id"])
        assert completed["state"] == "done"
        assert completed["bundle_path"] == str(tmp_path / "result.hawavoclean.zip")
        assert completed["bundle"]["archive_sha256"] == hash_file(
            tmp_path / "result.hawavoclean.zip"
        )
        assert completed["bundle"]["internal_hashes_verified"] is True
        assert completed["bundle"]["authenticated_publisher"] is False
    finally:
        manager.shutdown()

    recovered = JobManager(store_path=database)
    try:
        snapshot = recovered.get_status(submitted["job_id"])
        assert snapshot is not None
        assert snapshot["state"] == "done"
        assert snapshot["bundle"]["archive_sha256"] == completed["bundle"]["archive_sha256"]
    finally:
        recovered.shutdown()


def test_missing_bundle_fails_closed_even_when_child_claims_done(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    def dishonest(record: JobRecord) -> list[str]:
        return [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import json
                open({str(record.output_path)!r}, "wb").write(b"RIFF")
                open({str(record.report_path)!r}, "w").write("{{}}")
                print(json.dumps({{"event":"done", "report_path":{str(record.report_path)!r}}}))
                """
            ),
        ]

    manager = JobManager(command_factory=dishonest)
    try:
        submitted = manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            record_bundle=True,
        )
        failed = _wait_terminal(manager, submitted["job_id"])
        assert failed["state"] == "failed"
        assert failed["error"]["code"] == "PUBLICATION_FAILURE"
        assert "Processing Record verification failed" in failed["error"]["message"]
        assert "bundle" not in failed
    finally:
        manager.shutdown()


def test_conflict_policy_applies_to_report_summary_and_bundle_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    (tmp_path / "result.hawavoclean.txt").write_text("belongs to someone else")
    manager = JobManager(command_factory=lambda _record: [sys.executable, "-c", "pass"])
    try:
        with pytest.raises(OutputConflictError, match="already exists"):
            manager.submit(
                input_path=source,
                output_path=tmp_path / "result.wav",
                profile="production",
                overwrite=False,
                conflict_policy="fail",
                record_bundle=True,
            )
        unique = manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            conflict_policy="unique",
            record_bundle=True,
        )
        assert unique["output_path"] == str(tmp_path / "result (2).wav")
        assert unique["report_path"] == str(tmp_path / "result (2).hawavoclean.json")
        assert unique["bundle_path"] == str(tmp_path / "result (2).hawavoclean.zip")
    finally:
        manager.shutdown()


def test_in_memory_nonbundle_replace_refuses_stale_same_stem_record(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    (tmp_path / "result.hawavoclean.zip").write_bytes(b"prior Processing Record")
    manager = JobManager(command_factory=lambda _record: [sys.executable, "-c", "pass"])
    try:
        with pytest.raises(OutputConflictError, match="Processing Record already exists"):
            manager.submit(
                input_path=source,
                output_path=tmp_path / "result.wav",
                profile="production",
                overwrite=True,
                conflict_policy="replace",
                record_bundle=False,
            )
        unique = manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            conflict_policy="unique",
            record_bundle=False,
        )
        assert unique["output_path"] == str(tmp_path / "result (2).wav")
    finally:
        manager.shutdown()


def test_record_bundle_is_part_of_idempotency_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    manager = JobManager(command_factory=lambda _record: [sys.executable, "-c", "pass"])
    try:
        manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            idempotency_key="same-key",
            record_bundle=False,
        )
        with pytest.raises(IdempotencyConflictError, match="different request"):
            manager.submit(
                input_path=source,
                output_path=tmp_path / "result.wav",
                profile="production",
                overwrite=False,
                idempotency_key="same-key",
                record_bundle=True,
            )
    finally:
        manager.shutdown()


def test_startup_never_attributes_valid_prior_bundle_to_abandoned_replace_job(
    tmp_path: Path,
) -> None:
    fixtures = _fixture_export(tmp_path / "fixtures")
    master, report, summary = fixtures
    output = tmp_path / "result.wav"
    report_out = tmp_path / "result.hawavoclean.json"
    summary_out = tmp_path / "result.hawavoclean.txt"
    bundle = tmp_path / "result.hawavoclean.zip"
    shutil.copyfile(master, output)
    shutil.copyfile(report, report_out)
    shutil.copyfile(summary, summary_out)
    create_processing_record(
        master_path=output,
        report_path=report_out,
        summary_path=summary_out,
        destination=bundle,
    )

    database = tmp_path / "jobs.sqlite3"
    from hawavoclean.server.job_store import DurableJobStore

    record = JobRecord(
        job_id="j_abandoned",
        input_path=tmp_path / "source.wav",
        output_path=output,
        report_path=report_out,
        profile="production",
        overwrite=False,
        state="running",
        record_bundle=True,
        bundle_path=bundle,
    )
    store = DurableJobStore(database)
    request_hash = canonical_request_hash({"test": "recovery"})
    store.reserve(
        record=record.storage_record(),
        request_hash=request_hash,
        idempotency_key="recovery-key",
        conflict_policy="replace",
    )
    store.close()

    manager = JobManager(store_path=database)
    try:
        recovered = manager.get_status(record.job_id)
        assert recovered is not None
        assert recovered["state"] == "interrupted"
        assert recovered["error"]["code"] == "INTERRUPTED"
        assert "bundle" not in recovered
    finally:
        manager.shutdown()


def test_cancel_during_bundle_phase_never_reports_complete(tmp_path: Path) -> None:
    fixtures = _fixture_export(tmp_path / "fixtures")
    master, report, summary = fixtures
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    def pauses_before_bundle(record: JobRecord) -> list[str]:
        summary_out = record.output_path.parent / f"{record.output_path.stem}.hawavoclean.txt"
        script = f"""
            import shutil, time
            shutil.copyfile({str(master)!r}, {str(record.output_path)!r})
            shutil.copyfile({str(report)!r}, {str(record.report_path)!r})
            shutil.copyfile({str(summary)!r}, {str(summary_out)!r})
            print('{{"event":"progress","stage":"record_bundle","progress":0.995}}', flush=True)
            time.sleep(60)
        """
        return [sys.executable, "-c", textwrap.dedent(script)]

    manager = JobManager(command_factory=pauses_before_bundle, kill_grace_s=0.1)
    try:
        submitted = manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            record_bundle=True,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snapshot = manager.get_status(submitted["job_id"])
            assert snapshot is not None
            if snapshot["stage"] == "record_bundle":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("job never reached record_bundle phase")
        assert manager.cancel(submitted["job_id"])
        cancelled = _wait_terminal(manager, submitted["job_id"])
        assert cancelled["state"] == "cancelled"
        assert "bundle" not in cancelled
        assert not Path(cancelled["bundle_path"]).exists()
    finally:
        manager.shutdown()


def test_startup_fails_closed_if_completed_bundle_was_tampered(tmp_path: Path) -> None:
    fixtures = _fixture_export(tmp_path / "fixtures")
    database = tmp_path / "jobs.sqlite3"
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    first = JobManager(command_factory=_bundle_success_factory(fixtures), store_path=database)
    try:
        submitted = first.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            record_bundle=True,
        )
        completed = _wait_terminal(first, submitted["job_id"])
        assert completed["state"] == "done"
    finally:
        first.shutdown()

    bundle = tmp_path / "result.hawavoclean.zip"
    bundle.write_bytes(bundle.read_bytes()[:-16] + b"deliberate-tamper")
    second = JobManager(store_path=database)
    try:
        invalid = second.get_status(submitted["job_id"])
        assert invalid is not None
        assert invalid["state"] == "failed"
        assert invalid["error"]["code"] == "ARTIFACT_INVALID"
        assert "bundle" not in invalid
    finally:
        second.shutdown()


def test_startup_repairs_mixed_public_aliases_before_bundle_validation(
    tmp_path: Path,
) -> None:
    old_master, old_report, old_summary = _fixture_export(tmp_path / "old", amplitude=0.01)
    new_master, new_report, new_summary = _fixture_export(tmp_path / "new", amplitude=0.02)
    output = tmp_path / "result.wav"
    publish_output_generation(
        old_master,
        output,
        old_report.read_text(encoding="utf-8"),
        old_summary.read_text(encoding="utf-8"),
    )
    old_generation = resolve_committed_publication(output)
    assert old_generation is not None
    publish_output_generation(
        new_master,
        output,
        new_report.read_text(encoding="utf-8"),
        new_summary.read_text(encoding="utf-8"),
        overwrite=True,
    )
    new_generation = resolve_committed_publication(output)
    assert new_generation is not None
    bundle = tmp_path / "result.hawavoclean.zip"
    evidence = create_processing_record(
        master_path=new_generation[0],
        report_path=new_generation[1],
        summary_path=new_generation[2],
        destination=bundle,
    )

    record = JobRecord(
        job_id="j_mixed",
        input_path=tmp_path / "source.wav",
        output_path=output,
        report_path=publication_paths(output).json,
        profile="production",
        overwrite=True,
        conflict_policy="replace",
        state="done",
        stage="done",
        progress=1.0,
        record_bundle=True,
        bundle_path=bundle,
        bundle={
            "path": str(evidence.path),
            "archive_sha256": evidence.archive_sha256,
            "content_sha256": evidence.content_sha256,
            "master_sha256": evidence.master_sha256,
            "report_sha256": evidence.report_sha256,
            "summary_sha256": evidence.summary_sha256,
            "total_uncompressed_bytes": evidence.total_uncompressed_bytes,
            "internal_hashes_verified": True,
            "authenticated_publisher": evidence.authenticated_publisher,
        },
    )
    database = tmp_path / "jobs.sqlite3"
    from hawavoclean.server.job_store import DurableJobStore

    store = DurableJobStore(database)
    request_hash = canonical_request_hash({"test": "mixed-alias-recovery"})
    reserved = store.reserve(
        record={**record.storage_record(), "state": "queued"},
        request_hash=request_hash,
        idempotency_key="mixed-key",
        conflict_policy="replace",
    )
    stored = {**reserved.record, **record.storage_record()}
    store.update(stored, terminal=True)
    store.close()

    # Simulate the hard-kill window: current points at new, but only the audio
    # alias has advanced. The old report/summary are both known generations,
    # so the official resolver can safely repair them forward.
    paths = publication_paths(output)
    paths.audio.write_bytes(new_generation[0].read_bytes())
    paths.json.write_bytes(old_generation[1].read_bytes())
    paths.txt.write_bytes(old_generation[2].read_bytes())

    manager = JobManager(store_path=database)
    try:
        recovered = manager.get_status(record.job_id)
        assert recovered is not None and recovered["state"] == "done"
        assert recovered["bundle"]["archive_sha256"] == evidence.archive_sha256
        assert paths.json.read_bytes() == new_generation[1].read_bytes()
        assert paths.txt.read_bytes() == new_generation[2].read_bytes()
    finally:
        manager.shutdown()


def test_replaced_bundle_job_remains_exact_as_pruned_durable_resource(
    tmp_path: Path,
) -> None:
    old_master, old_report, old_summary = _fixture_export(tmp_path / "old", amplitude=0.01)
    new_master, new_report, new_summary = _fixture_export(tmp_path / "new", amplitude=0.02)
    output = tmp_path / "result.wav"
    publish_output_generation(
        old_master,
        output,
        old_report.read_text(encoding="utf-8"),
        old_summary.read_text(encoding="utf-8"),
    )
    old_generation = resolve_committed_publication(output)
    assert old_generation is not None
    bundle_path = tmp_path / "result.hawavoclean.zip"
    old_bundle = create_processing_record(
        master_path=old_generation[0],
        report_path=old_generation[1],
        summary_path=old_generation[2],
        destination=bundle_path,
    )

    publish_output_generation(
        new_master,
        output,
        new_report.read_text(encoding="utf-8"),
        new_summary.read_text(encoding="utf-8"),
        overwrite=True,
    )
    new_generation = resolve_committed_publication(output)
    assert new_generation is not None
    new_bundle = create_processing_record(
        master_path=new_generation[0],
        report_path=new_generation[1],
        summary_path=new_generation[2],
        destination=bundle_path,
        overwrite=True,
    )
    assert old_bundle.archive_sha256 != new_bundle.archive_sha256

    database = tmp_path / "jobs.sqlite3"
    store = DurableJobStore(database)
    for name, generation, bundle in (
        ("old", old_generation, old_bundle),
        ("new", new_generation, new_bundle),
    ):
        record = JobRecord(
            job_id=f"j_{name}",
            input_path=tmp_path / f"{name}-source.wav",
            output_path=output,
            report_path=publication_paths(output).json,
            profile="production",
            overwrite=True,
            idempotency_key=f"{name}-request",
            conflict_policy="replace",
            request_hash=("a" if name == "old" else "b") * 64,
            state="done",
            stage="done",
            progress=1.0,
            record_bundle=True,
            bundle_path=bundle_path,
            bundle=_record_evidence(bundle),
            report=json.loads(generation[1].read_text(encoding="utf-8")),
        )
        reserved = store.reserve(
            record={**record.storage_record(), "state": "queued"},
            request_hash=record.request_hash,
            idempotency_key=record.idempotency_key,
            conflict_policy="replace",
        )
        store.update({**reserved.record, **record.storage_record()}, terminal=True)
    store.close()
    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE jobs SET terminal_at = 1 WHERE job_id = 'j_old'")
        connection.execute("UPDATE jobs SET terminal_at = 2 WHERE job_id = 'j_new'")
        connection.commit()
    finally:
        connection.close()

    manager = JobManager(
        store_path=database,
        max_terminal_jobs=1,
        terminal_ttl_s=100.0,
        wall_clock=lambda: 3.0,
    )
    try:
        assert [item["job_id"] for item in manager.list_jobs()] == ["j_new"]
        old_status = manager.get_status("j_old")
        assert old_status is not None and old_status["state"] == "done"
        assert old_status["bundle"]["archive_sha256"] == old_bundle.archive_sha256
        assert old_status["report"]["output"]["sha256"] == old_bundle.master_sha256

        app = create_app(
            "bundle-resource-token",
            job_manager=manager,
            on_shutdown=lambda: None,
            min_free_bytes=0,
        )
        headers = {"X-Hawa-Token": "bundle-resource-token"}
        with TestClient(app, base_url="http://127.0.0.1") as client:
            status = client.get("/api/v1/jobs/j_old", headers=headers)
            exact_master = client.get("/api/v1/jobs/j_old/artifacts/master", headers=headers)
            exact_report = client.get("/api/v1/jobs/j_old/artifacts/report", headers=headers)
            stale_record = client.get("/api/v1/jobs/j_old/artifacts/record", headers=headers)
        assert status.status_code == 200 and status.json()["state"] == "completed"
        assert exact_master.status_code == 200
        assert exact_master.content == old_generation[0].read_bytes()
        assert exact_report.status_code == 200
        assert exact_report.json()["output"]["sha256"] == old_bundle.master_sha256
        assert stale_record.status_code == 409
        assert stale_record.json()["error"] == "artifact_unavailable"
    finally:
        manager.shutdown()


def test_verified_bundle_is_not_reported_complete_when_terminal_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _fixture_export(tmp_path / "fixtures")
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    manager = JobManager(
        command_factory=_bundle_success_factory(fixtures),
        store_path=tmp_path / "jobs.sqlite3",
    )
    store = manager._store
    assert store is not None
    real_update = store.update

    def fail_terminal(record: dict[str, Any], *, terminal: bool) -> None:
        if terminal:
            raise JobStoreError("simulated SQLite fsync failure")
        real_update(record, terminal=terminal)

    monkeypatch.setattr(store, "update", fail_terminal)
    try:
        submitted = manager.submit(
            input_path=source,
            output_path=tmp_path / "result.wav",
            profile="production",
            overwrite=False,
            record_bundle=True,
        )
        failed = _wait_terminal(manager, submitted["job_id"])
        assert failed["state"] == "failed"
        assert failed["error"]["code"] == "DURABILITY_FAILURE"
        assert "bundle" not in failed
        assert manager.persistence_error == "simulated SQLite fsync failure"
    finally:
        manager.shutdown()
