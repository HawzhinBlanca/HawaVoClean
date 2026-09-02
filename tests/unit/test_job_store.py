"""Durable job identity, recovery, and cross-process output leases."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from hawavoclean.publication import publish_output_generation
from hawavoclean.server.job_store import (
    DurableJobStore,
    IdempotencyConflictError,
    JobStoreError,
    OutputConflictError,
    canonical_request_hash,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord


def _stored_record(tmp_path: Path, *, job_id: str = "j_one") -> dict[str, Any]:
    output = tmp_path / "master.wav"
    return {
        "job_id": job_id,
        "input_path": str(tmp_path / "input.wav"),
        "output_path": str(output),
        "report_path": str(tmp_path / "master.hawavoclean.json"),
        "profile": "production",
        "overwrite": False,
        "idempotency_key": "request-one",
        "conflict_policy": "fail",
        "request_hash": "a" * 64,
        "mode": "natural",
        "speaker_id": None,
        "cutoff_hz": None,
        "state": "queued",
        "stage": "preflight",
        "progress": 0.0,
        "message": "Queued",
        "unit": None,
        "created_at": "2026-08-27T00:00:00.000Z",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "report": None,
        "cancel_requested": False,
        "seq": 0,
    }


def test_canonical_request_hash_is_order_independent_and_strict() -> None:
    assert canonical_request_hash({"a": 1, "b": [2]}) == canonical_request_hash({"b": [2], "a": 1})
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_request_hash({"cutoff_hz": float("nan")})


def test_idempotent_reservation_reuses_original_and_rejects_changed_request(
    tmp_path: Path,
) -> None:
    store = DurableJobStore(tmp_path / "state" / "jobs.sqlite3")
    try:
        record = _stored_record(tmp_path)
        first = store.reserve(
            record=record,
            request_hash="a" * 64,
            idempotency_key="request-one",
            conflict_policy="fail",
        )
        again = store.reserve(
            record={**record, "job_id": "j_other"},
            request_hash="a" * 64,
            idempotency_key="request-one",
            conflict_policy="fail",
        )
        assert not first.reused
        assert again.reused and again.record["job_id"] == "j_one"
        with pytest.raises(IdempotencyConflictError, match="different request"):
            store.reserve(
                record={**record, "job_id": "j_bad"},
                request_hash="b" * 64,
                idempotency_key="request-one",
                conflict_policy="fail",
            )
    finally:
        store.close()


def test_active_output_lease_is_cross_connection_and_unique_policy_is_deterministic(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    first_store = DurableJobStore(database)
    second_store = DurableJobStore(database)
    try:
        first_store.reserve(
            record=_stored_record(tmp_path),
            request_hash="a" * 64,
            idempotency_key="request-one",
            conflict_policy="replace",
        )
        second = _stored_record(tmp_path, job_id="j_two")
        second["idempotency_key"] = "request-two"
        with pytest.raises(OutputConflictError, match="active job"):
            second_store.reserve(
                record=second,
                request_hash="b" * 64,
                idempotency_key="request-two",
                conflict_policy="replace",
            )
        unique = second_store.reserve(
            record=second,
            request_hash="b" * 64,
            idempotency_key="request-two",
            conflict_policy="unique",
        )
        assert Path(str(unique.record["output_path"])).name == "master (2).wav"
        assert Path(str(unique.record["report_path"])).name == "master (2).hawavoclean.json"
    finally:
        second_store.close()
        first_store.close()


def test_fail_policy_refuses_regular_file_and_dangling_symlink(tmp_path: Path) -> None:
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    output = tmp_path / "master.wav"
    output.write_bytes(b"existing")
    try:
        with pytest.raises(OutputConflictError, match="already exists"):
            store.reserve(
                record=_stored_record(tmp_path),
                request_hash="a" * 64,
                idempotency_key="request-one",
                conflict_policy="fail",
            )
        output.unlink()
        output.symlink_to("missing-generation.wav")
        with pytest.raises(OutputConflictError, match="already exists"):
            store.reserve(
                record=_stored_record(tmp_path),
                request_hash="a" * 64,
                idempotency_key="request-one",
                conflict_policy="fail",
            )
    finally:
        store.close()


def test_durable_reservation_applies_conflicts_to_processing_record_sidecar(
    tmp_path: Path,
) -> None:
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    (tmp_path / "master.hawavoclean.zip").write_bytes(b"prior record")
    record = {
        **_stored_record(tmp_path),
        "record_bundle": True,
        "bundle_path": str(tmp_path / "master.hawavoclean.zip"),
    }
    try:
        with pytest.raises(OutputConflictError, match="sidecar already exists"):
            store.reserve(
                record=record,
                request_hash="a" * 64,
                idempotency_key="request-one",
                conflict_policy="fail",
            )
        unique = store.reserve(
            record=record,
            request_hash="a" * 64,
            idempotency_key="request-one",
            conflict_policy="unique",
        )
        assert Path(str(unique.record["output_path"])).name == "master (2).wav"
        assert Path(str(unique.record["bundle_path"])).name == "master (2).hawavoclean.zip"
    finally:
        store.close()


def test_nonbundle_reservation_never_leaves_a_stale_processing_record(
    tmp_path: Path,
) -> None:
    store = DurableJobStore(tmp_path / "jobs.sqlite3")
    (tmp_path / "master.hawavoclean.zip").write_bytes(b"describes prior audio")
    record = {**_stored_record(tmp_path), "record_bundle": False, "bundle_path": None}
    try:
        with pytest.raises(OutputConflictError, match="sidecar already exists"):
            store.reserve(
                record=record,
                request_hash="a" * 64,
                idempotency_key="request-one",
                conflict_policy="fail",
            )
        with pytest.raises(OutputConflictError, match="Processing Record already exists"):
            store.reserve(
                record=record,
                request_hash="a" * 64,
                idempotency_key="request-one",
                conflict_policy="replace",
            )
        unique = store.reserve(
            record=record,
            request_hash="a" * 64,
            idempotency_key="request-one",
            conflict_policy="unique",
        )
        assert Path(str(unique.record["output_path"])).name == "master (2).wav"
        assert unique.record["bundle_path"] is None
    finally:
        store.close()


def test_restart_marks_abandoned_work_interrupted_and_releases_output(tmp_path: Path) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    first = DurableJobStore(database)
    first.reserve(
        record=_stored_record(tmp_path),
        request_hash="a" * 64,
        idempotency_key="request-one",
        conflict_policy="replace",
    )
    first.close()  # simulate a process that vanished without a terminal update

    recovered = DurableJobStore(database)
    try:
        records = recovered.load_and_interrupt()
        assert records[0]["state"] == "interrupted"
        assert records[0]["error"]["code"] == "INTERRUPTED"
        replacement = _stored_record(tmp_path, job_id="j_retry")
        replacement["idempotency_key"] = "request-retry"
        reserved = recovered.reserve(
            record=replacement,
            request_hash="b" * 64,
            idempotency_key="request-retry",
            conflict_policy="replace",
        )
        assert reserved.record["output_path"] == replacement["output_path"]
    finally:
        recovered.close()


def test_store_refuses_future_schema(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 999")
    connection.close()
    with pytest.raises(JobStoreError, match="newer than supported"):
        DurableJobStore(database)


def _success_command(record: Any) -> list[str]:
    audio = b"RIFF-durable-test"
    report = json.dumps({"ok": True, "output": {"sha256": hashlib.sha256(audio).hexdigest()}})
    event = json.dumps({"event": "done", "report_path": str(record.report_path)})
    summary = record.output_path.parent / f"{record.output_path.stem}.hawavoclean.txt"
    code = (
        "from pathlib import Path; "
        f"Path({str(record.output_path)!r}).write_bytes({audio!r}); "
        f"Path({str(record.report_path)!r}).write_text({report!r}, encoding='utf-8'); "
        f"Path({str(summary)!r}).write_text('summary', encoding='utf-8'); "
        f"print({event!r})"
    )
    return [sys.executable, "-c", code]


def _wait_terminal(manager: JobManager, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        snap = manager.get_status(job_id)
        assert snap is not None
        if snap["state"] in TERMINAL_STATES:
            return snap
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {manager.get_status(job_id)}")


def test_job_manager_persists_history_and_deduplicates_submission(tmp_path: Path) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    first = JobManager(command_factory=_success_command, store_path=database)
    try:
        submitted = first.submit(
            input_path=source,
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
            idempotency_key="desktop-request-1",
            conflict_policy="fail",
        )
        repeated = first.submit(
            input_path=source,
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
            idempotency_key="desktop-request-1",
            conflict_policy="fail",
        )
        assert repeated["job_id"] == submitted["job_id"]
        assert _wait_terminal(first, submitted["job_id"])["state"] == "done"
    finally:
        first.shutdown()

    second = JobManager(command_factory=_success_command, store_path=database)
    try:
        recovered = second.get_status(submitted["job_id"])
        assert recovered is not None and recovered["state"] == "done"
        assert recovered["report"]["ok"] is True
        assert (
            recovered["report"]["output"]["sha256"]
            == hashlib.sha256(b"RIFF-durable-test").hexdigest()
        )
    finally:
        second.shutdown()


@pytest.mark.parametrize("artifact", ["audio", "report", "summary"])
def test_restart_fails_closed_when_completed_nonbundle_artifact_changed(
    tmp_path: Path, artifact: str
) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    first = JobManager(command_factory=_success_command, store_path=database)
    try:
        submitted = first.submit(
            input_path=source,
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
            idempotency_key="artifact-bound-request",
        )
        assert _wait_terminal(first, submitted["job_id"])["state"] == "done"
    finally:
        first.shutdown()

    paths = {
        "audio": tmp_path / "master.wav",
        "report": tmp_path / "master.hawavoclean.json",
        "summary": tmp_path / "master.hawavoclean.txt",
    }
    with open(paths[artifact], "ab") as stream:
        stream.write(b"tamper")

    recovered = JobManager(store_path=database)
    try:
        snapshot = recovered.get_status(submitted["job_id"])
        assert snapshot is not None
        assert snapshot["state"] == "failed"
        assert snapshot["error"]["code"] == "ARTIFACT_INVALID"
    finally:
        recovered.shutdown()


def test_restart_resolves_exact_old_generation_when_later_run_has_same_audio(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "candidate.wav"
    audio.write_bytes(b"identical-master")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    output = tmp_path / "master.wav"
    first_report = json.dumps({"output": {"sha256": digest}, "run": "first"})
    publish_output_generation(audio, output, first_report, "first summary")

    record = JobRecord(
        job_id="j_exact_generation",
        input_path=tmp_path / "input.wav",
        output_path=output,
        report_path=tmp_path / "master.hawavoclean.json",
        profile="production",
        overwrite=True,
        idempotency_key="exact-generation",
        conflict_policy="replace",
        request_hash="a" * 64,
        state="done",
        stage="done",
        progress=1.0,
        report=json.loads(first_report),
    )
    capture = JobManager()
    try:
        capture._capture_nonbundle_artifacts(record)
    finally:
        capture.shutdown()
    assert record.artifact_evidence is not None

    database = tmp_path / "jobs.sqlite3"
    store = DurableJobStore(database)
    reserved = store.reserve(
        record={**record.storage_record(), "state": "queued"},
        request_hash=record.request_hash,
        idempotency_key=record.idempotency_key,
        conflict_policy="replace",
    )
    store.update({**reserved.record, **record.storage_record()}, terminal=True)
    store.close()

    second_report = json.dumps({"output": {"sha256": digest}, "run": "second"})
    publish_output_generation(audio, output, second_report, "second summary", overwrite=True)

    manager = JobManager(store_path=database)
    try:
        recovered = manager.get_status(record.job_id)
        assert recovered is not None and recovered["state"] == "done"
        assert recovered["report"]["run"] == "first"
    finally:
        manager.shutdown()


def test_restart_does_not_refresh_terminal_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "jobs.sqlite3"
    terminal_wall_time = [1_000.0]
    monkeypatch.setattr("hawavoclean.server.job_store.time.time", lambda: terminal_wall_time[0])
    store = DurableJobStore(database)
    record = _stored_record(tmp_path)
    reserved = store.reserve(
        record=record,
        request_hash="a" * 64,
        idempotency_key="request-one",
        conflict_policy="replace",
    ).record
    failed = {
        **reserved,
        "state": "failed",
        "stage": "error",
        "finished_at": "2026-08-27T00:00:00.000Z",
        "error": {"code": "TEST", "message": "terminal"},
    }
    store.update(failed, terminal=True)
    store.close()

    monotonic_now = [100.0]
    first = JobManager(
        store_path=database,
        terminal_ttl_s=10.0,
        clock=lambda: monotonic_now[0],
        wall_clock=lambda: 1_009.0,
    )
    try:
        assert first.get_status("j_one") is not None
    finally:
        first.shutdown()

    second = JobManager(
        store_path=database,
        terminal_ttl_s=10.0,
        clock=lambda: monotonic_now[0],
        wall_clock=lambda: 1_010.0,
    )
    try:
        assert second.list_jobs() == []
        resource = second.get_status("j_one")
        assert resource is not None and resource["job_id"] == "j_one"
        receipt = second.get_by_idempotency("request-one")
        assert receipt is not None and receipt["job_id"] == "j_one"
    finally:
        second.shutdown()


def test_durable_history_is_bounded_and_old_keyed_rows_become_compact_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "jobs.sqlite3"
    wall_time = [1_000.0]
    monkeypatch.setattr("hawavoclean.server.job_store.time.time", lambda: wall_time[0])
    store = DurableJobStore(database)
    for index in range(9):
        record = _stored_record(tmp_path, job_id=f"j_{index}")
        record["output_path"] = str(tmp_path / f"master-{index}.wav")
        record["report_path"] = str(tmp_path / f"master-{index}.hawavoclean.json")
        record["idempotency_key"] = f"key-{index}" if index % 2 == 0 else None
        reserved = store.reserve(
            record=record,
            request_hash=f"{index:064x}",
            idempotency_key=record["idempotency_key"],
            conflict_policy="replace",
        ).record
        wall_time[0] = 1_000.0 + index
        terminal = {
            **reserved,
            "state": "done",
            "stage": "done",
            "progress": 1.0,
            "record_bundle": True,
            "bundle_path": str(tmp_path / f"master-{index}.hawavoclean.zip"),
            "artifact_evidence": {"schema_version": 1},
            "report": {"large": "x" * 20_000},
        }
        store.update(terminal, terminal=True)
    store.close()

    manager = JobManager(
        store_path=database,
        max_terminal_jobs=3,
        terminal_ttl_s=1_000.0,
        wall_clock=lambda: 1_010.0,
    )
    try:
        assert len(manager.list_jobs()) == 3
    finally:
        manager.shutdown()

    connection = sqlite3.connect(database)
    try:
        visible = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE history_visible = 1"
        ).fetchone()[0]
        anonymous = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE history_visible = 0 AND idempotency_key IS NULL"
        ).fetchone()[0]
        compact_raw = connection.execute(
            "SELECT record_json FROM jobs WHERE history_visible = 0 LIMIT 1"
        ).fetchone()[0]
    finally:
        connection.close()
    assert visible == 3
    assert anonymous == 0
    assert json.loads(compact_raw)["report"] is None


def test_pruned_exact_retry_precedes_queue_capacity_rejection(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    now = [0.0]

    def command(record: Any) -> list[str]:
        if record.output_path.name == "busy.wav":
            return [sys.executable, "-c", "import time; time.sleep(60)"]
        return _success_command(record)

    manager = JobManager(
        command_factory=command,
        store_path=database,
        max_active_jobs=1,
        terminal_ttl_s=1.0,
        clock=lambda: now[0],
    )
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    try:
        original = manager.submit(
            input_path=source,
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
            idempotency_key="exact-retry",
        )
        assert _wait_terminal(manager, original["job_id"])["state"] == "done"
        now[0] = 2.0
        assert manager.list_jobs() == []
        durable_resource = manager.get_status(original["job_id"])
        assert durable_resource is not None and durable_resource["state"] == "done"

        busy = manager.submit(
            input_path=source,
            output_path=tmp_path / "busy.wav",
            profile="production",
            overwrite=False,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state = manager.get_status(busy["job_id"])
            if state is not None and state["state"] == "running":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("busy job did not start")

        repeated = manager.submit(
            input_path=source,
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
            idempotency_key="exact-retry",
        )
        assert repeated["job_id"] == original["job_id"]
        assert repeated["state"] == "done"
    finally:
        manager.shutdown(grace_s=0.1)


def test_only_one_live_broker_can_own_and_recover_a_job_store(tmp_path: Path) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    first = JobManager(
        command_factory=lambda _record: [sys.executable, "-c", "import time; time.sleep(2)"],
        store_path=database,
    )
    try:
        submitted = first.submit(
            input_path=tmp_path / "input.wav",
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
        )
        with pytest.raises(JobStoreError, match="already owns"):
            JobManager(store_path=database)
        still_live = first.get_status(submitted["job_id"])
        assert still_live is not None
        assert still_live["state"] in {"queued", "running"}
        assert still_live["state"] != "interrupted"
    finally:
        first.shutdown()

    replacement = JobManager(store_path=database)
    try:
        recovered = replacement.get_status(submitted["job_id"])
        assert recovered is not None
        assert recovered["state"] in {"cancelled", "interrupted"}
    finally:
        replacement.shutdown()


def test_prepare_batch_late_failure_rolls_back_rows_keys_and_output_leases(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state" / "jobs.sqlite3"
    marker = tmp_path / "worker-started"

    def command(_record: Any) -> list[str]:
        return [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('started')"]

    manager = JobManager(command_factory=command, store_path=database)
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    try:
        with (
            pytest.raises(OutputConflictError, match="active job"),
            manager.prepare_batch(),
        ):
            manager.submit(
                input_path=source,
                output_path=tmp_path / "master.wav",
                profile="production",
                overwrite=False,
                idempotency_key="batch-item-one",
                conflict_policy="fail",
            )
            manager.submit(
                input_path=source,
                output_path=tmp_path / "master.wav",
                profile="studio",
                overwrite=False,
                idempotency_key="batch-item-two",
                conflict_policy="fail",
            )
        assert not marker.exists()
        assert manager.list_jobs() == []
        assert manager.get_by_idempotency("batch-item-one") is None

        # Both the durable idempotency row and output lease were deleted in
        # the same rollback, so the exact first item can be accepted again.
        retried = manager.submit(
            input_path=source,
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
            idempotency_key="batch-item-one",
            conflict_policy="fail",
        )
        assert _wait_terminal(manager, retried["job_id"])["state"] == "failed"
        # The fake child does not speak the done-event protocol, but it did
        # execute only after the successful standalone submission.
        assert marker.exists()
    finally:
        manager.shutdown()


def test_job_store_edge_cases_and_branch_coverage(tmp_path: Path) -> None:
    from hawavoclean.server.job_store import unique_candidate

    # 1. unique_candidate
    with pytest.raises(ValueError, match="ordinal must be positive"):
        unique_candidate(Path("a.wav"), 0)
    assert unique_candidate(Path("a.wav"), 1) == Path("a.wav")

    # 2. _encode and _decode validation
    with pytest.raises(JobStoreError, match="not canonical JSON"):
        DurableJobStore._encode({"bad": object()})
    with pytest.raises(JobStoreError, match="invalid JSON"):
        DurableJobStore._decode("{invalid json")
    with pytest.raises(JobStoreError, match="not an object"):
        DurableJobStore._decode("123")

    # 3. Schema Version 1 migration
    v1_db = tmp_path / "v1.sqlite3"
    conn = sqlite3.connect(v1_db)
    try:
        conn.executescript(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE,
                request_hash TEXT NOT NULL,
                conflict_policy TEXT NOT NULL,
                output_path TEXT NOT NULL,
                output_key TEXT NOT NULL,
                terminal INTEGER NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE UNIQUE INDEX active_output_lease ON jobs(output_key) WHERE terminal = 0;
            PRAGMA user_version = 1;
            """
        )
        conn.execute(
            "INSERT INTO jobs VALUES ('j1', 'k1', 'h1', 'fail', '/out.wav', '/out.wav', 1, "
            "'done', '2026-08-01T00:00:00Z', '2026-08-01T00:00:00Z', '{}')"
        )
    finally:
        conn.close()
    v1_store = DurableJobStore(v1_db)
    v1_store.close()

    # 4. Closed store methods
    store = DurableJobStore(tmp_path / "closed.sqlite3")
    store.close()
    # Double close is a no-op
    store.close()

    record = _stored_record(tmp_path)
    with pytest.raises(JobStoreError, match="closed"):
        store.reserve(
            record=record, request_hash="a" * 64, idempotency_key=None, conflict_policy="fail"
        )
    with pytest.raises(JobStoreError, match="closed"):
        store.find_idempotent("key", request_hash="a" * 64)
    with pytest.raises(JobStoreError, match="closed"):
        store.find_job("j_one")
    with pytest.raises(JobStoreError, match="closed"):
        store.update(record, terminal=True)
    with pytest.raises(JobStoreError, match="closed"):
        store.load_and_interrupt()
    with pytest.raises(JobStoreError, match="closed"):
        store.prune_terminal({"j_one"})
    with pytest.raises(JobStoreError, match="closed"):
        store.delete_queued(["j_one"])

    # 5. Unsupported conflict policy, unknown job update, and invalid load params
    live_store = DurableJobStore(tmp_path / "live.sqlite3")
    try:
        with pytest.raises(ValueError, match="unsupported conflict policy"):
            live_store.reserve(
                record=record,
                request_hash="a" * 64,
                idempotency_key=None,
                conflict_policy="bad_policy",  # type: ignore[arg-type]
            )
        with pytest.raises(JobStoreError, match="durable job is missing"):
            live_store.update({**record, "job_id": "nonexistent_job"}, terminal=True)
        with pytest.raises(ValueError, match="max_terminal_jobs"):
            live_store.load_and_interrupt(max_terminal_jobs=0)
        with pytest.raises(ValueError, match="terminal_ttl_s"):
            live_store.load_and_interrupt(terminal_ttl_s=0.0)
        # Empty prune and delete_queued are no-ops
        live_store.prune_terminal(set())
        live_store.delete_queued([])
        assert live_store.find_job("nonexistent") is None

        # Stale processing record in replace mode without record_bundle
        stale_out = tmp_path / "stale_out.wav"
        stale_zip = tmp_path / "stale_out.hawavoclean.zip"
        stale_zip.write_bytes(b"fake zip")
        stale_record = {**record, "job_id": "j_stale", "output_path": str(stale_out)}
        with pytest.raises(OutputConflictError, match="same-stem Processing Record already exists"):
            live_store.reserve(
                record=stale_record,
                request_hash="b" * 64,
                idempotency_key="stale_key",
                conflict_policy="replace",
            )
    finally:
        live_store.close()


def test_job_store_error_branches(tmp_path: Path) -> None:
    store = DurableJobStore(tmp_path / "errors.sqlite3")
    try:
        record = _stored_record(tmp_path, job_id="j_err1")
        store.reserve(
            record=record,
            request_hash="a" * 64,
            idempotency_key="k1",
            conflict_policy="fail",
        )
        # Update to terminal state
        record["state"] = "done"
        store.update(record, terminal=True)

        # Attempt to delete_queued on a finished/terminal job
        with pytest.raises(JobStoreError, match="no longer queued"):
            store.delete_queued(["j_err1"])

        # Attempt to delete_queued with non-existent job
        with pytest.raises(JobStoreError, match="no longer queued"):
            store.delete_queued(["nonexistent_job_id"])

        # Injected SQLite error on operations
        class _FailingConnProxy:
            def __init__(self, real: sqlite3.Connection) -> None:
                self._real = real

            def execute(self, *_args: object, **_kwargs: object) -> Any:
                raise sqlite3.OperationalError("disk I/O error")

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

        real_conn = store._conn
        store._conn = _FailingConnProxy(real_conn)  # type: ignore[assignment]
        with pytest.raises(JobStoreError, match="could not look up"):
            store.find_job("j_err1")
        with pytest.raises(JobStoreError, match="could not update"):
            store.update(record, terminal=True)
        with pytest.raises(JobStoreError, match="could not prune"):
            store.prune_terminal({"j_err1"})
        with pytest.raises(JobStoreError, match="could not recover"):
            store.load_and_interrupt()
        store._conn = real_conn
    finally:
        store.close()
