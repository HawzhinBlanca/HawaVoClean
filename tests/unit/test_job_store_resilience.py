"""Resilience, recovery, and degradation matrix for DurableJobStore (Phase E1.8).

Qualifies:
1. Schema migration from N-1 (v1) to N (v2) with terminal_at and history_visible.
2. Interrupted migration and future schema version rejection with rollback.
3. Corrupt-row resilience: corrupted active and terminal record_json blobs
   are quarantined without destroying readable history or holding output leases.
4. Corrupt artifact / missing output verification with actionable error reporting.
5. Disk full (ENOSPC) during reservation and update triggers clean rollback and DiskFullError.
6. Read-only volume loss fails closed with StorageReadOnlyError.
7. WAL crash recovery: uncheckpointed WAL frames are replayed upon restart,
   interrupted jobs are transitioned cleanly, and output leases are released.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from hawavoclean.errors import PublicationError
from hawavoclean.record_bundle import verify_processing_record
from hawavoclean.server.app import _job_artifact_path
from hawavoclean.server.job_store import (
    DiskFullError,
    DurableJobStore,
    JobStoreError,
    StorageReadOnlyError,
    _now_iso,
)


class _FailingConnectionProxy:
    """Proxy around sqlite3.Connection to simulate SQLite errors on demand."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        fail_executescript_phrase: str | None = None,
        fail_execute_phrase: str | None = None,
        error_to_raise: Exception | None = None,
    ) -> None:
        self._conn = conn
        self._fail_executescript_phrase = fail_executescript_phrase
        self._fail_execute_phrase = fail_execute_phrase
        self._error_to_raise = error_to_raise or sqlite3.OperationalError("simulated error")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def executescript(self, script: str) -> sqlite3.Cursor:
        if self._fail_executescript_phrase and self._fail_executescript_phrase in script:
            raise self._error_to_raise
        return self._conn.executescript(script)

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if self._fail_execute_phrase and self._fail_execute_phrase in sql:
            raise self._error_to_raise
        return self._conn.execute(sql, *args, **kwargs)


def _create_v1_store(db_path: Path) -> None:
    """Create a legacy schema version 1 job store database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(
        """
        BEGIN IMMEDIATE;
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE,
            request_hash TEXT NOT NULL,
            output_key TEXT NOT NULL,
            output_path TEXT NOT NULL,
            state TEXT NOT NULL,
            terminal INTEGER NOT NULL CHECK (terminal IN (0, 1)),
            record_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX active_output_lease
            ON jobs(output_key) WHERE terminal = 0;
        CREATE INDEX jobs_updated_at ON jobs(updated_at DESC);
        PRAGMA user_version = 1;
        COMMIT;
        """
    )
    now_iso = _now_iso()
    completed_record = {
        "job_id": "v1_completed_job",
        "state": "done",
        "output_path": "/tmp/out_v1.wav",
        "created_at": now_iso,
    }
    active_record = {
        "job_id": "v1_active_job",
        "state": "running",
        "output_path": "/tmp/out_v1_active.wav",
        "created_at": now_iso,
    }
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, idempotency_key, request_hash, output_key,
            output_path, state, terminal, record_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "v1_completed_job",
            "idem_v1",
            "req_hash_1",
            "out_v1",
            "/tmp/out_v1.wav",
            "done",
            1,
            json.dumps(completed_record),
            now_iso,
            now_iso,
        ),
    )
    conn.execute(
        """
        INSERT INTO jobs (
            job_id, idempotency_key, request_hash, output_key,
            output_path, state, terminal, record_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "v1_active_job",
            "idem_v1_active",
            "req_hash_2",
            "out_v1_active",
            "/tmp/out_v1_active.wav",
            "running",
            0,
            json.dumps(active_record),
            now_iso,
            now_iso,
        ),
    )
    conn.commit()
    conn.close()


def test_schema_migration_v1_to_v3(tmp_path: Path) -> None:
    """Test legacy (v1) to current (v3) schema migration backfills terminal_at, history_visible, and batch_id."""
    db_path = tmp_path / "ledger_v1.sqlite3"
    _create_v1_store(db_path)

    store = DurableJobStore(db_path)
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
            version = raw_conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == 3

            cols = {row[1] for row in raw_conn.execute("PRAGMA table_info(jobs)").fetchall()}
            assert "terminal_at" in cols
            assert "history_visible" in cols
            assert "batch_id" in cols

            tables = {
                row[0]
                for row in raw_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "batches" in tables

            indexes = {row[1] for row in raw_conn.execute("PRAGMA index_list(jobs)").fetchall()}
            assert "visible_terminal_retention" in indexes
            assert "jobs_batch_id" in indexes

        loaded = store.load_and_interrupt(max_terminal_jobs=10, terminal_ttl_s=7 * 86400.0)
        by_id = {j["job_id"]: j for j in loaded}
        assert "v1_completed_job" in by_id
        assert by_id["v1_completed_job"]["state"] == "done"
        assert by_id["v1_completed_job"]["_history_visible"] is True
        assert by_id["v1_completed_job"]["_terminal_at_epoch"] is not None

        interrupted = store.find_job("v1_active_job")
        assert interrupted is not None
        assert interrupted.record["state"] == "interrupted"
        assert interrupted.record["error"]["code"] == "INTERRUPTED"
    finally:
        store.close()


def test_schema_migration_v2_to_v3(tmp_path: Path) -> None:
    """Test N-1 (v2) to N (v3) schema migration adds batch_id and batches table."""
    db_path = tmp_path / "ledger_v2.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(
        """
        BEGIN IMMEDIATE;
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE,
            request_hash TEXT NOT NULL,
            output_key TEXT NOT NULL,
            output_path TEXT NOT NULL,
            state TEXT NOT NULL,
            terminal INTEGER NOT NULL CHECK (terminal IN (0, 1)),
            record_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            terminal_at REAL,
            history_visible INTEGER NOT NULL DEFAULT 1
                CHECK (history_visible IN (0, 1))
        );
        CREATE UNIQUE INDEX active_output_lease
            ON jobs(output_key) WHERE terminal = 0;
        CREATE INDEX jobs_updated_at ON jobs(updated_at DESC);
        CREATE INDEX visible_terminal_retention
            ON jobs(history_visible, terminal, terminal_at DESC, created_at DESC);
        PRAGMA user_version = 2;
        COMMIT;
        """
    )
    conn.close()

    store = DurableJobStore(db_path)
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
            version = raw_conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == 3

            cols = {row[1] for row in raw_conn.execute("PRAGMA table_info(jobs)").fetchall()}
            assert "batch_id" in cols

            tables = {
                row[0]
                for row in raw_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "batches" in tables

            indexes = {row[1] for row in raw_conn.execute("PRAGMA index_list(jobs)").fetchall()}
            assert "jobs_batch_id" in indexes

        # Verify batch methods work seamlessly on migrated store
        store.create_batch("batch_test_1", 2, options={"profile": "balanced"})
        b = store.get_batch("batch_test_1")
        assert b is not None
        assert b["batch_id"] == "batch_test_1"
        assert b["total_items"] == 2
        assert b["state"] == "queued"
        assert b["options"] == {"profile": "balanced"}

        store.update_batch_state("batch_test_1", "running")
        b = store.get_batch("batch_test_1")
        assert b is not None
        assert b["state"] == "running"

        batches = store.list_batches()
        assert len(batches) == 1
        assert batches[0]["batch_id"] == "batch_test_1"
    finally:
        store.close()


def test_future_schema_version_rejected(tmp_path: Path) -> None:
    """Test opening a store with a future schema version raises actionable JobStoreError."""
    db_path = tmp_path / "future_store.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 999")
    conn.commit()
    conn.close()

    with pytest.raises(JobStoreError, match="newer than supported"):
        DurableJobStore(db_path)


def test_interrupted_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a failure during migration executes rollback and does not corrupt the db."""
    db_path = tmp_path / "interrupted_migration.sqlite3"
    _create_v1_store(db_path)

    orig_connect = sqlite3.connect

    def failing_connect(database: Any, **kwargs: Any) -> Any:
        conn = orig_connect(database, **kwargs)
        return _FailingConnectionProxy(
            conn,
            fail_executescript_phrase="ALTER TABLE jobs ADD COLUMN terminal_at",
            error_to_raise=sqlite3.OperationalError(
                "simulated interrupted disk failure during DDL"
            ),
        )

    monkeypatch.setattr(sqlite3, "connect", failing_connect)

    with pytest.raises(JobStoreError, match="migration failed"):
        DurableJobStore(db_path)

    with contextlib.closing(orig_connect(str(db_path))) as raw_conn:
        assert raw_conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_corrupt_row_does_not_destroy_readable_history(tmp_path: Path) -> None:
    """Test that corrupted JSON in individual rows is quarantined and history is preserved."""
    db_path = tmp_path / "corrupted_rows.sqlite3"
    store = DurableJobStore(db_path)
    output_base = tmp_path / "outputs"
    output_base.mkdir()

    out1 = output_base / "job1.wav"
    out2 = output_base / "job2.wav"
    out3 = output_base / "job3.wav"

    store.reserve(
        record={"job_id": "job1", "state": "queued", "output_path": str(out1)},
        request_hash="hash1",
        idempotency_key="key1",
        conflict_policy="fail",
    )
    store.reserve(
        record={"job_id": "job2", "state": "queued", "output_path": str(out2)},
        request_hash="hash2",
        idempotency_key="key2",
        conflict_policy="fail",
    )
    store.reserve(
        record={"job_id": "job3", "state": "running", "output_path": str(out3)},
        request_hash="hash3",
        idempotency_key="key3",
        conflict_policy="fail",
    )

    store.update({"job_id": "job1", "state": "done", "output_path": str(out1)}, terminal=True)
    store.update({"job_id": "job2", "state": "done", "output_path": str(out2)}, terminal=True)
    store.close()

    with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
        raw_conn.execute(
            "UPDATE jobs SET record_json = '{\"corrupted_json_truncated:' WHERE job_id = 'job2'"
        )
        raw_conn.execute("UPDATE jobs SET record_json = '<<NOT_EVEN_JSON>>' WHERE job_id = 'job3'")
        raw_conn.commit()

    reopened = DurableJobStore(db_path)
    try:
        loaded = reopened.load_and_interrupt(max_terminal_jobs=10)
        by_id = {j["job_id"]: j for j in loaded}

        # 1. Valid job1 was loaded intact
        assert "job1" in by_id
        assert by_id["job1"]["state"] == "done"

        # 2. Corrupted terminal job2 was quarantined as failed without crashing history
        assert "job2" in by_id
        assert by_id["job2"]["state"] == "failed"
        assert by_id["job2"]["error"]["code"] == "CORRUPT_LEDGER_ROW"

        # 3. Corrupted active job3 was transitioned to interrupted and terminal=1
        job3_res = reopened.find_job("job3")
        assert job3_res is not None
        assert job3_res.record["state"] == "interrupted"

        # 4. Output lease for out3 was released
        res4 = reopened.reserve(
            record={"job_id": "job4", "state": "queued", "output_path": str(out3)},
            request_hash="hash4",
            idempotency_key="key4",
            conflict_policy="fail",
        )
        assert res4.reused is False
        assert res4.record["job_id"] == "job4"

        # 5. find_job on corrupt row returns quarantined tombstone
        with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
            raw_conn.execute(
                "UPDATE jobs SET record_json = 'INVALID_JSON_BLOB' WHERE job_id = 'job1'"
            )
            raw_conn.commit()

        res1_corrupt = reopened.find_job("job1")
        assert res1_corrupt is not None
        assert res1_corrupt.record["state"] == "failed"
        assert res1_corrupt.record["error"]["code"] == "CORRUPT_LEDGER_ROW"
    finally:
        reopened.close()


def test_disk_full_rolls_back_without_corruption(tmp_path: Path) -> None:
    """Test disk full (ENOSPC) during reserve or update raises DiskFullError and rolls back."""
    db_path = tmp_path / "disk_full.sqlite3"
    store = DurableJobStore(db_path)
    out1 = tmp_path / "out1.wav"

    # Wrap store._conn with proxy
    real_conn = store._conn
    store._conn = _FailingConnectionProxy(  # type: ignore[assignment]
        real_conn,
        fail_execute_phrase="INSERT INTO jobs",
        error_to_raise=sqlite3.OperationalError("database or disk is full"),
    )

    with pytest.raises(DiskFullError, match="disk is full"):
        store.reserve(
            record={"job_id": "full_job", "state": "queued", "output_path": str(out1)},
            request_hash="full_hash",
            idempotency_key="full_key",
            conflict_policy="fail",
        )

    # Restore real connection and verify DB is cleanly usable
    store._conn = real_conn

    res = store.reserve(
        record={"job_id": "full_job", "state": "queued", "output_path": str(out1)},
        request_hash="full_hash",
        idempotency_key="full_key",
        conflict_policy="fail",
    )
    assert res.reused is False
    assert res.record["job_id"] == "full_job"

    # Test disk full on update
    store._conn = _FailingConnectionProxy(  # type: ignore[assignment]
        real_conn,
        fail_execute_phrase="UPDATE jobs",
        error_to_raise=sqlite3.OperationalError("database or disk is full"),
    )
    with pytest.raises(DiskFullError, match="disk is full"):
        store.update({"job_id": "full_job", "state": "done"}, terminal=True)

    store._conn = real_conn
    store.close()


def test_storage_read_only_volume_loss(tmp_path: Path) -> None:
    """Test write operations fail closed with StorageReadOnlyError when volume is read-only."""
    db_path = tmp_path / "readonly.sqlite3"
    store = DurableJobStore(db_path)
    out1 = tmp_path / "readonly.wav"

    real_conn = store._conn
    store._conn = _FailingConnectionProxy(  # type: ignore[assignment]
        real_conn,
        fail_execute_phrase="INSERT INTO jobs",
        error_to_raise=sqlite3.OperationalError("attempt to write a readonly database"),
    )

    with pytest.raises(StorageReadOnlyError, match="read-only"):
        store.reserve(
            record={"job_id": "ro_job", "state": "queued", "output_path": str(out1)},
            request_hash="ro_hash",
            idempotency_key="ro_key",
            conflict_policy="fail",
        )
    store._conn = real_conn
    store.close()


def test_wal_recovery_after_simulated_crash(tmp_path: Path) -> None:
    """Test that SQLite WAL recovery automatically replays uncheckpointed frames after abrupt kill."""
    db_path = tmp_path / "wal_crash.sqlite3"

    store = DurableJobStore(db_path)
    out1 = tmp_path / "crash_job1.wav"
    out2 = tmp_path / "crash_job2.wav"

    store.reserve(
        record={"job_id": "crash1", "state": "running", "output_path": str(out1)},
        request_hash="hash1",
        idempotency_key="crash1",
        conflict_policy="fail",
    )
    store.reserve(
        record={"job_id": "crash2", "state": "done", "output_path": str(out2)},
        request_hash="hash2",
        idempotency_key="crash2",
        conflict_policy="fail",
    )
    store.update({"job_id": "crash2", "state": "done", "output_path": str(out2)}, terminal=True)

    # Abrupt kill without checkpoint
    store._conn.close()
    store._closed = True

    reopened = DurableJobStore(db_path)
    try:
        loaded = reopened.load_and_interrupt(max_terminal_jobs=10)
        by_id = {j["job_id"]: j for j in loaded}

        assert "crash2" in by_id
        assert by_id["crash2"]["state"] == "done"

        job1 = reopened.find_job("crash1")
        assert job1 is not None
        assert job1.record["state"] == "interrupted"

        res = reopened.reserve(
            record={"job_id": "crash1_new", "state": "queued", "output_path": str(out1)},
            request_hash="hash1_new",
            idempotency_key="crash1_retry",
            conflict_policy="fail",
        )
        assert res.reused is False
    finally:
        reopened.close()


def test_corrupt_artifact_surfaces_actionable_error(tmp_path: Path) -> None:
    """Test that missing or corrupted output artifacts fail verification cleanly."""
    missing_path = tmp_path / "missing_master.wav"
    snapshot_missing: dict[str, Any] = {
        "output_path": str(missing_path),
        "state": "done",
        "audio_sha256": "abcdef1234567890" * 4,
    }

    assert _job_artifact_path(snapshot_missing, "master") is None

    corrupt_record = tmp_path / "corrupt_record.zip"
    corrupt_record.write_bytes(b"PK\x03\x04corrupted_truncated_bytes")

    with pytest.raises((PublicationError, zipfile.BadZipFile)):
        verify_processing_record(corrupt_record)
