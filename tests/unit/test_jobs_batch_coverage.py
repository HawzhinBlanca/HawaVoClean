"""Targeted coverage tests for hawavoclean.server.jobs batch, retry, and lifecycle operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from hawavoclean.errors import MediaPreflightError
from hawavoclean.server.jobs import JobManager


def _dummy_command(record: Any) -> list[str]:
    return ["echo", record.job_id]


def test_job_manager_init_and_submit_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_active_jobs must be at least 1"):
        JobManager(max_active_jobs=0)
    with pytest.raises(ValueError, match="max_terminal_jobs must be at least 1"):
        JobManager(max_terminal_jobs=0)
    with pytest.raises(ValueError, match="terminal_ttl_s must be positive"):
        JobManager(terminal_ttl_s=0)

    manager = JobManager(store_path=None, command_factory=_dummy_command)
    try:
        in_f = tmp_path / "val_in.wav"
        in_f.write_bytes(b"RIFF" + b"\x00" * 40)
        out_f = tmp_path / "val_out.wav"

        # 1. Empty idempotency key
        with pytest.raises(ValueError, match="idempotency_key must contain 1-128 characters"):
            manager.submit(
                input_path=in_f,
                output_path=out_f,
                profile="production",
                overwrite=True,
                idempotency_key="   ",
            )

        # 2. Too long idempotency key
        with pytest.raises(ValueError, match="idempotency_key must contain 1-128 characters"):
            manager.submit(
                input_path=in_f,
                output_path=out_f,
                profile="production",
                overwrite=True,
                idempotency_key="a" * 129,
            )

        # 3. Non-visible ASCII idempotency key
        with pytest.raises(ValueError, match="visible ASCII"):
            manager.submit(
                input_path=in_f,
                output_path=out_f,
                profile="production",
                overwrite=True,
                idempotency_key="key with spaces",
            )

        # 4. Invalid conflict policy
        with pytest.raises(ValueError, match="unsupported conflict policy"):
            manager.submit(
                input_path=in_f,
                output_path=out_f,
                profile="production",
                overwrite=True,
                conflict_policy="invalid_policy",  # type: ignore[arg-type]
            )
    finally:
        manager.shutdown()


def test_jobs_batch_pause_resume_cancel(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    manager = JobManager(store_path=db_path, command_factory=_dummy_command)
    try:
        # 1. Batch does not exist
        assert manager.pause_batch("nonexistent_batch") is False
        assert manager.resume_batch("nonexistent_batch") is False
        assert manager.cancel_batch("nonexistent_batch") is False
        assert manager.get_batch_summary("nonexistent_batch") is None

        # 2. Register batch
        manager.register_batch("batch_1", 2, options={"profile": "production"})

        # Submit jobs under batch_1
        in_file1 = tmp_path / "in1.wav"
        in_file1.write_bytes(b"RIFF" + b"\x00" * 40)
        in_file2 = tmp_path / "in2.wav"
        in_file2.write_bytes(b"RIFF" + b"\x00" * 40)

        manager.submit(
            input_path=in_file1,
            output_path=tmp_path / "out1.wav",
            profile="production",
            overwrite=True,
            batch_id="batch_1",
        )
        manager.submit(
            input_path=in_file2,
            output_path=tmp_path / "out2.wav",
            profile="production",
            overwrite=True,
            batch_id="batch_1",
        )

        # 3. Pause batch
        assert manager.pause_batch("batch_1") is True
        summary = manager.get_batch_summary("batch_1")
        assert summary is not None
        assert summary["batch_id"] == "batch_1"
        assert summary["state"] in ("paused", "running")

        # 4. Resume batch
        assert manager.resume_batch("batch_1") is True

        # 5. List batches with store
        batches = manager.list_batches(limit=10)
        assert any(b["batch_id"] == "batch_1" for b in batches)

        # 6. Cancel batch
        assert manager.cancel_batch("batch_1", wait=True) is True
        summary_after_cancel = manager.get_batch_summary("batch_1")
        assert summary_after_cancel is not None
        assert summary_after_cancel["cancelled_items"] > 0
    finally:
        manager.shutdown()

    # When shut down, operations raise RuntimeError
    with pytest.raises(RuntimeError, match="shut down"):
        manager.pause_batch("batch_1")
    with pytest.raises(RuntimeError, match="shut down"):
        manager.resume_batch("batch_1")


def test_jobs_batch_summary_states_and_in_memory_listing(tmp_path: Path) -> None:
    # Test manager without durable store (store is None)
    manager = JobManager(store_path=None, command_factory=_dummy_command)
    try:
        # register_batch with no store is safe no-op
        manager.register_batch("mem_batch", 2)
        assert manager.list_batches() == []

        in_file1 = tmp_path / "mem1.wav"
        in_file1.write_bytes(b"RIFF" + b"\x00" * 40)
        manager.submit(
            input_path=in_file1,
            output_path=tmp_path / "mem_out1.wav",
            profile="production",
            overwrite=True,
            batch_id="mem_batch",
        )

        # Listing now includes mem_batch
        b_list = manager.list_batches(limit=5)
        assert len(b_list) == 1
        assert b_list[0]["batch_id"] == "mem_batch"

        summary = manager.get_batch_summary("mem_batch")
        assert summary is not None
        assert summary["total_items"] >= 1
    finally:
        manager.shutdown()


def test_jobs_retry_job_branches(tmp_path: Path) -> None:
    db_path = tmp_path / "retry_jobs.sqlite3"
    manager = JobManager(store_path=db_path, command_factory=_dummy_command)
    try:
        # 1. Non-existent job
        with pytest.raises(KeyError, match="not found"):
            manager.retry_job("nonexistent_id")

        # Submit a real job
        in_file = tmp_path / "in_retry.wav"
        in_file.write_bytes(b"RIFF" + b"\x00" * 40)
        snap = manager.submit(
            input_path=in_file,
            output_path=tmp_path / "out_retry.wav",
            profile="production",
            overwrite=True,
        )
        job_id = snap["job_id"]

        # 2. Non-terminal job retry returns current snapshot without error
        retried_snap = manager.retry_job(job_id)
        assert retried_snap["job_id"] == job_id

        # Cancel the job to make it terminal
        manager.cancel(job_id, wait=True)

        # 3. Successful retry of terminal job
        retried = manager.retry_job(job_id)
        assert retried["job_id"] == job_id
        assert retried["state"] == "queued"

        # Cancel again
        manager.cancel(job_id, wait=True)

        # 4. Retry when input file is missing raises MediaPreflightError
        in_file.unlink()
        with pytest.raises(MediaPreflightError):
            manager.retry_job(job_id)
    finally:
        manager.shutdown()

    # 5. Retry on shutdown manager raises RuntimeError
    with pytest.raises(RuntimeError, match="shut down"):
        manager.retry_job(job_id)


def test_jobs_subscribe_unsubscribe_and_callbacks(tmp_path: Path) -> None:
    manager = JobManager(store_path=None, command_factory=_dummy_command)
    try:
        # Subscribe to nonexistent job returns None
        async def _test_sub() -> None:
            q = manager.subscribe("no_such_job")
            assert q is None

            in_f = tmp_path / "sub_in.wav"
            in_f.write_bytes(b"RIFF" + b"\x00" * 40)
            j = manager.submit(
                input_path=in_f,
                output_path=tmp_path / "sub_out.wav",
                profile="production",
                overwrite=True,
            )
            q2 = manager.subscribe(j["job_id"])
            assert q2 is not None
            manager.unsubscribe(j["job_id"], q2)

        asyncio.run(_test_sub())

        # Terminal callback with exception is logged safely
        mock_cb = MagicMock(side_effect=RuntimeError("callback explosive failure"))
        manager.add_terminal_callback(mock_cb)

        in_cb = tmp_path / "cb_in.wav"
        in_cb.write_bytes(b"RIFF" + b"\x00" * 40)
        j_cb = manager.submit(
            input_path=in_cb,
            output_path=tmp_path / "cb_out.wav",
            profile="production",
            overwrite=True,
        )
        manager.cancel(j_cb["job_id"], wait=True)
        assert mock_cb.called
    finally:
        manager.shutdown()


def test_job_store_exclusive_lock_and_batch_state_variants(tmp_path: Path) -> None:
    from hawavoclean.server.job_store import JobStoreError

    db_path = tmp_path / "locked_jobs.sqlite3"
    manager1 = JobManager(store_path=db_path, command_factory=_dummy_command)
    try:
        # 1. Double acquisition of same store path fails with JobStoreError
        with pytest.raises(JobStoreError, match="another engine broker already owns"):
            JobManager(store_path=db_path, command_factory=_dummy_command)

        # 2. Batch summary with 0 items registered
        manager1.register_batch("empty_batch", 0)
        summary_empty = manager1.get_batch_summary("empty_batch")
        assert summary_empty is not None
        assert summary_empty["total_items"] == 0
        assert summary_empty["progress"] == 0.0

        # 3. Batch summary cancelled state when all cancelled
        in_f = tmp_path / "all_canc.wav"
        in_f.write_bytes(b"RIFF" + b"\x00" * 40)
        manager1.register_batch("canc_batch", 1)
        j = manager1.submit(
            input_path=in_f,
            output_path=tmp_path / "all_canc_out.wav",
            profile="production",
            overwrite=True,
            batch_id="canc_batch",
        )
        manager1.cancel(j["job_id"], wait=True)
        canc_summary = manager1.get_batch_summary("canc_batch")
        assert canc_summary is not None
        assert canc_summary["state"] == "cancelled"
    finally:
        manager1.shutdown()
