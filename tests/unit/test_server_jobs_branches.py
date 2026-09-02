"""Targeted branch coverage tests for server/jobs.py helper functions and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from hawavoclean.server.job_store import JobStoreError
from hawavoclean.server.jobs import JobManager, JobRecord


def _sample_record(tmp_path: Path) -> JobRecord:
    return JobRecord(
        job_id="j1",
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.json",
        profile="production",
        overwrite=False,
        idempotency_key="key-1",
        conflict_policy="fail",
        request_hash="a" * 64,
        mode="natural",
        speaker_id=None,
        cutoff_hz=None,
        record_bundle=False,
        bundle_path=None,
        state="queued",
        stage="preflight",
        progress=0.0,
        message="Queued",
        unit=None,
        created_at="2026-08-01T00:00:00Z",
        started_at=None,
        finished_at=None,
        error=None,
        report=None,
        cancel_requested=False,
        terminal_at=None,
        bundle=None,
        artifact_evidence=None,
        source_snapshot_path=None,
        source_snapshot_dir=None,
        source_sha256=None,
        source_size_bytes=None,
    )


def test_submit_idempotency_key_and_policy_validation(tmp_path: Path) -> None:
    manager = JobManager()
    try:
        in_wav = tmp_path / "in.wav"
        in_wav.touch()
        out_wav = tmp_path / "out.wav"

        # 1. Empty or whitespace idempotency key
        with pytest.raises(ValueError, match="1-128 characters"):
            manager.submit(
                input_path=in_wav,
                output_path=out_wav,
                profile="production",
                overwrite=False,
                idempotency_key="   ",
            )

        # 2. Non-visible ASCII idempotency key
        with pytest.raises(ValueError, match="visible ASCII"):
            manager.submit(
                input_path=in_wav,
                output_path=out_wav,
                profile="production",
                overwrite=False,
                idempotency_key="key with space",
            )

        # 3. Unsupported conflict policy
        with pytest.raises(ValueError, match="unsupported conflict policy"):
            manager.submit(
                input_path=in_wav,
                output_path=out_wav,
                profile="production",
                overwrite=False,
                conflict_policy="unsupported",  # type: ignore[arg-type]
            )

        # 4. Closed manager
        manager.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            manager.submit(
                input_path=in_wav, output_path=out_wav, profile="production", overwrite=False
            )
    finally:
        manager.shutdown()


def test_report_audio_sha256_validation() -> None:
    # 1. Missing output
    with pytest.raises(JobStoreError, match="canonical output SHA-256"):
        JobManager._report_audio_sha256({})

    # 2. Output sha256 not 64 hex characters
    with pytest.raises(JobStoreError, match="canonical output SHA-256"):
        JobManager._report_audio_sha256({"output": {"sha256": "not_hex"}})

    valid_sha = "a" * 64
    assert JobManager._report_audio_sha256({"output": {"sha256": valid_sha}}) == valid_sha


def test_closed_bundle_evidence_validation(tmp_path: Path) -> None:
    rec = _sample_record(tmp_path)

    # 1. Record bundle path is None
    with pytest.raises(JobStoreError, match="lacks closed durable evidence"):
        JobManager._closed_bundle_evidence(rec, {})

    rec.bundle_path = tmp_path / "bundle.zip"
    valid_evidence = {
        "path": str(tmp_path / "bundle.zip"),
        "archive_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "master_sha256": "c" * 64,
        "report_sha256": "d" * 64,
        "summary_sha256": "e" * 64,
        "total_uncompressed_bytes": 1000,
        "internal_hashes_verified": True,
        "authenticated_publisher": True,
    }

    # 2. Mismatched path
    bad_path_evidence = dict(valid_evidence, path=str(tmp_path / "different.zip"))
    with pytest.raises(JobStoreError, match="evidence path differs"):
        JobManager._closed_bundle_evidence(rec, bad_path_evidence)

    # 3. Bad digest
    bad_digest = dict(valid_evidence, archive_sha256="bad_sha")
    with pytest.raises(JobStoreError, match="archive_sha256 is invalid"):
        JobManager._closed_bundle_evidence(rec, bad_digest)

    # 4. Bad uncompressed bytes
    bad_size = dict(valid_evidence, total_uncompressed_bytes=0)
    with pytest.raises(JobStoreError, match="uncompressed size evidence is invalid"):
        JobManager._closed_bundle_evidence(rec, bad_size)

    # 5. Bad internal_hashes_verified
    bad_verified = dict(valid_evidence, internal_hashes_verified=False)
    with pytest.raises(JobStoreError, match="verification evidence is invalid"):
        JobManager._closed_bundle_evidence(rec, bad_verified)

    # 6. Valid evidence passes
    assert JobManager._closed_bundle_evidence(rec, valid_evidence) == valid_evidence


def test_artifact_record_and_capture_error_branches(tmp_path: Path) -> None:
    # 1. Missing file in _artifact_record
    with pytest.raises(JobStoreError, match="missing or unsafe"):
        JobManager._artifact_record(tmp_path / "nonexistent.wav")

    # 2. _validate_bundle_artifacts when record_bundle is False
    manager = JobManager()
    try:
        rec = _sample_record(tmp_path)
        rec.record_bundle = False
        with pytest.raises(JobStoreError, match="missing its durable bundle path"):
            manager._validate_bundle_artifacts(rec)

        # 3. _capture_nonbundle_artifacts when report is None
        rec.report = None
        with pytest.raises(JobStoreError, match="has no report object"):
            manager._capture_nonbundle_artifacts(rec)
    finally:
        manager.shutdown()
