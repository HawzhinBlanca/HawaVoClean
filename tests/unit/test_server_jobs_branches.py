from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from hawavoclean.server.job_store import JobStoreError
from hawavoclean.server.jobs import JobManager, JobRecord


def _dummy_command_factory(*_args: object, **_kwargs: object) -> list[str]:
    return ["true"]


def _dummy_record(tmp_path: Path) -> JobRecord:
    out = tmp_path / "out.wav"
    out.write_bytes(b"RIFFdummywav")
    rep = tmp_path / "out.hawavoclean.json"
    rep.write_text(json.dumps({"output": {"sha256": "fake_sha"}}), encoding="utf-8")
    summ = tmp_path / "out.hawavoclean.summary.json"
    summ.write_text("{}", encoding="utf-8")

    return JobRecord(
        job_id="job_test_01",
        input_path=tmp_path / "in.wav",
        output_path=out,
        profile="production",
        overwrite=False,
        report_path=rep,
        artifact_evidence={
            "schema_version": 1,
            "storage": "legacy_flat",
            "generation_id": None,
            "audio": {"sha256": "fake_sha", "size_bytes": 12},
            "report": {"sha256": "fake_rep", "size_bytes": 10},
            "summary": {"sha256": "fake_sum", "size_bytes": 2},
        },
    )


def test_validate_nonbundle_artifacts_schema_and_keys(tmp_path: Path) -> None:
    mgr = JobManager(command_factory=_dummy_command_factory, max_active_jobs=1)
    rec = _dummy_record(tmp_path)

    # 1. Invalid evidence type or keys
    rec.artifact_evidence = cast(Any, None)
    with pytest.raises(JobStoreError, match="lacks closed artifact evidence"):
        mgr._validate_nonbundle_artifacts(rec)

    # 2. Unsupported schema version
    rec.artifact_evidence = {"schema_version": 999}
    with pytest.raises(JobStoreError, match="lacks closed artifact evidence"):
        mgr._validate_nonbundle_artifacts(rec)

    # 3. Bad storage kind
    rec = _dummy_record(tmp_path)
    assert isinstance(rec.artifact_evidence, dict)
    rec.artifact_evidence["storage"] = "unknown_storage"
    with pytest.raises(JobStoreError, match="storage kind is invalid"):
        mgr._validate_nonbundle_artifacts(rec)

    # 4. Bad role evidence structure
    rec = _dummy_record(tmp_path)
    assert isinstance(rec.artifact_evidence, dict)
    rec.artifact_evidence["audio"] = {"bad": 1}
    with pytest.raises(JobStoreError, match="audio evidence is invalid"):
        mgr._validate_nonbundle_artifacts(rec)

    # 5. Missing digest string
    rec = _dummy_record(tmp_path)
    assert isinstance(rec.artifact_evidence, dict)
    rec.artifact_evidence["audio"] = {"sha256": 12345, "size_bytes": 10}
    with pytest.raises(JobStoreError, match="digest is missing"):
        mgr._validate_nonbundle_artifacts(rec)


def test_validate_nonbundle_artifacts_storage_and_report_failures(tmp_path: Path) -> None:
    mgr = JobManager(command_factory=_dummy_command_factory, max_active_jobs=1)
    rec = _dummy_record(tmp_path)
    assert isinstance(rec.artifact_evidence, dict)

    # 1. Immutable generation missing
    rec.artifact_evidence["storage"] = "immutable_generation"
    rec.artifact_evidence["generation_id"] = "gen_01"
    valid_sha = "a" * 64
    rec.artifact_evidence["audio"]["sha256"] = valid_sha
    rec.artifact_evidence["report"]["sha256"] = valid_sha
    rec.artifact_evidence["summary"]["sha256"] = valid_sha
    with (
        patch(
            "hawavoclean.server.jobs.resolve_immutable_publication_generation", return_value=None
        ),
        pytest.raises(JobStoreError, match="immutable publication generation is missing"),
    ):
        mgr._validate_nonbundle_artifacts(rec)

    # 2. Artifact record digest mismatch
    rec = _dummy_record(tmp_path)
    assert isinstance(rec.artifact_evidence, dict)
    rec.artifact_evidence["audio"]["sha256"] = valid_sha
    rec.artifact_evidence["report"]["sha256"] = valid_sha
    rec.artifact_evidence["summary"]["sha256"] = valid_sha
    with (
        patch(
            "hawavoclean.server.jobs.resolve_immutable_publication_generation", return_value=None
        ),
        patch.object(mgr, "_artifact_record", return_value={"sha256": "b" * 64, "size_bytes": 10}),
        pytest.raises(JobStoreError, match="failed digest validation"),
    ):
        mgr._validate_nonbundle_artifacts(rec)


def test_resolve_bundle_artifacts_failures(tmp_path: Path) -> None:
    mgr = JobManager(command_factory=_dummy_command_factory, max_active_jobs=1)
    rec = _dummy_record(tmp_path)
    valid_sha = "a" * 64
    evidence = {
        "master_sha256": valid_sha,
        "report_sha256": valid_sha,
        "summary_sha256": valid_sha,
    }

    # 1. Immutable generation missing when current is not None and exact is None
    with (
        patch(
            "hawavoclean.server.jobs.resolve_immutable_publication_generation", return_value=None
        ),
        patch(
            "hawavoclean.server.jobs.resolve_committed_publication",
            return_value=(rec.output_path, rec.report_path, Path("s")),
        ),
        pytest.raises(JobStoreError, match="immutable publication generation is missing"),
    ):
        mgr._bundle_artifact_paths(rec, evidence)

    # 2. Legacy differs from bundle evidence
    with (
        patch(
            "hawavoclean.server.jobs.resolve_immutable_publication_generation", return_value=None
        ),
        patch("hawavoclean.server.jobs.resolve_committed_publication", return_value=None),
        patch.object(mgr, "_artifacts_match_bundle_evidence", return_value=False),
        pytest.raises(JobStoreError, match="differs from the legacy export triplet"),
    ):
        mgr._bundle_artifact_paths(rec, evidence)


def test_prepare_batch_rollback_store_error(tmp_path: Path) -> None:
    fake_store = MagicMock()
    fake_store.delete_queued.side_effect = JobStoreError("simulated store delete failure")

    mgr = JobManager(
        command_factory=_dummy_command_factory,
        max_active_jobs=1,
    )
    mgr._store = fake_store

    in_file = tmp_path / "in.wav"
    in_file.write_bytes(b"RIFFtest")
    out_file = tmp_path / "out.wav"

    with (
        pytest.raises(JobStoreError, match="batch preparation failed and rollback was not durable"),
        mgr.prepare_batch(),
    ):
        mgr.submit(input_path=in_file, output_path=out_file, profile="production", overwrite=False)
        # Force an exception to trigger prepare_batch rollback
        raise RuntimeError("deliberate batch inner abort")

    assert mgr._persistence_error is not None
    assert "simulated store delete failure" in mgr._persistence_error


def test_retry_job_rejects_active_output_conflict_and_rolls_back_on_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from hawavoclean.server.job_store import OutputConflictError

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(work_dir))

    in_file = tmp_path / "in.wav"
    in_file.write_bytes(b"RIFFtest")
    out_file = tmp_path / "out.wav"

    # 1. Durable store: active output conflict raises OutputConflictError without
    #    mutating j1 to stuck "queued" or leaking PinnedSource directories.
    calls = [0]

    def _cmd_factory(_rec: JobRecord) -> list[str]:
        calls[0] += 1
        if calls[0] == 1:
            return [sys.executable, "-c", "import sys; sys.exit(1)"]
        return [sys.executable, "-c", "import time; time.sleep(30)"]

    import sys

    mgr = JobManager(
        command_factory=_cmd_factory,
        store_path=tmp_path / "jobs.sqlite",
        kill_grace_s=0.2,
    )
    try:
        j1 = mgr.submit(
            input_path=in_file, output_path=out_file, profile="production", overwrite=True
        )
        for _ in range(100):
            st1 = mgr.get_status(j1["job_id"])
            if st1 and st1["state"] == "failed":
                break
            time.sleep(0.02)
        st1_failed = mgr.get_status(j1["job_id"])
        assert st1_failed is not None and st1_failed["state"] == "failed"

        j2 = mgr.submit(
            input_path=in_file, output_path=out_file, profile="production", overwrite=True
        )
        for _ in range(100):
            st2 = mgr.get_status(j2["job_id"])
            if st2 and st2["state"] == "running":
                break
            time.sleep(0.02)

        before_dirs = set(work_dir.iterdir())
        with pytest.raises(OutputConflictError, match="reserved by an active job"):
            mgr.retry_job(j1["job_id"])
        assert set(work_dir.iterdir()) == before_dirs
        st1_conflict = mgr.get_status(j1["job_id"])
        assert st1_conflict is not None and st1_conflict["state"] == "failed"
        assert j1["job_id"] not in mgr._queue

        # After j2 is cancelled, a store.update error must roll back record and clean up snapshot.
        mgr.cancel(j2["job_id"], wait=True)
        after_cancel_dirs = set(work_dir.iterdir())
        assert mgr._store is not None
        with (
            patch.object(mgr._store, "update", side_effect=JobStoreError("disk busy")),
            pytest.raises(JobStoreError, match="disk busy"),
        ):
            mgr.retry_job(j1["job_id"])
        assert set(work_dir.iterdir()) == after_cancel_dirs
        st1_rollback = mgr.get_status(j1["job_id"])
        assert st1_rollback is not None and st1_rollback["state"] == "failed"
        assert j1["job_id"] not in mgr._queue

        # Successful retry clears stale bundle/report/artifact_evidence.
        rec1 = mgr._jobs[j1["job_id"]]
        rec1.bundle = {"path": str(tmp_path / "stale.zip")}
        rec1.report = {"stale": True}
        rec1.artifact_evidence = {"stale": True}
        retried = mgr.retry_job(j1["job_id"])
        assert retried["state"] in {"queued", "running"}
        assert "bundle" not in retried
        assert rec1.bundle is None and rec1.report is None and rec1.artifact_evidence is None
    finally:
        mgr.shutdown(grace_s=0.2)

    # 2. In-memory mode also blocks concurrent active output conflicts on retry.
    calls[0] = 0
    mem_mgr = JobManager(command_factory=_cmd_factory, store_path=None, kill_grace_s=0.2)
    try:
        m1 = mem_mgr.submit(
            input_path=in_file, output_path=out_file, profile="production", overwrite=True
        )
        for _ in range(100):
            m1_st = mem_mgr.get_status(m1["job_id"])
            if m1_st and m1_st["state"] == "failed":
                break
            time.sleep(0.02)
        m2 = mem_mgr.submit(
            input_path=in_file, output_path=out_file, profile="production", overwrite=True
        )
        for _ in range(100):
            m2_st = mem_mgr.get_status(m2["job_id"])
            if m2_st and m2_st["state"] == "running":
                break
            time.sleep(0.02)
        with pytest.raises(OutputConflictError, match="reserved by an active job"):
            mem_mgr.retry_job(m1["job_id"])
        m1_after = mem_mgr.get_status(m1["job_id"])
        assert m1_after is not None and m1_after["state"] == "failed"
    finally:
        mem_mgr.shutdown(grace_s=0.2)
