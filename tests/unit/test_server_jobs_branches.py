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
