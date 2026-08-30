"""The v1 processing contract rejects unsafe or ambiguous combinations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hawavoclean.server.app import _job_status_v1
from hawavoclean.server.contracts import JobStatusResponseV1, ProcessingRequestV1


def _manual(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "sourceIds": ["source_1"],
        "strategy": {
            "kind": "manual",
            "route": "production",
            "allowGenerativeReconstruction": False,
        },
        "executionPolicy": "offline_only",
        "conflictPolicy": "unique",
        "recordBundle": False,
        "idempotencyKey": "request-1",
    }
    value.update(changes)
    return value


def test_wire_contract_uses_camel_case_and_round_trips() -> None:
    request = ProcessingRequestV1.model_validate(_manual())
    assert request.strategy.kind == "manual"
    assert request.model_dump(by_alias=True, mode="json", exclude_none=True) == _manual()


def test_completed_record_bundle_status_requires_closed_verified_evidence() -> None:
    status: dict[str, object] = {
        "schemaVersion": 1,
        "jobId": "j_record",
        "state": "completed",
        "stage": "done",
        "progress": 1.0,
        "message": "Done",
        "outputPath": "/exports/result.wav",
        "reportPath": "/exports/result.hawavoclean.json",
        "recordBundle": True,
        "bundlePath": "/exports/result.hawavoclean.zip",
        "createdAt": "2026-08-27T00:00:00.000Z",
    }
    with pytest.raises(ValidationError, match="verified bundle evidence"):
        JobStatusResponseV1.model_validate(status)

    digest = "a" * 64
    parsed = JobStatusResponseV1.model_validate(
        {
            **status,
            "bundle": {
                "path": status["bundlePath"],
                "archiveSha256": digest,
                "contentSha256": digest,
                "masterSha256": digest,
                "reportSha256": digest,
                "summarySha256": digest,
                "totalUncompressedBytes": 4,
                "internalHashesVerified": True,
                "authenticatedPublisher": False,
            },
        }
    )
    assert parsed.bundle is not None
    assert parsed.bundle.archive_sha256 == digest


def test_unknown_fields_and_duplicate_sources_fail_closed() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProcessingRequestV1.model_validate({**_manual(), "profil": "studio"})
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ProcessingRequestV1.model_validate(_manual(sourceIds=["same", "same"]))


@pytest.mark.parametrize("route", ["restore_source", "restore_enrolled"])
def test_restore_requires_explicit_reconstruction_consent(route: str) -> None:
    strategy: dict[str, object] = {
        "kind": "manual",
        "route": route,
        "allowGenerativeReconstruction": False,
    }
    if route == "restore_enrolled":
        strategy["speakerProfileId"] = "speaker_1"
    with pytest.raises(ValidationError, match="explicit generative reconstruction consent"):
        ProcessingRequestV1.model_validate(_manual(strategy=strategy))


def test_enrolled_routes_require_profile_and_profile_is_not_accepted_elsewhere() -> None:
    with pytest.raises(ValidationError, match="requires speakerProfileId"):
        ProcessingRequestV1.model_validate(
            _manual(
                strategy={
                    "kind": "manual",
                    "route": "restore_enrolled",
                    "allowGenerativeReconstruction": True,
                }
            )
        )
    with pytest.raises(ValidationError, match="valid only for restore_enrolled"):
        ProcessingRequestV1.model_validate(
            _manual(
                strategy={
                    "kind": "manual",
                    "route": "production",
                    "speakerProfileId": "speaker_1",
                    "allowGenerativeReconstruction": False,
                }
            )
        )


def test_smart_restore_requires_consent_and_enrolled_profile() -> None:
    with pytest.raises(ValidationError, match="explicit generative reconstruction consent"):
        ProcessingRequestV1.model_validate(
            _manual(
                strategy={
                    "kind": "smart_safe",
                    "restorePolicy": "auto",
                    "allowGenerativeReconstruction": False,
                }
            )
        )
    with pytest.raises(ValidationError, match="requires speakerProfileId"):
        ProcessingRequestV1.model_validate(
            _manual(
                strategy={
                    "kind": "smart_safe",
                    "restorePolicy": "enrolled_only",
                    "allowGenerativeReconstruction": True,
                }
            )
        )


def test_cloud_needs_consent_and_offline_refuses_cloud_consent() -> None:
    with pytest.raises(ValidationError, match="requires a per-request cloudConsentId"):
        ProcessingRequestV1.model_validate(_manual(executionPolicy="cloud_allowed"))
    with pytest.raises(ValidationError, match="valid only with cloud_allowed"):
        ProcessingRequestV1.model_validate(_manual(cloudConsentId="consent-1"))


def test_nonfinite_cutoff_and_strategy_typo_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProcessingRequestV1.model_validate(
            _manual(
                strategy={
                    "kind": "manual",
                    "route": "restore_source",
                    "expertCutoffHz": float("nan"),
                    "allowGenerativeReconstruction": True,
                }
            )
        )
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        ProcessingRequestV1.model_validate(
            _manual(strategy={"kind": "smart", "restorePolicy": "disabled"})
        )


@pytest.mark.parametrize(
    ("legacy_state", "stage", "expected"),
    [
        ("queued", "preflight", "queued"),
        ("running", "decode", "analyzing"),
        ("running", "enhance", "rendering"),
        ("running", "guard", "guarding"),
        ("running", "publish", "publishing"),
        ("done", "done", "completed"),
        ("cancelled", "error", "cancelled"),
        ("interrupted", "error", "interrupted"),
        ("failed", "error", "failed"),
    ],
)
def test_legacy_job_states_map_to_the_closed_v1_lifecycle(
    legacy_state: str,
    stage: str,
    expected: str,
) -> None:
    response = _job_status_v1(
        {
            "job_id": "j_test",
            "state": legacy_state,
            "stage": stage,
            "progress": 0.5,
            "message": "test",
            "output_path": "/tmp/out.wav",
            "report_path": "/tmp/out.hawavoclean.json",
            "created_at": "2026-08-27T00:00:00Z",
        }
    )
    assert response.state == expected
