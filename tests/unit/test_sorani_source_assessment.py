from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from scripts import validate_sorani_sources as sources

ROOT = Path(__file__).resolve().parents[2]
ASSESSMENT_PATH = ROOT / "evidence" / "release" / "sorani-corpus-source-assessment.json"


def _locked() -> dict[str, object]:
    raw: object = json.loads(ASSESSMENT_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast(dict[str, object], raw)


def _rehash(value: dict[str, object]) -> None:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["design_sha256"] = sources.design_sha256(value)


def test_committed_source_assessment_is_valid_result_free_and_pending() -> None:
    value = _locked()
    digest = sources.validate_assessment(value)
    assert digest == value["integrity"]["design_sha256"]  # type: ignore[index]
    with pytest.raises(sources.SourceAssessmentError, match="pending explicit user approval"):
        sources.validate_assessment(value, require_approved=True)


def test_approval_must_bind_both_decisions_and_exact_design() -> None:
    value = _locked()
    approval = value["approval"]
    assert isinstance(approval, dict)
    approval.update(
        {
            "status": "approved",
            "approved_by": "release-owner",
            "approved_at": "2026-08-21T16:00:00Z",
            "approved_design_sha256": value["integrity"]["design_sha256"],  # type: ignore[index]
            "user_accepts_common_voice_terms": True,
            "user_authorizes_fresh_collection": True,
        }
    )
    sources.validate_assessment(value, require_approved=True)

    changed = copy.deepcopy(value)
    changed["recommended_route"]["rationale"] = "changed after approval"  # type: ignore[index]
    with pytest.raises(sources.SourceAssessmentError, match="design digest mismatch"):
        sources.validate_assessment(changed, require_approved=True)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["recommended_route"]["acceptance"]["common_voice_26_ckb"].update(
                minimum_speakers=20
            ),
            "at least 45 speakers",
        ),
        (
            lambda value: value["sources"][5].update(disposition="supporting_only"),
            "supporting source dispositions changed",
        ),
        (
            lambda value: value["sources"][6].update(licence="CC0-1.0"),
            "AsoSoft non-commercial restriction changed",
        ),
        (
            lambda value: value["selection_policy"].update(
                quarantined_cannot_enter_any_evaluation_split=False
            ),
            "quarantined sources must stay out",
        ),
    ],
)
def test_validator_rejects_weakened_source_safety(mutator: object, message: str) -> None:
    value = _locked()
    assert callable(mutator)
    mutator(value)
    _rehash(value)
    with pytest.raises(sources.SourceAssessmentError, match=message):
        sources.validate_assessment(value)


def test_source_assessment_cannot_smuggle_results_into_design() -> None:
    value = _locked()
    route = value["recommended_route"]
    assert isinstance(route, dict)
    route["observed_results"] = {"content_changes": 0}
    _rehash(value)
    with pytest.raises(sources.SourceAssessmentError, match="held-out result field is forbidden"):
        sources.validate_assessment(value)


def test_source_assessment_rejects_unknown_schema_fields() -> None:
    value = _locked()
    value["override"] = "allow everything"
    _rehash(value)
    with pytest.raises(sources.SourceAssessmentError, match="top-level fields differ"):
        sources.validate_assessment(value)


def test_loader_rejects_duplicate_keys_and_non_https_sources(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(sources.SourceAssessmentError, match="duplicate JSON key"):
        sources.load_assessment(duplicate)

    value = _locked()
    value["sources"][0]["authoritative_url"] = "http://example.invalid"  # type: ignore[index]
    _rehash(value)
    with pytest.raises(sources.SourceAssessmentError, match="must use HTTPS"):
        sources.validate_assessment(value)
