from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from scripts import validate_sorani_protocol as protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "evidence" / "release" / "sorani-evaluation-protocol.json"


def _locked() -> dict[str, object]:
    raw: object = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast(dict[str, object], raw)


def _rehash(value: dict[str, object]) -> None:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["design_sha256"] = protocol.design_sha256(value)


def test_committed_protocol_is_valid_result_free_and_pending() -> None:
    value = _locked()
    digest = protocol.validate_protocol(value)
    assert digest == value["integrity"]["design_sha256"]  # type: ignore[index]
    with pytest.raises(protocol.ProtocolError, match="pending explicit user approval"):
        protocol.validate_protocol(value, require_approved=True)


def test_approval_must_bind_the_exact_design() -> None:
    value = _locked()
    approval = value["approval"]
    assert isinstance(approval, dict)
    approval.update(
        {
            "status": "approved",
            "approved_by": "release-owner",
            "approved_at": "2026-08-21T14:00:00Z",
            "approved_design_sha256": value["integrity"]["design_sha256"],  # type: ignore[index]
        }
    )
    protocol.validate_protocol(value, require_approved=True)

    changed = copy.deepcopy(value)
    changed["scope"]["claim"] = "weaker claim after approval"  # type: ignore[index]
    with pytest.raises(protocol.ProtocolError, match="design digest mismatch"):
        protocol.validate_protocol(changed, require_approved=True)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["outcomes"][0].update(maximum_allowed=1),
            "confirmed_content_change must remain zero tolerance",
        ),
        (
            lambda value: value["population_and_splits"]["sample_size"].update(
                minimum_heldout_source_units_per_profile=300
            ),
            "at least 450 units/profile required",
        ),
        (
            lambda value: value["review"].update(asr_role="oracle"),
            "ASR cannot be the linguistic oracle",
        ),
        (
            lambda value: value["analysis"]["release_thresholds"].update(
                sig_noninferiority_margin_mos=-0.5
            ),
            "SIG margin changed",
        ),
        (
            lambda value: value["exclusions"].update(failed_units_are_not_replaced=False),
            "failed units cannot be replaced",
        ),
    ],
)
def test_validator_rejects_silent_safety_or_power_regressions(
    mutator: object, message: str
) -> None:
    value = _locked()
    assert callable(mutator)
    mutator(value)
    _rehash(value)
    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.validate_protocol(value)


def test_protocol_cannot_smuggle_heldout_results_into_the_design() -> None:
    value = _locked()
    scope = value["scope"]
    assert isinstance(scope, dict)
    scope["results"] = {"content_changes": 0}
    _rehash(value)
    with pytest.raises(protocol.ProtocolError, match="held-out result field is forbidden"):
        protocol.validate_protocol(value)


def test_loader_rejects_duplicate_keys_and_validator_reports_bad_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(protocol.ProtocolError, match="duplicate JSON key"):
        protocol.load_protocol(duplicate)

    value = _locked()
    sample = value["population_and_splits"]["sample_size"]  # type: ignore[index]
    assert isinstance(sample, dict)
    sample["minimum_heldout_speakers"] = "many"
    _rehash(value)
    with pytest.raises(protocol.ProtocolError, match="minimum_heldout_speakers must be an integer"):
        protocol.validate_protocol(value)
