#!/usr/bin/env python3
"""Validate the result-free Sorani corpus source assessment and approval lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSESSMENT = ROOT / "evidence" / "release" / "sorani-corpus-source-assessment.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELDS = {
    "schema_version",
    "assessment_id",
    "assessment_revision",
    "release",
    "assessed_at",
    "protocol_design_sha256",
    "scope",
    "selection_policy",
    "recommended_route",
    "sources",
    "upstream_model_overlap",
    "approval",
    "integrity",
}
SOURCE_REQUIRED_FIELDS = {
    "id",
    "name",
    "disposition",
    "source_version",
    "authoritative_url",
    "licence",
    "rights_basis",
    "speaker_key",
    "declared_inventory",
    "local_state",
    "constraints",
    "blocking_conditions",
}
SOURCE_ALLOWED_FIELDS = SOURCE_REQUIRED_FIELDS | {"local_artifact_sha256"}
EXPECTED_SOURCE_IDS = {
    "common_voice_26_ckb",
    "fresh_consented_field",
    "fleurs_ckb_iq",
    "gigant_ktts",
    "kaset_ldc2024s01",
    "comprehensive_ckb_sound",
    "asosoft_speech_subset",
    "cordi",
    "ckb_tts",
    "ckb_new_speech_corpus",
    "unclassified_private_audio",
}
PRIMARY_IDS = {"common_voice_26_ckb", "fresh_consented_field"}
SUPPORTING_IDS = {"fleurs_ckb_iq", "gigant_ktts", "kaset_ldc2024s01"}
QUARANTINED_IDS = {
    "comprehensive_ckb_sound",
    "asosoft_speech_subset",
    "cordi",
    "ckb_tts",
    "ckb_new_speech_corpus",
    "unclassified_private_audio",
}
FORBIDDEN_RESULT_KEYS = {
    "observed",
    "observations",
    "results",
    "result_artifact",
    "measured_effect",
    "measured_rate",
    "p_value",
}


class SourceAssessmentError(ValueError):
    """The source assessment is malformed, weakened, or not approved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAssessmentError(message)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourceAssessmentError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SourceAssessmentError(f"{label} must be a list")
    return value


def _integer(mapping: dict[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SourceAssessmentError(f"{label} must be an integer")
    return value


def _canonical_design(assessment: dict[str, Any]) -> bytes:
    design = {
        key: value for key, value in assessment.items() if key not in {"approval", "integrity"}
    }
    return json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def design_sha256(assessment: dict[str, Any]) -> str:
    """Hash the source design independently from its later approval record."""
    return hashlib.sha256(_canonical_design(assessment)).hexdigest()


def load_assessment(path: Path = DEFAULT_ASSESSMENT) -> dict[str, Any]:
    """Load JSON without accepting duplicate keys or non-finite numbers."""

    def reject_constant(value: str) -> None:
        raise SourceAssessmentError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise SourceAssessmentError(f"duplicate JSON key is forbidden: {key}")
            value[key] = child
        return value

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAssessmentError(f"cannot read source assessment {path}: {exc}") from exc
    return _object(raw, "assessment")


def _reject_embedded_results(value: object, path: str = "assessment") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            result_like = (
                lowered.startswith(("observed_", "measured_", "result_", "results_"))
                or lowered.endswith(("_result", "_results", "_p_value"))
                or lowered in FORBIDDEN_RESULT_KEYS
            )
            if result_like:
                raise SourceAssessmentError(
                    f"held-out result field is forbidden before execution: {path}.{key}"
                )
            _reject_embedded_results(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_results(child, f"{path}[{index}]")


def _sources_by_id(assessment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(_list(assessment.get("sources"), "sources")):
        source = _object(raw, f"sources[{index}]")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise SourceAssessmentError(f"sources[{index}].id must be a non-empty string")
        if source_id in sources:
            raise SourceAssessmentError(f"duplicate source id: {source_id}")
        _require(
            SOURCE_REQUIRED_FIELDS <= set(source) <= SOURCE_ALLOWED_FIELDS,
            f"source fields differ from schema v1: {source_id}",
        )
        sources[source_id] = source
    _require(
        set(sources) == EXPECTED_SOURCE_IDS, "the complete audited source inventory is required"
    )
    return sources


def _validate_urls_and_hashes(value: object, path: str = "assessment") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key.endswith("_url") and child is not None:
                _require(
                    isinstance(child, str) and child.startswith("https://"),
                    f"{child_path} must use HTTPS",
                )
            if key == "local_artifact_sha256":
                hashes = _object(child, child_path)
                _require(bool(hashes), f"{child_path} cannot be empty")
                for name, digest in hashes.items():
                    _require(
                        isinstance(name, str)
                        and isinstance(digest, str)
                        and HEX64.fullmatch(digest) is not None,
                        f"{child_path}.{name} must be a lowercase SHA-256",
                    )
            _validate_urls_and_hashes(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_urls_and_hashes(child, f"{path}[{index}]")


def validate_assessment(assessment: dict[str, Any], *, require_approved: bool = False) -> str:
    """Validate the exact safe source route and return its design digest."""
    _require(
        set(assessment) == TOP_LEVEL_FIELDS, "assessment top-level fields differ from schema v1"
    )
    _require(assessment.get("schema_version") == 1, "unsupported source assessment schema")
    _require(
        assessment.get("assessment_id") == "hawavoclean-sorani-corpus-sources-v1",
        "wrong source assessment id",
    )
    _require(assessment.get("assessment_revision") == "1.0.0", "wrong assessment revision")
    _require(assessment.get("assessed_at") == "2026-08-21T00:00:00Z", "unexpected assessment time")
    _require(
        assessment.get("release") == {"product": "hawavoclean", "version": "3.3.0"},
        "assessment must target the exact 3.3.0 release",
    )
    _require(
        assessment.get("protocol_design_sha256")
        == "896dfc12be5600705cd279b367fe5e28e6dfd3c6543a14977b4aa8e981bedd82",
        "assessment must bind the locked Sorani protocol",
    )
    _reject_embedded_results(assessment)
    _validate_urls_and_hashes(assessment)

    policy = _object(assessment.get("selection_policy"), "selection_policy")
    _require(
        policy.get("supporting_only_cannot_satisfy_primary_floors") is True,
        "supporting sources cannot satisfy primary floors",
    )
    _require(
        policy.get("quarantined_cannot_enter_any_evaluation_split") is True,
        "quarantined sources must stay out of every split",
    )
    _require(
        policy.get("speaker_key_required_for_primary") is True,
        "primary sources require speaker keys",
    )
    _require(
        policy.get("commercially_compatible_rights_required") is True,
        "commercially compatible rights are required",
    )

    route = _object(assessment.get("recommended_route"), "recommended_route")
    _require(
        route.get("status") == "proposed",
        "route must remain the immutable proposed design",
    )
    acceptance = _object(route.get("acceptance"), "recommended_route.acceptance")
    cv = _object(acceptance.get("common_voice_26_ckb"), "acceptance.common_voice_26_ckb")
    fresh = _object(acceptance.get("fresh_consented_field"), "acceptance.fresh_consented_field")
    _require(
        _integer(cv, "minimum_units", "Common Voice acceptance minimum") >= 450,
        "Common Voice acceptance requires at least 450 units",
    )
    _require(
        _integer(cv, "minimum_speakers", "Common Voice speaker minimum") >= 45,
        "Common Voice acceptance requires at least 45 speakers",
    )
    _require(
        _integer(cv, "maximum_units_per_speaker", "Common Voice speaker cap") <= 10,
        "Common Voice speaker concentration exceeds 10 units",
    )
    _require(
        _integer(fresh, "minimum_units", "fresh acceptance minimum") >= 120,
        "fresh acceptance requires at least 120 units",
    )
    _require(
        _integer(fresh, "minimum_speakers", "fresh speaker minimum") >= 24,
        "fresh acceptance requires at least 24 speakers",
    )
    _require(
        _integer(fresh, "maximum_units_per_speaker", "fresh speaker cap") <= 5,
        "fresh speaker concentration exceeds 5 units",
    )
    _require(
        _integer(acceptance, "minimum_combined_units", "combined acceptance minimum") >= 570,
        "combined acceptance requires at least 570 units",
    )
    _require(
        acceptance.get("all_three_profiles_use_the_same_source_units") is True,
        "all profiles must use the same source units",
    )

    reserve = _object(route.get("reserve"), "recommended_route.reserve")
    _require(
        reserve.get("disjoint_from_acceptance_and_calibration") is True,
        "reserve must remain disjoint",
    )
    _require(
        _integer(route, "fresh_collection_minimum_total_units", "fresh collection unit floor")
        >= 300,
        "fresh collection requires at least 300 units",
    )
    _require(
        _integer(route, "fresh_collection_minimum_total_speakers", "fresh collection speaker floor")
        >= 60,
        "fresh collection requires at least 60 speakers",
    )

    sources = _sources_by_id(assessment)
    _require(
        {
            source_id
            for source_id, source in sources.items()
            if source.get("disposition") in {"proposed_primary", "required_primary"}
        }
        == PRIMARY_IDS,
        "only Common Voice and fresh consented audio may be primary",
    )
    _require(
        {
            source_id
            for source_id, source in sources.items()
            if source.get("disposition") in {"supporting_only", "conditional_supporting"}
        }
        == SUPPORTING_IDS,
        "supporting source dispositions changed",
    )
    _require(
        {
            source_id
            for source_id, source in sources.items()
            if source.get("disposition") == "quarantined"
        }
        == QUARANTINED_IDS,
        "every ambiguous or incompatible source must remain quarantined",
    )

    common_voice = sources["common_voice_26_ckb"]
    inventory = _object(common_voice.get("declared_inventory"), "Common Voice inventory")
    _require(
        common_voice.get("source_version") == "cv-corpus-26.0-2026-06-12",
        "Common Voice version changed",
    )
    _require(common_voice.get("licence") == "CC0-1.0", "Common Voice licence changed")
    _require(
        common_voice.get("speaker_key") == "client_id hashed UUID",
        "Common Voice opaque speaker key required",
    )
    _require(
        _integer(inventory, "validated_clips", "Common Voice validated clips") == 121139,
        "Common Voice validated count changed",
    )
    _require(
        _integer(inventory, "speakers", "Common Voice speakers") == 2038,
        "Common Voice speaker count changed",
    )
    constraints = common_voice.get("constraints")
    _require(
        isinstance(constraints, list)
        and any("Do not attempt" in str(item) for item in constraints),
        "Common Voice identity prohibition must be recorded",
    )
    _require(
        isinstance(constraints, list)
        and any("Do not re-host" in str(item) for item in constraints),
        "Common Voice re-hosting prohibition must be recorded",
    )

    _require(
        sources["asosoft_speech_subset"].get("licence") == "research and non-commercial use only",
        "AsoSoft non-commercial restriction changed",
    )
    _require(
        sources["ckb_tts"].get("licence") == "not declared in dataset card",
        "ckb_tts must remain unlicensed",
    )
    _require(
        sources["cordi"].get("disposition") == "quarantined",
        "CORDI media-rights risk must remain quarantined",
    )

    integrity = _object(assessment.get("integrity"), "integrity")
    _require(
        integrity.get("algorithm") == "sha256-canonical-json-excluding-approval-and-integrity",
        "wrong design digest algorithm",
    )
    digest = design_sha256(assessment)
    _require(integrity.get("design_sha256") == digest, "source assessment design digest mismatch")

    approval = _object(assessment.get("approval"), "approval")
    status = approval.get("status")
    _require(status in {"pending_user_approval", "approved"}, "invalid approval status")
    if status == "pending_user_approval":
        _require(approval.get("approved_by") is None, "pending approval cannot name an approver")
        _require(approval.get("approved_at") is None, "pending approval cannot have a timestamp")
        _require(
            approval.get("approved_design_sha256") is None, "pending approval cannot bind a digest"
        )
        _require(
            approval.get("user_accepts_common_voice_terms") is False,
            "pending approval cannot accept terms",
        )
        _require(
            approval.get("user_authorizes_fresh_collection") is False,
            "pending approval cannot authorize collection",
        )
        if require_approved:
            raise SourceAssessmentError(
                "source assessment is valid but still pending explicit user approval"
            )
    else:
        _require(
            isinstance(approval.get("approved_by"), str) and bool(approval["approved_by"]),
            "approved_by is required",
        )
        _require(
            isinstance(approval.get("approved_at"), str) and approval["approved_at"].endswith("Z"),
            "approved_at must be UTC",
        )
        _require(
            approval.get("approved_design_sha256") == digest,
            "approval does not bind the exact source design",
        )
        _require(
            approval.get("user_accepts_common_voice_terms") is True,
            "Common Voice terms require explicit acceptance",
        )
        _require(
            approval.get("user_authorizes_fresh_collection") is True,
            "fresh collection requires explicit authorization",
        )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_ASSESSMENT)
    parser.add_argument("--require-approved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assessment = load_assessment(args.path)
        digest = validate_assessment(assessment, require_approved=args.require_approved)
    except SourceAssessmentError as exc:
        print(f"Sorani source assessment invalid: {exc}", file=sys.stderr)
        return 1
    print(f"Sorani source assessment valid: {digest}")
    if assessment["approval"]["status"] != "approved":
        print("Status: pending explicit user approval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
