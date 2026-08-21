#!/usr/bin/env python3
"""Validate the locked, result-free Sorani human-evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "evidence" / "release" / "sorani-evaluation-protocol.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "protocol_revision",
    "release",
    "frozen_at",
    "scope",
    "systems",
    "population_and_splits",
    "outcomes",
    "review",
    "listening_test",
    "analysis",
    "exclusions",
    "stopping_and_regression",
    "data_governance",
    "standards",
    "approval",
    "integrity",
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
PROFILE_IDS = {"production", "studio", "lowband"}
STANDARD_IDS = {
    "ITU-T P.800",
    "ITU-T P.835",
    "ITU-T P.808",
    "ITU-T P.807",
    "Hanley-Lippman-Hand zero numerator",
    "Hu-Loizou speech enhancement evaluation",
}
OUTCOME_IDS = {
    "confirmed_content_change",
    "guard_false_accept",
    "guard_false_revert",
    "intelligibility",
    "artifact_severity",
    "p835_sig",
    "p835_bak",
    "p835_ovrl",
    "blinded_preference",
}


class ProtocolError(ValueError):
    """The evaluation protocol is malformed, weakened, or not approved."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{label} must be a list")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _integer(mapping: dict[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{label} must be an integer")
    return value


def _number(mapping: dict[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ProtocolError(f"{label} must be a finite number")
    return float(value)


def _objects_by_id(values: list[Any], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        item = _object(value, f"{label}[{index}]")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ProtocolError(f"{label}[{index}].id must be a non-empty string")
        if item_id in indexed:
            raise ProtocolError(f"duplicate {label} id: {item_id}")
        indexed[item_id] = item
    return indexed


def _canonical_design(protocol: dict[str, Any]) -> bytes:
    """Return the immutable design bytes, deliberately excluding signatures."""
    design = {key: value for key, value in protocol.items() if key not in {"approval", "integrity"}}
    return json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def design_sha256(protocol: dict[str, Any]) -> str:
    """Hash the protocol design independently from its later approval record."""
    return hashlib.sha256(_canonical_design(protocol)).hexdigest()


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Load a JSON protocol object without accepting NaN or duplicate ambiguity."""

    def reject_constant(value: str) -> None:
        raise ProtocolError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ProtocolError(f"duplicate JSON key is forbidden: {key}")
            value[key] = child
        return value

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read protocol {path}: {exc}") from exc
    return _object(raw, "protocol")


def _reject_embedded_results(value: object, path: str = "protocol") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            result_like = (
                lowered.startswith(("observed_", "measured_", "result_", "results_"))
                or lowered.endswith(("_result", "_results", "_p_value"))
                or lowered in FORBIDDEN_RESULT_KEYS
            )
            if result_like:
                raise ProtocolError(
                    f"held-out result field is forbidden before execution: {path}.{key}"
                )
            _reject_embedded_results(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_results(child, f"{path}[{index}]")


def _outcomes_by_id(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outcomes = _list(protocol.get("outcomes"), "outcomes")
    indexed = _objects_by_id(outcomes, "outcomes")
    _require(set(indexed) == OUTCOME_IDS, "the complete locked outcome family is required")
    return indexed


def validate_protocol(protocol: dict[str, Any], *, require_approved: bool = False) -> str:
    """Validate structure and the safety/statistical floors; return the design digest."""
    _require(set(protocol) == TOP_LEVEL_FIELDS, "protocol top-level fields differ from schema v1")
    _require(protocol["schema_version"] == 1, "unsupported protocol schema version")
    _require(protocol["protocol_id"] == "hawavoclean-sorani-acceptance-v1", "wrong protocol id")
    _require(protocol["protocol_revision"] == "1.0.0", "wrong protocol revision")
    _require(protocol["frozen_at"] == "2026-08-21T00:00:00Z", "unexpected freeze time")
    _reject_embedded_results(protocol)

    release = _object(protocol["release"], "release")
    _require(
        release == {"product": "hawavoclean", "version": "3.3.0"},
        "protocol must target the exact 3.3.0 release",
    )
    systems = _objects_by_id(_list(protocol["systems"], "systems"), "systems")
    _require(
        set(systems) == {"original", "discarded_guard_candidate", *PROFILE_IDS},
        "original, discarded guard candidate, and all three shipped profiles required",
    )
    for system_id, system in systems.items():
        if system_id in PROFILE_IDS:
            _require(
                system.get("shipped_candidate") is True, "each profile must test shipped output"
            )
        else:
            _require(
                system.get("shipped_candidate") is False,
                "original and diagnostic guard candidates cannot be marked shipped",
            )

    population = _object(protocol["population_and_splits"], "population_and_splits")
    sample = _object(population.get("sample_size"), "population_and_splits.sample_size")
    n = _integer(
        sample,
        "minimum_heldout_source_units_per_profile",
        "minimum_heldout_source_units_per_profile",
    )
    _require(n >= 450, "at least 450 units/profile required")
    speakers = _integer(sample, "minimum_heldout_speakers", "minimum_heldout_speakers")
    _require(speakers >= 45, "at least 45 held-out speakers required")
    per_speaker = _integer(sample, "maximum_units_per_speaker", "maximum_units_per_speaker")
    _require(per_speaker <= 10, "speaker concentration exceeds 10 units")
    reviews = _integer(
        sample,
        "minimum_independent_content_reviews_per_profile",
        "minimum_independent_content_reviews_per_profile",
    )
    _require(reviews >= 2 * n, "every comparison requires two independent content reviews")
    alpha = _number(sample, "one_sided_alpha", "one_sided_alpha")
    family_alpha = _number(sample, "bonferroni_alpha_per_profile", "bonferroni_alpha_per_profile")
    _require(alpha == 0.05, "one-sided alpha must remain 0.05")
    _require(math.isclose(family_alpha, 0.05 / 3, abs_tol=1e-15), "profile family alpha changed")
    exact = 1.0 - math.pow(alpha, 1.0 / n)
    simultaneous = 1.0 - math.pow(family_alpha, 1.0 / n)
    _require(
        math.isclose(
            _number(sample, "zero_event_upper_95_per_profile", "zero_event_upper_95_per_profile"),
            exact,
            abs_tol=1e-15,
        ),
        "stored zero-event pointwise bound is not exact",
    )
    _require(
        math.isclose(
            _number(
                sample,
                "zero_event_upper_95_simultaneous_three_profiles",
                "zero_event_upper_95_simultaneous_three_profiles",
            ),
            simultaneous,
            abs_tol=1e-15,
        ),
        "stored simultaneous zero-event bound is not exact",
    )
    _require(
        simultaneous < 0.01, "sample no longer supports the declared sub-1% simultaneous bound"
    )
    _require(population.get("speaker_disjoint") is True, "splits must be speaker-disjoint")
    _require(population.get("heldout_labels_unseen") is True, "held-out labels must remain unseen")
    _require(
        population.get("reserve_holdout_required") is True, "untouched reserve holdout required"
    )

    outcomes = _outcomes_by_id(protocol)
    for outcome_id in ("confirmed_content_change", "guard_false_accept"):
        item = outcomes[outcome_id]
        _require(item.get("primary") is True, f"{outcome_id} must remain primary")
        _require(
            _integer(item, "maximum_allowed", f"{outcome_id}.maximum_allowed") == 0,
            f"{outcome_id} must remain zero tolerance",
        )
        _require(item.get("release_blocking") is True, f"{outcome_id} must block release")
    _require(
        outcomes["guard_false_revert"].get("release_blocking") is False,
        "guard false revert is an efficiency measure, not a linguistic-safety oracle",
    )

    review = _object(protocol["review"], "review")
    content_reviewers = _integer(
        review,
        "independent_content_reviewers_per_comparison",
        "independent_content_reviewers_per_comparison",
    )
    _require(content_reviewers >= 2, "dual review required")
    _require(review.get("adjudicator_required") is True, "disagreements require adjudication")
    _require(review.get("profile_blinding_until_verdict_lock") is True, "profile blinding required")
    _require(
        review.get("asr_role") == "triage_only_not_oracle", "ASR cannot be the linguistic oracle"
    )
    _require(
        review.get("source_transcript_locked_first") is True, "source transcript must lock first"
    )

    listening = _object(protocol["listening_test"], "listening_test")
    listeners = _integer(
        listening, "minimum_valid_sorani_listeners", "minimum_valid_sorani_listeners"
    )
    _require(listeners >= 32, "at least 32 valid listeners required")
    ratings = _integer(
        listening,
        "minimum_valid_ratings_per_item_condition",
        "minimum_valid_ratings_per_item_condition",
    )
    _require(ratings >= 16, "16 ratings/condition required")
    _require(listening.get("double_blind") is True, "listening comparison must be double blind")
    _require(listening.get("p835_dimensions") == ["SIG", "BAK", "OVRL"], "P.835 dimensions changed")
    _require(
        listening.get("balanced_scale_orders") == ["SIG-BAK-OVRL", "BAK-SIG-OVRL"],
        "P.835 scale order must be counter-balanced",
    )

    analysis = _object(protocol["analysis"], "analysis")
    thresholds = _object(analysis.get("release_thresholds"), "analysis.release_thresholds")
    _require(
        _integer(thresholds, "content_changes_maximum", "content_changes_maximum") == 0,
        "content threshold weakened",
    )
    _require(
        _integer(thresholds, "guard_false_accepts_maximum", "guard_false_accepts_maximum") == 0,
        "false-accept threshold weakened",
    )
    _require(
        _number(thresholds, "sig_noninferiority_margin_mos", "sig_noninferiority_margin_mos")
        == -0.25,
        "SIG margin changed",
    )
    _require(
        _number(thresholds, "ovrl_noninferiority_margin_mos", "ovrl_noninferiority_margin_mos")
        == -0.25,
        "OVRL margin changed",
    )
    _require(
        _number(thresholds, "bak_noninferiority_margin_mos", "bak_noninferiority_margin_mos")
        == -0.25,
        "BAK margin changed",
    )
    _require(
        _number(
            thresholds,
            "intelligibility_noninferiority_margin",
            "intelligibility_noninferiority_margin",
        )
        == -0.02,
        "intelligibility margin changed",
    )
    _require(
        _integer(
            thresholds,
            "processing_induced_severe_artifacts_maximum",
            "processing_induced_severe_artifacts_maximum",
        )
        == 0,
        "severe-artifact threshold weakened",
    )
    _require(
        _number(
            thresholds,
            "bak_superiority_lower_bound_on_noisy_intended_strata",
            "bak_superiority_lower_bound_on_noisy_intended_strata",
        )
        == 0.0,
        "BAK superiority gate changed",
    )
    _require(
        analysis.get("multiplicity") == "holm_5_percent_within_profile_primary_family",
        "multiplicity control changed",
    )
    _require(
        analysis.get("cluster_units") == ["speaker", "source_unit", "listener"],
        "cluster plan changed",
    )
    _require(
        analysis.get("no_missing_value_imputation") is True, "missing values cannot be imputed"
    )

    exclusions = _object(protocol["exclusions"], "exclusions")
    _require(
        exclusions.get("locked_before_candidate_generation") is True, "exclusions must lock first"
    )
    _require(
        exclusions.get("technical_processing_failure_is_failure") is True,
        "processing failures cannot be excluded",
    )
    _require(
        exclusions.get("failed_units_are_not_replaced") is True, "failed units cannot be replaced"
    )
    stopping = _object(protocol["stopping_and_regression"], "stopping_and_regression")
    _require(
        stopping.get("any_confirmed_content_change_stops_release") is True,
        "content stop rule required",
    )
    _require(
        stopping.get("output_changing_fix_invalidates_exposed_holdout") is True,
        "holdout reuse forbidden",
    )

    governance = _object(protocol["data_governance"], "data_governance")
    _require(
        governance.get("rights_approval_required_before_collection") is True,
        "rights approval required",
    )
    _require(
        governance.get("raw_audio_and_pii_forbidden_in_git") is True,
        "raw audio/PII cannot enter Git",
    )
    _require(
        governance.get("reviewer_identity_mapping_outside_repository") is True,
        "reviewers must stay anonymous in Git",
    )
    standards = _objects_by_id(_list(protocol["standards"], "standards"), "standards")
    _require(set(standards) == STANDARD_IDS, "standards are incomplete or unexpected")
    for standard_id, standard in standards.items():
        url = standard.get("url")
        _require(
            isinstance(url, str) and url.startswith("https://"),
            f"{standard_id} must have an HTTPS source URL",
        )

    digest = design_sha256(protocol)
    integrity = _object(protocol["integrity"], "integrity")
    _require(
        set(integrity) == {"algorithm", "design_sha256"},
        "integrity must contain exactly algorithm and design_sha256",
    )
    _require(integrity.get("algorithm") == "sha256-canonical-json-v1", "wrong integrity algorithm")
    claimed = integrity.get("design_sha256")
    _require(
        isinstance(claimed, str) and HEX64.fullmatch(claimed) is not None, "invalid design digest"
    )
    _require(claimed == digest, f"design digest mismatch: claimed={claimed}, actual={digest}")

    approval = _object(protocol["approval"], "approval")
    _require(
        set(approval)
        == {
            "status",
            "approved_by",
            "approved_at",
            "approved_design_sha256",
            "heldout_examined_before_approval",
        },
        "approval fields differ from schema v1",
    )
    status = approval.get("status")
    _require(
        isinstance(status, str) and status in {"pending_user_approval", "approved"},
        "invalid approval status",
    )
    if status == "approved":
        _require(
            isinstance(approval.get("approved_by"), str) and approval["approved_by"],
            "approver required",
        )
        approved_at = approval.get("approved_at")
        _require(
            isinstance(approved_at, str) and approved_at.endswith("Z"), "UTC approval time required"
        )
        _require(
            approval.get("approved_design_sha256") == digest, "approval does not bind this design"
        )
        _require(
            approval.get("heldout_examined_before_approval") is False,
            "approval must precede held-out examination",
        )
    else:
        _require(approval.get("approved_by") is None, "pending protocol cannot name an approver")
        _require(
            approval.get("approved_at") is None, "pending protocol cannot have an approval time"
        )
        _require(
            approval.get("approved_design_sha256") is None,
            "pending protocol cannot claim a signed design",
        )
    if require_approved and status != "approved":
        raise ProtocolError("protocol is valid but still pending explicit user approval")
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--print-digest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        digest = validate_protocol(protocol, require_approved=args.require_approved)
    except ProtocolError as exc:
        print(f"Sorani protocol: FAILED: {exc}", file=sys.stderr)
        return 1
    status = _object(protocol["approval"], "approval")["status"]
    print(f"Sorani protocol: VALID ({status})")
    if args.print_digest:
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
