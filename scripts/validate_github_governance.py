#!/usr/bin/env python3
"""Validate the immutable CI/governance design and emit a non-mutating API plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "evidence" / "release" / "github-governance-contract.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
USE_PATTERN = re.compile(r"^\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
TOP_LEVEL_FIELDS = {
    "schema_version",
    "contract_id",
    "contract_revision",
    "release",
    "status",
    "repository",
    "ci",
    "protected_branch",
    "protected_tags",
    "release_environment",
    "approval",
    "known_limits",
    "integrity",
}
EXPECTED_ACTIONS = {
    "actions/checkout": "11bd71901bbe5b1630ceea73d27597364c9af683",
    "actions/setup-node": "49933ea5288caeca8642d1e84afbd3f7d6820020",
    "actions/setup-python": "e797f83bcb11b83ae66e0230d6156d7c80228e7c",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "astral-sh/setup-uv": "1e862dfacbd1d6d858c55d9b792c756523627244",
}


class GovernanceError(ValueError):
    """The GitHub CI/governance contract is malformed or weakened."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GovernanceError(f"{label} must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GovernanceError(message)


def _canonical_contract(contract: dict[str, Any]) -> bytes:
    design = {key: value for key, value in contract.items() if key not in {"approval", "integrity"}}
    return json.dumps(
        design, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_contract(contract)).hexdigest()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise GovernanceError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise GovernanceError(f"duplicate JSON key is forbidden: {key}")
            value[key] = child
        return value

    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"cannot read governance contract {path}: {exc}") from exc
    return _object(raw, "governance contract")


def _validate_workflow(contract: dict[str, Any], workflow_text: str) -> None:
    ci = _object(contract["ci"], "ci")
    _require("pull_request_target:" not in workflow_text, "pull_request_target is forbidden")
    _require(
        "permissions:\n  contents: read" in workflow_text, "workflow permissions are not least"
    )
    _require("persist-credentials: false" in workflow_text, "checkout credentials remain persisted")
    _require("name: release" in workflow_text, "workflow name differs from the contract")
    _require("  required:\n    name: required" in workflow_text, "stable required job is absent")
    _require(
        "runs-on: [self-hosted, macOS, ARM64, hawavoclean-release]" in workflow_text,
        "release runner labels differ from the contract",
    )
    _require(
        "environment: release-candidate" in workflow_text,
        "private release gate is not environment-protected",
    )
    _require(
        "github.event.pull_request.head.repo.full_name == github.repository" in workflow_text,
        "fork pull requests are not excluded from the private release runner",
    )
    _require(
        "run: bash scripts/run_release_checks.sh" in workflow_text,
        "CI does not invoke the exact local release contract",
    )
    _require(
        "os: [ubuntu-24.04, macos-15]" in workflow_text
        and 'python: ["3.11", "3.12", "3.13", "3.14"]' in workflow_text,
        "hosted support matrix differs from the declared contract",
    )
    _require(
        '--cov-fail-under="$COVERAGE_FLOOR"' in workflow_text
        and 'COVERAGE_FLOOR: "92.49"' in workflow_text,
        "coverage floor is weakened or bypassed",
    )
    _require(
        "uv pip install --python" in workflow_text and '--no-deps "$WHEEL"' in workflow_text,
        "matrix does not install the built wheel into an isolated environment",
    )
    _require(
        "name: exact-release-gate-${{ github.sha }}-${{ github.run_attempt }}" in workflow_text,
        "full-gate evidence artifact does not bind the source SHA",
    )
    _require(
        "retention-days: 90" in workflow_text and "include-hidden-files: true" in workflow_text,
        "full-gate evidence retention contract is missing",
    )

    uses = USE_PATTERN.findall(workflow_text)
    _require(bool(uses), "workflow uses no pinned actions")
    seen: set[str] = set()
    for action, revision in uses:
        _require(action in EXPECTED_ACTIONS, f"unapproved workflow action: {action}")
        _require(HEX40.fullmatch(revision) is not None, f"action is not SHA-pinned: {action}")
        _require(revision == EXPECTED_ACTIONS[action], f"unexpected action revision: {action}")
        seen.add(action)
    _require(seen == set(EXPECTED_ACTIONS), "workflow action inventory is incomplete")
    _require(ci.get("actions") == EXPECTED_ACTIONS, "contract action pins differ from validator")


def validate_contract(contract: dict[str, Any], *, root: Path = ROOT) -> str:
    _require(set(contract) == TOP_LEVEL_FIELDS, "contract top-level fields differ from schema v1")
    _require(contract["schema_version"] == 1, "unsupported governance schema version")
    _require(contract["contract_id"] == "hawavoclean-github-governance-v1", "wrong contract id")
    _require(contract["contract_revision"] == "1.0.0", "wrong contract revision")
    _require(
        contract["release"] == {"product": "hawavoclean", "version": "3.3.0"},
        "governance contract targets the wrong release",
    )
    _require(
        contract["status"] in {"designed_pending_u1", "approved_pending_activation"},
        "contract status overclaims activation",
    )
    repository = _object(contract["repository"], "repository")
    _require(
        repository
        == {
            "full_name": "HawzhinBlanca/HawaVoClean",
            "default_branch": "main",
            "visibility": "private",
            "api_version": "2026-03-10",
        },
        "repository identity differs from the approved boundary",
    )

    ci = _object(contract["ci"], "ci")
    _require(ci.get("workflow_name") == "release", "wrong workflow name")
    _require(ci.get("required_status_context") == "release / required", "wrong required check")
    _require(
        ci.get("jobs")
        == ["source-contract", "core-support", "web-resolve", "exact-release-gate", "required"],
        "CI job inventory differs from the contract",
    )
    matrix = _object(ci.get("hosted_support_matrix"), "hosted support matrix")
    _require(
        matrix.get("operating_systems") == ["ubuntu-24.04", "macos-15"],
        "declared OS matrix is incomplete",
    )
    _require(
        matrix.get("python") == ["3.11", "3.12", "3.13", "3.14"], "Python matrix is incomplete"
    )
    _require(matrix.get("coverage_floor_percent") == 92.49, "coverage floor changed")
    _require(
        matrix.get("wheel_smoke_outside_repository_environment") is True,
        "isolated wheel smoke is required",
    )
    exact = _object(ci.get("exact_release_gate"), "exact release gate")
    _require(
        exact.get("runner_labels") == ["self-hosted", "macOS", "ARM64", "hawavoclean-release"],
        "exact release runner labels changed",
    )
    _require(exact.get("environment") == "release-candidate", "release environment changed")
    _require(
        exact.get("command") == "bash scripts/run_release_checks.sh", "release gate command changed"
    )
    _require(
        exact.get("fork_pull_requests_forbidden") is True,
        "fork PRs must not reach private evidence",
    )
    _require(
        exact.get("environment_approval_required_before_private_evidence") is True,
        "private evidence requires environment approval",
    )
    _require(ci.get("permissions") == {"contents": "read"}, "CI permissions are not least")
    _require(
        ci.get("artifact_names_bind_source_sha") is True, "evidence names must bind source SHA"
    )

    branch = _object(contract["protected_branch"], "protected branch")
    _require(branch.get("name") == "main", "wrong protected branch")
    _require(
        branch.get("required_status_contexts") == ["release / required"], "required checks changed"
    )
    for key in (
        "strict_status_checks",
        "enforce_admins",
        "dismiss_stale_reviews",
        "require_last_push_approval",
        "required_linear_history",
        "required_conversation_resolution",
    ):
        _require(branch.get(key) is True, f"branch safety requirement disabled: {key}")
    _require(branch.get("required_approving_review_count") == 1, "at least one review is required")
    _require(branch.get("allow_force_pushes") is False, "force pushes must stay blocked")
    _require(branch.get("allow_deletions") is False, "branch deletion must stay blocked")

    tags = _object(contract["protected_tags"], "protected tags")
    _require(tags.get("include") == ["refs/tags/v*"], "release tag selection changed")
    _require(tags.get("enforcement") == "active", "tag ruleset must be active")
    _require(tags.get("bypass_actors") == [], "release tags cannot have bypass actors")
    _require(tags.get("restrict_updates") is True, "release tag movement is not blocked")
    _require(tags.get("restrict_deletions") is True, "release tag deletion is not blocked")

    environment = _object(contract["release_environment"], "release environment")
    _require(environment.get("name") == "release-candidate", "release environment name changed")
    _require(environment.get("minimum_required_reviewers") == 1, "release gate requires a reviewer")
    _require(
        environment.get("private_evidence_must_be_outside_runner_workspace") is True,
        "private evidence isolation is required",
    )
    _require(
        environment.get("runner_must_be_ephemeral_or_single_purpose") is True,
        "release runner isolation is required",
    )

    approval = _object(contract["approval"], "approval")
    integrity = _object(contract["integrity"], "integrity")
    digest = contract_sha256(contract)
    _require(integrity.get("contract_sha256") == digest, "governance contract digest mismatch")
    _require(approval.get("checkpoint") == "U1", "wrong approval checkpoint")
    if approval.get("status") == "pending":
        _require(contract["status"] == "designed_pending_u1", "pending U1 status is inconsistent")
        _require(
            approval.get("approved_by") is None
            and approval.get("approved_at") is None
            and approval.get("approved_contract_sha256") is None,
            "pending U1 cannot carry approval fields",
        )
    elif approval.get("status") == "approved":
        _require(
            contract["status"] == "approved_pending_activation",
            "approved U1 status is inconsistent",
        )
        _require(
            isinstance(approval.get("approved_by"), str) and bool(approval["approved_by"]),
            "approved U1 must name its approver",
        )
        _require(
            isinstance(approval.get("approved_at"), str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", approval["approved_at"])
            is not None,
            "approved U1 must have a UTC approval time",
        )
        _require(
            approval.get("approved_contract_sha256") == digest,
            "U1 approval does not bind the exact contract design",
        )
    else:
        raise GovernanceError("approval status must be pending or approved")

    workflow_path = root / str(ci.get("workflow"))
    try:
        workflow_text = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GovernanceError(f"cannot read CI workflow {workflow_path}: {exc}") from exc
    workflow_digest = hashlib.sha256(workflow_text.encode("utf-8")).hexdigest()
    _require(ci.get("workflow_sha256") == workflow_digest, "CI workflow digest mismatch")
    _validate_workflow(contract, workflow_text)
    return digest


def api_plan(contract: dict[str, Any]) -> dict[str, Any]:
    """Return exact external mutations for later review; never execute them."""
    repository = _object(contract["repository"], "repository")["full_name"]
    branch = _object(contract["protected_branch"], "protected branch")
    tags = _object(contract["protected_tags"], "protected tags")
    return {
        "mutating": False,
        "requires_checkpoint": "U1",
        "api_version": _object(contract["repository"], "repository")["api_version"],
        "operations": [
            {
                "method": "PUT",
                "endpoint": f"/repos/{repository}/branches/{branch['name']}/protection",
                "body": {
                    "required_status_checks": {
                        "strict": branch["strict_status_checks"],
                        "contexts": branch["required_status_contexts"],
                    },
                    "enforce_admins": branch["enforce_admins"],
                    "required_pull_request_reviews": {
                        "dismiss_stale_reviews": branch["dismiss_stale_reviews"],
                        "require_code_owner_reviews": branch["require_code_owner_reviews"],
                        "required_approving_review_count": branch[
                            "required_approving_review_count"
                        ],
                        "require_last_push_approval": branch["require_last_push_approval"],
                    },
                    "restrictions": None,
                    "required_linear_history": branch["required_linear_history"],
                    "allow_force_pushes": branch["allow_force_pushes"],
                    "allow_deletions": branch["allow_deletions"],
                    "required_conversation_resolution": branch["required_conversation_resolution"],
                },
            },
            {
                "method": "POST",
                "endpoint": f"/repos/{repository}/rulesets",
                "body": {
                    "name": tags["ruleset_name"],
                    "target": "tag",
                    "enforcement": tags["enforcement"],
                    "bypass_actors": tags["bypass_actors"],
                    "conditions": {
                        "ref_name": {"include": tags["include"], "exclude": []},
                    },
                    "rules": [{"type": "update"}, {"type": "deletion"}],
                },
            },
        ],
        "manual_prerequisites": [
            "Resolve U1 and preserve the private repository boundary.",
            "Create the release-candidate environment with at least one named reviewer.",
            "Register an ephemeral or single-purpose Apple-silicon runner with the contracted labels.",
            "Set HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT to an external hash-locked evidence mirror.",
            "Review this plan again immediately before applying it.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--print-api-plan", action="store_true")
    args = parser.parse_args()
    try:
        contract = load_contract(args.contract)
        digest = validate_contract(contract)
    except GovernanceError as exc:
        print(f"GitHub governance validation failed: {exc}", file=sys.stderr)
        return 2
    if args.print_api_plan:
        print(json.dumps(api_plan(contract), indent=2, sort_keys=True))
    else:
        print(f"GitHub governance design is valid and pending U1: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
