from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from scripts import hydrate_release_evidence as hydration
from scripts import validate_github_governance as governance

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "evidence" / "release" / "github-governance-contract.json"


def _contract() -> dict[str, object]:
    raw: object = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast(dict[str, object], raw)


def _rehash(value: dict[str, object]) -> None:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["contract_sha256"] = governance.contract_sha256(value)


def test_committed_ci_governance_design_is_valid_and_pending_u1() -> None:
    value = governance.load_contract()
    assert governance.validate_contract(value) == value["integrity"]["contract_sha256"]
    assert value["status"] == "designed_pending_u1"
    assert value["approval"]["status"] == "pending"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["protected_branch"].update(allow_force_pushes=True),
            "force pushes must stay blocked",
        ),
        (
            lambda value: value["protected_tags"].update(restrict_updates=False),
            "release tag movement is not blocked",
        ),
        (
            lambda value: value["ci"]["hosted_support_matrix"].update(python=["3.14"]),
            "Python matrix is incomplete",
        ),
        (
            lambda value: value["ci"].update(permissions={"contents": "write"}),
            "CI permissions are not least",
        ),
    ],
)
def test_governance_validator_rejects_weakened_controls(mutator: object, message: str) -> None:
    value = _contract()
    assert callable(mutator)
    mutator(value)
    _rehash(value)
    with pytest.raises(governance.GovernanceError, match=message):
        governance.validate_contract(value)


def test_governance_api_plan_is_non_mutating_and_complete() -> None:
    plan = governance.api_plan(governance.load_contract())
    assert plan["mutating"] is False
    assert plan["requires_checkpoint"] == "U1"
    operations = plan["operations"]
    assert [item["method"] for item in operations] == ["PUT", "POST"]
    assert operations[0]["body"]["required_status_checks"]["contexts"] == ["required"]
    assert operations[1]["body"]["rules"] == [{"type": "update"}, {"type": "deletion"}]


def test_u1_approval_must_bind_the_exact_design() -> None:
    value = _contract()
    value["status"] = "approved_pending_activation"
    _rehash(value)
    approval = value["approval"]
    assert isinstance(approval, dict)
    approval.update(
        {
            "status": "approved",
            "approved_by": "release-owner",
            "approved_at": "2026-08-21T19:00:00Z",
            "approved_contract_sha256": value["integrity"]["contract_sha256"],  # type: ignore[index]
        }
    )
    governance.validate_contract(value)

    changed = copy.deepcopy(value)
    known_limits = changed["known_limits"]
    assert isinstance(known_limits, list)
    known_limits.append("changed after approval")
    with pytest.raises(governance.GovernanceError, match="governance contract digest mismatch"):
        governance.validate_contract(changed)


def test_workflow_validator_rejects_floating_actions_and_reduced_gate(tmp_path: Path) -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    target = tmp_path / ".github" / "workflows" / "ci.yml"
    target.parent.mkdir(parents=True)
    floating = workflow.replace(
        "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
        "actions/checkout@v4",
    )
    target.write_text(floating, encoding="utf-8")
    floating_contract = _contract()
    floating_contract["ci"]["workflow_sha256"] = hashlib.sha256(  # type: ignore[index]
        floating.encode()
    ).hexdigest()
    _rehash(floating_contract)
    with pytest.raises(governance.GovernanceError, match="not SHA-pinned"):
        governance.validate_contract(floating_contract, root=tmp_path)

    reduced = workflow.replace(
        "run: bash scripts/run_release_checks.sh",
        "run: uv run pytest -q",
    )
    target.write_text(reduced, encoding="utf-8")
    reduced_contract = _contract()
    reduced_contract["ci"]["workflow_sha256"] = hashlib.sha256(  # type: ignore[index]
        reduced.encode()
    ).hexdigest()
    _rehash(reduced_contract)
    with pytest.raises(governance.GovernanceError, match="exact local release contract"):
        governance.validate_contract(reduced_contract, root=tmp_path)


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", path], check=True)


def _hydration_fixture(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    _git_init(destination)
    (destination / ".gitignore").write_text("private/\n", encoding="utf-8")
    payload = b"private regression bytes"
    artifact = source / "private" / "input.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "input": "private/input.bin",
                        "input_sha256": digest,
                        "reference_audio": "private/input.bin",
                        "audio_sha256": digest,
                        "reference_report": "private/input.bin",
                        "report_sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return source, destination, manifest, payload


def test_private_evidence_hydration_checks_hash_ignore_rule_and_mode(tmp_path: Path) -> None:
    source, destination, manifest, payload = _hydration_fixture(tmp_path)
    hydrated = hydration.hydrate(source, destination, manifest)
    target = destination / "private" / "input.bin"
    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o777 == 0o600
    assert hydrated == {"private/input.bin": hashlib.sha256(payload).hexdigest()}
    assert hydration.hydrate(source, destination, manifest) == hydrated


def test_private_evidence_hydration_fails_closed(tmp_path: Path) -> None:
    source, destination, manifest, _payload = _hydration_fixture(tmp_path)
    value: dict[str, object] = json.loads(manifest.read_text(encoding="utf-8"))
    changed_hash = copy.deepcopy(value)
    changed_hash["cases"][0]["input_sha256"] = "0" * 64  # type: ignore[index]
    manifest.write_text(json.dumps(changed_hash), encoding="utf-8")
    with pytest.raises(hydration.HydrationError, match="hash drift"):
        hydration.hydrate(source, destination, manifest)

    manifest.write_text(json.dumps(value), encoding="utf-8")
    (destination / ".gitignore").write_text("", encoding="utf-8")
    with pytest.raises(hydration.HydrationError, match="not Git-ignored"):
        hydration.hydrate(source, destination, manifest)
