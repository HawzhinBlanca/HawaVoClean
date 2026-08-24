from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "evidence" / "release" / "t7.3-prefinal-audit.json"
CHECKPOINT = ROOT / "evidence" / "release" / "t3.1-release-gate-refresh.json"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_prefinal_audit_is_integral_complete_in_scope_and_explicitly_not_independent() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    claimed = audit.pop("audit_sha256")
    assert claimed == _canonical_sha256(audit)
    assert audit["task_id"] == "T7.3"
    assert audit["classification"] == "prefinal_self_audit_not_independent_t7_3_remains_open"
    assert audit["independent_signoff"]["status"] == "not_performed"
    assert "does not complete T7.3" in audit["known_limits"][0]
    assert re.fullmatch(r"[0-9a-f]{40}", audit["source_commit"])
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", f"{audit['source_commit']}^{{commit}}"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )

    required_surfaces = {
        "publication",
        "interruption",
        "overwrite",
        "provenance",
        "security_boundary",
        "profile_routing",
        "corpus_leakage",
        "installer_rollback",
        "long_run_resources",
        "documentation_claims",
    }
    assert {item["surface"] for item in audit["attack_surfaces"]} == required_surfaces
    assert audit["findings"]
    assert all(
        finding["status"] == "closed"
        for finding in audit["findings"]
        if finding["severity"] in {"P0", "P1"}
    )


def test_prefinal_audit_names_the_exact_validated_release_gate_checkpoint() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    gate = audit["release_gate"]
    integrity = checkpoint["proof_integrity"]
    assert audit["source_commit"] == checkpoint["source_commit"]
    assert gate["retained_proof_path"] == integrity["retained_proof_path"]
    assert gate["full_proof_file_sha256"] == integrity["full_proof_file_sha256"]
    assert gate["full_proof_canonical_sha256"] == integrity["full_proof_canonical_sha256"]
    assert gate["default_suite_per_pass"] == {
        key: checkpoint["result"]["default_test_suite_per_pass"][key]
        for key in ("passed", "skipped", "fuzz_deselected", "branch_coverage_percent")
    }
    assert gate["artifact_reproducibility"] == checkpoint["reproducibility"]["status"]
