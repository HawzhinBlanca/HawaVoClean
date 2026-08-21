from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import validate_release_gate_checkpoint as validator

ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    checkpoint = json.loads(validator.DEFAULT_CHECKPOINT.read_text(encoding="utf-8"))
    result = checkpoint["result"]
    suite = result["default_test_suite_per_pass"]
    audit = json.dumps(
        {
            "metadata": {
                "vulnerabilities": {
                    "info": 0,
                    "low": 0,
                    "moderate": 0,
                    "high": 0,
                    "critical": 0,
                }
            }
        }
    )
    logs = {
        "default-tests-branch-coverage": (
            f"Total coverage: {suite['branch_coverage_percent']}%\n"
            f"{suite['passed']} passed, {suite['skipped']} skipped, "
            f"{suite['fuzz_deselected']} deselected\n"
        ),
        "fuzz-tests": f"{result['separate_fuzz_tests_per_pass']} passed in 1.00s\n",
        "mutation-gate": f"{result['owner_scoped_mutations_caught_per_pass']} caught\n",
        "real-audio-regressions": json.dumps(
            {
                "cases": [
                    {"status": "passed", "runs": result["runs_per_real_audio_case"]}
                    for _ in range(result["real_audio_cases"])
                ]
            }
        ),
        "ui-tests": f"Tests {result['ui_tests_per_pass']} passed\n",
        "ui-audit": audit,
        "plugin-audit": audit,
        "toolchain-audit": audit,
        "python-audit": json.dumps({"dependencies": []}),
        "resolve-plugin-self-test": "passed\n",
        "container-doctor": "passed\n",
        "container-process": "passed\n",
        "container-verify": "passed\n",
        "container-vulnerability-scan": "",
        "container-configuration-scan": "",
    }
    while len(logs) < checkpoint["result"]["steps_per_pass"]:
        number = len(logs) + 1
        logs[f"fixture-step-{number}"] = "passed\n"

    identities = checkpoint["reproducibility"]["artifact_sha256"]
    artifacts = {name: {"sha256": digest} for name, digest in identities.items()}
    runs: list[dict[str, Any]] = []
    for index, duration in enumerate(checkpoint["result"]["pass_duration_seconds"], 1):
        log_root = tmp_path / f"pass-{index}" / "logs"
        log_root.mkdir(parents=True)
        steps: list[dict[str, Any]] = []
        for offset, (name, content) in enumerate(logs.items(), 1):
            log_name = f"{offset:02d}-{name}.log"
            path = log_root / log_name
            path.write_text(content, encoding="utf-8")
            steps.append(
                {
                    "name": name,
                    "command": ["fixture"],
                    "duration_seconds": 0.01,
                    "exit_code": 0,
                    "log": log_name,
                    "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        runs.append(
            {
                "index": index,
                "status": "passed",
                "duration_seconds": duration,
                "steps": steps,
                "artifacts": artifacts,
                "container_packages": {"verified": 77, "locked": 77},
                "python_tools": {
                    name: checkpoint["toolchain"][name]
                    for name in ("python", "ruff", "mypy", "pytest")
                },
            }
        )

    toolchain_map = {
        "python": "python",
        "resolve_engine_python": "resolve-engine-python",
        "uv": "uv",
        "node": "node",
        "npm": "npm",
        "pnpm": "pnpm",
        "docker": "docker",
        "trivy": "trivy",
        "pip_audit": "pip-audit",
        "check_jsonschema": "check-jsonschema",
        "ffmpeg": "ffmpeg",
        "trivy_database_updated_at": "trivy-database-updated-at",
        "trivy_check_bundle_digest": "trivy-check-bundle-digest",
    }
    proof: dict[str, Any] = {
        "schema_version": 1,
        "source_commit": checkpoint["source_commit"],
        "status": "passed",
        "toolchain_lock_sha256": checkpoint["toolchain"]["toolchain_lock_sha256"],
        "toolchain": {
            raw: checkpoint["toolchain"][compact] for compact, raw in toolchain_map.items()
        },
        "external_inputs": {},
        "runs": runs,
        "reproducibility": {
            "status": "passed",
            "passes": 2,
            "artifact_sha256": identities,
        },
        "known_limits": checkpoint["known_limits"],
    }
    proof["proof_sha256"] = _canonical(proof)
    proof_path = tmp_path / "release-gate-proof.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint["reproducibility"]["pass_artifact_inventory_canonical_sha256"] = _canonical(
        artifacts
    )
    checkpoint["proof_integrity"].update(
        {
            "full_proof_canonical_sha256": proof["proof_sha256"],
            "full_proof_file_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            "external_input_inventory_canonical_sha256": _canonical({}),
            "pass_step_records_canonical_sha256": [_canonical(run["steps"]) for run in runs],
        }
    )
    return checkpoint, proof_path


def test_committed_checkpoint_has_the_complete_valid_contract() -> None:
    assert validator.validate_checkpoint()["result"]["status"] == "passed"


def test_full_proof_validator_derives_every_compact_metric(tmp_path: Path) -> None:
    checkpoint, proof_path = _fixture(tmp_path)
    assert validator.validate_full_proof(checkpoint, proof_path) == {
        "passes": 2,
        "steps_per_pass": 41,
    }


def test_full_proof_validator_rejects_a_fabricated_test_count(tmp_path: Path) -> None:
    checkpoint, proof_path = _fixture(tmp_path)
    fabricated = copy.deepcopy(checkpoint)
    fabricated["result"]["default_test_suite_per_pass"]["passed"] += 1
    with pytest.raises(validator.CheckpointError, match="test counts differ"):
        validator.validate_full_proof(fabricated, proof_path)


def test_full_proof_validator_rejects_a_tampered_bound_log(tmp_path: Path) -> None:
    checkpoint, proof_path = _fixture(tmp_path)
    log = next((tmp_path / "pass-1" / "logs").glob("*-ui-tests.log"))
    log.write_text("Tests 343 passed\n", encoding="utf-8")
    with pytest.raises(validator.CheckpointError, match="log digest mismatch"):
        validator.validate_full_proof(checkpoint, proof_path)


def test_checkpoint_rejects_an_incomplete_artifact_set(tmp_path: Path) -> None:
    checkpoint = json.loads(validator.DEFAULT_CHECKPOINT.read_text(encoding="utf-8"))
    checkpoint["reproducibility"]["artifact_sha256"].pop("wheel")
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(validator.CheckpointError, match="promised set"):
        validator.validate_checkpoint(path)


def test_checkpoint_rejects_a_tool_version_that_differs_from_its_lock(tmp_path: Path) -> None:
    checkpoint = json.loads(validator.DEFAULT_CHECKPOINT.read_text(encoding="utf-8"))
    checkpoint["toolchain"]["pnpm"] = "999.0.0"
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(validator.CheckpointError, match="pnpm differs from its committed lock"):
        validator.validate_checkpoint(path)
