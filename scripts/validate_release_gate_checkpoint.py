#!/usr/bin/env python3
"""Validate the committed release-gate checkpoint and, optionally, its raw proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "evidence" / "release" / "t3.1-release-gate-refresh.json"
DEFAULT_TOOLCHAIN_LOCK = ROOT / "evidence" / "release" / "toolchain-lock.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_SUMMARY = re.compile(
    r"(?P<passed>\d+) passed, (?P<skipped>\d+) skipped, "
    r"(?P<deselected>\d+) deselected"
)
COVERAGE = re.compile(r"Total coverage: (?P<coverage>\d+(?:\.\d+)?)%")
FUZZ_SUMMARY = re.compile(r"(?P<passed>\d+) passed in")
MUTATION_SUMMARY = re.compile(r"(?P<caught>\d+)/(?P<total>\d+) caught")
UI_SUMMARY = re.compile(r"Tests\s+(?P<passed>\d+) passed")
ARTIFACTS = {
    "audio-regression",
    "container-audio",
    "container-image",
    "resolve-engine",
    "resolve-plugin",
    "sbom",
    "sdist",
    "ui",
    "wheel",
    "wheel-smoke-audio",
}


class CheckpointError(RuntimeError):
    """A compact checkpoint or retained release proof is invalid."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CheckpointError(f"cannot read retained proof input {path}: {exc}") from exc
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CheckpointError(f"{label} must be an integer >= {minimum}")
    return value


def _hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CheckpointError(f"{label} is not a valid digest")
    return value


def validate_checkpoint(path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    """Validate the complete shape and internal claims of the compact checkpoint."""
    checkpoint = _load_object(path, "release-gate checkpoint")
    expected_top = {
        "schema_version",
        "task_id",
        "recorded_on",
        "source_commit",
        "command",
        "result",
        "reproducibility",
        "proof_integrity",
        "toolchain",
        "known_limits",
    }
    if set(checkpoint) != expected_top or checkpoint.get("schema_version") != 1:
        raise CheckpointError("release-gate checkpoint top-level contract differs")
    if checkpoint.get("task_id") != "T3.1" or checkpoint.get("command") != (
        "bash scripts/run_release_checks.sh"
    ):
        raise CheckpointError("release-gate checkpoint identity differs")
    if (
        not isinstance(checkpoint.get("recorded_on"), str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", checkpoint["recorded_on"]) is None
    ):
        raise CheckpointError("checkpoint recorded_on must be an ISO calendar date")
    _hex(checkpoint.get("source_commit"), HEX40, "checkpoint source commit")

    result = _object(checkpoint.get("result"), "checkpoint result")
    expected_result = {
        "status",
        "isolated_clean_checkout_passes",
        "steps_per_pass",
        "pass_duration_seconds",
        "default_test_suite_per_pass",
        "separate_fuzz_tests_per_pass",
        "owner_scoped_mutations_caught_per_pass",
        "real_audio_cases",
        "runs_per_real_audio_case",
        "ui_tests_per_pass",
        "python_ui_plugin_and_toolchain_audits",
        "resolve_staged_lifecycle_self_test",
        "container",
        "generated_identity_schema_status_and_design_drift",
        "tracked_tree_drift",
    }
    if set(result) != expected_result:
        raise CheckpointError("checkpoint result fields differ from the version-1 contract")
    if result.get("status") != "passed":
        raise CheckpointError("checkpoint result is not passing")
    passes = _integer(result.get("isolated_clean_checkout_passes"), "checkout passes", minimum=2)
    _integer(result.get("steps_per_pass"), "steps per pass", minimum=1)
    durations = result.get("pass_duration_seconds")
    if (
        not isinstance(durations, list)
        or len(durations) != passes
        or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and item > 0
            for item in durations
        )
    ):
        raise CheckpointError("pass durations do not match the declared pass count")
    suite = _object(result.get("default_test_suite_per_pass"), "default suite")
    if set(suite) != {
        "passed",
        "skipped",
        "fuzz_deselected",
        "branch_coverage_percent",
        "required_branch_coverage_percent",
    }:
        raise CheckpointError("default-suite fields differ from the version-1 contract")
    for field in ("passed", "skipped", "fuzz_deselected"):
        _integer(suite.get(field), f"default suite {field}")
    coverage = suite.get("branch_coverage_percent")
    floor = suite.get("required_branch_coverage_percent")
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in (coverage, floor)
    ):
        raise CheckpointError("coverage values must be numeric")
    if float(cast(float, coverage)) < float(cast(float, floor)):
        raise CheckpointError("recorded branch coverage is below its floor")
    _integer(result.get("separate_fuzz_tests_per_pass"), "separate fuzz tests", minimum=1)
    mutations = result.get("owner_scoped_mutations_caught_per_pass")
    match = re.fullmatch(r"(?P<caught>\d+)/(?P<total>\d+)", str(mutations))
    if match is None or match.group("caught") != match.group("total"):
        raise CheckpointError("owner-scoped mutation result is not a complete catch")
    _integer(result.get("real_audio_cases"), "real-audio cases", minimum=1)
    _integer(result.get("runs_per_real_audio_case"), "real-audio runs", minimum=2)
    _integer(result.get("ui_tests_per_pass"), "UI tests", minimum=1)
    if result.get("python_ui_plugin_and_toolchain_audits") != "zero known vulnerabilities":
        raise CheckpointError("dependency audits are not recorded clean")
    if result.get("resolve_staged_lifecycle_self_test") != "passed in both checkouts":
        raise CheckpointError("Resolve staged lifecycle is not recorded passing")
    for field in ("generated_identity_schema_status_and_design_drift", "tracked_tree_drift"):
        if result.get(field) != 0:
            raise CheckpointError(f"checkpoint records non-zero {field}")
    container = _object(result.get("container"), "container result")
    if set(container) != {
        "exact_locked_packages",
        "non_root_read_only_doctor_process_verify",
        "high_or_critical_vulnerabilities",
        "high_or_critical_configuration_findings",
    }:
        raise CheckpointError("container fields differ from the version-1 contract")
    if container.get("non_root_read_only_doctor_process_verify") != "passed in both checkouts":
        raise CheckpointError("container smoke path is not recorded passing")
    if (
        container.get("high_or_critical_vulnerabilities") != 0
        or container.get("high_or_critical_configuration_findings") != 0
    ):
        raise CheckpointError("container checkpoint records high/critical findings")
    package_match = re.fullmatch(
        r"(?P<verified>\d+)/(?P<locked>\d+)", str(container.get("exact_locked_packages"))
    )
    if package_match is None or package_match.group("verified") != package_match.group("locked"):
        raise CheckpointError("container package count is malformed")

    reproducibility = _object(checkpoint.get("reproducibility"), "reproducibility")
    if set(reproducibility) != {
        "status",
        "artifact_sha256",
        "pass_artifact_inventory_canonical_sha256",
    }:
        raise CheckpointError("reproducibility fields differ from the version-1 contract")
    if reproducibility.get("status") != "passed":
        raise CheckpointError("artifact reproducibility is not passing")
    artifacts = _object(reproducibility.get("artifact_sha256"), "artifact identities")
    if set(artifacts) != ARTIFACTS:
        raise CheckpointError("artifact identity set differs from the promised set")
    for name, digest in artifacts.items():
        _hex(digest, HEX64, f"artifact {name}")
    _hex(
        reproducibility.get("pass_artifact_inventory_canonical_sha256"),
        HEX64,
        "artifact inventory",
    )

    integrity = _object(checkpoint.get("proof_integrity"), "proof integrity")
    if set(integrity) != {
        "retained_proof_path",
        "full_proof_canonical_sha256",
        "full_proof_file_sha256",
        "external_input_inventory_canonical_sha256",
        "pass_step_records_canonical_sha256",
    }:
        raise CheckpointError("proof-integrity fields differ from the version-1 contract")
    for field in (
        "full_proof_canonical_sha256",
        "full_proof_file_sha256",
        "external_input_inventory_canonical_sha256",
    ):
        _hex(integrity.get(field), HEX64, field)
    step_hashes = integrity.get("pass_step_records_canonical_sha256")
    if not isinstance(step_hashes, list) or len(step_hashes) != passes:
        raise CheckpointError("step-record digests do not match the pass count")
    for index, digest in enumerate(step_hashes, 1):
        _hex(digest, HEX64, f"pass {index} step records")
    retained = integrity.get("retained_proof_path")
    if (
        not isinstance(retained, str)
        or PurePosixPath(retained).is_absolute()
        or ".." in PurePosixPath(retained).parts
    ):
        raise CheckpointError("retained proof path must be a safe repository-relative path")

    toolchain = _object(checkpoint.get("toolchain"), "checkpoint toolchain")
    expected_toolchain = {
        "toolchain_lock_sha256",
        "python",
        "resolve_engine_python",
        "uv",
        "ruff",
        "mypy",
        "pytest",
        "node",
        "npm",
        "pnpm",
        "docker",
        "trivy",
        "pip_audit",
        "check_jsonschema",
        "ffmpeg",
        "trivy_database_updated_at",
        "trivy_check_bundle_digest",
    }
    if set(toolchain) != expected_toolchain or any(
        not isinstance(value, str) or not value for value in toolchain.values()
    ):
        raise CheckpointError("toolchain fields differ from the version-1 contract")
    for field in ("toolchain_lock_sha256", "trivy_check_bundle_digest"):
        value = toolchain.get(field)
        pattern = re.compile(r"^sha256:[0-9a-f]{64}$") if field.endswith("bundle_digest") else HEX64
        _hex(value, pattern, f"toolchain {field}")
    if (
        not isinstance(checkpoint.get("known_limits"), list)
        or not checkpoint["known_limits"]
        or not all(isinstance(limit, str) and limit for limit in checkpoint["known_limits"])
    ):
        raise CheckpointError("checkpoint must retain its known limits")
    lock = _load_object(DEFAULT_TOOLCHAIN_LOCK, "toolchain lock")
    if _file_sha256(DEFAULT_TOOLCHAIN_LOCK) != toolchain["toolchain_lock_sha256"]:
        raise CheckpointError("committed toolchain lock digest differs from the checkpoint")
    locked_tools = _object(lock.get("tools"), "locked tools")
    tool_name_map = {
        "python": "python",
        "resolve_engine_python": "resolve-engine-python",
        "uv": "uv",
        "ruff": "ruff",
        "mypy": "mypy",
        "pytest": "pytest",
        "node": "node",
        "npm": "npm",
        "pnpm": "pnpm",
        "docker": "docker",
        "trivy": "trivy",
        "pip_audit": "pip-audit",
        "check_jsonschema": "check-jsonschema",
        "ffmpeg": "ffmpeg",
    }
    for compact_name, lock_name in tool_name_map.items():
        if toolchain[compact_name] != locked_tools.get(lock_name):
            raise CheckpointError(f"toolchain {compact_name} differs from its committed lock")
    return checkpoint


def _step_logs(run: dict[str, Any], root: Path, expected_steps: int) -> dict[str, str]:
    steps = run.get("steps")
    if not isinstance(steps, list) or len(steps) != expected_steps:
        raise CheckpointError("raw proof step count differs from the checkpoint")
    logs: dict[str, str] = {}
    for step in steps:
        item = _object(step, "raw proof step")
        name = item.get("name")
        log_name = item.get("log")
        if not isinstance(name, str) or not name or name in logs:
            raise CheckpointError("raw proof step names must be unique strings")
        if item.get("exit_code") != 0 or not isinstance(log_name, str):
            raise CheckpointError(f"raw proof step did not pass cleanly: {name}")
        log_path = root / log_name
        if _file_sha256(log_path) != item.get("log_sha256"):
            raise CheckpointError(f"raw proof log digest mismatch: {name}")
        logs[name] = log_path.read_text(encoding="utf-8", errors="replace")
    return logs


def _match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    found = pattern.search(text)
    if found is None:
        raise CheckpointError(f"cannot derive {label} from its retained log")
    return found


def _zero_vulnerabilities(text: str, label: str) -> None:
    start = text.find("{")
    if start < 0:
        raise CheckpointError(f"{label} retained log does not contain JSON")
    try:
        report = _object(json.loads(text[start:]), f"{label} report")
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{label} retained log contains invalid JSON: {exc}") from exc
    metadata = report.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("vulnerabilities"), dict):
        vulnerabilities = metadata["vulnerabilities"]
        if any(not isinstance(value, int) or value != 0 for value in vulnerabilities.values()):
            raise CheckpointError(f"{label} contains a non-zero vulnerability count")
        return
    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(dependency, dict) or dependency.get("vulns") != []
        for dependency in dependencies
    ):
        raise CheckpointError(f"{label} does not prove zero known vulnerabilities")


def _verify_logged_metrics(logs: dict[str, str], result: dict[str, Any]) -> None:
    default = _match(DEFAULT_SUMMARY, logs["default-tests-branch-coverage"], "default tests")
    suite = _object(result.get("default_test_suite_per_pass"), "default suite")
    if [int(default.group(name)) for name in ("passed", "skipped", "deselected")] != [
        suite["passed"],
        suite["skipped"],
        suite["fuzz_deselected"],
    ]:
        raise CheckpointError("default test counts differ from the retained log")
    coverage = _match(COVERAGE, logs["default-tests-branch-coverage"], "branch coverage")
    if float(coverage.group("coverage")) != float(suite["branch_coverage_percent"]):
        raise CheckpointError("branch coverage differs from the retained log")
    fuzz = _match(FUZZ_SUMMARY, logs["fuzz-tests"], "fuzz tests")
    if int(fuzz.group("passed")) != result["separate_fuzz_tests_per_pass"]:
        raise CheckpointError("fuzz count differs from the retained log")
    mutation = _match(MUTATION_SUMMARY, logs["mutation-gate"], "mutation result")
    if (
        f"{mutation.group('caught')}/{mutation.group('total')}"
        != result["owner_scoped_mutations_caught_per_pass"]
    ):
        raise CheckpointError("mutation result differs from the retained log")
    audio_text = logs["real-audio-regressions"]
    audio = json.loads(audio_text[audio_text.find("{") :])
    cases = audio.get("cases") if isinstance(audio, dict) else None
    if (
        not isinstance(cases, list)
        or len(cases) != result["real_audio_cases"]
        or any(
            not isinstance(case, dict)
            or case.get("status") != "passed"
            or case.get("runs") != result["runs_per_real_audio_case"]
            for case in cases
        )
    ):
        raise CheckpointError("real-audio counts or results differ from the retained log")
    ui = _match(UI_SUMMARY, logs["ui-tests"], "UI tests")
    if int(ui.group("passed")) != result["ui_tests_per_pass"]:
        raise CheckpointError("UI test count differs from the retained log")
    for name in ("ui-audit", "plugin-audit", "toolchain-audit", "python-audit"):
        _zero_vulnerabilities(logs[name], name)


def validate_full_proof(checkpoint: dict[str, Any], proof_path: Path) -> dict[str, int]:
    """Derive the compact checkpoint's hashes and metrics from a retained raw proof."""
    proof = _load_object(proof_path, "retained release-gate proof")
    integrity = _object(checkpoint["proof_integrity"], "proof integrity")
    if _file_sha256(proof_path) != integrity["full_proof_file_sha256"]:
        raise CheckpointError("retained proof file digest differs from the checkpoint")
    claimed = proof.get("proof_sha256")
    covered = dict(proof)
    covered.pop("proof_sha256", None)
    actual = _canonical_sha256(covered)
    if claimed != actual or actual != integrity["full_proof_canonical_sha256"]:
        raise CheckpointError("retained proof canonical digest does not recompute")
    if proof.get("status") != "passed" or proof.get("source_commit") != checkpoint["source_commit"]:
        raise CheckpointError("retained proof status/source differs from the checkpoint")
    if (
        _canonical_sha256(proof.get("external_inputs"))
        != integrity["external_input_inventory_canonical_sha256"]
    ):
        raise CheckpointError("external-input inventory digest differs from the checkpoint")
    if proof.get("toolchain_lock_sha256") != checkpoint["toolchain"]["toolchain_lock_sha256"]:
        raise CheckpointError("toolchain lock digest differs from the retained proof")
    raw_toolchain = _object(proof.get("toolchain"), "raw proof toolchain")
    toolchain_map = {
        "python": "python",
        "resolve_engine_python": "resolve-engine-python",
        "uv": "uv",
        "node": "node",
        "npm": "npm",
        "docker": "docker",
        "trivy": "trivy",
        "pip_audit": "pip-audit",
        "check_jsonschema": "check-jsonschema",
        "ffmpeg": "ffmpeg",
        "trivy_database_updated_at": "trivy-database-updated-at",
        "trivy_check_bundle_digest": "trivy-check-bundle-digest",
    }
    for compact_name, raw_name in toolchain_map.items():
        if checkpoint["toolchain"][compact_name] != raw_toolchain.get(raw_name):
            raise CheckpointError(f"toolchain {compact_name} differs from the retained proof")

    result = _object(checkpoint["result"], "checkpoint result")
    reproducibility = _object(checkpoint["reproducibility"], "checkpoint reproducibility")
    runs = proof.get("runs")
    pass_count = result["isolated_clean_checkout_passes"]
    if not isinstance(runs, list) or len(runs) != pass_count:
        raise CheckpointError("retained proof pass count differs from the checkpoint")
    for offset, raw_run in enumerate(runs):
        run = _object(raw_run, f"raw proof pass {offset + 1}")
        if run.get("status") != "passed" or run.get("index") != offset + 1:
            raise CheckpointError(f"raw proof pass {offset + 1} is not passing")
        if run.get("duration_seconds") != result["pass_duration_seconds"][offset]:
            raise CheckpointError("pass duration differs from the retained proof")
        if (
            _canonical_sha256(run.get("steps"))
            != integrity["pass_step_records_canonical_sha256"][offset]
        ):
            raise CheckpointError("step-record digest differs from the retained proof")
        artifacts = _object(run.get("artifacts"), "raw proof artifacts")
        if (
            _canonical_sha256(artifacts)
            != reproducibility["pass_artifact_inventory_canonical_sha256"]
        ):
            raise CheckpointError("artifact-inventory digest differs from the checkpoint")
        digests = {
            name: _object(identity, f"artifact {name}").get("sha256")
            for name, identity in artifacts.items()
        }
        if digests != reproducibility["artifact_sha256"]:
            raise CheckpointError("artifact identities differ from the checkpoint")
        packages = _object(run.get("container_packages"), "container packages")
        expected_packages = f"{packages.get('verified')}/{packages.get('locked')}"
        if expected_packages != result["container"]["exact_locked_packages"]:
            raise CheckpointError("container package count differs from the retained proof")
        python_tools = _object(run.get("python_tools"), "raw proof Python tools")
        for name in ("python", "ruff", "mypy", "pytest"):
            if python_tools.get(name) != checkpoint["toolchain"][name]:
                raise CheckpointError(f"toolchain {name} differs from raw proof pass {offset + 1}")
        log_root = proof_path.parent / f"pass-{offset + 1}" / "logs"
        logs = _step_logs(run, log_root, result["steps_per_pass"])
        _verify_logged_metrics(logs, result)
        required = {
            "resolve-plugin-self-test",
            "container-doctor",
            "container-process",
            "container-verify",
            "container-vulnerability-scan",
            "container-configuration-scan",
        }
        if not required.issubset(logs):
            raise CheckpointError("retained proof omits a required runtime/security step")
    raw_repro = _object(proof.get("reproducibility"), "raw reproducibility")
    if raw_repro.get("status") != "passed" or raw_repro.get("passes") != pass_count:
        raise CheckpointError("raw reproducibility result differs from the checkpoint")
    if raw_repro.get("artifact_sha256") != reproducibility["artifact_sha256"]:
        raise CheckpointError("raw reproducibility identities differ from the checkpoint")
    return {"passes": pass_count, "steps_per_pass": result["steps_per_pass"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--full-proof",
        type=Path,
        help="also bind the checkpoint to a retained raw proof and all of its logs",
    )
    args = parser.parse_args()
    try:
        checkpoint = validate_checkpoint(args.checkpoint)
        summary: dict[str, Any] = {"checkpoint": "passed"}
        if args.full_proof is not None:
            summary.update(validate_full_proof(checkpoint, args.full_proof))
            summary["full_proof"] = "passed"
    except (CheckpointError, json.JSONDecodeError) as exc:
        print(f"release-gate checkpoint validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
