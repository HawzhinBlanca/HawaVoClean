#!/usr/bin/env python3
"""Generate or verify the evidence-derived release status snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

if not __package__:  # Direct ``python scripts/generate_release_status.py`` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_release_gate_checkpoint import CheckpointError, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "generated-release-status.md"
RELEASE_IDENTITY = ROOT / "src" / "hawavoclean" / "release.json"
FULL_GATE_PROOF = ROOT / "evidence" / "release" / "t3.1-release-gate-refresh.json"
RUNTIME_RISK_PROOF = ROOT / "evidence" / "release" / "t4.6-resolve-runtime-proof.json"
PROTOCOL = ROOT / "evidence" / "release" / "sorani-evaluation-protocol.json"
SOURCE_ASSESSMENT = ROOT / "evidence" / "release" / "sorani-corpus-source-assessment.json"
LEDGER = ROOT / "evidence" / "release" / "ledger.jsonl"
PLAN = ROOT / "docs" / "true-10-plan.md"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TASK = re.compile(r"^- \[(?P<done>[ x])\] \*\*T[0-9]+\.[0-9]+\b", re.MULTILINE)


class ReleaseStatusError(RuntimeError):
    """A source proof is invalid or the generated status is stale."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseStatusError(f"{label} must be an object")
    return value


def _integer(mapping: dict[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReleaseStatusError(f"{label} must be an integer")
    return value


def _string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ReleaseStatusError(f"{label} must be a non-empty string")
    return value


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseStatusError(f"cannot read {label}: {exc}") from exc


def _validate_ledger() -> None:
    try:
        lines = [line for line in LEDGER.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:
        raise ReleaseStatusError(f"cannot read evidence ledger: {exc}") from exc
    if not lines:
        raise ReleaseStatusError("evidence ledger is empty")
    try:
        entries = [
            _object(json.loads(line), f"ledger line {index}") for index, line in enumerate(lines, 1)
        ]
    except json.JSONDecodeError as exc:
        raise ReleaseStatusError(f"invalid evidence ledger JSON: {exc}") from exc
    for index, entry in enumerate(entries, 1):
        if entry.get("sequence") != index:
            raise ReleaseStatusError("evidence ledger sequence is not contiguous")
        digest = entry.get("entry_sha256")
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise ReleaseStatusError(f"ledger entry {index} has an invalid digest")
        expected_previous = None if index == 1 else entries[index - 2]["entry_sha256"]
        if entry.get("previous_entry_sha256") != expected_previous:
            raise ReleaseStatusError(f"ledger entry {index} does not link to its predecessor")
        covered = dict(entry)
        covered.pop("entry_sha256", None)
        canonical = json.dumps(
            covered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        actual = hashlib.sha256(canonical).hexdigest()
        if digest != actual:
            raise ReleaseStatusError(f"ledger entry {index} digest does not recompute")


def _task_counts() -> tuple[int, int]:
    try:
        matches = list(TASK.finditer(PLAN.read_text(encoding="utf-8")))
    except OSError as exc:
        raise ReleaseStatusError(f"cannot read true-10 plan: {exc}") from exc
    if not matches:
        raise ReleaseStatusError("true-10 plan contains no task checkboxes")
    return sum(match.group("done") == "x" for match in matches), len(matches)


def _approval(value: dict[str, Any], label: str) -> tuple[str, str]:
    approval = _object(value.get("approval"), f"{label} approval")
    integrity = _object(value.get("integrity"), f"{label} integrity")
    status = _string(approval, "status", f"{label} approval status")
    digest = _string(integrity, "design_sha256", f"{label} design digest")
    if status not in {"pending_user_approval", "approved"}:
        raise ReleaseStatusError(f"{label} approval status is invalid")
    if HEX64.fullmatch(digest) is None:
        raise ReleaseStatusError(f"{label} design digest is invalid")
    return status, digest


def render_status() -> str:
    """Return the deterministic Markdown snapshot derived from committed evidence."""
    identity = _json(RELEASE_IDENTITY, "release identity")
    try:
        full_gate = validate_checkpoint(FULL_GATE_PROOF)
    except CheckpointError as exc:
        raise ReleaseStatusError(f"invalid T3.1 full-gate checkpoint: {exc}") from exc
    runtime = _json(RUNTIME_RISK_PROOF, "T4.6 runtime proof")
    protocol = _json(PROTOCOL, "Sorani protocol")
    sources = _json(SOURCE_ASSESSMENT, "Sorani source assessment")
    _validate_ledger()
    completed, total = _task_counts()

    version = _string(identity, "version", "release version")
    report_schema = _integer(identity, "report_schema_version", "report schema version")

    gate_result = _object(full_gate.get("result"), "T3.1 result")
    if gate_result.get("status") != "passed":
        raise ReleaseStatusError("the recorded T3.1 proof is not passing")
    source_commit = _string(full_gate, "source_commit", "T3.1 source commit")
    if HEX40.fullmatch(source_commit) is None:
        raise ReleaseStatusError("T3.1 source commit is invalid")
    suite = _object(gate_result.get("default_test_suite_per_pass"), "T3.1 default suite")
    container = _object(gate_result.get("container"), "T3.1 container result")
    reproducibility = _object(full_gate.get("reproducibility"), "T3.1 reproducibility")
    if reproducibility.get("status") != "passed":
        raise ReleaseStatusError("the recorded T3.1 artifacts are not reproducible")
    artifacts = _object(reproducibility.get("artifact_sha256"), "T3.1 artifact identities")
    expected_artifacts = {
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
    if set(artifacts) != expected_artifacts or any(
        not isinstance(digest, str) or HEX64.fullmatch(digest) is None
        for digest in artifacts.values()
    ):
        raise ReleaseStatusError("T3.1 artifact identities are incomplete or invalid")
    proof_integrity = _object(full_gate.get("proof_integrity"), "T3.1 proof integrity")
    for field in (
        "full_proof_canonical_sha256",
        "full_proof_file_sha256",
        "external_input_inventory_canonical_sha256",
    ):
        if HEX64.fullmatch(_string(proof_integrity, field, f"T3.1 {field}")) is None:
            raise ReleaseStatusError(f"T3.1 {field} is invalid")

    assessment = _object(runtime.get("assessment"), "T4.6 assessment")
    controlled = _object(runtime.get("controlled_runtime"), "controlled runtime")
    vendor = _object(runtime.get("vendor_runtime"), "vendor runtime")
    host = _object(runtime.get("resolve_host"), "Resolve host")
    vendor_advisories = _object(runtime.get("vendor_advisories"), "vendor advisories")

    protocol_status, protocol_digest = _approval(protocol, "Sorani protocol")
    source_status, source_digest = _approval(sources, "Sorani source assessment")
    lines = [
        "# Generated release status",
        "",
        "<!-- Generated by scripts/generate_release_status.py. Do not edit by hand. -->",
        "",
        "This is a deterministic snapshot of committed evidence, not a claim that the current HEAD",
        "has completed the human, host, governance, or final-release gates.",
        "",
        "| Fact | Evidence-derived value |",
        "|---|---|",
        f"| Candidate identity | HawaVoClean {version}; report schema {report_schema} |",
        f"| True-10 plan | {completed}/{total} tasks have committed completion proof |",
        f"| Last full local gate | Passed on `{source_commit}`; {_integer(gate_result, 'isolated_clean_checkout_passes', 'isolated passes')} isolated passes × {_integer(gate_result, 'steps_per_pass', 'steps per pass')} steps |",
        f"| Default suite in each full-gate pass | {_integer(suite, 'passed', 'default passed')} passed, {_integer(suite, 'skipped', 'default skipped')} skipped, {_integer(suite, 'fuzz_deselected', 'fuzz deselected')} fuzz-only deselected; {suite.get('branch_coverage_percent')}% branch coverage (floor {suite.get('required_branch_coverage_percent')}%) |",
        f"| Separate mutation/fuzz/UI gates | {gate_result.get('owner_scoped_mutations_caught_per_pass')} mutations; {_integer(gate_result, 'separate_fuzz_tests_per_pass', 'fuzz count')} fuzz cases; {_integer(gate_result, 'ui_tests_per_pass', 'UI test count')} UI tests per pass |",
        "| Reproduced release identities | Wheel, sdist, UI, Resolve engine/plugin, container, SBOM and engineering-audio identities matched across both passes |",
        f"| CPU container at the full gate | {container.get('exact_locked_packages')} packages; {_integer(container, 'high_or_critical_vulnerabilities', 'container vulnerabilities')} high/critical vulnerabilities; {_integer(container, 'high_or_critical_configuration_findings', 'container configuration findings')} high/critical configuration findings |",
        f"| Sorani protocol | `{protocol_status}`; design `{protocol_digest}` |",
        f"| Sorani source route | `{source_status}`; design `{source_digest}` |",
        f"| Controlled standalone Electron | {controlled.get('electron')}; {_integer(assessment, 'controlled_high_or_critical_count', 'controlled advisories')} high/critical advisories at capture |",
        f"| Resolve host boundary | Resolve {host.get('version')} embeds Electron {vendor.get('electron')}; {_integer(vendor_advisories, 'total', 'vendor advisories')} advisories including {_integer(assessment, 'vendor_high_or_critical_count', 'vendor high advisories')} high/critical; explicit acceptance/update required |",
        "",
        "## What this does not prove",
        "",
        "- The full two-pass proof is bound to the named source commit, not automatically to later",
        "  documentation, protocol, source-audit, or release commits. The final candidate must rerun it.",
        "- Protocol and source designs are structurally valid but unapproved; no held-out Sorani",
        "  corpus, dual-review verdict, listening score, or product-quality result exists yet.",
        "- The staged Resolve lifecycle is not the real in-host workflow, keyboard, VoiceOver, or",
        "  timeline matrix. Those require the user-authorized installation and DaVinci Resolve run.",
        "- GitHub CI, branch protection, vendor-risk acceptance, final artifact signing, protected",
        "  merge, tag, and publication remain open.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the generated Markdown")
    args = parser.parse_args(argv)
    try:
        rendered = render_status()
        if args.write:
            OUTPUT.write_text(rendered, encoding="utf-8")
            print(f"wrote {OUTPUT.relative_to(ROOT)}")
            return 0
        current = OUTPUT.read_text(encoding="utf-8")
        if current != rendered:
            raise ReleaseStatusError(
                "generated release status is stale; run scripts/generate_release_status.py --write"
            )
    except (OSError, ReleaseStatusError) as exc:
        print(f"release status generation failed: {exc}", file=sys.stderr)
        return 1
    print("generated release status is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
