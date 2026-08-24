#!/usr/bin/env python3
"""Verify and append HawaVoClean's hash-chained release evidence ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "evidence" / "release"
DEFAULT_BASELINE = EVIDENCE_DIR / "baseline.json"
DEFAULT_LEDGER = EVIDENCE_DIR / "ledger.jsonl"
DEFAULT_SCHEMA = EVIDENCE_DIR / "evidence-entry.schema.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TASK_ID = re.compile(r"^T[0-9]+\.[0-9]+$")
ENTRY_REQUIRED = {
    "schema_version",
    "sequence",
    "recorded_at",
    "task_id",
    "source_commit",
    "command",
    "tools",
    "inputs",
    "result",
    "outputs",
    "known_limits",
    "previous_entry_sha256",
    "entry_sha256",
}


class EvidenceError(ValueError):
    """Release evidence is malformed, incomplete, or does not verify."""


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading a release artifact fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_entry_bytes(entry: dict[str, Any]) -> bytes:
    """Canonical bytes covered by ``entry_sha256``."""
    payload = dict(entry)
    payload.pop("entry_sha256", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def compute_entry_sha256(entry: dict[str, Any]) -> str:
    """Compute an evidence entry's self digest."""
    return sha256_bytes(canonical_entry_bytes(entry))


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object in {path}")
    return value


def _safe_git_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError("tracked artifact path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise EvidenceError(f"unsafe tracked artifact path: {value!r}")
    return value


def _git_blob(repo_root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceError(f"cannot read {path} at {commit}: {detail}")
    return result.stdout


def _verify_commit_exists(repo_root: Path, commit: str) -> None:
    if not HEX40.fullmatch(commit):
        raise EvidenceError(f"invalid full commit hash: {commit!r}")
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise EvidenceError(f"missing commit object: {commit}")


def _artifact_list(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError(f"{field} must be a list")
    artifacts: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise EvidenceError(f"{field}[{index}] must be an object")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path:
            raise EvidenceError(f"{field}[{index}].path must be a non-empty string")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise EvidenceError(f"{field}[{index}].sha256 must be 64 lowercase hex")
        artifacts.append(item)
    return artifacts


def verify_schema_contract(path: Path = DEFAULT_SCHEMA) -> None:
    """Check that the committed JSON schema requires the fields the verifier enforces."""
    schema = _json_object(path)
    required = schema.get("required")
    if not isinstance(required, list) or set(required) != ENTRY_REQUIRED:
        raise EvidenceError("evidence schema required fields differ from verifier contract")
    if schema.get("additionalProperties") is not False:
        raise EvidenceError("evidence schema must reject additional top-level properties")


def validate_entry(
    entry: dict[str, Any], *, expected_sequence: int, expected_previous: str | None
) -> None:
    """Validate one ledger entry's shape, chain position, and self digest."""
    if set(entry) != ENTRY_REQUIRED:
        missing = sorted(ENTRY_REQUIRED - set(entry))
        extra = sorted(set(entry) - ENTRY_REQUIRED)
        raise EvidenceError(f"entry fields differ: missing={missing}, extra={extra}")
    if entry["schema_version"] != 1:
        raise EvidenceError("unsupported evidence schema version")
    if entry["sequence"] != expected_sequence:
        raise EvidenceError(
            f"sequence {entry['sequence']!r} does not equal expected {expected_sequence}"
        )
    if entry["previous_entry_sha256"] != expected_previous:
        raise EvidenceError(f"entry {expected_sequence} does not link to its predecessor")
    if not isinstance(entry["recorded_at"], str) or not entry["recorded_at"].endswith("Z"):
        raise EvidenceError("recorded_at must be a UTC ISO-8601 string ending in Z")
    if not isinstance(entry["task_id"], str) or not TASK_ID.fullmatch(entry["task_id"]):
        raise EvidenceError("task_id must match T<number>.<number>")
    if not isinstance(entry["source_commit"], str) or not HEX40.fullmatch(entry["source_commit"]):
        raise EvidenceError("source_commit must be a full lowercase Git hash")
    command = entry["command"]
    if not isinstance(command, dict) or set(command) != {"text", "cwd"}:
        raise EvidenceError("command must contain exactly text and cwd")
    if not all(isinstance(command[key], str) and command[key] for key in command):
        raise EvidenceError("command text and cwd must be non-empty strings")
    tools = entry["tools"]
    if not isinstance(tools, dict) or not all(
        isinstance(key, str) and key and isinstance(value, str) for key, value in tools.items()
    ):
        raise EvidenceError("tools must map non-empty names to string versions")
    _artifact_list(entry["inputs"], "inputs")
    _artifact_list(entry["outputs"], "outputs")
    result = entry["result"]
    if not isinstance(result, dict) or set(result) != {"status", "summary"}:
        raise EvidenceError("result must contain exactly status and summary")
    if result["status"] not in {"passed", "failed", "blocked"}:
        raise EvidenceError("invalid evidence result status")
    if not isinstance(result["summary"], str) or not result["summary"]:
        raise EvidenceError("result summary must be non-empty")
    limits = entry["known_limits"]
    if not isinstance(limits, list) or not all(isinstance(limit, str) for limit in limits):
        raise EvidenceError("known_limits must be a list of strings")
    claimed = entry["entry_sha256"]
    actual = compute_entry_sha256(entry)
    if not isinstance(claimed, str) or claimed != actual:
        raise EvidenceError(
            f"entry {expected_sequence} self digest mismatch: claimed={claimed}, actual={actual}"
        )


def read_and_verify_ledger(path: Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    """Read and verify the complete append-only ledger."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceError(f"cannot read evidence ledger {path}: {exc}") from exc
    if not lines:
        raise EvidenceError("evidence ledger is empty")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"ledger line {number} is not complete JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise EvidenceError(f"ledger line {number} is not an object")
        entry: dict[str, Any] = raw
        validate_entry(entry, expected_sequence=number, expected_previous=previous)
        previous = entry["entry_sha256"]
        entries.append(entry)
    return entries


def verify_baseline(
    path: Path = DEFAULT_BASELINE, *, repo_root: Path = REPO_ROOT, check_external: bool = False
) -> dict[str, int]:
    """Verify baseline Git blobs and, optionally, ignored local real-audio references."""
    baseline = _json_object(path)
    if baseline.get("schema_version") != 1:
        raise EvidenceError("unsupported baseline schema version")
    commits = baseline.get("source_commits")
    if not isinstance(commits, dict) or not commits:
        raise EvidenceError("baseline source_commits must be a non-empty object")
    for commit in commits.values():
        if not isinstance(commit, str):
            raise EvidenceError("baseline commit values must be strings")
        _verify_commit_exists(repo_root, commit)

    tracked = _artifact_list(baseline.get("tracked_artifacts"), "tracked_artifacts")
    for item in tracked:
        commit = item.get("source_commit")
        if not isinstance(commit, str):
            raise EvidenceError("tracked artifact source_commit must be a string")
        _verify_commit_exists(repo_root, commit)
        rel = _safe_git_path(item["path"])
        actual = sha256_bytes(_git_blob(repo_root, commit, rel))
        if actual != item["sha256"]:
            raise EvidenceError(f"tracked baseline digest mismatch: {rel} at {commit}")

    external = _artifact_list(baseline.get("local_external_artifacts"), "local_external_artifacts")
    checked = 0
    if check_external:
        for item in external:
            candidate = (repo_root / item["path"]).resolve()
            try:
                candidate.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise EvidenceError(
                    f"local external artifact escapes repository: {item['path']}"
                ) from exc
            if not candidate.is_file():
                raise EvidenceError(f"local external artifact is unavailable: {item['path']}")
            if sha256_file(candidate) != item["sha256"]:
                raise EvidenceError(f"local external artifact digest mismatch: {item['path']}")
            size = item.get("size_bytes")
            if size is not None and (not isinstance(size, int) or candidate.stat().st_size != size):
                raise EvidenceError(f"local external artifact size mismatch: {item['path']}")
            checked += 1
    return {"commits": len(commits), "tracked_artifacts": len(tracked), "external_checked": checked}


def _parse_tool(values: list[str]) -> dict[str, str]:
    tools: dict[str, str] = {}
    for value in values:
        name, separator, version = value.partition("=")
        if not separator or not name or not version:
            raise EvidenceError(f"tool must be NAME=VERSION, got {value!r}")
        tools[name] = version
    return tools


def _working_artifacts(paths: list[str], repo_root: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for value in paths:
        candidate = (repo_root / value).resolve()
        try:
            rel = candidate.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise EvidenceError(f"evidence path escapes repository: {value}") from exc
        if not candidate.is_file():
            raise EvidenceError(f"evidence artifact does not exist: {value}")
        artifacts.append({"path": rel.as_posix(), "sha256": sha256_file(candidate)})
    return artifacts


def build_entry(
    *,
    sequence: int,
    previous: str | None,
    recorded_at: str,
    task_id: str,
    source_commit: str,
    command: str,
    tools: dict[str, str],
    inputs: list[dict[str, str]],
    status: str,
    summary: str,
    outputs: list[dict[str, str]],
    known_limits: list[str],
) -> dict[str, Any]:
    """Build and self-hash one evidence entry."""
    entry: dict[str, Any] = {
        "schema_version": 1,
        "sequence": sequence,
        "recorded_at": recorded_at,
        "task_id": task_id,
        "source_commit": source_commit,
        "command": {"text": command, "cwd": "."},
        "tools": dict(sorted(tools.items())),
        "inputs": inputs,
        "result": {"status": status, "summary": summary},
        "outputs": outputs,
        "known_limits": known_limits,
        "previous_entry_sha256": previous,
    }
    entry["entry_sha256"] = compute_entry_sha256(entry)
    validate_entry(entry, expected_sequence=sequence, expected_previous=previous)
    return entry


def append_entry(args: argparse.Namespace) -> dict[str, Any]:
    """Append a single flushed line after verifying the existing chain."""
    ledger = Path(args.ledger).resolve()
    entries = read_and_verify_ledger(ledger)
    _verify_commit_exists(REPO_ROOT, args.source_commit)
    entry = build_entry(
        sequence=len(entries) + 1,
        previous=entries[-1]["entry_sha256"],
        recorded_at=args.recorded_at,
        task_id=args.task_id,
        source_commit=args.source_commit,
        command=args.command,
        tools=_parse_tool(args.tool),
        inputs=_working_artifacts(args.input, REPO_ROOT),
        status=args.status,
        summary=args.summary,
        outputs=_working_artifacts(args.output, REPO_ROOT),
        known_limits=args.known_limit,
    )
    encoded = (
        json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(ledger, os.O_APPEND | os.O_WRONLY)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise EvidenceError(f"short append to evidence ledger: {written}/{len(encoded)} bytes")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return entry


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)

    verify = sub.add_parser("verify", help="verify schema, baseline, and ledger")
    verify.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    verify.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    verify.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    verify.add_argument("--check-external", action="store_true")

    append = sub.add_parser("append", help="append one evidence entry")
    append.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    append.add_argument("--recorded-at", default=_utc_now())
    append.add_argument("--task-id", required=True)
    append.add_argument("--source-commit", required=True)
    append.add_argument("--command", required=True)
    append.add_argument("--tool", action="append", default=[])
    append.add_argument("--input", action="append", default=[])
    append.add_argument("--status", choices=("passed", "failed", "blocked"), required=True)
    append.add_argument("--summary", required=True)
    append.add_argument("--output", action="append", default=[])
    append.add_argument("--known-limit", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parser().parse_args(argv)
    try:
        if args.operation == "verify":
            verify_schema_contract(Path(args.schema))
            baseline = verify_baseline(
                Path(args.baseline), repo_root=REPO_ROOT, check_external=args.check_external
            )
            entries = read_and_verify_ledger(Path(args.ledger))
            result: dict[str, Any] = {
                "status": "passed",
                "baseline": baseline,
                "ledger_entries": len(entries),
                "ledger_head": entries[-1]["entry_sha256"],
            }
        else:
            entry = append_entry(args)
            result = {
                "status": "appended",
                "sequence": entry["sequence"],
                "entry_sha256": entry["entry_sha256"],
            }
    except EvidenceError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
