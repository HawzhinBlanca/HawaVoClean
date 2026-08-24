#!/usr/bin/env python3
"""Hydrate CI with hash-locked private regressions without weakening Git hygiene."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from re import fullmatch
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evidence" / "release" / "audio-regressions.json"
PRIVATE_FIELDS = (
    ("input", "input_sha256"),
    ("reference_audio", "audio_sha256"),
    ("reference_report", "report_sha256"),
)


class HydrationError(ValueError):
    """Private regression evidence could not be safely hydrated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise HydrationError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise HydrationError(f"duplicate JSON key is forbidden: {key}")
            value[key] = child
        return value

    try:
        value: object = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HydrationError(f"cannot read regression manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise HydrationError("regression manifest must contain a cases array")
    return value


def _is_ignored(relative: str, destination_root: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=destination_root,
        check=False,
    )
    return result.returncode == 0


def hydrate(source_root: Path, destination_root: Path, manifest_path: Path) -> dict[str, str]:
    """Copy only manifest-named, hash-matching, Git-ignored files into a checkout."""
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if not source_root.is_dir():
        raise HydrationError(f"private evidence root is not a directory: {source_root}")
    if not (destination_root / ".git").exists():
        raise HydrationError(f"destination is not a Git checkout: {destination_root}")
    if source_root.is_relative_to(destination_root) or destination_root.is_relative_to(source_root):
        raise HydrationError("private evidence root and destination checkout must be disjoint")

    manifest = _load_manifest(manifest_path)
    cases = manifest["cases"]
    if not cases:
        raise HydrationError("regression manifest has no cases")

    copied: dict[str, str] = {}
    for case_index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            raise HydrationError(f"case {case_index} is not an object")
        for path_field, hash_field in PRIVATE_FIELDS:
            relative = raw_case.get(path_field)
            expected = raw_case.get(hash_field)
            if (
                not isinstance(relative, str)
                or not relative
                or not isinstance(expected, str)
                or fullmatch(r"[0-9a-f]{64}", expected) is None
            ):
                raise HydrationError(f"case {case_index} has invalid {path_field}/{hash_field}")
            if relative in copied:
                if copied[relative] != expected:
                    raise HydrationError(f"conflicting expected hashes for {relative}")
                continue

            source_unresolved = source_root / relative
            source = source_unresolved.resolve()
            destination = destination_root / relative
            try:
                source.relative_to(source_root)
                destination.resolve(strict=False).relative_to(destination_root)
            except ValueError as exc:
                raise HydrationError(
                    f"private regression path escapes its root: {relative}"
                ) from exc
            if source_unresolved.is_symlink() or not source.is_file():
                raise HydrationError(f"private regression artifact is unavailable: {relative}")
            if not _is_ignored(relative, destination_root):
                raise HydrationError(f"private regression target is not Git-ignored: {relative}")

            actual = _sha256(source)
            if actual != expected:
                raise HydrationError(
                    f"private regression artifact hash drift for {relative}: {actual} != {expected}"
                )
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or not destination.is_file():
                    raise HydrationError(f"private regression target is unsafe: {relative}")
                if _sha256(destination) != expected:
                    raise HydrationError(
                        f"private regression target already has wrong bytes: {relative}"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                destination.chmod(0o600)
            copied[relative] = expected
    return dict(sorted(copied.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        hydrated = hydrate(args.source_root, args.destination_root, args.manifest)
    except HydrationError as exc:
        print(f"release evidence hydration failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "artifacts": hydrated}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
