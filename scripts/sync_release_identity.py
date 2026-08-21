#!/usr/bin/env python3
"""Synchronize or verify generated version mirrors from release.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = ROOT / "src" / "hawavoclean" / "release.json"


class IdentitySyncError(RuntimeError):
    """A generated release-version mirror is missing or stale."""


def _identity_version() -> str:
    value = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "identity_schema_version",
        "product",
        "version",
        "report_schema_version",
    }:
        raise IdentitySyncError("release.json has an unexpected shape")
    version = value["version"]
    if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise IdentitySyncError("release.json version is not canonical SemVer")
    return version


def _json_version(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    version = value.get("version") if isinstance(value, dict) else None
    if not isinstance(version, str):
        raise IdentitySyncError(f"{path.relative_to(ROOT)} has no string version")
    return version


def _manifest_version(path: Path) -> str:
    match = re.search(r"<Version>([^<]+)</Version>", path.read_text(encoding="utf-8"))
    if match is None:
        raise IdentitySyncError(f"{path.relative_to(ROOT)} has no Version element")
    return match.group(1)


def _lock_version(path: Path) -> str:
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    matches = [
        item
        for item in value.get("package", [])
        if item.get("name") == "hawavoclean" and item.get("source") == {"editable": "."}
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("version"), str):
        raise IdentitySyncError("uv.lock must contain exactly one editable hawavoclean package")
    return str(matches[0]["version"])


def _mirrors() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject.get("project", {}).get("version")
    if not isinstance(project_version, str):
        raise IdentitySyncError("pyproject.toml project.version is missing")
    return {
        "pyproject.toml": project_version,
        "uv.lock": _lock_version(ROOT / "uv.lock"),
        "ui/package.json": _json_version(ROOT / "ui" / "package.json"),
        "resolve-plugin/com.hawavoclean.resolve/package.json": _json_version(
            ROOT / "resolve-plugin" / "com.hawavoclean.resolve" / "package.json"
        ),
        "resolve-plugin/com.hawavoclean.resolve/manifest.xml": _manifest_version(
            ROOT / "resolve-plugin" / "com.hawavoclean.resolve" / "manifest.xml"
        ),
    }


def _replace_one(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise IdentitySyncError(f"could not uniquely update {path.relative_to(ROOT)}")
    path.write_text(updated, encoding="utf-8")


def _write_mirrors(version: str) -> None:
    _replace_one(
        ROOT / "pyproject.toml",
        r'(?s)(\[project\].*?\nversion = ")[^"]+("\n)',
        rf"\g<1>{version}\g<2>",
    )
    _replace_one(
        ROOT / "uv.lock",
        r'(\[\[package\]\]\nname = "hawavoclean"\nversion = ")[^"]+("\n)',
        rf"\g<1>{version}\g<2>",
    )
    for relative in (
        "ui/package.json",
        "resolve-plugin/com.hawavoclean.resolve/package.json",
    ):
        path = ROOT / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        value["version"] = version
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    _replace_one(
        ROOT / "resolve-plugin" / "com.hawavoclean.resolve" / "manifest.xml",
        r"(<Version>)[^<]+(</Version>)",
        rf"\g<1>{version}\g<2>",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite generated version mirrors before checking them",
    )
    args = parser.parse_args()
    try:
        version = _identity_version()
        if args.write:
            _write_mirrors(version)
        stale = {path: value for path, value in _mirrors().items() if value != version}
        if stale:
            details = ", ".join(f"{path}={value}" for path, value in stale.items())
            raise IdentitySyncError(f"release identity is {version}; stale mirrors: {details}")
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError, IdentitySyncError) as exc:
        print(f"release identity check failed: {exc}", file=sys.stderr)
        return 1
    print(f"release identity {version}: {len(_mirrors())} generated mirrors agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
