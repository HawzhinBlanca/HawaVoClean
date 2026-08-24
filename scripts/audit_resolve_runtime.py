#!/usr/bin/env python3
"""Capture the exact Resolve-owned Electron runtime and its advisory exposure.

This is deliberately separate from JavaScript lockfile auditing: DaVinci Resolve ships its own
Electron application, outside HawaVoClean's package locks and upgrade authority. The report is JSON
so release evidence can bind a vendor runtime to its on-disk hash, signature and advisory snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_RESOLVE_APP = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app")
DEFAULT_ELECTRON_RELATIVE = Path("Contents/Applications/.hidden/Electron.app")
GITHUB_ADVISORIES_API = "https://api.github.com/advisories"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_plist(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = plistlib.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a plist dictionary")
    return value


def _codesign_evidence(app: Path) -> dict[str, Any]:
    details = subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(app)],
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in (details.stdout, details.stderr) if part)

    def field(name: str) -> str | None:
        match = re.search(rf"^{re.escape(name)}=(.+)$", combined, re.MULTILINE)
        return match.group(1).strip() if match else None

    verify = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "valid_on_disk": verify.returncode == 0,
        "identifier": field("Identifier"),
        "team_identifier": field("TeamIdentifier"),
        "authority": re.findall(r"^Authority=(.+)$", combined, re.MULTILINE),
        "cdhash": field("CDHash"),
        "timestamp": field("Timestamp"),
        "hardened_runtime": "flags=0x10000(runtime)" in combined,
    }


def _fetch_advisories(version: str) -> tuple[str, list[dict[str, Any]]]:
    query = urllib.parse.urlencode(
        {"ecosystem": "npm", "affects": f"electron@{version}", "per_page": "100"}
    )
    url = f"{GITHUB_ADVISORIES_API}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "HawaVoClean-release-audit",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
        value = json.load(response)
    if not isinstance(value, list):
        raise ValueError("GitHub advisory response was not a list")
    return url, value


def _summarize_advisories(
    version: str, query_url: str, advisories: list[dict[str, Any]]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for advisory in advisories:
        ranges = []
        for vulnerability in advisory.get("vulnerabilities", []):
            package = vulnerability.get("package", {})
            if package.get("ecosystem") == "npm" and package.get("name") == "electron":
                patched = vulnerability.get("first_patched_version")
                ranges.append(
                    {
                        "vulnerable_version_range": vulnerability.get("vulnerable_version_range"),
                        "first_patched_version": patched.get("identifier")
                        if isinstance(patched, dict)
                        else None,
                    }
                )
        records.append(
            {
                "ghsa_id": advisory.get("ghsa_id"),
                "cve_id": advisory.get("cve_id"),
                "severity": advisory.get("severity"),
                "summary": advisory.get("summary"),
                "published_at": advisory.get("published_at"),
                "updated_at": advisory.get("updated_at"),
                "url": advisory.get("html_url"),
                "electron_ranges": ranges,
            }
        )
    records.sort(key=lambda record: str(record["ghsa_id"]))
    severities = Counter(str(record["severity"]) for record in records)
    return {
        "package": "electron",
        "version": version,
        "query_url": query_url,
        "response_canonical_sha256": _canonical_sha256(advisories),
        "total": len(records),
        "severity_counts": dict(sorted(severities.items())),
        "high_or_critical": [
            record for record in records if record["severity"] in {"high", "critical"}
        ],
        "advisory_ids": [record["ghsa_id"] for record in records],
    }


def capture(
    resolve_app: Path,
    controlled_package: Path,
    supported_majors: set[int],
) -> dict[str, Any]:
    resolve_info_path = resolve_app / "Contents/Info.plist"
    electron_app = resolve_app / DEFAULT_ELECTRON_RELATIVE
    electron_info_path = electron_app / "Contents/Info.plist"
    electron_binary = electron_app / "Contents/MacOS/Electron"
    for required in (resolve_info_path, electron_info_path, electron_binary, controlled_package):
        if not required.exists():
            raise FileNotFoundError(required)

    resolve_info = _read_plist(resolve_info_path)
    electron_info = _read_plist(electron_info_path)
    package = json.loads(controlled_package.read_text())
    vendor_version = str(electron_info["CFBundleShortVersionString"])
    controlled_version = str(package["devDependencies"]["electron"])
    vendor_url, vendor_raw = _fetch_advisories(vendor_version)
    controlled_url, controlled_raw = _fetch_advisories(controlled_version)
    vendor_scan = _summarize_advisories(vendor_version, vendor_url, vendor_raw)
    controlled_scan = _summarize_advisories(controlled_version, controlled_url, controlled_raw)
    vendor_major = int(vendor_version.split(".", 1)[0])

    return {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "resolve_host": {
            "path": str(resolve_app),
            "bundle_identifier": resolve_info.get("CFBundleIdentifier"),
            "version": resolve_info.get("CFBundleShortVersionString"),
            "build": resolve_info.get("CFBundleVersion"),
        },
        "vendor_runtime": {
            "owner": "Blackmagic Design",
            "path": str(electron_app),
            "bundle_identifier": electron_info.get("CFBundleIdentifier"),
            "electron": vendor_version,
            "executable_sha256": _sha256_file(electron_binary),
            "signature": _codesign_evidence(electron_app),
            "app_transport_security": electron_info.get("NSAppTransportSecurity"),
            "declares_camera_usage": bool(electron_info.get("NSCameraUsageDescription")),
            "declares_microphone_usage": bool(electron_info.get("NSMicrophoneUsageDescription")),
            "supported_major_at_capture": vendor_major in supported_majors,
        },
        "controlled_runtime": {
            "owner": "HawaVoClean",
            "package_path": str(controlled_package),
            "electron": controlled_version,
            "supported_major_at_capture": int(controlled_version.split(".", 1)[0])
            in supported_majors,
        },
        "support_policy": {
            "source": "https://www.electronjs.org/docs/latest/tutorial/electron-timelines",
            "rule": "latest three stable major versions",
            "supported_majors_at_capture": sorted(supported_majors),
        },
        "advisory_source": {
            "provider": "GitHub Advisory Database API",
            "documentation": "https://docs.github.com/en/rest/security-advisories/global-advisories",
        },
        "vendor_advisories": vendor_scan,
        "controlled_advisories": controlled_scan,
        "assessment": {
            "dependency_estate_clean": vendor_scan["total"] == 0 and controlled_scan["total"] == 0,
            "vendor_high_or_critical_count": sum(
                vendor_scan["severity_counts"].get(severity, 0) for severity in ("high", "critical")
            ),
            "controlled_high_or_critical_count": sum(
                controlled_scan["severity_counts"].get(severity, 0)
                for severity in ("high", "critical")
            ),
            "requires_explicit_user_acceptance": bool(
                vendor_scan["severity_counts"].get("high", 0)
                or vendor_scan["severity_counts"].get("critical", 0)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve-app", type=Path, default=DEFAULT_RESOLVE_APP)
    parser.add_argument(
        "--controlled-package",
        type=Path,
        default=Path("resolve-plugin/com.hawavoclean.resolve/package.json"),
    )
    parser.add_argument(
        "--supported-majors",
        default="41,42,43",
        help="Electron majors supported on the capture date, from the official schedule",
    )
    args = parser.parse_args()
    try:
        supported = {int(value) for value in args.supported_majors.split(",") if value}
        report = capture(args.resolve_app.resolve(), args.controlled_package.resolve(), supported)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"resolve runtime audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
