#!/usr/bin/env python3
"""Validate an unsigned, source-bound macOS desktop packaging proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from scripts import generate_sbom  # noqa: E402

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROVENANCE_NAME = "HAWAVOCLEAN-PACKAGE-PROVENANCE.json"
MAX_ASAR_HEADER_BYTES = 128 * 1024 * 1024


class DesktopProofError(RuntimeError):
    """The packaged app is incomplete, mislabelled, or not integrity-bound."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DesktopProofError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _asar_header_sha256(path: Path) -> str:
    """Recompute Electron Builder's SHA-256 over the ASAR header JSON UTF-8 bytes.

    ASAR stores an eight-byte Chromium Pickle containing the byte size of a
    second Pickle.  The second Pickle contains a length-prefixed JSON string.
    Electron Builder hashes that decoded string as UTF-8, not the complete
    archive and not the Pickle framing.  Parse the framing strictly so random
    bytes cannot be accepted merely because Info.plist contains a hash-shaped
    value.
    """

    try:
        archive_size = path.stat().st_size
        with path.open("rb") as stream:
            size_pickle = stream.read(8)
            if len(size_pickle) != 8:
                raise DesktopProofError("desktop ASAR omits its Chromium Pickle size header")
            size_payload_bytes, header_pickle_size = struct.unpack("<II", size_pickle)
            if size_payload_bytes != 4:
                raise DesktopProofError("desktop ASAR size Pickle has an invalid payload size")
            if (
                header_pickle_size < 8
                or header_pickle_size > MAX_ASAR_HEADER_BYTES
                or 8 + header_pickle_size > archive_size
            ):
                raise DesktopProofError("desktop ASAR header Pickle size is invalid")
            header_pickle = stream.read(header_pickle_size)
    except DesktopProofError:
        raise
    except OSError as exc:
        raise DesktopProofError(f"cannot read desktop ASAR header: {exc}") from exc

    if len(header_pickle) != header_pickle_size:
        raise DesktopProofError("desktop ASAR header Pickle is truncated")
    payload_size = struct.unpack_from("<I", header_pickle, 0)[0]
    if payload_size != header_pickle_size - 4 or payload_size < 4:
        raise DesktopProofError("desktop ASAR header Pickle framing is invalid")
    json_size = struct.unpack_from("<i", header_pickle, 4)[0]
    if json_size < 0:
        raise DesktopProofError("desktop ASAR header JSON length is negative")
    aligned_json_size = (json_size + 3) & ~3
    if payload_size != 4 + aligned_json_size:
        raise DesktopProofError("desktop ASAR header JSON length differs from its Pickle payload")
    json_start = 8
    json_end = json_start + json_size
    padding_end = json_start + aligned_json_size
    json_bytes = header_pickle[json_start:json_end]
    if len(json_bytes) != json_size or any(header_pickle[json_end:padding_end]):
        raise DesktopProofError("desktop ASAR header JSON or alignment padding is invalid")
    try:
        header_text = json_bytes.decode("utf-8", errors="strict")
        header: Any = json.loads(header_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DesktopProofError(f"desktop ASAR header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict) or not isinstance(header.get("files"), dict):
        raise DesktopProofError("desktop ASAR header does not contain a files object")
    if header_text.encode("utf-8") != json_bytes:
        raise DesktopProofError("desktop ASAR header does not round-trip as exact UTF-8")
    return hashlib.sha256(json_bytes).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DesktopProofError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise DesktopProofError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_regular(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        details = path.lstat()
    except OSError as exc:
        raise DesktopProofError(f"{label} is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise DesktopProofError(f"{label} is not a real regular file: {path}")
    if executable and details.st_mode & 0o111 == 0:
        raise DesktopProofError(f"{label} is not executable: {path}")


def _safe_relative(value: str, label: str) -> Path:
    relative = Path(value.removeprefix("./"))
    if (
        not value.startswith("./")
        or relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
    ):
        raise DesktopProofError(f"{label} contains an unsafe path: {value!r}")
    return relative


def _verify_engine(engine: Path, source_revision: str) -> dict[str, Any]:
    if engine.is_symlink() or not engine.is_dir():
        raise DesktopProofError("desktop engine payload is not a real directory")
    placeholder = engine / "README.txt"
    if placeholder.exists():
        raise DesktopProofError(
            "archive validation failed: placeholder engine resource detected (README.txt)"
        )
    launcher = engine / "hawavoclean-engine"
    manifest_path = engine / "ENGINE-MANIFEST.json"
    checksum_path = engine / "ENGINE-SHA256SUMS"
    symlink_path = engine / "ENGINE-SYMLINKS"
    for path, label in (
        (launcher, "desktop engine launcher"),
        (manifest_path, "desktop engine manifest"),
        (checksum_path, "desktop engine checksum inventory"),
        (symlink_path, "desktop engine symlink inventory"),
    ):
        _require_regular(path, label, executable=path == launcher)

    manifest = _load_json(manifest_path, "desktop engine manifest")
    if (
        manifest.get("bundle_schema_version") != 1
        or manifest.get("artifact_type") != "resolve-engine-directory"
        or manifest.get("product_version") != "3.3.0"
        or manifest.get("source_revision") != source_revision
        or manifest.get("platform") != "macos-arm64"
    ):
        raise DesktopProofError(
            "desktop engine manifest differs from the qualified payload contract"
        )

    checksums: dict[Path, str] = {}
    try:
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DesktopProofError(f"cannot read engine checksums: {exc}") from exc
    for line in checksum_lines:
        digest, separator, raw_relative = line.partition("  ")
        if separator != "  " or HEX64.fullmatch(digest) is None:
            raise DesktopProofError("desktop engine checksum inventory is malformed")
        relative = _safe_relative(raw_relative, "desktop engine checksum inventory")
        if relative in checksums:
            raise DesktopProofError(f"duplicate desktop engine checksum path: {relative}")
        target = engine / relative
        _require_regular(target, f"checksummed desktop engine file {relative}")
        if _sha256(target) != digest:
            raise DesktopProofError(f"desktop engine checksum differs: {relative}")
        checksums[relative] = digest

    actual_files = {
        path.relative_to(engine)
        for path in engine.rglob("*")
        if path.is_file() and not path.is_symlink() and path != checksum_path
    }
    if set(checksums) != actual_files:
        raise DesktopProofError("desktop engine regular-file inventory differs from its checksums")

    declared_links: dict[Path, str] = {}
    try:
        symlink_lines = symlink_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DesktopProofError(f"cannot read engine symlink inventory: {exc}") from exc
    for line in symlink_lines:
        raw_relative, separator, link_target = line.partition("\t")
        if (
            separator != "\t"
            or not link_target
            or Path(link_target).is_absolute()
            or ".." in Path(link_target).parts
        ):
            raise DesktopProofError("desktop engine symlink inventory is malformed")
        relative = _safe_relative(raw_relative, "desktop engine symlink inventory")
        if relative in declared_links:
            raise DesktopProofError(f"duplicate desktop engine symlink path: {relative}")
        link = engine / relative
        if not link.is_symlink() or os.readlink(link) != link_target:
            raise DesktopProofError(f"desktop engine symlink differs: {relative}")
        declared_links[relative] = link_target
    actual_links = {path.relative_to(engine) for path in engine.rglob("*") if path.is_symlink()}
    if set(declared_links) != actual_links:
        raise DesktopProofError("desktop engine symlink inventory is incomplete")

    return {
        "manifest_sha256": _sha256(manifest_path),
        "regular_files": len(actual_files),
        "symlinks": len(actual_links),
        "launcher_sha256": _sha256(launcher),
    }


def _signing_classification(app: Path) -> dict[str, Any]:
    verification = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", os.fspath(app)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if verification.returncode != 0:
        detail = (verification.stderr or verification.stdout)[-1000:]
        raise DesktopProofError(f"desktop proof ad-hoc signature is invalid: {detail}")
    completed = subprocess.run(
        ["codesign", "-d", "--verbose=4", os.fspath(app)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    detail = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise DesktopProofError("codesign could not classify the unsigned desktop proof")
    if "TeamIdentifier=not set" not in detail or "Authority=" in detail:
        raise DesktopProofError(
            "desktop proof unexpectedly carries a distribution signing identity"
        )
    if "Signature=adhoc" not in detail and "flags=0x20002(adhoc,linker-signed)" not in detail:
        raise DesktopProofError("desktop proof is not explicitly ad-hoc/unsigned")
    return {"developer_id": False, "team_identifier": None, "classification": "adhoc_unsigned"}


def validate_app(app: Path, source_revision: str, *, engine_mode: str) -> dict[str, Any]:
    if HEX40.fullmatch(source_revision) is None:
        raise DesktopProofError("source revision must be a lowercase full Git SHA")
    if app.is_symlink() or not app.is_dir() or app.name != "HawaVoClean.app":
        raise DesktopProofError("desktop proof must be the real HawaVoClean.app directory")
    app = app.resolve()
    contents = app / "Contents"
    resources = contents / "Resources"
    executable = contents / "MacOS" / "HawaVoClean"
    app_asar = resources / "app.asar"
    ui_index = resources / "ui" / "index.html"
    provenance_path = resources / PROVENANCE_NAME
    info_path = contents / "Info.plist"
    icon_path = resources / "icon.icns"
    for path, label in (
        (executable, "desktop app executable"),
        (app_asar, "desktop app ASAR"),
        (ui_index, "desktop UI entrypoint"),
        (provenance_path, "desktop package provenance"),
        (info_path, "desktop Info.plist"),
        (icon_path, "desktop branded icon"),
    ):
        _require_regular(path, label, executable=path == executable)

    if icon_path.stat().st_size < 10000:
        raise DesktopProofError("desktop branded icon is too small or corrupt")

    try:
        with info_path.open("rb") as stream:
            info: Any = plistlib.load(stream)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise DesktopProofError(f"cannot read desktop Info.plist: {exc}") from exc
    if not isinstance(info, dict):
        raise DesktopProofError("desktop Info.plist is not a dictionary")
    if (
        info.get("CFBundleIdentifier") != "com.hawavoclean.desktop"
        or info.get("CFBundleShortVersionString") != "3.3.0"
        or info.get("LSMinimumSystemVersion") != "14.0.0"
        or info.get("CFBundleExecutable") != "HawaVoClean"
    ):
        raise DesktopProofError("desktop Info.plist differs from the product contract")
    icon_file = info.get("CFBundleIconFile")
    if icon_file == "electron.icns":
        raise DesktopProofError("desktop Info.plist retains stock/ad-hoc electron.icns identity")
    if icon_file not in ("icon.icns", "icon"):
        raise DesktopProofError(
            f"desktop Info.plist does not declare branded CFBundleIconFile: {icon_file!r}"
        )
    asar_integrity = info.get("ElectronAsarIntegrity")
    if not isinstance(asar_integrity, dict):
        raise DesktopProofError("desktop Info.plist omits Electron ASAR integrity")
    integrity_item = asar_integrity.get("Resources/app.asar")
    if (
        not isinstance(integrity_item, dict)
        or integrity_item.get("algorithm") != "SHA256"
        or HEX64.fullmatch(str(integrity_item.get("hash"))) is None
    ):
        raise DesktopProofError("desktop ASAR integrity metadata is malformed")
    asar_header_sha256 = _asar_header_sha256(app_asar)
    if integrity_item.get("hash") != asar_header_sha256:
        raise DesktopProofError("desktop ASAR header hash differs from ElectronAsarIntegrity")

    provenance = _load_json(provenance_path, "desktop package provenance")
    expected_signing = {"developer_id": False, "notarized": False, "stapled": False}
    if (
        provenance.get("schema_version") != 1
        or provenance.get("artifact_type") != "unsigned-macos-app-proof"
        or provenance.get("distribution_eligible") is not False
        or provenance.get("product") != "hawavoclean"
        or provenance.get("product_version") != "3.3.0"
        or provenance.get("source_revision") != source_revision
        or provenance.get("target") != "macos-arm64"
        or provenance.get("engine_mode") != engine_mode
        or provenance.get("packaged_selftest_allowed") is not (engine_mode == "full")
        or provenance.get("signing") != expected_signing
    ):
        raise DesktopProofError("desktop package provenance differs from the proof contract")

    engine_root = resources / "engine"
    if (engine_root / "README.txt").exists() and engine_mode == "full":
        raise DesktopProofError(
            "archive validation failed: placeholder engine resource detected (README.txt)"
        )
    if engine_mode == "full":
        engine = _verify_engine(engine_root, source_revision)
        if provenance.get("engine_manifest_sha256") != engine["manifest_sha256"]:
            raise DesktopProofError("desktop package provenance does not bind its engine manifest")
    elif engine_mode == "shell-only":
        _require_regular(engine_root / "README.txt", "desktop shell-only engine marker")
        if (engine_root / "hawavoclean-engine").exists():
            raise DesktopProofError("shell-only desktop proof unexpectedly contains an engine")
        if provenance.get("engine_manifest_sha256") is not None:
            raise DesktopProofError("shell-only desktop proof claims a full engine manifest")
        engine = {"status": "intentionally_absent"}
    else:
        raise DesktopProofError(f"unsupported desktop engine mode: {engine_mode}")

    component = generate_sbom._artifact_component("desktop-app", app)
    hashes = component.get("hashes")
    properties = component.get("properties")
    if not isinstance(hashes, list) or not isinstance(properties, list):
        raise DesktopProofError("desktop app has no canonical tree identity")
    tree_sha256 = next(
        (
            item.get("content")
            for item in hashes
            if isinstance(item, dict) and item.get("alg") == "SHA-256"
        ),
        None,
    )
    if not isinstance(tree_sha256, str) or HEX64.fullmatch(tree_sha256) is None:
        raise DesktopProofError("desktop app tree identity is malformed")
    details = {
        str(item["name"]): str(item["value"])
        for item in properties
        if isinstance(item, dict) and "name" in item and "value" in item
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "artifact_type": "unsigned-macos-app-proof",
        "distribution_eligible": False,
        "source_revision": source_revision,
        "engine_mode": engine_mode,
        "tree_sha256": tree_sha256,
        "tree_regular_files": int(details["hawavoclean:artifact-tree-regular-files"]),
        "tree_symlinks": int(details["hawavoclean:artifact-tree-symlink-count"]),
        "app_asar_sha256": _sha256(app_asar),
        "app_asar_header_sha256": asar_header_sha256,
        "engine": engine,
        "signing": _signing_classification(app),
    }


def validate_archive(
    target: Path,
    source_revision: str,
    *,
    engine_mode: str = "full",
) -> dict[str, Any]:
    """Validate a .dmg, .zip, or .app directory package against the True-10 sealed contract."""
    target = target.resolve()
    if not target.exists():
        raise DesktopProofError(f"archive target does not exist: {target}")

    if target.is_file():
        if target.suffix == ".zip":
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                try:
                    with zipfile.ZipFile(target) as zf:
                        for info in zf.infolist():
                            extracted = temp_path / info.filename
                            mode = info.external_attr >> 16
                            if mode & 0o170000 == 0o120000:
                                link_target = zf.read(info).decode("utf-8")
                                extracted.parent.mkdir(parents=True, exist_ok=True)
                                if extracted.is_symlink() or extracted.exists():
                                    extracted.unlink()
                                extracted.symlink_to(link_target)
                            else:
                                zf.extract(info, temp_path)
                                if mode:
                                    os.chmod(extracted, mode)
                except (OSError, zipfile.BadZipFile) as exc:
                    raise DesktopProofError(f"cannot extract ZIP archive: {exc}") from exc
                apps = list(temp_path.glob("**/HawaVoClean.app"))
                if not apps:
                    raise DesktopProofError("ZIP archive does not contain HawaVoClean.app")
                return validate_app(apps[0], source_revision, engine_mode=engine_mode)
        elif target.suffix == ".dmg":
            if sys.platform != "darwin":
                raise DesktopProofError("DMG archive validation requires a macOS host")
            with tempfile.TemporaryDirectory() as mount_dir:
                mount_path = Path(mount_dir)
                attach_cmd = [
                    "hdiutil",
                    "attach",
                    "-nobrowse",
                    "-readonly",
                    "-noverify",
                    os.fspath(target),
                    "-mountpoint",
                    os.fspath(mount_path),
                ]
                proc = subprocess.run(attach_cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    raise DesktopProofError(f"hdiutil attach failed: {proc.stderr or proc.stdout}")
                try:
                    apps = list(mount_path.glob("**/HawaVoClean.app"))
                    if not apps:
                        raise DesktopProofError("DMG archive does not contain HawaVoClean.app")
                    return validate_app(apps[0], source_revision, engine_mode=engine_mode)
                finally:
                    eject = subprocess.run(
                        ["diskutil", "eject", os.fspath(mount_path)],
                        capture_output=True,
                        check=False,
                    )
                    if eject.returncode != 0:
                        subprocess.run(
                            ["hdiutil", "detach", os.fspath(mount_path), "-force"],
                            capture_output=True,
                            check=False,
                        )
        else:
            raise DesktopProofError(f"unsupported archive format: {target.suffix}")
    elif target.is_dir():
        if target.name == "HawaVoClean.app":
            return validate_app(target, source_revision, engine_mode=engine_mode)
        apps = list(target.glob("**/HawaVoClean.app"))
        if not apps:
            raise DesktopProofError(f"directory does not contain HawaVoClean.app: {target}")
        return validate_app(apps[0], source_revision, engine_mode=engine_mode)
    raise DesktopProofError(f"unsupported archive target: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--source-sha", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--require-engine", action="store_true")
    mode.add_argument("--shell-only", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        result = validate_archive(
            args.app,
            args.source_sha,
            engine_mode="full" if args.require_engine else "shell-only",
        )
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(encoded, encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
    except (
        DesktopProofError,
        OSError,
        generate_sbom.SbomError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"desktop app proof validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
