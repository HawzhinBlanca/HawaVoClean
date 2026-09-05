#!/usr/bin/env python3
"""Build a relocatable, version-bound macOS Resolve engine directory."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HEX40 = set("0123456789abcdef")
LAUNCHER_SOURCE = """from pathlib import Path
import multiprocessing
import os
import sys


def _run() -> int:
    engine_root = Path(__file__).resolve().parent
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(engine_root / "site-packages"))
    from hawavoclean.cli import main

    return main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(_run())
"""
SHELL_LAUNCHER_SOURCE = """#!/bin/sh
set -eu
engine_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if [ -d "$engine_dir/bin" ]; then
    export PATH="$engine_dir/bin:$PATH"
fi
exec "$engine_dir/python/bin/python3.11" -I -B "$engine_dir/launcher.py" "$@"
"""


class BundleError(RuntimeError):
    """The requested bundle cannot be built truthfully."""


def _run(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wheel_provenance(wheel: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            value: Any = json.loads(archive.read("hawavoclean/build-provenance.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise BundleError(f"wheel has no readable build provenance: {exc}") from exc
    if not isinstance(value, dict) or value.get("artifact_type") != "wheel":
        raise BundleError("engine input must be a provenance-bearing wheel")
    revision = value.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40 or set(revision) - HEX40:
        raise BundleError("wheel source revision is invalid")
    if value.get("source_dirty") is not False:
        raise BundleError("engine input wheel claims dirty source")
    return value


def _iter_entries(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    links: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in sorted((*dirnames, *filenames)):
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                links.append(path)
            elif stat.S_ISREG(mode):
                files.append(path)
            elif not stat.S_ISDIR(mode):
                raise BundleError(f"bundle contains unsupported filesystem node: {path}")
    return sorted(files), sorted(links)


def _write_manifests(stage: Path, metadata_value: dict[str, Any]) -> None:
    manifest_path = stage / "ENGINE-MANIFEST.json"
    manifest_path.write_text(json.dumps(metadata_value, indent=2, sort_keys=True) + "\n")
    _, links = _iter_entries(stage)
    symlink_lines: list[str] = []
    for path in links:
        rel = "./" + path.relative_to(stage).as_posix()
        target = os.readlink(path)
        if target.startswith("/") or ".." in Path(target).parts:
            raise BundleError(f"bundle symlink escapes its root: {rel} -> {target}")
        symlink_lines.append(f"{rel}\t{target}\n")
    (stage / "ENGINE-SYMLINKS").write_text("".join(symlink_lines))

    files, _ = _iter_entries(stage)
    checksum_path = stage / "ENGINE-SHA256SUMS"
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(stage).as_posix()}\n"
        for path in files
        if path != checksum_path
    ]
    checksum_path.write_text("".join(lines))


def _remove_bytecode(root: Path) -> None:
    for directory in sorted(root.rglob("__pycache__"), reverse=True):
        if directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)
    for bytecode in root.rglob("*.py[co]"):
        bytecode.unlink()


def _record_digest(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _prune_and_normalize_install(stage: Path, site_packages: Path) -> None:
    """Remove build-only files and make every installed RECORD truthful."""
    scripts = site_packages / "bin"
    if scripts.exists():
        shutil.rmtree(scripts)
    base_site = stage / "python" / "lib" / "python3.11" / "site-packages"
    if base_site.exists():
        shutil.rmtree(base_site)

    for build_metadata in sorted(site_packages.glob("*.dist-info/uv_cache.json")):
        build_metadata.unlink()
    for direct_url in sorted(site_packages.glob("*.dist-info/direct_url.json")):
        direct_url.unlink()

    site_root = site_packages.resolve()
    for record in sorted(site_packages.glob("*.dist-info/RECORD")):
        rows: list[tuple[str, str, str]] = []
        with record.open(newline="") as stream:
            original = list(csv.reader(stream))
        for row in original:
            if len(row) != 3:
                raise BundleError(f"invalid installed RECORD row in {record}")
            relative = row[0]
            target = (site_packages / relative).resolve()
            if not target.is_relative_to(site_root):
                raise BundleError(f"installed RECORD escapes site-packages: {relative}")
            if target == record.resolve() or not target.is_file():
                continue
            rows.append((relative, _record_digest(target), str(target.stat().st_size)))
        record_relative = record.relative_to(site_packages).as_posix()
        rows.append((record_relative, "", ""))
        with record.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerows(sorted(rows))


def _write_launchers(stage: Path) -> None:
    (stage / "launcher.py").write_text(LAUNCHER_SOURCE)
    launcher = stage / "hawavoclean-engine"
    launcher.write_text(SHELL_LAUNCHER_SOURCE)
    launcher.chmod(0o755)


def build_bundle(wheel: Path, output: Path, python_spec: str) -> None:
    wheel = wheel.resolve()
    output = output.resolve()
    if output.exists():
        raise BundleError(f"output already exists: {output}")
    if not wheel.is_file():
        raise BundleError(f"wheel not found: {wheel}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise BundleError("Resolve v3.3 engine bundles are qualified only on Apple silicon macOS")

    provenance = _wheel_provenance(wheel)
    revision = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=normal"])
    if dirty:
        raise BundleError("refusing to package a Resolve engine from a dirty source tree")
    if provenance["source_revision"] != revision:
        raise BundleError("wheel source revision does not match the clean checkout")
    lock_sha = _sha256(ROOT / "uv.lock")
    if provenance.get("dependency_lock_sha256") != lock_sha:
        raise BundleError("wheel dependency lock digest does not match uv.lock")

    managed_python = Path(_run(["uv", "python", "find", "--managed-python", python_spec])).resolve()
    prefix = managed_python.parent.parent
    python_version = _run(
        [str(managed_python), "-c", "import platform;print(platform.python_version())"]
    )
    if not python_version.startswith("3.11."):
        raise BundleError(f"Resolve engine runtime must be CPython 3.11, got {python_version}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    try:
        shutil.copytree(prefix, stage / "python", symlinks=True)
        site_packages = stage / "site-packages"
        site_packages.mkdir()
        requirements = stage / ".requirements.txt"
        subprocess.run(
            [
                "uv",
                "export",
                "--quiet",
                "--frozen",
                "--extra",
                "studio",
                "--extra",
                "ui",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(requirements),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(stage / "python" / "bin" / "python3.11"),
                "--target",
                str(site_packages),
                "--requirements",
                str(requirements),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(stage / "python" / "bin" / "python3.11"),
                "--target",
                str(site_packages),
                "--no-deps",
                str(wheel),
            ],
            cwd=ROOT,
            check=True,
        )
        requirements.unlink()
        _prune_and_normalize_install(stage, site_packages)
        _remove_bytecode(stage)

        # Carry the product's generated third-party inventory inside the
        # self-contained runtime. It is covered by ENGINE-SHA256SUMS below.
        shutil.copy2(ROOT / "THIRD_PARTY_LICENSES.md", stage / "THIRD_PARTY_LICENSES.md")

        # Bundle pinned FFmpeg and ffprobe binaries into engine bin/
        bin_dir = stage / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for tool_name in ("ffmpeg", "ffprobe"):
            tool_path = shutil.which(tool_name)
            if tool_path:
                dest_tool = bin_dir / tool_name
                shutil.copy2(tool_path, dest_tool)
                dest_tool.chmod(0o755)

        _write_launchers(stage)
        launcher = stage / "hawavoclean-engine"

        version_output = _run([str(launcher), "--version"], cwd=stage)
        if version_output != "hawavoclean 3.3.0":
            raise BundleError(f"bundled engine reported unexpected version: {version_output}")
        _remove_bytecode(stage)
        distributions = sorted(
            {
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions(path=[str(site_packages)])
                if dist.metadata.get("Name")
            }
        )
        metadata_value = {
            "bundle_schema_version": 1,
            "artifact_type": "resolve-engine-directory",
            "product_version": "3.3.0",
            "source_revision": revision,
            "source_date_epoch": provenance["source_date_epoch"],
            "dependency_lock_sha256": lock_sha,
            "wheel_filename": wheel.name,
            "wheel_sha256": _sha256(wheel),
            "python_version": python_version,
            "platform": "macos-arm64",
            "installed_distributions": distributions,
        }
        _write_manifests(stage, metadata_value)

        epoch = int(provenance["source_date_epoch"])
        for path in sorted(stage.rglob("*"), reverse=True):
            if not path.is_symlink():
                os.utime(path, (epoch, epoch), follow_symlinks=False)
        os.utime(stage, (epoch, epoch), follow_symlinks=False)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="3.11")
    args = parser.parse_args()
    try:
        build_bundle(args.wheel, args.output, args.python)
    except (BundleError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
