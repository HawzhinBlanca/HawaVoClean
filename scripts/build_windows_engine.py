#!/usr/bin/env python3
"""Build a relocatable, version-bound Windows x64 standalone HawaVoClean engine directory."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE_IDENTITY_PATH = ROOT / "src" / "hawavoclean" / "release.json"
HEX40 = set("0123456789abcdef")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_LAUNCHER_NAME = "hawavoclean-engine.exe"
WINDOWS_CMD_WRAPPER_NAME = "hawavoclean-engine.cmd"
PE_MAGIC = b"MZ"

LAUNCHER_SOURCE = """from pathlib import Path
import multiprocessing
import os
import sys


def _run() -> int:
    engine_root = Path(__file__).resolve().parent
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    bin_dir = engine_root / "bin"
    if bin_dir.is_dir():
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}" if current_path else str(bin_dir)
    sys.path.insert(0, str(engine_root / "site-packages"))
    from hawavoclean.cli import main

    return main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(_run())
"""

CMD_WRAPPER_SOURCE = """@echo off
setlocal
set "ENGINE_DIR=%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
if exist "%ENGINE_DIR%bin" set "PATH=%ENGINE_DIR%bin;%PATH%"
"%ENGINE_DIR%python\\python.exe" -I -B "%ENGINE_DIR%launcher.py" %*
exit /b %ERRORLEVEL%
"""

# Minimal C source for compiling a native Win32 launcher stub (`hawavoclean-engine.exe`)
# when a Windows C compiler (cl.exe / clang / gcc) is available on the build host.
NATIVE_STUB_C_SOURCE = r"""#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>

static int append_quoted(wchar_t *buf, size_t cap, size_t *len, const wchar_t *arg) {
    if (*len + 3 >= cap) return 0;
    if (*len > 0) buf[(*len)++] = L' ';
    buf[(*len)++] = L'"';
    for (const wchar_t *p = arg; *p; ++p) {
        if (*p == L'"') {
            if (*len + 2 >= cap) return 0;
            buf[(*len)++] = L'\\';
        }
        if (*len + 2 >= cap) return 0;
        buf[(*len)++] = *p;
    }
    buf[(*len)++] = L'"';
    buf[*len] = L'\0';
    return 1;
}

int wmain(int argc, wchar_t **argv) {
    wchar_t exe_path[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, exe_path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return 1;
    wchar_t *slash = wcsrchr(exe_path, L'\\');
    if (!slash) return 1;
    *(slash + 1) = L'\0';

    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONDONTWRITEBYTECODE", L"1");

    wchar_t bin_dir[MAX_PATH];
    swprintf(bin_dir, MAX_PATH, L"%sbin", exe_path);
    wchar_t old_path[32768];
    DWORD old_len = GetEnvironmentVariableW(L"PATH", old_path, 32768);
    wchar_t new_path[32768];
    if (old_len > 0 && old_len < 32000) {
        swprintf(new_path, 32768, L"%s;%s", bin_dir, old_path);
    } else {
        swprintf(new_path, 32768, L"%s", bin_dir);
    }
    SetEnvironmentVariableW(L"PATH", new_path);

    wchar_t python_exe[MAX_PATH];
    wchar_t launcher_py[MAX_PATH];
    swprintf(python_exe, MAX_PATH, L"%spython\\python.exe", exe_path);
    swprintf(launcher_py, MAX_PATH, L"%slauncher.py", exe_path);

    wchar_t cmdline[32768];
    cmdline[0] = L'\0';
    size_t len = 0;
    if (!append_quoted(cmdline, 32768, &len, python_exe)) return 1;
    if (!append_quoted(cmdline, 32768, &len, L"-I")) return 1;
    if (!append_quoted(cmdline, 32768, &len, L"-B")) return 1;
    if (!append_quoted(cmdline, 32768, &len, launcher_py)) return 1;
    for (int i = 1; i < argc; ++i) {
        if (!append_quoted(cmdline, 32768, &len, argv[i])) return 1;
    }

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    if (!CreateProcessW(python_exe, cmdline, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)exit_code;
}
"""


class WindowsBundleError(RuntimeError):
    """The requested Windows engine bundle cannot be built or verified truthfully."""


def _run(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_identity(root: Path = ROOT) -> dict[str, Any]:
    """Load and validate canonical release identity metadata from src/hawavoclean/release.json."""
    identity_path = root / "src" / "hawavoclean" / "release.json"
    try:
        value: Any = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowsBundleError(f"cannot read release identity {identity_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WindowsBundleError("release.json must be a JSON object")
    version = value.get("version")
    product = value.get("product")
    if not isinstance(version, str) or re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise WindowsBundleError(f"release.json version is invalid: {version!r}")
    if not isinstance(product, str) or not product.strip():
        raise WindowsBundleError("release.json product is invalid")
    return value


def _wheel_provenance(wheel: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            value: Any = json.loads(archive.read("hawavoclean/build-provenance.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise WindowsBundleError(f"wheel has no readable build provenance: {exc}") from exc
    if not isinstance(value, dict) or value.get("artifact_type") != "wheel":
        raise WindowsBundleError("engine input must be a provenance-bearing wheel")
    revision = value.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40 or set(revision) - HEX40:
        raise WindowsBundleError("wheel source revision is invalid")
    if value.get("source_dirty") is not False:
        raise WindowsBundleError("engine input wheel claims dirty source")
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
                raise WindowsBundleError(f"bundle contains unsupported filesystem node: {path}")
    return sorted(files), sorted(links)


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
    """Remove build-only files and normalize installed dist-info RECORD entries."""
    for scripts_name in ("bin", "Scripts"):
        scripts = site_packages / scripts_name
        if scripts.exists():
            shutil.rmtree(scripts)
    base_site = stage / "python" / "Lib" / "site-packages"
    if base_site.exists():
        shutil.rmtree(base_site)

    for build_metadata in sorted(site_packages.glob("*.dist-info/uv_cache.json")):
        build_metadata.unlink()
    for direct_url in sorted(site_packages.glob("*.dist-info/direct_url.json")):
        direct_url.unlink()

    site_root = site_packages.resolve()
    for record in sorted(site_packages.glob("*.dist-info/RECORD")):
        rows: list[tuple[str, str, str]] = []
        with record.open(newline="", encoding="utf-8") as stream:
            original = list(csv.reader(stream))
        for row in original:
            if len(row) != 3:
                raise WindowsBundleError(f"invalid installed RECORD row in {record}")
            relative = row[0]
            target = (site_packages / relative).resolve()
            if not target.is_relative_to(site_root):
                raise WindowsBundleError(f"installed RECORD escapes site-packages: {relative}")
            if target == record.resolve() or not target.is_file():
                continue
            rows.append((relative, _record_digest(target), str(target.stat().st_size)))
        record_relative = record.relative_to(site_packages).as_posix()
        rows.append((record_relative, "", ""))
        with record.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerows(sorted(rows))


def _verify_pe_header(binary: Path) -> None:
    try:
        header = binary.read_bytes()[:2]
    except OSError as exc:
        raise WindowsBundleError(f"cannot read Windows binary {binary}: {exc}") from exc
    if header != PE_MAGIC:
        raise WindowsBundleError(
            f"Windows binary {binary.name} does not have a valid PE (MZ) header"
        )


def stage_windows_ffmpeg(
    bin_dir: Path,
    *,
    ffmpeg_dir: Path | None = None,
    allow_missing_ffmpeg: bool = False,
    verify_pe: bool = True,
) -> list[str]:
    """Copy ffmpeg.exe, ffprobe.exe, and adjacent DLLs into *bin_dir*."""
    staged: list[str] = []
    search_dirs: list[Path] = []
    if ffmpeg_dir is not None:
        if not ffmpeg_dir.is_dir():
            raise WindowsBundleError(f"specified --ffmpeg-dir is not a directory: {ffmpeg_dir}")
        search_dirs.append(ffmpeg_dir)

    copied_dlls: set[str] = set()
    for tool_name in ("ffmpeg.exe", "ffprobe.exe"):
        found: Path | None = None
        for directory in search_dirs:
            candidate = directory / tool_name
            if candidate.is_file():
                found = candidate
                break
        if found is None and ffmpeg_dir is None:
            on_path = shutil.which(tool_name)
            if on_path:
                found = Path(on_path)
        if found is None:
            if allow_missing_ffmpeg:
                continue
            raise WindowsBundleError(
                f"Required Windows media binary '{tool_name}' not found. "
                "Provide --ffmpeg-dir containing ffmpeg.exe and ffprobe.exe, or pass "
                "--allow-missing-ffmpeg for non-release dev builds."
            )
        if verify_pe:
            _verify_pe_header(found)
        bin_dir.mkdir(parents=True, exist_ok=True)
        dest = bin_dir / tool_name
        shutil.copy2(found.resolve(), dest)
        staged.append(tool_name)

        # Copy adjacent DLLs when bundling a shared-library Windows FFmpeg build.
        if ffmpeg_dir is not None:
            for dll in sorted(found.parent.glob("*.dll")):
                if dll.is_file() and dll.name not in copied_dlls:
                    if verify_pe:
                        _verify_pe_header(dll)
                    shutil.copy2(dll.resolve(), bin_dir / dll.name)
                    copied_dlls.add(dll.name)
    return staged


def _compile_native_windows_launcher(output_exe: Path) -> bool:
    """Attempt to compile NATIVE_STUB_C_SOURCE into a Win32 PE executable."""
    with tempfile.TemporaryDirectory(prefix="hawa-win-stub-") as tmp:
        src = Path(tmp) / "launcher_stub.c"
        src.write_text(NATIVE_STUB_C_SOURCE, encoding="utf-8")
        cl = shutil.which("cl.exe") or shutil.which("cl")
        if cl:
            result = subprocess.run(
                [cl, "/nologo", "/O2", "/W3", str(src), f"/Fe:{output_exe}"],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and output_exe.is_file():
                return True
        for compiler in ("clang.exe", "clang", "gcc.exe", "gcc"):
            cc = shutil.which(compiler)
            if not cc:
                continue
            result = subprocess.run(
                [cc, "-O2", "-municode", str(src), "-o", str(output_exe)],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and output_exe.is_file():
                return True
    return False


def write_windows_launchers(
    stage: Path,
    *,
    launcher_stub: Path | None = None,
    verify_pe: bool = True,
) -> None:
    """Write launcher.py, hawavoclean-engine.cmd, and hawavoclean-engine.exe into *stage*."""
    (stage / "launcher.py").write_text(LAUNCHER_SOURCE, encoding="utf-8")
    (stage / WINDOWS_CMD_WRAPPER_NAME).write_text(CMD_WRAPPER_SOURCE, encoding="utf-8")
    launcher_exe = stage / WINDOWS_LAUNCHER_NAME

    if launcher_stub is not None:
        if not launcher_stub.is_file():
            raise WindowsBundleError(f"Windows launcher stub not found: {launcher_stub}")
        if verify_pe:
            _verify_pe_header(launcher_stub)
        shutil.copy2(launcher_stub.resolve(), launcher_exe)
        return

    if platform.system() == "Windows" and _compile_native_windows_launcher(launcher_exe):
        if verify_pe:
            _verify_pe_header(launcher_exe)
        return

    raise WindowsBundleError(
        "Unable to produce hawavoclean-engine.exe: provide --launcher-stub pointing to a signed/built "
        "Win32 PE launcher stub or run on a Windows host with MSVC cl.exe / clang / mingw gcc."
    )


def build_manifest_dict(
    *,
    version: str,
    product: str,
    source_revision: str,
    source_date_epoch: int,
    dependency_lock_sha256: str,
    wheel_filename: str,
    wheel_sha256: str,
    python_version: str,
    installed_distributions: list[str],
    bundled_tools: list[str],
) -> dict[str, Any]:
    """Build the canonical ENGINE-MANIFEST.json payload for win32-x64."""
    return {
        "bundle_schema_version": 1,
        "artifact_type": "resolve-engine-directory",
        "product": product,
        "product_version": version,
        "version": version,
        "platform": "win32",
        "arch": "x64",
        "launcher": WINDOWS_LAUNCHER_NAME,
        "source_revision": source_revision,
        "source_date_epoch": source_date_epoch,
        "dependency_lock_sha256": dependency_lock_sha256,
        "wheel_filename": wheel_filename,
        "wheel_sha256": wheel_sha256,
        "python_version": python_version,
        "bundled_tools": sorted(bundled_tools),
        "installed_distributions": sorted(installed_distributions),
    }


def write_manifests(stage: Path, metadata_value: dict[str, Any]) -> None:
    """Write ENGINE-MANIFEST.json, ENGINE-SYMLINKS, and ENGINE-SHA256SUMS."""
    manifest_path = stage / "ENGINE-MANIFEST.json"
    manifest_path.write_text(
        json.dumps(metadata_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _, links = _iter_entries(stage)
    symlink_lines: list[str] = []
    for path in links:
        rel = "./" + path.relative_to(stage).as_posix()
        target = os.readlink(path)
        if target.startswith("/") or ".." in Path(target).parts:
            raise WindowsBundleError(f"bundle symlink escapes its root: {rel} -> {target}")
        symlink_lines.append(f"{rel}\t{target}\n")
    (stage / "ENGINE-SYMLINKS").write_text("".join(symlink_lines), encoding="utf-8")

    files, _ = _iter_entries(stage)
    checksum_path = stage / "ENGINE-SHA256SUMS"
    lines = [
        f"{_sha256(path)}  ./{path.relative_to(stage).as_posix()}\n"
        for path in files
        if path != checksum_path
    ]
    checksum_path.write_text("".join(lines), encoding="utf-8")


def _safe_relative_path(raw: str) -> Path:
    if "\x00" in raw or "\\" in raw:
        raise WindowsBundleError(f"unsafe checksum path: {raw!r}")
    if not raw.startswith("./"):
        raise WindowsBundleError(f"checksum path must start with './': {raw!r}")
    stripped = raw.removeprefix("./")
    rel = Path(stripped)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or "." in rel.parts:
        raise WindowsBundleError(f"checksum path escapes bundle root: {raw!r}")
    return rel


def verify_staged_bundle(
    stage: Path,
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Verify manifest contract, placeholder absence, launcher presence, and exhaustive SHA-256 sums."""
    if stage.is_symlink() or not stage.is_dir():
        raise WindowsBundleError(f"staged engine path is not a real directory: {stage}")
    placeholder = stage / "README.txt"
    if placeholder.exists():
        raise WindowsBundleError(
            f"archive validation rejected placeholder engine resource: {placeholder}"
        )

    launcher = stage / WINDOWS_LAUNCHER_NAME
    manifest_path = stage / "ENGINE-MANIFEST.json"
    checksums_path = stage / "ENGINE-SHA256SUMS"
    symlinks_path = stage / "ENGINE-SYMLINKS"
    for path, label in (
        (launcher, "Windows engine launcher"),
        (manifest_path, "ENGINE-MANIFEST.json"),
        (checksums_path, "ENGINE-SHA256SUMS"),
        (symlinks_path, "ENGINE-SYMLINKS"),
    ):
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise WindowsBundleError(f"missing {label}: {path}") from exc
        if not stat.S_ISREG(mode):
            raise WindowsBundleError(f"{label} must be a regular file: {path}")

    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowsBundleError(f"ENGINE-MANIFEST.json is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise WindowsBundleError("ENGINE-MANIFEST.json must be a JSON object")

    canonical_version = expected_version or load_release_identity()["version"]
    if (
        manifest.get("bundle_schema_version") != 1
        or manifest.get("artifact_type") != "resolve-engine-directory"
        or manifest.get("platform") != "win32"
        or manifest.get("arch") != "x64"
        or manifest.get("launcher") != WINDOWS_LAUNCHER_NAME
        or manifest.get("version") != canonical_version
        or manifest.get("product_version") != canonical_version
    ):
        raise WindowsBundleError(
            "ENGINE-MANIFEST.json does not conform to the win32-x64 HawaVoClean engine contract"
        )

    lines = [
        line for line in checksums_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise WindowsBundleError("ENGINE-SHA256SUMS is empty")

    stage_root = stage.resolve()
    verified: dict[Path, str] = {}
    for line in lines:
        digest, sep, raw_rel = line.partition("  ")
        if sep != "  " or HEX64_RE.fullmatch(digest.lower()) is None:
            raise WindowsBundleError(f"malformed ENGINE-SHA256SUMS line: {line!r}")
        rel = _safe_relative_path(raw_rel)
        if rel in verified:
            raise WindowsBundleError(f"duplicate ENGINE-SHA256SUMS entry: {rel}")
        target = (stage / rel).resolve()
        if not target.is_relative_to(stage_root):
            raise WindowsBundleError(f"checksummed file escapes bundle root: {rel}")
        try:
            mode = (stage / rel).lstat().st_mode
        except OSError as exc:
            raise WindowsBundleError(f"checksummed file is missing: {rel}") from exc
        if not stat.S_ISREG(mode):
            raise WindowsBundleError(f"checksummed entry is not a regular file: {rel}")
        actual = _sha256(stage / rel)
        if actual != digest.lower():
            raise WindowsBundleError(
                f"checksum mismatch for {rel}: expected {digest.lower()}, got {actual}"
            )
        verified[rel] = actual

    actual_files = {
        path.relative_to(stage)
        for path in stage.rglob("*")
        if path.is_file() and not path.is_symlink() and path != checksums_path
    }
    if set(verified) != actual_files:
        missing = sorted(str(p) for p in (actual_files - set(verified)))
        extra = sorted(str(p) for p in (set(verified) - actual_files))
        raise WindowsBundleError(
            f"ENGINE-SHA256SUMS inventory drift (unlisted={missing}, missing={extra})"
        )

    return {
        "verified_files": len(verified),
        "manifest_sha256": _sha256(manifest_path),
        "launcher_sha256": _sha256(launcher),
        "version": canonical_version,
        "platform": "win32",
        "arch": "x64",
    }


def build_bundle(
    wheel: Path,
    output: Path,
    python_spec: str,
    *,
    ffmpeg_dir: Path | None = None,
    launcher_stub: Path | None = None,
    allow_missing_ffmpeg: bool = False,
) -> dict[str, Any]:
    """Build and verify a self-contained Windows x64 standalone engine directory."""
    wheel = wheel.resolve()
    output = output.resolve()
    if output.exists():
        raise WindowsBundleError(f"output already exists: {output}")
    if not wheel.is_file():
        raise WindowsBundleError(f"wheel not found: {wheel}")
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise WindowsBundleError(
            "Windows win32-x64 engine bundles must be built on a native Windows x64 host"
        )

    identity = load_release_identity()
    release_version = str(identity["version"])
    product = str(identity["product"])

    provenance = _wheel_provenance(wheel)
    revision = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=normal"])
    if dirty:
        raise WindowsBundleError("refusing to package a Windows engine from a dirty source tree")
    if provenance["source_revision"] != revision:
        raise WindowsBundleError("wheel source revision does not match the clean checkout")
    lock_sha = _sha256(ROOT / "uv.lock")
    if provenance.get("dependency_lock_sha256") != lock_sha:
        raise WindowsBundleError("wheel dependency lock digest does not match uv.lock")

    managed_python = Path(_run(["uv", "python", "find", "--managed-python", python_spec])).resolve()
    prefix = managed_python.parent
    python_version = _run(
        [str(managed_python), "-c", "import platform;print(platform.python_version())"]
    )
    if not python_version.startswith("3.11."):
        raise WindowsBundleError(
            f"Windows engine runtime must be CPython 3.11, got {python_version}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    try:
        shutil.copytree(prefix, stage / "python", symlinks=False)
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
        stage_python = stage / "python" / "python.exe"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(stage_python),
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
                str(stage_python),
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

        shutil.copy2(ROOT / "THIRD_PARTY_LICENSES.md", stage / "THIRD_PARTY_LICENSES.md")

        bundled_tools = stage_windows_ffmpeg(
            stage / "bin",
            ffmpeg_dir=ffmpeg_dir,
            allow_missing_ffmpeg=allow_missing_ffmpeg,
        )
        write_windows_launchers(stage, launcher_stub=launcher_stub)

        launcher = stage / WINDOWS_LAUNCHER_NAME
        version_output = _run([str(launcher), "--version"], cwd=stage)
        if version_output != f"{product} {release_version}":
            raise WindowsBundleError(
                f"bundled Windows engine reported unexpected version: {version_output}"
            )
        _remove_bytecode(stage)

        distributions = sorted(
            {
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions(path=[str(site_packages)])
                if dist.metadata.get("Name")
            }
        )
        manifest_data = build_manifest_dict(
            version=release_version,
            product=product,
            source_revision=revision,
            source_date_epoch=int(provenance["source_date_epoch"]),
            dependency_lock_sha256=lock_sha,
            wheel_filename=wheel.name,
            wheel_sha256=_sha256(wheel),
            python_version=python_version,
            installed_distributions=distributions,
            bundled_tools=bundled_tools,
        )
        write_manifests(stage, manifest_data)
        verification = verify_staged_bundle(stage, expected_version=release_version)

        epoch = int(provenance["source_date_epoch"])
        for path in sorted(stage.rglob("*"), reverse=True):
            if not path.is_symlink():
                os.utime(path, (epoch, epoch), follow_symlinks=False)
        os.utime(stage, (epoch, epoch), follow_symlinks=False)
        stage.rename(output)
        return verification
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="3.11")
    parser.add_argument(
        "--ffmpeg-dir",
        type=Path,
        default=None,
        help="Directory containing pinned ffmpeg.exe, ffprobe.exe, and optional FFmpeg DLLs.",
    )
    parser.add_argument(
        "--launcher-stub",
        type=Path,
        default=None,
        help="Optional prebuilt Win32 PE executable stub to install as hawavoclean-engine.exe.",
    )
    parser.add_argument(
        "--allow-missing-ffmpeg",
        action="store_true",
        help="Allow building without ffmpeg.exe/ffprobe.exe (non-release dev builds only).",
    )
    args = parser.parse_args()
    try:
        build_bundle(
            args.wheel,
            args.output,
            args.python,
            ffmpeg_dir=args.ffmpeg_dir,
            launcher_stub=args.launcher_stub,
            allow_missing_ffmpeg=args.allow_missing_ffmpeg,
        )
    except (WindowsBundleError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
