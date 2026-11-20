"""Request-side path policy for the engine API.

Every path a client sends (analyze, jobs, audio) must be absolute and must
resolve under the user's home directory, ``/Volumes`` (mounted media), the
HawaVoClean work directory, or (on Windows) user-accessible local secondary
fixed/removable drive roots (e.g. ``D:\\``, ``E:\\``) outside Windows system
directories (``Windows``, ``Program Files``, ``ProgramData``). Anything else
is refused with 403 — the engine never reads or writes arbitrary files on
behalf of a web page.

Refusals are *designed*: this module answers 400, 403 or 404 and never lets
a filesystem call raise through it. That matters because a client string is
not a path — ``"/x/a\\0b.wav"`` and a lone surrogate both make ``lstat``
raise ``ValueError`` deep inside ``Path.resolve()``, which the API would
otherwise publish as a 500 with a raw Python exception in the body. Text
that a POSIX filesystem *can* hold — newline, tab, ESC, DEL, non-UTF-8
bytes carried as surrogate escapes — is legal in a filename and is passed
through to the ordinary 403/404 answers.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hawavoclean.paths import work_root

_WINDOWS_DRIVE_REMOVABLE = 2
_WINDOWS_DRIVE_FIXED = 3
_WINDOWS_BLOCKED_SYSTEM_DIRS = frozenset(
    {
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "system volume information",
        "$recycle.bin",
    }
)


class PathPolicyError(Exception):
    """A client-supplied path was refused; ``status`` is the HTTP status."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _active_platform(platform: str | None) -> str:
    return sys.platform if platform is None else platform


def _windows_system_drive() -> str:
    raw = os.environ.get("SYSTEMDRIVE") or Path.home().drive or "C:"
    cleaned = raw.strip().rstrip("\\/")
    if len(cleaned) == 1 and cleaned.isalpha():
        cleaned = f"{cleaned}:"
    return cleaned.upper()


def _is_system_drive(candidate: Path, system_drive: str) -> bool:
    drive = candidate.drive.rstrip("\\/").upper()
    return bool(drive) and drive == system_drive


def windows_local_drives(*, include_system_drive: bool = False) -> list[Path]:
    """Return accessible local fixed/removable drive roots on Windows.

    By default, excludes the OS system drive (e.g. ``C:\\``) so system-drive
    access remains scoped to the user's home directory and work directory
    while secondary media/data drives (e.g. ``D:\\``, ``E:\\``) are allowed.
    """
    if sys.platform != "win32":
        return []
    import ctypes

    windll: Any = getattr(ctypes, "windll", None)
    if windll is None:
        return []
    kernel32 = windll.kernel32
    bitmask = int(kernel32.GetLogicalDrives())
    system_drive = _windows_system_drive()
    drives: list[Path] = []
    for idx in range(26):
        if not (bitmask & (1 << idx)):
            continue
        letter = chr(ord("A") + idx)
        root_str = f"{letter}:\\"
        drive_type = int(kernel32.GetDriveTypeW(root_str))
        if drive_type not in (_WINDOWS_DRIVE_REMOVABLE, _WINDOWS_DRIVE_FIXED):
            continue
        root = Path(root_str)
        if not include_system_drive and _is_system_drive(root, system_drive):
            continue
        drives.append(root)
    return drives


def _resolve_windows_secondary_drives(
    *,
    platform: str | None = None,
    windows_drives: Iterable[Path] | None = None,
) -> list[Path]:
    if _active_platform(platform) != "win32":
        return []
    system_drive = _windows_system_drive()
    raw_drives = (
        list(windows_drives)
        if windows_drives is not None
        else windows_local_drives(include_system_drive=False)
    )
    resolved_drives: list[Path] = []
    for candidate in raw_drives:
        if _is_system_drive(candidate, system_drive):
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved not in resolved_drives:
            resolved_drives.append(resolved)
    return resolved_drives


def is_blocked_windows_system_path(
    resolved: Path,
    *,
    platform: str | None = None,
    windows_drives: Iterable[Path] | None = None,
) -> bool:
    """Return True if ``resolved`` targets a blocked Windows system directory."""
    if _active_platform(platform) != "win32":
        return False

    # 1. Real Windows drive-qualified paths (e.g. C:\Windows, D:\Program Files).
    if (
        resolved.drive
        and len(resolved.parts) >= 2
        and resolved.parts[1].casefold() in _WINDOWS_BLOCKED_SYSTEM_DIRS
    ):
        return True

    # 2. Explicit OS environment system directories when running on real Windows.
    if sys.platform == "win32":
        for env_key in ("SYSTEMROOT", "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA"):
            raw_sys = os.environ.get(env_key)
            if not raw_sys:
                continue
            try:
                sys_path = Path(raw_sys).resolve()
            except OSError:
                continue
            if resolved == sys_path or resolved.is_relative_to(sys_path):
                return True

    # 3. Secondary drive roots (including POSIX-injected test roots like tmp/drive_D).
    for drive_root in _resolve_windows_secondary_drives(
        platform=platform, windows_drives=windows_drives
    ):
        if resolved != drive_root and resolved.is_relative_to(drive_root):
            rel = resolved.relative_to(drive_root)
            if rel.parts and rel.parts[0].casefold() in _WINDOWS_BLOCKED_SYSTEM_DIRS:
                return True

    return False


def allowed_roots(
    *,
    platform: str | None = None,
    windows_drives: Iterable[Path] | None = None,
) -> list[Path]:
    """Roots a client path may resolve under (resolved, deduplicated)."""
    candidates: list[Path] = [Path.home(), Path("/Volumes"), work_root()]
    if _active_platform(platform) == "win32":
        candidates.extend(
            _resolve_windows_secondary_drives(platform=platform, windows_drives=windows_drives)
        )
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - pathological filesystem state
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def refuse_unusable_filename_text(raw: str, *, what: str = "path") -> None:
    """Raise 400 if ``raw`` is text no filesystem call can even accept.

    Two cases, both of which make ``os``/``pathlib`` raise ``ValueError``
    rather than return a refusal:

    * a NUL byte — it terminates the C string, so it can never be part of a
      name (and silently truncating it would open a policy bypass);
    * text that does not survive ``os.fsencode`` — an unpaired surrogate,
      which ``json.loads`` produces from a ``"\\ud800"`` escape and which
      the UTF-8 codec refuses. (Surrogate *escapes* in the U+DC80–DCFF
      range do encode: they are how Python carries non-UTF-8 filenames, and
      they stay legal here.)
    """
    nul = raw.find("\x00")
    if nul >= 0:
        raise PathPolicyError(400, "bad_request", f"{what} contains a NUL byte at position {nul}")
    try:
        raw.encode("utf-8", "surrogateescape")
    except (UnicodeEncodeError, ValueError) as e:
        detail = getattr(e, "reason", None) or str(e)
        raise PathPolicyError(
            400, "bad_request", f"{what} is not a usable filename: {detail}"
        ) from e


def _refuse_windows_unc_or_device(raw: str, *, platform: str | None = None) -> None:
    if _active_platform(platform) != "win32":
        return
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise PathPolicyError(
            403,
            "forbidden",
            f"network UNC or device paths are not allowed: {raw}",
        )


def resolve_client_path(
    raw: str,
    *,
    must_exist: bool = False,
    platform: str | None = None,
    windows_drives: Iterable[Path] | None = None,
) -> Path:
    """Validate ``raw`` against the policy and return its resolved ``Path``.

    Raises :class:`PathPolicyError` with 400 (empty, not absolute, or text
    that cannot be a filename at all), 403 (outside the allowed roots or
    inside a blocked Windows system directory), or 404 (``must_exist`` and no
    such file). It never raises anything else.
    """
    if not raw or not raw.strip():
        raise PathPolicyError(400, "bad_request", "path is required")
    refuse_unusable_filename_text(raw)
    _refuse_windows_unc_or_device(raw, platform=platform)
    p = Path(raw)
    if not p.is_absolute():
        raise PathPolicyError(400, "bad_request", f"path must be absolute: {raw}")
    try:
        resolved = p.resolve()
    except (OSError, ValueError) as e:  # pragma: no cover - defence in depth
        raise PathPolicyError(400, "bad_request", f"path cannot be resolved: {e}") from e
    if is_blocked_windows_system_path(resolved, platform=platform, windows_drives=windows_drives):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is inside a protected Windows system directory: {raw}",
        )
    roots = allowed_roots(platform=platform, windows_drives=windows_drives)
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is outside the allowed locations (home, /Volumes, work dir): {raw}",
        )
    if must_exist:
        try:
            exists = resolved.is_file()
        except (OSError, ValueError):  # pragma: no cover - defence in depth
            exists = False
        if not exists:
            raise PathPolicyError(404, "not_found", f"file not found: {raw}")
    return resolved


def resolve_client_output_path(
    raw: str,
    *,
    platform: str | None = None,
    windows_drives: Iterable[Path] | None = None,
) -> Path:
    """Validate a writable public path without following its final alias.

    A committed HawaVoClean output is a symlink into its adjacent generation
    bundle. Following that final symlink during an overwrite would silently
    retarget the job at the immutable old generation. The parent is still
    fully resolved and checked against policy; only the final public name is
    preserved lexically for the publisher.
    """
    if not raw or not raw.strip():
        raise PathPolicyError(400, "bad_request", "path is required")
    refuse_unusable_filename_text(raw)
    _refuse_windows_unc_or_device(raw, platform=platform)
    path = Path(raw)
    if not path.is_absolute():
        raise PathPolicyError(400, "bad_request", f"path must be absolute: {raw}")
    if path.name in {".", ".."}:
        return resolve_client_path(
            raw, must_exist=False, platform=platform, windows_drives=windows_drives
        )
    try:
        parent = path.parent.resolve()
    except (OSError, ValueError) as exc:  # pragma: no cover - defence in depth
        raise PathPolicyError(400, "bad_request", f"path cannot be resolved: {exc}") from exc
    candidate = parent / path.name
    if is_blocked_windows_system_path(
        parent, platform=platform, windows_drives=windows_drives
    ) or is_blocked_windows_system_path(
        candidate, platform=platform, windows_drives=windows_drives
    ):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is inside a protected Windows system directory: {raw}",
        )
    roots = allowed_roots(platform=platform, windows_drives=windows_drives)
    if not any(parent == root or parent.is_relative_to(root) for root in roots):
        raise PathPolicyError(
            403,
            "forbidden",
            f"path is outside the allowed locations (home, /Volumes, work dir): {raw}",
        )
    return candidate
