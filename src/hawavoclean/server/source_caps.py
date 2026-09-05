"""Opaque capabilities for native files selected outside the renderer.

The desktop and Resolve main processes own the broker root secret.  They use
that authority once, immediately after an OS/Resolve selection, to register an
exact regular file.  Renderer sessions never gain a general "read a path"
capability: they can name only a path that is still backed by the same file
identity registered here, or use the opaque 128-bit source id.

Registrations are intentionally process-local.  A broker restart invalidates
them along with every renderer session, so stale capabilities cannot silently
survive a new trust boundary.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hawavoclean.server.policy import PathPolicyError, refuse_unusable_filename_text

MAX_NATIVE_SOURCES = 4096
SOURCE_ID_HEX_LENGTH = 32


def _is_redirect(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _open_nofollow_descriptor(path: Path) -> int:
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        FILE_SHARE_DELETE = 0x00000004
        OPEN_EXISTING = 3
        FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
        FILE_ATTRIBUTE_NORMAL = 0x00000080
        GENERIC_READ = 0x80000000

        kernel32 = vars(ctypes)["windll"].kernel32
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        handle = kernel32.CreateFileW(
            str(path),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_ATTRIBUTE_NORMAL,
            None,
        )
        invalid_handle = wintypes.HANDLE(-1).value
        if handle == invalid_handle or handle == -1 or handle is None:
            err = int(kernel32.GetLastError())
            raise OSError(None, f"CreateFileW failed with error {err}", str(path), err)
        return int(msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)))

    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def resolve_native_selected_path(raw_path: str) -> Path:
    """Resolve one root-authorized OS selection without a root allowlist.

    A native file picker can legitimately select a secondary Windows drive,
    network mount, or another OS location that is not below the home/work
    roots used for legacy browser paths.  The root-only registration call is
    the authority; subsequent renderer access remains exact-capability based.
    """

    if not raw_path or not raw_path.strip():
        raise PathPolicyError(400, "bad_request", "selected source path is required")
    refuse_unusable_filename_text(raw_path, what="selected source path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise PathPolicyError(400, "bad_request", "selected source path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise PathPolicyError(404, "not_found", "selected source file was not found") from exc
    try:
        info = resolved.stat()
    except OSError as exc:
        raise PathPolicyError(404, "not_found", "selected source file was not found") from exc
    if not stat.S_ISREG(info.st_mode):
        raise PathPolicyError(404, "not_found", "selected source is not a regular file")
    return resolved


@dataclass(frozen=True)
class NativeSource:
    """One exact native file authorized by a root-owned selection."""

    source_id: str
    path: Path
    device: int
    inode: int
    descriptor: int


class NativeSourceRegistry:
    """Bounded, thread-safe native-file capabilities backed by stable OS handles.

    Holding an open no-follow descriptor prevents filesystem inode-reuse and
    uncoordinated file replacement while registered. Canonical path and
    filesystem identity are re-verified via fstat/lstat on every use. A path
    swapped to a symlink, a different regular file, or a different hard link
    after selection immediately loses authority.
    """

    def __init__(self, maximum: int = MAX_NATIVE_SOURCES) -> None:
        if maximum < 1 or maximum > 65_536:
            raise ValueError("maximum native sources must be between 1 and 65536")
        self.maximum = maximum
        self._entries: OrderedDict[str, NativeSource] = OrderedDict()
        self._by_path: dict[Path, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _identity(path: Path) -> tuple[int, int] | None:
        try:
            current = path.resolve(strict=True)
            info = current.stat()
        except (OSError, ValueError):
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        return int(info.st_dev), int(info.st_ino)

    @staticmethod
    def _close_source(source: NativeSource) -> None:
        if source.descriptor >= 0:
            with contextlib.suppress(OSError):
                os.close(source.descriptor)

    def close(self) -> None:
        """Close all retained open OS handles and clear capability registry."""
        with self._lock:
            for source in self._entries.values():
                self._close_source(source)
            self._entries.clear()
            self._by_path.clear()

    def __enter__(self) -> NativeSourceRegistry:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock") and hasattr(self, "_entries"):
            self.close()

    def register(self, raw_path: str) -> NativeSource:
        """Register one exact, existing regular file and return its capability."""

        path = resolve_native_selected_path(raw_path)
        with self._lock:
            previous_id = self._by_path.get(path)
            if previous_id is not None:
                previous = self._entries.get(previous_id)
                if previous is not None and self._valid_locked(previous):
                    self._entries.move_to_end(previous_id)
                    return previous
                if previous is not None:
                    self._close_source(previous)
                self._entries.pop(previous_id, None)
                self._by_path.pop(path, None)

            # Open with O_NOFOLLOW to avoid symlink traversal during open
            try:
                descriptor = _open_nofollow_descriptor(path)
            except OSError as exc:
                raise PathPolicyError(
                    404, "not_found", f"selected source cannot be opened safely: {path}"
                ) from exc

            try:
                fd_info = os.fstat(descriptor)
                path_info = os.lstat(path)
                if (
                    not stat.S_ISREG(fd_info.st_mode)
                    or not stat.S_ISREG(path_info.st_mode)
                    or _is_redirect(path_info)
                    or (fd_info.st_dev, fd_info.st_ino) != (path_info.st_dev, path_info.st_ino)
                ):
                    raise PathPolicyError(404, "not_found", "selected source is not a regular file")

                device = int(fd_info.st_dev)
                inode = int(fd_info.st_ino)

                source_id = secrets.token_hex(SOURCE_ID_HEX_LENGTH // 2)
                while source_id in self._entries:  # pragma: no cover - 128-bit collision defence
                    source_id = secrets.token_hex(SOURCE_ID_HEX_LENGTH // 2)
                source = NativeSource(source_id, path, device, inode, descriptor)
                self._entries[source_id] = source
                self._by_path[path] = source_id
                while len(self._entries) > self.maximum:
                    _, evicted = self._entries.popitem(last=False)
                    self._close_source(evicted)
                    self._by_path.pop(evicted.path, None)
                return source
            except Exception:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                raise

    def _valid_locked(self, source: NativeSource) -> bool:
        if source.descriptor < 0:
            return False
        try:
            fd_info = os.fstat(source.descriptor)
            path_info = os.lstat(source.path)
        except OSError:
            return False
        if not stat.S_ISREG(fd_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
            return False
        if _is_redirect(path_info):
            return False
        if (fd_info.st_dev, fd_info.st_ino) != (source.device, source.inode):
            return False
        return (path_info.st_dev, path_info.st_ino) == (source.device, source.inode)

    def resolve_source(self, source_id: str) -> Path | None:
        """Resolve a 32-hex capability if the selected file is still identical."""

        if len(source_id) != SOURCE_ID_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in source_id
        ):
            return None
        with self._lock:
            source = self._entries.get(source_id)
            if source is None:
                return None
            if not self._valid_locked(source):
                self._close_source(source)
                self._entries.pop(source_id, None)
                self._by_path.pop(source.path, None)
                return None
            self._entries.move_to_end(source_id)
            return source.path

    def authorizes(self, path: Path) -> bool:
        """Return whether ``path`` is the still-identical registered file."""

        candidate = path.resolve()
        with self._lock:
            source_id = self._by_path.get(candidate)
            if source_id is None:
                return False
            source = self._entries.get(source_id)
            if source is None or not self._valid_locked(source):
                if source is not None:
                    self._close_source(source)
                self._entries.pop(source_id, None)
                self._by_path.pop(candidate, None)
                return False
            self._entries.move_to_end(source_id)
            return True

    def resolve_registered_path(self, raw_path: str) -> Path | None:
        """Resolve a path-form legacy request only when root registered it."""

        try:
            candidate = resolve_native_selected_path(raw_path)
        except PathPolicyError:
            return None
        return candidate if self.authorizes(candidate) else None

    @contextmanager
    def lease_source(self, source_id: str) -> Iterator[Path | None]:
        """Match the upload-store lease interface for versioned source ids.

        Native inputs are user-owned and never scavenged by HawaVoClean, so no
        retention lease is required after the identity check.
        """

        yield self.resolve_source(source_id)
