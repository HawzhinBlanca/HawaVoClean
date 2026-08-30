"""Small cross-platform filesystem primitives used by publication.

The module deliberately avoids importing ``fcntl`` or ``msvcrt`` at import
time.  A base wheel must be importable on both POSIX and Windows; the native
locking module is selected only when a lock is actually acquired.
"""

from __future__ import annotations

import errno
import importlib
import os
import stat
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WINDOWS_REPLACE_ATTEMPTS = 8
_WINDOWS_REPLACE_INITIAL_DELAY_SECONDS = 0.005
_WINDOWS_SHARING_ERRORS = {5, 32, 33}
_WINDOWS_LOCK_RETRY_SECONDS = 0.05
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, tuple[threading.RLock, int]] = {}
_PROCESS_LEASES: set[str] = set()


def _platform_name() -> str:
    """Return the active OS family through a testable seam."""
    return os.name


def _platform_system() -> str:
    """Return the concrete POSIX family through a testable seam."""

    return sys.platform


def _load_native_module(name: str) -> Any:
    return importlib.import_module(name)


def _is_reparse_or_symlink(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare an opened file with its directory entry on POSIX and NTFS."""
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def is_reparse_or_symlink(path: Path) -> bool:
    """Return whether an existing path redirects through a filesystem reparse point."""
    try:
        return _is_reparse_or_symlink(os.lstat(path))
    except FileNotFoundError:
        return False


def _lock_registry_key(path: Path) -> str:
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    return key.casefold() if _platform_name() == "nt" else key


@contextmanager
def _process_file_lock(path: Path) -> Iterator[None]:
    """Serialize threads even where native record locks are process-scoped."""
    key = _lock_registry_key(path)
    with _PROCESS_LOCKS_GUARD:
        lock, users = _PROCESS_LOCKS.get(key, (threading.RLock(), 0))
        _PROCESS_LOCKS[key] = (lock, users + 1)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _PROCESS_LOCKS_GUARD:
            registered, users = _PROCESS_LOCKS[key]
            if registered is lock and users == 1:
                del _PROCESS_LOCKS[key]
            else:
                _PROCESS_LOCKS[key] = (registered, users - 1)


def _open_safe_lock(path: Path) -> int:
    """Open a regular, non-reparse lock file without trusting a path race."""
    before: os.stat_result | None
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        before = None
    if before is not None and (_is_reparse_or_symlink(before) or not stat.S_ISREG(before.st_mode)):
        raise OSError(
            errno.ELOOP,
            "publication lock is not a regular file or is a reparse point",
            path,
        )

    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        after = os.lstat(path)
        if (
            _is_reparse_or_symlink(after)
            or not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or not _same_identity(opened, after)
            or (before is not None and not _same_identity(before, opened))
        ):
            raise OSError(
                errno.ELOOP,
                "publication lock changed identity or is not a regular file",
                path,
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _lock_descriptor(descriptor: int) -> None:
    if _platform_name() == "nt":
        msvcrt = _load_native_module("msvcrt")
        if os.fstat(descriptor).st_size < 1:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        lock_mode = getattr(msvcrt, "LK_NBLCK", msvcrt.LK_LOCK)
        while True:
            try:
                msvcrt.locking(descriptor, lock_mode, 1)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                    raise
                time.sleep(_WINDOWS_LOCK_RETRY_SECONDS)
        return
    fcntl = _load_native_module("fcntl")
    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if _platform_name() == "nt":
        msvcrt = _load_native_module("msvcrt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl = _load_native_module("fcntl")
    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _try_lock_descriptor(descriptor: int) -> None:
    """Acquire once or raise ``BlockingIOError`` without sleeping."""

    if _platform_name() == "nt":
        msvcrt = _load_native_module("msvcrt")
        if os.fstat(descriptor).st_size < 1:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise BlockingIOError(errno.EWOULDBLOCK, "file lease is already held") from exc
            raise
        return
    fcntl = _load_native_module("fcntl")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise BlockingIOError(errno.EWOULDBLOCK, "file lease is already held") from exc
        raise


@dataclass(slots=True)
class ExclusiveFileLease:
    """A nonblocking exclusive lock held until explicit release."""

    path: Path
    _descriptor: int
    _registry_key: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        try:
            with suppress(OSError):
                _unlock_descriptor(self._descriptor)
        finally:
            try:
                os.close(self._descriptor)
            finally:
                with _PROCESS_LOCKS_GUARD:
                    _PROCESS_LEASES.discard(self._registry_key)
                self._released = True


def try_acquire_exclusive_file_lease(path: Path) -> ExclusiveFileLease:
    """Acquire a cross-process owner lease immediately or fail closed."""

    key = _lock_registry_key(path)
    with _PROCESS_LOCKS_GUARD:
        if key in _PROCESS_LEASES:
            raise BlockingIOError(errno.EWOULDBLOCK, "file lease is already held", path)
        _PROCESS_LEASES.add(key)
    descriptor: int | None = None
    try:
        descriptor = _open_safe_lock(path)
        _try_lock_descriptor(descriptor)
        return ExclusiveFileLease(path, descriptor, key)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        with _PROCESS_LOCKS_GUARD:
            _PROCESS_LEASES.discard(key)
        raise


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an inter-process exclusive lock backed by a safe regular file."""
    with _process_file_lock(path):
        descriptor = _open_safe_lock(path)
        acquired = False
        try:
            _lock_descriptor(descriptor)
            acquired = True
            yield
        finally:
            try:
                if acquired:
                    with suppress(OSError):
                        _unlock_descriptor(descriptor)
            finally:
                os.close(descriptor)


def flush_directory(path: Path) -> None:
    """Persist directory metadata where the host exposes directory fsync.

    Windows path moves below use ``MOVEFILE_WRITE_THROUGH``.  Windows does not
    expose POSIX directory descriptors, so there is no second directory fsync.
    """
    if _platform_name() == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _windows_move(source: Path, destination: Path, *, replace: bool) -> None:
    """Move one path on Windows with metadata write-through semantics."""
    ctypes = _load_native_module("ctypes")
    wintypes = _load_native_module("ctypes.wintypes")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file = kernel32.MoveFileExW
    move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file.restype = wintypes.BOOL
    flags = 0x8  # MOVEFILE_WRITE_THROUGH
    if replace:
        flags |= 0x1  # MOVEFILE_REPLACE_EXISTING
    if not move_file(str(source), str(destination), flags):
        error = ctypes.get_last_error()
        raise ctypes.WinError(error)


def _raise_native_os_error(error: int, destination: Path) -> None:
    """Raise Python's specific ``OSError`` subclass for a native errno."""

    raise OSError(error, os.strerror(error), destination)


def _posix_rename_new_path(source: Path, destination: Path) -> None:
    """Use the host's atomic no-replace rename primitive.

    Plain ``rename(2)`` replaces an existing destination, so checking first is
    inherently racy.  HawaVoClean's supported POSIX host is macOS, whose
    ``renamex_np(RENAME_EXCL)`` provides the required single atomic decision.
    Linux CI and development hosts use ``renameat2(RENAME_NOREPLACE)``.  Other
    POSIX families fail closed instead of silently weakening publication.
    """

    ctypes = _load_native_module("ctypes")
    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    system = _platform_system()
    if system == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(encoded_source, encoded_destination, 0x00000004)
    elif system.startswith("linux"):
        try:
            rename_exclusive = libc.renameat2
        except AttributeError as exc:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable on this POSIX host",
                destination,
            ) from exc
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            -100,  # AT_FDCWD
            encoded_source,
            -100,  # AT_FDCWD
            encoded_destination,
            0x00000001,  # RENAME_NOREPLACE
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rename is unavailable on this POSIX host",
            destination,
        )
    if result != 0:
        _raise_native_os_error(int(ctypes.get_errno()), destination)


def _flush_rename_directories(source: Path, destination: Path) -> None:
    """Persist both directory-entry changes made by a cross-directory rename."""

    flush_directory(destination.parent)
    if source.parent != destination.parent:
        flush_directory(source.parent)


def _replace_once(source: Path, destination: Path) -> None:
    if _platform_name() == "nt":
        _windows_move(source, destination, replace=True)
    else:
        os.replace(source, destination)


def replace_path(source: Path, destination: Path) -> None:
    """Atomically replace a same-volume path.

    POSIX callers persist the parent with :func:`flush_directory` after any
    fault-injection checkpoint they need. Windows replacement is write-through
    because Windows has no equivalent directory descriptor to flush later.
    """
    delay = _WINDOWS_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(_WINDOWS_REPLACE_ATTEMPTS):
        try:
            _replace_once(source, destination)
            return
        except OSError as exc:
            windows_error = getattr(exc, "winerror", None) or exc.errno
            retryable = _platform_name() == "nt" and windows_error in _WINDOWS_SHARING_ERRORS
            if not retryable or attempt + 1 == _WINDOWS_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay *= 2


def rename_new_path(source: Path, destination: Path) -> None:
    """Atomically commit a new same-volume path without replacing a winner."""
    if _platform_name() == "nt":
        _windows_move(source, destination, replace=False)
    else:
        _posix_rename_new_path(source, destination)
        _flush_rename_directories(source, destination)
