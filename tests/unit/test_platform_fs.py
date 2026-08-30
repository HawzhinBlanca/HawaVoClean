"""Portable publication primitives select safe native branches lazily."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import hawavoclean.platform_fs as platform_fs


def test_publication_import_does_not_require_posix_or_windows_lock_module() -> None:
    script = r"""
import builtins
import sys
import pathlib
import hawavoclean

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name in {"fcntl", "msvcrt"}:
        raise AssertionError(f"eager native lock import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.modules.pop("fcntl", None)
sys.modules.pop("msvcrt", None)
import hawavoclean.publication
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_windows_lock_branch_locks_one_real_byte_and_unlocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int]] = []
    fake_msvcrt = SimpleNamespace(
        LK_LOCK=10,
        LK_NBLCK=12,
        LK_UNLCK=11,
        locking=lambda _descriptor, mode, count: calls.append((mode, count)),
    )
    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(platform_fs, "_load_native_module", lambda _name: fake_msvcrt)

    lock_path = tmp_path / "publication.lock"
    with platform_fs.exclusive_file_lock(lock_path):
        assert lock_path.read_bytes() == b"\0"

    assert calls == [(fake_msvcrt.LK_NBLCK, 1), (fake_msvcrt.LK_UNLCK, 1)]


def test_windows_lock_retries_only_contention_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0
    delays: list[float] = []

    def locking(_descriptor: int, mode: int, _count: int) -> None:
        nonlocal attempts
        if mode == 12:
            attempts += 1
            if attempts < 3:
                raise OSError(13, "lock contention")

    fake_msvcrt = SimpleNamespace(LK_LOCK=10, LK_NBLCK=12, LK_UNLCK=11, locking=locking)
    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(platform_fs, "_load_native_module", lambda _name: fake_msvcrt)
    monkeypatch.setattr("hawavoclean.platform_fs.time.sleep", delays.append)

    with platform_fs.exclusive_file_lock(tmp_path / "publication.lock"):
        pass

    assert attempts == 3
    assert delays == [0.05, 0.05]


def test_process_lock_serializes_threads_when_native_lock_is_process_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_entered = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []
    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(platform_fs, "_lock_descriptor", lambda _descriptor: None)
    monkeypatch.setattr(platform_fs, "_unlock_descriptor", lambda _descriptor: None)
    path = tmp_path / "publication.lock"

    def first() -> None:
        with platform_fs.exclusive_file_lock(path):
            order.append("first-enter")
            first_entered.set()
            assert release_first.wait(timeout=2)
            order.append("first-exit")

    def second() -> None:
        assert first_entered.wait(timeout=2)
        second_started.set()
        with platform_fs.exclusive_file_lock(path):
            order.append("second-enter")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first)
        second_future = pool.submit(second)
        assert second_started.wait(timeout=2)
        assert order == ["first-enter"]
        release_first.set()
        first_future.result(timeout=2)
        second_future.result(timeout=2)

    assert order == ["first-enter", "first-exit", "second-enter"]
    assert platform_fs._PROCESS_LOCKS == {}


def test_windows_replace_retries_sharing_violation_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "new"
    destination = tmp_path / "current"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    attempts = 0
    delays: list[float] = []

    def replace_once(left: Path, right: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(32, "sharing violation")
        os.replace(left, right)

    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(platform_fs, "_replace_once", replace_once)
    monkeypatch.setattr("hawavoclean.platform_fs.time.sleep", delays.append)

    platform_fs.replace_path(source, destination)

    assert attempts == 3
    assert delays == [0.005, 0.01]
    assert destination.read_bytes() == b"new"


def test_windows_replace_does_not_retry_non_sharing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail(_source: Path, _destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise OSError(87, "invalid parameter")

    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(platform_fs, "_replace_once", fail)
    with pytest.raises(OSError, match="invalid parameter"):
        platform_fs.replace_path(tmp_path / "source", tmp_path / "destination")
    assert calls == 1


def test_windows_new_path_move_is_write_through_without_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, Path, bool]] = []

    def move(source: Path, destination: Path, *, replace: bool) -> None:
        calls.append((source, destination, replace))

    source = tmp_path / "staging"
    destination = tmp_path / "generation"
    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(platform_fs, "_windows_move", move)

    platform_fs.rename_new_path(source, destination)

    assert calls == [(source, destination, False)]


@pytest.mark.skipif(os.name == "nt", reason="exercises the native POSIX primitive")
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_posix_new_path_rename_never_replaces_an_existing_winner(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    if kind == "directory":
        source.mkdir()
        destination.mkdir()
        (source / "payload").write_bytes(b"candidate")
        (destination / "payload").write_bytes(b"winner")
    else:
        source.write_bytes(b"candidate")
        destination.write_bytes(b"winner")

    with pytest.raises(FileExistsError):
        platform_fs.rename_new_path(source, destination)

    assert source.exists()
    payload = destination / "payload" if kind == "directory" else destination
    assert payload.read_bytes() == b"winner"


@pytest.mark.skipif(os.name == "nt", reason="exercises POSIX directory fsync")
def test_posix_new_path_rename_flushes_both_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "candidate"
    destination = destination_parent / "winner"
    source.write_bytes(b"complete")
    flushed: list[Path] = []
    real_rename = platform_fs._posix_rename_new_path
    monkeypatch.setattr(platform_fs, "_posix_rename_new_path", real_rename)
    monkeypatch.setattr(platform_fs, "flush_directory", flushed.append)

    platform_fs.rename_new_path(source, destination)

    assert destination.read_bytes() == b"complete"
    assert flushed == [destination_parent, source_parent]


def test_windows_native_move_uses_replace_and_write_through_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, int]] = []

    class MoveFile:
        argtypes: object = None
        restype: object = None

        def __call__(self, source: str, destination: str, flags: int) -> int:
            calls.append((source, destination, flags))
            return 1

    move_file = MoveFile()
    fake_ctypes = SimpleNamespace(
        WinDLL=lambda *_args, **_kwargs: SimpleNamespace(MoveFileExW=move_file),
        get_last_error=lambda: 0,
    )
    fake_wintypes = SimpleNamespace(LPCWSTR=str, DWORD=int, BOOL=int)

    def load(name: str) -> object:
        return fake_wintypes if name == "ctypes.wintypes" else fake_ctypes

    monkeypatch.setattr(platform_fs, "_load_native_module", load)
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    platform_fs._windows_move(source, destination, replace=True)
    platform_fs._windows_move(source, destination, replace=False)

    assert calls == [
        (str(source), str(destination), 0x9),
        (str(source), str(destination), 0x8),
    ]


def test_windows_directory_flush_uses_write_through_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform_fs, "_platform_name", lambda: "nt")
    monkeypatch.setattr(
        "hawavoclean.platform_fs.os.open",
        lambda *_args, **_kwargs: pytest.fail("Windows must not POSIX-open a directory"),
    )
    platform_fs.flush_directory(tmp_path)


def test_lifetime_file_lease_is_nonblocking_across_processes_and_reusable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broker.owner.lock"
    script = r"""
import pathlib
import sys
from hawavoclean.platform_fs import try_acquire_exclusive_file_lease
try:
    lease = try_acquire_exclusive_file_lease(pathlib.Path(sys.argv[1]))
except BlockingIOError:
    raise SystemExit(9)
lease.release()
"""
    lease = platform_fs.try_acquire_exclusive_file_lease(path)
    try:
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert blocked.returncode == 9, blocked.stderr
    finally:
        lease.release()

    acquired = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert acquired.returncode == 0, acquired.stderr
