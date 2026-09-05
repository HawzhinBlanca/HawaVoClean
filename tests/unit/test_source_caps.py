"""Native source capabilities bind a selection to one exact regular file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hawavoclean.paths import work_root
from hawavoclean.server.policy import PathPolicyError
from hawavoclean.server.source_caps import NativeSourceRegistry

pytestmark = pytest.mark.unit


def _source(name: str, value: bytes = b"audio") -> Path:
    path = work_root() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_registration_is_stable_bounded_and_opaque() -> None:
    registry = NativeSourceRegistry(maximum=1)
    first = _source("first.wav")
    registered = registry.register(str(first))
    repeated = registry.register(str(first))
    assert repeated == registered
    assert len(registered.source_id) == 32
    assert set(registered.source_id) <= set("0123456789abcdef")
    assert registry.resolve_source(registered.source_id) == first.resolve()
    assert registry.authorizes(first)

    second = registry.register(str(_source("second.wav")))
    assert second.source_id != registered.source_id
    assert registry.resolve_source(registered.source_id) is None
    assert not registry.authorizes(first)


def test_replacing_or_redirecting_selected_path_revokes_authority() -> None:
    registry = NativeSourceRegistry()
    source = _source("replace.wav", b"first")
    registered = registry.register(str(source))

    replacement = _source("replacement.wav", b"different inode")
    replacement.replace(source)
    assert registry.resolve_source(registered.source_id) is None
    assert not registry.authorizes(source)

    target = _source("target.wav", b"target")
    source.unlink()
    os.symlink(target.name, source)
    assert not registry.authorizes(source.resolve())


def test_invalid_ids_directories_and_limits_fail_closed() -> None:
    registry = NativeSourceRegistry()
    assert registry.resolve_source("") is None
    assert registry.resolve_source("g" * 32) is None
    directory = work_root() / "folder"
    directory.mkdir(parents=True)
    with pytest.raises(PathPolicyError):
        registry.register(str(directory))
    with pytest.raises(ValueError):
        NativeSourceRegistry(maximum=0)
    with pytest.raises(ValueError):
        NativeSourceRegistry(maximum=65_537)


def test_native_lease_never_deletes_user_owned_file() -> None:
    registry = NativeSourceRegistry()
    source = _source("leased.wav")
    registered = registry.register(str(source))
    with registry.lease_source(registered.source_id) as leased:
        assert leased == source.resolve()
    assert source.is_file()


def test_resolve_native_selected_path_and_registry_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawavoclean.server.source_caps import resolve_native_selected_path

    # Empty path
    with pytest.raises(PathPolicyError, match="selected source path is required"):
        resolve_native_selected_path("")
    with pytest.raises(PathPolicyError, match="selected source path is required"):
        resolve_native_selected_path("   ")

    # Relative path
    with pytest.raises(PathPolicyError, match="must be absolute"):
        resolve_native_selected_path("relative/path.wav")

    # Non-existent path
    with pytest.raises(PathPolicyError, match="not found"):
        resolve_native_selected_path(
            str(Path.cwd().resolve() / "nonexistent" / "file" / "path_12345.wav")
        )

    registry = NativeSourceRegistry()
    source = _source("reg_path.wav")

    # resolve_registered_path before registering
    assert registry.resolve_registered_path(str(source)) is None
    # resolve_registered_path with invalid path
    assert registry.resolve_registered_path("relative/path.wav") is None

    # Register and resolve
    registry.register(str(source))
    assert registry.resolve_registered_path(str(source)) == source.resolve()

    # Re-register path after inode changed
    source.write_bytes(b"modified bytes")
    # Replace file to change inode
    temp_new = _source("temp_new.wav", b"new content")
    os.replace(temp_new, source)
    reg_new = registry.register(str(source))
    assert reg_new.path == source.resolve()

    # Failing open during register raises PathPolicyError
    def failing_open(_p: Path) -> int:
        raise OSError("Permission denied")

    monkeypatch.setattr("hawavoclean.server.source_caps._open_nofollow_descriptor", failing_open)
    fail_source = _source("fail_open.wav")
    with pytest.raises(PathPolicyError, match="cannot be opened safely"):
        registry.register(str(fail_source))


def test_source_caps_validation_edge_branches() -> None:
    from hawavoclean.server.source_caps import NativeSource

    registry = NativeSourceRegistry()
    source = _source("caps_val.wav")
    registered = registry.register(str(source))

    # 1. Closed descriptor (< 0) fails validation
    invalid_source = NativeSource(
        source_id=registered.source_id,
        path=registered.path,
        device=registered.device,
        inode=registered.inode,
        descriptor=-1,
    )
    assert not registry._valid_locked(invalid_source)

    # 2. Device/inode mismatch fails validation
    mismatch_source = NativeSource(
        source_id=registered.source_id,
        path=registered.path,
        device=registered.device + 1,
        inode=registered.inode + 1,
        descriptor=registered.descriptor,
    )
    assert not registry._valid_locked(mismatch_source)


def test_open_nofollow_descriptor_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ctypes
    import sys
    import types
    from unittest.mock import MagicMock

    from hawavoclean.server.source_caps import _open_nofollow_descriptor

    test_file = tmp_path / "win_test.wav"
    test_file.write_bytes(b"data")

    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.open_osfhandle = MagicMock(return_value=42)  # type: ignore[attr-defined]

    fake_windll = MagicMock()
    fake_kernel32 = MagicMock()
    fake_kernel32.CreateFileW.return_value = 1234
    fake_windll.kernel32 = fake_kernel32

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setitem(vars(ctypes), "windll", fake_windll)

    fd = _open_nofollow_descriptor(test_file)
    assert fd == 42

    # Error branch: invalid handle returned by CreateFileW
    fake_kernel32.CreateFileW.return_value = -1
    fake_kernel32.GetLastError.return_value = 5
    with pytest.raises(OSError, match="CreateFileW failed"):
        _open_nofollow_descriptor(test_file)
