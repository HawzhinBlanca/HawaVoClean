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
        resolve_native_selected_path("/nonexistent/file/path_12345.wav")

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

    # _identity returns None when file is missing during register
    monkeypatch.setattr(registry, "_identity", lambda _p: None)
    with pytest.raises(PathPolicyError, match="not a regular file"):
        registry.register(str(source))
