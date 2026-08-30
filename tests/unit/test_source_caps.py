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

    source.unlink()
    source.write_bytes(b"different inode")
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
