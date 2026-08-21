"""The build hook fails closed while supporting metadata-free build contexts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest


def _load_build_hook() -> types.ModuleType:
    if importlib.util.find_spec("hatchling") is not None:
        return importlib.import_module("hatch_build")

    class BuildHookInterface:
        pass

    modules = {
        "hatchling": types.ModuleType("hatchling"),
        "hatchling.builders": types.ModuleType("hatchling.builders"),
        "hatchling.builders.hooks": types.ModuleType("hatchling.builders.hooks"),
        "hatchling.builders.hooks.plugin": types.ModuleType("hatchling.builders.hooks.plugin"),
        "hatchling.builders.hooks.plugin.interface": types.ModuleType(
            "hatchling.builders.hooks.plugin.interface"
        ),
    }
    interface = cast(Any, modules["hatchling.builders.hooks.plugin.interface"])
    interface.BuildHookInterface = BuildHookInterface
    sys.modules.update(modules)
    try:
        return importlib.import_module("hatch_build")
    finally:
        for name in modules:
            sys.modules.pop(name, None)


hatch_build = _load_build_hook()


def _detached_root(tmp_path: Path) -> Path:
    (tmp_path / "uv.lock").write_text("locked\n", encoding="utf-8")
    release = tmp_path / "src" / "hawavoclean" / "release.json"
    release.parent.mkdir(parents=True)
    release.write_text('{"version":"test"}\n', encoding="utf-8")
    return tmp_path


def test_detached_build_uses_explicit_validated_source_anchors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _detached_root(tmp_path)
    monkeypatch.setenv("HAWAVOCLEAN_SOURCE_REVISION", "a" * 40)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "123")

    value = hatch_build.provenance_payload(root, "wheel")

    assert value["source_revision"] == "a" * 40
    assert value["source_date_epoch"] == 123
    assert value["source_dirty"] is False
    assert value["dependency_lock_sha256"] == hashlib.sha256(b"locked\n").hexdigest()


def test_detached_build_requires_both_source_anchors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _detached_root(tmp_path)
    monkeypatch.setenv("HAWAVOCLEAN_SOURCE_REVISION", "a" * 40)

    with pytest.raises(RuntimeError, match="require both"):
        hatch_build.provenance_payload(root, "wheel")


@pytest.mark.parametrize(
    ("revision", "epoch", "message"),
    [
        ("short", "123", "full lowercase Git SHA"),
        ("A" * 40, "123", "full lowercase Git SHA"),
        ("a" * 40, "not-an-int", "positive integer"),
        ("a" * 40, "0", "positive integer"),
    ],
)
def test_detached_build_rejects_malformed_source_anchors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    revision: str,
    epoch: str,
    message: str,
) -> None:
    root = _detached_root(tmp_path)
    monkeypatch.setenv("HAWAVOCLEAN_SOURCE_REVISION", revision)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)

    with pytest.raises(RuntimeError, match=message):
        hatch_build.provenance_payload(root, "wheel")


def test_detached_build_without_source_anchors_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires explicit source anchors"):
        hatch_build.provenance_payload(_detached_root(tmp_path), "wheel")


def test_checkout_rejects_conflicting_explicit_anchors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _detached_root(tmp_path)
    (root / ".git").mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "123")

    def fake_git(_root: Path, *args: str) -> str:
        if args[0] == "status":
            return ""
        if args[0] == "rev-parse":
            return "a" * 40
        return "123"

    monkeypatch.setattr(hatch_build, "_run_git", fake_git)

    with pytest.raises(RuntimeError, match="do not match"):
        hatch_build.provenance_payload(root, "wheel")
