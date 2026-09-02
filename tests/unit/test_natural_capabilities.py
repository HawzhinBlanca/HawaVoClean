"""Natural capability truth follows the exact model-cold pipeline contract."""

from __future__ import annotations

import importlib.machinery
import os
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import hawavoclean.server.app as app_module
from hawavoclean.enhancement.studio import StudioVoiceCore
from hawavoclean.paths import models_dir
from hawavoclean.runtime import DEVICE_ENV_VAR, MEMORY_LIMIT_ENV_VAR, THREAD_ENV_VARS
from hawavoclean.server.app import capabilities_v1, create_app
from hawavoclean.server.contracts import CapabilityStatusV1
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit


def _capability(capability_id: str) -> CapabilityStatusV1:
    return next(
        item for item in capabilities_v1().capabilities if item.capability_id == capability_id
    )


def test_qualified_natural_routes_publish_exact_identity_and_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def model_must_remain_cold(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("capability inspection constructed a neural model")

    monkeypatch.setattr(StudioVoiceCore, "__init__", model_must_remain_cold)
    routes = [_capability(profile) for profile in ("production", "studio", "lowband")]
    assert all(route.available and route.maturity == "qualified" for route in routes)
    assert all(route.reason and "verified" in route.reason for route in routes)
    assert all(
        route.manifest_sha256 and re.fullmatch(r"[0-9a-f]{64}", route.manifest_sha256)
        for route in routes
    )
    assert len({route.manifest_sha256 for route in routes}) == 3
    assert all(route.providers == ["cpu"] for route in routes)


def test_discoverable_but_unimportable_optional_dependency_blocks_studio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Package discovery alone is not readiness: this simulates a present
    # native wheel whose import fails at dynamic-library load time.
    (tmp_path / "torch.py").write_text(
        "raise OSError('simulated broken native dependency')\n",
        encoding="utf-8",
    )
    assert importlib.machinery.PathFinder.find_spec("torch", [str(tmp_path)]) is not None
    monkeypatch.syspath_prepend(str(tmp_path))

    capability = _capability("studio")
    assert capability.available is False
    assert capability.maturity == "blocked"
    assert capability.manifest_sha256 is None
    assert capability.providers == []
    assert capability.reason is not None
    assert "optional runtime import/contract failed" in capability.reason
    assert "torch import failed (OSError)" in capability.reason
    assert "simulated broken native dependency" not in capability.reason


def test_importable_dependency_missing_used_symbol_blocks_studio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "nara_wpe"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "wpe.py").write_text("wpe = object()\n", encoding="utf-8")
    assert importlib.machinery.PathFinder.find_spec("nara_wpe", [str(tmp_path)]) is not None
    monkeypatch.syspath_prepend(str(tmp_path))

    capability = _capability("studio")
    assert capability.available is False
    assert capability.maturity == "blocked"
    assert capability.manifest_sha256 is None
    assert capability.reason is not None
    assert "nara_wpe.wpe.wpe is missing or is not callable" in capability.reason


def test_capability_inspection_preserves_broker_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = (DEVICE_ENV_VAR, MEMORY_LIMIT_ENV_VAR, *THREAD_ENV_VARS)
    values = {
        DEVICE_ENV_VAR: "mps",
        MEMORY_LIMIT_ENV_VAR: "4321",
        THREAD_ENV_VARS[0]: "7",
        THREAD_ENV_VARS[1]: "11",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    before = {name: os.environ.get(name) for name in protected}
    optional_prefixes = ("torch", "torchaudio", "df", "nara_wpe")
    modules_before = {
        name: module
        for name, module in sys.modules.items()
        if name in optional_prefixes
        or name.startswith(tuple(f"{item}." for item in optional_prefixes))
    }

    response = capabilities_v1()

    assert all(
        capability.available
        for capability in response.capabilities
        if capability.capability_id in {"production", "studio", "lowband"}
    )
    assert {name: os.environ.get(name) for name in protected} == before
    modules_after = {
        name: module
        for name, module in sys.modules.items()
        if name in optional_prefixes
        or name.startswith(tuple(f"{item}." for item in optional_prefixes))
    }
    assert modules_after == modules_before


def test_missing_config_is_blocked_and_local_root_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_configs = tmp_path / "private-configs"
    empty_configs.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_CONFIG_DIR", str(empty_configs))
    capability = _capability("production")
    assert capability.available is False and capability.maturity == "blocked"
    assert capability.reason is not None
    assert "Configuration file not found" in capability.reason
    assert str(tmp_path) not in capability.reason
    assert "<config-dir>" in capability.reason


def test_missing_locked_weight_is_blocked_and_model_root_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged_models = models_dir()
    staged_models = tmp_path / "private-models"
    staged_models.mkdir()
    (staged_models / "studio-core.lock.toml").write_bytes(
        (packaged_models / "studio-core.lock.toml").read_bytes()
    )
    (staged_models / "guard-calibration-studio.json").write_bytes(
        (packaged_models / "guard-calibration-studio.json").read_bytes()
    )
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(staged_models))
    capability = _capability("studio")
    assert capability.available is False and capability.maturity == "blocked"
    assert capability.reason is not None and "Locked weights file missing" in capability.reason
    assert str(tmp_path) not in capability.reason
    assert "<model-dir>" in capability.reason


def test_tampered_core_lock_blocks_production_instead_of_false_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged_models = models_dir()
    staged_models = tmp_path / "private-models"
    staged_models.mkdir()
    lock = (packaged_models / "production-core.lock.toml").read_text(encoding="utf-8")
    lock = re.sub(r'params_hash = "[0-9a-f]{64}"', f'params_hash = "{"0" * 64}"', lock)
    (staged_models / "production-core.lock.toml").write_text(lock, encoding="utf-8")
    (staged_models / "guard-calibration.json").write_bytes(
        (packaged_models / "guard-calibration.json").read_bytes()
    )
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(staged_models))
    capability = _capability("production")
    assert capability.available is False and capability.maturity == "blocked"
    assert capability.reason is not None and "Core parameter drift" in capability.reason
    assert capability.manifest_sha256 is None


def test_v1_submission_rechecks_selected_route_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(profile: str) -> CapabilityStatusV1:
        return CapabilityStatusV1(
            capability_id=profile,
            available=False,
            maturity="blocked",
            reason="selected route failed its runtime contract",
        )

    monkeypatch.setattr(app_module, "_natural_route_capability", blocked)
    manager = JobManager()
    app = create_app(
        "capability-token",
        job_manager=manager,
        on_shutdown=lambda: None,
        min_free_bytes=0,
    )
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/api/v1/jobs",
                headers={"X-Hawa-Token": "capability-token"},
                json={
                    "schemaVersion": 1,
                    "sourceIds": ["a" * 32],
                    "strategy": {
                        "kind": "manual",
                        "route": "production",
                        "allowGenerativeReconstruction": False,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "blocked-runtime-route",
                },
            )
        assert response.status_code == 503
        assert response.json() == {
            "error": "capability_blocked",
            "message": "selected route failed its runtime contract",
        }
    finally:
        manager.shutdown()


def test_load_core_lock_and_natural_contract_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hawavoclean.errors import PreflightError
    from hawavoclean.natural_contract import load_core_lock

    # 1. Unknown core id
    with pytest.raises(PreflightError, match="Unknown enhancement core"):
        load_core_lock("nonexistent_core_id")

    # 2. Corrupted / unreadable lockfile
    staged_models = tmp_path / "corrupt-models"
    staged_models.mkdir()
    (staged_models / "production-core.lock.toml").write_text("invalid [[ toml", encoding="utf-8")
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(staged_models))
    with pytest.raises(PreflightError, match="unreadable"):
        load_core_lock("wiener-dd-48k-v1")

    # 3. Lockfile core_id mismatch
    (staged_models / "production-core.lock.toml").write_text(
        'core_id = "wrong_core"\nparams_hash = "abc"\n', encoding="utf-8"
    )
    with pytest.raises(PreflightError, match="does not match"):
        load_core_lock("wiener-dd-48k-v1")

    # 4. _probe_optional_runtime_contract timeout and OSError branches
    import subprocess

    from hawavoclean.natural_contract import _probe_optional_runtime_contract

    def failing_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="probe", timeout=0.1)

    monkeypatch.setattr(subprocess, "run", failing_run)
    with pytest.raises(PreflightError, match="timed out"):
        _probe_optional_runtime_contract("test-core", "probe_ref", ("test_mod",), ("/fake/path",))

    def oserror_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("exec failed")

    monkeypatch.setattr(subprocess, "run", oserror_run)
    with pytest.raises(PreflightError, match="could not start"):
        _probe_optional_runtime_contract("test-core2", "probe_ref", ("test_mod",), ("/fake/path",))
