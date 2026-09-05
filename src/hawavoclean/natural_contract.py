"""Fail-closed readiness contract for every shipped Natural route.

Capability reporting and the processing pipeline must answer the same
question: can this profile run with the exact config, guard calibration,
core implementation, lockfile, optional dependencies, and weights installed
right now?  Keeping that preflight here prevents the API from advertising a
route which the pipeline would immediately refuse.

The inspection is intentionally model-cold.  Optional runtime imports and
their used symbols are verified in an isolated child interpreter, core
parameter callables recompute only small hashes, and model files are streamed
through SHA-256.  No enhancer is constructed and no neural weights are loaded
into memory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from hawavoclean.config import GuardConfig, HawaVoCleanConfig, load_config
from hawavoclean.enhancement.factory import resolve_core
from hawavoclean.errors import CalibrationError, ConfigError, PreflightError
from hawavoclean.guard.calibration import (
    apply_calibrated_thresholds,
    load_calibration_artifact,
)
from hawavoclean.hashing import hash_bytes, hash_file, hash_json_canonical
from hawavoclean.paths import (
    models_dir,
    profile_config_path,
    resolve_calibration_file,
)
from hawavoclean.runtime import resolve_device

_DEPENDENCY_PROBE_TIMEOUT_S = 30.0
_DEPENDENCY_PROBE_SENTINEL = "HAWAVOCLEAN_DEPENDENCY_PROBE_V1="
_DEPENDENCY_PROBE_PROGRAM = r"""
import importlib
import json
import sys

sentinel = "HAWAVOCLEAN_DEPENDENCY_PROBE_V1="
search_path = json.loads(sys.argv[1])
if not isinstance(search_path, list) or not all(isinstance(item, str) for item in search_path):
    print(sentinel + json.dumps({"ok": False, "error": "invalid_search_path"}))
    raise SystemExit(3)
sys.path[:] = search_path

try:
    module_name, separator, attribute_path = sys.argv[2].partition(":")
    if not module_name or separator != ":" or not attribute_path:
        raise ValueError("invalid probe reference")
    value = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError("probe target is not callable")
    value()
except Exception as exc:
    # Probe implementations expose only fixed, path-free contract messages.
    # Unknown exceptions are reduced to their type so capability responses do
    # not disclose package locations or arbitrary third-party error strings.
    safe_detail = str(exc) if type(exc).__name__ == "RuntimeDependencyContractError" else ""
    print(
        sentinel
        + json.dumps(
            {"ok": False, "error": type(exc).__name__, "detail": safe_detail},
            sort_keys=True,
        )
    )
    raise SystemExit(3)

print(sentinel + json.dumps({"ok": True}, sort_keys=True))
"""


def _runtime_import_search_path() -> tuple[str, ...]:
    """Return the broker's exact import roots for the isolated interpreter."""

    paths: list[str] = []
    for entry in sys.path:
        try:
            value = os.fspath(entry)
        except TypeError:
            continue
        if value not in paths:
            paths.append(value)
    return tuple(paths)


@lru_cache(maxsize=16)
def _probe_optional_runtime_contract(
    core_id: str,
    probe_reference: str,
    required_modules: tuple[str, ...],
    search_path: tuple[str, ...],
) -> None:
    """Run one optional dependency contract outside the broker process."""

    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _DEPENDENCY_PROBE_PROGRAM,
                json.dumps(search_path),
                probe_reference,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_DEPENDENCY_PROBE_TIMEOUT_S,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(
            f"Core {core_id!r} optional runtime import/contract probe timed out after "
            f"{_DEPENDENCY_PROBE_TIMEOUT_S:g}s (required: {', '.join(required_modules)})"
        ) from exc
    except OSError as exc:
        raise PreflightError(
            f"Core {core_id!r} optional runtime import/contract probe could not start "
            f"({type(exc).__name__}; required: {', '.join(required_modules)})"
        ) from exc

    record: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(_DEPENDENCY_PROBE_SENTINEL):
            continue
        try:
            value: Any = json.loads(line.removeprefix(_DEPENDENCY_PROBE_SENTINEL))
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            record = value
        break
    if completed.returncode == 0 and record == {"ok": True}:
        return

    error_type = "invalid probe response"
    detail = ""
    if record is not None:
        if isinstance(record.get("error"), str):
            error_type = str(record["error"])
        if isinstance(record.get("detail"), str):
            detail = str(record["detail"])
    safe_detail = f": {detail}" if detail else ""
    raise PreflightError(
        f"Core {core_id!r} optional runtime import/contract failed "
        f"({error_type}{safe_detail}; required: {', '.join(required_modules)}). "
        "Reinstall the signed application runtime."
    )


@dataclass(frozen=True)
class NaturalRouteContract:
    """Verified inputs shared by capability reporting and pipeline preflight."""

    profile: str
    config: HawaVoCleanConfig
    active_guard: GuardConfig
    calibration: dict[str, Any]
    calibration_sha256: str
    core_lock: dict[str, Any]
    core_lock_sha256: str
    provider: str
    manifest_sha256: str


def load_core_lock(core_id: str) -> tuple[dict[str, Any], str]:
    """Load and verify a core lock exactly, without constructing its model."""

    try:
        registration = resolve_core(core_id)
    except KeyError as exc:
        raise PreflightError(str(exc)) from exc

    if registration.requires_modules and registration.dependency_probe is None:
        raise PreflightError(
            f"Core {core_id!r} declares optional dependencies without a runtime contract"
        )
    if registration.dependency_probe is not None:
        _probe_optional_runtime_contract(
            core_id,
            registration.dependency_probe,
            registration.requires_modules,
            _runtime_import_search_path(),
        )

    model_root = models_dir()
    lock_path = model_root / registration.lock_filename
    if not lock_path.is_file():
        raise PreflightError(
            f"Core lockfile missing: {lock_path}. Refusing to run without "
            "verifiable core provenance."
        )
    try:
        raw_lock = lock_path.read_bytes()
        lock_value = tomllib.loads(raw_lock.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PreflightError(f"Core lockfile is unreadable: {lock_path} ({exc})") from exc
    lock = dict(lock_value)

    if lock.get("core_id") != core_id:
        raise PreflightError(
            f"Configured core_id {core_id!r} does not match lockfile core {lock.get('core_id')!r}"
        )
    actual_params_hash = registration.implementation_params_hash()
    if lock.get("params_hash") != actual_params_hash:
        raise PreflightError(
            "Core parameter drift: lockfile params_hash "
            f"{str(lock.get('params_hash'))[:16]}... does not match the "
            f"implemented core {actual_params_hash[:16]}..."
        )

    params = lock.get("params", {})
    weights = lock.get("weight_sha256", {})
    if not isinstance(params, dict) or not isinstance(weights, dict):
        raise PreflightError("Core lockfile params/weight tables are invalid")
    payload: dict[str, Any] = dict(params)
    weight_table = {str(key): str(value) for key, value in weights.items()}
    if weight_table:
        payload["weights_sha256"] = weight_table
    if hash_json_canonical(payload) != actual_params_hash:
        raise PreflightError(
            "Core lockfile tables do not recompute to params_hash; "
            "the lockfile has been hand-edited."
        )

    for relative, expected_digest in sorted(weight_table.items()):
        weight_path = model_root / relative
        if not weight_path.is_file():
            raise PreflightError(f"Locked weights file missing: {weight_path}")
        if hash_file(weight_path) != expected_digest:
            raise PreflightError(f"Weights digest mismatch for {relative}")
    return lock, hash_bytes(raw_lock)


def load_natural_route_contract(
    profile: str,
    *,
    config: HawaVoCleanConfig | None = None,
    config_path: Path | str | None = None,
    activate_runtime_config: bool = True,
) -> NaturalRouteContract:
    """Verify and identify one Natural route using the pipeline's exact inputs."""

    if config is None:
        selected_config = (
            Path(config_path) if config_path is not None else profile_config_path(profile)
        )
        config = load_config(
            selected_config,
            is_production=profile == "production",
            activate=activate_runtime_config,
        )

    # Provider validation is pure: explicit unavailable devices still block
    # readiness, while a capability query does not publish the result into
    # process-global environment or thread-budget state.
    provider = resolve_device(
        config.runtime.device,
        core_id=config.enhancement.core_id,
    ).resolved

    calibration_path = resolve_calibration_file(config.guard.calibration_file)
    calibration = load_calibration_artifact(calibration_path)
    if hash_json_canonical(calibration["thresholds"]) != calibration.get("calibration_id"):
        raise CalibrationError(
            f"Guard calibration artifact {calibration_path} has been edited: calibration_id "
            "does not recompute from its thresholds. Refusing to run with a tampered guard."
        )
    active_guard = apply_calibrated_thresholds(config.guard, calibration)
    calibration_sha256 = hash_file(calibration_path)

    core_lock, core_lock_sha256 = load_core_lock(config.enhancement.core_id)
    registration = resolve_core(config.enhancement.core_id)
    if bool(core_lock.get("phase_coherent", True)) != config.enhancement.phase_coherent:
        raise ConfigError(
            f"enhancement.phase_coherent = {config.enhancement.phase_coherent} but core "
            f"{config.enhancement.core_id!r} is "
            f"{'phase-coherent' if core_lock.get('phase_coherent', True) else 'NOT phase-coherent'}; "
            "the report would misstate the core and the policy would blend residuals incorrectly."
        )
    try:
        expected_rates = [int(rate) for rate in core_lock.get("expected_sample_rates", [])]
    except (TypeError, ValueError) as exc:
        raise ConfigError("Core lockfile expected_sample_rates is invalid") from exc
    if expected_rates and config.enhancement.model_sample_rate not in expected_rates:
        raise ConfigError(
            f"enhancement.model_sample_rate = {config.enhancement.model_sample_rate} but core "
            f"{config.enhancement.core_id!r} runs at {expected_rates}"
        )

    # This digest is the capability's exact runnable identity.  It binds all
    # values later attested in the processing report, including the effective
    # parsed config rather than only a mutable filename.
    manifest_sha256 = hash_json_canonical(
        {
            "schema_version": 1,
            "profile": profile,
            "config_sha256": config.compute_hash(),
            "core": {
                "id": config.enhancement.core_id,
                "lock_sha256": core_lock_sha256,
                "params_hash": core_lock.get("params_hash"),
                "dependency_probe": registration.dependency_probe,
                "required_modules": list(registration.requires_modules),
                "weight_sha256": {
                    str(key): str(value)
                    for key, value in dict(core_lock.get("weight_sha256", {})).items()
                },
            },
            "guard": {
                "id": active_guard.guard_id,
                "probe_id": active_guard.probe_id,
                "calibration_id": calibration.get("calibration_id"),
                "calibration_sha256": calibration_sha256,
            },
        }
    )
    return NaturalRouteContract(
        profile=profile,
        config=config,
        active_guard=active_guard,
        calibration=calibration,
        calibration_sha256=calibration_sha256,
        core_lock=core_lock,
        core_lock_sha256=core_lock_sha256,
        provider=provider,
        manifest_sha256=manifest_sha256,
    )


__all__ = ["NaturalRouteContract", "load_core_lock", "load_natural_route_contract"]
