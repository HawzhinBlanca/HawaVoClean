"""Model-cold runtime contracts for optional enhancement dependencies.

These probes deliberately import the exact modules and symbols used by the
neural cores without constructing a model or reading its weights.  Readiness
inspection executes them in an isolated child interpreter so imports cannot
alter the broker's ``sys.modules``, native runtime state, or environment.

The torchaudio compatibility shim lives here because the real DFN3 loader and
the readiness probe must exercise the same import path.  Keeping two copies
would let a capability probe pass while the worker later fails (or vice
versa).
"""

from __future__ import annotations

import sys
import types
from typing import Any


class RuntimeDependencyContractError(RuntimeError):
    """An optional dependency imported incorrectly or lacks a used symbol."""


def _require_callable(value: Any, attribute: str, label: str) -> None:
    if not callable(getattr(value, attribute, None)):
        raise RuntimeDependencyContractError(f"{label} is missing or is not callable")


def install_torchaudio_compat() -> None:
    """Install the annotation-only shim required by DeepFilterNet 0.5.6.

    ``df.io`` still imports ``torchaudio.backend.common.AudioMetaData``, which
    torchaudio >=2.2 removed.  DeepFilterNet uses it only as a type
    annotation in a path HawaVoClean does not call, so the runtime supplies
    the same narrow compatibility object before importing ``df``.
    """

    if "torchaudio.backend.common" in sys.modules:
        return
    try:
        import torchaudio  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"torchaudio import failed ({type(exc).__name__})"
        ) from None

    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # annotation-only stand-in
        pass

    common.AudioMetaData = getattr(torchaudio, "AudioMetaData", AudioMetaData)  # type: ignore[attr-defined]
    backend = types.ModuleType("torchaudio.backend")
    backend.common = common  # type: ignore[attr-defined]
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def _probe_deepfilternet_runtime_contract() -> None:
    """Import and validate every symbol used by the shared DFN3 path."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"torch import failed ({type(exc).__name__})"
        ) from None
    for attribute in ("device", "from_numpy", "no_grad"):
        _require_callable(torch, attribute, f"torch.{attribute}")

    install_torchaudio_compat()
    try:
        from df.config import config as df_config
        from df.enhance import enhance as df_enhance
        from df.enhance import init_df
        from df.utils import get_device
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"DeepFilterNet import failed ({type(exc).__name__})"
        ) from None

    _require_callable(df_config, "set", "df.config.config.set")
    if not callable(init_df):
        raise RuntimeDependencyContractError("df.enhance.init_df is missing or is not callable")
    if not callable(df_enhance):
        raise RuntimeDependencyContractError("df.enhance.enhance is missing or is not callable")
    if not callable(get_device):
        raise RuntimeDependencyContractError("df.utils.get_device is missing or is not callable")


def probe_studio_runtime_contract() -> None:
    """Validate Studio's DFN3 and WPE imports without creating a model."""

    _probe_deepfilternet_runtime_contract()
    try:
        from nara_wpe.wpe import wpe
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"nara_wpe import failed ({type(exc).__name__})"
        ) from None
    if not callable(wpe):
        raise RuntimeDependencyContractError("nara_wpe.wpe.wpe is missing or is not callable")


def probe_lowband_runtime_contract() -> None:
    """Validate Lowband's shared DFN3 imports without creating a model."""

    _probe_deepfilternet_runtime_contract()


__all__ = [
    "RuntimeDependencyContractError",
    "install_torchaudio_compat",
    "probe_lowband_runtime_contract",
    "probe_studio_runtime_contract",
]
