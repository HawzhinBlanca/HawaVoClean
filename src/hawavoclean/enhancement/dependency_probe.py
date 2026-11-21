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

import importlib
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


def probe_studio_pro_runtime_contract() -> None:
    """Validate Studio Pro's MossFormer2_SE_48K imports and SHA-256 weight lock.

    Deliberately model-cold: verifies required Python/native symbols and
    streams the checkpoint through SHA-256 without constructing the neural
    network or allocating tensor weights. All failure messages are path-free.
    """

    try:
        import torch
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"torch import failed ({type(exc).__name__})"
        ) from None
    for attribute in ("device", "from_numpy", "no_grad", "stft", "istft", "load"):
        _require_callable(torch, attribute, f"torch.{attribute}")

    try:
        ta_kaldi = importlib.import_module("torchaudio.compliance.kaldi")
        ta_functional = importlib.import_module("torchaudio.functional")
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"torchaudio import failed ({type(exc).__name__})"
        ) from None
    _require_callable(ta_kaldi, "fbank", "torchaudio.compliance.kaldi.fbank")
    _require_callable(ta_functional, "compute_deltas", "torchaudio.functional.compute_deltas")

    try:
        arch_mod = importlib.import_module("hawavoclean.enhancement.mossformer2_arch")
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"mossformer2_arch import failed ({type(exc).__name__})"
        ) from None
    _require_callable(
        arch_mod,
        "MossFormer2SE48K",
        "hawavoclean.enhancement.mossformer2_arch.MossFormer2SE48K",
    )
    _require_callable(
        arch_mod,
        "extract_mossformer2_features",
        "hawavoclean.enhancement.mossformer2_arch.extract_mossformer2_features",
    )
    _require_callable(
        arch_mod,
        "apply_mossformer2_mask",
        "hawavoclean.enhancement.mossformer2_arch.apply_mossformer2_mask",
    )

    from hawavoclean.enhancement.studio_pro import verify_studio_pro_weights

    weights_ok, reason = verify_studio_pro_weights()
    if not weights_ok:
        raise RuntimeDependencyContractError(reason)


def probe_dereverb_runtime_contract() -> None:
    """Validate Dereverb's pure-PyTorch TF-GridNetV3 imports and SHA-256 weight lock.

    Deliberately model-cold: verifies required Python/native symbols and
    streams ``Predictor.pt`` through SHA-256 without constructing the neural
    network or allocating tensor weights. All failure messages are path-free.
    """

    try:
        import torch
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"torch import failed ({type(exc).__name__})"
        ) from None
    for attribute in ("device", "from_numpy", "inference_mode", "stft", "istft", "load"):
        _require_callable(torch, attribute, f"torch.{attribute}")

    try:
        importlib.import_module("torchaudio")
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"torchaudio import failed ({type(exc).__name__})"
        ) from None

    try:
        dereverb_mod = importlib.import_module("hawavoclean.enhancement.dereverb_neural")
    except Exception as exc:
        raise RuntimeDependencyContractError(
            f"dereverb_neural import failed ({type(exc).__name__})"
        ) from None
    _require_callable(
        dereverb_mod,
        "NeuralDereverbCore",
        "hawavoclean.enhancement.dereverb_neural.NeuralDereverbCore",
    )
    _require_callable(
        dereverb_mod,
        "load_dereverb_model",
        "hawavoclean.enhancement.dereverb_neural.load_dereverb_model",
    )
    _require_callable(
        dereverb_mod,
        "verify_dereverb_weights",
        "hawavoclean.enhancement.dereverb_neural.verify_dereverb_weights",
    )

    weights_ok, reason = dereverb_mod.verify_dereverb_weights()
    if not weights_ok:
        raise RuntimeDependencyContractError(reason)


__all__ = [
    "RuntimeDependencyContractError",
    "install_torchaudio_compat",
    "probe_dereverb_runtime_contract",
    "probe_lowband_runtime_contract",
    "probe_studio_pro_runtime_contract",
    "probe_studio_runtime_contract",
]
