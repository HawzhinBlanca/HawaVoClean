"""The studio core's compute device: pinned, reported, and never assumed.

Gated on torch/DeepFilterNet being installed. The GPU comparison is skipped
on a machine that has no GPU — it is a measurement, and there is nothing to
measure without one.
"""

import numpy as np
import pytest

from hawavoclean import runtime
from hawavoclean.errors import ConfigError


def _speech_like(seconds: float = 1.0, sr: int = 48000) -> np.ndarray:
    rng = np.random.default_rng(5)
    t = np.arange(int(sr * seconds)) / sr
    sig = 0.3 * np.sin(2 * np.pi * 190 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return (sig + 0.02 * rng.standard_normal(t.shape)).astype(np.float32)


def test_core_takes_its_device_from_the_armed_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio import StudioVoiceCore

    monkeypatch.setenv(runtime.DEVICE_ENV_VAR, "cpu")
    assert StudioVoiceCore().device == "cpu"


def test_core_refuses_a_device_this_machine_cannot_provide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio import StudioVoiceCore

    monkeypatch.setattr(runtime, "device_available", lambda n: n == "cpu")
    with pytest.raises(ConfigError):
        StudioVoiceCore(device="cuda")


def test_lowband_core_takes_its_device_from_the_armed_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio_lowband import StudioLowBandCore

    monkeypatch.setenv(runtime.DEVICE_ENV_VAR, "cpu")
    assert StudioLowBandCore().device == "cpu"


def test_lowband_core_refuses_a_device_this_machine_cannot_provide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio_lowband import StudioLowBandCore

    monkeypatch.setattr(runtime, "device_available", lambda n: n == "cpu")
    with pytest.raises(ConfigError):
        StudioLowBandCore(device="cuda")


def test_loading_the_model_pins_deepfilternets_own_device_lookup() -> None:
    """df.enhance re-reads df's global device for every feature tensor. If the
    pin does not take, model and features land on different devices."""
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio import StudioVoiceCore

    core = StudioVoiceCore(device="cpu")
    core.warmup()
    from df.utils import get_device

    assert str(get_device()) == "cpu"


def test_pinning_the_device_never_rewrites_the_vendored_config() -> None:
    """df.config.set marks the parser dirty; only df.config.save() would touch
    the file, and the file's digest is locked."""
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio import _MODEL_DIR, StudioVoiceCore
    from hawavoclean.hashing import hash_file

    cfg_ini = _MODEL_DIR / "config.ini"
    before = hash_file(cfg_ini)
    StudioVoiceCore(device="cpu").warmup()
    assert hash_file(cfg_ini) == before


@pytest.mark.gpu
def test_gpu_output_differs_from_cpu_which_is_why_auto_stays_on_cpu() -> None:
    """The measurement behind ``AUTO_DEVICE_PREFERENCE == ("cpu",)``."""
    pytest.importorskip("torch")
    import torch

    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        pytest.skip("no GPU backend on this machine")
    gpu = "mps" if torch.backends.mps.is_available() else "cuda"

    from hawavoclean.enhancement.studio import StudioVoiceCore

    x = _speech_like(2.0)
    cpu_out = StudioVoiceCore(device="cpu").enhance(x, 48000).waveform
    gpu_out = StudioVoiceCore(device=gpu).enhance(x, 48000).waveform

    assert len(cpu_out) == len(gpu_out)
    # Close, but NOT the same numbers — which is the entire reason a GPU is
    # opt-in and is recorded in the report.
    assert not np.array_equal(cpu_out, gpu_out)
    assert float(np.abs(cpu_out - gpu_out).max()) < 1e-4
