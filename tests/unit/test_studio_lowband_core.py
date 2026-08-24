"""Band-split core: the crossover's exactness, its confinement, and its lock.

The crossover is the whole core. Two properties carry it, and both are
pinned here rather than left to inspection:

* it reconstructs the input EXACTLY when the model changes nothing, so the
  crossover itself can never colour the signal;
* nothing the model does can escape it upward, so the consonant band is
  preserved by construction and not by the model's good behaviour.
"""

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hawavoclean.config import load_config
from hawavoclean.enhancement.factory import resolve_core
from hawavoclean.enhancement.studio_lowband import (
    STUDIO_LOWBAND_PARAMS,
    StudioLowBandCore,
    crossover_mix,
    lowpass_zero_phase,
    studio_lowband_params_hash,
)
from hawavoclean.guard.signal import check_signal_integrity
from hawavoclean.paths import profile_config_path

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "src" / "hawavoclean" / "resources" / "models"
SR = 48000
CORE_ID = "studio-dfn3-lowband-48k-v1"


def _lock() -> dict[str, Any]:
    with open(MODELS_DIR / "studio-lowband-core.lock.toml", "rb") as f:
        return dict(tomllib.load(f))


def _tone(freq: float, seconds: float = 1.0, amp: float = 0.3) -> np.ndarray[Any, Any]:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_registry_resolves_the_lowband_core() -> None:
    reg = resolve_core(CORE_ID)
    assert reg.lock_filename == "studio-lowband-core.lock.toml"
    assert reg.enhancer_class is StudioLowBandCore
    # It shares the studio core's DFN3 weights but runs no WPE, so nara_wpe
    # must NOT be among its requirements or a base install would be told to
    # go and get a dependency this core never imports.
    assert "nara_wpe" not in reg.requires_modules
    assert set(reg.requires_modules) == {"df", "torch"}
    assert reg.device_aware is True


def test_crossover_reconstructs_the_input_when_the_model_changes_nothing() -> None:
    """The crossover must be transparent on its own.

    Both bands come from the same lowpass and the high one by subtraction, so
    an identity 'enhancement' has to come back out as the input — otherwise
    the crossover is colouring every file it touches, at the exact frequency
    where the voice's first formant lives.
    """
    rng = np.random.default_rng(3)
    x = (_tone(180.0, 2.0) + _tone(3000.0, 2.0, 0.2) + 0.05 * rng.standard_normal(SR * 2)).astype(
        np.float32
    )
    out = crossover_mix(x, x, SR, 1000.0, 4)
    assert out.shape == x.shape
    # float32 round-trip through one subtraction and one addition: a couple of
    # ULPs. A real crossover error (a notch or a bump) would be ~1e-1.
    assert np.max(np.abs(out - x)) < 1e-5, float(np.max(np.abs(out - x)))


def test_crossover_confines_the_model_to_the_low_band() -> None:
    """Even a model that destroys everything cannot touch the high band.

    'enhanced' here is silence — the worst thing a model could hand back. The
    tone above the crossover must survive it almost untouched; the one below
    must not.
    """
    fc = float(STUDIO_LOWBAND_PARAMS["crossover_hz"])
    order = int(STUDIO_LOWBAND_PARAMS["crossover_order"])
    low_tone, high_tone = _tone(150.0, 2.0), _tone(4000.0, 2.0)
    x = (low_tone + high_tone).astype(np.float32)

    out = crossover_mix(np.zeros_like(x), x, SR, fc, order)

    def band_rms(sig: np.ndarray[Any, Any], f: float) -> float:
        t = np.arange(len(sig)) / SR
        ref = np.exp(-2j * np.pi * f * t)
        return float(np.abs(np.mean(sig * ref)))

    assert band_rms(out, 4000.0) > 0.95 * band_rms(x, 4000.0)
    assert band_rms(out, 150.0) < 0.02 * band_rms(x, 150.0)


def test_crossover_frequency_and_order_are_inside_the_params_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the crossover must be a relock, not an edit.

    The crossover is the line between a core that ships audio and one whose
    every unit the guard reverts, so it has to be impossible to change it
    without the lockfile noticing.
    """
    baseline = studio_lowband_params_hash()
    assert baseline == _lock()["params_hash"]

    for key, value in (("crossover_hz", 1500.0), ("crossover_order", 8)):
        moved = dict(STUDIO_LOWBAND_PARAMS)
        moved[key] = value
        monkeypatch.setattr("hawavoclean.enhancement.studio_lowband.STUDIO_LOWBAND_PARAMS", moved)
        assert studio_lowband_params_hash() != baseline, f"{key} is not in the params hash"
        monkeypatch.undo()


def test_lock_declares_the_shipped_crossover() -> None:
    lock = _lock()
    assert lock["core_id"] == CORE_ID
    assert lock["phase_coherent"] is False
    assert lock["params"]["crossover_hz"] == STUDIO_LOWBAND_PARAMS["crossover_hz"]
    assert lock["params"]["crossover_order"] == STUDIO_LOWBAND_PARAMS["crossover_order"]
    # It borrows the studio core's weights; both locks must name the same files
    # at the same digests or one of them is describing a model it does not run.
    with open(MODELS_DIR / "studio-core.lock.toml", "rb") as f:
        studio_lock = tomllib.load(f)
    assert lock["weight_sha256"] == studio_lock["weight_sha256"]


def test_lowband_profile_agrees_with_the_lock() -> None:
    cfg = load_config(profile_config_path("lowband"), is_production=True)
    lock = _lock()
    assert cfg.enhancement.core_id == CORE_ID
    # The pipeline refuses to run when these disagree; catch it here instead.
    assert cfg.enhancement.phase_coherent == lock["phase_coherent"]
    assert cfg.enhancement.model_sample_rate in lock["expected_sample_rates"]
    # Not phase-coherent, so a partial-strength blend would comb-filter.
    assert cfg.policy.strength_ladder == [1.0]
    # Restoration changes the spectrum by design; identity is not enforced,
    # but every other protection is.
    assert cfg.guard.mode == "integrity"
    assert cfg.guard.enforce_signal_integrity is True


def test_core_refuses_configurations_it_cannot_honour() -> None:
    with pytest.raises(ValueError, match="not phase-coherent"):
        StudioLowBandCore(phase_coherent=True)
    with pytest.raises(ValueError, match="48000"):
        StudioLowBandCore(sample_rate=16000)


def test_lowpass_survives_signals_shorter_than_its_own_pad() -> None:
    """A degenerate unit must not raise out of the filter.

    scipy's filtfilt refuses a signal shorter than its default pad, which is
    an exception in the middle of a run rather than a short unit handled.
    """
    for n in (0, 1, 2, 7, 40):
        x = np.linspace(-0.5, 0.5, n, dtype=np.float32)
        out = lowpass_zero_phase(x, SR, 1000.0, 4)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))
        mixed = crossover_mix(x, x, SR, 1000.0, 4)
        assert mixed.shape == x.shape
        if n >= 2:
            assert np.max(np.abs(mixed - x)) < 1e-5


def test_empty_input_is_returned_unchanged_without_loading_a_model() -> None:
    core = StudioLowBandCore()
    res = core.enhance(np.zeros(0, dtype=np.float32), SR)
    assert res.output_samples == 0 and len(res.waveform) == 0


def test_metadata_describes_the_locked_core() -> None:
    meta = StudioLowBandCore().metadata
    lock = _lock()
    assert meta.core_id == lock["core_id"] == CORE_ID
    assert meta.params_hash == lock["params_hash"]
    assert meta.algorithm == lock["algorithm"]
    assert meta.phase_coherent is False
    assert meta.sample_rate == STUDIO_LOWBAND_PARAMS["internal_sample_rate"]


@pytest.mark.parametrize("delta", [-777, 0, 1024])
def test_a_model_that_returns_a_different_length_is_realigned(
    monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    """DFN3 works in frames and may hand back a different sample count.

    If that were added to the high band as-is, every sample above the
    crossover would land at the wrong offset — an inaudible-looking bug that
    smears the whole unit. The model here returns silence at the wrong
    length; the high band must still come through in the right place.
    """
    calls: list[int] = []

    def fake_load(_device: str) -> tuple[object, object]:
        calls.append(1)
        return object(), object()

    def fake_run(_m: object, _s: object, audio: Any, _atten: float) -> Any:
        return np.zeros(max(0, len(audio) + delta), dtype=np.float32)

    monkeypatch.setattr("hawavoclean.enhancement.studio_lowband.load_deepfilternet3", fake_load)
    monkeypatch.setattr("hawavoclean.enhancement.studio_lowband.run_deepfilternet3", fake_run)

    x = (_tone(150.0, 2.0) + _tone(4000.0, 2.0)).astype(np.float32)
    core = StudioLowBandCore()
    out = core.enhance(x, SR).waveform

    assert len(out) == len(x)
    assert np.all(np.isfinite(out))
    # The model contributed nothing but silence, so the output must be the
    # complementary high band of the input — in phase with it, not shifted.
    expected = crossover_mix(
        np.zeros_like(x),
        x,
        SR,
        float(STUDIO_LOWBAND_PARAMS["crossover_hz"]),
        int(STUDIO_LOWBAND_PARAMS["crossover_order"]),
    )
    assert np.max(np.abs(out - expected)) < 1e-5

    # A second call reuses the loaded model; warmup does not reload it either.
    core.enhance(x, SR)
    core.warmup()
    assert calls == [1]


def test_lowband_inference_keeps_the_consonant_band_and_drops_the_rumble() -> None:
    """The measured claim, on synthetic material: rumble out, consonants in.

    A 70 Hz drone under modulated speech-band content. The band under the
    crossover must lose most of its energy; the consonant band must come
    through at essentially unity, and the guard's own integrity check must
    accept the result.
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    seconds = 3.0
    t = np.arange(int(SR * seconds)) / SR
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t)
    voice = 0.25 * np.sin(2 * np.pi * 220 * t) * envelope
    fricative = 0.05 * rng.standard_normal(len(t)) * envelope
    rumble = 0.15 * np.sin(2 * np.pi * 70 * t)
    x = (voice + fricative + rumble).astype(np.float32)

    res = StudioLowBandCore().enhance(x, SR)
    y = res.waveform
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))

    integrity = check_signal_integrity(
        x,
        y,
        SR,
        spectral_hole_thresh=0.10,
        musical_noise_thresh=0.30,
        min_hf_preservation_ratio=0.50,
    )
    assert integrity.passed, integrity.failure_reasons

    def band_energy(sig: np.ndarray[Any, Any], lo: float, hi: float) -> float:
        spec = np.abs(np.fft.rfft(sig.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / SR)
        sel = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spec[sel] ** 2))

    assert band_energy(y, 50.0, 100.0) < 0.25 * band_energy(x, 50.0, 100.0)
    assert integrity.consonant_retention_ratio > 0.90
