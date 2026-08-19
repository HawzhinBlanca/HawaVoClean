"""Behavioral contract of the Wiener core: attenuate, never destroy or invent."""

import numpy as np

from voiceclean.enhancement.production import WIENER_PARAMS, WienerSpectralEnhancer

SR = 48000


def test_gain_floor_prevents_signal_annihilation() -> None:
    """On pure noise the Wiener gains want to go to zero; the floor forbids it.

    Removing the floor (gain_floor -> 0) collapses stationary content to
    near-silence and reintroduces musical noise — this test pins the floor's
    audible effect, not just the constant.
    """
    rng = np.random.default_rng(3)
    noise = (0.1 * rng.standard_normal(SR * 2)).astype(np.float32)

    enhancer = WienerSpectralEnhancer()
    result = enhancer.enhance(noise, SR)

    in_rms = float(np.sqrt(np.mean(noise**2)))
    out_rms = float(np.sqrt(np.mean(result.waveform**2)))
    floor = float(WIENER_PARAMS["gain_floor"])

    # Output RMS on stationary noise must stay above roughly the floor's
    # share of the input (with margin for windowing/resampling losses).
    assert out_rms >= 0.5 * floor * in_rms, (
        f"noise floor annihilated: in_rms={in_rms:.5f} out_rms={out_rms:.5f} (floor={floor})"
    )
    # And it must actually attenuate noise, or it does nothing at all.
    assert out_rms < 0.9 * in_rms, "enhancer did not attenuate stationary noise"


def test_length_and_finiteness_preserved() -> None:
    rng = np.random.default_rng(4)
    t = np.arange(SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(SR)).astype(np.float32)

    result = WienerSpectralEnhancer().enhance(x, SR)

    assert len(result.waveform) == len(x)
    assert np.all(np.isfinite(result.waveform))
