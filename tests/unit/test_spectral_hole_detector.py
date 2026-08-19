"""The spectral-hole detector must flag missing bands IN SIGNAL, not a
lowered noise floor BETWEEN signal.

Red on the original detector: a clean denoise (voiced content kept to within
1 dB, the floor in the gaps dropped 30 dB) scored ~0.6 and was rejected as a
'spectral hole artifact' — the detector was measuring the cleanup it exists
to protect. Measured on a real DJI field recording, 2026-08-19.
"""

from typing import Any

import numpy as np
import scipy.signal

from hawavoclean.guard.signal import check_signal_integrity

SR = 48000


def _speech_like_with_gaps(seed: int = 0) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Voiced bursts separated by silence, sitting on a -40 dBFS noise bed."""
    rng = np.random.default_rng(seed)
    n = SR * 6
    t = np.arange(n) / SR
    harm = np.zeros(n)
    for h in range(1, 30):
        harm += (0.3 / h) * np.sin(2 * np.pi * 160 * h * t)
    gate = ((t % 1.5) < 0.7).astype(np.float64)  # 0.7 s on, 0.8 s off
    voiced = harm * gate * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t))
    noise = 0.01 * rng.standard_normal(n)
    return np.asarray(voiced + noise, dtype=np.float32)


def _clean_denoise(
    x: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Ideal denoiser stand-in: full content kept, floor in the gaps -30 dB."""
    n = len(x)
    t = np.arange(n) / SR
    gate = (t % 1.5) < 0.7
    y = x.copy()
    y[~gate] *= 10 ** (-30 / 20)
    return np.asarray(y, dtype=np.float32)


def _real_hole(x: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    """A genuine artifact: the 1-3 kHz band wiped out of the SIGNAL itself."""
    sos = scipy.signal.butter(8, [1000.0, 3000.0], btype="bandstop", fs=SR, output="sos")
    return np.asarray(scipy.signal.sosfiltfilt(sos, x), dtype=np.float32)


def test_clean_denoise_is_not_a_spectral_hole() -> None:
    x = _speech_like_with_gaps()
    y = _clean_denoise(x)
    res = check_signal_integrity(x, y, SR, spectral_hole_thresh=0.10)
    assert res.spectral_hole_score < 0.10, (
        f"a clean floor drop in the gaps was scored as a hole "
        f"({res.spectral_hole_score:.3f}); the detector is measuring the denoising "
        f"it exists to protect"
    )
    assert not any("hole" in r.lower() for r in res.failure_reasons), res.failure_reasons


def test_real_band_wipe_in_signal_is_still_caught() -> None:
    x = _speech_like_with_gaps()
    y = _real_hole(x)
    res = check_signal_integrity(x, y, SR, spectral_hole_thresh=0.10, min_hf_preservation_ratio=0.0)
    assert res.spectral_hole_score >= 0.10, (
        f"a 1-3 kHz wipe inside the signal was NOT detected (score {res.spectral_hole_score:.3f})"
    )
