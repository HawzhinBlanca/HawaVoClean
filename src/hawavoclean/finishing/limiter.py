"""Transparent look-ahead true-peak limiter.

Correctness contract: the output's true peak (8x oversampled, float64) is at
or below the configured ceiling, with no hard clipping anywhere. The gain
envelope anticipates each peak across the full lookahead window (a sliding
minimum, not a shift), ramps in over the lookahead, and releases smoothly.
A final verified trim guarantees the ceiling; if it cannot, the limiter
raises instead of silently clipping.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal

from hawavoclean.errors import OutputValidationError
from hawavoclean.finishing.truepeak import oversampled_peak_envelope, true_peak_linear


@dataclass(frozen=True)
class LimiterResult:
    """Limiter output waveform and diagnostic statistics."""

    limited_waveform: np.ndarray[Any, np.dtype[np.float32]]
    max_gain_reduction_db: float
    ceiling_dbtp: float
    gain_envelope: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )


def _true_peak_8x(waveform: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    """8x oversampled true peak across all channels, memory-bounded."""
    return true_peak_linear(waveform, factor=8)


def _slope_limited_min_envelope(
    gain: np.ndarray[Any, np.dtype[np.float32]], lookahead: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Anticipating gain envelope: g[i] = min_k(gain[i+k] + k*delta), k in [0, L].

    This is the slope-limited lower envelope: gain reaches each required
    minimum exactly on time and ramps toward it over at most L samples.
    Computed with the shift-doubling trick in O(n log L).
    """
    n = len(gain)
    if lookahead <= 0 or n == 0:
        return np.asarray(gain, dtype=np.float32)
    delta = np.float32(1.0 / float(lookahead))
    # In-place float32 doubling: each step needs one scratch array of the
    # same size (not two), and no float64 promotion — memory stays at
    # ~2 arrays of n float32 instead of ~4 arrays of n float64.
    env = np.array(gain, dtype=np.float32, copy=True)
    scratch = np.empty_like(env)
    shift = 1
    while shift <= lookahead:
        if shift >= n:
            break  # nothing left to look ahead into
        np.add(env[shift:], np.float32(shift) * delta, out=scratch[: n - shift])
        scratch[n - shift :] = np.inf
        np.minimum(env, scratch, out=env)
        shift *= 2
    np.minimum(env, np.float32(1.0), out=env)
    return env


def apply_lookahead_limiter(
    waveform: np.ndarray[Any, np.dtype[np.float32]],  # shape (channels, samples)
    sample_rate: int,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 50.0,
) -> LimiterResult:
    """Apply a lookahead true-peak limiter enforcing ceiling_dbtp across channels."""
    channels, samples = waveform.shape
    if samples == 0:
        return LimiterResult(waveform.copy(), 0.0, ceiling_dbtp)

    ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))
    lookahead_samples = int(round(sample_rate * (lookahead_ms / 1000.0)))
    release_coeff = float(np.exp(-1.0 / (sample_rate * (release_ms / 1000.0))))

    # 1. 4x oversampled peak envelope, one value per sample, chunk-wise.
    peak_envelope = oversampled_peak_envelope(waveform, factor=4)

    # 2. Required instantaneous gain per sample.
    inst_gain = np.ones(samples, dtype=np.float32)
    over_idx = peak_envelope > ceiling_linear
    inst_gain[over_idx] = ceiling_linear / (peak_envelope[over_idx] + 1e-12)

    # 3. Anticipating envelope: sliding minimum over the lookahead window,
    # then a slope-limited ramp so the reduction arrives smoothly and on time.
    if lookahead_samples > 0 and samples > 1:
        size = min(lookahead_samples + 1, samples)
        # Look-AHEAD window [i, i+size-1]: origin = -(size//2) is always within
        # scipy's valid range (-(size//2) .. (size-1)//2) for any size parity.
        windowed_min = scipy.ndimage.minimum_filter1d(
            inst_gain, size=size, origin=-(size // 2), mode="nearest"
        )
    else:
        windowed_min = inst_gain
    del peak_envelope, inst_gain  # consumed; free before the next full-size array
    anticipated = _slope_limited_min_envelope(windowed_min, lookahead_samples)
    del windowed_min

    # 4. Asymmetric smoothing: instantaneous attack (already ramped by the
    # envelope), one-pole release. The smoothed gain never exceeds the
    # anticipated envelope, so every sample stays within its required gain.
    # Smoothing is done IN PLACE on `anticipated` (it becomes smooth_gain).
    smooth_gain = anticipated
    current_g = 1.0
    for i in range(samples):
        target = float(smooth_gain[i])
        current_g = target if target < current_g else target + release_coeff * (current_g - target)
        smooth_gain[i] = current_g

    limited = np.multiply(waveform, smooth_gain, dtype=np.float32)

    # 5. Verified ceiling: inter-sample peaks can still exceed the envelope
    # estimate marginally; a single transparent trim closes the gap. No clip.
    tp = _true_peak_8x(limited)
    if tp > ceiling_linear:
        trim = np.float32((ceiling_linear / tp) * (1.0 - 1e-6))
        np.multiply(limited, trim, out=limited)
        np.multiply(smooth_gain, trim, out=smooth_gain)
        tp = _true_peak_8x(limited)
    if tp > ceiling_linear:
        raise OutputValidationError(
            f"Limiter failed to enforce ceiling: true peak {tp:.6f} > {ceiling_linear:.6f}"
        )

    min_gain = float(np.min(smooth_gain))
    max_gr_db = float(-20.0 * np.log10(max(min_gain, 1e-6)))

    return LimiterResult(
        limited_waveform=limited,
        max_gain_reduction_db=max_gr_db,
        ceiling_dbtp=ceiling_dbtp,
        gain_envelope=smooth_gain,
    )
