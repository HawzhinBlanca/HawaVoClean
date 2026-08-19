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

from voiceclean.errors import OutputValidationError


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
    """8x oversampled true peak in float64 across all channels."""
    over = scipy.signal.resample_poly(waveform.astype(np.float64), up=8, down=1, axis=-1)
    return float(np.max(np.abs(over)))


def _slope_limited_min_envelope(
    gain: np.ndarray[Any, np.dtype[np.float32]], lookahead: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Anticipating gain envelope: g[i] = min_k(gain[i+k] + k*delta), k in [0, L].

    This is the slope-limited lower envelope: gain reaches each required
    minimum exactly on time and ramps toward it over at most L samples.
    Computed with the shift-doubling trick in O(n log L).
    """
    if lookahead <= 0:
        return np.asarray(gain, dtype=np.float32)
    delta = 1.0 / float(lookahead)
    env = gain.astype(np.float64)
    shift = 1
    while shift <= lookahead:
        shifted = np.concatenate([env[shift:], np.full(shift, np.inf)]) + shift * delta
        env = np.minimum(env, shifted)
        shift *= 2
    return np.asarray(np.minimum(env, 1.0), dtype=np.float32)


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

    # 1. 4x oversampled peak envelope, folded back to one value per sample.
    oversampled = scipy.signal.resample_poly(waveform, up=4, down=1, axis=-1)
    max_env_4x = np.max(np.abs(oversampled), axis=0)
    pad_rem = (4 - (len(max_env_4x) % 4)) % 4
    if pad_rem > 0:
        max_env_4x = np.pad(max_env_4x, (0, pad_rem), mode="constant")
    peak_envelope = np.max(max_env_4x.reshape(-1, 4), axis=1)[:samples]
    if len(peak_envelope) < samples:
        peak_envelope = np.pad(peak_envelope, (0, samples - len(peak_envelope)), mode="edge")

    # 2. Required instantaneous gain per sample.
    inst_gain = np.ones(samples, dtype=np.float32)
    over_idx = peak_envelope > ceiling_linear
    inst_gain[over_idx] = ceiling_linear / (peak_envelope[over_idx] + 1e-12)

    # 3. Anticipating envelope: sliding minimum over the lookahead window,
    # then a slope-limited ramp so the reduction arrives smoothly and on time.
    if lookahead_samples > 0:
        size = lookahead_samples + 1
        windowed_min = scipy.ndimage.minimum_filter1d(
            inst_gain, size=size, origin=size // 2, mode="nearest"
        )
    else:
        windowed_min = inst_gain
    anticipated = _slope_limited_min_envelope(windowed_min, lookahead_samples)

    # 4. Asymmetric smoothing: instantaneous attack (already ramped by the
    # envelope), one-pole release. The smoothed gain never exceeds the
    # anticipated envelope, so every sample stays within its required gain.
    smooth_gain = np.ones(samples, dtype=np.float32)
    current_g = 1.0
    for i in range(samples):
        target = float(anticipated[i])
        current_g = target if target < current_g else target + release_coeff * (current_g - target)
        smooth_gain[i] = current_g

    limited = (waveform * smooth_gain).astype(np.float32)

    # 5. Verified ceiling: inter-sample peaks can still exceed the envelope
    # estimate marginally; a single transparent trim closes the gap. No clip.
    tp = _true_peak_8x(limited)
    if tp > ceiling_linear:
        trim = (ceiling_linear / tp) * (1.0 - 1e-6)
        limited = (limited * trim).astype(np.float32)
        smooth_gain = (smooth_gain * trim).astype(np.float32)
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
