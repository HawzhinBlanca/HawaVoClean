"""Transparent look-ahead true-peak limiter with bounded gain reduction."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.signal


@dataclass(frozen=True)
class LimiterResult:
    """Limiter output waveform and diagnostic statistics."""

    limited_waveform: np.ndarray[Any, np.dtype[np.float32]]
    max_gain_reduction_db: float
    ceiling_dbtp: float


def apply_lookahead_limiter(
    waveform: np.ndarray[Any, np.dtype[np.float32]],  # shape (channels, samples)
    sample_rate: int,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 50.0,
) -> LimiterResult:
    """Apply lookahead true-peak limiter enforcing ceiling_dbtp across all channels."""
    channels, samples = waveform.shape
    if samples == 0:
        return LimiterResult(waveform.copy(), 0.0, ceiling_dbtp)

    ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))

    # Lookahead and release parameters
    lookahead_samples = int(round(sample_rate * (lookahead_ms / 1000.0)))
    release_coeff = float(np.exp(-1.0 / (sample_rate * (release_ms / 1000.0))))

    # 1. 4x Oversampling to estimate true peak envelope
    oversampled = scipy.signal.resample_poly(waveform, up=4, down=1, axis=-1)
    # Peak across all channels at each 4x sample point
    max_env_4x = np.max(np.abs(oversampled), axis=0)

    # Downsample envelope back to original rate by taking max across each 4-sample block
    pad_rem = (4 - (len(max_env_4x) % 4)) % 4
    if pad_rem > 0:
        max_env_4x = np.pad(max_env_4x, (0, pad_rem), mode="constant")
    env_blocks = max_env_4x.reshape(-1, 4)
    peak_envelope = np.max(env_blocks, axis=1)[:samples]

    if len(peak_envelope) < samples:
        peak_envelope = np.pad(peak_envelope, (0, samples - len(peak_envelope)), mode="edge")

    # 2. Compute required instantaneous gain
    inst_gain = np.ones(samples, dtype=np.float32)
    over_idx = peak_envelope > ceiling_linear
    inst_gain[over_idx] = ceiling_linear / (peak_envelope[over_idx] + 1e-12)

    # 3. Apply lookahead shift (shift gain earlier)
    if lookahead_samples > 0:
        shifted_gain = np.pad(inst_gain[lookahead_samples:], (0, lookahead_samples), mode="edge")
    else:
        shifted_gain = inst_gain

    # 4. Smooth with 1-pole release filter
    smooth_gain = np.ones(samples, dtype=np.float32)
    current_g = 1.0

    for i in range(samples):
        target = shifted_gain[i]
        current_g = target if target < current_g else target + release_coeff * (current_g - target)
        smooth_gain[i] = current_g

    # 5. Apply gain to all channels
    limited = (waveform * smooth_gain).astype(np.float32)
    # Hard safety clamp to ceiling linear
    limited = np.clip(limited, -ceiling_linear, ceiling_linear)

    min_gain = float(np.min(smooth_gain))
    max_gr_db = float(-20.0 * np.log10(max(min_gain, 1e-6)))

    return LimiterResult(
        limited_waveform=limited,
        max_gain_reduction_db=max_gr_db,
        ceiling_dbtp=ceiling_dbtp,
    )
