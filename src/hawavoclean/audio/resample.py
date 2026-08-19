"""High-quality band-limited audio resampling with exact sample count preservation."""

from math import gcd
from typing import Any

import numpy as np
import scipy.signal


def resample_audio(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    orig_sr: int,
    target_sr: int,
    target_samples: int | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Resample 1D or 2D audio array between sample rates with band-limiting and delay compensation.

    If target_samples is provided, the output length is strictly padded or truncated to target_samples.
    """
    if orig_sr == target_sr:
        if target_samples is not None:
            if waveform.shape[-1] < target_samples:
                pad_width = [(0, 0)] * (waveform.ndim - 1) + [
                    (0, target_samples - waveform.shape[-1])
                ]
                return np.pad(waveform, pad_width, mode="constant").astype(np.float32)
            elif waveform.shape[-1] > target_samples:
                return waveform[..., :target_samples].astype(np.float32)
        return waveform.astype(np.float32)

    # Calculate integer up/down ratio
    common_gcd = gcd(orig_sr, target_sr)
    up = target_sr // common_gcd
    down = orig_sr // common_gcd

    # Use scipy polyphase filtering
    if waveform.ndim == 1:
        resampled = scipy.signal.resample_poly(waveform, up, down, axis=0)
    else:
        resampled = scipy.signal.resample_poly(waveform, up, down, axis=-1)

    expected_len = (
        target_samples
        if target_samples is not None
        else int(round(waveform.shape[-1] * target_sr / orig_sr))
    )

    actual_len = resampled.shape[-1]
    if actual_len < expected_len:
        pad_width = [(0, 0)] * (resampled.ndim - 1) + [(0, expected_len - actual_len)]
        resampled = np.pad(resampled, pad_width, mode="constant")
    elif actual_len > expected_len:
        resampled = resampled[..., :expected_len]

    return np.ascontiguousarray(resampled, dtype=np.float32)
