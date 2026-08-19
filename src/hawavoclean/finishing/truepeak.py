"""Memory-bounded oversampled true-peak measurement.

Oversampling a whole file at 4x/8x in float64 costs tens of bytes per input
sample per copy; on long recordings that reached gigabytes. These helpers
process fixed-size chunks with overlap (so the polyphase filter's edge
transient never lands inside a kept region) and keep memory proportional to
the chunk, not the file.
"""

from typing import Any

import numpy as np
import scipy.signal

CHUNK = 1 << 20  # 1,048,576 samples per chunk (~22 s at 48 kHz)
EDGE = 4096  # overlap on each side; comfortably beyond resample_poly's FIR half-length


def oversampled_peak_envelope(
    waveform: np.ndarray[Any, np.dtype[np.float32]],  # (channels, samples)
    factor: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Per-sample max-abs over the `factor`x oversampled signal, across channels.

    Returns an array of length `samples`: for each original sample i, the
    maximum absolute oversampled value in its block [i*factor, (i+1)*factor).
    Computed chunk-wise in float32.
    """
    channels, samples = waveform.shape
    out = np.zeros(samples, dtype=np.float32)
    if samples == 0:
        return out
    start = 0
    while start < samples:
        end = min(samples, start + CHUNK)
        lo = max(0, start - EDGE)
        hi = min(samples, end + EDGE)
        piece = waveform[:, lo:hi].astype(np.float32, copy=False)
        over = scipy.signal.resample_poly(piece, up=factor, down=1, axis=-1)
        env = np.max(np.abs(over), axis=0)  # (len(piece)*factor,)
        # Fold to one value per original sample, then keep only the core.
        core_lo = (start - lo) * factor
        core_hi = core_lo + (end - start) * factor
        core = env[core_lo:core_hi]
        pad = (-len(core)) % factor
        if pad:
            core = np.pad(core, (0, pad), mode="edge")
        out[start:end] = core.reshape(-1, factor).max(axis=1)[: end - start]
        start = end
    return out


def true_peak_linear(waveform: np.ndarray[Any, np.dtype[np.float32]], factor: int = 4) -> float:
    """Scalar oversampled true peak (linear), memory-bounded."""
    if waveform.size == 0:
        return 0.0
    return float(np.max(oversampled_peak_envelope(waveform, factor)))
