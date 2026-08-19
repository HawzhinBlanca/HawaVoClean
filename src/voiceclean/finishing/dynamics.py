"""Level riding and gentle speech compression with strict gain reduction bounds."""

from typing import Any

import numpy as np


def apply_dialogue_leveler(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    target_rms_db: float = -20.0,
    max_gain_reduction_db: float = 3.0,
    max_gain_boost_db: float = 2.0,
    window_s: float = 0.40,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
    """Apply gentle RMS level riding to smooth extreme vocal volume fluctuations."""
    n = len(waveform)
    if n < 128:
        return waveform.copy(), 0.0

    win_samples = int(round(sample_rate * window_s))
    hop_samples = win_samples // 4
    num_frames = max(1, (n - win_samples) // hop_samples + 1)

    frame_gains_db = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_samples
        chunk = waveform[start : start + win_samples]
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        rms_db = 20.0 * np.log10(rms)

        if rms_db < -45.0:
            # Silence / room tone -> unity gain
            frame_gains_db[i] = 0.0
        else:
            diff = target_rms_db - rms_db
            # Apply soft compression ratio (2:1)
            applied_db = diff * 0.35
            frame_gains_db[i] = np.clip(applied_db, -max_gain_reduction_db, max_gain_boost_db)

    # Smooth gain changes
    sample_indices = np.linspace(0, num_frames - 1, n)
    smooth_gains_db = np.interp(sample_indices, np.arange(num_frames), frame_gains_db)
    linear_gains = 10.0 ** (smooth_gains_db / 20.0)

    leveled = waveform * linear_gains
    max_gr = float(np.max(np.abs(smooth_gains_db)))

    return np.ascontiguousarray(leveled, dtype=np.float32), max_gr
