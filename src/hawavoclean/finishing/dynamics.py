"""Level riding and gentle speech compression with strict gain reduction bounds."""

import math
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
    """Apply gentle speech-relative RMS level riding with soft-knee gating and exact alignment."""
    n = len(waveform)
    if n < 128:
        return waveform.copy(), 0.0

    win_samples = max(16, int(round(sample_rate * window_s)))
    hop_samples = max(4, win_samples // 4)
    num_frames = max(1, (n - win_samples) // hop_samples + 1)

    x64 = waveform.astype(np.float64, copy=False)
    frame_rms_db = np.empty(num_frames, dtype=np.float64)
    frame_centers = np.empty(num_frames, dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_samples
        end = min(n, start + win_samples)
        chunk = x64[start:end]
        rms = math.sqrt(float(np.mean(chunk * chunk)) + 1e-12)
        frame_rms_db[i] = 20.0 * math.log10(rms)
        frame_centers[i] = 0.5 * float(start + end - 1)

    # Measure active speech reference so quiet recordings (e.g. -34 LUFS voice memos)
    # get proper relative leveling and soft-knee silence gating instead of being ignored.
    loud_ref_db = float(np.percentile(frame_rms_db, 85.0))
    if loud_ref_db < -62.0:
        return waveform.copy(), 0.0

    gate_db = min(-45.0, max(-62.0, loud_ref_db - 22.0))
    knee_db = 6.0
    active_mask = frame_rms_db >= gate_db
    if bool(np.any(active_mask)):
        median_active_db = float(np.median(frame_rms_db[active_mask]))
        eff_target_db = (
            median_active_db if median_active_db < target_rms_db - 6.0 else float(target_rms_db)
        )
    else:
        eff_target_db = float(target_rms_db)

    frame_gains_db = np.zeros(num_frames, dtype=np.float64)
    for i in range(num_frames):
        rms_db = float(frame_rms_db[i])
        if rms_db <= gate_db - knee_db:
            gate_weight = 0.0
        elif rms_db >= gate_db:
            gate_weight = 1.0
        else:
            frac = (rms_db - (gate_db - knee_db)) / knee_db
            gate_weight = 0.5 * (1.0 - math.cos(math.pi * frac))

        if gate_weight > 0.0:
            diff = eff_target_db - rms_db
            applied_db = float(np.clip(diff * 0.35, -max_gain_reduction_db, max_gain_boost_db))
            frame_gains_db[i] = applied_db * gate_weight

    # Interpolate smooth gain changes aligned to exact frame centers
    smooth_gains_db = np.interp(np.arange(n, dtype=np.float64), frame_centers, frame_gains_db)
    linear_gains = np.power(10.0, smooth_gains_db / 20.0)

    leveled = x64 * linear_gains
    max_gr = float(np.max(np.abs(smooth_gains_db)))

    return np.ascontiguousarray(leveled, dtype=np.float32), max_gr
