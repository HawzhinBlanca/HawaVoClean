"""Conservative split-band de-esser with envelope smoothing and bounded gain reduction."""

import math
from typing import Any

import numpy as np
import scipy.signal


def apply_split_band_deesser(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    crossover_hz: float = 5500.0,
    threshold_db: float = -22.0,
    ratio: float = 3.0,
    max_reduction_db: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 50.0,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
    """Split-band dynamic de-esser with 10ms/50ms envelope smoothing and level-relative threshold."""
    n = len(waveform)
    if n < 128 or (sample_rate / 2.0) <= crossover_hz:
        return waveform.copy(), 0.0

    # 4th-order Linkwitz-Riley crossover (2nd-order Butterworth applied zero-phase via sosfiltfilt)
    sos_lp = scipy.signal.butter(2, crossover_hz, btype="lowpass", fs=sample_rate, output="sos")
    sos_hp = scipy.signal.butter(2, crossover_hz, btype="highpass", fs=sample_rate, output="sos")

    x64 = waveform.astype(np.float64, copy=False)
    low_band = np.asarray(scipy.signal.sosfiltfilt(sos_lp, x64), dtype=np.float64)
    high_band = np.asarray(scipy.signal.sosfiltfilt(sos_hp, x64), dtype=np.float64)

    # Frame-level RMS detection on high band (5 ms hop, 10 ms window)
    hop = max(1, int(round(sample_rate * 0.005)))
    win = max(1, int(round(sample_rate * 0.010)))
    num_frames = max(1, (n + hop - 1) // hop)

    frame_rms_db = np.empty(num_frames, dtype=np.float64)
    low_frame_rms_db = np.empty(num_frames, dtype=np.float64)
    frame_centers = np.empty(num_frames, dtype=np.float64)
    for i in range(num_frames):
        s = i * hop
        e = min(n, s + win)
        chunk_hi = high_band[s:e]
        chunk_lo = low_band[s:e]
        frame_rms_db[i] = 20.0 * math.log10(float(np.sqrt(np.mean(chunk_hi * chunk_hi) + 1e-12)))
        low_frame_rms_db[i] = 20.0 * math.log10(
            float(np.sqrt(np.mean(chunk_lo * chunk_lo) + 1e-12))
        )
        frame_centers[i] = 0.5 * float(s + e - 1)

    # Adapt threshold relative to active low-band speech RMS so quiet voice memos
    # (e.g. -34 LUFS) are de-essed cleanly while pure quiet HF noise stays untouched.
    eff_threshold_db = float(threshold_db)
    speech_p85_db = float(np.percentile(low_frame_rms_db, 85.0))
    if speech_p85_db > -52.0:
        eff_threshold_db = min(eff_threshold_db, max(-56.0, speech_p85_db - 2.0))

    # Instantaneous target gain reduction in dB, capped to max_reduction_db
    raw_gr_db = np.zeros(num_frames, dtype=np.float64)
    over_thresh = frame_rms_db > eff_threshold_db
    raw_gr_db[over_thresh] = (frame_rms_db[over_thresh] - eff_threshold_db) * (1.0 - 1.0 / ratio)
    raw_gr_db = np.clip(raw_gr_db, 0.0, max_reduction_db)

    # Real asymmetric exponential envelope smoothing (default 10 ms attack / 50 ms release)
    hop_s = float(hop) / float(sample_rate)
    alpha_attack = math.exp(-hop_s / max(attack_ms / 1000.0, 1e-4)) if attack_ms > 0.0 else 0.0
    alpha_release = math.exp(-hop_s / max(release_ms / 1000.0, 1e-4)) if release_ms > 0.0 else 0.0
    gr_db = np.empty(num_frames, dtype=np.float64)
    env = 0.0
    for i in range(num_frames):
        target = float(raw_gr_db[i])
        if target > env:
            env = alpha_attack * env + (1.0 - alpha_attack) * target
        else:
            env = alpha_release * env + (1.0 - alpha_release) * target
        gr_db[i] = env
    gr_db = np.clip(gr_db, 0.0, max_reduction_db)

    # Interpolate gain reduction curve back to exact sample grid using true frame centers
    sample_gr_db = np.interp(np.arange(n, dtype=np.float64), frame_centers, gr_db)
    linear_gain = np.power(10.0, -sample_gr_db / 20.0)

    processed = low_band + high_band * linear_gain

    max_gr_applied = float(np.max(gr_db))
    return np.ascontiguousarray(processed, dtype=np.float32), max_gr_applied
