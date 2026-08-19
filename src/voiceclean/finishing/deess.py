"""Conservative split-band de-esser with bounded maximum gain reduction."""

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
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float]:
    """Split-band dynamic de-esser reducing harsh sibilance with strictly bounded gain reduction."""
    n = len(waveform)
    if n < 128 or (sample_rate / 2.0) <= crossover_hz:
        return waveform.copy(), 0.0

    # 4th-order Linkwitz-Riley crossover (cascaded Butterworth)
    sos_lp = scipy.signal.butter(2, crossover_hz, btype="lowpass", fs=sample_rate, output="sos")
    sos_hp = scipy.signal.butter(2, crossover_hz, btype="highpass", fs=sample_rate, output="sos")

    low_band = scipy.signal.sosfiltfilt(sos_lp, waveform)
    high_band = scipy.signal.sosfiltfilt(sos_hp, waveform)

    # Envelope detection on high band (10ms attack, 50ms release)
    hop = int(round(sample_rate * 0.005))
    win = int(round(sample_rate * 0.010))
    num_frames = max(1, n // hop)

    frame_rms_db = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        chunk = high_band[i * hop : min(n, i * hop + win)]
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        frame_rms_db[i] = 20.0 * np.log10(rms)

    # Compute gain reduction in dB
    gr_db = np.zeros(num_frames, dtype=np.float32)
    over_thresh = frame_rms_db > threshold_db
    gr_db[over_thresh] = (frame_rms_db[over_thresh] - threshold_db) * (1.0 - 1.0 / ratio)
    # Strictly cap gain reduction
    gr_db = np.clip(gr_db, 0.0, max_reduction_db)

    # Interpolate gain reduction curve back to sample level
    sample_indices = np.linspace(0, num_frames - 1, n)
    sample_gr_db = np.interp(sample_indices, np.arange(num_frames), gr_db)
    linear_gain = 10.0 ** (-sample_gr_db / 20.0)

    compressed_high = high_band * linear_gain
    processed = low_band + compressed_high

    max_gr_applied = float(np.max(gr_db))
    return np.ascontiguousarray(processed, dtype=np.float32), max_gr_applied
