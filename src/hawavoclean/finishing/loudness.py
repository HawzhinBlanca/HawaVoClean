"""ITU-R BS.1770-4 / EBU R128 loudness measurement and static gain calculation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyloudnorm as pyln


@dataclass(frozen=True)
class LoudnessMeasurement:
    """Standardized BS.1770 loudness and peak metrics."""

    integrated_lufs: float
    sample_peak_dbfs: float
    true_peak_dbtp: float


def measure_loudness_and_peaks(
    waveform: np.ndarray[Any, np.dtype[np.float32]],  # shape (channels, samples)
    sample_rate: int,
) -> LoudnessMeasurement:
    """Measure integrated LUFS, loudness range, sample peak, and oversampled true peak."""
    channels, samples = waveform.shape
    if samples < sample_rate * 0.4:
        # File is too short for standard gating (<400ms)
        sample_peak = float(np.max(np.abs(waveform)))
        sp_db = float(20.0 * np.log10(sample_peak + 1e-9))
        return LoudnessMeasurement(
            integrated_lufs=-70.0 if sample_peak < 1e-4 else sp_db,
            sample_peak_dbfs=sp_db,
            true_peak_dbtp=sp_db,
        )

    # Transpose for pyloudnorm (samples, channels)
    data_for_meter = waveform.T

    # 1. Integrated LUFS
    meter = pyln.Meter(sample_rate)
    try:
        integrated_lufs = float(meter.integrated_loudness(data_for_meter))
        if np.isneginf(integrated_lufs) or np.isnan(integrated_lufs):
            integrated_lufs = -70.0
    except Exception:
        integrated_lufs = -70.0

    # 2. Sample peak
    sample_peak = float(np.max(np.abs(waveform)))
    sample_peak_dbfs = float(20.0 * np.log10(sample_peak + 1e-9))

    # 3. 4x Oversampled True Peak calculation
    # Resample 4x to capture inter-sample peaks
    from scipy.signal import resample_poly

    oversampled = resample_poly(waveform, up=4, down=1, axis=-1)
    true_peak = float(np.max(np.abs(oversampled)))
    true_peak_dbtp = float(20.0 * np.log10(true_peak + 1e-9))

    return LoudnessMeasurement(
        integrated_lufs=integrated_lufs,
        sample_peak_dbfs=sample_peak_dbfs,
        true_peak_dbtp=true_peak_dbtp,
    )


def compute_static_master_gain(
    measured_lufs: float,
    target_lufs: float,
    current_true_peak_dbtp: float,
    true_peak_ceiling_dbtp: float = -1.0,
    max_limiter_reduction_db: float = 2.5,
) -> float:
    """Calculate single static gain (dB) required to hit target LUFS while respecting limiter headroom."""
    if measured_lufs <= -69.0:
        return 0.0

    needed_gain_db = target_lufs - measured_lufs

    # Projected true peak under needed gain
    projected_peak_dbtp = current_true_peak_dbtp + needed_gain_db

    # Maximum allowable projected peak before exceeding limiter reduction cap
    max_allowable_projected_peak = true_peak_ceiling_dbtp + max_limiter_reduction_db

    if projected_peak_dbtp > max_allowable_projected_peak:
        # Back off static gain to prevent crushing peaks beyond limiter budget
        needed_gain_db = max_allowable_projected_peak - current_true_peak_dbtp

    return float(needed_gain_db)
