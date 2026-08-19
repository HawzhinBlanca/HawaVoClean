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
        # Too short for BS.1770 gating (<400 ms). Use the ungated mean-square
        # loudness (-0.691 + 10 log10 of channel-summed mean square), which
        # is what the gated measure converges to for short steady content —
        # NOT the sample peak, which sat ~9 dB higher and produced a gain
        # jump at the 400 ms boundary. True peak is still oversampled.
        from hawavoclean.finishing.truepeak import true_peak_linear

        sample_peak = float(np.max(np.abs(waveform))) if samples else 0.0
        sp_db = float(20.0 * np.log10(sample_peak + 1e-9))
        if sample_peak < 1e-4:
            return LoudnessMeasurement(
                integrated_lufs=-70.0, sample_peak_dbfs=sp_db, true_peak_dbtp=sp_db
            )
        mean_sq = float(np.sum(np.mean(waveform.astype(np.float64) ** 2, axis=1)))
        ungated_lufs = float(-0.691 + 10.0 * np.log10(mean_sq + 1e-20))
        tp = true_peak_linear(waveform, factor=4)
        return LoudnessMeasurement(
            integrated_lufs=ungated_lufs,
            sample_peak_dbfs=sp_db,
            true_peak_dbtp=float(20.0 * np.log10(tp + 1e-9)),
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

    # 3. 4x oversampled true peak, computed chunk-wise (memory-bounded)
    from hawavoclean.finishing.truepeak import true_peak_linear

    true_peak = true_peak_linear(waveform, factor=4)
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
