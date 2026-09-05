"""ITU-R BS.1770-4 / EBU R128 loudness measurement and static gain calculation."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyloudnorm as pyln
import scipy.signal

from hawavoclean.finishing.truepeak import true_peak_linear
from hawavoclean.runtime import evict_memmap_pages

LOUDNESS_BLOCK_SIZE_S = 0.4
LOUDNESS_BLOCK_OVERLAP = 0.75
LOUDNESS_ABSOLUTE_GATE_LUFS = -70.0
LOUDNESS_RELATIVE_GATE_LU = -10.0
LOUDNESS_CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)
LOUDNESS_CHUNK_SAMPLES = 1 << 20


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


def _loudness_block_bounds(j: int, sample_rate: int) -> tuple[int, int]:
    """Match pyloudnorm's floating-point operand order at every block edge."""
    step = 1.0 - LOUDNESS_BLOCK_OVERLAP
    return (
        int(LOUDNESS_BLOCK_SIZE_S * (j * step) * sample_rate),
        int(LOUDNESS_BLOCK_SIZE_S * (j * step + 1) * sample_rate),
    )


class StreamingLoudnessMeter:
    """Exact BS.1770 block reduction with bounded sample storage.

    Only the per-400 ms block energies survive a push. A three-hour recording
    contributes roughly 108,000 small block records; PCM and K-weighted audio
    are released after each chunk. Filter state is carried between chunks and
    float32 quantisation between pyloudnorm's two biquads is preserved.
    """

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.total = 0
        self.sample_peak = 0.0
        self.sum_squares = np.zeros(max(self.channels, 1), dtype=np.float64)
        self._blocks: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        self._weighted_buffer = np.empty((max(self.channels, 1), 0), dtype=np.float32)
        self._weighted_offset = 0
        self._next_block = 0
        self._stages: list[
            tuple[
                np.ndarray[Any, np.dtype[np.float64]],
                np.ndarray[Any, np.dtype[np.float64]],
                float,
                list[np.ndarray[Any, np.dtype[np.float64]]],
            ]
        ] = []
        self.supported = 0 < self.channels <= len(LOUDNESS_CHANNEL_GAINS)
        if self.supported:
            meter = pyln.Meter(self.sample_rate)
            for stage in meter._filters.values():  # noqa: SLF001 -- no public coefficients
                b = np.asarray(stage.b, dtype=np.float64)
                a = np.asarray(stage.a, dtype=np.float64)
                order = max(len(a), len(b)) - 1
                zi = [np.zeros(order, dtype=np.float64) for _ in range(self.channels)]
                self._stages.append((b, a, float(stage.passband_gain), zi))

    def push(self, data: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        if data.ndim != 2 or int(data.shape[0]) != self.channels:
            raise ValueError(
                f"Loudness chunk has shape {data.shape}; expected ({self.channels}, samples)"
            )
        n = int(data.shape[1])
        if n == 0:
            return
        if not np.all(np.isfinite(data)):
            raise ValueError("Loudness chunk contains NaN or Infinite samples")
        self.sample_peak = max(self.sample_peak, float(np.max(np.abs(data))))
        wide = data.astype(np.float64, copy=False)
        self.sum_squares += np.sum(wide * wide, axis=1)
        offset = self.total
        self.total += n
        if not self.supported:
            return

        weighted = data
        for b, a, gain, zi in self._stages:
            out = np.empty_like(weighted)
            for channel in range(self.channels):
                filtered, zi[channel] = scipy.signal.lfilter(
                    b, a, weighted[channel], zi=zi[channel]
                )
                out[channel] = (gain * filtered).astype(np.float32)
            weighted = out
        if self._weighted_buffer.shape[1]:
            weighted = np.concatenate((self._weighted_buffer, weighted), axis=1)
        else:
            self._weighted_offset = offset
        available_end = self._weighted_offset + int(weighted.shape[1])
        while True:
            lo, hi = _loudness_block_bounds(self._next_block, self.sample_rate)
            if hi > available_end:
                break
            local_lo = lo - self._weighted_offset
            local_hi = hi - self._weighted_offset
            segment = weighted[:, local_lo:local_hi]
            # Match pyloudnorm exactly: square in float32, then call np.sum
            # on the complete 400 ms block. Summing chunk partials in float64
            # was accurate but moved the last 1e-7 LU and could change a
            # float32 static-gain sample at a mastering threshold.
            self._blocks.append(
                np.asarray(
                    [
                        (1.0 / (LOUDNESS_BLOCK_SIZE_S * self.sample_rate))
                        * np.sum(np.square(segment[channel]))
                        for channel in range(self.channels)
                    ],
                    dtype=np.float64,
                )
            )
            self._next_block += 1

        next_lo, _ = _loudness_block_bounds(self._next_block, self.sample_rate)
        keep_from = max(0, next_lo - self._weighted_offset)
        self._weighted_buffer = np.array(weighted[:, keep_from:], dtype=np.float32, copy=True)
        self._weighted_offset += keep_from

    def _integrated_lufs(self) -> float:
        if not self.supported:
            return -70.0
        step = 1.0 - LOUDNESS_BLOCK_OVERLAP
        duration_s = self.total / self.sample_rate
        n_blocks = int(
            np.round((duration_s - LOUDNESS_BLOCK_SIZE_S) / (LOUDNESS_BLOCK_SIZE_S * step)) + 1
        )
        if n_blocks <= 0:
            return -70.0
        z = np.zeros((self.channels, n_blocks), dtype=np.float64)
        for j in range(min(n_blocks, len(self._blocks))):
            z[:, j] = self._blocks[j]
        gains = np.asarray(LOUDNESS_CHANNEL_GAINS[: self.channels], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            loud = -0.691 + 10.0 * np.log10(gains @ z)
            above_absolute = loud >= LOUDNESS_ABSOLUTE_GATE_LUFS
            if not bool(np.any(above_absolute)):
                return -70.0
            z_gated = z[:, above_absolute].mean(axis=1)
            relative = -0.691 + 10.0 * np.log10(float(gains @ z_gated)) + LOUDNESS_RELATIVE_GATE_LU
            keep = (loud > relative) & (loud > LOUDNESS_ABSOLUTE_GATE_LUFS)
            if not bool(np.any(keep)):
                return -70.0
            result = float(-0.691 + 10.0 * np.log10(float(gains @ z[:, keep].mean(axis=1))))
        return -70.0 if math.isnan(result) or math.isinf(result) else result

    def finish(self, true_peak: float) -> LoudnessMeasurement:
        peak_db = float(20.0 * np.log10(self.sample_peak + 1e-9))
        if self.total < self.sample_rate * LOUDNESS_BLOCK_SIZE_S:
            if self.sample_peak < 1e-4:
                return LoudnessMeasurement(-70.0, peak_db, peak_db)
            mean_sq = float(np.sum(self.sum_squares / max(self.total, 1)))
            return LoudnessMeasurement(
                float(-0.691 + 10.0 * np.log10(mean_sq + 1e-20)),
                peak_db,
                float(20.0 * np.log10(true_peak + 1e-9)),
            )
        return LoudnessMeasurement(
            self._integrated_lufs(),
            peak_db,
            float(20.0 * np.log10(true_peak + 1e-9)),
        )


def measure_loudness_and_peaks_streaming(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    *,
    chunk_samples: int = LOUDNESS_CHUNK_SAMPLES,
) -> LoudnessMeasurement:
    """Measure a memory-mapped recording with bounded PCM allocations."""
    if waveform.ndim != 2:
        raise ValueError(f"Waveform must have shape (channels, samples), got {waveform.shape}")
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")
    meter = StreamingLoudnessMeter(sample_rate, int(waveform.shape[0]))
    samples = int(waveform.shape[1])
    for start in range(0, samples, chunk_samples):
        end = min(samples, start + chunk_samples)
        meter.push(waveform[:, start:end])
        evict_memmap_pages(waveform, start, end)
    return meter.finish(true_peak_linear(waveform, factor=4))
