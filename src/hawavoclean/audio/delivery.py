"""Broadcast delivery loudness, EBU Tech 3342 LRA, BWF/iXML metadata, and multi-format export."""

from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import pyloudnorm as pyln
import scipy.ndimage
import scipy.signal
import soundfile as sf

from hawavoclean.audio.encode import _finalize_deterministic_wav
from hawavoclean.errors import OutputValidationError, PreflightError
from hawavoclean.finishing.limiter import (
    _future_gain_dependency,
    _release_scalar,
    _slope_limited_min_envelope,
    apply_lookahead_limiter,
)
from hawavoclean.finishing.loudness import (
    LOUDNESS_ABSOLUTE_GATE_LUFS,
    LOUDNESS_CHANNEL_GAINS,
    LoudnessMeasurement,
    StreamingLoudnessMeter,
    measure_loudness_and_peaks,
)
from hawavoclean.finishing.truepeak import EDGE as TRUE_PEAK_EDGE
from hawavoclean.finishing.truepeak import oversampled_peak_envelope_window
from hawavoclean.paths import ffmpeg_bin_path
from hawavoclean.platform_fs import is_reparse_or_symlink

Float32Array = np.ndarray[Any, np.dtype[np.float32]]

DeliveryPreset = Literal["match_input", "ebu_r128", "atsc_a85", "podcast", "streaming"]
ExportFormat = Literal["wav24", "wav32f", "flac", "aac", "video_mux"]

EXPORT_FORMAT_EXTENSIONS: dict[ExportFormat, frozenset[str]] = {
    "wav24": frozenset({".wav"}),
    "wav32f": frozenset({".wav"}),
    "flac": frozenset({".flac"}),
    "aac": frozenset({".m4a", ".aac"}),
    "video_mux": frozenset({".mp4", ".mov", ".mkv"}),
}

_WAV_CLASSIC_LIMIT_BYTES = (1 << 32) - 1
_WAV_HEADER_RESERVE_BYTES = 1 << 20
_STREAM_COPY_BYTES = 1 << 20
_STREAM_AUDIO_FRAMES = 1 << 18
_MAX_STATIC_GAIN_DB = 36.0
MAX_METADATA_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB DoS cap per BWF/iXML chunk
_MAX_DS64_CHUNK_BYTES = 1 * 1024 * 1024  # 1 MiB DoS cap for RF64 ds64 chunk
_MAX_WAV_CHUNKS = 4096
_MAX_RF64_SIZE = (1 << 63) - 1
_F32_MAX = float(np.finfo(np.float32).max)

_LRA_BLOCK_SIZE_S = 3.0
_LRA_HOP_SIZE_S = 0.1
_LRA_RELATIVE_GATE_LU = -20.0

PRESERVED_WAV_METADATA_CHUNKS: frozenset[bytes] = frozenset(
    {b"bext", b"iXML", b"axml", b"LIST", b"cue ", b"smpl"}
)
_PROTECTED_WAV_CHUNKS: frozenset[bytes] = frozenset({b"fmt ", b"data", b"ds64", b"fact", b"PEAK"})


@dataclass(frozen=True)
class DeliveryPresetSpec:
    """Broadcast and streaming loudness delivery target specification."""

    name: DeliveryPreset
    target_lufs_stereo: float | None
    target_lufs_mono: float | None
    true_peak_ceiling_dbtp: float | None
    bypass_limiter: bool
    description: str


DELIVERY_PRESETS: dict[DeliveryPreset, DeliveryPresetSpec] = {
    "match_input": DeliveryPresetSpec(
        name="match_input",
        target_lufs_stereo=None,
        target_lufs_mono=None,
        true_peak_ceiling_dbtp=None,
        bypass_limiter=True,
        description="Preserve input integrated loudness with limiter bypassed",
    ),
    "ebu_r128": DeliveryPresetSpec(
        name="ebu_r128",
        target_lufs_stereo=-23.0,
        target_lufs_mono=-23.0,
        true_peak_ceiling_dbtp=-1.0,
        bypass_limiter=False,
        description="EBU R128 broadcast (-23.0 LUFS, -1.0 dBTP)",
    ),
    "atsc_a85": DeliveryPresetSpec(
        name="atsc_a85",
        target_lufs_stereo=-24.0,
        target_lufs_mono=-24.0,
        true_peak_ceiling_dbtp=-2.0,
        bypass_limiter=False,
        description="ATSC A/85 CALM Act broadcast (-24.0 LKFS, -2.0 dBTP)",
    ),
    "podcast": DeliveryPresetSpec(
        name="podcast",
        target_lufs_stereo=-16.0,
        target_lufs_mono=-19.0,
        true_peak_ceiling_dbtp=-1.0,
        bypass_limiter=False,
        description="Podcast delivery (-16.0 LUFS stereo / -19.0 LUFS mono, -1.0 dBTP)",
    ),
    "streaming": DeliveryPresetSpec(
        name="streaming",
        target_lufs_stereo=-14.0,
        target_lufs_mono=-14.0,
        true_peak_ceiling_dbtp=-1.0,
        bypass_limiter=False,
        description="Streaming platform delivery (-14.0 LUFS, -1.0 dBTP)",
    ),
}


def _validate_audio(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
) -> tuple[Float32Array, bool]:
    """Validate audio array and normalize to contiguous 2D (channels, samples) float32."""
    if isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    arr = np.asarray(waveform)
    if arr.ndim not in (1, 2):
        raise ValueError(f"waveform must be 1D or 2D, got shape {arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("waveform contains NaN or Inf values")
    was_1d = arr.ndim == 1
    with np.errstate(over="ignore", invalid="ignore"):
        f32 = np.ascontiguousarray(arr, dtype=np.float32)
    if f32.size > 0 and not np.all(np.isfinite(f32)):
        raise ValueError("waveform contains NaN or Inf values after float32 conversion")
    if was_1d:
        f32 = f32[np.newaxis, :]
    return f32, was_1d


def _scale_to_finite_float32(wave_2d: Float32Array, gain_linear: float) -> Float32Array:
    """Scale float32 audio in float64 and clamp to finite float32 bounds without warnings."""
    scaled = wave_2d.astype(np.float64, copy=False) * gain_linear
    return np.ascontiguousarray(np.clip(scaled, -_F32_MAX, _F32_MAX), dtype=np.float32)


def _restore_shape(wave_2d: Float32Array, was_1d: bool) -> Float32Array:
    """Restore original 1D/2D shape as contiguous float32."""
    out = wave_2d[0] if was_1d else wave_2d
    return np.ascontiguousarray(out, dtype=np.float32)


def compute_loudness_range_lu(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
) -> float:
    """Compute EBU Tech 3342 Loudness Range (LRA, 10th-95th percentile of 3 s blocks).

    Uses ITU-R BS.1770 K-weighting, 3.0 s short-term blocks with 0.1 s hop,
    -70 LUFS absolute gating, and -20 LU relative gating.
    """
    wave_2d, _ = _validate_audio(waveform, sample_rate)
    channels, samples = wave_2d.shape
    block_len = int(round(_LRA_BLOCK_SIZE_S * sample_rate))
    if channels == 0 or samples < block_len or block_len <= 0:
        return 0.0

    # Apply BS.1770 K-weighting filters using pyloudnorm's exact biquad stages.
    meter = pyln.Meter(sample_rate)
    weighted = wave_2d.astype(np.float64, copy=True)
    for stage in meter._filters.values():  # noqa: SLF001 -- matches StreamingLoudnessMeter
        b = np.asarray(stage.b, dtype=np.float64)
        a = np.asarray(stage.a, dtype=np.float64)
        gain = float(stage.passband_gain)
        for ch in range(channels):
            weighted[ch] = gain * scipy.signal.lfilter(b, a, weighted[ch])

    hop_len = max(1, int(round(_LRA_HOP_SIZE_S * sample_rate)))
    starts = np.arange(0, samples - block_len + 1, hop_len, dtype=np.intp)
    if starts.size < 2:
        return 0.0
    ends = starts + block_len

    # Exact O(N) sliding mean-square per channel via cumulative sums.
    cs = np.empty((channels, samples + 1), dtype=np.float64)
    cs[:, 0] = 0.0
    np.cumsum(weighted * weighted, axis=1, out=cs[:, 1:])
    mean_sq = np.maximum(cs[:, ends] - cs[:, starts], 0.0) / float(block_len)

    gains = np.asarray(
        [
            LOUDNESS_CHANNEL_GAINS[ch] if ch < len(LOUDNESS_CHANNEL_GAINS) else 1.0
            for ch in range(channels)
        ],
        dtype=np.float64,
    )
    block_power = gains @ mean_sq

    with np.errstate(divide="ignore", invalid="ignore"):
        block_lufs = -0.691 + 10.0 * np.log10(block_power)

    above_abs = np.isfinite(block_lufs) & (block_lufs >= LOUDNESS_ABSOLUTE_GATE_LUFS)
    if int(np.count_nonzero(above_abs)) < 2:
        return 0.0

    mean_power_abs = float(np.mean(block_power[above_abs]))
    if mean_power_abs <= 0.0:
        return 0.0
    rel_threshold = -0.691 + 10.0 * math.log10(mean_power_abs) + _LRA_RELATIVE_GATE_LU

    keep = above_abs & (block_lufs >= rel_threshold)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0

    kept = block_lufs[keep]
    low_p = float(np.percentile(kept, 10.0))
    high_p = float(np.percentile(kept, 95.0))
    lra = max(0.0, high_p - low_p)
    return 0.0 if not math.isfinite(lra) else lra


def apply_delivery_loudness(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    preset: DeliveryPreset,
    *,
    input_lufs: float | None = None,
) -> tuple[Float32Array, dict[str, float | str]]:
    """Apply preset loudness normalization and true-peak limiting (or match-input bypass)."""
    if preset not in DELIVERY_PRESETS:
        raise ValueError(f"Unsupported delivery preset: {preset!r}")
    if input_lufs is not None and not math.isfinite(input_lufs):
        raise ValueError(f"input_lufs must be finite when provided, got {input_lufs}")

    wave_2d, was_1d = _validate_audio(waveform, sample_rate)
    spec = DELIVERY_PRESETS[preset]
    channels, samples = wave_2d.shape

    if channels == 0 or samples == 0:
        empty_metrics: dict[str, float | str] = {
            "preset": preset,
            "limiter_mode": "bypass" if spec.bypass_limiter else "lookahead",
            "input_lufs": -70.0,
            "target_lufs": -70.0,
            "applied_gain_db": 0.0,
            "integrated_lufs": -70.0,
            "true_peak_dbtp": -180.0,
            "sample_peak_dbfs": -180.0,
            "loudness_range_lu": 0.0,
            "limiter_max_gr_db": 0.0,
            "ceiling_dbtp": (
                "bypass"
                if spec.true_peak_ceiling_dbtp is None
                else float(spec.true_peak_ceiling_dbtp)
            ),
        }
        return _restore_shape(wave_2d.copy(), was_1d), empty_metrics

    pre = measure_loudness_and_peaks(wave_2d, sample_rate)

    if spec.bypass_limiter:
        target_lufs = float(input_lufs) if input_lufs is not None else float(pre.integrated_lufs)
        if input_lufs is None or pre.integrated_lufs <= -69.0 or input_lufs <= -69.0:
            gain_db = 0.0
            processed = wave_2d.copy()
        else:
            gain_db = float(
                np.clip(input_lufs - pre.integrated_lufs, -_MAX_STATIC_GAIN_DB, _MAX_STATIC_GAIN_DB)
            )
            if abs(gain_db) < 1e-6:
                gain_db = 0.0
                processed = wave_2d.copy()
            else:
                gain_linear = float(10.0 ** (gain_db / 20.0))
                processed = _scale_to_finite_float32(wave_2d, gain_linear)

        post = measure_loudness_and_peaks(processed, sample_rate)
        lra = compute_loudness_range_lu(processed, sample_rate)
        metrics: dict[str, float | str] = {
            "preset": preset,
            "limiter_mode": "bypass",
            "input_lufs": float(pre.integrated_lufs),
            "target_lufs": target_lufs,
            "applied_gain_db": gain_db,
            "integrated_lufs": float(post.integrated_lufs),
            "true_peak_dbtp": float(post.true_peak_dbtp),
            "sample_peak_dbfs": float(post.sample_peak_dbfs),
            "loudness_range_lu": float(lra),
            "limiter_max_gr_db": 0.0,
            "ceiling_dbtp": "bypass",
        }
        return _restore_shape(processed, was_1d), metrics

    target = spec.target_lufs_mono if channels == 1 else spec.target_lufs_stereo
    ceiling = spec.true_peak_ceiling_dbtp
    assert target is not None and ceiling is not None

    if pre.integrated_lufs <= -69.0:
        gain_db = 0.0
        gained = wave_2d.copy()
    else:
        gain_db = float(
            np.clip(target - pre.integrated_lufs, -_MAX_STATIC_GAIN_DB, _MAX_STATIC_GAIN_DB)
        )
        gain_linear = float(10.0 ** (gain_db / 20.0))
        gained = _scale_to_finite_float32(wave_2d, gain_linear)

    limiter_res = apply_lookahead_limiter(gained, sample_rate, ceiling_dbtp=ceiling)
    processed = np.ascontiguousarray(limiter_res.limited_waveform, dtype=np.float32)
    post = measure_loudness_and_peaks(processed, sample_rate)
    lra = compute_loudness_range_lu(processed, sample_rate)

    metrics = {
        "preset": preset,
        "limiter_mode": "lookahead",
        "input_lufs": float(pre.integrated_lufs),
        "target_lufs": float(target),
        "applied_gain_db": gain_db,
        "integrated_lufs": float(post.integrated_lufs),
        "true_peak_dbtp": float(post.true_peak_dbtp),
        "sample_peak_dbfs": float(post.sample_peak_dbfs),
        "loudness_range_lu": float(lra),
        "limiter_max_gr_db": float(limiter_res.max_gain_reduction_db),
        "ceiling_dbtp": float(ceiling),
    }
    return _restore_shape(processed, was_1d), metrics


_STREAM_LIMITER_MAX_STEP = 1 << 17
_STREAM_TRUE_PEAK_MAX_STEP = 1 << 17
_DELIVERY_STREAM_SUBTYPES: frozenset[str] = frozenset({"PCM_16", "PCM_24", "PCM_32", "FLOAT"})


def _release_smooth_carried(
    gain: Float32Array,
    release_coeff: float,
    current: float,
) -> float:
    """Sample-exact carried run-walk release smoother matching limiter._release_smooth."""
    n = int(gain.shape[0])
    if n == 0:
        return current
    bits = gain.view(np.uint32)
    changed = bits[1:] != bits[:-1]
    runs = int(np.count_nonzero(changed)) + 1
    if runs * 5 > n:
        return _release_scalar(gain, release_coeff, 0, n, current)

    starts = [0, *(np.flatnonzero(changed) + 1).tolist(), n]
    targets = gain[np.asarray(starts[:-1], dtype=np.intp)].tolist()
    for k, target in enumerate(targets):
        lo = starts[k]
        hi = starts[k + 1]
        if target == 0.0 or target != target:
            current = _release_scalar(gain, release_coeff, lo, hi, current)
            continue
        if current == target:
            continue
        if target < current:
            current = target
            continue
        ramp: list[float] = []
        for _ in range(lo, hi):
            current = target + release_coeff * (current - target)
            if current == target:
                break
            ramp.append(current)
        if ramp:
            gain[lo : lo + len(ramp)] = ramp
    return current


class _StreamingLraMeter:
    """Exact EBU Tech 3342 Loudness Range (LRA) accumulator with bounded memory."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._block_len = int(round(_LRA_BLOCK_SIZE_S * self.sample_rate))
        self._hop_len = max(1, int(round(_LRA_HOP_SIZE_S * self.sample_rate)))
        self._gains = np.asarray(
            [
                LOUDNESS_CHANNEL_GAINS[ch] if ch < len(LOUDNESS_CHANNEL_GAINS) else 1.0
                for ch in range(self.channels)
            ],
            dtype=np.float64,
        )
        self._stages: list[
            tuple[
                np.ndarray[Any, np.dtype[np.float64]],
                np.ndarray[Any, np.dtype[np.float64]],
                float,
                list[np.ndarray[Any, np.dtype[np.float64]]],
            ]
        ] = []
        if self.channels > 0 and self._block_len > 0:
            meter = pyln.Meter(self.sample_rate)
            for stage in meter._filters.values():  # noqa: SLF001 -- matches compute_loudness_range_lu
                b = np.asarray(stage.b, dtype=np.float64)
                a = np.asarray(stage.a, dtype=np.float64)
                order = max(len(a), len(b)) - 1
                zi = [np.zeros(order, dtype=np.float64) for _ in range(self.channels)]
                self._stages.append((b, a, float(stage.passband_gain), zi))
        self._power_tail = np.empty(0, dtype=np.float64)
        self._tail_offset = 0
        self._total = 0
        self._next_start = 0
        self._block_powers: list[np.ndarray[Any, np.dtype[np.float64]]] = []

    def push(self, data: Float32Array) -> None:
        if self.channels <= 0 or self._block_len <= 0:
            return
        n = int(data.shape[1])
        if n == 0:
            return
        weighted = data.astype(np.float64, copy=True)
        for b, a, gain, zi in self._stages:
            for ch in range(self.channels):
                filtered, zi[ch] = scipy.signal.lfilter(b, a, weighted[ch], zi=zi[ch])
                weighted[ch] = gain * filtered
        chunk_power = self._gains @ (weighted * weighted)
        if self._power_tail.size > 0:
            buf = np.concatenate((self._power_tail, chunk_power))
        else:
            buf = chunk_power
            self._tail_offset = self._total
        self._total += n

        available_end = self._tail_offset + int(buf.shape[0])
        if available_end >= self._next_start + self._block_len:
            last_start = available_end - self._block_len
            n_emit = ((last_start - self._next_start) // self._hop_len) + 1
            global_starts = self._next_start + np.arange(n_emit, dtype=np.intp) * self._hop_len
            local_starts = global_starts - self._tail_offset
            local_ends = local_starts + self._block_len
            cs = np.empty(buf.shape[0] + 1, dtype=np.float64)
            cs[0] = 0.0
            np.cumsum(buf, out=cs[1:])
            mean_power = np.maximum(cs[local_ends] - cs[local_starts], 0.0) / float(self._block_len)
            self._block_powers.append(mean_power)
            self._next_start += n_emit * self._hop_len
            keep_from = min(int(buf.shape[0]), max(0, self._next_start - self._tail_offset))
            self._power_tail = np.array(buf[keep_from:], dtype=np.float64, copy=True)
            self._tail_offset += keep_from
        else:
            self._power_tail = np.array(buf, dtype=np.float64, copy=True)

    def finish(self) -> float:
        if not self._block_powers:
            return 0.0
        block_power = np.concatenate(self._block_powers)
        if block_power.size < 2:
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            block_lufs = -0.691 + 10.0 * np.log10(block_power)
        above_abs = np.isfinite(block_lufs) & (block_lufs >= LOUDNESS_ABSOLUTE_GATE_LUFS)
        if int(np.count_nonzero(above_abs)) < 2:
            return 0.0
        mean_power_abs = float(np.mean(block_power[above_abs]))
        if mean_power_abs <= 0.0:
            return 0.0
        rel_threshold = -0.691 + 10.0 * math.log10(mean_power_abs) + _LRA_RELATIVE_GATE_LU
        keep = above_abs & (block_lufs >= rel_threshold)
        if int(np.count_nonzero(keep)) < 2:
            return 0.0
        kept = block_lufs[keep]
        low_p = float(np.percentile(kept, 10.0))
        high_p = float(np.percentile(kept, 95.0))
        lra = max(0.0, high_p - low_p)
        return 0.0 if not math.isfinite(lra) else lra


class _StreamingTruePeakMeter:
    """Bounded-memory exact oversampled true-peak meter with FIR overlap."""

    def __init__(
        self,
        channels: int,
        *,
        factor: int = 4,
        max_step: int = _STREAM_TRUE_PEAK_MAX_STEP,
    ) -> None:
        self._channels = int(channels)
        self._factor = int(factor)
        self._max_step = max(TRUE_PEAK_EDGE + 1, int(max_step))
        self._tail = np.empty((self._channels, 0), dtype=np.float32)
        self._pending = 0
        self._peak = 0.0

    def _push_slice(self, data: Float32Array) -> None:
        n = int(data.shape[1])
        if n == 0:
            return
        work = np.concatenate((self._tail, data), axis=1) if self._tail.shape[1] > 0 else data
        total_work = int(work.shape[1])
        core_start = int(self._tail.shape[1]) - self._pending
        core_end = total_work - TRUE_PEAK_EDGE
        if core_end > core_start:
            env = oversampled_peak_envelope_window(work, self._factor, core_start, core_end)
            if env.size:
                self._peak = max(self._peak, float(np.max(env)))
            keep_from = max(0, core_end - TRUE_PEAK_EDGE)
            self._tail = np.ascontiguousarray(work[:, keep_from:], dtype=np.float32)
            self._pending = total_work - core_end
        else:
            self._tail = np.ascontiguousarray(work, dtype=np.float32)
            self._pending += n

    def push(self, data: Float32Array) -> None:
        n = int(data.shape[1])
        for offset in range(0, n, self._max_step):
            self._push_slice(data[:, offset : min(n, offset + self._max_step)])

    def finish(self) -> float:
        if self._pending > 0 and self._tail.shape[1] > 0:
            total_tail = int(self._tail.shape[1])
            core_start = max(0, total_tail - self._pending)
            env = oversampled_peak_envelope_window(self._tail, self._factor, core_start, total_tail)
            if env.size:
                self._peak = max(self._peak, float(np.max(env)))
            self._pending = 0
            self._tail = np.empty((self._channels, 0), dtype=np.float32)
        return self._peak


def _wav_container_for_frames(channels: int, frames: int, subtype: str) -> Literal["WAV", "RF64"]:
    """Select classic RIFF WAV or RF64 based on estimated data size."""
    bytes_per_sample = 3 if subtype == "PCM_24" else 4
    estimated = channels * frames * bytes_per_sample + _WAV_HEADER_RESERVE_BYTES
    return "RF64" if estimated > _WAV_CLASSIC_LIMIT_BYTES else "WAV"


def measure_delivery_loudness_streaming(
    wav_path: Path,
    *,
    chunk_samples: int = 1 << 19,
) -> tuple[LoudnessMeasurement, float]:
    """Measure integrated LUFS, peaks, and EBU Tech 3342 LRA in one streaming pass."""
    if isinstance(chunk_samples, bool) or chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples!r}")
    resolved = _validate_path_string(wav_path, "wav_path")
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file not found: {resolved}")

    with sf.SoundFile(str(resolved), mode="r") as reader:
        sample_rate = int(reader.samplerate)
        channels = int(reader.channels)
        frames = int(reader.frames)
        if sample_rate <= 0 or channels <= 0 or frames <= 0:
            raise OutputValidationError(f"Audio file is empty or invalid: {resolved}")

        loudness_meter = StreamingLoudnessMeter(sample_rate, channels)
        tp_meter = _StreamingTruePeakMeter(channels, factor=4)
        lra_meter = _StreamingLraMeter(sample_rate, channels)
        step = min(int(chunk_samples), _STREAM_TRUE_PEAK_MAX_STEP)
        while True:
            block = reader.read(frames=step, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            chunk_2d = np.ascontiguousarray(block.T, dtype=np.float32)
            if not np.all(np.isfinite(chunk_2d)):
                raise ValueError("waveform contains NaN or Inf values")
            loudness_meter.push(chunk_2d)
            tp_meter.push(chunk_2d)
            lra_meter.push(chunk_2d)

    meas = loudness_meter.finish(tp_meter.finish())
    lra = lra_meter.finish()
    return meas, lra


def apply_delivery_loudness_streaming(
    master_wav: Path,
    out_wav: Path,
    preset: DeliveryPreset,
    *,
    input_lufs: float | None = None,
    chunk_samples: int = 1 << 19,
) -> dict[str, float | str]:
    """Apply preset loudness normalization and lookahead limiting in bounded RAM.

    Pass 1 streams ``master_wav`` through :class:`StreamingLoudnessMeter`,
    chunked 4x true-peak, and (when bypassing limiting) streaming EBU Tech 3342
    LRA without holding the full recording in memory. Pass 2 streams static
    gain and overlap-buffered lookahead limiting (carrying release state and
    FIR/lookahead context across chunk boundaries) into ``out_wav`` atomically.
    """
    if preset not in DELIVERY_PRESETS:
        raise ValueError(f"Unsupported delivery preset: {preset!r}")
    if input_lufs is not None and (isinstance(input_lufs, bool) or not math.isfinite(input_lufs)):
        raise ValueError(f"input_lufs must be finite when provided, got {input_lufs}")
    if isinstance(chunk_samples, bool) or chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples!r}")

    master_resolved = _validate_path_string(master_wav, "master_wav")
    if not master_resolved.is_file():
        raise FileNotFoundError(f"Master WAV not found: {master_resolved}")
    out_resolved = _validate_dest_path(out_wav, "wav32f")
    if out_resolved == master_resolved:
        raise ValueError("Destination path must not overwrite master_wav in place")

    spec = DELIVERY_PRESETS[preset]

    # Pass 1: streaming loudness + 4x true peak (+ LRA for bypass fast path).
    with sf.SoundFile(str(master_resolved), mode="r") as reader:
        sample_rate = int(reader.samplerate)
        channels = int(reader.channels)
        total_samples = int(reader.frames)
        source_subtype = str(reader.subtype)
        if sample_rate <= 0 or channels <= 0 or total_samples <= 0:
            raise OutputValidationError(f"Master WAV is empty or invalid: {master_resolved}")

        pre_loudness = StreamingLoudnessMeter(sample_rate, channels)
        pre_tp4 = _StreamingTruePeakMeter(channels, factor=4)
        pre_lra_meter = _StreamingLraMeter(sample_rate, channels) if spec.bypass_limiter else None
        pass1_step = min(int(chunk_samples), _STREAM_TRUE_PEAK_MAX_STEP)
        while True:
            block = reader.read(frames=pass1_step, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            chunk_2d = np.ascontiguousarray(block.T, dtype=np.float32)
            if not np.all(np.isfinite(chunk_2d)):
                raise ValueError("waveform contains NaN or Inf values")
            pre_loudness.push(chunk_2d)
            pre_tp4.push(chunk_2d)
            if pre_lra_meter is not None:
                pre_lra_meter.push(chunk_2d)

    pre = pre_loudness.finish(pre_tp4.finish())
    pre_lra = pre_lra_meter.finish() if pre_lra_meter is not None else 0.0

    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = (
        out_resolved.parent
        / f".{out_resolved.stem}.{uuid.uuid4().hex[:8]}.tmp{out_resolved.suffix or '.wav'}"
    )
    tmp_trim_path = (
        out_resolved.parent
        / f".{out_resolved.stem}.{uuid.uuid4().hex[:8]}.trim.tmp{out_resolved.suffix or '.wav'}"
    )

    try:
        if spec.bypass_limiter:
            target_lufs = (
                float(input_lufs) if input_lufs is not None else float(pre.integrated_lufs)
            )
            if input_lufs is None or pre.integrated_lufs <= -69.0 or input_lufs <= -69.0:
                gain_db = 0.0
            else:
                gain_db = float(
                    np.clip(
                        input_lufs - pre.integrated_lufs,
                        -_MAX_STATIC_GAIN_DB,
                        _MAX_STATIC_GAIN_DB,
                    )
                )
                if abs(gain_db) < 1e-6:
                    gain_db = 0.0

            if gain_db == 0.0:
                copy_subtype = (
                    source_subtype if source_subtype in _DELIVERY_STREAM_SUBTYPES else "FLOAT"
                )
                container = _wav_container_for_frames(channels, total_samples, copy_subtype)
                with (
                    sf.SoundFile(str(master_resolved), mode="r") as reader,
                    sf.SoundFile(
                        str(tmp_path),
                        mode="w",
                        samplerate=sample_rate,
                        channels=channels,
                        subtype=copy_subtype,
                        format=container,
                    ) as writer,
                ):
                    while True:
                        block = reader.read(frames=pass1_step, dtype="float32", always_2d=True)
                        if block.size == 0:
                            break
                        writer.write(block)
                post = pre
                lra = pre_lra
            else:
                gain_linear = float(10.0 ** (gain_db / 20.0))
                container = _wav_container_for_frames(channels, total_samples, "FLOAT")
                post_loudness = StreamingLoudnessMeter(sample_rate, channels)
                post_tp4 = _StreamingTruePeakMeter(channels, factor=4)
                post_lra = _StreamingLraMeter(sample_rate, channels)
                with (
                    sf.SoundFile(str(master_resolved), mode="r") as reader,
                    sf.SoundFile(
                        str(tmp_path),
                        mode="w",
                        samplerate=sample_rate,
                        channels=channels,
                        subtype="FLOAT",
                        format=container,
                    ) as writer,
                ):
                    while True:
                        block = reader.read(frames=pass1_step, dtype="float32", always_2d=True)
                        if block.size == 0:
                            break
                        chunk_2d = np.ascontiguousarray(block.T, dtype=np.float32)
                        scaled = _scale_to_finite_float32(chunk_2d, gain_linear)
                        writer.write(scaled.T)
                        post_loudness.push(scaled)
                        post_tp4.push(scaled)
                        post_lra.push(scaled)
                post = post_loudness.finish(post_tp4.finish())
                lra = post_lra.finish()

            _finalize_deterministic_wav(tmp_path)
            if is_reparse_or_symlink(out_resolved):
                raise ValueError(
                    f"Destination path must not be a symlink or reparse point: {out_resolved}"
                )
            os.replace(tmp_path, out_resolved)
            return {
                "preset": preset,
                "limiter_mode": "bypass",
                "input_lufs": float(pre.integrated_lufs),
                "target_lufs": target_lufs,
                "applied_gain_db": gain_db,
                "integrated_lufs": float(post.integrated_lufs),
                "true_peak_dbtp": float(post.true_peak_dbtp),
                "sample_peak_dbfs": float(post.sample_peak_dbfs),
                "loudness_range_lu": float(lra),
                "limiter_max_gr_db": 0.0,
                "ceiling_dbtp": "bypass",
            }

        target = spec.target_lufs_mono if channels == 1 else spec.target_lufs_stereo
        ceiling = spec.true_peak_ceiling_dbtp
        assert target is not None and ceiling is not None

        if pre.integrated_lufs <= -69.0:
            gain_db = 0.0
            gain_linear = 1.0
        else:
            gain_db = float(
                np.clip(target - pre.integrated_lufs, -_MAX_STATIC_GAIN_DB, _MAX_STATIC_GAIN_DB)
            )
            gain_linear = float(10.0 ** (gain_db / 20.0))

        ceiling_linear = float(10.0 ** (ceiling / 20.0))
        lookahead_samples = int(round(sample_rate * (5.0 / 1000.0)))
        release_coeff = float(np.exp(-1.0 / (sample_rate * (50.0 / 1000.0))))
        dependency = _future_gain_dependency(lookahead_samples)
        limiter_step = min(int(chunk_samples), _STREAM_LIMITER_MAX_STEP)
        container = _wav_container_for_frames(channels, total_samples, "FLOAT")

        post_loudness = StreamingLoudnessMeter(sample_rate, channels)
        post_tp4 = _StreamingTruePeakMeter(channels, factor=4)
        post_tp8 = _StreamingTruePeakMeter(channels, factor=8)
        post_lra = _StreamingLraMeter(sample_rate, channels)
        release_state = 1.0
        min_gain = 1.0

        with (
            sf.SoundFile(str(master_resolved), mode="r") as reader,
            sf.SoundFile(
                str(tmp_path),
                mode="w",
                samplerate=sample_rate,
                channels=channels,
                subtype="FLOAT",
                format=container,
            ) as writer,
        ):
            for start in range(0, total_samples, limiter_step):
                end = min(total_samples, start + limiter_step)
                needed_end = min(total_samples, end + dependency)
                source_start = max(0, start - TRUE_PEAK_EDGE)
                source_end = min(total_samples, needed_end + TRUE_PEAK_EDGE)
                reader.seek(source_start)
                raw_block = reader.read(
                    frames=source_end - source_start, dtype="float32", always_2d=True
                )
                if raw_block.shape[0] != source_end - source_start:
                    raise OutputValidationError(
                        f"Unexpected EOF while streaming master WAV: {master_resolved}"
                    )
                chunk_src = np.ascontiguousarray(raw_block.T, dtype=np.float32)
                if not np.all(np.isfinite(chunk_src)):
                    raise ValueError("waveform contains NaN or Inf values")

                gained_slice = (
                    chunk_src
                    if gain_linear == 1.0
                    else _scale_to_finite_float32(chunk_src, gain_linear)
                )
                local_start = start - source_start
                local_end = end - source_start
                local_needed_end = needed_end - source_start

                peak_env = oversampled_peak_envelope_window(
                    gained_slice, 4, local_start, local_needed_end
                )
                inst_gain = np.ones(len(peak_env), dtype=np.float32)
                over = peak_env > ceiling_linear
                inst_gain[over] = ceiling_linear / (peak_env[over] + 1e-12)
                if lookahead_samples > 0 and total_samples > 1:
                    size = min(lookahead_samples + 1, total_samples)
                    windowed = scipy.ndimage.minimum_filter1d(
                        inst_gain,
                        size=size,
                        origin=-(size // 2),
                        mode="nearest",
                    )
                else:
                    windowed = inst_gain
                anticipated = _slope_limited_min_envelope(windowed, lookahead_samples)
                smooth = np.ascontiguousarray(anticipated[: end - start], dtype=np.float32)
                release_state = _release_smooth_carried(smooth, release_coeff, release_state)
                if smooth.size:
                    min_gain = min(min_gain, float(np.min(smooth)))

                limited_core = np.multiply(
                    gained_slice[:, local_start:local_end], smooth, dtype=np.float32
                )
                writer.write(limited_core.T)
                post_loudness.push(limited_core)
                post_tp4.push(limited_core)
                post_tp8.push(limited_core)
                post_lra.push(limited_core)

        tp8 = post_tp8.finish()
        final_tmp = tmp_path
        if tp8 > ceiling_linear:
            trim = float(np.float32((ceiling_linear / tp8) * (1.0 - 1e-6)))
            min_gain *= trim
            post_loudness = StreamingLoudnessMeter(sample_rate, channels)
            post_tp4 = _StreamingTruePeakMeter(channels, factor=4)
            post_tp8 = _StreamingTruePeakMeter(channels, factor=8)
            post_lra = _StreamingLraMeter(sample_rate, channels)
            with (
                sf.SoundFile(str(tmp_path), mode="r") as reader,
                sf.SoundFile(
                    str(tmp_trim_path),
                    mode="w",
                    samplerate=sample_rate,
                    channels=channels,
                    subtype="FLOAT",
                    format=container,
                ) as writer,
            ):
                while True:
                    block = reader.read(frames=limiter_step, dtype="float32", always_2d=True)
                    if block.size == 0:
                        break
                    trimmed = np.multiply(
                        np.ascontiguousarray(block.T, dtype=np.float32),
                        np.float32(trim),
                        dtype=np.float32,
                    )
                    writer.write(trimmed.T)
                    post_loudness.push(trimmed)
                    post_tp4.push(trimmed)
                    post_tp8.push(trimmed)
                    post_lra.push(trimmed)
            tp8 = post_tp8.finish()
            if tp8 > ceiling_linear:
                raise OutputValidationError(
                    f"Limiter failed to enforce ceiling: true peak {tp8:.6f} > {ceiling_linear:.6f}"
                )
            tmp_path.unlink(missing_ok=True)
            final_tmp = tmp_trim_path

        post = post_loudness.finish(post_tp4.finish())
        lra = post_lra.finish()
        max_gr_db = float(-20.0 * np.log10(max(min_gain, 1e-6)))

        _finalize_deterministic_wav(final_tmp)
        if is_reparse_or_symlink(out_resolved):
            raise ValueError(
                f"Destination path must not be a symlink or reparse point: {out_resolved}"
            )
        os.replace(final_tmp, out_resolved)
        return {
            "preset": preset,
            "limiter_mode": "lookahead",
            "input_lufs": float(pre.integrated_lufs),
            "target_lufs": float(target),
            "applied_gain_db": gain_db,
            "integrated_lufs": float(post.integrated_lufs),
            "true_peak_dbtp": float(post.true_peak_dbtp),
            "sample_peak_dbfs": float(post.sample_peak_dbfs),
            "loudness_range_lu": float(lra),
            "limiter_max_gr_db": max_gr_db,
            "ceiling_dbtp": float(ceiling),
        }
    except Exception:
        tmp_path.unlink(missing_ok=True)
        tmp_trim_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class _WavChunkRecord:
    """Indexed RIFF/RF64 chunk location inside a WAV file."""

    chunk_id: bytes
    payload_offset: int
    payload_size: int
    raw_header_size: int


def _parse_ds64_sizes(payload: bytes) -> tuple[int, int, dict[bytes, int]]:
    """Parse RF64 ds64 chunk returning (riff_size, data_size, extra_chunk_sizes)."""
    if len(payload) < 28:
        raise ValueError(f"Malformed RF64 ds64 chunk: expected >=28 bytes, got {len(payload)}")
    riff_size, data_size, _sample_count, table_len = struct.unpack_from("<QQQI", payload, 0)
    if riff_size > _MAX_RF64_SIZE or data_size > _MAX_RF64_SIZE:
        raise ValueError("RF64 ds64 chunk size overflow")
    expected_min = 28 + table_len * 12
    if len(payload) < expected_min:
        raise ValueError(
            f"Malformed RF64 ds64 size table: expected >={expected_min} bytes, got {len(payload)}"
        )
    extra: dict[bytes, int] = {}
    offset = 28
    for _ in range(table_len):
        cid = payload[offset : offset + 4]
        csize = struct.unpack_from("<Q", payload, offset + 4)[0]
        if csize > _MAX_RF64_SIZE:
            raise ValueError(f"RF64 ds64 size table overflow for chunk {cid!r}")
        extra[cid] = int(csize)
        offset += 12
    return int(riff_size), int(data_size), extra


def _scan_wav_chunks(
    handle: BinaryIO,
    file_size: int,
) -> tuple[bytes, list[_WavChunkRecord]]:
    """Scan RIFF/RF64 WAVE container and return (container_magic, chunk_records)."""
    header = handle.read(12)
    if len(header) < 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
        raise ValueError("File is not a valid RIFF/RF64 WAVE container")

    magic = header[:4]
    ds64_data_size: int | None = None
    ds64_extra: dict[bytes, int] = {}
    records: list[_WavChunkRecord] = []

    pos = 12
    while pos < file_size:
        if len(records) >= _MAX_WAV_CHUNKS:
            raise ValueError(
                f"WAV container exceeds maximum allowed chunk count ({_MAX_WAV_CHUNKS})"
            )
        if file_size - pos < 8:
            raise ValueError(f"Truncated WAV chunk header at offset {pos}")
        handle.seek(pos)
        chunk_hdr = handle.read(8)
        if len(chunk_hdr) < 8:
            raise ValueError(f"Truncated WAV chunk header at offset {pos}")
        chunk_id = chunk_hdr[:4]
        raw_size = int.from_bytes(chunk_hdr[4:8], "little")
        payload_offset = pos + 8

        actual_size = raw_size
        if magic == b"RIFF" and raw_size == 0xFFFFFFFF:
            raise ValueError(
                f"Classic RIFF chunk {chunk_id!r} uses invalid 0xFFFFFFFF size sentinel"
            )
        if magic == b"RF64" and raw_size == 0xFFFFFFFF:
            if chunk_id == b"data" and ds64_data_size is not None:
                actual_size = ds64_data_size
            elif chunk_id in ds64_extra:
                actual_size = ds64_extra[chunk_id]
            else:
                raise ValueError(
                    f"RF64 chunk {chunk_id!r} uses 0xFFFFFFFF sentinel without ds64 entry"
                )

        if payload_offset + actual_size > file_size:
            raise ValueError(
                f"Truncated WAV chunk {chunk_id!r} at offset {pos}: "
                f"need {actual_size} bytes, only {file_size - payload_offset} available"
            )

        if magic == b"RF64" and chunk_id == b"ds64":
            if records:
                raise ValueError("RF64 ds64 chunk must be the first chunk")
            if actual_size > _MAX_DS64_CHUNK_BYTES:
                raise ValueError(f"RF64 ds64 chunk exceeds {_MAX_DS64_CHUNK_BYTES} byte safety cap")
            ds64_bytes = handle.read(actual_size)
            if len(ds64_bytes) < actual_size:
                raise ValueError("Truncated RF64 ds64 chunk payload")
            _, ds64_data_size, ds64_extra = _parse_ds64_sizes(ds64_bytes)
            if ds64_data_size > file_size:
                raise ValueError("RF64 ds64 data_size exceeds file size")
            for extra_cid, extra_size in ds64_extra.items():
                if extra_size > file_size:
                    raise ValueError(f"RF64 ds64 size for chunk {extra_cid!r} exceeds file size")

        records.append(_WavChunkRecord(chunk_id, payload_offset, actual_size, raw_size))
        pos = payload_offset + actual_size + (actual_size & 1)

    return magic, records


def extract_wav_metadata_chunks(wav_path: Path) -> dict[bytes, bytes]:
    """Extract preserved Broadcast Wave / iXML metadata chunks losslessly from RIFF/RF64 WAV.

    Returns an empty dict for non-WAV containers (e.g. MP4/FLAC). Raises
    ``FileNotFoundError`` if ``wav_path`` does not exist and ``ValueError`` if a
    RIFF/RF64 WAVE file is structurally truncated, corrupt, or exceeds metadata
    safety caps.
    """
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    file_size = path.stat().st_size
    if file_size < 12:
        return {}

    with open(path, "rb") as handle:
        head = handle.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return {}
        handle.seek(0)
        _, records = _scan_wav_chunks(handle, file_size)
        extracted: dict[bytes, bytes] = {}
        for rec in records:
            if rec.chunk_id in PRESERVED_WAV_METADATA_CHUNKS:
                if rec.payload_size > MAX_METADATA_CHUNK_BYTES:
                    raise ValueError(
                        f"WAV metadata chunk {rec.chunk_id!r} exceeds "
                        f"{MAX_METADATA_CHUNK_BYTES} byte safety cap (got {rec.payload_size})"
                    )
                handle.seek(rec.payload_offset)
                payload = handle.read(rec.payload_size)
                if len(payload) < rec.payload_size:
                    raise ValueError(f"Truncated WAV metadata chunk {rec.chunk_id!r}")
                extracted[rec.chunk_id] = payload
        return extracted


def _stream_copy_range(
    src: BinaryIO,
    dst: BinaryIO,
    offset: int,
    length: int,
) -> None:
    """Copy exact byte range [offset, offset + length) in bounded memory."""
    src.seek(offset)
    remaining = length
    while remaining > 0:
        to_read = min(remaining, _STREAM_COPY_BYTES)
        buf = src.read(to_read)
        if not buf:
            raise OutputValidationError("Unexpected EOF while copying WAV chunk payload")
        dst.write(buf)
        remaining -= len(buf)


def inject_wav_metadata_chunks(wav_path: Path, chunks: dict[bytes, bytes]) -> bool:
    """Losslessly inject or replace BWF/iXML metadata chunks in a RIFF/RF64 WAV file.

    Rewrites atomically via a sibling temporary file without modifying audio
    ``data`` samples or corrupting ``RIFF`` / ``ds64`` size headers.
    """
    if not chunks:
        return False

    for chunk_id, payload in chunks.items():
        if not isinstance(chunk_id, bytes) or len(chunk_id) != 4:
            raise ValueError(f"WAV chunk ID must be exactly 4 bytes, got {chunk_id!r}")
        if chunk_id in _PROTECTED_WAV_CHUNKS:
            raise ValueError(f"Cannot inject protected structural WAV chunk {chunk_id!r}")
        if not isinstance(payload, bytes):
            raise ValueError(f"WAV chunk payload for {chunk_id!r} must be bytes")
        if len(payload) > MAX_METADATA_CHUNK_BYTES:
            raise ValueError(
                f"WAV metadata chunk {chunk_id!r} exceeds {MAX_METADATA_CHUNK_BYTES} byte safety cap"
            )

    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    file_size = path.stat().st_size

    tmp_path = path.parent / f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp{path.suffix or '.wav'}"
    try:
        with open(path, "rb") as src:
            magic, records = _scan_wav_chunks(src, file_size)
            if not any(rec.chunk_id == b"data" for rec in records):
                raise ValueError(f"WAV file has no audio data chunk: {path}")

            # Filter out any pre-existing copies of chunks we are injecting.
            inject_ids = set(chunks.keys())
            kept_records = [rec for rec in records if rec.chunk_id not in inject_ids]

            # Build ordered output plan: place injected metadata immediately before `data`.
            # RF64 requires `ds64` to remain the very first chunk.
            before_data: list[_WavChunkRecord] = []
            at_and_after_data: list[_WavChunkRecord] = []
            seen_data = False
            for rec in kept_records:
                if rec.chunk_id == b"data":
                    seen_data = True
                if seen_data:
                    at_and_after_data.append(rec)
                else:
                    before_data.append(rec)

            # Place even-length `bext` before `data` (standard EBU Tech 3285 BWF layout);
            # place `iXML` and all other/odd-length metadata chunks after `data` (standard
            # iXML layout and compatible with libsndfile's RF64 pre-data chunk scanner).
            inject_before: list[tuple[bytes, bytes]] = []
            inject_after: list[tuple[bytes, bytes]] = []
            if b"bext" in chunks and (len(chunks[b"bext"]) & 1) == 0:
                inject_before.append((b"bext", chunks[b"bext"]))
            for preferred in (b"bext", b"iXML"):
                if preferred in chunks and not (
                    preferred == b"bext" and (len(chunks[b"bext"]) & 1) == 0
                ):
                    inject_after.append((preferred, chunks[preferred]))
            for cid, payload in chunks.items():
                if cid not in (b"bext", b"iXML"):
                    inject_after.append((cid, payload))

            total_riff_size = 4  # b"WAVE"
            for rec in before_data:
                total_riff_size += 8 + rec.payload_size + (rec.payload_size & 1)
            for _, payload in inject_before:
                total_riff_size += 8 + len(payload) + (len(payload) & 1)
            for rec in at_and_after_data:
                total_riff_size += 8 + rec.payload_size + (rec.payload_size & 1)
            for _, payload in inject_after:
                total_riff_size += 8 + len(payload) + (len(payload) & 1)

            if magic == b"RIFF" and total_riff_size > _WAV_CLASSIC_LIMIT_BYTES:
                raise OutputValidationError(
                    "Injecting metadata would exceed classic 4 GiB RIFF limit; use RF64 container"
                )

            with open(tmp_path, "wb") as dst:
                if magic == b"RIFF":
                    dst.write(b"RIFF")
                    dst.write(struct.pack("<I", total_riff_size))
                    dst.write(b"WAVE")
                else:
                    dst.write(b"RF64")
                    dst.write(struct.pack("<I", 0xFFFFFFFF))
                    dst.write(b"WAVE")

                def write_existing(rec: _WavChunkRecord) -> None:
                    if magic == b"RF64" and rec.chunk_id == b"ds64":
                        src.seek(rec.payload_offset)
                        ds64_raw = bytearray(src.read(rec.payload_size))
                        struct.pack_into("<Q", ds64_raw, 0, total_riff_size)
                        dst.write(b"ds64")
                        dst.write(struct.pack("<I", len(ds64_raw)))
                        dst.write(ds64_raw)
                        if len(ds64_raw) & 1:
                            dst.write(b"\x00")
                        return

                    hdr_size = (
                        0xFFFFFFFF
                        if (
                            magic == b"RF64"
                            and (
                                rec.raw_header_size == 0xFFFFFFFF
                                or rec.payload_size > _WAV_CLASSIC_LIMIT_BYTES
                            )
                        )
                        else rec.payload_size
                    )
                    dst.write(rec.chunk_id)
                    dst.write(struct.pack("<I", hdr_size))
                    _stream_copy_range(src, dst, rec.payload_offset, rec.payload_size)
                    if rec.payload_size & 1:
                        dst.write(b"\x00")

                for rec in before_data:
                    write_existing(rec)

                for cid, payload in inject_before:
                    dst.write(cid)
                    dst.write(struct.pack("<I", len(payload)))
                    dst.write(payload)
                    if len(payload) & 1:
                        dst.write(b"\x00")

                for rec in at_and_after_data:
                    write_existing(rec)

                for cid, payload in inject_after:
                    dst.write(cid)
                    dst.write(struct.pack("<I", len(payload)))
                    dst.write(payload)
                    if len(payload) & 1:
                        dst.write(b"\x00")

                dst.flush()
                os.fsync(dst.fileno())

        os.replace(tmp_path, path)
        return True
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _validate_path_string(path: Path, label: str) -> Path:
    """Reject empty paths and NUL bytes before resolving."""
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"Invalid {label} path: {path!r}")
    return Path(path).resolve()


def _is_system_prefix(candidate: Path) -> bool:
    """Return True if ``candidate`` is a host/system root alias (e.g. macOS ``/var``)."""
    for anchor in (Path.home(), Path(tempfile.gettempdir()), Path("/tmp")):
        if anchor == candidate or anchor.is_relative_to(candidate):
            return True
    return False


def _validate_dest_path(path: Path, fmt: ExportFormat) -> Path:
    """Validate destination path without resolving a final symlink component."""
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"Invalid dest_path path: {path!r}")
    dest_raw = Path(path)
    if not dest_raw.name or dest_raw.name in (".", ".."):
        raise ValueError(f"Invalid dest_path path: {path!r}")
    if is_reparse_or_symlink(dest_raw):
        raise ValueError(f"Destination path must not be a symlink or reparse point: {dest_raw}")
    for ancestor in dest_raw.parents:
        if ancestor == ancestor.parent or _is_system_prefix(ancestor):
            continue
        if is_reparse_or_symlink(ancestor):
            raise ValueError(
                f"Destination path ancestor must not be a symlink or reparse point: {ancestor}"
            )
    parent = dest_raw.parent.resolve()
    dest_no_follow = parent / dest_raw.name
    if is_reparse_or_symlink(dest_no_follow):
        raise ValueError(
            f"Destination path must not be a symlink or reparse point: {dest_no_follow}"
        )
    for ancestor in dest_no_follow.parents:
        if ancestor == ancestor.parent:
            continue
        if is_reparse_or_symlink(ancestor):
            raise ValueError(
                f"Destination path ancestor must not be a symlink or reparse point: {ancestor}"
            )
    if dest_no_follow.is_dir():
        raise ValueError(f"Destination path is a directory: {dest_no_follow}")
    allowed_exts = EXPORT_FORMAT_EXTENSIONS[fmt]
    if dest_no_follow.suffix.lower() not in allowed_exts:
        raise ValueError(
            f"Destination extension {dest_no_follow.suffix!r} is not valid for format {fmt!r}; "
            f"expected one of {sorted(allowed_exts)}"
        )
    return dest_no_follow


def _export_soundfile(
    master_resolved: Path,
    tmp_path: Path,
    *,
    fmt: Literal["wav24", "wav32f", "flac"],
    meta_chunks: dict[bytes, bytes],
) -> None:
    """Stream-transcode master WAV to wav24, wav32f, or flac via soundfile."""
    with sf.SoundFile(str(master_resolved), mode="r") as reader:
        sample_rate = int(reader.samplerate)
        channels = int(reader.channels)
        frames = int(reader.frames)
        if sample_rate <= 0 or channels <= 0 or frames <= 0:
            raise OutputValidationError(f"Master WAV is empty or invalid: {master_resolved}")

        if fmt == "flac":
            container = "FLAC"
            subtype = "PCM_24"
        else:
            subtype = "PCM_24" if fmt == "wav24" else "FLOAT"
            bytes_per_sample = 3 if subtype == "PCM_24" else 4
            estimated = channels * frames * bytes_per_sample + _WAV_HEADER_RESERVE_BYTES
            container = "RF64" if estimated > _WAV_CLASSIC_LIMIT_BYTES else "WAV"

        with sf.SoundFile(
            str(tmp_path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            subtype=subtype,
            format=container,
        ) as writer:
            while True:
                block = reader.read(frames=_STREAM_AUDIO_FRAMES, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                if not np.all(np.isfinite(block)):
                    raise OutputValidationError(
                        f"Master WAV contains NaN or Inf samples: {master_resolved}"
                    )
                writer.write(block)

    if fmt in ("wav24", "wav32f") and meta_chunks:
        inject_wav_metadata_chunks(tmp_path, meta_chunks)


def _run_ffmpeg_checked(cmd: list[str], *, timeout_s: float, label: str) -> None:
    """Run an FFmpeg subprocess safely without shell interpolation."""
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise OutputValidationError(f"FFmpeg {label} timed out after {timeout_s}s") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise OutputValidationError(
            f"FFmpeg {label} failed (exit {exc.returncode}): {stderr}"
        ) from exc


def export_delivery_artifact(
    master_wav: Path,
    dest_path: Path,
    *,
    fmt: ExportFormat,
    source_media_path: Path | None = None,
    preserve_metadata: bool = True,
) -> Path:
    """Export master WAV to delivery format with strict path validation and atomic replace.

    Supported formats:
    - ``"wav24"``: 24-bit PCM WAV/RF64 (with BWF/iXML metadata preservation)
    - ``"wav32f"``: 32-bit float WAV/RF64 (with BWF/iXML metadata preservation)
    - ``"flac"``: 24-bit lossless FLAC
    - ``"aac"``: 256 kbps AAC via bundled/system FFmpeg
    - ``"video_mux"``: lossless ``-c:v copy`` video mux-back replacing audio stream
    """
    valid_formats: set[str] = {"wav24", "wav32f", "flac", "aac", "video_mux"}
    if fmt not in valid_formats:
        raise ValueError(f"Unsupported export format: {fmt!r}")

    master_resolved = _validate_path_string(master_wav, "master_wav")
    dest_resolved = _validate_dest_path(dest_path, fmt)

    if not master_resolved.is_file():
        raise FileNotFoundError(f"Master WAV not found: {master_resolved}")
    if dest_resolved == master_resolved:
        raise ValueError("Destination path must not overwrite master_wav in place")

    source_resolved: Path | None = None
    if source_media_path is not None:
        source_resolved = _validate_path_string(source_media_path, "source_media_path")
        if dest_resolved == source_resolved:
            raise ValueError("Destination path must not overwrite source_media_path in place")

    if fmt == "video_mux":
        if source_resolved is None:
            raise ValueError("source_media_path is required for video_mux export")
        if not source_resolved.is_file():
            raise FileNotFoundError(f"Source media not found: {source_resolved}")

    meta_chunks: dict[bytes, bytes] = {}
    if preserve_metadata and fmt in ("wav24", "wav32f"):
        if source_resolved is not None and source_resolved.is_file():
            meta_chunks.update(extract_wav_metadata_chunks(source_resolved))
        meta_chunks.update(extract_wav_metadata_chunks(master_resolved))

    dest_resolved.parent.mkdir(parents=True, exist_ok=True)
    default_suffix = {
        "wav24": ".wav",
        "wav32f": ".wav",
        "flac": ".flac",
        "aac": ".m4a",
        "video_mux": source_resolved.suffix
        if source_resolved and source_resolved.suffix
        else ".mp4",
    }[fmt]
    tmp_suffix = dest_resolved.suffix or default_suffix
    tmp_path = (
        dest_resolved.parent / f".{dest_resolved.stem}.{uuid.uuid4().hex[:8]}.tmp{tmp_suffix}"
    )

    try:
        if fmt in ("wav24", "wav32f", "flac"):
            _export_soundfile(
                master_resolved,
                tmp_path,
                fmt=fmt,
                meta_chunks=meta_chunks,
            )
        elif fmt == "aac":
            ffmpeg_bin = ffmpeg_bin_path()
            if not ffmpeg_bin:
                raise PreflightError("FFmpeg binary is required for AAC export but was not found.")
            cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(master_resolved),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
            ]
            if tmp_suffix.lower() in (".m4a", ".mp4"):
                cmd.extend(["-movflags", "+faststart"])
            cmd.append(str(tmp_path))
            _run_ffmpeg_checked(cmd, timeout_s=300.0, label="AAC export")
        else:
            assert source_resolved is not None
            ffmpeg_bin = ffmpeg_bin_path()
            if not ffmpeg_bin:
                raise PreflightError(
                    "FFmpeg binary is required for video mux-back but was not found."
                )
            ext = tmp_suffix.lower()
            cmd = [
                ffmpeg_bin,
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(source_resolved),
                "-i",
                str(master_resolved),
                "-map",
                "0:v",
                "-map",
                "1:a:0",
                "-map",
                "0:s?",
                "-c:v",
                "copy",
                "-c:s",
                "copy",
            ]
            if preserve_metadata:
                cmd.extend(["-map_metadata", "0"])
            if ext in (".mov", ".mkv", ".mxf"):
                cmd.extend(["-c:a", "pcm_s24le"])
            else:
                cmd.extend(["-c:a", "aac", "-b:a", "256k"])
                if ext in (".mp4", ".m4v"):
                    cmd.extend(["-movflags", "+faststart"])
            cmd.append(str(tmp_path))
            _run_ffmpeg_checked(cmd, timeout_s=600.0, label="video mux-back")

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise OutputValidationError(f"Export produced an empty artifact: {dest_resolved}")

        if is_reparse_or_symlink(dest_resolved):
            raise ValueError(
                f"Destination path must not be a symlink or reparse point: {dest_resolved}"
            )
        os.replace(tmp_path, dest_resolved)
        return dest_resolved
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
