"""2D STFT and delta spectrogram tile generator for windowed UI inspection.

Decodes *only* the requested ``[start_s, end_s]`` window via
:func:`~hawavoclean.audio.decode.decode_audio_window`, computes a real STFT
with :func:`scipy.signal.stft`, resamples frequency bins to ``log``, ``mel``,
or ``linear`` grids and time frames to ``width_cols``, and returns a compact
base64-encoded ``uint8`` row-major ``[n_bins * width_cols]`` tile guaranteed
free of NaN/Inf.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.signal

from hawavoclean.audio.decode import decode_audio_window, window_sample_bounds
from hawavoclean.audio.resample import resample_audio
from hawavoclean.server.analysis import cached_probe

SpectrogramScale = Literal["log", "mel", "linear"]

_VALID_SCALES: frozenset[str] = frozenset({"log", "mel", "linear"})
MAX_SPECTROGRAM_WINDOW_S = 300.0
MAX_TILE_WIDTH_COLS = 4096
MAX_TILE_N_BINS = 1024
MIN_N_FFT = 64
MAX_N_FFT = 16384
DELTA_DEFAULT_DB_FLOOR = 0.0
DELTA_DEFAULT_DB_CEIL = 40.0


@dataclass(frozen=True)
class SpectrogramTile:
    """Compact 8-bit normalized spectrogram tile (row-major ``[n_bins, width_cols]``)."""

    start_s: float
    end_s: float
    sample_rate: int
    f_min_hz: float
    f_max_hz: float
    n_bins: int
    width_cols: int
    scale: SpectrogramScale
    db_floor: float
    db_ceil: float
    data_b64: str

    def decode_grid(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Decode ``data_b64`` back into a ``(n_bins, width_cols)`` uint8 array."""
        raw = base64.b64decode(self.data_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return arr.reshape((self.n_bins, self.width_cols))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "sample_rate": self.sample_rate,
            "f_min_hz": self.f_min_hz,
            "f_max_hz": self.f_max_hz,
            "n_bins": self.n_bins,
            "width_cols": self.width_cols,
            "scale": self.scale,
            "db_floor": self.db_floor,
            "db_ceil": self.db_ceil,
            "data_b64": self.data_b64,
        }


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(700.0 * (np.power(10.0, mel / 2595.0) - 1.0), dtype=np.float64)


def _validate_tile_params(
    *,
    start_s: float,
    end_s: float,
    width_cols: int,
    n_bins: int,
    n_fft: int,
    scale: str,
    db_floor: float,
    db_ceil: float,
) -> None:
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise ValueError(f"start_s and end_s must be finite, got ({start_s}, {end_s})")
    if start_s < 0.0:
        raise ValueError(f"start_s must be >= 0, got {start_s}")
    if end_s <= start_s:
        raise ValueError(f"end_s must be greater than start_s, got ({start_s}, {end_s})")
    if end_s - start_s > MAX_SPECTROGRAM_WINDOW_S:
        raise ValueError(
            f"Spectrogram window duration ({end_s - start_s:.2f}s) exceeds maximum "
            f"{MAX_SPECTROGRAM_WINDOW_S:.1f}s"
        )
    if isinstance(width_cols, bool) or not 1 <= width_cols <= MAX_TILE_WIDTH_COLS:
        raise ValueError(f"width_cols must be in 1..{MAX_TILE_WIDTH_COLS}, got {width_cols!r}")
    if isinstance(n_bins, bool) or not 1 <= n_bins <= MAX_TILE_N_BINS:
        raise ValueError(f"n_bins must be in 1..{MAX_TILE_N_BINS}, got {n_bins!r}")
    if isinstance(n_fft, bool) or not MIN_N_FFT <= n_fft <= MAX_N_FFT:
        raise ValueError(f"n_fft must be in {MIN_N_FFT}..{MAX_N_FFT}, got {n_fft!r}")
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale must be one of {sorted(_VALID_SCALES)}, got {scale!r}")
    if not (math.isfinite(db_floor) and math.isfinite(db_ceil)) or db_floor >= db_ceil:
        raise ValueError(f"db_floor must be < db_ceil and finite, got ({db_floor}, {db_ceil})")


def _resolve_freq_bounds(
    sample_rate: int,
    scale: SpectrogramScale,
    f_min_hz: float | None,
    f_max_hz: float | None,
) -> tuple[float, float]:
    nyquist = sample_rate / 2.0
    hi = nyquist if f_max_hz is None else min(nyquist, float(f_max_hz))
    if scale == "log":
        default_lo = min(20.0, hi * 0.5)
        lo = default_lo if f_min_hz is None else max(1.0, float(f_min_hz))
    else:
        lo = 0.0 if f_min_hz is None else max(0.0, float(f_min_hz))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ValueError(f"Invalid frequency bounds [{lo}, {hi}] for sample_rate={sample_rate}")
    return lo, hi


def _target_freq_grid(
    f_min_hz: float,
    f_max_hz: float,
    n_bins: int,
    scale: SpectrogramScale,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if n_bins == 1:
        return np.asarray([(f_min_hz + f_max_hz) * 0.5], dtype=np.float64)
    if scale == "linear":
        return np.linspace(f_min_hz, f_max_hz, n_bins, dtype=np.float64)
    if scale == "log":
        return np.geomspace(max(f_min_hz, 1.0), f_max_hz, n_bins, dtype=np.float64)
    mel_grid = np.linspace(_hz_to_mel(f_min_hz), _hz_to_mel(f_max_hz), n_bins, dtype=np.float64)
    return _mel_to_hz(mel_grid)


def _freq_resample_matrix(
    stft_freqs: np.ndarray[Any, np.dtype[np.float64]],
    target_freqs: np.ndarray[Any, np.dtype[np.float64]],
    f_min_hz: float,
    f_max_hz: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Build a row-normalized ``(n_bins, n_stft)`` power-mapping matrix."""
    n_bins = int(target_freqs.shape[0])
    n_stft = int(stft_freqs.shape[0])
    weights = np.zeros((n_bins, n_stft), dtype=np.float64)

    if n_bins == 1:
        edges = np.asarray([f_min_hz, f_max_hz], dtype=np.float64)
    else:
        mids = 0.5 * (target_freqs[:-1] + target_freqs[1:])
        edges = np.concatenate(([f_min_hz], mids, [f_max_hz]))

    for idx in range(n_bins):
        lo_f = float(edges[idx])
        hi_f = float(edges[idx + 1])
        mask = (stft_freqs >= lo_f) & (stft_freqs <= hi_f)
        if bool(np.any(mask)):
            count = float(np.sum(mask))
            weights[idx, mask] = 1.0 / count
            continue

        # Target bin narrower than one STFT bin: linear interpolation between neighbors.
        fc = float(target_freqs[idx])
        right = int(np.searchsorted(stft_freqs, fc, side="left"))
        if right <= 0:
            weights[idx, 0] = 1.0
        elif right >= n_stft:
            weights[idx, n_stft - 1] = 1.0
        else:
            left = right - 1
            span = float(stft_freqs[right] - stft_freqs[left])
            if span <= 0.0:
                weights[idx, left] = 1.0
            else:
                alpha = (fc - float(stft_freqs[left])) / span
                weights[idx, left] = 1.0 - alpha
                weights[idx, right] = alpha
    return weights


def _resample_time_axis(
    power_grid: np.ndarray[Any, np.dtype[np.float64]],
    width_cols: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Resample ``(n_bins, n_frames)`` power matrix to ``(n_bins, width_cols)``."""
    n_bins, n_frames = power_grid.shape
    if n_frames == width_cols:
        return power_grid
    if n_frames == 1:
        return np.repeat(power_grid, width_cols, axis=1)
    if n_frames > width_cols:
        starts = (np.arange(width_cols, dtype=np.int64) * n_frames) // width_cols
        ends = (np.arange(1, width_cols + 1, dtype=np.int64) * n_frames) // width_cols
        ends = np.maximum(ends, starts + 1)
        out = np.empty((n_bins, width_cols), dtype=np.float64)
        for col in range(width_cols):
            out[:, col] = np.mean(power_grid[:, int(starts[col]) : int(ends[col])], axis=1)
        return out

    src_x = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, width_cols, dtype=np.float64)
    out = np.empty((n_bins, width_cols), dtype=np.float64)
    for row in range(n_bins):
        out[row] = np.interp(dst_x, src_x, power_grid[row])
    return out


def _compute_stft_db_grid(
    mono: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    *,
    width_cols: int,
    n_bins: int,
    n_fft: int,
    scale: SpectrogramScale,
    f_min_hz: float,
    f_max_hz: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute a finite ``(n_bins, width_cols)`` STFT magnitude-dB grid from mono PCM."""
    x = np.ascontiguousarray(mono, dtype=np.float32)
    if x.size == 0:
        x = np.zeros(n_fft, dtype=np.float32)
    elif x.shape[0] < n_fft:
        x = np.pad(x, (0, n_fft - x.shape[0]), mode="constant")

    n_samples = int(x.shape[0])
    # Keep hop small enough for >= 2x time oversampling while bounding frame count.
    desired_hop = max(1, n_samples // max(width_cols * 2, 1))
    hop = min(n_fft, max(1, min(n_fft // 4, desired_hop)))
    if n_samples > n_fft * width_cols:
        hop = n_fft
    noverlap = max(0, n_fft - hop)

    freqs, _times, zxx = scipy.signal.stft(
        x,
        fs=float(sample_rate),
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    # Spectrum scaling gives |Zxx| = 0.5 for a full-scale sine; scale by 2.0 so
    # full-scale sine peaks at 0 dBFS.
    mag = np.abs(zxx).astype(np.float64, copy=False) * 2.0
    power = np.square(mag)

    target_freqs = _target_freq_grid(f_min_hz, f_max_hz, n_bins, scale)
    weights = _freq_resample_matrix(
        np.asarray(freqs, dtype=np.float64),
        target_freqs,
        f_min_hz,
        f_max_hz,
    )
    freq_power = weights @ power
    grid_power = _resample_time_axis(freq_power, width_cols)
    grid_power = np.maximum(grid_power, 1e-24)
    db_grid = 10.0 * np.log10(grid_power)
    return np.asarray(np.where(np.isfinite(db_grid), db_grid, -240.0), dtype=np.float64)


def _quantize_db_to_b64(
    db_grid: np.ndarray[Any, np.dtype[np.float64]],
    db_floor: float,
    db_ceil: float,
) -> str:
    span = db_ceil - db_floor
    norm = np.clip((db_grid - db_floor) / span, 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    quantized = np.ascontiguousarray(np.rint(norm * 255.0), dtype=np.uint8)
    return base64.b64encode(quantized.tobytes()).decode("ascii")


def _decode_window_mono(
    path: Path,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int, float, float]:
    resolved = Path(path).resolve()
    probe = cached_probe(resolved)
    duration_s = probe.samples / float(probe.sample_rate)
    if start_s >= duration_s:
        raise ValueError(
            f"start_s {start_s:.4f}s is at or past the end of audio ({duration_s:.4f}s)"
        )
    clamped_end_s = min(end_s, duration_s)
    start_sample, end_sample = window_sample_bounds(probe, start_s, clamped_end_s)
    buf = decode_audio_window(
        probe,
        start_sample / float(probe.sample_rate),
        end_sample / float(probe.sample_rate),
    )
    mono = buf.to_mono()
    actual_start = round(start_sample / float(probe.sample_rate), 6)
    actual_end = round((start_sample + int(mono.shape[0])) / float(probe.sample_rate), 6)
    return mono, probe.sample_rate, actual_start, actual_end


def compute_spectrogram_tile(
    path: Path,
    *,
    start_s: float,
    end_s: float,
    width_cols: int = 512,
    n_bins: int = 128,
    n_fft: int = 2048,
    scale: SpectrogramScale = "log",
    db_floor: float = -90.0,
    db_ceil: float = 0.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> SpectrogramTile:
    """Compute a 2D STFT spectrogram tile for ``path`` over ``[start_s, end_s]``."""
    _validate_tile_params(
        start_s=start_s,
        end_s=end_s,
        width_cols=width_cols,
        n_bins=n_bins,
        n_fft=n_fft,
        scale=scale,
        db_floor=db_floor,
        db_ceil=db_ceil,
    )
    mono, sample_rate, actual_start, actual_end = _decode_window_mono(path, start_s, end_s)
    lo_hz, hi_hz = _resolve_freq_bounds(sample_rate, scale, f_min_hz, f_max_hz)
    db_grid = _compute_stft_db_grid(
        mono,
        sample_rate,
        width_cols=width_cols,
        n_bins=n_bins,
        n_fft=n_fft,
        scale=scale,
        f_min_hz=lo_hz,
        f_max_hz=hi_hz,
    )
    data_b64 = _quantize_db_to_b64(db_grid, db_floor, db_ceil)
    return SpectrogramTile(
        start_s=actual_start,
        end_s=actual_end,
        sample_rate=sample_rate,
        f_min_hz=round(lo_hz, 3),
        f_max_hz=round(hi_hz, 3),
        n_bins=n_bins,
        width_cols=width_cols,
        scale=scale,
        db_floor=float(db_floor),
        db_ceil=float(db_ceil),
        data_b64=data_b64,
    )


def compute_delta_spectrogram_tile(
    original_path: Path,
    cleaned_path: Path,
    *,
    start_s: float,
    end_s: float,
    width_cols: int = 512,
    n_bins: int = 128,
    n_fft: int = 2048,
    scale: SpectrogramScale = "log",
    db_floor: float = DELTA_DEFAULT_DB_FLOOR,
    db_ceil: float = DELTA_DEFAULT_DB_CEIL,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
) -> SpectrogramTile:
    """Compute energy-removed delta tile ``max(0, orig_db - clean_db)`` normalized to ``0..255``."""
    _validate_tile_params(
        start_s=start_s,
        end_s=end_s,
        width_cols=width_cols,
        n_bins=n_bins,
        n_fft=n_fft,
        scale=scale,
        db_floor=db_floor,
        db_ceil=db_ceil,
    )
    orig_mono, orig_sr, actual_start, actual_end = _decode_window_mono(
        original_path, start_s, end_s
    )
    clean_mono, clean_sr, _clean_start, _clean_end = _decode_window_mono(
        cleaned_path, start_s, end_s
    )

    target_sr = max(orig_sr, clean_sr)
    if orig_sr != target_sr:
        orig_mono = resample_audio(orig_mono, orig_sr, target_sr)
    if clean_sr != target_sr:
        clean_mono = resample_audio(clean_mono, clean_sr, target_sr, target_samples=len(orig_mono))

    common_len = min(len(orig_mono), len(clean_mono))
    orig_mono = orig_mono[:common_len]
    clean_mono = clean_mono[:common_len]

    lo_hz, hi_hz = _resolve_freq_bounds(target_sr, scale, f_min_hz, f_max_hz)
    orig_db = _compute_stft_db_grid(
        orig_mono,
        target_sr,
        width_cols=width_cols,
        n_bins=n_bins,
        n_fft=n_fft,
        scale=scale,
        f_min_hz=lo_hz,
        f_max_hz=hi_hz,
    )
    clean_db = _compute_stft_db_grid(
        clean_mono,
        target_sr,
        width_cols=width_cols,
        n_bins=n_bins,
        n_fft=n_fft,
        scale=scale,
        f_min_hz=lo_hz,
        f_max_hz=hi_hz,
    )
    delta_db = np.maximum(0.0, orig_db - clean_db)
    data_b64 = _quantize_db_to_b64(delta_db, db_floor, db_ceil)
    return SpectrogramTile(
        start_s=actual_start,
        end_s=actual_end,
        sample_rate=target_sr,
        f_min_hz=round(lo_hz, 3),
        f_max_hz=round(hi_hz, 3),
        n_bins=n_bins,
        width_cols=width_cols,
        scale=scale,
        db_floor=float(db_floor),
        db_ceil=float(db_ceil),
        data_b64=data_b64,
    )
