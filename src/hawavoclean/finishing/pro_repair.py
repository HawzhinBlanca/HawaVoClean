"""Surgical dialogue repair (de-plosive, mouth de-click, de-clip) and room-tone floor."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.ndimage
import scipy.signal

Float32Array = np.ndarray[Any, np.dtype[np.float32]]


@dataclass(frozen=True)
class ProRepairOptions:
    """Configuration for the surgical dialogue repair and room-tone floor chain."""

    declip: bool = True
    clip_threshold: float = 0.985
    deplosive: bool = True
    deplosive_threshold_db: float = -18.0
    deplosive_max_reduction_db: float = 10.0
    declick: bool = True
    declick_sensitivity: float = 0.5
    declick_max_ms: float = 12.0
    room_tone_floor: bool = False
    room_tone_target_floor_db: float | None = -50.0
    room_tone_blend_ratio: float = 0.0


def _validate_waveform(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    name: str = "waveform",
) -> tuple[Float32Array, bool]:
    """Validate waveform shape/finiteness and normalize to contiguous 2D float32."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    arr = np.asarray(waveform)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    was_1d = arr.ndim == 1
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    if was_1d:
        f32 = f32[np.newaxis, :]
    return f32, was_1d


def _restore_shape(wave_2d: Float32Array, was_1d: bool) -> Float32Array:
    """Restore original 1D/2D shape as contiguous float32."""
    out = wave_2d[0] if was_1d else wave_2d
    return np.ascontiguousarray(out, dtype=np.float32)


def _find_true_runs(mask: np.ndarray[Any, np.dtype[np.bool_]]) -> list[tuple[int, int]]:
    """Return half-open [start, end) index runs where mask is True."""
    if mask.size == 0 or not bool(np.any(mask)):
        return []
    padded = np.empty(mask.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = mask.view(np.int8)
    padded[-1] = 0
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge adjacent [start, end) runs separated by at most max_gap samples."""
    if not runs:
        return []
    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _deplosive_channel(
    channel: Float32Array,
    sample_rate: int,
    threshold_db: float,
    max_reduction_db: float,
) -> tuple[Float32Array, int]:
    """Attenuate 20-160 Hz plosive bursts on one channel while preserving >160 Hz voice."""
    n = int(channel.shape[0])
    if n < 64 or max_reduction_db == 0.0 or sample_rate <= 400:
        return channel.copy(), 0

    low_hz = 20.0
    high_hz = min(160.0, 0.45 * float(sample_rate))
    if high_hz <= low_hz + 5.0:
        return channel.copy(), 0

    x64 = channel.astype(np.float64, copy=False)
    sos_bp = scipy.signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    plosive_band = np.asarray(scipy.signal.sosfiltfilt(sos_bp, x64), dtype=np.float64)
    residual = x64 - plosive_band

    fast_win = max(8, int(round(sample_rate * 0.015)))
    energy = scipy.ndimage.uniform_filter1d(
        plosive_band * plosive_band, size=fast_win, mode="nearest"
    )
    env_rms = np.sqrt(np.maximum(energy, 0.0) + 1e-12)
    env_db = 20.0 * np.log10(env_rms)

    excess_db = np.maximum(0.0, env_db - threshold_db)
    raw_gr_db = np.minimum(max_reduction_db, excess_db * 0.85)
    if float(np.max(raw_gr_db)) <= 0.1:
        return channel.copy(), 0

    # Anticipate plosive onset slightly and smooth envelope to prevent zipper artifacts.
    attack_win = max(3, int(round(sample_rate * 0.008)))
    smooth_win = max(5, int(round(sample_rate * 0.018)))
    anticipated = scipy.ndimage.maximum_filter1d(raw_gr_db, size=attack_win, mode="nearest")
    smooth_gr_db = scipy.ndimage.uniform_filter1d(anticipated, size=smooth_win, mode="nearest")
    smooth_gr_db = np.clip(smooth_gr_db, 0.0, max_reduction_db)

    active_runs = _find_true_runs(smooth_gr_db > 0.5)
    merge_gap = max(1, int(round(sample_rate * 0.040)))
    event_count = len(_merge_runs(active_runs, merge_gap))

    gain = np.power(10.0, -smooth_gr_db / 20.0)
    repaired = residual + plosive_band * gain
    return np.ascontiguousarray(repaired, dtype=np.float32), event_count


def apply_dynamic_deplosive(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    threshold_db: float = -18.0,
    max_reduction_db: float = 10.0,
) -> tuple[Float32Array, int]:
    """Sub-band 20-160 Hz dynamic de-plosive preserving voice fundamentals above 160 Hz."""
    wave_2d, was_1d = _validate_waveform(waveform, sample_rate)
    if not np.isfinite(threshold_db):
        raise ValueError(f"threshold_db must be finite, got {threshold_db}")
    if not np.isfinite(max_reduction_db) or max_reduction_db < 0.0:
        raise ValueError(f"max_reduction_db must be finite and >= 0, got {max_reduction_db}")

    channels, samples = wave_2d.shape
    if channels == 0 or samples == 0:
        return _restore_shape(wave_2d.copy(), was_1d), 0

    out = np.empty_like(wave_2d)
    total_events = 0
    for ch in range(channels):
        repaired, count = _deplosive_channel(
            wave_2d[ch], sample_rate, threshold_db, max_reduction_db
        )
        out[ch] = repaired
        total_events += count

    return _restore_shape(out, was_1d), total_events


def _fit_ar_coefficients(
    signal: np.ndarray[Any, np.dtype[np.float64]], order: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Solve regularized Yule-Walker equations for an AR(order) linear predictor."""
    n = int(signal.shape[0])
    if n <= order + 2 or order < 1:
        return np.zeros(order, dtype=np.float64)
    centered = signal - float(np.mean(signal))
    acf = np.empty(order + 1, dtype=np.float64)
    acf[0] = float(np.dot(centered, centered))
    if acf[0] <= 1e-18:
        return np.zeros(order, dtype=np.float64)
    for lag in range(1, order + 1):
        acf[lag] = float(np.dot(centered[lag:], centered[:-lag]))
    acf[0] *= 1.0 + 1e-5
    try:
        coeffs = scipy.linalg.solve_toeplitz((acf[:order], acf[:order]), acf[1 : order + 1])
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    except Exception:
        return np.zeros(order, dtype=np.float64)
    if not np.all(np.isfinite(coeffs_arr)):
        return np.zeros(order, dtype=np.float64)
    return coeffs_arr


def _hermite_fill_span(
    x: np.ndarray[Any, np.dtype[np.float64]],
    start: int,
    end: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Fill [start, end) using cubic Hermite spline from clean boundary samples."""
    n = int(x.shape[0])
    y0 = float(x[start - 1])
    y1 = float(x[end])
    m0 = float(x[start - 1] - x[start - 2]) if start >= 2 else 0.0
    m1 = float(x[end + 1] - x[end]) if end + 1 < n else 0.0
    spline = scipy.interpolate.CubicHermiteSpline([start - 1, end], [y0, y1], [m0, m1])
    idx = np.arange(start, end, dtype=np.float64)
    filled = np.asarray(spline(idx), dtype=np.float64)
    bound = max(abs(y0), abs(y1)) * 1.15 + 1e-4
    if not np.all(np.isfinite(filled)) or float(np.max(np.abs(filled))) > bound:
        return np.linspace(y0, y1, (end - start) + 2, dtype=np.float64)[1:-1]
    return filled


def _reconstruct_click_span(
    x: np.ndarray[Any, np.dtype[np.float64]],
    start: int,
    end: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Reconstruct click span [start, end) via forward-backward AR with Hermite fallback."""
    n = int(x.shape[0])
    span_len = end - start
    if start < 1 or end >= n or span_len <= 0:
        return x[start:end].copy()

    hermite = _hermite_fill_span(x, start, end)
    if span_len <= 4:
        return hermite

    order = min(12, max(4, span_len // 2))
    ctx = min(max(48, 4 * span_len), 256)
    left_lo = max(0, start - ctx)
    right_hi = min(n, end + ctx)
    left_ctx = x[left_lo:start]
    right_ctx = x[end:right_hi]

    if left_ctx.size < order + 4 or right_ctx.size < order + 4:
        return hermite

    a_fwd = _fit_ar_coefficients(left_ctx, order)
    a_bwd = _fit_ar_coefficients(right_ctx[::-1], order)

    fwd_buf = np.empty(order + span_len, dtype=np.float64)
    fwd_buf[:order] = left_ctx[-order:]
    for k in range(span_len):
        fwd_buf[order + k] = float(np.dot(a_fwd, fwd_buf[k : k + order][::-1]))
    fwd_pred = fwd_buf[order:]

    bwd_buf = np.empty(order + span_len, dtype=np.float64)
    bwd_buf[:order] = right_ctx[:order][::-1]
    for k in range(span_len):
        bwd_buf[order + k] = float(np.dot(a_bwd, bwd_buf[k : k + order][::-1]))
    bwd_pred = bwd_buf[order:][::-1]

    t = np.linspace(0.0, math.pi, span_len + 2, dtype=np.float64)[1:-1]
    w_fwd = 0.5 * (1.0 + np.cos(t))
    ar_interp = w_fwd * fwd_pred + (1.0 - w_fwd) * bwd_pred

    ctx_peak = max(
        float(np.max(np.abs(left_ctx))),
        float(np.max(np.abs(right_ctx))),
        1e-4,
    )
    if not np.all(np.isfinite(ar_interp)) or float(np.max(np.abs(ar_interp))) > 1.25 * ctx_peak:
        return hermite
    return np.asarray(ar_interp, dtype=np.float64)


def _declick_channel(
    channel: Float32Array,
    sample_rate: int,
    sensitivity: float,
    max_click_ms: float,
) -> tuple[Float32Array, int]:
    """Detect and repair mouth/saliva clicks on a single channel."""
    n = int(channel.shape[0])
    if n < 64 or sensitivity == 0.0:
        return channel.copy(), 0

    x64 = channel.astype(np.float64, copy=False)
    if float(np.max(np.abs(x64))) < 1e-5:
        return channel.copy(), 0

    hp_hz = min(1800.0, 0.4 * float(sample_rate))
    sos_hp = scipy.signal.butter(4, hp_hz, btype="highpass", fs=sample_rate, output="sos")
    hp = np.asarray(scipy.signal.sosfiltfilt(sos_hp, x64), dtype=np.float64)

    ar_order = min(8, max(2, n // 32))
    ar_coeffs = _fit_ar_coefficients(hp, ar_order)
    pred_filter = np.concatenate(([1.0], -ar_coeffs))
    residual = np.asarray(scipy.signal.lfilter(pred_filter, [1.0], hp), dtype=np.float64)
    abs_res = np.abs(residual)

    # Frame-level robust local scale (~1 ms frames, median over >= 3.5x max_click_ms)
    frame_len = max(1, int(round(sample_rate * 0.001)))
    n_frames = max(1, int(math.ceil(n / frame_len)))
    padded_len = n_frames * frame_len
    if padded_len == n:
        frames = abs_res.reshape(n_frames, frame_len)
    else:
        padded = np.pad(abs_res, (0, padded_len - n), mode="edge")
        frames = padded.reshape(n_frames, frame_len)
    frame_scale = np.mean(frames, axis=1)

    med_frames = max(41, int(round(max_click_ms * 3.5)) | 1)
    med_frames = min(med_frames, n_frames if n_frames % 2 == 1 else max(1, n_frames - 1))
    local_frame_base = scipy.ndimage.median_filter(frame_scale, size=med_frames, mode="nearest")
    local_base = np.repeat(local_frame_base, frame_len)[:n]

    # Higher sensitivity -> lower threshold multiplier.
    sigma_mult = 12.0 - 7.0 * sensitivity
    abs_floor = 0.004 * (1.4 - sensitivity)
    threshold = np.maximum(local_base * sigma_mult, abs_floor)

    anomalous = abs_res > threshold
    if not bool(np.any(anomalous)):
        return channel.copy(), 0

    raw_runs = _find_true_runs(anomalous)
    merge_gap = max(2, int(round(sample_rate * 0.0015)))
    merged = _merge_runs(raw_runs, merge_gap)

    max_click_samples = max(2, int(round(sample_rate * max_click_ms / 1000.0)))
    pad_samples = min(max(1, int(round(sample_rate * 0.0005))), max(1, max_click_samples // 6))

    repaired = x64.copy()
    repaired_count = 0
    for start, end in merged:
        span_len = end - start
        if span_len > max_click_samples:
            # Sustained high-frequency energy (e.g. sibilant fricative): leave untouched.
            continue
        rep_start = max(2, start - pad_samples)
        rep_end = min(n - 2, end + pad_samples)
        if rep_end <= rep_start or (rep_end - rep_start) > max_click_samples:
            rep_start = max(2, start)
            rep_end = min(n - 2, end)
        if rep_end <= rep_start:
            continue
        repaired[rep_start:rep_end] = _reconstruct_click_span(repaired, rep_start, rep_end)
        repaired_count += 1

    return np.ascontiguousarray(repaired, dtype=np.float32), repaired_count


def apply_mouth_declick(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    sensitivity: float = 0.5,
    max_click_ms: float = 12.0,
) -> tuple[Float32Array, int]:
    """Detect and repair mouth/saliva clicks using high-passed AR residual + AR/cubic fill."""
    wave_2d, was_1d = _validate_waveform(waveform, sample_rate)
    if not np.isfinite(sensitivity) or not (0.0 <= sensitivity <= 1.0):
        raise ValueError(f"sensitivity must be within [0.0, 1.0], got {sensitivity}")
    if not np.isfinite(max_click_ms) or max_click_ms <= 0.0:
        raise ValueError(f"max_click_ms must be finite and > 0, got {max_click_ms}")

    channels, samples = wave_2d.shape
    if channels == 0 or samples == 0:
        return _restore_shape(wave_2d.copy(), was_1d), 0

    out = np.empty_like(wave_2d)
    total_clicks = 0
    for ch in range(channels):
        repaired, count = _declick_channel(wave_2d[ch], sample_rate, sensitivity, max_click_ms)
        out[ch] = repaired
        total_clicks += count

    return _restore_shape(out, was_1d), total_clicks


def _declip_channel(
    channel: Float32Array,
    clip_threshold: float,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], int]:
    """Reconstruct clipped plateaus (>= 3 consecutive samples) on one channel in float64."""
    x = channel.astype(np.float64, copy=True)
    n = int(x.shape[0])
    if n < 8:
        return x, 0

    pos_runs = [r for r in _find_true_runs(x >= clip_threshold) if (r[1] - r[0]) >= 3]
    neg_runs = [r for r in _find_true_runs(x <= -clip_threshold) if (r[1] - r[0]) >= 3]
    if not pos_runs and not neg_runs:
        return x, 0

    repaired_count = 0
    for polarity, runs in ((1.0, pos_runs), (-1.0, neg_runs)):
        for start, end in runs:
            if start < 2 or end > n - 2:
                continue
            y0 = float(x[start - 1])
            y1 = float(x[end])
            m0 = float(x[start - 1] - x[start - 2])
            m1 = float(x[end + 1] - x[end])

            # Ensure tangents arch outward in the polarity direction of the clipped crest.
            if polarity > 0.0:
                m0 = max(0.0, m0)
                m1 = min(0.0, m1)
            else:
                m0 = min(0.0, m0)
                m1 = max(0.0, m1)

            spline = scipy.interpolate.CubicHermiteSpline([start - 1, end], [y0, y1], [m0, m1])
            idx = np.arange(start, end, dtype=np.float64)
            reconstructed = np.asarray(spline(idx), dtype=np.float64)

            # Bound extreme tangent overshoots while preserving peak smoothness.
            reconstructed = np.clip(reconstructed, -4.0, 4.0)
            if polarity > 0.0:
                reconstructed = np.maximum(reconstructed, min(y0, y1))
            else:
                reconstructed = np.minimum(reconstructed, max(y0, y1))

            x[start:end] = reconstructed
            repaired_count += 1

    return x, repaired_count


def apply_declip(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    clip_threshold: float = 0.985,
) -> tuple[Float32Array, int]:
    """Reconstruct clipped plateaus (>=3 samples) with Hermite splines + safe headroom scaling."""
    wave_2d, was_1d = _validate_waveform(waveform, sample_rate)
    if not np.isfinite(clip_threshold) or not (0.0 < clip_threshold <= 1.0):
        raise ValueError(f"clip_threshold must be within (0.0, 1.0], got {clip_threshold}")

    channels, samples = wave_2d.shape
    if channels == 0 or samples == 0:
        return _restore_shape(wave_2d.copy(), was_1d), 0

    out64 = np.empty((channels, samples), dtype=np.float64)
    total_clips = 0
    for ch in range(channels):
        repaired, count = _declip_channel(wave_2d[ch], clip_threshold)
        out64[ch] = repaired
        total_clips += count

    if total_clips > 0:
        peak = float(np.max(np.abs(out64)))
        if peak > 0.99:
            out64 *= 0.99 / peak

    out32 = np.ascontiguousarray(out64, dtype=np.float32)
    return _restore_shape(out32, was_1d), total_clips


def _extract_ambient_profile(
    orig: np.ndarray[Any, np.dtype[np.float64]],
    clean: np.ndarray[Any, np.dtype[np.float64]],
    sample_rate: int,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], float, float] | None:
    """Extract (median_mag_spectrum, original_ambient_rms, cleaned_floor_rms) from pauses."""
    n = int(orig.shape[0])
    if n < 64 or float(np.max(np.abs(orig))) < 1e-6:
        return None

    frame_len = max(32, min(int(round(sample_rate * 0.020)), n // 2))
    n_frames = n // frame_len
    if n_frames < 2:
        single_rms_o = float(np.sqrt(np.mean(orig * orig)))
        single_rms_c = float(np.sqrt(np.mean(clean * clean)))
        if single_rms_o < 1e-6:
            return None
        n_fft = 1 << max(5, min(10, int(math.floor(math.log2(n)))))
        win = np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(orig[:n_fft] * win))
        return mag, single_rms_o, single_rms_c

    trimmed = n_frames * frame_len
    o_frames = orig[:trimmed].reshape(n_frames, frame_len)
    c_frames = clean[:trimmed].reshape(n_frames, frame_len)

    rms_o = np.sqrt(np.mean(o_frames * o_frames, axis=1) + 1e-18)
    rms_c = np.sqrt(np.mean(c_frames * c_frames, axis=1) + 1e-18)

    # Identify non-speech pause frames via the lowest-energy frames in `cleaned`.
    c_thresh = float(np.percentile(rms_c, 25.0))
    max_c = float(np.max(rms_c))
    if max_c > 1e-4:
        c_thresh = max(c_thresh, 0.04 * max_c)
    pause_mask = rms_c <= c_thresh + 1e-9

    # Reject transient spikes in `original` during pauses (e.g. cough/bump removed by denoiser).
    if int(np.count_nonzero(pause_mask)) >= 2:
        med_pause_o = float(np.median(rms_o[pause_mask]))
        quiet_pause_mask: np.ndarray[Any, np.dtype[np.bool_]] = (
            pause_mask & (rms_o <= max(2.2 * med_pause_o, 1e-5)) & (rms_o >= 1e-6)
        )
    else:
        quiet_pause_mask = np.zeros(n_frames, dtype=np.bool_)

    if int(np.count_nonzero(quiet_pause_mask)) >= 2:
        source_frames = o_frames[quiet_pause_mask]
        ambient_rms = float(np.median(rms_o[quiet_pause_mask]))
        cleaned_floor_rms = float(np.median(rms_c[quiet_pause_mask]))
    else:
        # Fallback for continuous speech without clear pauses: inspect removed noise residual.
        residual = orig[:trimmed] - clean[:trimmed]
        r_frames = residual.reshape(n_frames, frame_len)
        rms_r = np.sqrt(np.mean(r_frames * r_frames, axis=1) + 1e-18)
        r_thresh = float(np.percentile(rms_r, 35.0))
        res_mask = (rms_r <= max(r_thresh, 1e-5)) & (rms_r >= 1e-6)
        if int(np.count_nonzero(res_mask)) == 0:
            return None
        source_frames = r_frames[res_mask]
        ambient_rms = float(np.median(rms_r[res_mask]))
        cleaned_floor_rms = float(np.percentile(rms_c, 15.0))

    if ambient_rms < 1e-6:
        return None

    n_fft = 1 << max(6, min(10, int(math.ceil(math.log2(frame_len)))))
    win = np.hanning(frame_len)
    spectra = np.abs(np.fft.rfft(source_frames * win[np.newaxis, :], n=n_fft, axis=1))
    mag_profile = np.median(spectra, axis=0)
    if float(np.max(mag_profile)) <= 1e-12:
        return None
    return np.asarray(mag_profile, dtype=np.float64), ambient_rms, cleaned_floor_rms


def _synthesize_room_tone(
    mag_profile: np.ndarray[Any, np.dtype[np.float64]],
    n_samples: int,
    target_rms: float,
    seed_key: str,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Synthesize continuous spectrally-matched room tone using overlap-add ISTFT."""
    n_fft = (int(mag_profile.shape[0]) - 1) * 2
    hop = max(1, n_fft // 2)
    seed_int = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)

    n_frames = max(2, int(math.ceil(n_samples / hop)) + 2)
    total_len = (n_frames - 1) * hop + n_fft
    win = np.hanning(n_fft)

    white = rng.standard_normal((n_frames, n_fft))
    shaped = np.fft.irfft(np.fft.rfft(white, axis=1) * mag_profile[np.newaxis, :], n=n_fft, axis=1)
    shaped *= win[np.newaxis, :]

    out = np.zeros(total_len, dtype=np.float64)
    norm = np.zeros(total_len, dtype=np.float64)
    win_sq = win * win
    for i in range(n_frames):
        offset = i * hop
        out[offset : offset + n_fft] += shaped[i]
        norm[offset : offset + n_fft] += win_sq

    # Trim edge ramp of the first hop so the noise bed is stationary from sample 0.
    start = hop
    bed = out[start : start + n_samples]
    bed_norm = np.maximum(norm[start : start + n_samples], 1e-8)
    bed = bed / np.sqrt(bed_norm)

    cur_rms = float(np.sqrt(np.mean(bed * bed)))
    if cur_rms <= 1e-12:
        return np.zeros(n_samples, dtype=np.float64)
    bed *= target_rms / cur_rms
    return np.asarray(bed, dtype=np.float64)


def apply_room_tone_floor(
    original: np.ndarray[Any, Any],
    cleaned: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    target_floor_db: float | None = -50.0,
    blend_ratio: float = 0.0,
) -> Float32Array:
    """Extract clean room tone from pauses in `original` and blend smoothly into `cleaned`.

    Supports two floor modes:
    - Proportional mode when ``blend_ratio > 0.0`` (in ``[0.0, 0.5]``): targets
      ``blend_ratio * ambient_rms``.
    - Explicit floor mode when ``blend_ratio == 0.0`` and ``target_floor_db`` is set:
      targets ``10**(target_floor_db / 20)``.
    - Bypass when ``blend_ratio == 0.0`` and ``target_floor_db is None``.
    """
    orig_2d, orig_1d = _validate_waveform(original, sample_rate, name="original")
    clean_2d, clean_1d = _validate_waveform(cleaned, sample_rate, name="cleaned")
    if orig_1d != clean_1d or orig_2d.shape != clean_2d.shape:
        raise ValueError(
            f"original shape {original.shape} does not match cleaned shape {cleaned.shape}"
        )
    if not np.isfinite(blend_ratio) or not (0.0 <= blend_ratio <= 0.5):
        raise ValueError(f"blend_ratio must be within [0.0, 0.5], got {blend_ratio}")
    if target_floor_db is not None and (
        not np.isfinite(target_floor_db) or target_floor_db > -12.0
    ):
        raise ValueError(f"target_floor_db must be finite and <= -12.0 dB, got {target_floor_db}")

    channels, samples = clean_2d.shape
    if channels == 0 or samples < 64 or (blend_ratio == 0.0 and target_floor_db is None):
        return _restore_shape(clean_2d.copy(), clean_1d)

    # Preserve bit-identity on dual-mono channels while decorrelating true stereo channels.
    seed_keys: list[str] = []
    for ch in range(channels):
        same_as = next(
            (
                prior
                for prior in range(ch)
                if np.array_equal(orig_2d[prior], orig_2d[ch])
                and np.array_equal(clean_2d[prior], clean_2d[ch])
            ),
            None,
        )
        seed_keys.append(seed_keys[same_as] if same_as is not None else f"hawavoclean:rt:ch_{ch}")

    out = clean_2d.astype(np.float64, copy=True)
    for ch in range(channels):
        profile = _extract_ambient_profile(
            orig_2d[ch].astype(np.float64, copy=False),
            clean_2d[ch].astype(np.float64, copy=False),
            sample_rate,
        )
        if profile is None:
            continue
        mag_profile, ambient_rms, cleaned_floor_rms = profile

        if blend_ratio > 0.0:
            target_rms = blend_ratio * ambient_rms
        else:
            assert target_floor_db is not None
            target_rms = float(10.0 ** (target_floor_db / 20.0))

        if cleaned_floor_rms >= target_rms:
            continue

        fill_rms = math.sqrt(
            max(0.0, target_rms * target_rms - cleaned_floor_rms * cleaned_floor_rms)
        )
        if fill_rms <= 1e-8:
            continue

        bed = _synthesize_room_tone(mag_profile, samples, fill_rms, seed_keys[ch])
        out[ch] += bed

    out32 = np.ascontiguousarray(out, dtype=np.float32)
    return _restore_shape(out32, clean_1d)


def apply_pro_repair_chain(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    options: ProRepairOptions | None = None,
    *,
    original_waveform: np.ndarray[Any, Any] | None = None,
) -> tuple[Float32Array, list[str]]:
    """Run the configured surgical repair and optional room-tone floor chain."""
    opts = options if options is not None else ProRepairOptions()
    wave_2d, was_1d = _validate_waveform(waveform, sample_rate)
    current = _restore_shape(wave_2d.copy(), was_1d)
    actions: list[str] = []

    if opts.declip:
        current, clips = apply_declip(current, sample_rate, clip_threshold=opts.clip_threshold)
        if clips > 0:
            actions.append(f"declip(count={clips})")

    if opts.deplosive:
        current, plosives = apply_dynamic_deplosive(
            current,
            sample_rate,
            threshold_db=opts.deplosive_threshold_db,
            max_reduction_db=opts.deplosive_max_reduction_db,
        )
        if plosives > 0:
            actions.append(f"deplosive(count={plosives})")

    if opts.declick:
        current, clicks = apply_mouth_declick(
            current,
            sample_rate,
            sensitivity=opts.declick_sensitivity,
            max_click_ms=opts.declick_max_ms,
        )
        if clicks > 0:
            actions.append(f"mouth_declick(count={clicks})")

    if opts.room_tone_floor and original_waveform is not None:
        before = current
        current = apply_room_tone_floor(
            original_waveform,
            current,
            sample_rate,
            target_floor_db=opts.room_tone_target_floor_db,
            blend_ratio=opts.room_tone_blend_ratio,
        )
        if not np.array_equal(current, before):
            if opts.room_tone_blend_ratio > 0.0:
                actions.append(f"room_tone_floor(blend_ratio={opts.room_tone_blend_ratio:.2f})")
            elif opts.room_tone_target_floor_db is not None:
                actions.append(f"room_tone_floor(target_db={opts.room_tone_target_floor_db:.1f})")

    return np.ascontiguousarray(current, dtype=np.float32), actions


def _raised_cosine_box_1d(
    coords: np.ndarray[Any, np.dtype[np.float64]],
    lo: float,
    hi: float,
    feather: float,
    *,
    taper_lo: bool = True,
    taper_hi: bool = True,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Smooth raised-cosine mask in [0, 1] supported on [lo, hi]."""
    mask = np.zeros(coords.shape, dtype=np.float64)
    inside = (coords >= lo) & (coords <= hi)
    if not bool(np.any(inside)):
        return mask

    mask[inside] = 1.0
    width = hi - lo
    eff_feather = min(max(0.0, feather), 0.25 * width)
    if eff_feather <= 0.0:
        return mask

    if taper_lo:
        lo_zone = inside & (coords < lo + eff_feather)
        if bool(np.any(lo_zone)):
            phase = (coords[lo_zone] - lo) / eff_feather
            mask[lo_zone] *= 0.5 * (1.0 - np.cos(math.pi * phase))

    if taper_hi:
        hi_zone = inside & (coords > hi - eff_feather)
        if bool(np.any(hi_zone)):
            phase = (hi - coords[hi_zone]) / eff_feather
            mask[hi_zone] *= 0.5 * (1.0 - np.cos(math.pi * phase))

    return mask


def _spectral_box_repair_channel(
    channel: Float32Array,
    sample_rate: int,
    *,
    start_s: float,
    end_s: float,
    f_low_hz: float,
    f_high_hz: float,
    attenuation_db: float,
    n_fft: int,
) -> Float32Array:
    """Apply a smooth 2D STFT attenuation box on one channel with bit-identical exterior."""
    n = int(channel.shape[0])
    start_idx = max(0, int(round(start_s * sample_rate)))
    end_idx = min(n, int(round(end_s * sample_rate)))
    if end_idx <= start_idx:
        return channel.copy()

    if n_fft % 8 == 0:
        hop = n_fft // 8
    elif n_fft % 4 == 0:
        hop = n_fft // 4
    else:
        hop = max(1, n_fft // 2)
    noverlap = n_fft - hop

    taper_s = 0.015
    fade_samples = int(round(taper_s * sample_rate))
    splice_start = max(0, start_idx - fade_samples) if start_idx > 0 else 0
    splice_end = min(n, end_idx + fade_samples) if end_idx < n else n

    pad = max(n_fft, fade_samples + hop)
    ctx_start = max(0, splice_start - pad)
    ctx_end = min(n, splice_end + pad)

    slice_x = channel[ctx_start:ctx_end].astype(np.float64, copy=False)
    orig_slice_len = int(slice_x.shape[0])
    if orig_slice_len < n_fft:
        work_x = np.pad(slice_x, (0, n_fft - orig_slice_len), mode="constant")
    else:
        work_x = slice_x

    freqs, times, zxx = scipy.signal.stft(
        work_x,
        fs=float(sample_rate),
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    times_arr = np.asarray(times, dtype=np.float64)
    zxx_arr = np.asarray(zxx, dtype=np.complex128)

    df = float(sample_rate) / float(n_fft)
    nyquist = 0.5 * float(sample_rate)
    f_mask = _raised_cosine_box_1d(
        freqs_arr,
        f_low_hz,
        f_high_hz,
        2.0 * df,
        taper_lo=f_low_hz > 0.0,
        taper_hi=f_high_hz < nyquist,
    )
    if not bool(np.any(f_mask > 0.0)) and freqs_arr.size > 0:
        nearest_bin = int(np.argmin(np.abs(freqs_arr - 0.5 * (f_low_hz + f_high_hz))))
        f_mask[nearest_bin] = 1.0

    t_abs = (ctx_start / float(sample_rate)) + times_arr
    stft_lo = max(0.0, start_s - taper_s) if start_idx > 0 else start_s
    stft_hi = min(n / float(sample_rate), end_s + taper_s) if end_idx < n else end_s
    t_mask = _raised_cosine_box_1d(
        t_abs,
        stft_lo,
        stft_hi,
        taper_s,
        taper_lo=start_idx > 0,
        taper_hi=end_idx < n,
    )
    if not bool(np.any(t_mask > 0.0)) and times_arr.size > 0:
        nearest_frame = int(np.argmin(np.abs(t_abs - 0.5 * (start_s + end_s))))
        t_mask[nearest_frame] = 1.0

    mask_2d = f_mask[:, np.newaxis] * t_mask[np.newaxis, :]
    target_gain = 10.0 ** (-attenuation_db / 20.0)
    gain_2d = 1.0 - mask_2d * (1.0 - target_gain)

    _, reconstructed = scipy.signal.istft(
        zxx_arr * gain_2d,
        fs=float(sample_rate),
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    repaired_slice = np.asarray(reconstructed[:orig_slice_len], dtype=np.float64)

    out = channel.copy()
    splice_len = splice_end - splice_start
    local_start = splice_start - ctx_start
    local_end = splice_end - ctx_start
    fade_in_len = start_idx - splice_start
    fade_out_len = splice_end - end_idx

    blend = np.ones(splice_len, dtype=np.float64)
    if fade_in_len > 0:
        k_in = np.arange(fade_in_len, dtype=np.float64)
        blend[:fade_in_len] = 0.5 * (1.0 - np.cos(math.pi * (k_in + 0.5) / float(fade_in_len)))
    if fade_out_len > 0:
        k_out = np.arange(fade_out_len, dtype=np.float64)
        blend[splice_len - fade_out_len :] = 0.5 * (
            1.0 + np.cos(math.pi * (k_out + 0.5) / float(fade_out_len))
        )

    orig_seg = channel[splice_start:splice_end].astype(np.float64, copy=False)
    rep_seg = repaired_slice[local_start:local_end]
    blended = orig_seg * (1.0 - blend) + rep_seg * blend
    out[splice_start:splice_end] = blended.astype(np.float32)
    return out


def apply_spectral_box_repair(
    waveform: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    start_s: float,
    end_s: float,
    f_low_hz: float,
    f_high_hz: float,
    attenuation_db: float = 24.0,
    n_fft: int = 2048,
) -> Float32Array:
    """Surgically attenuate a 2D time×frequency box via a smooth raised-cosine STFT mask.

    Applies a soft 15 ms raised-cosine time taper and 2-bin frequency feathering
    to prevent ringing/pre-echo while keeping audio outside the crossfaded time
    window bit-identical and audio outside ``[f_low_hz, f_high_hz]`` untouched.
    """
    wave_2d, was_1d = _validate_waveform(waveform, sample_rate)

    for label, val in (
        ("start_s", start_s),
        ("end_s", end_s),
        ("f_low_hz", f_low_hz),
        ("f_high_hz", f_high_hz),
        ("attenuation_db", attenuation_db),
    ):
        if isinstance(val, bool) or not np.isfinite(val):
            raise ValueError(f"{label} must be a finite number, got {val!r}")

    if isinstance(n_fft, bool) or not isinstance(n_fft, int) or n_fft < 64 or n_fft % 2 != 0:
        raise ValueError(f"n_fft must be an even integer >= 64, got {n_fft!r}")

    nyquist = 0.5 * float(sample_rate)
    if start_s < 0.0 or end_s <= start_s:
        raise ValueError(f"time window must satisfy 0 <= start_s < end_s, got ({start_s}, {end_s})")
    if f_low_hz < 0.0 or f_high_hz <= f_low_hz or f_high_hz > nyquist:
        raise ValueError(
            f"frequency band must satisfy 0 <= f_low_hz < f_high_hz <= {nyquist:g}, "
            f"got ({f_low_hz}, {f_high_hz})"
        )
    if not (3.0 <= attenuation_db <= 60.0):
        raise ValueError(f"attenuation_db must be within [3.0, 60.0], got {attenuation_db}")

    channels, samples = wave_2d.shape
    if channels == 0 or samples == 0:
        return _restore_shape(wave_2d.copy(), was_1d)

    out = np.empty_like(wave_2d)
    for ch in range(channels):
        out[ch] = _spectral_box_repair_channel(
            wave_2d[ch],
            sample_rate,
            start_s=float(start_s),
            end_s=float(end_s),
            f_low_hz=float(f_low_hz),
            f_high_hz=float(f_high_hz),
            attenuation_db=float(attenuation_db),
            n_fft=n_fft,
        )

    return _restore_shape(out, was_1d)
