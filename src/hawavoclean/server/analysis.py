"""``POST /api/analyze``: waveform overview, long-term spectrum, loudness.

Decoding goes through the existing ``hawavoclean.audio`` path (ffmpeg with a
soundfile fallback), so any container the pipeline accepts can be analysed.
The spectrum is a long-term average magnitude in 1/12-octave bands, in dB
relative to a full-scale sine (a full-scale sine at a band centre reads
≈ 0 dB); silence clamps at -120 dB.
"""

from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.finishing.loudness import measure_loudness_and_peaks

DEFAULT_BUCKETS = 1200
MAX_BUCKETS = 8000
FLOOR_DB = -120.0
SPECTRUM_N_FFT = 8192
SPECTRUM_FRAMES_PER_CHUNK = 256
BAND_LOW_HZ = 40.0
BAND_HIGH_HZ = 20000.0
BANDS_PER_OCTAVE = 12
# The hann main lobe is 4 bins wide: a band narrower than that can never
# capture a sine's leaked energy, so integration windows are clamped to it.
MIN_BAND_WIDTH_BINS = 4.0
# Analysis is read-only inspection; accept any rate the probe can describe
# (the pipeline enforces its own limits at job time).
ANALYSIS_MAX_SAMPLE_RATE = 384000

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _db(power: FloatArray) -> FloatArray:
    """10·log10 with the contract floor."""
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(np.maximum(power, 0.0))
    return np.asarray(np.where(np.isfinite(out), np.maximum(out, FLOOR_DB), FLOOR_DB))


def bucket_edges(n_samples: int, buckets: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Start/end sample of each bucket. Every bucket covers ≥ 1 sample, so a
    file shorter than ``buckets`` samples still yields ``buckets`` values."""
    starts = (np.arange(buckets, dtype=np.int64) * n_samples) // buckets
    ends = (np.arange(1, buckets + 1, dtype=np.int64) * n_samples) // buckets
    ends = np.maximum(ends, starts + 1)
    starts = np.minimum(starts, n_samples - 1)
    ends = np.minimum(ends, n_samples)
    return starts, ends


def waveform_overview(
    mono: np.ndarray[Any, np.dtype[np.float32]], buckets: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-bucket (min, max, rms_db) over the mono signal."""
    n = int(mono.shape[0])
    starts, ends = bucket_edges(n, buckets)
    x = mono.astype(np.float64, copy=False)
    mins = np.empty(buckets)
    maxs = np.empty(buckets)
    rms_db = np.empty(buckets)
    # Fast path when buckets are uniform-ish: use reduceat on the start indices.
    # reduceat needs strictly increasing indices with the last < n; fall back
    # to a loop for the overlapping (n < buckets) case.
    if n >= buckets:
        mins[:] = np.minimum.reduceat(x, starts)
        maxs[:] = np.maximum.reduceat(x, starts)
        sq = np.add.reduceat(x * x, starts)
        counts = np.diff(np.append(starts, n)).astype(np.float64)
        rms_db[:] = _db(sq / np.maximum(counts, 1.0))
    else:
        for i in range(buckets):
            seg = x[starts[i] : ends[i]]
            mins[i] = float(seg.min())
            maxs[i] = float(seg.max())
            rms_db[i] = float(_db(np.asarray([float(np.mean(seg * seg))]))[0])
    return mins, maxs, rms_db


def band_centres(sample_rate: int) -> FloatArray:
    """1/12-octave centre frequencies from 40 Hz up to min(20 kHz, Nyquist)."""
    f_max = min(BAND_HIGH_HZ, sample_rate / 2.0)
    centres: list[float] = []
    i = 0
    while True:
        fc = BAND_LOW_HZ * (2.0 ** (i / BANDS_PER_OCTAVE))
        if fc > f_max:
            break
        centres.append(fc)
        i += 1
    return np.asarray(centres, dtype=np.float64)


def average_power_spectrum(
    mono: np.ndarray[Any, np.dtype[np.float32]], n_fft: int = SPECTRUM_N_FFT
) -> FloatArray:
    """Long-term average power per rfft bin, normalised so that a full-scale
    sine sums to 1.0 over its bins (Parseval-exact, window-independent)."""
    x = mono.astype(np.float32, copy=False)
    n = int(x.shape[0])
    if n < n_fft:
        x = np.pad(x, (0, n_fft - n))
        n = n_fft
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (n - n_fft) // hop
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for start in range(0, n_frames, SPECTRUM_FRAMES_PER_CHUNK):
        stop = min(n_frames, start + SPECTRUM_FRAMES_PER_CHUNK)
        idx = (np.arange(start, stop)[:, None] * hop) + np.arange(n_fft)[None, :]
        frames = x[idx] * window[None, :]
        spec = np.fft.rfft(frames, axis=1)
        acc += np.sum(np.abs(spec).astype(np.float64) ** 2, axis=0)
    mean_power = acc / float(n_frames)
    norm = 4.0 / (n_fft * float(np.sum(window.astype(np.float64) ** 2)))
    return np.asarray(mean_power * norm, dtype=np.float64)


def band_integrate(
    bin_power: FloatArray, sample_rate: int, n_fft: int, centres: FloatArray
) -> FloatArray:
    """Integrate per-bin power into bands, treating each bin as a uniform
    density over its own width. Bands narrower than the window main lobe
    (low frequencies) widen their integration window to it, centred on the
    band centre: a sine's energy leaks over the whole main lobe, so without
    the clamp a full-scale sine at a 40 Hz band centre would read ~-6 dB
    instead of the contract-calibrated 0 dB."""
    n_bins = bin_power.shape[0]
    df = sample_rate / float(n_fft)
    bin_lo = (np.arange(n_bins) - 0.5) * df
    bin_hi = bin_lo + df
    half = 2.0 ** (1.0 / (2 * BANDS_PER_OCTAVE))
    lo = centres / half
    hi = centres * half
    min_width = MIN_BAND_WIDTH_BINS * df
    too_narrow = (hi - lo) < min_width
    lo = np.where(too_narrow, np.maximum(centres - min_width / 2.0, 0.0), lo)
    hi = np.where(too_narrow, centres + min_width / 2.0, hi)
    out = np.empty(centres.shape[0], dtype=np.float64)
    for i in range(centres.shape[0]):
        overlap = np.clip(np.minimum(bin_hi, hi[i]) - np.maximum(bin_lo, lo[i]), 0.0, None)
        out[i] = float(np.sum(bin_power * (overlap / df)))
    return out


def analyze_audio(path: Path, buckets: int = DEFAULT_BUCKETS) -> dict[str, Any]:
    """Full ``AudioAnalysis`` for ``path`` (contract section 1)."""
    buckets = int(buckets)
    if buckets < 1 or buckets > MAX_BUCKETS:
        raise ValueError(f"buckets must be in 1..{MAX_BUCKETS}, got {buckets}")
    probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    buf = decode_audio(probe)
    sample_rate = buf.sample_rate
    mono = buf.to_mono()

    mins, maxs, rms_db = waveform_overview(mono, buckets)
    centres = band_centres(sample_rate)
    n_fft = SPECTRUM_N_FFT
    bin_power = average_power_spectrum(mono, n_fft=n_fft)
    spectrum_db = _db(band_integrate(bin_power, sample_rate, n_fft, centres))

    loudness = measure_loudness_and_peaks(buf.data, sample_rate)

    live = rms_db[rms_db > FLOOR_DB]
    noise_floor_db = float(np.percentile(live, 10)) if live.size else FLOOR_DB

    return {
        "path": str(path),
        "duration_s": round(buf.duration_s, 4),
        "sample_rate": sample_rate,
        "channels": buf.channels,
        "peaks": {
            "min": [round(float(v), 4) for v in mins],
            "max": [round(float(v), 4) for v in maxs],
        },
        "rms_db": [round(float(v), 2) for v in rms_db],
        "spectrum": {
            "freqs_hz": [round(float(v), 2) for v in centres],
            "db": [round(float(v), 2) for v in spectrum_db],
        },
        "loudness": {
            "integrated_lufs": round(float(loudness.integrated_lufs), 2),
            "true_peak_dbtp": round(float(loudness.true_peak_dbtp), 2),
        },
        "noise_floor_db": round(noise_floor_db, 2),
    }
