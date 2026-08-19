"""``POST /api/analyze``: waveform overview, long-term spectrum, loudness.
``POST /api/peaks``: the same waveform maths over one time window only.

Decoding goes through the existing ``hawavoclean.audio`` path (ffmpeg with a
soundfile fallback), so any container the pipeline accepts can be analysed.
The spectrum is a long-term average magnitude in 1/12-octave bands, in dB
relative to a full-scale sine (a full-scale sine at a band centre reads
≈ 0 dB); silence clamps at -120 dB.
"""

import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.audio.decode import decode_audio, decode_audio_window, window_sample_bounds
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioProbeResult
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

# /api/analyze rounds the overview to 4 decimals; a zoomed window is where a
# client finally sees individual samples, so it keeps ~6 decimals (well below
# a 24-bit LSB) — otherwise deep zoom would show a quantisation staircase.
WINDOW_PEAK_DECIMALS = 6
# A window longer than this is bucketed by streaming reduction instead of one
# decode. Nothing on screen is ever this long (87 s at 48 kHz), but a client
# asking for a whole 3-hour file as one "window" must not cost 8.5 GB — which
# is exactly what one decode of that span measured before this was added.
# Chunked, peak RSS is constant in file length; measured on a 3-hour file,
# 4 Mi samples = 155 MB / 5.6 s, 2 Mi = 79 MB / 8.7 s, 1 Mi = 40 MB / 14.8 s.
WINDOW_CHUNK_SAMPLES = 4 * 1024 * 1024


PROBE_CACHE_SIZE = 8
_probe_cache: "OrderedDict[tuple[str, int, int], AudioProbeResult]" = OrderedDict()
_probe_cache_lock = threading.Lock()


class PeaksWindowError(ValueError):
    """A ``/api/peaks`` window request is unusable (bad range or bucket count).

    The route turns this into ``400 {"error":"bad_request"}``.
    """


def cached_probe(path: Path) -> AudioProbeResult:
    """``probe_audio`` for a file that is about to be windowed repeatedly.

    Probing SHA-256s the whole file — 0.8 s on a 2 GB three-hour recording —
    which is nothing next to a full decode but is the entire cost of serving a
    5 s window. A zoom gesture is a burst of windows over one file, so the last
    few probes are remembered, keyed on identity that changes whenever the file
    does (path + mtime + size). Bounded, so it cannot grow into a leak.
    """
    try:
        st = path.stat()
    except OSError:  # pragma: no cover - the route has already stat()ed the file
        return probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _probe_cache_lock:
        hit = _probe_cache.get(key)
        if hit is not None:
            _probe_cache.move_to_end(key)
            return hit
    probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    with _probe_cache_lock:
        _probe_cache[key] = probe
        _probe_cache.move_to_end(key)
        while len(_probe_cache) > PROBE_CACHE_SIZE:
            _probe_cache.popitem(last=False)
    return probe


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
        "channels": probe.channels,
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


def _chunked_overview(
    probe: AudioProbeResult, start: int, end: int, buckets: int
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    """``waveform_overview`` over ``[start, end)`` without ever holding the span.

    Buckets are a streaming reduction — running min, max and sum of squares —
    so a window of any length costs one chunk of memory. Chunks are decoded
    back to back, so a chunk that comes up short can only be the last one.
    """
    n = end - start
    starts, ends = bucket_edges(n, buckets)
    mins = np.full(buckets, np.inf)
    maxs = np.full(buckets, -np.inf)
    sumsq = np.zeros(buckets)
    counts = np.zeros(buckets, dtype=np.int64)
    sample_rate = probe.sample_rate

    pos = 0
    while pos < n:
        stop = min(pos + WINDOW_CHUNK_SAMPLES, n)
        requested = stop - pos
        chunk = decode_audio_window(
            probe, (start + pos) / sample_rate, (start + stop) / sample_rate
        ).to_mono()
        got = int(chunk.shape[0])
        x = chunk.astype(np.float64, copy=False)
        # Buckets are contiguous and sorted, so every index in [first, last)
        # genuinely overlaps this chunk.
        first = int(np.searchsorted(ends, pos, side="right"))
        last = int(np.searchsorted(starts, pos + got, side="left"))
        for i in range(first, last):
            lo = max(int(starts[i]), pos) - pos
            hi = min(int(ends[i]), pos + got) - pos
            seg = x[lo:hi]
            mins[i] = min(mins[i], float(seg.min()))
            maxs[i] = max(maxs[i], float(seg.max()))
            sumsq[i] += float(np.dot(seg, seg))
            counts[i] += hi - lo
        del chunk, x
        pos += got
        if got < requested:  # short read: the stream ended early
            break

    covered = int(counts.sum())
    keep = int(np.count_nonzero(counts))
    mins, maxs, sumsq, counts = mins[:keep], maxs[:keep], sumsq[:keep], counts[:keep]
    rms_db = _db(sumsq / np.maximum(counts.astype(np.float64), 1.0))
    return mins, maxs, rms_db, covered


def _window_overview(
    probe: AudioProbeResult, start: int, end: int, buckets: int
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    """Per-bucket (min, max, rms_db) plus the sample count actually covered."""
    if end - start > WINDOW_CHUNK_SAMPLES:
        return _chunked_overview(probe, start, end, buckets)
    sample_rate = probe.sample_rate
    mono = decode_audio_window(probe, start / sample_rate, end / sample_rate).to_mono()
    got = int(mono.shape[0])
    mins, maxs, rms_db = waveform_overview(mono, min(buckets, got))
    return mins, maxs, rms_db, got


def compute_peaks_window(
    path: Path,
    start_s: float,
    end_s: float,
    buckets: int = DEFAULT_BUCKETS,
) -> dict[str, Any]:
    """``PeaksWindow`` for ``path`` over ``[start_s, end_s)`` (contract addendum 1).

    Only the requested span is decoded — never the whole file — so the cost of
    a zoom re-query scales with what is on screen, and a span too long to hold
    at once is reduced chunk by chunk. ``buckets`` is clamped down to the
    number of samples in the window, which makes every bucket cover at least
    one sample; ``samples_per_bucket`` is the widest bucket, so a client that
    sees 1 knows it is looking at raw samples and cannot zoom further.
    """
    buckets = int(buckets)
    if buckets < 1 or buckets > MAX_BUCKETS:
        raise PeaksWindowError(f"buckets must be in 1..{MAX_BUCKETS}, got {buckets}")
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise PeaksWindowError(f"start_s and end_s must be finite, got {start_s}, {end_s}")
    if start_s < 0.0:
        raise PeaksWindowError(f"start_s must be >= 0, got {start_s}")
    if end_s <= start_s:
        raise PeaksWindowError(f"end_s must be greater than start_s, got {start_s}, {end_s}")

    probe = cached_probe(path)
    sample_rate = probe.sample_rate
    duration_s = probe.samples / sample_rate
    if start_s >= duration_s:
        raise PeaksWindowError(
            f"start_s {start_s} is at or past the end of the file ({duration_s:.4f} s)"
        )

    # end_s is clamped to the duration; the sample grid is shared with the
    # decoder so the reported span is exactly what the buckets cover.
    start_sample, end_sample = window_sample_bounds(probe, start_s, min(end_s, duration_s))
    buckets = min(buckets, end_sample - start_sample)
    mins, maxs, rms_db, n = _window_overview(probe, start_sample, end_sample, buckets)
    buckets = int(mins.shape[0])
    samples_per_bucket = -(-n // buckets)  # ceil: 1 only when every bucket is one sample

    return {
        "path": str(path),
        "start_s": round(start_sample / sample_rate, WINDOW_PEAK_DECIMALS),
        "end_s": round((start_sample + n) / sample_rate, WINDOW_PEAK_DECIMALS),
        "sample_rate": sample_rate,
        "channels": probe.channels,
        "duration_s": round(duration_s, 4),
        "samples_per_bucket": samples_per_bucket,
        "peaks": {
            "min": [round(float(v), WINDOW_PEAK_DECIMALS) for v in mins],
            "max": [round(float(v), WINDOW_PEAK_DECIMALS) for v in maxs],
        },
        "rms_db": [round(float(v), 2) for v in rms_db],
    }
