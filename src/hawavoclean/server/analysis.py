"""``POST /api/analyze``: waveform overview, long-term spectrum, loudness.
``POST /api/peaks``: the same waveform maths over one time window only.

Decoding goes through the existing ``hawavoclean.audio`` path (ffmpeg with a
soundfile fallback), so any container the pipeline accepts can be analysed.
The spectrum is a long-term average magnitude in 1/12-octave bands, in dB
relative to a full-scale sine (a full-scale sine at a band centre reads
≈ 0 dB); silence clamps at -120 dB.

**Nothing here ever holds a whole file.** ``analyze`` used to decode the file
into memory, which cost 12.8 GB of peak RSS on a three-hour recording; all
four products it returns are now streaming reductions over
:func:`iter_decode_audio` chunks — overview buckets, the long-term average
spectrum, the BS.1770 gated block statistics and the oversampled true peak.
Each accumulator is written to land on exactly the grid its whole-file
counterpart uses, so the numbers are the same numbers (see
``tests/unit/test_server_analyze_streaming.py``, which asserts it).
"""

import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import scipy.signal

from hawavoclean.audio.decode import (
    DECODE_CHUNK_SAMPLES,
    decode_audio_window,
    iter_decode_audio,
    window_sample_bounds,
)
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioProbeResult
from hawavoclean.finishing.loudness import LoudnessMeasurement
from hawavoclean.finishing.truepeak import EDGE as TRUE_PEAK_EDGE
from hawavoclean.finishing.truepeak import oversampled_peak_envelope

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


# ------------------------------------------------------- streaming reductions

# One decoded chunk fed to every accumulator in turn. The accumulators together
# hold a few multiples of it, which is why a three-hour analyze costs 0.2 GB of
# peak RSS instead of 12.8 GB.
ANALYZE_CHUNK_SAMPLES = DECODE_CHUNK_SAMPLES

# BS.1770-4 gating, mirrored from ``pyloudnorm.Meter`` so the streaming
# accumulation lands on exactly the block grid pyloudnorm would have used.
BLOCK_SIZE_S = 0.400
BLOCK_OVERLAP = 0.75
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
CHANNEL_GAINS = (1.0, 1.0, 1.0, 1.41, 1.41)
MAX_LOUDNESS_CHANNELS = len(CHANNEL_GAINS)
TRUE_PEAK_FACTOR = 4


def block_bounds(j: int, sample_rate: int) -> tuple[int, int]:
    """Sample range ``[lo, hi)`` of gating block ``j``.

    Written with the same operand order as ``pyloudnorm.Meter`` (``T_g * (j *
    step) * rate`` truncated by ``int``) because the truncation of a binary
    float is not associative: computing ``0.1 * j * rate`` instead would move
    some block edges by one sample.
    """
    step = 1.0 - BLOCK_OVERLAP
    return (
        int(BLOCK_SIZE_S * (j * step) * sample_rate),
        int(BLOCK_SIZE_S * (j * step + 1) * sample_rate),
    )


class _BucketReducer:
    """Streaming per-bucket (min, max, sum of squares) over a sample range.

    Buckets are laid out over ``n_expected`` up front — the reduction has to
    know where the boundaries are before it has seen the last sample — so
    ``n_expected`` is the *container* sample count, the same grid ``/api/peaks``
    and the ``<audio>`` element use. For PCM and FLAC that is also the decoded
    sample count, so the result is identical to reducing the whole decoded
    array. A lossy decoder that runs past the declared length (AAC emits 71
    samples of frame padding past it in the project's test file) has that tail
    dropped rather than folded into the last bucket, because a bucket grid that
    disagreed with the playhead by 1.5 ms is what this endpoint used to do.
    A stream that stops *early* is stretched back over the buckets it did cover.
    """

    def __init__(self, n_expected: int, buckets: int) -> None:
        self.buckets = int(buckets)
        self.n_expected = max(int(n_expected), 1)
        self.seen = 0
        self.covered = 0
        self.starts, self.ends = bucket_edges(self.n_expected, self.buckets)
        self.mins = np.full(self.buckets, np.inf)
        self.maxs = np.full(self.buckets, -np.inf)
        self.sumsq = np.zeros(self.buckets)
        self.counts = np.zeros(self.buckets, dtype=np.int64)

    def push(self, offset: int, mono: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        """Fold ``mono`` — the samples at absolute index ``offset`` — into the buckets."""
        got = int(mono.shape[0])
        if got == 0:
            return
        self.seen += got
        x = mono.astype(np.float64, copy=False)
        first = int(np.searchsorted(self.ends, offset, side="right"))
        last = int(np.searchsorted(self.starts, offset + got, side="left"))
        for i in range(first, last):
            lo = max(int(self.starts[i]), offset) - offset
            hi = min(int(self.ends[i]), offset + got) - offset
            if hi <= lo:
                continue
            seg = x[lo:hi]
            self.mins[i] = min(self.mins[i], float(seg.min()))
            self.maxs[i] = max(self.maxs[i], float(seg.max()))
            self.sumsq[i] += float(np.dot(seg, seg))
            self.counts[i] += hi - lo

    def finish(self, *, trim: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``trim`` keeps only the buckets that received samples (``/api/peaks``,
        whose bucket count is advisory); otherwise the covered prefix is
        stretched back over the full bucket count, because ``/api/analyze``
        promises exactly ``buckets`` values."""
        # Distinct samples the grid covers. Not ``counts.sum()``: a file with
        # fewer samples than buckets has overlapping buckets, and that sum
        # would count the same sample several times and inflate the duration.
        self.covered = min(self.seen, self.n_expected)
        keep = int(np.count_nonzero(self.counts))
        mins, maxs = self.mins[:keep], self.maxs[:keep]
        rms_db = _db(self.sumsq[:keep] / np.maximum(self.counts[:keep].astype(np.float64), 1.0))
        if not trim and 0 < keep < self.buckets:
            idx = (np.arange(self.buckets) * keep) // self.buckets
            mins, maxs, rms_db = mins[idx], maxs[idx], rms_db[idx]
        return mins, maxs, rms_db


class _SpectrumAccumulator:
    """Long-term average power spectrum, one STFT frame at a time.

    The average over the whole file is a running sum of per-frame power over a
    frame count — mathematically identical to averaging the whole file's frames
    at the end, because every frame contributes exactly once and the frame grid
    (``hop = n_fft/2`` from sample 0) does not depend on where the chunk edges
    fell. The tail that is shorter than a frame is carried into the next chunk.
    """

    def __init__(self, n_fft: int = SPECTRUM_N_FFT) -> None:
        self.n_fft = int(n_fft)
        self.hop = self.n_fft // 2
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.acc = np.zeros(self.n_fft // 2 + 1, dtype=np.float64)
        self.frames = 0
        self.total = 0
        self._tail: np.ndarray[Any, np.dtype[np.float32]] = np.empty(0, dtype=np.float32)

    def _transform(self, buf: np.ndarray[Any, np.dtype[np.float32]], n_frames: int) -> None:
        for start in range(0, n_frames, SPECTRUM_FRAMES_PER_CHUNK):
            stop = min(n_frames, start + SPECTRUM_FRAMES_PER_CHUNK)
            idx = (np.arange(start, stop)[:, None] * self.hop) + np.arange(self.n_fft)[None, :]
            frames = buf[idx] * self.window[None, :]
            spec = np.fft.rfft(frames, axis=1)
            self.acc += np.sum(np.abs(spec).astype(np.float64) ** 2, axis=0)

    def push(self, mono: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        self.total += int(mono.shape[0])
        buf = np.concatenate((self._tail, mono)) if self._tail.size else mono
        n = int(buf.shape[0])
        if n >= self.n_fft:
            new = 1 + (n - self.n_fft) // self.hop
            self._transform(buf, new)
            self.frames += new
            buf = buf[new * self.hop :]
        # An explicit copy, not a view: a slice of the chunk would keep the
        # whole chunk alive for one more iteration for the sake of < n_fft
        # samples, which is exactly the kind of retention this file exists to
        # avoid.
        self._tail = np.array(buf, dtype=np.float32, copy=True)

    def finish(self) -> FloatArray:
        if self.frames == 0:
            # Shorter than one FFT: the whole-file path zero-pads to n_fft.
            padded = np.pad(self._tail, (0, self.n_fft - int(self._tail.shape[0])))
            self._transform(padded, 1)
            self.frames = 1
        mean_power = self.acc / float(self.frames)
        norm = 4.0 / (self.n_fft * float(np.sum(self.window.astype(np.float64) ** 2)))
        self._tail = np.empty(0, dtype=np.float32)
        return np.asarray(mean_power * norm, dtype=np.float64)


class _TruePeakAccumulator:
    """Oversampled true peak over a stream, exact to the whole-file value.

    ``oversampled_peak_envelope`` is already chunked internally with ``EDGE``
    samples of context on each side of every core region; the only thing a
    stream adds is that chunk edges are also *buffer* edges. So the accumulator
    keeps ``EDGE`` samples of decoded audio behind the last finalised sample
    and another ``EDGE`` ahead of it, and only ever trusts envelope values that
    had real audio on both sides. ``EDGE`` is 4096 samples against a polyphase
    FIR whose half-length is ten input samples, so this is exact, not close.
    """

    def __init__(self, factor: int = TRUE_PEAK_FACTOR) -> None:
        self.factor = factor
        self.peak = 0.0
        self._buf: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._context = 0

    def _take(self, buf: np.ndarray[Any, np.dtype[np.float32]], stop: int) -> None:
        env = oversampled_peak_envelope(buf, self.factor)
        core = env[self._context : stop]
        if core.size:
            self.peak = max(self.peak, float(core.max()))

    def push(self, data: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        buf = data if self._buf is None else np.concatenate((self._buf, data), axis=1)
        n = int(buf.shape[1])
        ready = max(self._context, n - TRUE_PEAK_EDGE)
        if ready > self._context:
            self._take(buf, ready)
        keep_from = max(0, ready - TRUE_PEAK_EDGE)
        # Copy for the same reason as the spectrum tail: a mono slice of the
        # chunk is still contiguous, so a view here would pin the chunk.
        self._buf = np.array(buf[:, keep_from:], dtype=np.float32, copy=True)
        self._context = ready - keep_from

    def finish(self) -> float:
        if self._buf is not None and int(self._buf.shape[1]) > self._context:
            self._take(self._buf, int(self._buf.shape[1]))
        self._buf = None
        return self.peak


class _LoudnessAccumulator:
    """BS.1770 integrated loudness + peaks, accumulated block by block.

    ``pyloudnorm`` needs the whole signal because it K-weights it, squares it
    into 400 ms blocks at 75 % overlap, and only then applies the absolute
    (-70 LUFS) and relative (-10 LU) gates. Only the *gates* need all the
    blocks — the block statistics themselves are a per-sample reduction. So
    this keeps the two K-weighting biquads' filter state across chunks (an IIR
    split with ``lfilter``'s ``zi`` is exact, not approximate), accumulates one
    mean square per block per channel, and gates at the end exactly as
    pyloudnorm does. The block list is 108 000 entries for three hours: under
    a megabyte.

    The coefficients come from ``pyloudnorm.Meter`` itself rather than being
    re-derived here, so a pyloudnorm upgrade cannot silently desynchronise the
    two paths.
    """

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.total = 0
        self.sample_peak = 0.0
        self.sum_squares = np.zeros(max(self.channels, 1), dtype=np.float64)
        self._blocks: list[FloatArray] = []
        self._stages: list[tuple[FloatArray, FloatArray, float, list[FloatArray]]] = []
        # pyloudnorm raises for >5 channels and the caller maps that to -70.
        self.supported = 0 < self.channels <= MAX_LOUDNESS_CHANNELS
        if self.supported:
            meter = pyln.Meter(self.sample_rate)
            for stage in meter._filters.values():  # noqa: SLF001 - no public accessor
                b = np.asarray(stage.b, dtype=np.float64)
                a = np.asarray(stage.a, dtype=np.float64)
                order = max(len(a), len(b)) - 1
                zi = [np.zeros(order, dtype=np.float64) for _ in range(self.channels)]
                self._stages.append((b, a, float(stage.passband_gain), zi))

    def push(self, data: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        n = int(data.shape[1])
        if n == 0:
            return
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
            for ch in range(self.channels):
                filtered, zi[ch] = scipy.signal.lfilter(b, a, weighted[ch], zi=zi[ch])
                # pyloudnorm writes each stage back into a float32 array, so the
                # quantisation between the two biquads is part of the reference.
                out[ch] = (gain * filtered).astype(np.float32)
            weighted = out
        wide = weighted.astype(np.float64, copy=False)

        span = BLOCK_SIZE_S * (1.0 - BLOCK_OVERLAP) * self.sample_rate
        first = max(0, int((offset - BLOCK_SIZE_S * self.sample_rate) / span) - 2)
        last = int((offset + n) / span) + 2
        for j in range(first, last + 1):
            lo, hi = block_bounds(j, self.sample_rate)
            start = max(lo, offset)
            stop = min(hi, offset + n)
            if stop <= start:
                continue
            while len(self._blocks) <= j:
                self._blocks.append(np.zeros(self.channels))
            seg = wide[:, start - offset : stop - offset]
            self._blocks[j] += np.einsum("cs,cs->c", seg, seg)

    def _integrated_lufs(self) -> float:
        """The gates, applied exactly as ``pyloudnorm.Meter`` applies them."""
        if not self.supported:
            return -70.0
        step = 1.0 - BLOCK_OVERLAP
        duration_s = self.total / self.sample_rate
        n_blocks = int(np.round((duration_s - BLOCK_SIZE_S) / (BLOCK_SIZE_S * step)) + 1)
        if n_blocks <= 0:
            return -70.0
        z = np.zeros((self.channels, n_blocks), dtype=np.float64)
        for j in range(min(n_blocks, len(self._blocks))):
            z[:, j] = self._blocks[j] / (BLOCK_SIZE_S * self.sample_rate)
        gains = np.asarray(CHANNEL_GAINS[: self.channels], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            loud = -0.691 + 10.0 * np.log10(gains @ z)
            above_absolute = loud >= ABSOLUTE_GATE_LUFS
            if not bool(np.any(above_absolute)):
                return -70.0
            z_gated = z[:, above_absolute].mean(axis=1)
            relative = -0.691 + 10.0 * np.log10(float(gains @ z_gated)) + RELATIVE_GATE_LU
            keep = (loud > relative) & (loud > ABSOLUTE_GATE_LUFS)
            if not bool(np.any(keep)):
                return -70.0
            lufs = float(-0.691 + 10.0 * np.log10(float(gains @ z[:, keep].mean(axis=1))))
        if math.isnan(lufs) or math.isinf(lufs):
            return -70.0
        return lufs

    def finish(self, true_peak: float) -> LoudnessMeasurement:
        """Same contract as ``finishing.loudness.measure_loudness_and_peaks``,
        including its short-signal branch (under 400 ms there are no gating
        blocks, so it reports the ungated mean-square loudness instead)."""
        peak_db = float(20.0 * np.log10(self.sample_peak + 1e-9))
        if self.total < self.sample_rate * BLOCK_SIZE_S:
            if self.sample_peak < 1e-4:
                return LoudnessMeasurement(
                    integrated_lufs=-70.0, sample_peak_dbfs=peak_db, true_peak_dbtp=peak_db
                )
            mean_sq = float(np.sum(self.sum_squares / max(self.total, 1)))
            return LoudnessMeasurement(
                integrated_lufs=float(-0.691 + 10.0 * np.log10(mean_sq + 1e-20)),
                sample_peak_dbfs=peak_db,
                true_peak_dbtp=float(20.0 * np.log10(true_peak + 1e-9)),
            )
        return LoudnessMeasurement(
            integrated_lufs=self._integrated_lufs(),
            sample_peak_dbfs=peak_db,
            true_peak_dbtp=float(20.0 * np.log10(true_peak + 1e-9)),
        )


def stream_measurements(
    probe: AudioProbeResult, buckets: int, chunk_samples: int = ANALYZE_CHUNK_SAMPLES
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, LoudnessMeasurement, int]:
    """One decode pass, four reductions: buckets, spectrum, loudness, true peak.

    Returns ``(mins, maxs, rms_db, bin_power, loudness, covered_samples)``,
    where ``covered_samples`` is the span the bucket grid covers — the decoded
    length, clamped to the container's declared length so that the overview,
    ``/api/peaks`` and the audio element all describe the same timeline.
    Spectrum, loudness and true peak see *every* decoded sample, which is what
    makes them bit-comparable with the whole-file functions. The file is read
    once; nothing larger than a chunk is ever resident.
    """
    overview = _BucketReducer(int(probe.samples), buckets)
    spectrum = _SpectrumAccumulator(SPECTRUM_N_FFT)
    loudness = _LoudnessAccumulator(probe.sample_rate, probe.channels)
    true_peak = _TruePeakAccumulator()

    total = 0
    for chunk in iter_decode_audio(probe, chunk_samples):
        mono = chunk.to_mono()
        overview.push(total, mono)
        spectrum.push(mono)
        loudness.push(chunk.data)
        true_peak.push(chunk.data)
        total += int(mono.shape[0])
        del mono, chunk

    mins, maxs, rms_db = overview.finish(trim=False)
    return (
        mins,
        maxs,
        rms_db,
        spectrum.finish(),
        loudness.finish(true_peak.finish()),
        overview.covered,
    )


def analyze_audio(path: Path, buckets: int = DEFAULT_BUCKETS) -> dict[str, Any]:
    """Full ``AudioAnalysis`` for ``path`` (contract section 1).

    The file is decoded once, as a stream: peak RSS is a function of the chunk
    size, not of the file. Measured on a 3 h / 2.07 GB recording, 12.76 GB ->
    0.23 GB. The response is byte-for-byte the shape it always was.
    """
    buckets = int(buckets)
    if buckets < 1 or buckets > MAX_BUCKETS:
        raise ValueError(f"buckets must be in 1..{MAX_BUCKETS}, got {buckets}")
    probe = cached_probe(path)
    sample_rate = probe.sample_rate

    mins, maxs, rms_db, bin_power, loudness, covered = stream_measurements(probe, buckets)

    centres = band_centres(sample_rate)
    spectrum_db = _db(band_integrate(bin_power, sample_rate, SPECTRUM_N_FFT, centres))

    live = rms_db[rms_db > FLOOR_DB]
    noise_floor_db = float(np.percentile(live, 10)) if live.size else FLOOR_DB

    return {
        "path": str(path),
        "duration_s": round(covered / sample_rate, 4),
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
    reducer = _BucketReducer(n, buckets)
    sample_rate = probe.sample_rate

    pos = 0
    while pos < n:
        stop = min(pos + WINDOW_CHUNK_SAMPLES, n)
        requested = stop - pos
        chunk = decode_audio_window(
            probe, (start + pos) / sample_rate, (start + stop) / sample_rate
        ).to_mono()
        got = int(chunk.shape[0])
        reducer.push(pos, chunk)
        del chunk
        pos += got
        if got < requested:  # short read: the stream ended early
            break

    mins, maxs, rms_db = reducer.finish(trim=True)
    return mins, maxs, rms_db, reducer.covered


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
