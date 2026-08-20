"""Memory-bounded oversampled true-peak measurement.

Oversampling a whole file at 4x/8x in float64 costs tens of bytes per input
sample per copy; on long recordings that reached gigabytes. These helpers
process fixed-size chunks with overlap (so the polyphase filter's edge
transient never lands inside a kept region) and keep memory proportional to
the chunk, not the file.

The chunks are independent: every kept ("core") sample is computed with
``EDGE`` real samples of context on each side, against a polyphase FIR whose
half-length is ten input samples, so a core value is bit-for-bit the same
whatever chunk it lands in and wherever that chunk starts. That independence
is what lets the chunks run on a bounded thread pool — ``scipy``'s ``upfirdn``
releases the GIL, so the oversampling scales across cores, and each worker
writes a disjoint slice of the output. The result does not depend on the
chunk size, the worker count, or the machine: see
``tests/unit/test_truepeak_parallel.py``, which pins that invariance.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import scipy.signal

CHUNK = 1 << 20  # 1,048,576 samples per chunk (~22 s at 48 kHz)
EDGE = 4096  # overlap on each side; comfortably beyond resample_poly's FIR half-length
MIN_CHUNK = 1 << 16  # never split so fine that the EDGE overlap dominates the work
MAX_WORKERS = 8  # oversampling is memory-bandwidth bound; more threads stop paying
INFLIGHT_BYTES = 64 << 20  # ceiling on oversampled audio held across all workers


def _plan(channels: int, samples: int, factor: int) -> tuple[int, int]:
    """Choose (chunk_size, worker_count) under a fixed in-flight memory budget.

    Only performance depends on this: the envelope itself is invariant to the
    chunk size, so a machine with more cores computes the same numbers.
    """
    workers = max(1, min(MAX_WORKERS, os.cpu_count() or 1))
    per_sample = channels * factor * 4  # oversampled float32 bytes per input sample
    while workers > 1 and INFLIGHT_BYTES // (workers * per_sample) - 2 * EDGE < MIN_CHUNK:
        workers -= 1
    budget = INFLIGHT_BYTES // (workers * per_sample) - 2 * EDGE
    even_split = -(-samples // workers)
    chunk = max(MIN_CHUNK, min(CHUNK, budget, even_split))
    n_chunks = -(-samples // chunk)
    return chunk, max(1, min(workers, n_chunks))


def _core_envelope(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    factor: int,
    samples: int,
    start: int,
    end: int,
    out: np.ndarray[Any, np.dtype[np.float32]],
) -> None:
    """Fill ``out[start:end]`` from one overlapped chunk. Touches no other index."""
    lo = max(0, start - EDGE)
    hi = min(samples, end + EDGE)
    piece = waveform[:, lo:hi].astype(np.float32, copy=False)
    over = scipy.signal.resample_poly(piece, up=factor, down=1, axis=-1)
    np.abs(over, out=over)  # resample_poly hands us a fresh array; fold in place
    env = np.max(over, axis=0)  # (len(piece)*factor,)
    # Fold to one value per original sample, then keep only the core.
    core_lo = (start - lo) * factor
    core_hi = core_lo + (end - start) * factor
    core = env[core_lo:core_hi]
    pad = (-len(core)) % factor
    if pad:
        core = np.pad(core, (0, pad), mode="edge")
    out[start:end] = core.reshape(-1, factor).max(axis=1)[: end - start]


def oversampled_peak_envelope(
    waveform: np.ndarray[Any, np.dtype[np.float32]],  # (channels, samples)
    factor: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Per-sample max-abs over the `factor`x oversampled signal, across channels.

    Returns an array of length `samples`: for each original sample i, the
    maximum absolute oversampled value in its block [i*factor, (i+1)*factor).
    Computed chunk-wise in float32, on a bounded thread pool.
    """
    channels, samples = waveform.shape
    out = np.zeros(samples, dtype=np.float32)
    if samples == 0:
        return out
    chunk, workers = _plan(channels, samples, factor)
    bounds = [(s, min(samples, s + chunk)) for s in range(0, samples, chunk)]
    if workers <= 1 or len(bounds) <= 1:
        for start, end in bounds:
            _core_envelope(waveform, factor, samples, start, end, out)
        return out

    def run(span: tuple[int, int]) -> None:
        _core_envelope(waveform, factor, samples, span[0], span[1], out)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # list() drains the iterator, so a worker exception is re-raised here
        # rather than being swallowed — the caller's fail-closed path sees it.
        list(pool.map(run, bounds))
    return out


def true_peak_linear(waveform: np.ndarray[Any, np.dtype[np.float32]], factor: int = 4) -> float:
    """Scalar oversampled true peak (linear), memory-bounded."""
    if waveform.size == 0:
        return 0.0
    return float(np.max(oversampled_peak_envelope(waveform, factor)))
