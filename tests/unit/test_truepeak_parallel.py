"""The oversampled peak envelope must not depend on how the work was split.

``oversampled_peak_envelope`` splits the file into overlapped chunks and runs
them on a thread pool. That is only safe because a chunk's kept region is
computed with ``EDGE`` real samples of context on each side, so its values do
not depend on the chunk size, the number of workers, or the machine. These
tests pin exactly that: same numbers, bit for bit, under every split — and a
worker failure propagating rather than silently leaving zeros in the output.
"""

import threading
from typing import Any

import numpy as np
import pytest
import scipy.signal

from hawavoclean.finishing import truepeak
from hawavoclean.finishing.truepeak import (
    EDGE,
    _plan,
    oversampled_peak_envelope,
    true_peak_linear,
)

FloatArray = np.ndarray[Any, np.dtype[np.float32]]


def _unchunked_envelope(waveform: FloatArray, factor: int) -> FloatArray:
    """Reference: oversample the whole file at once, no chunking, no threads."""
    over = scipy.signal.resample_poly(waveform, up=factor, down=1, axis=-1)
    env = np.max(np.abs(over), axis=0)
    samples = waveform.shape[1]
    pad = (-len(env)) % factor
    if pad:
        env = np.pad(env, (0, pad), mode="edge")
    return np.asarray(env.reshape(-1, factor).max(axis=1)[:samples], dtype=np.float32)


def _signal(channels: int, samples: int, seed: int = 4) -> FloatArray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples, dtype=np.float64)
    base = 0.6 * np.sin(2 * np.pi * 997.0 * t / 48000.0)
    out = np.empty((channels, samples), dtype=np.float32)
    for c in range(channels):
        out[c] = (base + 0.05 * rng.standard_normal(samples)).astype(np.float32)
    return out


@pytest.mark.unit
@pytest.mark.parametrize("factor", [4, 8])
@pytest.mark.parametrize("channels", [1, 2])
def test_envelope_matches_unchunked_reference(channels: int, factor: int) -> None:
    """Chunk + thread split reproduces the whole-file transform exactly."""
    wave = _signal(channels, 40_000)
    assert np.array_equal(
        oversampled_peak_envelope(wave, factor), _unchunked_envelope(wave, factor)
    )


@pytest.mark.unit
@pytest.mark.parametrize("factor", [4, 8])
def test_envelope_is_invariant_to_chunk_size_and_worker_count(
    factor: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every split of the same audio yields bit-identical values."""
    wave = _signal(2, 300_000)
    reference = oversampled_peak_envelope(wave, factor)
    for chunk in (1 << 13, 1 << 14, 1 << 16, 1 << 18, 1 << 22):
        for workers in (1, 2, 3, 8):
            monkeypatch.setattr(truepeak, "CHUNK", chunk)
            monkeypatch.setattr(truepeak, "MIN_CHUNK", min(chunk, 1 << 13))
            monkeypatch.setattr(truepeak, "MAX_WORKERS", workers)
            got = oversampled_peak_envelope(wave, factor)
            assert np.array_equal(reference, got), f"chunk={chunk} workers={workers}"


@pytest.mark.unit
def test_envelope_handles_degenerate_shapes() -> None:
    """Empty, sub-EDGE and single-sample inputs stay on the serial path."""
    assert oversampled_peak_envelope(np.zeros((2, 0), dtype=np.float32), 4).size == 0
    assert true_peak_linear(np.zeros((2, 0), dtype=np.float32)) == 0.0
    for samples in (1, 7, EDGE - 1, EDGE + 1):
        wave = _signal(1, samples)
        assert np.array_equal(oversampled_peak_envelope(wave, 4), _unchunked_envelope(wave, 4)), (
            samples
        )


@pytest.mark.unit
def test_plan_keeps_in_flight_memory_under_the_budget() -> None:
    """Worker count shrinks rather than the memory ceiling being blown."""
    for channels in (1, 2, 8):
        for factor in (4, 8):
            for samples in (1_000, 1_154_304, 4_541_440, 300_000_000):
                chunk, workers = _plan(channels, samples, factor)
                assert workers >= 1
                assert chunk >= 1
                assert workers <= truepeak.MAX_WORKERS
                in_flight = workers * channels * (chunk + 2 * EDGE) * factor * 4
                # One worker on a tiny minimum chunk is the floor we accept.
                floor = channels * (truepeak.MIN_CHUNK + 2 * EDGE) * factor * 4
                assert in_flight <= max(truepeak.INFLIGHT_BYTES, floor)


@pytest.mark.unit
def test_worker_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash inside a chunk must raise, never return a half-filled envelope."""
    wave = _signal(1, 400_000)
    monkeypatch.setattr(truepeak, "CHUNK", 1 << 13)
    monkeypatch.setattr(truepeak, "MIN_CHUNK", 1 << 13)

    lock = threading.Lock()
    calls = {"n": 0}
    real = scipy.signal.resample_poly

    def exploding(*args: Any, **kwargs: Any) -> Any:
        with lock:
            calls["n"] += 1
            nth = calls["n"]
        if nth == 3:
            raise MemoryError("simulated chunk failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.signal, "resample_poly", exploding)
    with pytest.raises(MemoryError):
        oversampled_peak_envelope(wave, 4)


@pytest.mark.unit
def test_true_peak_is_the_max_of_the_envelope() -> None:
    wave = _signal(2, 200_000)
    env = oversampled_peak_envelope(wave, 8)
    assert true_peak_linear(wave, factor=8) == float(np.max(env))
