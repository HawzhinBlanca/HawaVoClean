"""Batching the guard's FFTs must not move a single bin.

Both guard front-ends used to call ``np.fft.rfft`` once per frame — tens of
thousands of calls on a 24-second file. They now stack frames and take one
transform over the last axis. That is only acceptable if the batched transform
is *exactly* the per-row transform: the guard's scores are compared against
calibrated thresholds, so a change of one ULP is a change of verdict risk.
"""

from typing import Any

import numpy as np
import pytest

from hawavoclean.guard import signal as guard_signal
from hawavoclean.guard import spectral_probe as probe_mod
from hawavoclean.guard.signal import _magnitude_stft, check_signal_integrity
from hawavoclean.guard.spectral_probe import SpectralSignatureProbe

FloatArray = np.ndarray[Any, np.dtype[np.float32]]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "length", "n_fft"), [(1, 1024, 1024), (997, 1024, 1024), (13, 400, 512)]
)
def test_batched_rfft_is_exactly_the_per_row_rfft(rows: int, length: int, n_fft: int) -> None:
    """Max absolute difference must be exactly 0.0, not merely small."""
    rng = np.random.default_rng(99)
    block = rng.standard_normal((rows, length))
    batched = np.fft.rfft(block, n=n_fft, axis=-1)
    worst = 0.0
    for i in range(rows):
        row = np.fft.rfft(block[i], n=n_fft)
        assert np.array_equal(row, batched[i]), f"row {i} differs bitwise"
        worst = max(worst, float(np.max(np.abs(row - batched[i]))))
    assert worst == 0.0


def _per_frame_signal_stft(
    wave: FloatArray, n_fft: int, hop: int, win: np.ndarray[Any, np.dtype[np.float64]], frames: int
) -> FloatArray:
    """The original one-rfft-per-frame loop, kept as the oracle."""
    stft = np.zeros((frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(frames):
        start = i * hop
        stft[i] = np.abs(np.fft.rfft(wave[start : start + n_fft] * win, n=n_fft))
    return stft


@pytest.mark.unit
@pytest.mark.parametrize("length", [1024, 1025, 1280, 5_000, 60_000])
@pytest.mark.parametrize("scale", [1.0, 1e-8, 3.0])
def test_signal_magnitude_stft_matches_the_per_frame_loop(length: int, scale: float) -> None:
    rng = np.random.default_rng(7)
    wave = (rng.standard_normal(length) * scale).astype(np.float32)
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    frames = (length - n_fft) // hop + 1
    expected = _per_frame_signal_stft(wave, n_fft, hop, win, frames)
    got = _magnitude_stft(wave, n_fft, hop, win, frames)
    assert np.array_equal(expected, got)
    assert float(np.max(np.abs(expected - got))) == 0.0


@pytest.mark.unit
def test_signal_magnitude_stft_handles_an_offset_slice() -> None:
    """The guard hands in a slice of a longer buffer, not a fresh array."""
    rng = np.random.default_rng(8)
    backing = (rng.standard_normal(80_000) * 0.5).astype(np.float32)
    wave = backing[7:60_007]
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    frames = (wave.size - n_fft) // hop + 1
    assert np.array_equal(
        _per_frame_signal_stft(wave, n_fft, hop, win, frames),
        _magnitude_stft(wave, n_fft, hop, win, frames),
    )


@pytest.mark.unit
def test_signal_stft_block_size_does_not_change_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(9)
    wave = (rng.standard_normal(40_000) * 0.3).astype(np.float32)
    n_fft, hop = 1024, 256
    win = np.hanning(n_fft)
    frames = (wave.size - n_fft) // hop + 1
    reference = _magnitude_stft(wave, n_fft, hop, win, frames)
    for block in (1, 7, 64, 512, 4096):
        monkeypatch.setattr(guard_signal, "STFT_FRAME_BLOCK", block)
        assert np.array_equal(reference, _magnitude_stft(wave, n_fft, hop, win, frames)), block


@pytest.mark.unit
@pytest.mark.parametrize("length", [100, 400, 401, 1_000, 16_000, 48_000])
def test_probe_features_match_the_per_frame_loop(length: int) -> None:
    rng = np.random.default_rng(11)
    wave = (rng.standard_normal(length) * 0.3).astype(np.float32)
    probe = SpectralSignatureProbe()

    sr, n_fft = 16_000, 512
    hop = int(round(sr * 0.020))
    win_length = int(round(sr * 0.025))
    window = np.hanning(win_length)
    padded = np.pad(wave, (0, win_length - len(wave))) if len(wave) < win_length else wave
    frames = max(1, (len(padded) - win_length) // hop + 1)

    expected_mag = np.zeros((frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(frames):
        chunk = padded[i * hop : i * hop + win_length] * window
        expected_mag[i] = np.abs(np.fft.rfft(chunk, n=n_fft))
    expected = np.log1p(expected_mag[:, :80]).astype(np.float32)

    got = probe._compute_spectral_features(wave)
    assert np.array_equal(expected, got)
    assert float(np.max(np.abs(expected - got))) == 0.0


@pytest.mark.unit
def test_probe_block_size_does_not_change_the_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(13)
    wave = (rng.standard_normal(32_000) * 0.3).astype(np.float32)
    probe = SpectralSignatureProbe()
    reference = probe.infer(wave, 16_000)
    for block in (1, 3, 128, 4096):
        monkeypatch.setattr(probe_mod, "STFT_FRAME_BLOCK", block)
        got = probe.infer(wave, 16_000)
        assert got.raw_signature == reference.raw_signature, block
        assert np.array_equal(got.frame_distributions, reference.frame_distributions), block


@pytest.mark.unit
def test_identical_waveforms_still_score_clean() -> None:
    """End-to-end sanity: batching did not perturb the guard's own scores."""
    rng = np.random.default_rng(17)
    wave = (rng.standard_normal(48_000) * 0.2).astype(np.float32)
    res = check_signal_integrity(wave, wave.copy(), 48_000)
    assert res.passed
    assert res.spectral_hole_score == 0.0
    assert res.musical_noise_score == 0.0
    assert res.consonant_retention_ratio == pytest.approx(1.0, abs=1e-6)
