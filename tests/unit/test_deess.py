"""Unit tests for conservative split-band de-esser."""

from __future__ import annotations

import numpy as np

from hawavoclean.finishing.deess import apply_split_band_deesser


def test_deesser_short_or_low_sample_rate() -> None:
    # 1. Short input < 128 samples
    short_sig = np.zeros(64, dtype=np.float32)
    out, gr = apply_split_band_deesser(short_sig, 48000)
    assert len(out) == 64
    assert gr == 0.0

    # 2. Nyquist below crossover
    sig = np.zeros(1000, dtype=np.float32)
    out, gr = apply_split_band_deesser(sig, sample_rate=8000, crossover_hz=5500.0)
    assert len(out) == 1000
    assert gr == 0.0


def test_deesser_compresses_sibilance() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # High frequency 7 kHz harsh sibilance tone
    sibilant = (0.8 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)

    out, gr = apply_split_band_deesser(
        sibilant,
        sample_rate=sr,
        crossover_hz=5500.0,
        threshold_db=-20.0,
        ratio=3.0,
        max_reduction_db=4.0,
    )
    assert gr > 0.0
    assert gr <= 4.0
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(sibilant)))


def test_deesser_leaves_quiet_signal_untouched() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Very quiet high frequency tone below threshold
    quiet = (0.001 * np.sin(2 * np.pi * 7000 * t)).astype(np.float32)

    out, gr = apply_split_band_deesser(
        quiet,
        sample_rate=sr,
        threshold_db=-20.0,
    )
    assert gr == 0.0
    np.testing.assert_allclose(out, quiet, atol=1e-5)
