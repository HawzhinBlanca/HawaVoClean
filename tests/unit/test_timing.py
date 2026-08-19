"""Unit tests for timing and landmark drift integrity."""

import numpy as np
import pytest

from voiceclean.guard.timing import check_timing_integrity


@pytest.mark.unit
def test_timing_integrity_identical() -> None:
    sr = 48000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    wave = 0.4 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    res = check_timing_integrity(wave, wave, sr)
    assert res.passed is True
    assert res.duration_ratio == pytest.approx(1.0, abs=1e-4)
    assert res.envelope_correlation == pytest.approx(1.0, abs=1e-3)


@pytest.mark.unit
def test_timing_integrity_shifted_drift() -> None:
    sr = 48000
    orig = np.zeros(2 * sr, dtype=np.float32)
    # Burst at 0.5s
    orig[int(0.5 * sr) : int(0.7 * sr)] = 0.5

    # Shifted burst at 0.65s (150ms drift)
    cand = np.zeros(2 * sr, dtype=np.float32)
    cand[int(0.65 * sr) : int(0.85 * sr)] = 0.5

    res = check_timing_integrity(orig, cand, sr, max_allowed_drift_ms=40.0)
    assert res.passed is False
    assert res.max_drift_ms > 40.0
