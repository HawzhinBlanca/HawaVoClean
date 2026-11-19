"""Unit tests for delay estimation."""

import numpy as np
import pytest

from hawavoclean.alignment.delay import estimate_gcc_phat_delay


@pytest.mark.unit
def test_fractional_delay_zero() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    res = estimate_gcc_phat_delay(sig, sig, sr)
    assert res.delay_samples == pytest.approx(0.0, abs=0.1)
    assert res.delay_ms == pytest.approx(0.0, abs=0.01)


@pytest.mark.unit
def test_align_and_delay_compensate() -> None:
    sr = 48000
    sig = np.random.default_rng(42).normal(0.0, 0.2, size=4800).astype(np.float32)
    delayed = np.roll(sig, 5)
    res = estimate_gcc_phat_delay(sig, delayed, sr)
    assert len(res.aligned_candidate) == len(sig)
    assert abs(res.delay_samples) == pytest.approx(5.0, abs=0.5)
