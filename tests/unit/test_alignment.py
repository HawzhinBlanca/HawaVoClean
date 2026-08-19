"""Unit tests for delay estimation, drift tracking, and phase/magnitude coherence."""

import numpy as np
import pytest

from voiceclean.alignment.coherence import estimate_coherence
from voiceclean.alignment.delay import estimate_gcc_phat_delay
from voiceclean.alignment.drift import analyze_local_drift


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


@pytest.mark.unit
def test_landmark_drift_flat() -> None:
    sr = 48000
    sig = np.random.default_rng(42).normal(0.0, 0.2, size=9600).astype(np.float32)
    res = analyze_local_drift(sig, sig, sr, window_s=0.10, hop_s=0.05)
    assert res.passed is True
    assert res.max_window_drift_ms < 1.0


@pytest.mark.unit
def test_coherence_identical() -> None:
    sig = np.random.default_rng(42).normal(0.0, 0.2, size=4800).astype(np.float32)
    res = estimate_coherence(sig, sig)
    assert res.passed is True
    assert res.phase_coherence == pytest.approx(1.0, abs=1e-3)
    assert res.magnitude_coherence == pytest.approx(1.0, abs=1e-3)
