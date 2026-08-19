"""Unit tests for acoustic signal integrity detectors (consonants, spectral holes, clipping)."""

import numpy as np
import pytest

from hawavoclean.guard.signal import check_signal_integrity


@pytest.mark.unit
def test_signal_integrity_clean_pass() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 3kHz consonant tone + 300Hz vowel tone
    wave = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)).astype(
        np.float32
    )

    res = check_signal_integrity(wave, wave, sr)
    assert res.passed is True
    assert res.clipping_samples_count == 0
    assert res.consonant_retention_ratio == pytest.approx(1.0, abs=0.05)


@pytest.mark.unit
def test_signal_integrity_detects_clipping() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    orig = 0.5 * np.sin(2 * np.pi * 400 * t).astype(np.float32)
    clipped = np.clip(1.5 * np.sin(2 * np.pi * 400 * t), -1.0, 1.0).astype(np.float32)

    res = check_signal_integrity(orig, clipped, sr, max_allowed_clipping_samples=0)
    assert res.passed is False
    assert res.clipping_samples_count > 0
    assert any("clipping" in r for r in res.failure_reasons)


@pytest.mark.unit
def test_signal_integrity_detects_wiped_consonants() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    orig = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 3500 * t)).astype(
        np.float32
    )
    # Wiped 3.5kHz consonant band in candidate
    cand = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    res = check_signal_integrity(orig, cand, sr, min_hf_preservation_ratio=0.60)
    assert res.passed is False
    assert any("Consonant presence" in r for r in res.failure_reasons)
