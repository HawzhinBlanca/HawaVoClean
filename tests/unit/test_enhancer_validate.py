"""Unit tests for enhancement candidate validation and production model classes."""

import numpy as np
import pytest

from hawavoclean.enhancement.production import NoOpEnhancer
from hawavoclean.enhancement.validate import validate_enhancer_output


@pytest.mark.unit
def test_validate_enhancement_candidate_valid() -> None:
    orig = np.zeros(4800, dtype=np.float32)
    cand = 0.5 * np.ones(4800, dtype=np.float32)
    passed, reason = validate_enhancer_output(orig, cand, is_speech=False)
    assert passed is True
    assert reason == ""


@pytest.mark.unit
def test_validate_enhancement_candidate_nan_rejected() -> None:
    orig = np.zeros(4800, dtype=np.float32)
    cand = np.zeros(4800, dtype=np.float32)
    cand[10] = np.nan
    passed, reason = validate_enhancer_output(orig, cand, is_speech=False)
    assert passed is False
    assert "NaN or Infinite" in reason


@pytest.mark.unit
def test_validate_enhancement_candidate_length_mismatch() -> None:
    orig = np.zeros(4800, dtype=np.float32)
    cand = np.zeros(4000, dtype=np.float32)
    passed, reason = validate_enhancer_output(orig, cand, is_speech=False)
    assert passed is False
    assert "length mismatch" in reason.lower()


@pytest.mark.unit
def test_noop_enhancer() -> None:
    enh = NoOpEnhancer()
    sig = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = enh.enhance(sig, 48000)
    assert np.array_equal(res.waveform, sig)
    assert res.model_runtime_ms >= 0.0
