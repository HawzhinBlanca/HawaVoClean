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


@pytest.mark.unit
def test_validate_enhancer_output_additional_branches() -> None:
    orig = 0.5 * np.ones(4800, dtype=np.float32)

    # 1. Not an array or None
    passed, reason = validate_enhancer_output(orig, None, is_speech=False)  # type: ignore[arg-type]
    assert not passed and "not a valid numpy array" in reason

    passed, reason = validate_enhancer_output(orig, [1.0, 2.0], is_speech=False)  # type: ignore[arg-type]
    assert not passed and "not a valid numpy array" in reason

    # 2. Wrong dtype
    wrong_dtype = 0.5 * np.ones(4800, dtype=np.float64)
    passed, reason = validate_enhancer_output(orig, wrong_dtype, is_speech=False)  # type: ignore[arg-type]
    assert not passed and "Expected float32" in reason

    # 3. Speech signal collapse
    silent = np.zeros(4800, dtype=np.float32)
    passed, reason = validate_enhancer_output(orig, silent, is_speech=True)
    assert not passed and "collapsed to near-zero" in reason

    # 4. Energy explosion (> 10x) and normal speech non-explosion
    normal_speech = 0.6 * np.ones(4800, dtype=np.float32)
    passed, reason = validate_enhancer_output(orig, normal_speech, is_speech=True)
    assert passed and reason == ""

    exploded = 6.0 * np.ones(4800, dtype=np.float32)
    passed, reason = validate_enhancer_output(orig, exploded, is_speech=True)
    assert not passed and "energy explosion" in reason

    # 5. Newly introduced hard clipping (cand_max > 1.05 when orig_max < 0.99)
    clipped = 0.5 * np.ones(4800, dtype=np.float32)
    clipped[0] = 1.10
    passed, reason = validate_enhancer_output(orig, clipped, is_speech=False)
    assert not passed and "hard clipping" in reason

    # 6. Empty waveform handling
    empty = np.array([], dtype=np.float32)
    passed, reason = validate_enhancer_output(empty, empty, is_speech=False)
    assert passed and reason == ""
