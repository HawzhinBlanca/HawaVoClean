from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from hawavoclean.audio.types import AudioBuffer
from hawavoclean.smart_safe.analyzer import (
    ProbabilityEstimate,
    StreamingAcousticAnalyzer,
    _clip_probability,
)


def test_clip_probability_nan() -> None:
    assert _clip_probability(float("nan")) == 0.0
    assert _clip_probability(1.5) == 1.0
    assert _clip_probability(-0.5) == 0.0


def test_probability_estimate_validations() -> None:
    # Value out of range or non-finite
    with pytest.raises(ValueError, match="value must be finite"):
        ProbabilityEstimate(
            value=float("nan"),
            confidence=0.5,
            conservative=0.5,
            direction="lower",
            rationale="test",
        )

    with pytest.raises(ValueError, match="value must be finite"):
        ProbabilityEstimate(
            value=1.5, confidence=0.5, conservative=0.5, direction="lower", rationale="test"
        )

    # Invalid direction
    with pytest.raises(ValueError, match="direction must be lower or upper"):
        ProbabilityEstimate(
            value=0.5,
            confidence=0.5,
            conservative=0.5,
            direction=cast(Any, "invalid"),
            rationale="test",
        )

    # Empty rationale
    with pytest.raises(ValueError, match="rationale must not be empty"):
        ProbabilityEstimate(
            value=0.5, confidence=0.5, conservative=0.5, direction="lower", rationale=""
        )


def test_analyzer_accept_edge_cases() -> None:
    analyzer = StreamingAcousticAnalyzer()

    # 1. Non AudioBuffer
    analyzer.accept("not_an_audio_buffer")  # type: ignore[arg-type]
    assert not analyzer.valid

    # Reset with fresh analyzer
    analyzer = StreamingAcousticAnalyzer()

    # 2. Unsupported sample rate (< 8000 or > 192000)
    bad_chunk = AudioBuffer(data=np.zeros((1, 100), dtype=np.float32), sample_rate=4000)
    analyzer.accept(bad_chunk)
    assert not analyzer.valid

    # 3. Empty chunk (0 samples)
    good_chunk = AudioBuffer(data=np.zeros((1, 2048), dtype=np.float32), sample_rate=48000)
    analyzer = StreamingAcousticAnalyzer()
    analyzer.accept(good_chunk)
    assert analyzer.valid

    empty_chunk = AudioBuffer(data=np.zeros((1, 0), dtype=np.float32), sample_rate=48000)
    analyzer.accept(empty_chunk)
    assert analyzer.valid

    # 4. Accept after finish
    analyzer.finish()
    with pytest.raises(RuntimeError, match="cannot accept audio after finish"):
        analyzer.accept(good_chunk)
