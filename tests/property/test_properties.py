"""Property-based tests with Hypothesis validating invariants across arbitrary audio waveforms."""

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from hawavoclean.audio.resample import resample_audio
from hawavoclean.finishing.limiter import apply_lookahead_limiter
from hawavoclean.hashing import hash_numpy


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    waveform=npst.arrays(
        dtype=np.float32,
        shape=st.integers(min_value=256, max_value=4800),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
)
def test_property_resampling_finite_and_exact_target_length(waveform: np.ndarray) -> None:
    res = resample_audio(waveform, orig_sr=48000, target_sr=16000)
    assert np.all(np.isfinite(res))
    assert res.dtype == np.float32


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    waveform=npst.arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=1, max_value=2), st.integers(min_value=512, max_value=4800)
        ),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
)
def test_property_limiter_strictly_enforces_ceiling(waveform: np.ndarray) -> None:
    res = apply_lookahead_limiter(waveform, sample_rate=48000, ceiling_dbtp=-1.0)
    ceiling_lin = 10.0 ** (-1.0 / 20.0)  # ~0.89125
    assert np.all(np.isfinite(res.limited_waveform))
    assert float(np.max(np.abs(res.limited_waveform))) <= ceiling_lin + 1e-4


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    waveform=npst.arrays(
        dtype=np.float32,
        shape=st.integers(min_value=10, max_value=500),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
)
def test_property_hash_numpy_deterministic(waveform: np.ndarray) -> None:
    h1 = hash_numpy(waveform)
    h2 = hash_numpy(waveform.copy())
    assert h1 == h2
    assert len(h1) == 64
