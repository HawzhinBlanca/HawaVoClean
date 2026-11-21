"""Hypothesis property tests for restoration invariances."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hawavoclean.restoration.protected_band import (
    compute_transition_mask,
    merge_protected_spectrum,
)


@given(
    cutoff_hz=st.floats(min_value=2000.0, max_value=20000.0),
    transition_hz=st.floats(min_value=100.0, max_value=1000.0),
)
@settings(max_examples=30, deadline=None)
def test_property_transition_mask_monotonic(cutoff_hz: float, transition_hz: float) -> None:
    """Property: Transition mask is always bounded in [0, 1] and monotonic non-decreasing."""
    mask = compute_transition_mask(
        n_freqs=1025,
        sample_rate=48000,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )
    assert np.all(mask >= 0.0)
    assert np.all(mask <= 1.0)
    assert np.all(np.diff(mask) >= -1e-6)


@given(
    cutoff_hz=st.floats(min_value=3000.0, max_value=16000.0),
    strength=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=30, deadline=None)
def test_property_protected_band_perfect_preservation(cutoff_hz: float, strength: float) -> None:
    """Property: Frequency bins below cutoff transition are perfectly identical to observed spectrum."""
    n_freqs = 513
    n_frames = 20
    sample_rate = 48000
    transition_hz = 500.0

    mask = compute_transition_mask(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )

    obs = (np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)).astype(
        np.complex64
    )
    gen = (np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)).astype(
        np.complex64
    )

    merged = merge_protected_spectrum(obs, gen, mask, strength=strength)

    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    passband = freqs < (cutoff_hz - transition_hz / 2.0)

    np.testing.assert_array_equal(merged[passband, :], obs[passband, :])


@given(
    cutoff_hz=st.floats(min_value=3000.0, max_value=16000.0),
    transition_hz=st.floats(min_value=200.0, max_value=1000.0),
)
@settings(max_examples=30, deadline=None)
def test_property_protected_band_stopband_replacement(
    cutoff_hz: float, transition_hz: float
) -> None:
    """Property: At strength 1.0, bins above upper transition edge match generated spectrum."""
    n_freqs = 513
    n_frames = 15
    sample_rate = 48000

    mask = compute_transition_mask(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )

    obs = (np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)).astype(
        np.complex64
    )
    gen = (np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)).astype(
        np.complex64
    )

    merged = merge_protected_spectrum(obs, gen, mask, strength=1.0)

    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    stopband = freqs > (cutoff_hz + transition_hz / 2.0)

    np.testing.assert_array_equal(merged[stopband, :], gen[stopband, :])


@given(
    cutoff_hz=st.floats(min_value=3000.0, max_value=16000.0),
    alpha=st.floats(min_value=0.1, max_value=5.0),
    strength=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=30, deadline=None)
def test_property_merge_spectrum_scaling_homogeneity(
    cutoff_hz: float, alpha: float, strength: float
) -> None:
    """Property: Scaling both observed and generated inputs by alpha scales output by alpha."""
    n_freqs = 257
    n_frames = 10
    sample_rate = 48000

    mask = compute_transition_mask(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=500.0,
    )

    obs = (np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)).astype(
        np.complex64
    )
    gen = (np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)).astype(
        np.complex64
    )

    merged_scaled = merge_protected_spectrum(alpha * obs, alpha * gen, mask, strength=strength)
    merged_direct = alpha * merge_protected_spectrum(obs, gen, mask, strength=strength)

    np.testing.assert_allclose(merged_scaled, merged_direct, rtol=1e-5, atol=1e-6)


@given(
    cutoff_hz=st.floats(min_value=3000.0, max_value=16000.0),
    strength1=st.floats(min_value=0.0, max_value=1.0),
    strength2=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=30, deadline=None)
def test_property_merge_spectrum_strength_monotonicity(
    cutoff_hz: float, strength1: float, strength2: float
) -> None:
    """Property: Generated band energy grows monotonically with strength."""
    n_freqs = 257
    n_frames = 10
    sample_rate = 48000
    s_low, s_high = min(strength1, strength2), max(strength1, strength2)

    mask = compute_transition_mask(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=500.0,
    )

    obs = np.zeros((n_freqs, n_frames), dtype=np.complex64)
    gen = np.ones((n_freqs, n_frames), dtype=np.complex64)

    m_low = merge_protected_spectrum(obs, gen, mask, strength=s_low)
    m_high = merge_protected_spectrum(obs, gen, mask, strength=s_high)

    e_low = np.sum(np.abs(m_low) ** 2)
    e_high = np.sum(np.abs(m_high) ** 2)

    assert e_high >= e_low - 1e-6
