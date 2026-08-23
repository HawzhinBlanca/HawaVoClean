"""Unit tests for protected-band invariance and complementary masking."""

import numpy as np
import scipy.signal as signal

from hawavoclean.restoration.protected_band import (
    compute_transition_mask,
    merge_protected_spectrum,
    verify_protected_band_invariance,
)


def test_transition_mask_shape_and_values() -> None:
    """Test transition mask values in passband, transition band, and stopband."""
    n_freqs = 1025
    sample_rate = 48000
    cutoff_hz = 8000.0
    transition_hz = 500.0

    mask = compute_transition_mask(
        n_freqs=n_freqs,
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
    )

    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    passband_indices = np.where(freqs < (cutoff_hz - transition_hz / 2.0))[0]
    stopband_indices = np.where(freqs > (cutoff_hz + transition_hz / 2.0))[0]

    # Below cutoff transition, mask should be strictly 0.0 (preserved observed band)
    assert np.all(mask[passband_indices] == 0.0)
    # Above cutoff transition, mask should be strictly 1.0 (synthesized band)
    assert np.all(mask[stopband_indices] == 1.0)
    # In transition band, mask values should be monotonically non-decreasing
    assert np.all(np.diff(mask) >= 0.0)


def test_merge_protected_spectrum_invariance() -> None:
    """Test that below cutoff, merged spectrum is strictly identical to observed spectrum."""
    n_freqs = 1025
    n_frames = 100
    sample_rate = 48000
    cutoff_hz = 8000.0

    mask = compute_transition_mask(n_freqs=n_freqs, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    obs_spec = (
        np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)
    ).astype(np.complex64)
    gen_spec = (
        np.random.randn(n_freqs, n_frames) + 1j * np.random.randn(n_freqs, n_frames)
    ).astype(np.complex64)

    merged = merge_protected_spectrum(obs_spec, gen_spec, mask, strength=1.0)

    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    passband_mask = freqs < (cutoff_hz - 250.0)

    np.testing.assert_array_equal(merged[passband_mask, :], obs_spec[passband_mask, :])


def test_verify_protected_band_invariance_passes() -> None:
    """Test invariance verification on waveform with unaltered low band."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    clean = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t)).astype(
        np.float32
    )
    sos = signal.butter(6, 4000 / 24000, btype="lowpass", output="sos")
    natural_audio = signal.sosfiltfilt(sos, clean).astype(np.float32)

    # Identical waveform passes with zero error
    chk = verify_protected_band_invariance(
        original_audio=natural_audio,
        restored_audio=natural_audio,
        sample_rate=sr,
        cutoff_hz=4000.0,
    )

    assert chk.passes_invariance is True
    assert chk.rms_waveform_error < 1e-4
