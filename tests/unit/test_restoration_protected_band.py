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


def test_a_gutted_sub_band_inside_the_protected_region_is_refused() -> None:
    """Deleting a slice of the protected band must fail invariance.

    The check used to be a single relative norm over the whole protected
    region. Protected-band energy is dominated by sub-1 kHz speech, so an
    entire multi-kHz slice holding a small share of it could be removed and
    still land two orders of magnitude inside tolerance: measured, a band-stop
    from 2.6 kHz up to the protected boundary took 20.7 dB out of the region
    the guard calls protected, and Guard R returned PASS at strength 1.00 and
    handed back the gutted audio. That is the product's central safety claim
    failing open, so the region is now judged band by band.
    """
    sr = 48000
    rng = np.random.default_rng(3)
    t = np.arange(int(sr * 2.0)) / sr
    harmonics = np.zeros_like(t)
    k = 1
    while 120.0 * k < sr / 2:
        harmonics += (1.0 / (k**1.5)) * np.sin(
            2 * np.pi * 120.0 * k * t + rng.uniform(0, 2 * np.pi)
        )
        k += 1
    voiced = harmonics * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t))
    band_limited = signal.sosfiltfilt(
        signal.butter(12, 4000 / (sr / 2), btype="lowpass", output="sos"), voiced
    )
    natural = (band_limited / np.max(np.abs(band_limited)) * 0.7).astype(np.float32)

    cutoff_hz = 4094.0
    gutted = signal.sosfiltfilt(
        signal.butter(8, [1200.0, 2000.0], btype="bandstop", fs=sr, output="sos"), natural
    ).astype(np.float32)

    verified = verify_protected_band_invariance(
        natural,
        gutted,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        tolerance_rms=0.05,
        tolerance_stft=0.10,
    )

    assert not verified.passes_invariance, (
        "a sub-band of the protected region was deleted and invariance still passed"
    )
    assert verified.worst_band_energy_deviation_db > 6.0
    assert 1200.0 <= verified.worst_band_center_hz <= 2000.0, (
        "the report must name the band that actually failed"
    )
    # The global norm alone would have missed it — that is the whole point.
    assert verified.complex_stft_relative_error < 0.10


def test_the_crossover_skirt_is_outside_the_per_band_promise() -> None:
    """Pin the limit of the per-band check instead of overstating its reach.

    The restorer's crossover reaches below the nominal protected boundary by
    design, moving the topmost third-octave bands by several dB on perfectly
    good output. A band-stop confined to that same skirt is not separable from
    that legitimate work by this statistic, so the check deliberately stops at
    85% of the boundary and those bands are left to the global norm and Guard
    R's other layers. This test exists so the gap is a stated property with a
    number on it, not something a reader has to discover.
    """
    sr = 48000
    rng = np.random.default_rng(3)
    t = np.arange(int(sr * 2.0)) / sr
    harmonics = np.zeros_like(t)
    k = 1
    while 120.0 * k < sr / 2:
        harmonics += (1.0 / (k**1.5)) * np.sin(
            2 * np.pi * 120.0 * k * t + rng.uniform(0, 2 * np.pi)
        )
        k += 1
    band_limited = signal.sosfiltfilt(
        signal.butter(12, 4000 / (sr / 2), btype="lowpass", output="sos"), harmonics
    )
    natural = (band_limited / np.max(np.abs(band_limited)) * 0.7).astype(np.float32)

    cutoff_hz = 4094.0
    boundary = max(500.0, cutoff_hz - 250.0)
    skirt_only = signal.sosfiltfilt(
        signal.butter(8, [boundary * 0.87, boundary], btype="bandstop", fs=sr, output="sos"),
        natural,
    ).astype(np.float32)

    verified = verify_protected_band_invariance(
        natural,
        skirt_only,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        tolerance_rms=0.05,
        tolerance_stft=0.10,
    )
    assert verified.worst_band_center_hz <= boundary * 0.85, (
        "the per-band check must not reach into the crossover skirt"
    )


def test_untouched_audio_still_passes_band_by_band() -> None:
    """The per-band test must not reject a candidate that changed nothing."""
    sr = 48000
    t = np.arange(int(sr * 1.0)) / sr
    sig = (0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1400 * t)).astype(
        np.float32
    )
    verified = verify_protected_band_invariance(
        sig, sig.copy(), sample_rate=sr, cutoff_hz=6000.0, tolerance_rms=0.05, tolerance_stft=0.10
    )
    assert verified.passes_invariance
    assert verified.worst_band_energy_deviation_db == 0.0
