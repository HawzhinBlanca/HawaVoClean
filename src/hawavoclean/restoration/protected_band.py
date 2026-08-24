"""Protected-band masking, spectrum merging, and numerical invariance verification."""

from dataclasses import dataclass

import numpy as np
from scipy import signal

#: Third-octave band ratio used to judge the protected region band by band.
_THIRD_OCTAVE = 2.0 ** (1.0 / 3.0)
#: Bands below this frequency are pooled into the first band.
_MIN_BAND_HZ = 50.0
#: A band holding this little of the loudest bin's energy is numerical noise,
#: and a large relative error on silence is not a violation of anything.
_BAND_ENERGY_FLOOR_RATIO = 1e-4


@dataclass(frozen=True)
class ProtectedBandVerification:
    """Numerical metrics evaluating protected-band preservation below cutoff."""

    max_waveform_abs_error: float
    rms_waveform_error: float
    complex_stft_relative_error: float
    max_phase_deviation_rad: float
    #: Largest third-octave relative error inside the protected region, and the
    #: centre of the band that produced it. The global norm alone cannot see a
    #: gutted sub-band; this is the number that can.
    worst_band_relative_error: float
    worst_band_center_hz: float
    passes_invariance: bool


def compute_transition_mask(
    n_freqs: int,
    sample_rate: int,
    cutoff_hz: float,
    transition_hz: float = 500.0,
) -> np.ndarray:
    """Compute smooth complementary transition mask across STFT frequency bins.

    Returns an array of shape (n_freqs,) where:
      - 0.0 indicates fully protected (observed band kept 100%)
      - 1.0 indicates fully generated (missing high band)
      - Smooth raised-cosine transition in [cutoff_hz - transition_hz/2, cutoff_hz + transition_hz/2]

    Vectorized implementation for performance on large FFTs.
    """
    if n_freqs <= 0:
        return np.array([], dtype=np.float32)

    freqs = np.fft.rfftfreq((n_freqs - 1) * 2, d=1.0 / sample_rate)
    half_t = max(0.0, transition_hz) / 2.0
    nyquist = sample_rate / 2.0
    cutoff_hz = float(np.clip(cutoff_hz, 0.0, nyquist))
    f_low = max(0.0, cutoff_hz - half_t)
    f_high = min(nyquist, cutoff_hz + half_t)

    mask = np.zeros(n_freqs, dtype=np.float32)

    # Vectorized: bins above upper transition edge are 1.0
    mask[freqs >= f_high] = 1.0

    # Vectorized: bins in the transition band get raised-cosine interpolation
    if f_high > f_low:
        in_transition = (freqs > f_low) & (freqs < f_high)
        if np.any(in_transition):
            phase = (freqs[in_transition] - f_low) / (f_high - f_low)
            mask[in_transition] = (0.5 * (1.0 - np.cos(np.pi * phase))).astype(np.float32)

    return mask


def merge_protected_spectrum(
    observed_stft: np.ndarray,  # (..., n_freqs, n_frames), complex64
    generated_high_stft: np.ndarray,  # (..., n_freqs, n_frames), complex64
    mask: np.ndarray,  # (n_freqs,), float32
    strength: float = 1.0,
) -> np.ndarray:
    """Merge observed low-band spectrum with generated missing high-band spectrum.

    The trusted observed spectrum is preserved identically below the transition band.
    """
    mask_expanded = mask.reshape((1,) * (observed_stft.ndim - 2) + (len(mask), 1))
    complement = 1.0 - mask_expanded

    # Observed band kept via complement; generated high band scaled by strength and mask
    merged = (observed_stft * complement) + (strength * generated_high_stft * mask_expanded)
    return np.asarray(merged, dtype=np.complex64)


def verify_protected_band_invariance(
    original_audio: np.ndarray,
    restored_audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    transition_hz: float = 500.0,
    tolerance_rms: float = 1e-4,
    tolerance_stft: float = 1e-3,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> ProtectedBandVerification:
    """Verify that audio content strictly below the transition cutoff remains unmodified."""
    orig_mono = np.mean(original_audio, axis=0) if original_audio.ndim == 2 else original_audio
    rest_mono = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio

    # Guard against degenerate input
    if orig_mono.size == 0 or rest_mono.size == 0:
        return ProtectedBandVerification(
            max_waveform_abs_error=0.0,
            rms_waveform_error=0.0,
            complex_stft_relative_error=0.0,
            max_phase_deviation_rad=0.0,
            worst_band_relative_error=0.0,
            worst_band_center_hz=0.0,
            passes_invariance=True,
        )

    # Length match
    min_len = min(len(orig_mono), len(rest_mono))
    if min_len < n_fft:
        # Too short for meaningful STFT analysis — compare waveforms directly
        diff = rest_mono[:min_len] - orig_mono[:min_len]
        rms_err = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff))) if min_len > 0 else 0.0
        return ProtectedBandVerification(
            max_waveform_abs_error=max_abs,
            rms_waveform_error=rms_err,
            complex_stft_relative_error=0.0,
            max_phase_deviation_rad=0.0,
            worst_band_relative_error=0.0,
            worst_band_center_hz=0.0,
            passes_invariance=(rms_err <= tolerance_rms),
        )

    orig_mono = orig_mono[:min_len]
    rest_mono = rest_mono[:min_len]

    # Compute STFT
    _, _, Z_orig = signal.stft(
        orig_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )
    _, _, Z_rest = signal.stft(
        rest_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=True,
    )

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    protected_boundary = max(500.0, cutoff_hz - transition_hz / 2.0)
    protected_bins = freqs < protected_boundary

    if not np.any(protected_bins):
        return ProtectedBandVerification(
            max_waveform_abs_error=0.0,
            rms_waveform_error=0.0,
            complex_stft_relative_error=0.0,
            max_phase_deviation_rad=0.0,
            worst_band_relative_error=0.0,
            worst_band_center_hz=0.0,
            passes_invariance=True,
        )

    Z_orig_prot = Z_orig[protected_bins, :]
    Z_rest_prot = Z_rest[protected_bins, :]

    diff = Z_rest_prot - Z_orig_prot
    stft_rel_err = float(np.linalg.norm(diff) / (np.linalg.norm(Z_orig_prot) + 1e-9))

    # Per-band error, because the global norm above cannot enforce the promise
    # this function exists to make. Protected-band energy is dominated by
    # sub-1 kHz speech, so a whole multi-kHz slice holding a small share of it
    # can be deleted, phase-inverted or fabricated and still land two orders of
    # magnitude inside tolerance. Measured: a band-stop from 2.6 kHz to the
    # protected boundary removed 20.7 dB from inside the protected region and
    # the guard returned verdict=PASS at strength 1.00, handing back the gutted
    # audio with passes_invariance=true. Each third-octave band is therefore
    # normalised by its OWN energy and judged on its own.
    prot_freqs = freqs[protected_bins]
    band_energy = np.abs(Z_orig_prot) ** 2
    per_band_energy = np.sum(band_energy, axis=1)
    loudest_bin = float(np.max(per_band_energy)) if per_band_energy.size else 0.0

    worst_band_error = 0.0
    worst_band_hz = 0.0
    if loudest_bin > 0.0:
        lo_hz = max(float(prot_freqs[0]), _MIN_BAND_HZ)
        while lo_hz < protected_boundary:
            hi_hz = min(lo_hz * _THIRD_OCTAVE, protected_boundary)
            in_band = (prot_freqs >= lo_hz) & (prot_freqs < hi_hz)
            lo_hz = hi_hz
            if not np.any(in_band):
                continue
            orig_band = Z_orig_prot[in_band, :]
            energy = float(np.sum(np.abs(orig_band) ** 2))
            # Bands carrying essentially nothing are numerical noise: a large
            # relative error on silence is not a violation of anything.
            if energy <= loudest_bin * _BAND_ENERGY_FLOOR_RATIO:
                continue
            band_err = float(
                np.linalg.norm(Z_rest_prot[in_band, :] - orig_band)
                / (np.linalg.norm(orig_band) + 1e-12)
            )
            if band_err > worst_band_error:
                worst_band_error = band_err
                worst_band_hz = float(np.mean(prot_freqs[in_band]))

    # Phase deviation
    phase_diff = np.angle(Z_rest_prot * np.conj(Z_orig_prot))
    max_phase_dev = float(np.max(np.abs(phase_diff)))

    # Low-pass filtered waveform error
    sos = signal.butter(6, protected_boundary, btype="lowpass", fs=sample_rate, output="sos")
    orig_lp = signal.sosfiltfilt(sos, orig_mono)
    rest_lp = signal.sosfiltfilt(sos, rest_mono)
    lp_diff = rest_lp - orig_lp

    max_abs = float(np.max(np.abs(lp_diff)))
    rms_err = float(np.sqrt(np.mean(lp_diff**2)))

    passes = (
        rms_err <= tolerance_rms
        and stft_rel_err <= tolerance_stft
        and worst_band_error <= tolerance_stft
    )

    return ProtectedBandVerification(
        max_waveform_abs_error=max_abs,
        rms_waveform_error=rms_err,
        complex_stft_relative_error=stft_rel_err,
        max_phase_deviation_rad=max_phase_dev,
        worst_band_relative_error=worst_band_error,
        worst_band_center_hz=worst_band_hz,
        passes_invariance=passes,
    )
