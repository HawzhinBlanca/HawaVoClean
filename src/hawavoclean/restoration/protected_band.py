"""Protected-band masking, spectrum merging, and numerical invariance verification."""

import math
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
#: Per-band checking stops at this fraction of the protected boundary. Above
#: it lies the restorer's crossover skirt, which reaches below the nominal
#: boundary by design: measured across five speech-like sources, a legitimate
#: full-strength restoration moves the top third-octave band by up to 6.3 dB
#: while a band-stop across the same band moves it 9.3 dB. Those two are not
#: separable by this statistic, so the skirt is left to the global norm and to
#: Guard R's other layers rather than given a threshold that would either
#: revert good restorations or wave through bad ones.
_BAND_CHECK_CEILING = 0.85
#: Largest energy change a third-octave band inside the enforceable region may
#: show. Legitimate restorations measured 0.3-2.7 dB there; band-stops that the
#: global norm waved through measured 9.8-19.2 dB.
_MAX_BAND_DEVIATION_DB = 6.0


@dataclass(frozen=True)
class ProtectedBandVerification:
    """Numerical metrics evaluating protected-band preservation below cutoff."""

    max_waveform_abs_error: float
    rms_waveform_error: float
    complex_stft_relative_error: float
    max_phase_deviation_rad: float
    #: Largest third-octave energy change inside the enforceable part of the
    #: protected region, in dB, and the centre of the band that produced it.
    #: The global norm alone cannot see a gutted sub-band; this is the number
    #: that can. Energy rather than complex difference on purpose: a crossover
    #: shifts phase legitimately, and a complex-difference norm reads that
    #: shift as destruction.
    worst_band_energy_deviation_db: float
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
    max_third_octave_deviation_db: float = _MAX_BAND_DEVIATION_DB,
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
            worst_band_energy_deviation_db=0.0,
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
            worst_band_energy_deviation_db=0.0,
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
            worst_band_energy_deviation_db=0.0,
            worst_band_center_hz=0.0,
            passes_invariance=True,
        )

    Z_orig_prot = Z_orig[protected_bins, :]
    Z_rest_prot = Z_rest[protected_bins, :]

    diff = Z_rest_prot - Z_orig_prot
    stft_rel_err = float(np.linalg.norm(diff) / (np.linalg.norm(Z_orig_prot) + 1e-9))

    # Per-band energy, because the global norm above cannot enforce the promise
    # this function exists to make. Protected-band energy is dominated by
    # sub-1 kHz speech, so a whole slice higher up holds a small enough share
    # of the total that removing it barely moves the norm. Measured on a
    # speech-like source, against a 0.10 tolerance: band-stops at 600-900 Hz,
    # 900-1400, 1200-2000, 1800-2600 and 2600-3400 scored 0.089, 0.057, 0.042,
    # 0.025 and 0.014 -- every one of them accepted, each having gutted a whole
    # third-octave slice of protected speech. The same cases move their band's
    # OWN energy by 9.8 to 19.2 dB, which is what this loop measures.
    #
    # Energy, not complex difference: a crossover shifts phase legitimately,
    # and ``|Z_rest - Z_orig|`` counts that shift as if the content had been
    # destroyed -- it rated a good restoration worse than a band-stop.
    prot_freqs = freqs[protected_bins]
    band_energy = np.abs(Z_orig_prot) ** 2
    per_band_energy = np.sum(band_energy, axis=1)
    loudest_bin = float(np.max(per_band_energy)) if per_band_energy.size else 0.0
    check_ceiling = protected_boundary * _BAND_CHECK_CEILING

    worst_band_error = 0.0
    worst_band_hz = 0.0
    if loudest_bin > 0.0:
        lo_hz = max(float(prot_freqs[0]), _MIN_BAND_HZ)
        while lo_hz < check_ceiling:
            hi_hz = min(lo_hz * _THIRD_OCTAVE, protected_boundary)
            in_band = (prot_freqs >= lo_hz) & (prot_freqs < hi_hz)
            lo_hz = hi_hz
            # Only whole bands below the ceiling: a band straddling it would
            # drag the skirt's legitimate reshaping into the strict check.
            if hi_hz > check_ceiling or not np.any(in_band):
                continue
            orig_band = Z_orig_prot[in_band, :]
            energy = float(np.sum(np.abs(orig_band) ** 2))
            # Bands carrying essentially nothing are numerical noise, and a
            # large relative change on silence is not a violation of anything.
            if energy <= loudest_bin * _BAND_ENERGY_FLOOR_RATIO:
                continue
            rest_energy = float(np.sum(np.abs(Z_rest_prot[in_band, :]) ** 2))
            band_err = abs(10.0 * math.log10((rest_energy + 1e-20) / (energy + 1e-20)))
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
        and worst_band_error <= max_third_octave_deviation_db
    )

    return ProtectedBandVerification(
        max_waveform_abs_error=max_abs,
        rms_waveform_error=rms_err,
        complex_stft_relative_error=stft_rel_err,
        max_phase_deviation_rad=max_phase_dev,
        worst_band_energy_deviation_db=worst_band_error,
        worst_band_center_hz=worst_band_hz,
        passes_invariance=passes,
    )
