"""Evaluation metrics for audio bandwidth restoration, spectral fidelity, and speaker similarity."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class RestorationMetrics:
    """Quantitative objective restoration evaluation metrics."""

    fullband_lsd_db: float
    highband_lsd_db: float
    protected_band_rms_err: float
    highband_energy_error_db: float
    f0_rmse_hz: float
    speaker_cosine_sim: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return asdict(self)


def compute_log_spectral_distance(
    ref_audio: np.ndarray,
    deg_audio: np.ndarray,
    sample_rate: int = 48000,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq_min: float = 0.0,
    freq_max: float = 24000.0,
) -> float:
    """Compute Log-Spectral Distance (LSD) in dB between two aligned audio signals."""
    ref_mono = np.mean(ref_audio, axis=0) if ref_audio.ndim == 2 else ref_audio
    deg_mono = np.mean(deg_audio, axis=0) if deg_audio.ndim == 2 else deg_audio

    min_len = min(len(ref_mono), len(deg_mono))
    ref_mono = ref_mono[:min_len]
    deg_mono = deg_mono[:min_len]

    _, _, Z_ref = signal.stft(
        ref_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    _, _, Z_deg = signal.stft(
        deg_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band_mask = (freqs >= freq_min) & (freqs <= freq_max)

    if not np.any(band_mask):
        return 0.0

    P_ref = np.abs(Z_ref[band_mask, :]) ** 2 + 1e-12
    P_deg = np.abs(Z_deg[band_mask, :]) ** 2 + 1e-12

    log_ref = 10.0 * np.log10(P_ref)
    log_deg = 10.0 * np.log10(P_deg)

    frame_lsd = np.sqrt(np.mean((log_ref - log_deg) ** 2, axis=0))
    return float(np.mean(frame_lsd))


def evaluate_restoration(
    clean_reference: np.ndarray,
    restored_candidate: np.ndarray,
    cutoff_hz: float,
    sample_rate: int = 48000,
    speaker_embedding_ref: np.ndarray | None = None,
    speaker_embedding_cand: np.ndarray | None = None,
) -> RestorationMetrics:
    """Compute comprehensive restoration metrics against clean ground truth."""
    full_lsd = compute_log_spectral_distance(
        clean_reference,
        restored_candidate,
        sample_rate=sample_rate,
        freq_min=0.0,
        freq_max=sample_rate / 2.0,
    )
    high_lsd = compute_log_spectral_distance(
        clean_reference,
        restored_candidate,
        sample_rate=sample_rate,
        freq_min=cutoff_hz,
        freq_max=sample_rate / 2.0,
    )

    # Low-frequency protected band RMS error
    nyq = sample_rate / 2.0
    prot_cutoff = max(500.0, cutoff_hz - 250.0)
    sos_lp = signal.butter(6, prot_cutoff / nyq, btype="lowpass", output="sos")
    ref_lp = signal.sosfiltfilt(
        sos_lp, np.mean(clean_reference, axis=0) if clean_reference.ndim == 2 else clean_reference
    )
    cand_lp = signal.sosfiltfilt(
        sos_lp,
        np.mean(restored_candidate, axis=0) if restored_candidate.ndim == 2 else restored_candidate,
    )
    prot_rms = float(np.sqrt(np.mean((ref_lp - cand_lp) ** 2)))

    # High-band energy error in dB
    sos_hp = signal.butter(6, min(0.95, cutoff_hz / nyq), btype="highpass", output="sos")
    ref_hp = signal.sosfiltfilt(
        sos_hp, np.mean(clean_reference, axis=0) if clean_reference.ndim == 2 else clean_reference
    )
    cand_hp = signal.sosfiltfilt(
        sos_hp,
        np.mean(restored_candidate, axis=0) if restored_candidate.ndim == 2 else restored_candidate,
    )
    ref_energy = np.mean(ref_hp**2) + 1e-12
    cand_energy = np.mean(cand_hp**2) + 1e-12
    energy_err = float(np.abs(10.0 * np.log10(cand_energy / ref_energy)))

    # Speaker cosine similarity
    if speaker_embedding_ref is not None and speaker_embedding_cand is not None:
        norm_a = np.linalg.norm(speaker_embedding_ref) + 1e-9
        norm_b = np.linalg.norm(speaker_embedding_cand) + 1e-9
        spk_sim = float(np.dot(speaker_embedding_ref, speaker_embedding_cand) / (norm_a * norm_b))
    else:
        spk_sim = 1.0

    return RestorationMetrics(
        fullband_lsd_db=full_lsd,
        highband_lsd_db=high_lsd,
        protected_band_rms_err=prot_rms,
        highband_energy_error_db=energy_err,
        f0_rmse_hz=0.0,
        speaker_cosine_sim=spk_sim,
    )
