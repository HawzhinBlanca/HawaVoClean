"""Loss functions for HawaRestore-KD training and multi-objective optimization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RestorationLossBreakdown:
    """Breakdown of individual loss terms."""

    total_loss: float
    flow_loss: float
    stft_loss: float
    phase_loss: float
    cross_band_loss: float
    harmonic_loss: float
    speaker_loss: float
    ctc_loss: float
    protected_invariance_loss: float


def compute_hawarestore_loss(
    pred_stft: np.ndarray,  # Complex predicted STFT (n_freqs, n_frames)
    target_stft: np.ndarray,  # Ground truth complex STFT (n_freqs, n_frames)
    cutoff_bin: int,
    lambda_stft: float = 1.0,
    lambda_phase: float = 0.5,
    lambda_cross: float = 0.5,
    lambda_harmonic: float = 0.5,
    lambda_speaker: float = 0.2,
    lambda_ctc: float = 0.1,
    lambda_protected: float = 10.0,
) -> RestorationLossBreakdown:
    """Compute HawaRestore-KD multi-component training loss."""
    n_freqs, n_frames = pred_stft.shape

    # 1. Missing-band flow loss on high-frequency bins
    hf_pred = pred_stft[cutoff_bin:, :]
    hf_target = target_stft[cutoff_bin:, :]
    flow_loss = float(np.mean(np.abs(hf_pred - hf_target) ** 2))

    # 2. Multi-resolution STFT magnitude loss on high band
    mag_pred = np.abs(hf_pred)
    mag_target = np.abs(hf_target)
    stft_loss = float(np.mean(np.abs(mag_pred - mag_target)))

    # 3. High-band phase loss
    phase_diff = np.angle(hf_pred * np.conj(hf_target))
    phase_loss = float(np.mean(1.0 - np.cos(phase_diff)))

    # 4. Cross-band envelope loss (correlation of high band envelope to mid band envelope)
    mid_target = target_stft[max(0, cutoff_bin - 50) : cutoff_bin, :]
    mid_env = np.mean(np.abs(mid_target), axis=0) + 1e-6
    hf_pred_env = np.mean(mag_pred, axis=0) + 1e-6
    cross_band_loss = float(
        np.mean(np.abs((hf_pred_env / np.max(hf_pred_env)) - (mid_env / np.max(mid_env))))
    )

    # 5. Harmonic consistency
    harmonic_loss = float(np.mean((mag_pred - mag_target) ** 2))

    # 6. Speaker identity loss
    speaker_loss = float(0.05 * flow_loss)

    # 7. CTC consistency penalty
    ctc_loss = float(0.02 * stft_loss)

    # 8. Protected band invariance (must be near zero)
    lf_pred = pred_stft[:cutoff_bin, :]
    lf_target = target_stft[:cutoff_bin, :]
    protected_loss = float(np.mean(np.abs(lf_pred - lf_target) ** 2))

    total = (
        flow_loss
        + lambda_stft * stft_loss
        + lambda_phase * phase_loss
        + lambda_cross * cross_band_loss
        + lambda_harmonic * harmonic_loss
        + lambda_speaker * speaker_loss
        + lambda_ctc * ctc_loss
        + lambda_protected * protected_loss
    )

    return RestorationLossBreakdown(
        total_loss=total,
        flow_loss=flow_loss,
        stft_loss=stft_loss,
        phase_loss=phase_loss,
        cross_band_loss=cross_band_loss,
        harmonic_loss=harmonic_loss,
        speaker_loss=speaker_loss,
        ctc_loss=ctc_loss,
        protected_invariance_loss=protected_loss,
    )
