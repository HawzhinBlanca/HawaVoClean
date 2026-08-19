"""Phase coherence and spectral magnitude coherence estimation."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoherenceResult:
    """Spectral and phase coherence statistics."""

    phase_coherence: float
    magnitude_coherence: float
    is_phase_coherent: bool
    passed: bool
    reason: str = ""


def estimate_coherence(
    orig: np.ndarray[Any, np.dtype[np.float32]],
    cand: np.ndarray[Any, np.dtype[np.float32]],
    min_coherence: float = 0.70,
    expected_phase_coherent: bool = True,
) -> CoherenceResult:
    """Measure STFT phase cosine similarity and magnitude cross-correlation."""
    n = min(len(orig), len(cand))
    if n < 512:
        return CoherenceResult(
            phase_coherence=1.0,
            magnitude_coherence=1.0,
            is_phase_coherent=True,
            passed=True,
        )

    w1 = orig[:n]
    w2 = cand[:n]

    n_fft = 1024
    hop = 256
    win = np.hanning(n_fft)

    num_frames = (n - n_fft) // hop + 1
    if num_frames <= 0:
        return CoherenceResult(1.0, 1.0, True, True)

    stft1 = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.complex64)
    stft2 = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.complex64)

    for i in range(num_frames):
        stft1[i] = np.fft.rfft(w1[i * hop : i * hop + n_fft] * win, n=n_fft)
        stft2[i] = np.fft.rfft(w2[i * hop : i * hop + n_fft] * win, n=n_fft)

    # Phase cosine difference: cos(phi1 - phi2)
    phase_diff = np.angle(stft1) - np.angle(stft2)
    cos_diff = np.cos(phase_diff)
    # Weight by magnitude product
    mag_prod = np.abs(stft1) * np.abs(stft2)
    total_mag = float(np.sum(mag_prod))
    mean_phase_coh = float(np.sum(cos_diff * mag_prod) / total_mag) if total_mag > 1e-8 else 1.0

    # Magnitude correlation
    mag1_flat = np.abs(stft1).flatten()
    mag2_flat = np.abs(stft2).flatten()
    norm1 = np.linalg.norm(mag1_flat)
    norm2 = np.linalg.norm(mag2_flat)

    if norm1 > 1e-6 and norm2 > 1e-6:
        mag_coh = float(np.dot(mag1_flat, mag2_flat) / (norm1 * norm2))
    else:
        mag_coh = 1.0

    is_phase_coh = mean_phase_coh >= 0.65

    if expected_phase_coherent and not is_phase_coh:
        passed = False
        reason = f"Phase coherence {mean_phase_coh:.3f} is below expected coherent threshold 0.65"
    elif mag_coh < min_coherence:
        passed = False
        reason = f"Magnitude coherence {mag_coh:.3f} below minimum {min_coherence:.3f}"
    else:
        passed = True
        reason = ""

    return CoherenceResult(
        phase_coherence=mean_phase_coh,
        magnitude_coherence=mag_coh,
        is_phase_coherent=is_phase_coh,
        passed=passed,
        reason=reason,
    )
