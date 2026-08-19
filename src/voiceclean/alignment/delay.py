"""GCC-PHAT cross-correlation and fractional delay alignment."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DelayAlignmentResult:
    """Delay estimation and alignment outcome."""

    delay_samples: float
    delay_ms: float
    correlation_peak: float
    aligned_candidate: np.ndarray[Any, np.dtype[np.float32]]
    passed: bool
    reason: str = ""


def estimate_gcc_phat_delay(
    orig: np.ndarray[Any, np.dtype[np.float32]],
    cand: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    max_delay_ms: float = 50.0,
) -> DelayAlignmentResult:
    """Estimate fractional delay using GCC-PHAT cross-correlation."""
    n = min(len(orig), len(cand))
    if n < 512:
        return DelayAlignmentResult(
            delay_samples=0.0,
            delay_ms=0.0,
            correlation_peak=1.0,
            aligned_candidate=cand.copy(),
            passed=True,
        )

    w1 = orig[:n]
    w2 = cand[:n]

    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    X1 = np.fft.rfft(w1, n=n_fft)
    X2 = np.fft.rfft(w2, n=n_fft)

    # GCC-PHAT cross-power spectrum
    cross_power = X1 * np.conj(X2)
    cross_power_norm = cross_power / (np.abs(cross_power) + 1e-9)

    cc = np.fft.irfft(cross_power_norm, n=n_fft)
    # Shift zero-lag to center
    cc = np.fft.fftshift(cc)

    mid = len(cc) // 2
    max_lag_samples = int(round(sample_rate * (max_delay_ms / 1000.0)))
    search_region = cc[mid - max_lag_samples : mid + max_lag_samples + 1]

    peak_idx = int(np.argmax(search_region))
    peak_val = float(search_region[peak_idx])
    int_lag = peak_idx - max_lag_samples

    # Parabolic sub-sample fractional interpolation
    if 0 < peak_idx < len(search_region) - 1:
        alpha = float(search_region[peak_idx - 1])
        beta = float(search_region[peak_idx])
        gamma = float(search_region[peak_idx + 1])
        denom = 2.0 * (2.0 * beta - alpha - gamma)
        frac_offset = (alpha - gamma) / denom if abs(denom) > 1e-6 else 0.0
    else:
        frac_offset = 0.0

    delay_samples = float(int_lag + frac_offset)
    delay_ms = float((delay_samples / sample_rate) * 1000.0)

    # Shift candidate by integer delay (pad/trim to match original length)
    shift = int(round(delay_samples))
    if shift > 0:
        # Candidate is delayed relative to original -> shift left
        aligned = np.pad(cand[shift:], (0, min(shift, len(cand))), mode="constant")[: len(cand)]
    elif shift < 0:
        # Candidate is ahead of original -> shift right
        shift_abs = abs(shift)
        aligned = np.pad(cand, (shift_abs, 0), mode="constant")[: len(cand)]
    else:
        aligned = cand.copy()

    passed = abs(delay_ms) <= max_delay_ms
    reason = "" if passed else f"Delay {abs(delay_ms):.2f}ms exceeds maximum {max_delay_ms}ms"

    return DelayAlignmentResult(
        delay_samples=delay_samples,
        delay_ms=delay_ms,
        correlation_peak=peak_val,
        aligned_candidate=aligned.astype(np.float32),
        passed=passed,
        reason=reason,
    )
