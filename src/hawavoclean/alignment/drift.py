"""Local landmark drift and temporal warping detection."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DriftAnalysisResult:
    """Multi-window delay variance and drift metrics."""

    max_window_drift_ms: float
    delay_variance_ms: float
    passed: bool
    reason: str = ""


def analyze_local_drift(
    orig: np.ndarray[Any, np.dtype[np.float32]],
    cand: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    max_drift_thresh_ms: float = 20.0,
) -> DriftAnalysisResult:
    """Analyze whether delay varies locally across time windows (detecting time-stretching or warping)."""
    n = min(len(orig), len(cand))
    win_samples = int(round(sample_rate * window_s))
    hop_samples = int(round(sample_rate * hop_s))

    if n < win_samples:
        return DriftAnalysisResult(
            max_window_drift_ms=0.0,
            delay_variance_ms=0.0,
            passed=True,
        )

    delays_ms: list[float] = []
    num_windows = (n - win_samples) // hop_samples + 1

    for i in range(num_windows):
        start = i * hop_samples
        w_orig = orig[start : start + win_samples]
        w_cand = cand[start : start + win_samples]

        # Only evaluate active speech windows (RMS > 1e-4)
        if np.sqrt(np.mean(w_orig**2)) < 1e-4:
            continue

        corr = np.correlate(w_cand, w_orig, mode="full")
        mid = len(corr) // 2
        search_radius = int(round(sample_rate * 0.05))  # 50ms radius
        region = corr[max(0, mid - search_radius) : min(len(corr), mid + search_radius + 1)]
        if len(region) == 0:
            continue

        best_lag = int(np.argmax(region) - (len(region) // 2))
        delays_ms.append(float((best_lag / sample_rate) * 1000.0))

    if len(delays_ms) < 2:
        return DriftAnalysisResult(
            max_window_drift_ms=0.0,
            delay_variance_ms=0.0,
            passed=True,
        )

    drift_range = float(np.max(delays_ms) - np.min(delays_ms))
    drift_std = float(np.std(delays_ms))

    passed = drift_range <= max_drift_thresh_ms
    reason = (
        ""
        if passed
        else f"Local drift range {drift_range:.1f}ms exceeds threshold {max_drift_thresh_ms}ms"
    )

    return DriftAnalysisResult(
        max_window_drift_ms=drift_range,
        delay_variance_ms=drift_std,
        passed=passed,
        reason=reason,
    )
