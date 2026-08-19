"""Timing, duration integrity, and landmark drift analysis."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimingIntegrityResult:
    """Timing and landmark stability statistics."""

    passed: bool
    max_drift_ms: float
    duration_ratio: float
    envelope_correlation: float
    failure_reasons: list[str] = field(default_factory=list)


def check_timing_integrity(
    orig_waveform: np.ndarray[Any, np.dtype[np.float32]],
    cand_waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    max_allowed_drift_ms: float = 40.0,
    min_envelope_correlation: float = 0.80,
) -> TimingIntegrityResult:
    """Validate that candidate audio maintains strict monotonic timing alignment with original."""
    reasons: list[str] = []

    orig_len = len(orig_waveform)
    cand_len = len(cand_waveform)

    if orig_len == 0 or cand_len == 0:
        return TimingIntegrityResult(
            passed=(orig_len == cand_len),
            max_drift_ms=0.0,
            duration_ratio=1.0 if orig_len == cand_len else 0.0,
            envelope_correlation=1.0 if orig_len == cand_len else 0.0,
        )

    duration_ratio = float(cand_len / orig_len)
    if abs(duration_ratio - 1.0) > 0.01:
        reasons.append(
            f"Length mismatch: candidate length ratio is {duration_ratio:.4f} (expected 1.0)"
        )

    # Compute short-term RMS energy envelope (20ms windows, 10ms hop)
    hop = int(round(sample_rate * 0.010))
    win = int(round(sample_rate * 0.020))

    num_frames = max(1, min(orig_len, cand_len) // hop)
    env_orig = np.zeros(num_frames, dtype=np.float32)
    env_cand = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop
        end = min(min(orig_len, cand_len), start + win)
        env_orig[i] = np.sqrt(np.mean(orig_waveform[start:end] ** 2) + 1e-12)
        env_cand[i] = np.sqrt(np.mean(cand_waveform[start:end] ** 2) + 1e-12)

    # Cross-correlation between energy envelopes
    norm_orig = np.linalg.norm(env_orig)
    norm_cand = np.linalg.norm(env_cand)

    if norm_orig > 1e-6 and norm_cand > 1e-6:
        corr = float(np.dot(env_orig, env_cand) / (norm_orig * norm_cand))
    else:
        corr = 1.0

    if corr < min_envelope_correlation:
        reasons.append(
            f"Energy envelope correlation {corr:.3f} below minimum {min_envelope_correlation:.3f}"
        )

    # Estimate timing lag of onset landmarks via cross-correlation peak
    std_orig = float(np.std(env_orig))
    std_cand = float(np.std(env_cand))
    mean_orig = float(np.mean(env_orig))

    if std_orig > 1e-5 and std_cand > 1e-5 and (std_orig / (mean_orig + 1e-8)) > 0.05:
        xcorr = np.correlate(
            env_cand - np.mean(env_cand), env_orig - np.mean(env_orig), mode="full"
        )
        lag_frames = int(np.argmax(xcorr) - (num_frames - 1))
        drift_ms = abs(lag_frames * 10.0)
    else:
        # Stationary signal (e.g. steady tone) has no landmark onsets to drift
        drift_ms = 0.0

    if drift_ms > max_allowed_drift_ms:
        reasons.append(
            f"Estimated envelope drift {drift_ms:.1f}ms exceeds threshold {max_allowed_drift_ms:.1f}ms"
        )

    passed = len(reasons) == 0

    return TimingIntegrityResult(
        passed=passed,
        max_drift_ms=drift_ms,
        duration_ratio=duration_ratio,
        envelope_correlation=corr,
        failure_reasons=reasons,
    )
