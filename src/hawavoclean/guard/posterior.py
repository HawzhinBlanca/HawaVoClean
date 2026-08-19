"""Frame-level CTC posterior Jensen-Shannon divergence comparison."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PosteriorComparisonResult:
    """Statistics from CTC posterior distribution divergence analysis."""

    passed: bool
    mean_js_divergence: float
    max_peak_js_divergence: float
    voiced_frames_count: int
    failure_reasons: list[str] = field(default_factory=list)


def compute_js_divergence(
    p: np.ndarray[Any, np.dtype[np.float32]],
    q: np.ndarray[Any, np.dtype[np.float32]],
    eps: float = 1e-12,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute element-wise Jensen-Shannon divergence between probability vectors."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    m = 0.5 * (p_safe + q_safe)

    kl_pm = np.sum(p_safe * np.log2(p_safe / m), axis=-1)
    kl_qm = np.sum(q_safe * np.log2(q_safe / m), axis=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return np.clip(js, 0.0, 1.0).astype(np.float32)


def compare_ctc_posteriors(
    orig_posteriors: np.ndarray[Any, np.dtype[np.float32]],
    cand_posteriors: np.ndarray[Any, np.dtype[np.float32]],
    max_mean_js_div: float = 0.25,
    max_peak_js_div: float = 0.60,
    blank_threshold: float = 0.90,
) -> PosteriorComparisonResult:
    """Compare frame-level CTC posteriors across voiced/speech frames."""
    reasons: list[str] = []

    if orig_posteriors.size == 0 or cand_posteriors.size == 0:
        return PosteriorComparisonResult(
            passed=True,
            mean_js_divergence=0.0,
            max_peak_js_divergence=0.0,
            voiced_frames_count=0,
        )

    # Align frame counts if slight difference
    min_frames = min(len(orig_posteriors), len(cand_posteriors))
    p1 = orig_posteriors[:min_frames]
    p2 = cand_posteriors[:min_frames]

    # Focus on speech/voiced frames where blank probability < blank_threshold
    is_speech_frame = p1[:, 0] < blank_threshold
    voiced_count = int(np.sum(is_speech_frame))

    if voiced_count == 0:
        # Check all frames if all were classified as blank
        js_frames = compute_js_divergence(p1, p2)
    else:
        js_frames = compute_js_divergence(p1[is_speech_frame], p2[is_speech_frame])

    mean_js = float(np.mean(js_frames))
    max_js = float(np.max(js_frames)) if len(js_frames) > 0 else 0.0

    if mean_js > max_mean_js_div:
        reasons.append(
            f"Mean CTC posterior JS divergence {mean_js:.3f} exceeded threshold {max_mean_js_div:.3f}"
        )
    if max_js > max_peak_js_div:
        reasons.append(
            f"Peak CTC posterior JS divergence {max_js:.3f} exceeded threshold {max_peak_js_div:.3f}"
        )

    passed = len(reasons) == 0

    return PosteriorComparisonResult(
        passed=passed,
        mean_js_divergence=mean_js,
        max_peak_js_divergence=max_js,
        voiced_frames_count=voiced_count,
        failure_reasons=reasons,
    )
