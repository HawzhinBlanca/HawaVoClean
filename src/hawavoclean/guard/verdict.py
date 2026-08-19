"""Verdict evaluation aggregating token anchors, CTC posteriors, timing, and signal integrity."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hawavoclean.config import GuardConfig
from hawavoclean.guard.posterior import compare_ctc_posteriors
from hawavoclean.guard.protocol import ProbeResult, SpectralProbe
from hawavoclean.guard.signal import check_signal_integrity
from hawavoclean.guard.timing import check_timing_integrity
from hawavoclean.guard.token_anchor import compare_token_anchors


class GuardVerdict(StrEnum):
    """Authoritative verdicts as specified in BLUEPRINT.md section 13.4."""

    PASS = "PASS"
    REVERT = "REVERT"
    UNVERIFIED = "UNVERIFIED"
    ERROR = "ERROR"
    NO_SPEECH = "NO_SPEECH"


@dataclass
class GuardEvaluationResult:
    """Complete evaluation outcome and diagnostic scores for a single speech unit."""

    verdict: GuardVerdict
    scores: dict[str, float | str | bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def evaluate_guard_pass(
    orig_waveform: np.ndarray[Any, np.dtype[np.float32]],
    cand_waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    is_speech: bool,
    probe: SpectralProbe,
    config: GuardConfig,
    cached_orig_probe: ProbeResult | None = None,
    is_finishing_pass: bool = False,
) -> tuple[GuardEvaluationResult, ProbeResult]:
    """Run full Hawzhin Sorani Fidelity Guard comparison."""
    if not is_speech:
        return (
            GuardEvaluationResult(
                verdict=GuardVerdict.NO_SPEECH,
                scores={"is_speech": False},
                reasons=["Unit is non-speech; guard pass bypassed."],
            ),
            cached_orig_probe or probe.infer(orig_waveform, sample_rate),
        )

    # Step 0: structural sanity — these are not judgement calls, they are
    # hard failures. Every later check compares nan > thresh == False, so a
    # non-finite candidate would otherwise FAIL OPEN straight to PASS.
    if len(cand_waveform) == 0 or len(cand_waveform) != len(orig_waveform):
        return (
            GuardEvaluationResult(
                verdict=GuardVerdict.REVERT,
                scores={"cand_len": len(cand_waveform), "orig_len": len(orig_waveform)},
                reasons=[f"Candidate length {len(cand_waveform)} != original {len(orig_waveform)}"],
            ),
            cached_orig_probe or probe.infer(orig_waveform, sample_rate),
        )
    if not np.all(np.isfinite(cand_waveform)):
        return (
            GuardEvaluationResult(
                verdict=GuardVerdict.REVERT,
                scores={"non_finite_samples": int(np.sum(~np.isfinite(cand_waveform)))},
                reasons=["Candidate contains NaN/Inf samples"],
            ),
            cached_orig_probe or probe.infer(orig_waveform, sample_rate),
        )

    try:
        # Step 1: Probe inference
        orig_probe = cached_orig_probe or probe.infer(orig_waveform, sample_rate)
        cand_probe = probe.infer(cand_waveform, sample_rate)

        scores: dict[str, float | str | bool] = {}
        reasons: list[str] = []

        # Step 2: High-confidence token anchor comparison
        anchor_res = compare_token_anchors(
            orig_tokens=orig_probe.tokens,
            cand_tokens=cand_probe.tokens,
            min_anchor_confidence=config.min_anchor_confidence,
            max_timestamp_drift_ms=config.max_timing_drift_ms,
        )
        scores["high_conf_anchors"] = anchor_res.high_conf_anchors_count
        scores["deleted_anchors"] = anchor_res.deleted_anchors_count
        scores["substituted_anchors"] = anchor_res.substituted_anchors_count
        scores["anchor_drift_ms"] = anchor_res.max_timestamp_drift_ms

        strict = config.mode == "strict_spectral"
        if not is_finishing_pass and strict:
            if anchor_res.insufficient_anchors:
                return (
                    GuardEvaluationResult(
                        verdict=GuardVerdict.UNVERIFIED,
                        scores=scores,
                        reasons=anchor_res.failure_reasons,
                    ),
                    orig_probe,
                )

            if not anchor_res.passed:
                reasons.extend(anchor_res.failure_reasons)

        # Step 3: CTC posterior JS divergence
        post_res = compare_ctc_posteriors(
            orig_posteriors=orig_probe.frame_distributions,
            cand_posteriors=cand_probe.frame_distributions,
            max_mean_js_div=config.max_posterior_js_div,
            max_peak_js_div=config.max_peak_js_div,
        )
        scores["mean_js_div"] = post_res.mean_js_divergence
        scores["peak_js_div"] = post_res.max_peak_js_divergence

        if not post_res.passed:
            reasons.extend(post_res.failure_reasons)

        # Step 4: Timing & envelope integrity
        timing_res = check_timing_integrity(
            orig_waveform=orig_waveform,
            cand_waveform=cand_waveform,
            sample_rate=sample_rate,
            max_allowed_drift_ms=config.max_timing_drift_ms,
        )
        scores["envelope_correlation"] = timing_res.envelope_correlation
        scores["timing_drift_ms"] = timing_res.max_drift_ms

        if not timing_res.passed:
            reasons.extend(timing_res.failure_reasons or ["Timing integrity check failed"])

        # Step 5: Acoustic Signal integrity
        if config.enforce_signal_integrity:
            sig_res = check_signal_integrity(
                orig_waveform=orig_waveform,
                cand_waveform=cand_waveform,
                sample_rate=sample_rate,
                spectral_hole_thresh=config.spectral_hole_thresh,
                musical_noise_thresh=config.musical_noise_thresh,
                min_hf_preservation_ratio=config.min_hf_preservation_ratio,
            )
            scores["consonant_retention"] = sig_res.consonant_retention_ratio
            scores["spectral_hole_score"] = sig_res.spectral_hole_score
            scores["musical_noise_score"] = sig_res.musical_noise_score
            scores["clipping_samples"] = sig_res.clipping_samples_count

            if not sig_res.passed:
                reasons.extend(sig_res.failure_reasons)

        verdict = GuardVerdict.PASS if len(reasons) == 0 else GuardVerdict.REVERT
        return (
            GuardEvaluationResult(verdict=verdict, scores=scores, reasons=reasons),
            orig_probe,
        )

    except Exception as e:
        return (
            GuardEvaluationResult(
                verdict=GuardVerdict.ERROR,
                scores={"error": str(e)},
                reasons=[f"Guard evaluation execution failed with exception: {e}"],
            ),
            cached_orig_probe or ProbeResult("", ""),
        )
