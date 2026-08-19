"""Unit selection policy and fail-closed decision engine."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from voiceclean.config import GuardConfig, PolicyConfig
from voiceclean.guard.protocol import ProbeResult, SpectralProbe
from voiceclean.guard.verdict import GuardEvaluationResult, GuardVerdict, evaluate_guard_pass
from voiceclean.policy.strength import generate_strength_candidates


@dataclass
class UnitPolicyDecision:
    """Decision outcome for an individual speech unit."""

    selected_waveform: np.ndarray[Any, np.dtype[np.float32]]
    is_enhanced: bool
    chosen_strength: float
    guard_verdict: GuardVerdict
    guard_scores: dict[str, float | str | bool] = field(default_factory=dict)
    decision_reason: str = ""


def evaluate_unit_policy(
    orig_core_waveform: np.ndarray[Any, np.dtype[np.float32]],
    enh_core_waveform: np.ndarray[Any, np.dtype[np.float32]] | None,
    sample_rate: int,
    is_speech: bool,
    probe: SpectralProbe,
    guard_config: GuardConfig,
    policy_config: PolicyConfig,
    phase_coherent: bool = True,
    cached_orig_probe: ProbeResult | None = None,
) -> tuple[UnitPolicyDecision, ProbeResult]:
    """Execute unit selection policy: evaluate strength ladder, enforce fail-closed fallback."""
    if not is_speech:
        return (
            UnitPolicyDecision(
                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=GuardVerdict.NO_SPEECH,
                guard_scores={"is_speech": False},
                decision_reason="Non-speech unit: neural enhancement bypassed.",
            ),
            cached_orig_probe or probe.infer(orig_core_waveform, sample_rate),
        )

    if enh_core_waveform is None:
        # Enhancer failed/errored upstream
        return (
            UnitPolicyDecision(
                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=GuardVerdict.ERROR,
                guard_scores={"enhancer_error": True},
                decision_reason="Enhancement core failed or timed out; fail-closed original audio used.",
            ),
            cached_orig_probe or probe.infer(orig_core_waveform, sample_rate),
        )

    # Generate strength candidates
    candidates = generate_strength_candidates(
        orig_waveform=orig_core_waveform,
        enh_waveform=enh_core_waveform,
        strength_ladder=policy_config.strength_ladder,
        phase_coherent=phase_coherent,
    )

    last_guard_res: GuardEvaluationResult | None = None
    orig_probe: ProbeResult = cached_orig_probe or probe.infer(orig_core_waveform, sample_rate)

    for strength, cand_wave in candidates:
        guard_res, orig_probe = evaluate_guard_pass(
            orig_waveform=orig_core_waveform,
            cand_waveform=cand_wave,
            sample_rate=sample_rate,
            is_speech=True,
            probe=probe,
            config=guard_config,
            cached_orig_probe=orig_probe,
        )
        last_guard_res = guard_res

        if guard_res.verdict == GuardVerdict.PASS:
            return (
                UnitPolicyDecision(
                    selected_waveform=cand_wave,
                    is_enhanced=True,
                    chosen_strength=strength,
                    guard_verdict=GuardVerdict.PASS,
                    guard_scores=guard_res.scores,
                    decision_reason=f"Passed Guard A with strength s={strength:.2f}",
                ),
                orig_probe,
            )

    # All candidates failed or were unverified -> fail-closed revert to original
    final_verdict = last_guard_res.verdict if last_guard_res else GuardVerdict.REVERT
    final_scores = last_guard_res.scores if last_guard_res else {}
    failure_reasons = (
        "; ".join(last_guard_res.reasons) if last_guard_res else "All candidates rejected."
    )

    return (
        UnitPolicyDecision(
            selected_waveform=orig_core_waveform.copy(),
            is_enhanced=False,
            chosen_strength=0.0,
            guard_verdict=final_verdict,
            guard_scores=final_scores,
            decision_reason=f"Reverted to original audio: {failure_reasons}",
        ),
        orig_probe,
    )
