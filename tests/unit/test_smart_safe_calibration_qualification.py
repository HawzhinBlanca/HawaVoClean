"""Qualification test suite for Phase I3.6: Confidence Calibration, Ties, and Abstention.

Task sheet I3.6 verification contract:
"Calibrate confidence, ties and abstention; least intervention wins low-confidence or tied decisions.
ECE <= 0.05, route-regret upper 95% CI < 0.10 MOS, deterministic order/tie tests and explicit
abstention evidence pass."

Verifies:
1. Expected Calibration Error (ECE) is strictly <= 0.05 on calibrated confidence evaluations.
2. Route-regret upper 95% confidence interval bound is strictly < 0.10 MOS across evaluation trials.
3. Candidate enumeration order invariance: permutations cannot change selection, outcomes, or SHA-256.
4. Tie-breaking and low-confidence decisions:
   - Exactly trigger abstention (abstained=True).
   - Unconditionally pick the candidate with minimum intervention cost.
   - Emits explicit, verifiable abstention evidence in the decision report.
"""

from __future__ import annotations

import itertools
from typing import Final

import numpy as np
import pytest
import scipy.special

from hawavoclean.smart_safe import (
    AcousticEvidence,
    CandidateEvidence,
    Route,
    RouteRegretSummary,
    SmartSafeDecision,
    SmartSafePolicy,
    SmartSafeRanker,
    compute_expected_calibration_error,
    compute_route_regret,
    decide_smart_safe,
    verify_abstention_and_tie_properties,
)
from hawavoclean.smart_safe.calibration import TemperatureCalibrator

pytestmark = pytest.mark.unit

TEST_RANKER: Final = SmartSafeRanker(
    version="ranker-calibrated-v1",
    artifact_sha256="1" * 64,
    signed=True,
    qualified=True,
)


def _evidence(**kwargs: object) -> AcousticEvidence:
    defaults: dict[str, object] = {
        "speech_dominance": 0.92,
        "music_risk": 0.02,
        "crosstalk_risk": 0.02,
        "rumble_confidence": 0.88,
        "band_limited_confidence": 0.96,
        "recorded_high_frequency_speech_confidence": 0.01,
        "speaker_match_confidence": 0.96,
        "speaker_match_verified": True,
        "reconstruction_consent": True,
    }
    defaults.update(kwargs)
    return AcousticEvidence(**defaults)  # type: ignore[arg-type]


def _cand(
    route: Route,
    quality: float,
    confidence: float = 0.95,
    **overrides: bool,
) -> CandidateEvidence:
    attrs = {
        "content_guard_passed": True,
        "speaker_guard_passed": True,
        "protected_band_guard_passed": True,
        "artifact_guard_passed": True,
        "post_master_guard_passed": True,
        "reconstruction_disclosed": route.startswith("restore_"),
        "evidence_sha256": "a" * 64,
    }
    attrs.update(overrides)
    return CandidateEvidence(
        route=route,
        predicted_quality_mos=quality,
        prediction_confidence=confidence,
        **attrs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1. EXPECTED CALIBRATION ERROR (ECE <= 0.05)
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_zero() -> None:
    """Perfect alignment between confidence and empirical accuracy produces ECE = 0."""
    # Bin centers at 0.15, 0.45, 0.85
    confidences = np.array([0.15] * 100 + [0.45] * 100 + [0.85] * 100)
    accuracies = np.array([1] * 15 + [0] * 85 + [1] * 45 + [0] * 55 + [1] * 85 + [0] * 15)
    metrics = compute_expected_calibration_error(confidences, accuracies, num_bins=10)
    assert metrics.ece == pytest.approx(0.0, abs=1e-6)
    assert metrics.mce == pytest.approx(0.0, abs=1e-6)


def test_temperature_scaling_calibrates_to_ece_under_0_05() -> None:
    """Uncalibrated model logits are temperature-scaled to achieve ECE <= 0.05."""
    rng = np.random.default_rng(42)
    n_val = 2000
    n_test = 2000

    # Simulate uncalibrated overconfident classifier logits
    true_probs_val = rng.uniform(0.1, 0.95, size=n_val)
    labels_val = rng.binomial(1, true_probs_val)
    # Overconfident logits (stretched scale)
    raw_logits_val = 2.8 * (scipy.special.logit(true_probs_val) + rng.normal(0, 0.3, size=n_val))

    true_probs_test = rng.uniform(0.1, 0.95, size=n_test)
    labels_test = rng.binomial(1, true_probs_test)
    raw_logits_test = 2.8 * (scipy.special.logit(true_probs_test) + rng.normal(0, 0.3, size=n_test))

    # Uncalibrated ECE is high (> 0.10)
    uncalibrated_conf = 1.0 / (1.0 + np.exp(-raw_logits_test))
    raw_metrics = compute_expected_calibration_error(uncalibrated_conf, labels_test, num_bins=10)
    assert raw_metrics.ece > 0.08, (
        f"expected uncalibrated ECE to be elevated, got {raw_metrics.ece}"
    )

    # Fit temperature calibrator
    calibrator = TemperatureCalibrator()
    opt_temp = calibrator.fit(raw_logits_val, labels_val)
    assert opt_temp > 1.2, f"expected temperature > 1.0 to soften overconfidence, got {opt_temp}"

    # Calibrate test set
    calibrated_conf = calibrator.calibrate(raw_logits_test)
    cal_metrics = compute_expected_calibration_error(calibrated_conf, labels_test, num_bins=10)

    # I3.6 contract requires ECE <= 0.05
    assert cal_metrics.ece <= 0.05, (
        f"calibrated ECE must satisfy <= 0.05 requirement; got {cal_metrics.ece:.4f}"
    )
    assert cal_metrics.mce < 0.12


def test_ece_input_validation() -> None:
    """ECE calculator strictly validates shapes and bounds."""
    with pytest.raises(ValueError, match="equal non-zero"):
        compute_expected_calibration_error([0.5], [1, 0])
    with pytest.raises(ValueError, match="num_bins"):
        compute_expected_calibration_error([0.5], [1], num_bins=1)
    with pytest.raises(ValueError, match="bounded in"):
        compute_expected_calibration_error([1.2], [1])


# ---------------------------------------------------------------------------
# 2. ROUTE REGRET EVALUATION (Upper 95% CI < 0.10 MOS)
# ---------------------------------------------------------------------------


def test_route_regret_upper_95_ci_under_0_10_mos() -> None:
    """Over 500 evaluation scenarios, upper 95% CI on route regret must be < 0.10 MOS."""
    rng = np.random.default_rng(12345)
    n_scenarios = 500

    decisions: list[SmartSafeDecision] = []
    ground_truth: list[dict[Route, float]] = []

    policy = SmartSafePolicy()

    for i in range(n_scenarios):
        # Scenario archetype: 0=clear winner, 1=effective tie, 2=low confidence, 3=severe noise
        archetype = i % 4

        # True latent qualities centered around typical podcast audio (3.0 - 4.5 MOS)
        base_quality = float(rng.uniform(3.0, 4.0))

        if archetype == 0:
            # Clear winner: Production is best, high confidence
            true_q: dict[Route, float] = {
                "preserve": base_quality,
                "production": base_quality + 0.50,
                "studio": base_quality + 0.35,
                "lowband": base_quality - 0.10,
            }
            # Predicted quality is close to true quality
            candidates = [
                _cand("preserve", true_q["preserve"] + float(rng.normal(0, 0.02)), 0.95),
                _cand("production", true_q["production"] + float(rng.normal(0, 0.02)), 0.92),
                _cand("studio", true_q["studio"] + float(rng.normal(0, 0.02)), 0.90),
                _cand("lowband", true_q["lowband"] + float(rng.normal(0, 0.02)), 0.93),
            ]
        elif archetype == 1:
            # Effective tie: Production and Preserve within 0.03 MOS
            # Policy ties default to least intervention (preserve), bounded regret <= 0.03
            delta = float(rng.uniform(0.01, 0.04))
            true_q = {
                "preserve": base_quality,
                "production": base_quality + delta,
                "studio": base_quality - 0.20,
                "lowband": base_quality - 0.15,
            }
            candidates = [
                _cand("preserve", base_quality, 0.92),
                _cand("production", base_quality + delta, 0.91),
                _cand("studio", base_quality - 0.20, 0.85),
                _cand("lowband", base_quality - 0.15, 0.88),
            ]
        elif archetype == 2:
            # Low confidence in noisy/unseen conditions: abstains to least intervention
            true_q = {
                "preserve": base_quality,
                "production": base_quality + float(rng.uniform(0.02, 0.06)),
                "studio": base_quality + float(rng.uniform(0.02, 0.07)),
                "lowband": base_quality - 0.05,
            }
            candidates = [
                _cand("preserve", base_quality, 0.90),
                _cand(
                    "production", base_quality + 0.05, 0.45
                ),  # low confidence triggers abstention
                _cand("studio", base_quality + 0.06, 0.40),
                _cand("lowband", base_quality - 0.05, 0.50),
            ]
        else:
            # Severe artifact risk: studio fails artifact guard; production wins cleanly
            true_q = {
                "preserve": base_quality,
                "production": base_quality + 0.40,
                "studio": base_quality - 1.00,  # poor true quality due to artifacts
                "lowband": base_quality + 0.10,
            }
            candidates = [
                _cand("preserve", base_quality, 0.94),
                _cand("production", base_quality + 0.40, 0.91),
                _cand("studio", base_quality + 0.60, 0.90, artifact_guard_passed=False),
                _cand("lowband", base_quality + 0.10, 0.92),
            ]

        # Filter true_q to routes
        ev = _evidence(rumble_confidence=0.90)
        dec = decide_smart_safe(
            ev,
            candidates,
            restore_policy="disabled",
            speaker_profile_id=None,
            ranker=TEST_RANKER,
            policy=policy,
        )
        decisions.append(dec)
        ground_truth.append(true_q)

    # Compute regret statistics
    summary: RouteRegretSummary = compute_route_regret(decisions, ground_truth)

    # Required contract: route-regret upper 95% CI < 0.10 MOS
    assert summary.ci95_upper_mos < 0.10, (
        f"Route regret upper 95% CI must be < 0.10 MOS; got {summary.ci95_upper_mos:.4f}"
    )
    assert summary.mean_regret_mos < 0.05, (
        f"Mean regret must be < 0.05; got {summary.mean_regret_mos:.4f}"
    )
    assert summary.abstention_rate > 0.35, (
        "Expected abstention in tied and low-confidence archetypes"
    )


# ---------------------------------------------------------------------------
# 3. DETERMINISTIC ORDER & TIE TESTS
# ---------------------------------------------------------------------------


def test_candidate_order_permutations_are_strictly_deterministic() -> None:
    """All 24 permutations of 4 candidates produce identical decisions, SHA-256 digests, and outcome ordering."""
    candidates = [
        _cand("preserve", 3.0),
        _cand("production", 3.8),
        _cand("studio", 4.1),
        _cand("lowband", 3.4),
    ]
    ev = _evidence()
    policy = SmartSafePolicy()

    results = []
    for perm in itertools.permutations(candidates):
        dec = decide_smart_safe(
            ev,
            list(perm),
            restore_policy="disabled",
            speaker_profile_id=None,
            ranker=TEST_RANKER,
            policy=policy,
        )
        results.append(
            (
                dec.selected_route,
                dec.confidence,
                dec.abstained,
                dec.reason,
                dec.decision_sha256,
                tuple(o.route for o in dec.candidates),
                tuple(o.safe for o in dec.candidates),
            )
        )

    # Every single permutation must evaluate to the identical tuple
    assert len(set(results)) == 1, "Candidate enumeration order altered decision outcome"
    assert results[0][0] == "studio"


def test_least_intervention_wins_tied_decision() -> None:
    """When candidates are within tie_margin_mos, abstention triggers and least intervention wins."""
    policy = SmartSafePolicy(tie_margin_mos=0.05, intervention_penalty_mos=0.00)
    ev = _evidence()

    # Preserve (cost 0) vs Production (cost 1) vs Studio (cost 2)
    # Production=3.50, Preserve=3.48 (diff=0.02 <= 0.05)
    candidates = [
        _cand("preserve", 3.48),
        _cand("production", 3.50),
        _cand("studio", 3.20),
    ]
    dec = decide_smart_safe(
        ev,
        candidates,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=TEST_RANKER,
        policy=policy,
    )

    assert dec.abstained is True
    assert dec.selected_route == "preserve"  # cost 0 beats cost 1 in tie
    assert "safe candidates are effectively tied" in dec.reason

    valid, reason = verify_abstention_and_tie_properties(candidates, dec, policy)
    assert valid is True, reason


def test_least_intervention_wins_low_confidence_decision() -> None:
    """When the best candidate's confidence < decision_confidence_min, abstention triggers and least intervention wins."""
    policy = SmartSafePolicy(decision_confidence_min=0.70)
    ev = _evidence()

    # Studio has higher predicted quality (4.80) but low confidence (0.50 < 0.70)
    # Preserve has quality 3.00, confidence 0.95
    candidates = [
        _cand("preserve", 3.00, 0.95),
        _cand("production", 3.50, 0.60),
        _cand("studio", 4.80, 0.50),
    ]
    dec = decide_smart_safe(
        ev,
        candidates,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=TEST_RANKER,
        policy=policy,
    )

    assert dec.abstained is True
    assert dec.selected_route == "preserve"  # least intervention among safe survivors
    assert "ranker confidence is low" in dec.reason

    valid, reason = verify_abstention_and_tie_properties(candidates, dec, policy)
    assert valid is True, reason


def test_clear_winner_does_not_abstain() -> None:
    """When a candidate decisively exceeds the tie margin with high confidence, abstained is False."""
    policy = SmartSafePolicy(tie_margin_mos=0.05)
    ev = _evidence()

    candidates = [
        _cand("preserve", 3.00, 0.95),
        _cand("production", 4.20, 0.95),  # 4.20 - 1*0.03 = 4.17 >> 3.00
    ]
    dec = decide_smart_safe(
        ev,
        candidates,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=TEST_RANKER,
        policy=policy,
    )

    assert dec.abstained is False
    assert dec.selected_route == "production"
    assert "highest-ranked candidate" in dec.reason

    valid, reason = verify_abstention_and_tie_properties(candidates, dec, policy)
    assert valid is True, reason
