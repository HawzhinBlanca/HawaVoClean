"""Targeted branch tests for hawavoclean.smart_safe.calibration."""

from __future__ import annotations

import pytest

from hawavoclean.smart_safe.calibration import (
    TemperatureCalibrator,
    compute_expected_calibration_error,
    compute_route_regret,
    verify_abstention_and_tie_properties,
)
from hawavoclean.smart_safe.decision import (
    CandidateEvidence,
    CandidateOutcome,
    Route,
    SmartSafeDecision,
    SmartSafePolicy,
)


def test_temperature_calibrator_validation_and_properties() -> None:
    # 1. Invalid initial temperature
    with pytest.raises(ValueError, match="initial_temperature must be a finite positive"):
        TemperatureCalibrator(initial_temperature=0.0)
    with pytest.raises(ValueError, match="initial_temperature must be a finite positive"):
        TemperatureCalibrator(initial_temperature=-1.5)
    with pytest.raises(ValueError, match="initial_temperature must be a finite positive"):
        TemperatureCalibrator(initial_temperature=float("nan"))

    cal = TemperatureCalibrator(initial_temperature=2.0)
    assert cal.temperature == 2.0
    assert not cal.fitted

    # 2. Fit validation: empty or length mismatch
    with pytest.raises(ValueError, match="matching lengths"):
        cal.fit([], [])
    with pytest.raises(ValueError, match="matching lengths"):
        cal.fit([1.0, 2.0], [1])

    # 3. Fit validation: non-binary labels
    with pytest.raises(ValueError, match="labels must be binary"):
        cal.fit([1.0, 2.0], [0, 2])


def test_ece_validation_and_empty_bins() -> None:
    # 1. Length mismatch or empty
    with pytest.raises(ValueError, match="equal non-zero lengths"):
        compute_expected_calibration_error([], [])
    with pytest.raises(ValueError, match="equal non-zero lengths"):
        compute_expected_calibration_error([0.5], [1, 0])

    # 2. num_bins < 2
    with pytest.raises(ValueError, match="num_bins must be >= 2"):
        compute_expected_calibration_error([0.5], [1], num_bins=1)

    # 3. Out of bounds confidence
    with pytest.raises(ValueError, match="bounded in"):
        compute_expected_calibration_error([-0.1], [1])
    with pytest.raises(ValueError, match="bounded in"):
        compute_expected_calibration_error([1.1], [1])

    # 4. Empty bins: samples only in first bin
    m = compute_expected_calibration_error([0.05, 0.08], [1, 0], num_bins=10)
    assert len(m.bin_counts) == 10
    assert m.bin_counts[0] == 2
    assert m.bin_counts[1] == 0


def test_compute_route_regret_validation_and_single_sample() -> None:
    # 1. Length mismatch or empty
    with pytest.raises(ValueError, match="matching non-zero lengths"):
        compute_route_regret([], [])
    with pytest.raises(ValueError, match="matching non-zero lengths"):
        compute_route_regret([None], [])  # type: ignore[list-item]

    cand_pass = CandidateOutcome(
        route="production",
        eligible=True,
        safe=True,
        reasons=(),
        rank_score=3.5,
        prediction_confidence=0.9,
    )
    cand_studio = CandidateOutcome(
        route="studio",
        eligible=True,
        safe=True,
        reasons=(),
        rank_score=4.0,
        prediction_confidence=0.9,
    )
    cand_unsafe = CandidateOutcome(
        route="lowband",
        eligible=True,
        safe=False,
        reasons=("unsafe",),
        rank_score=2.0,
        prediction_confidence=0.9,
    )

    dec = SmartSafeDecision(
        selected_route="production",
        confidence=0.9,
        abstained=False,
        reason="preferred",
        candidates=(cand_pass, cand_studio, cand_unsafe),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )

    # 2. Missing rating for safe route
    with pytest.raises(ValueError, match="missing rating for safe route"):
        compute_route_regret([dec], [{"production": 3.5}])

    # 3. Missing rating for selected route
    dec_unrated_selected = SmartSafeDecision(
        selected_route="preserve",
        confidence=0.5,
        abstained=True,
        reason="low confidence",
        candidates=(cand_pass, cand_studio),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="missing rating for selected route"):
        compute_route_regret([dec_unrated_selected], [{"production": 3.5, "studio": 4.0}])

    # 4. Decision with no safe candidate
    dec_no_safe = SmartSafeDecision(
        selected_route="preserve",
        confidence=0.5,
        abstained=True,
        reason="none safe",
        candidates=(cand_unsafe,),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    no_safe_qualities: dict[Route, float] = {"preserve": 1.0}
    with pytest.raises(ValueError, match="contains no safe candidate route"):
        compute_route_regret([dec_no_safe], [no_safe_qualities])

    # 5. Single sample (n = 1) coverage for standard error and CI
    qualities: dict[Route, float] = {"production": 3.5, "studio": 4.0}
    summary = compute_route_regret([dec], [qualities])
    assert summary.num_evaluations == 1
    assert summary.mean_regret_mos == pytest.approx(0.5)
    assert summary.standard_error_mos == 0.0
    assert summary.ci95_upper_mos == summary.mean_regret_mos


def test_verify_abstention_and_tie_properties_branches() -> None:
    policy = SmartSafePolicy()

    # 1. No safe survivors
    ev_unsafe = CandidateEvidence(
        route="production",
        predicted_quality_mos=3.5,
        prediction_confidence=0.95,
        content_guard_passed=False,
        speaker_guard_passed=False,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
    )
    out_unsafe = CandidateOutcome(
        route="production",
        eligible=True,
        safe=False,
        reasons=("guard failure",),
    )
    dec = SmartSafeDecision(
        selected_route="preserve",
        confidence=0.5,
        abstained=True,
        reason="no safe",
        candidates=(out_unsafe,),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    ok, msg = verify_abstention_and_tie_properties([ev_unsafe], dec, policy)
    assert not ok and "no safe survivors" in msg

    # 2. Abstention mismatch (decision says not abstained when low confidence)
    ev_low_conf = CandidateEvidence(
        route="production",
        predicted_quality_mos=3.5,
        prediction_confidence=0.4,
        content_guard_passed=True,
        speaker_guard_passed=True,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
    )
    out_low_conf = CandidateOutcome(
        route="production",
        eligible=True,
        safe=True,
        reasons=(),
    )
    dec_wrong_abstention = SmartSafeDecision(
        selected_route="production",
        confidence=0.4,
        abstained=False,
        reason="forced",
        candidates=(out_low_conf,),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    ok, msg = verify_abstention_and_tie_properties([ev_low_conf], dec_wrong_abstention, policy)
    assert not ok and "abstention mismatch" in msg

    # 3. Abstained decision did not select least intervention
    ev_studio = CandidateEvidence(
        route="studio",
        predicted_quality_mos=3.5,
        prediction_confidence=0.4,
        content_guard_passed=True,
        speaker_guard_passed=True,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
    )
    out_studio = CandidateOutcome(
        route="studio",
        eligible=True,
        safe=True,
        reasons=(),
    )
    dec_wrong_route = SmartSafeDecision(
        selected_route="studio",
        confidence=0.4,
        abstained=True,
        reason="confidence is low",
        candidates=(out_low_conf, out_studio),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    ok, msg = verify_abstention_and_tie_properties(
        [ev_low_conf, ev_studio], dec_wrong_route, policy
    )
    assert not ok and "did not select least intervention" in msg

    # 4. Low-confidence reason missing from decision
    dec_missing_reason = SmartSafeDecision(
        selected_route="production",
        confidence=0.4,
        abstained=True,
        reason="arbitrary reason without key phrase",
        candidates=(out_low_conf,),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    ok, msg = verify_abstention_and_tie_properties([ev_low_conf], dec_missing_reason, policy)
    assert not ok and "low-confidence reason missing" in msg

    # 5. Tied reason missing from decision
    ev_tied1 = CandidateEvidence(
        route="production",
        predicted_quality_mos=4.0,
        prediction_confidence=0.9,
        content_guard_passed=True,
        speaker_guard_passed=True,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
    )
    ev_tied2 = CandidateEvidence(
        route="lowband",
        predicted_quality_mos=4.0,
        prediction_confidence=0.9,
        content_guard_passed=True,
        speaker_guard_passed=True,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
    )
    out_tied1 = CandidateOutcome(route="production", eligible=True, safe=True, reasons=())
    out_tied2 = CandidateOutcome(route="lowband", eligible=True, safe=True, reasons=())
    dec_tied_missing_reason = SmartSafeDecision(
        selected_route="lowband",
        confidence=0.9,
        abstained=True,
        reason="some arbitrary reason",
        candidates=(out_tied1, out_tied2),
        ranker_version="1.0",
        ranker_sha256="0" * 64,
        decision_sha256="1" * 64,
    )
    ok, msg = verify_abstention_and_tie_properties(
        [ev_tied1, ev_tied2], dec_tied_missing_reason, policy
    )
    assert not ok and "tied reason missing" in msg
