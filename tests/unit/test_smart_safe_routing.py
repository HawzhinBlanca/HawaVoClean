import random

import pytest

from hawavoclean.smart_safe.decision import (
    AcousticEvidence,
    CandidateEvidence,
    RegionRecommendation,
    SmartSafeRanker,
    decide_smart_safe,
    stabilize_region_routes,
)


def _sample_evidence() -> AcousticEvidence:
    return AcousticEvidence(
        speech_dominance=0.90,
        music_risk=0.05,
        crosstalk_risk=0.05,
        rumble_confidence=0.85,
        band_limited_confidence=0.95,
        recorded_high_frequency_speech_confidence=0.02,
        speaker_match_confidence=0.95,
        speaker_match_verified=True,
        reconstruction_consent=True,
    )


def _sample_ranker() -> SmartSafeRanker:
    return SmartSafeRanker(
        version="v1.0-test",
        artifact_sha256="0" * 64,
        signed=True,
        qualified=True,
    )


def test_candidate_order_invariance() -> None:
    """Decision must be strictly identical regardless of the order candidates are supplied."""
    evidence = _sample_evidence()
    ranker = _sample_ranker()

    candidates = [
        CandidateEvidence("preserve", 3.0, 0.95, True, True, True, True, True),
        CandidateEvidence("production", 4.2, 0.90, True, True, True, True, True),
        CandidateEvidence("studio", 4.5, 0.88, True, True, True, True, True),
        CandidateEvidence("lowband", 3.8, 0.92, True, True, True, True, True),
        CandidateEvidence("lowband_then_production", 4.3, 0.89, True, True, True, True, True),
        CandidateEvidence(
            "restore_source",
            4.6,
            0.85,
            True,
            True,
            True,
            True,
            True,
            reconstruction_disclosed=True,
        ),
        CandidateEvidence(
            "restore_enrolled",
            4.8,
            0.87,
            True,
            True,
            True,
            True,
            True,
            reconstruction_disclosed=True,
        ),
    ]

    base_decision = decide_smart_safe(
        evidence,
        candidates,
        restore_policy="auto",
        speaker_profile_id="speaker_01",
        ranker=ranker,
        require_qualified_ranker=False,
    )

    rng = random.Random(42)
    for _ in range(10):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        shuffled_decision = decide_smart_safe(
            evidence,
            shuffled,
            restore_policy="auto",
            speaker_profile_id="speaker_01",
            ranker=ranker,
            require_qualified_ranker=False,
        )
        assert shuffled_decision.selected_route == base_decision.selected_route
        assert shuffled_decision.decision_sha256 == base_decision.decision_sha256
        assert shuffled_decision.confidence == base_decision.confidence
        assert shuffled_decision.abstained == base_decision.abstained


def test_low_confidence_and_ties_abstain_to_least_intervention() -> None:
    evidence = _sample_evidence()
    ranker = _sample_ranker()

    # 1. Low confidence candidate (< decision_confidence_min = 0.70)
    low_conf_candidates = [
        CandidateEvidence("preserve", 3.0, 0.95, True, True, True, True, True),
        CandidateEvidence("production", 4.8, 0.50, True, True, True, True, True),  # low conf
    ]
    dec_low = decide_smart_safe(
        evidence,
        low_conf_candidates,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=ranker,
        require_qualified_ranker=False,
    )
    assert dec_low.abstained is True
    assert dec_low.selected_route == "preserve"

    # 2. Effectively tied candidates (within tie_margin_mos = 0.05)
    tied_candidates = [
        CandidateEvidence("preserve", 3.0, 0.95, True, True, True, True, True),
        CandidateEvidence("production", 4.00, 0.95, True, True, True, True, True),
        CandidateEvidence(
            "studio", 4.03, 0.95, True, True, True, True, True
        ),  # studio has penalty 0.06 -> 3.97 vs prod penalty 0.03 -> 3.97 (tied!)
    ]
    dec_tied = decide_smart_safe(
        evidence,
        tied_candidates,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=ranker,
        require_qualified_ranker=False,
    )
    assert dec_tied.abstained is True
    # Ties pick least intervention cost: production (cost 1) vs studio (cost 2)
    assert dec_tied.selected_route in {"preserve", "production"}


def test_hard_guard_failure_eliminates_candidate() -> None:
    evidence = _sample_evidence()
    ranker = _sample_ranker()

    failing_guards = [
        CandidateEvidence("preserve", 3.0, 0.95, True, True, True, True, True),
        CandidateEvidence(
            "production",
            4.9,
            0.95,
            content_guard_passed=False,
            speaker_guard_passed=True,
            protected_band_guard_passed=True,
            artifact_guard_passed=True,
            post_master_guard_passed=True,
        ),
    ]
    decision = decide_smart_safe(
        evidence,
        failing_guards,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=ranker,
        require_qualified_ranker=False,
    )
    assert decision.selected_route == "preserve"
    prod_outcome = next(c for c in decision.candidates if c.route == "production")
    assert prod_outcome.safe is False
    assert "content guard failed" in prod_outcome.reasons


def test_stabilize_region_routes_hysteresis_and_inheritance() -> None:
    # 1. Contiguous regions with a short uncertain region in the middle
    regions = [
        RegionRecommendation(0.0, 5.0, "studio", confidence=0.90, boundary_confidence=0.90),
        # Short 0.5s uncertain region
        RegionRecommendation(5.0, 5.5, "restore_source", confidence=0.40, boundary_confidence=0.50),
        RegionRecommendation(5.5, 10.0, "production", confidence=0.88, boundary_confidence=0.85),
    ]

    stabilized = stabilize_region_routes(regions)
    assert len(stabilized) == 3
    # The middle uncertain region should inherit the safer (lower intervention) neighbor: production (cost 1) over studio (cost 2)
    assert stabilized[1].route == "production"
    assert stabilized[1].start_s == 5.0
    assert stabilized[1].end_s == 5.5

    # 2. Non-contiguous regions raise ValueError
    bad_regions = [
        RegionRecommendation(0.0, 2.0, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(3.0, 5.0, "production", confidence=0.90, boundary_confidence=0.90),
    ]
    with pytest.raises(ValueError, match="contiguous and ordered"):
        stabilize_region_routes(bad_regions)
