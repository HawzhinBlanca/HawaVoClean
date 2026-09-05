from __future__ import annotations

import hashlib
import itertools

import pytest

from hawavoclean.smart_safe import (
    AcousticEvidence,
    CandidateEvidence,
    RegionRecommendation,
    SmartSafeRanker,
    UnqualifiedRankerError,
    decide_smart_safe,
    eligible_routes,
    stabilize_region_routes,
)

pytestmark = pytest.mark.unit


def _ranker(*, qualified: bool = True) -> SmartSafeRanker:
    return SmartSafeRanker(
        version="ranker-test-v1",
        artifact_sha256=hashlib.sha256(b"ranker").hexdigest(),
        signed=True,
        qualified=qualified,
    )


def _evidence(**changes: object) -> AcousticEvidence:
    values: dict[str, object] = {
        "speech_dominance": 0.90,
        "music_risk": 0.05,
        "crosstalk_risk": 0.05,
        "rumble_confidence": 0.85,
        "band_limited_confidence": 0.95,
        "recorded_high_frequency_speech_confidence": 0.01,
        "speaker_match_confidence": 0.98,
        "speaker_match_verified": True,
        "reconstruction_consent": True,
    }
    values.update(changes)
    return AcousticEvidence(**values)  # type: ignore[arg-type]


def _candidate(
    route: str, quality: float, confidence: float = 0.95, **changes: bool
) -> CandidateEvidence:
    values: dict[str, object] = {
        "route": route,
        "predicted_quality_mos": quality,
        "prediction_confidence": confidence,
        "content_guard_passed": True,
        "speaker_guard_passed": True,
        "protected_band_guard_passed": True,
        "artifact_guard_passed": True,
        "post_master_guard_passed": True,
        "reconstruction_disclosed": route.startswith("restore_"),
        "evidence_sha256": hashlib.sha256(route.encode()).hexdigest(),
    }
    values.update(changes)
    return CandidateEvidence(**values)  # type: ignore[arg-type]


def test_production_and_preserve_are_always_eligible() -> None:
    routes = eligible_routes(
        _evidence(
            speech_dominance=0.0,
            music_risk=1.0,
            crosstalk_risk=1.0,
            rumble_confidence=0.0,
            band_limited_confidence=0.0,
            speaker_match_verified=False,
            speaker_match_confidence=0.0,
            reconstruction_consent=False,
        ),
        restore_policy="disabled",
        speaker_profile_id=None,
    )
    assert routes["preserve"] == (True, ())
    assert routes["production"] == (True, ())
    assert all(not routes[route][0] for route in routes if route not in {"preserve", "production"})


def test_studio_requires_speech_without_music_or_crosstalk_risk() -> None:
    routes = eligible_routes(
        _evidence(speech_dominance=0.4, music_risk=0.8, crosstalk_risk=0.7),
        restore_policy="auto",
        speaker_profile_id="speaker_1",
    )
    eligible, reasons = routes["studio"]
    assert eligible is False
    assert len(reasons) == 3


def test_restore_requires_consent_band_limit_and_no_recorded_hf_content() -> None:
    routes = eligible_routes(
        _evidence(
            reconstruction_consent=False,
            band_limited_confidence=0.2,
            recorded_high_frequency_speech_confidence=0.8,
        ),
        restore_policy="auto",
        speaker_profile_id="speaker_1",
    )
    assert routes["restore_source"][0] is False
    assert len(routes["restore_source"][1]) == 3
    assert routes["restore_enrolled"][0] is False


def test_enrolled_restore_requires_verified_matching_profile() -> None:
    routes = eligible_routes(
        _evidence(speaker_match_verified=False, speaker_match_confidence=0.4),
        restore_policy="enrolled_only",
        speaker_profile_id=None,
    )
    assert routes["restore_source"][0] is False
    assert routes["restore_enrolled"][0] is False
    assert "no enrolled speaker profile" in " ".join(routes["restore_enrolled"][1])


def test_unqualified_ranker_fails_closed_by_default() -> None:
    with pytest.raises(UnqualifiedRankerError, match="not independently qualified"):
        decide_smart_safe(
            _evidence(),
            [_candidate("preserve", 3.0), _candidate("production", 3.5)],
            restore_policy="disabled",
            speaker_profile_id=None,
            ranker=_ranker(qualified=False),
        )


@pytest.mark.parametrize(
    "failed_guard",
    [
        "content_guard_passed",
        "speaker_guard_passed",
        "protected_band_guard_passed",
        "artifact_guard_passed",
        "post_master_guard_passed",
    ],
)
def test_every_hard_guard_rejects_before_ranking(failed_guard: str) -> None:
    dangerous = _candidate("studio", 5.0, **{failed_guard: False})
    decision = decide_smart_safe(
        _evidence(),
        [_candidate("preserve", 3.0), _candidate("production", 3.5), dangerous],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(),
    )
    assert decision.selected_route == "production"
    studio = next(item for item in decision.candidates if item.route == "studio")
    assert studio.safe is False
    assert any("guard failed" in reason for reason in studio.reasons)


def test_restore_without_disclosure_is_rejected() -> None:
    decision = decide_smart_safe(
        _evidence(),
        [
            _candidate("preserve", 3.0),
            _candidate("production", 3.5),
            _candidate("restore_source", 5.0, reconstruction_disclosed=False),
        ],
        restore_policy="source_allowed",
        speaker_profile_id=None,
        ranker=_ranker(),
    )
    assert decision.selected_route == "production"
    restore = next(item for item in decision.candidates if item.route == "restore_source")
    assert restore.reasons == ("reconstruction disclosure is absent",)


def test_candidate_enumeration_order_cannot_change_decision_or_digest() -> None:
    candidates = [
        _candidate("preserve", 3.0),
        _candidate("production", 3.8),
        _candidate("studio", 4.2),
        _candidate("lowband", 3.6),
    ]
    results = {
        (
            decision.selected_route,
            decision.abstained,
            decision.decision_sha256,
            tuple(item.route for item in decision.candidates),
        )
        for order in itertools.permutations(candidates)
        for decision in [
            decide_smart_safe(
                _evidence(),
                list(order),
                restore_policy="disabled",
                speaker_profile_id=None,
                ranker=_ranker(),
            )
        ]
    }
    assert len(results) == 1
    assert next(iter(results))[0] == "studio"


def test_tie_abstains_to_least_intervention() -> None:
    decision = decide_smart_safe(
        _evidence(),
        [_candidate("preserve", 3.50), _candidate("production", 3.53)],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(),
    )
    assert decision.abstained is True
    assert decision.selected_route == "preserve"
    assert "tied" in decision.reason


def test_low_confidence_abstains_to_least_intervention_even_when_quality_is_high() -> None:
    decision = decide_smart_safe(
        _evidence(),
        [_candidate("preserve", 2.5, 0.95), _candidate("studio", 5.0, 0.2)],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(),
    )
    assert decision.abstained is True
    assert decision.selected_route == "preserve"
    assert "confidence is low" in decision.reason


def test_missing_preview_is_explicitly_rejected() -> None:
    decision = decide_smart_safe(
        _evidence(),
        [_candidate("preserve", 3.0), _candidate("production", 3.5)],
        restore_policy="auto",
        speaker_profile_id="speaker_1",
        ranker=_ranker(),
    )
    restore = next(item for item in decision.candidates if item.route == "restore_enrolled")
    assert restore.eligible is True
    assert restore.safe is False
    assert restore.reasons == ("preview evidence is missing",)


def test_no_safe_candidate_fails_instead_of_silently_selecting_one() -> None:
    with pytest.raises(RuntimeError, match="no Smart Safe candidate survived"):
        decide_smart_safe(
            _evidence(),
            [
                _candidate("preserve", 3.0, artifact_guard_passed=False),
                _candidate("production", 3.5, content_guard_passed=False),
            ],
            restore_policy="disabled",
            speaker_profile_id=None,
            ranker=_ranker(),
        )


def test_region_hysteresis_inherits_safer_neighbour_for_short_region() -> None:
    regions = (
        RegionRecommendation(0.0, 5.0, "production", 0.95, 1.0),
        RegionRecommendation(5.0, 5.5, "restore_source", 0.95, 0.95),
        RegionRecommendation(5.5, 10.0, "studio", 0.95, 0.95),
    )
    stabilized = stabilize_region_routes(regions)
    assert [region.route for region in stabilized] == ["production", "production", "studio"]


def test_unstable_boundary_inherits_previous_safer_route() -> None:
    regions = (
        RegionRecommendation(0.0, 5.0, "preserve", 0.95, 1.0),
        RegionRecommendation(5.0, 10.0, "studio", 0.95, 0.2),
    )
    stabilized = stabilize_region_routes(regions)
    assert stabilized[1].route == "preserve"


def test_region_input_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        stabilize_region_routes(
            (
                RegionRecommendation(0.0, 1.0, "preserve", 1.0, 1.0),
                RegionRecommendation(2.0, 3.0, "production", 1.0, 1.0),
            )
        )


def test_smart_safe_decision_error_branches() -> None:
    from hawavoclean.smart_safe.decision import SmartSafePolicy, _probability

    # 1. _probability validation
    with pytest.raises(ValueError, match="between 0 and 1"):
        _probability(1.5, "test_field")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _probability(-0.1, "test_field")
    with pytest.raises(ValueError, match="between 0 and 1"):
        _probability(float("nan"), "test_field")

    # 2. AcousticEvidence validation
    with pytest.raises(ValueError, match="non-zero confidence"):
        AcousticEvidence(
            speech_dominance=0.9,
            music_risk=0.1,
            crosstalk_risk=0.1,
            rumble_confidence=0.1,
            band_limited_confidence=0.1,
            recorded_high_frequency_speech_confidence=0.1,
            speaker_match_confidence=0.0,
            speaker_match_verified=True,
        )

    # 3. SmartSafePolicy validation
    with pytest.raises(ValueError, match="tie_margin_mos"):
        SmartSafePolicy(tie_margin_mos=-1.0)
    with pytest.raises(ValueError, match="intervention_penalty_mos"):
        SmartSafePolicy(intervention_penalty_mos=-1.0)
    with pytest.raises(ValueError, match="uncertain_region_max_s"):
        SmartSafePolicy(uncertain_region_max_s=-1.0)

    # 4. CandidateEvidence validation
    with pytest.raises(ValueError, match="unknown candidate route"):
        CandidateEvidence(
            route="unknown_route",  # type: ignore[arg-type]
            predicted_quality_mos=4.0,
            prediction_confidence=0.9,
            content_guard_passed=True,
            speaker_guard_passed=True,
            protected_band_guard_passed=True,
            artifact_guard_passed=True,
            post_master_guard_passed=True,
        )
    with pytest.raises(ValueError, match="between 1 and 5"):
        CandidateEvidence(
            route="preserve",
            predicted_quality_mos=0.5,
            prediction_confidence=0.9,
            content_guard_passed=True,
            speaker_guard_passed=True,
            protected_band_guard_passed=True,
            artifact_guard_passed=True,
            post_master_guard_passed=True,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CandidateEvidence(
            route="preserve",
            predicted_quality_mos=4.0,
            prediction_confidence=0.9,
            content_guard_passed=True,
            speaker_guard_passed=True,
            protected_band_guard_passed=True,
            artifact_guard_passed=True,
            post_master_guard_passed=True,
            evidence_sha256="bad_sha",
        )

    # 5. RegionRecommendation validation
    with pytest.raises(ValueError, match="0 <= start < end"):
        RegionRecommendation(5.0, 4.0, "preserve", 1.0, 1.0)
    with pytest.raises(ValueError, match="0 <= start < end"):
        RegionRecommendation(-1.0, 1.0, "preserve", 1.0, 1.0)
