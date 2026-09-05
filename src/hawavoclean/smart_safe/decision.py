"""Deterministic, fail-closed Smart Safe decision foundation.

This module deliberately does not claim that Smart Safe is qualified.  It
implements the product invariants that must surround a future listener-trained,
signed ranker:

* route eligibility follows measured evidence and explicit consent;
* every hard guard is evaluated before quality ranking;
* candidate input order cannot change the result;
* ties and low-confidence decisions choose the least intervention;
* an unqualified ranker is rejected when production qualification is required.

The analyzer, preview renderer, signed ranker artifact, and post-master guard
runner are separate boundaries.  Callers must provide their evidence for every
candidate; a missing preview is a rejection, never an implicit pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final, Literal

Route = Literal[
    "preserve",
    "production",
    "studio",
    "lowband",
    "lowband_then_production",
    "restore_source",
    "restore_enrolled",
]
RestorePolicy = Literal["disabled", "source_allowed", "enrolled_only", "auto"]

ROUTES: Final[tuple[Route, ...]] = (
    "preserve",
    "production",
    "studio",
    "lowband",
    "lowband_then_production",
    "restore_source",
    "restore_enrolled",
)
RESTORE_ROUTES: Final[frozenset[Route]] = frozenset({"restore_source", "restore_enrolled"})
INTERVENTION_COST: Final[dict[Route, int]] = {
    "preserve": 0,
    "production": 1,
    "lowband": 1,
    "studio": 2,
    "lowband_then_production": 2,
    "restore_source": 3,
    "restore_enrolled": 4,
}


class UnqualifiedRankerError(RuntimeError):
    """Production Smart Safe was asked to use an unqualified ranker."""


def _probability(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and between 0 and 1")
    return result


@dataclass(frozen=True, slots=True)
class AcousticEvidence:
    """Normalized file/region evidence emitted by the acoustic analyzer."""

    speech_dominance: float
    music_risk: float
    crosstalk_risk: float
    rumble_confidence: float
    band_limited_confidence: float
    recorded_high_frequency_speech_confidence: float
    speaker_match_confidence: float = 0.0
    speaker_match_verified: bool = False
    reconstruction_consent: bool = False

    def __post_init__(self) -> None:
        for field in (
            "speech_dominance",
            "music_risk",
            "crosstalk_risk",
            "rumble_confidence",
            "band_limited_confidence",
            "recorded_high_frequency_speech_confidence",
            "speaker_match_confidence",
        ):
            object.__setattr__(self, field, _probability(getattr(self, field), field))
        if self.speaker_match_verified and self.speaker_match_confidence == 0.0:
            raise ValueError("a verified speaker match must carry non-zero confidence")


@dataclass(frozen=True, slots=True)
class SmartSafePolicy:
    """Locked thresholds around the future signed ranker."""

    speech_dominance_min: float = 0.70
    music_risk_max: float = 0.20
    crosstalk_risk_max: float = 0.20
    rumble_confidence_min: float = 0.70
    band_limited_confidence_min: float = 0.85
    recorded_hf_speech_max: float = 0.10
    speaker_match_confidence_min: float = 0.90
    decision_confidence_min: float = 0.70
    tie_margin_mos: float = 0.05
    intervention_penalty_mos: float = 0.03
    stable_boundary_confidence_min: float = 0.80
    uncertain_region_max_s: float = 1.0

    def __post_init__(self) -> None:
        for field in (
            "speech_dominance_min",
            "music_risk_max",
            "crosstalk_risk_max",
            "rumble_confidence_min",
            "band_limited_confidence_min",
            "recorded_hf_speech_max",
            "speaker_match_confidence_min",
            "decision_confidence_min",
            "stable_boundary_confidence_min",
        ):
            object.__setattr__(self, field, _probability(getattr(self, field), field))
        if not math.isfinite(self.tie_margin_mos) or self.tie_margin_mos < 0.0:
            raise ValueError("tie_margin_mos must be finite and non-negative")
        if not math.isfinite(self.intervention_penalty_mos) or self.intervention_penalty_mos < 0.0:
            raise ValueError("intervention_penalty_mos must be finite and non-negative")
        if not math.isfinite(self.uncertain_region_max_s) or self.uncertain_region_max_s < 0.0:
            raise ValueError("uncertain_region_max_s must be finite and non-negative")


DEFAULT_POLICY: Final = SmartSafePolicy()


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Preview and post-master evidence for one candidate route."""

    route: Route
    predicted_quality_mos: float
    prediction_confidence: float
    content_guard_passed: bool
    speaker_guard_passed: bool
    protected_band_guard_passed: bool
    artifact_guard_passed: bool
    post_master_guard_passed: bool
    reconstruction_disclosed: bool = False
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"unknown candidate route: {self.route!r}")
        quality = float(self.predicted_quality_mos)
        if not math.isfinite(quality) or not 1.0 <= quality <= 5.0:
            raise ValueError("predicted_quality_mos must be finite and between 1 and 5")
        object.__setattr__(self, "predicted_quality_mos", quality)
        object.__setattr__(
            self,
            "prediction_confidence",
            _probability(self.prediction_confidence, "prediction_confidence"),
        )
        if self.evidence_sha256 is not None:
            value = self.evidence_sha256
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SmartSafeRanker:
    """Identity and qualification state of the monotonic ranker artifact.

    ``qualified`` is meaningful only when the artifact's signature, comparison
    corpus, calibration, and locked acceptance evidence have been verified by
    the caller.  This foundation intentionally ships no qualified artifact.
    """

    version: str
    artifact_sha256: str
    signed: bool
    qualified: bool

    def __post_init__(self) -> None:
        if not self.version or len(self.version) > 128:
            raise ValueError("ranker version must contain 1-128 characters")
        if len(self.artifact_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.artifact_sha256
        ):
            raise ValueError("ranker artifact_sha256 must be a lowercase SHA-256 digest")
        if self.qualified and not self.signed:
            raise ValueError("a qualified ranker must be signed")


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    route: Route
    eligible: bool
    safe: bool
    reasons: tuple[str, ...]
    rank_score: float | None = None
    prediction_confidence: float | None = None
    evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SmartSafeDecision:
    selected_route: Route
    confidence: float
    abstained: bool
    reason: str
    candidates: tuple[CandidateOutcome, ...]
    ranker_version: str
    ranker_sha256: str
    decision_sha256: str


def eligible_routes(
    evidence: AcousticEvidence,
    *,
    restore_policy: RestorePolicy,
    speaker_profile_id: str | None,
    policy: SmartSafePolicy = DEFAULT_POLICY,
) -> dict[Route, tuple[bool, tuple[str, ...]]]:
    """Return canonical route eligibility plus every failed rule."""

    if restore_policy not in {"disabled", "source_allowed", "enrolled_only", "auto"}:
        raise ValueError(f"unknown restore policy: {restore_policy!r}")

    result: dict[Route, tuple[bool, tuple[str, ...]]] = {
        "preserve": (True, ()),
        "production": (True, ()),
    }

    studio_reasons: list[str] = []
    if evidence.speech_dominance < policy.speech_dominance_min:
        studio_reasons.append("speech dominance is below the Studio threshold")
    if evidence.music_risk > policy.music_risk_max:
        studio_reasons.append("protected music risk is too high for Studio")
    if evidence.crosstalk_risk > policy.crosstalk_risk_max:
        studio_reasons.append("protected crosstalk risk is too high for Studio")
    result["studio"] = (not studio_reasons, tuple(studio_reasons))

    lowband_reasons = (
        ()
        if evidence.rumble_confidence >= policy.rumble_confidence_min
        else ("rumble or low-frequency contamination is not demonstrated",)
    )
    result["lowband"] = (not lowband_reasons, lowband_reasons)
    result["lowband_then_production"] = (not lowband_reasons, lowband_reasons)

    common_restore_reasons: list[str] = []
    if not evidence.reconstruction_consent:
        common_restore_reasons.append("generative reconstruction consent is absent")
    if evidence.band_limited_confidence < policy.band_limited_confidence_min:
        common_restore_reasons.append("a band-limited region is not verified")
    if evidence.recorded_high_frequency_speech_confidence > policy.recorded_hf_speech_max:
        common_restore_reasons.append("recorded high-frequency speech is protected")

    source_reasons = list(common_restore_reasons)
    if restore_policy not in {"source_allowed", "auto"}:
        source_reasons.append("the Restore policy does not allow source mode")
    result["restore_source"] = (not source_reasons, tuple(source_reasons))

    enrolled_reasons = list(common_restore_reasons)
    if restore_policy not in {"enrolled_only", "auto"}:
        enrolled_reasons.append("the Restore policy does not allow enrolled mode")
    if not speaker_profile_id:
        enrolled_reasons.append("no enrolled speaker profile was selected")
    if not evidence.speaker_match_verified:
        enrolled_reasons.append("the selected speaker profile was not verified")
    if evidence.speaker_match_confidence < policy.speaker_match_confidence_min:
        enrolled_reasons.append("speaker-match confidence is below the safety threshold")
    result["restore_enrolled"] = (not enrolled_reasons, tuple(enrolled_reasons))

    return {route: result[route] for route in ROUTES}


def _hard_guard_reasons(candidate: CandidateEvidence) -> tuple[str, ...]:
    reasons: list[str] = []
    if not candidate.content_guard_passed:
        reasons.append("content guard failed")
    if not candidate.speaker_guard_passed:
        reasons.append("speaker guard failed")
    if not candidate.protected_band_guard_passed:
        reasons.append("protected-band guard failed")
    if not candidate.artifact_guard_passed:
        reasons.append("artifact guard failed")
    if not candidate.post_master_guard_passed:
        reasons.append("post-master guard failed")
    if candidate.route in RESTORE_ROUTES and not candidate.reconstruction_disclosed:
        reasons.append("reconstruction disclosure is absent")
    return tuple(reasons)


def _least_intervention(candidates: list[CandidateEvidence]) -> CandidateEvidence:
    return min(candidates, key=lambda item: (INTERVENTION_COST[item.route], item.route))


def _decision_digest(
    selected: CandidateEvidence,
    outcomes: tuple[CandidateOutcome, ...],
    *,
    abstained: bool,
    ranker: SmartSafeRanker,
) -> str:
    payload = {
        "selected_route": selected.route,
        "abstained": abstained,
        "ranker_sha256": ranker.artifact_sha256,
        "candidates": [
            {
                "route": outcome.route,
                "eligible": outcome.eligible,
                "safe": outcome.safe,
                "reasons": list(outcome.reasons),
                "rank_score": outcome.rank_score,
                "prediction_confidence": outcome.prediction_confidence,
                "evidence_sha256": outcome.evidence_sha256,
            }
            for outcome in outcomes
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decide_smart_safe(
    evidence: AcousticEvidence,
    candidates: list[CandidateEvidence] | tuple[CandidateEvidence, ...],
    *,
    restore_policy: RestorePolicy,
    speaker_profile_id: str | None,
    ranker: SmartSafeRanker,
    policy: SmartSafePolicy = DEFAULT_POLICY,
    require_qualified_ranker: bool = True,
) -> SmartSafeDecision:
    """Reject unsafe candidates, then rank survivors deterministically."""

    if require_qualified_ranker and not ranker.qualified:
        raise UnqualifiedRankerError(
            "Smart Safe ranker is not independently qualified; use manual routes or an "
            "explicit experimental workflow"
        )
    by_route: dict[Route, CandidateEvidence] = {}
    for candidate in candidates:
        if candidate.route in by_route:
            raise ValueError(f"duplicate candidate evidence for {candidate.route}")
        by_route[candidate.route] = candidate

    eligibility = eligible_routes(
        evidence,
        restore_policy=restore_policy,
        speaker_profile_id=speaker_profile_id,
        policy=policy,
    )
    survivors: list[CandidateEvidence] = []
    outcomes: list[CandidateOutcome] = []
    for route in ROUTES:
        is_eligible, eligibility_reasons = eligibility[route]
        preview = by_route.get(route)
        if not is_eligible:
            outcomes.append(CandidateOutcome(route, False, False, eligibility_reasons))
            continue
        if preview is None:
            outcomes.append(CandidateOutcome(route, True, False, ("preview evidence is missing",)))
            continue
        guard_reasons = _hard_guard_reasons(preview)
        if guard_reasons:
            outcomes.append(
                CandidateOutcome(
                    route,
                    True,
                    False,
                    guard_reasons,
                    prediction_confidence=preview.prediction_confidence,
                    evidence_sha256=preview.evidence_sha256,
                )
            )
            continue
        score = preview.predicted_quality_mos - (
            INTERVENTION_COST[route] * policy.intervention_penalty_mos
        )
        survivors.append(preview)
        outcomes.append(
            CandidateOutcome(
                route,
                True,
                True,
                (),
                rank_score=score,
                prediction_confidence=preview.prediction_confidence,
                evidence_sha256=preview.evidence_sha256,
            )
        )

    if not survivors:
        raise RuntimeError("no Smart Safe candidate survived the hard guards")

    scores = {
        candidate.route: candidate.predicted_quality_mos
        - INTERVENTION_COST[candidate.route] * policy.intervention_penalty_mos
        for candidate in survivors
    }
    ranked = sorted(
        survivors,
        key=lambda item: (-scores[item.route], INTERVENTION_COST[item.route], item.route),
    )
    best = ranked[0]
    low_confidence = best.prediction_confidence < policy.decision_confidence_min
    tied = (
        len(ranked) > 1 and (scores[best.route] - scores[ranked[1].route]) <= policy.tie_margin_mos
    )
    abstained = low_confidence or tied
    selected = _least_intervention(survivors) if abstained else best

    if low_confidence:
        reason = "ranker confidence is low; selected the least-modified safe candidate"
    elif tied:
        reason = "safe candidates are effectively tied; selected the least-modified candidate"
    else:
        reason = "selected the highest-ranked candidate after all hard guards passed"
    canonical_outcomes = tuple(outcomes)
    return SmartSafeDecision(
        selected_route=selected.route,
        confidence=selected.prediction_confidence,
        abstained=abstained,
        reason=reason,
        candidates=canonical_outcomes,
        ranker_version=ranker.version,
        ranker_sha256=ranker.artifact_sha256,
        decision_sha256=_decision_digest(
            selected, canonical_outcomes, abstained=abstained, ranker=ranker
        ),
    )


@dataclass(frozen=True, slots=True)
class RegionRecommendation:
    start_s: float
    end_s: float
    route: Route
    confidence: float
    boundary_confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("region bounds must be finite")
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("region bounds must satisfy 0 <= start < end")
        if self.route not in ROUTES:
            raise ValueError(f"unknown region route: {self.route!r}")
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(
            self,
            "boundary_confidence",
            _probability(self.boundary_confidence, "boundary_confidence"),
        )


def stabilize_region_routes(
    regions: list[RegionRecommendation] | tuple[RegionRecommendation, ...],
    *,
    policy: SmartSafePolicy = DEFAULT_POLICY,
) -> tuple[RegionRecommendation, ...]:
    """Apply boundary hysteresis and safer-neighbour inheritance.

    This chooses routes only.  The renderer remains responsible for sample-
    aligned crossfades and for rerunning guards on every final region.
    """

    if not regions:
        return ()
    ordered = list(regions)
    for index, region in enumerate(ordered):
        if index and not math.isclose(
            ordered[index - 1].end_s, region.start_s, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("Smart Safe regions must be contiguous and ordered")

    stabilized: list[RegionRecommendation] = []
    for index, region in enumerate(ordered):
        duration = region.end_s - region.start_s
        uncertain = (
            duration <= policy.uncertain_region_max_s
            or region.confidence < policy.decision_confidence_min
        )
        unstable_boundary = (
            index > 0 and region.boundary_confidence < policy.stable_boundary_confidence_min
        )
        if not uncertain and not unstable_boundary:
            stabilized.append(region)
            continue

        neighbours: list[RegionRecommendation] = []
        if stabilized:
            neighbours.append(stabilized[-1])
        if index + 1 < len(ordered):
            neighbours.append(ordered[index + 1])
        if not neighbours:
            stabilized.append(region)
            continue
        inherited = min(
            neighbours,
            key=lambda item: (INTERVENTION_COST[item.route], item.route),
        )
        stabilized.append(
            RegionRecommendation(
                start_s=region.start_s,
                end_s=region.end_s,
                route=inherited.route,
                confidence=min(region.confidence, inherited.confidence),
                boundary_confidence=region.boundary_confidence,
            )
        )
    return tuple(stabilized)
