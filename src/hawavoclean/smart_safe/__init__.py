"""Fail-closed Smart Safe analysis, eligibility, and ranking contracts."""

from hawavoclean.smart_safe.analyzer import (
    DEFAULT_ANALYZER_CONFIG,
    AnalyzerConfig,
    ProbabilityEstimate,
    StreamingAcousticAnalyzer,
    StreamingAcousticReport,
    analyze_audio_stream,
)
from hawavoclean.smart_safe.decision import (
    ROUTES,
    AcousticEvidence,
    CandidateEvidence,
    CandidateOutcome,
    RegionRecommendation,
    SmartSafeDecision,
    SmartSafePolicy,
    SmartSafeRanker,
    UnqualifiedRankerError,
    decide_smart_safe,
    eligible_routes,
    stabilize_region_routes,
)

__all__ = [
    "ROUTES",
    "AcousticEvidence",
    "AnalyzerConfig",
    "CandidateEvidence",
    "CandidateOutcome",
    "DEFAULT_ANALYZER_CONFIG",
    "ProbabilityEstimate",
    "RegionRecommendation",
    "StreamingAcousticAnalyzer",
    "StreamingAcousticReport",
    "SmartSafeDecision",
    "SmartSafePolicy",
    "SmartSafeRanker",
    "UnqualifiedRankerError",
    "analyze_audio_stream",
    "decide_smart_safe",
    "eligible_routes",
    "stabilize_region_routes",
]
