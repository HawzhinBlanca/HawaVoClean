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
from hawavoclean.smart_safe.preview import (
    CandidatePreview,
    SmartSafePreviewEngine,
    abstain_to_least_intervention,
    compute_evidence_sha256,
    extract_preview_slice,
    verify_candidate_evidence_integrity,
    verify_post_master_invariants,
)
from hawavoclean.smart_safe.region import (
    RegionalAssemblyResult,
    filter_region_routes_for_acoustics,
    render_and_stitch_regions,
)

__all__ = [
    "ROUTES",
    "AcousticEvidence",
    "AnalyzerConfig",
    "CandidateEvidence",
    "CandidateOutcome",
    "CandidatePreview",
    "DEFAULT_ANALYZER_CONFIG",
    "ProbabilityEstimate",
    "RegionRecommendation",
    "RegionalAssemblyResult",
    "SmartSafeDecision",
    "SmartSafePolicy",
    "SmartSafePreviewEngine",
    "SmartSafeRanker",
    "StreamingAcousticAnalyzer",
    "StreamingAcousticReport",
    "UnqualifiedRankerError",
    "abstain_to_least_intervention",
    "analyze_audio_stream",
    "compute_evidence_sha256",
    "decide_smart_safe",
    "eligible_routes",
    "extract_preview_slice",
    "filter_region_routes_for_acoustics",
    "render_and_stitch_regions",
    "stabilize_region_routes",
    "verify_candidate_evidence_integrity",
    "verify_post_master_invariants",
]
