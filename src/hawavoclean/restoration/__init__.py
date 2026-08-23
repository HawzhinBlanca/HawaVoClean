"""HawaVoClean spectral restoration subsystem (HawaRestore-KD)."""

from hawavoclean.restoration.bandwidth import (
    BandwidthDetector,
    BandwidthEstimate,
    BandwidthEvidence,
)
from hawavoclean.restoration.base import RestorationCandidate, Restorer
from hawavoclean.restoration.config import (
    RestorationConfig,
    RestorationGuardConfig,
)
from hawavoclean.restoration.f0 import (
    F0Extractor,
    F0Statistics,
    F0Trajectory,
)
from hawavoclean.restoration.guard import (
    GuardRResult,
    RestorationGuard,
)
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.highband_events import (
    HighBandEventDetector,
    HighBandEventResult,
)
from hawavoclean.restoration.policy import (
    RestorationPolicyManager,
    SegmentRestorationDecision,
)
from hawavoclean.restoration.profiles import (
    ProfileValidationError,
    SpeakerF0Stats,
    SpeakerProfile,
    load_speaker_profile,
    validate_speaker_profile,
)
from hawavoclean.restoration.protected_band import (
    ProtectedBandVerification,
    compute_transition_mask,
    merge_protected_spectrum,
    verify_protected_band_invariance,
)
from hawavoclean.restoration.report import (
    RestorationReport,
    RestorationSegmentCounts,
)
from hawavoclean.restoration.universr_upstream import UniverSRBaseline

__all__ = [
    "BandwidthDetector",
    "BandwidthEstimate",
    "BandwidthEvidence",
    "F0Extractor",
    "F0Statistics",
    "F0Trajectory",
    "GuardRResult",
    "HawaRestoreKD",
    "HighBandEventDetector",
    "HighBandEventResult",
    "ProfileValidationError",
    "ProtectedBandVerification",
    "RestorationCandidate",
    "RestorationConfig",
    "RestorationGuard",
    "RestorationGuardConfig",
    "RestorationPolicyManager",
    "RestorationReport",
    "RestorationSegmentCounts",
    "Restorer",
    "SegmentRestorationDecision",
    "SpeakerF0Stats",
    "SpeakerProfile",
    "UniverSRBaseline",
    "compute_transition_mask",
    "load_speaker_profile",
    "merge_protected_spectrum",
    "validate_speaker_profile",
    "verify_protected_band_invariance",
]
