"""HawaVoClean spectral restoration subsystem (HawaRestore-KD)."""

import importlib

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
    PostMasteringSegmentEvidence,
    PostMasteringVerificationResult,
    RestorationGuard,
)
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

#: Names whose modules define ``torch.nn.Module`` subclasses, so importing
#: them needs torch at class-definition time. Torch is an optional extra: a
#: natural-mode-only install is a supported configuration, and the CLI has to
#: come up on one. Eagerly re-exporting these made ``import hawavoclean.cli``
#: -- via multipass, via pipeline, via this package -- a hard torch
#: dependency, so the published wheel could not even print its own version
#: without the restore extra installed.
_TORCH_BACKED: dict[str, str] = {
    "HawaRestoreKD": "hawavoclean.restoration.hawarestore_kd",
    "UniverSRBaseline": "hawavoclean.restoration.universr_upstream",
    "compute_code_provenance": "hawavoclean.restoration.checkpoint",
    "compute_dependency_provenance": "hawavoclean.restoration.checkpoint",
    "load_safe_checkpoint": "hawavoclean.restoration.checkpoint",
    "save_safe_checkpoint": "hawavoclean.restoration.checkpoint",
}


def __getattr__(name: str) -> object:
    """Import a torch-backed restorer on first use (PEP 562)."""
    module_path = _TORCH_BACKED.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without torch
        raise ModuleNotFoundError(
            f"{name} needs the optional restore dependencies: {exc}. "
            "Install them with `pip install hawavoclean[restore]`; natural mode "
            "does not require them."
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache, so this runs once
    return value


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
    "PostMasteringSegmentEvidence",
    "PostMasteringVerificationResult",
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
    "compute_code_provenance",
    "compute_dependency_provenance",
    "compute_transition_mask",
    "load_safe_checkpoint",
    "load_speaker_profile",
    "merge_protected_spectrum",
    "save_safe_checkpoint",
    "validate_speaker_profile",
    "verify_protected_band_invariance",
]
