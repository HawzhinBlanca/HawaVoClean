"""Audit report models and serialization for spectral restoration mode."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RestorationSegmentCounts:
    """Breakdown of segment-level restoration actions."""

    restored: int
    reduced: int
    reverted: int
    bypassed: int
    errors: int


@dataclass(frozen=True)
class RestorationReport:
    """Restoration-specific section of the immutable HawaVoClean report."""

    mode: str
    speaker_id: str | None
    profile_hash: str | None
    natural_output_hash: str | None
    bandwidth: dict[str, Any]
    restorer: dict[str, Any]
    segments: RestorationSegmentCounts
    guard_r: dict[str, Any]
    review_timecodes: list[dict[str, Any]]
    post_mastering_verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return asdict(self)
