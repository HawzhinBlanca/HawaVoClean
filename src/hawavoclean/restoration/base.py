"""Base interfaces and protocols for HawaVoClean spectral restoration."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RestorationCandidate:
    """A generated restoration candidate at a specific high-band residual strength."""

    strength: float
    audio: np.ndarray  # Shape: (channels, samples) or (samples,), float32, 48 kHz
    cutoff_hz: float
    protected_band_error: float = 0.0


@runtime_checkable
class Restorer(Protocol):
    """Protocol implemented by bandwidth restoration backends."""

    def restore(
        self,
        audio_48k: np.ndarray,
        sample_rate: int,
        effective_cutoff_hz: float,
        speaker_id: str | None = None,
        speaker_embedding: np.ndarray | None = None,
        f0_trajectory: np.ndarray | None = None,
        vuv_mask: np.ndarray | None = None,
        strengths: list[float] | None = None,
        seed: int = 42,
    ) -> list[RestorationCandidate]:
        """Generate candidates for candidate high-band strengths from strongest to weakest."""
        ...
