"""Protocol and data definitions for the spectral fidelity probe.

The probe is NOT a speech recognizer. It maps audio to a deterministic
symbolic signature derived from short-term spectral shape, so that two
renderings of the same audio can be compared for spectral change. It has
no acoustic model, recognizes no phonemes, and cannot detect a linguistic
substitution that preserves spectral shape.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TokenInfo:
    """A sustained spectral state with timestamps and a stability score.

    'Tokens' here are symbols from an arbitrary alphabet assigned to
    spectral-shape classes — they are not recognized words or phonemes.
    """

    token_id: int
    text: str
    start_time_s: float
    end_time_s: float
    confidence: float


@dataclass
class ProbeResult:
    """Complete spectral-signature analysis of one waveform."""

    raw_signature: str
    normalized_signature: str
    tokens: list[TokenInfo] = field(default_factory=list)
    frame_distributions: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    frame_timestamps: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    mean_confidence: float = 1.0
    probe_id: str = "spectral-signature-v1"
    probe_hash: str = ""


class SpectralProbe(Protocol):
    """Protocol interface for fidelity probe backends."""

    @property
    def probe_id(self) -> str: ...

    @property
    def probe_hash(self) -> str: ...

    def infer(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> ProbeResult: ...
