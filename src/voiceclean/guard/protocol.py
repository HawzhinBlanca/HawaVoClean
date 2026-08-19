"""Protocol and data definitions for Sorani ASR and CTC acoustic analysis."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TokenInfo:
    """Individual recognized token with timestamps and calibrated confidence."""

    token_id: int
    text: str
    start_time_s: float
    end_time_s: float
    confidence: float


@dataclass
class ASRResult:
    """Complete acoustic and textual inference result from Sorani ASR."""

    raw_transcript: str
    normalized_transcript: str
    tokens: list[TokenInfo] = field(default_factory=list)
    frame_posteriors: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    frame_timestamps: np.ndarray[Any, np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    mean_confidence: float = 1.0
    model_id: str = "hawzhin-sorani-asr-v1"
    model_hash: str = ""


class SoraniASR(Protocol):
    """Protocol interface for Sorani ASR backends as specified in BLUEPRINT.md section 13.1."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_hash(self) -> str: ...

    def infer(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> ASRResult: ...
