"""Protocol and data models for neural audio enhancement cores."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EnhancerMetadata:
    """Identification and provenance metadata for an enhancement core."""

    core_id: str
    version: str
    sample_rate: int
    phase_coherent: bool
    commit: str = ""
    weights_sha256: str = ""


@dataclass
class EnhancementResult:
    """Inference output from an enhancement core."""

    waveform: np.ndarray[Any, np.dtype[np.float32]]
    sample_rate: int
    model_runtime_ms: float
    input_samples: int
    output_samples: int
    peak_vram_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


class Enhancer(Protocol):
    """Protocol interface for frozen neural enhancement cores as defined in BLUEPRINT.md section 11.1."""

    @property
    def metadata(self) -> EnhancerMetadata: ...

    def warmup(self) -> None: ...

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult: ...
