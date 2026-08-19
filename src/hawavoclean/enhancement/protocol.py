"""Protocol and data models for deterministic enhancement cores.

HawaVoClean v1 ships no learned models. Every core is classical DSP with
explicit parameters; provenance is the parameter set itself, hashed
canonically, never a weights digest.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EnhancerMetadata:
    """Identification and provenance metadata for an enhancement core."""

    core_id: str
    version: str
    algorithm: str
    sample_rate: int
    phase_coherent: bool
    params_hash: str = ""


@dataclass
class EnhancementResult:
    """Inference output from an enhancement core."""

    waveform: np.ndarray[Any, np.dtype[np.float32]]
    sample_rate: int
    model_runtime_ms: float
    input_samples: int
    output_samples: int
    warnings: list[str] = field(default_factory=list)


class Enhancer(Protocol):
    """Protocol interface for enhancement cores."""

    def __init__(
        self,
        core_id: str = ...,
        sample_rate: int = ...,
        phase_coherent: bool = ...,
    ) -> None: ...

    @property
    def metadata(self) -> EnhancerMetadata: ...

    def warmup(self) -> None: ...

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult: ...
