"""Candidate runner interface for empirical model evaluation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class CandidateResult:
    """Candidate model execution result according to BLUEPRINT.md section 5.2."""

    candidate_name: str
    version: str
    repo_url: str
    commit: str
    weight_sha256: dict[str, str]
    code_license: str
    weight_license: str
    input_sample_rate: int
    output_sample_rate: int
    input_samples: int
    output_samples: int
    runtime_ms: float
    peak_vram_bytes: int
    phase_coherent: bool
    warnings: list[str] = field(default_factory=list)


class CandidateRunner(Protocol):
    """Opaque executable runner contract."""

    def run(self, input_wav: Path, output_wav: Path) -> CandidateResult: ...
