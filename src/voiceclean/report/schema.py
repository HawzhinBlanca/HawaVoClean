"""Pydantic v2 schemas for audit reports, unit decisions, and corpus manifests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ReportBaseModel(BaseModel):
    """Base model with strict forbidden extra fields and JSON serialization support."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MediaStats(ReportBaseModel):
    """File metadata and audio statistics."""

    path: str
    sha256: str
    sample_rate: int
    channels: int
    samples: int
    duration_s: float
    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None


class CoreMetadata(ReportBaseModel):
    """Neural enhancement model identification and lock hashes."""

    id: str
    commit: str
    weight_sha256: dict[str, str] = Field(default_factory=dict)
    phase_coherent: bool = True


class GuardMetadata(ReportBaseModel):
    """Guard identification and calibration artifact hashes."""

    id: str
    model_sha256: str
    calibration_id: str


class EnvironmentMetadata(ReportBaseModel):
    """Hardware and runtime stack metadata for audit repeatability."""

    platform: str
    os_version: str
    python_version: str
    torch_version: str
    cuda_version: str | None = None
    gpu_name: str | None = None
    driver_version: str | None = None
    cpu_model: str | None = None


class UnitSummary(ReportBaseModel):
    """Aggregate statistics of all processed units."""

    units_total: int = 0
    enhanced: int = 0
    reverted: int = 0
    unverified: int = 0
    error_passthrough: int = 0
    no_speech: int = 0
    finish_applied: int = 0
    finish_bypassed: int = 0


class ReviewTimecode(ReportBaseModel):
    """Timecode flagged for human listening review."""

    unit_id: int
    start_time_s: float
    end_time_s: float
    channel: int
    verdict: str
    reason: str


class UnitDecisionRecord(ReportBaseModel):
    """Full forensic audit trail for a single speech unit."""

    unit_id: int
    channel: int
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    is_speech: bool
    input_sha256: str
    candidate_sha256: str | None = None
    output_sha256: str = ""
    guard_a_verdict: str
    guard_a_scores: dict[str, float | str | bool] = Field(default_factory=dict)
    guard_b_verdict: str | None = None
    guard_b_scores: dict[str, float | str | bool] = Field(default_factory=dict)
    chosen_strength: float = 0.0
    finish_preset_applied: str = "bypass"
    finish_actions: list[str] = Field(default_factory=list)
    final_decision: str
    decision_reason: str = ""
    runtime_ms: float = 0.0


class VoiceCleanReport(ReportBaseModel):
    """Master immutable JSON audit report format as defined in BLUEPRINT.md section 18.1."""

    schema_version: int = 1
    job_id: str
    config_hash: str
    input: MediaStats
    output: MediaStats
    core: CoreMetadata
    guard: GuardMetadata
    environment: EnvironmentMetadata
    summary: UnitSummary
    review_timecodes: list[ReviewTimecode] = Field(default_factory=list)
    units: list[UnitDecisionRecord] = Field(default_factory=list)


class CorpusItem(ReportBaseModel):
    """Metadata schema for dataset manifests in calibration, benchmark, and acceptance sets."""

    id: str
    audio_path: str
    audio_sha256: str
    duration_s: float
    speaker_id: str
    dialect: str  # "slemani", "erbil", "general"
    gender: Literal["male", "female", "unknown"]
    environment: str  # "studio", "untreated", "street", "fan_noise", "reverb"
    degradation_type: str  # "clean", "noise", "reverb", "clipping", "codec", "consonant_cut", etc.
    transcript_sorani: str
    verified_by_human: bool = True
    split: Literal["calibration", "development", "acceptance", "corruption"]


class CorpusManifest(ReportBaseModel):
    """Collection manifest for dataset splits."""

    schema_version: int = 1
    manifest_id: str
    split_name: str
    items_count: int
    manifest_sha256: str = ""
    items: list[CorpusItem] = Field(default_factory=list)
