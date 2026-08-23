"""Pydantic v2 schemas for audit reports, unit decisions, and corpus manifests."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hawavoclean.provenance import (
    ProvenanceError,
    current_build_report_fields,
    verify_report_build,
)
from hawavoclean.release import RELEASE_IDENTITY, REPORT_SCHEMA_VERSION
from hawavoclean.runtime import active_device


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
    """Enhancement core identification. The core is deterministic DSP; its
    provenance is its parameter set, hashed canonically."""

    id: str
    algorithm: str
    params_hash: str
    phase_coherent: bool = True
    lock_sha256: str | None = None
    weight_sha256: dict[str, str] = Field(default_factory=dict)


class GuardMetadata(ReportBaseModel):
    """Guard identification and calibration artifact hashes.

    The guard compares spectral signatures; it does not verify linguistic
    content."""

    id: str
    probe_hash: str
    calibration_id: str
    calibration_sha256: str | None = None


class EnvironmentMetadata(ReportBaseModel):
    """Hardware and runtime stack metadata for audit repeatability."""

    platform: str
    os_version: str
    python_version: str
    numpy_version: str
    scipy_version: str
    soundfile_version: str
    cpu_model: str | None = None
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    deterministic_settings: dict[str, str | int | bool] = Field(default_factory=dict)
    #: The compute device the enhancement core actually ran on. A GPU does not
    #: compute the same samples as the CPU, so a result that did not say which
    #: one produced it could be attributed to the wrong compute path — and two
    #: runs of the same config on two machines can legitimately differ here
    #: while sharing a config hash. Defaults from the live process (see
    #: :func:`hawavoclean.runtime.active_device`) exactly as the platform and
    #: library versions above do; a classical-DSP core always reports ``cpu``
    #: whatever ``runtime.device`` asked for, because that is what ran.
    compute_device: str = Field(default_factory=active_device)


class BuildMetadata(ReportBaseModel):
    """Exact source, dependency lock and installed-distribution identity."""

    provenance_schema_version: Literal[1]
    artifact_type: Literal["wheel", "sdist", "source-tree"]
    source_revision: str
    source_date_epoch: int
    source_dirty: bool
    dependency_lock_sha256: str
    release_identity_sha256: str
    build_id: str
    distribution_record_sha256: str | None = None

    @model_validator(mode="after")
    def identity_is_self_verifying(self) -> "BuildMetadata":
        try:
            verify_report_build(self.model_dump())
        except ProvenanceError as exc:
            raise ValueError(str(exc)) from exc
        return self


class ReleaseMetadata(ReportBaseModel):
    """Release identity copied from, and checked against, packaged bytes."""

    product: str
    version: str
    report_schema_version: int
    identity_sha256: str

    @model_validator(mode="after")
    def matches_packaged_release(self) -> "ReleaseMetadata":
        if self.model_dump() != RELEASE_IDENTITY.report_fields():
            raise ValueError("report release identity does not match the packaged release")
        return self


def current_release_metadata() -> ReleaseMetadata:
    """Construct the only release identity valid for a newly emitted report."""
    return ReleaseMetadata(**RELEASE_IDENTITY.report_fields())


def current_build_metadata() -> BuildMetadata:
    """Construct provenance for the code and distribution running now."""
    return BuildMetadata(**current_build_report_fields())


class UnitSummary(ReportBaseModel):
    """Aggregate statistics of all processed units."""

    units_total: int = 0
    enhanced: int = 0
    reverted: int = 0
    unverified: int = 0
    error_passthrough: int = 0
    continuity_reverted: int = 0
    #: Enhanced units that kept their enhancement across a forced mid-speech
    #: cut by fading back to the original recording at the joint, instead of
    #: being reverted whole. See :mod:`hawavoclean.policy.continuity`.
    continuity_crossfaded: int = 0
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


class PassRecord(ReportBaseModel):
    """One pass of a multi-pass run: what went in, what came out, what the
    guard did, and — when auto mode discarded the pass — why it did not ship.

    ``input_sha256``/``output_sha256`` chain pass to pass (pass k's input is
    pass k-1's output), so the journey from source to shipped master is
    auditable even though only the final pass's unit records are the report's
    ``units``.
    """

    pass_index: int
    input_sha256: str
    output_sha256: str
    units_total: int = 0
    enhanced: int = 0
    reverted: int = 0
    chosen_strengths: list[float] = Field(default_factory=list)
    separation_db: float
    integrated_lufs: float | None = None
    discarded: bool = False
    discard_reason: str | None = None


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
    output_sha256: str
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


class HawaVoCleanReport(ReportBaseModel):
    """Master immutable JSON audit report format as defined in BLUEPRINT.md section 18.1."""

    schema_version: Literal[1, 2] = REPORT_SCHEMA_VERSION
    release: ReleaseMetadata | None = None
    build: BuildMetadata | None = None
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
    #: Multi-pass audit trail (empty for the ordinary single-pass run, so
    #: every pre-existing report, test, and consumer is untouched). The
    #: report's ``units`` are always the FINAL (shipped) pass's records;
    #: this list is the journey, including any auto-discarded pass.
    passes: list[PassRecord] = Field(default_factory=list)
    restoration: dict[str, Any] | None = None

    @model_validator(mode="after")
    def release_matches_schema(self) -> "HawaVoCleanReport":
        if self.schema_version == 1:
            if self.release is not None:
                raise ValueError("schema-v1 reports cannot claim schema-v2 release identity")
            if self.build is not None:
                raise ValueError("schema-v1 reports cannot claim schema-v2 build identity")
            return self
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError("report schema does not match the packaged release identity")
        if self.release is None:
            raise ValueError("schema-v2 reports require release identity")
        if self.release.report_schema_version != self.schema_version:
            raise ValueError("report schema and embedded release identity disagree")
        if self.build is None:
            raise ValueError("schema-v2 reports require build provenance")
        if self.core.lock_sha256 is None:
            raise ValueError("schema-v2 reports require the selected core lock digest")
        if self.guard.calibration_sha256 is None:
            raise ValueError("schema-v2 reports require the guard calibration digest")
        required_versions = {
            "hawavoclean",
            "numpy",
            "scipy",
            "soundfile",
            "pyloudnorm",
            "pydantic",
            "libsndfile",
            "ffmpeg",
            "ffprobe",
        }
        missing_versions = required_versions - self.environment.runtime_versions.keys()
        if missing_versions:
            raise ValueError(
                "schema-v2 reports are missing runtime versions: "
                + ", ".join(sorted(missing_versions))
            )
        settings = self.environment.deterministic_settings
        required_settings = {
            "compute_device",
            "requested_device",
            "worker_pool_size",
            "threads_per_worker",
            "omp_num_threads",
            "mkl_num_threads",
            "python_hash_seed",
            "output_bit_depth",
            "tpdf_dither",
            "dither_seed_derivation",
            "result_order",
            "torch_deterministic_algorithms",
        }
        missing_settings = required_settings - settings.keys()
        if missing_settings:
            raise ValueError(
                "schema-v2 reports are missing deterministic settings: "
                + ", ".join(sorted(missing_settings))
            )
        if settings.get("compute_device") != self.environment.compute_device:
            raise ValueError("report device disagrees with deterministic settings")
        return self


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
    verified_by_human: bool = False
    split: Literal["calibration", "development", "acceptance", "corruption"]


class CorpusManifest(ReportBaseModel):
    """Collection manifest for dataset splits."""

    schema_version: int = 1
    manifest_id: str
    split_name: str
    items_count: int
    manifest_sha256: str = ""
    items: list[CorpusItem] = Field(default_factory=list)
