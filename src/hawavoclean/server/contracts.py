"""Versioned public processing contracts.

Legacy ``POST /api/jobs`` remains available for one compatibility release.
New clients use these discriminated, fail-closed shapes so a misspelled or
unsupported strategy can never degrade into some other processing route.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def _reject_bool_number(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("Numeric field must not be a boolean")
    return value


def _require_bool_or_none(value: Any) -> Any:
    if value is not None and not isinstance(value, bool):
        raise ValueError("Boolean field must be a boolean")
    return value


def _validate_pro_repair_dict(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("proRepair must be an object mapping repair flag names to booleans")
    unknown = sorted(str(k) for k in value if k not in _ALLOWED_PRO_REPAIR_KEYS)
    if unknown:
        raise ValueError(f"Unsupported proRepair flag(s): {', '.join(unknown)}")
    for key, flag in value.items():
        if not isinstance(flag, bool):
            raise ValueError(f"proRepair[{key!r}] must be a boolean")
    return value


def _validate_time_range(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    start_s, end_s = value
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise ValueError("timeRange values must be finite numbers")
    if start_s < 0.0 or end_s <= start_s or end_s > 86400.0:
        raise ValueError("timeRange must satisfy 0 <= start < end <= 86400")
    return (float(start_s), float(end_s))


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


RestorePolicy = Literal["disabled", "source_allowed", "enrolled_only", "auto"]
DeliveryPresetV1 = Literal[
    "match_input", "ebu_r128", "atsc_a85", "dialogue_gated", "podcast", "streaming"
]
StereoModeV1 = Literal["auto", "dual_mono_same", "split_speakers", "true_stereo_ms"]
ExportFormatV1 = Literal["wav24", "wav32f", "flac", "aac", "video_mux"]
PreviewProfileV1 = Literal["production", "studio", "studio_pro", "dereverb", "lowband"]
SpectrogramModeV1 = Literal["original", "cleaned", "delta"]
SpectrogramScaleV1 = Literal["log", "mel", "linear"]
MAX_SPECTROGRAM_WINDOW_S = 300.0

_ALLOWED_PRO_REPAIR_KEYS: frozenset[str] = frozenset(
    {
        "declip",
        "deplosive",
        "declick",
        "mouthDeclick",
        "mouth_declick",
        "deess",
        "tonalRestoration",
        "tonal_restoration",
        "levelRider",
        "level_rider",
    }
)
ManualRoute = Literal[
    "production",
    "studio",
    "studio_pro",
    "dereverb",
    "lowband",
    "lowband_then_production",
    "restore_source",
    "restore_enrolled",
]
JobLifecycleStateV1 = Literal[
    "queued",
    "analyzing",
    "rendering",
    "guarding",
    "publishing",
    "completed",
    "cancelled",
    "interrupted",
    "failed",
]


class SmartSafeStrategyV1(ContractModel):
    kind: Literal["smart_safe"]
    restore_policy: RestorePolicy = "disabled"
    speaker_profile_id: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    allow_generative_reconstruction: bool = False
    hpf_enabled: bool | None = None
    hpf_cutoff_hz: float | None = Field(default=None, ge=20.0, le=250.0, allow_inf_nan=False)
    ambience_floor_db: float | None = Field(default=None, ge=-120.0, le=-12.0, allow_inf_nan=False)
    delivery_preset: DeliveryPresetV1 | None = None
    stereo_mode: StereoModeV1 | None = None
    pro_repair: dict[str, bool] | None = None

    @field_validator("hpf_enabled", mode="before")
    @classmethod
    def _validate_hpf_enabled(cls, value: Any) -> Any:
        return _require_bool_or_none(value)

    @field_validator("hpf_cutoff_hz", "ambience_floor_db", mode="before")
    @classmethod
    def _validate_numeric_not_bool(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @field_validator("pro_repair", mode="before")
    @classmethod
    def _validate_pro_repair(cls, value: Any) -> Any:
        return _validate_pro_repair_dict(value)

    @model_validator(mode="after")
    def validate_restore_consent(self) -> SmartSafeStrategyV1:
        if self.restore_policy != "disabled" and not self.allow_generative_reconstruction:
            raise ValueError("Smart Restore requires explicit generative reconstruction consent")
        if self.restore_policy == "disabled" and self.speaker_profile_id is not None:
            raise ValueError("speakerProfileId is invalid when Restore is disabled")
        if self.restore_policy == "enrolled_only" and self.speaker_profile_id is None:
            raise ValueError("enrolled_only requires speakerProfileId")
        return self


class ManualStrategyV1(ContractModel):
    kind: Literal["manual"]
    route: ManualRoute
    speaker_profile_id: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    expert_cutoff_hz: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    allow_generative_reconstruction: bool = False
    strength: float | None = Field(default=None, ge=0.1, le=1.0, allow_inf_nan=False)
    time_range: tuple[float, float] | None = None
    ambience_floor_db: float | None = Field(default=None, ge=-120.0, le=-12.0, allow_inf_nan=False)
    delivery_preset: DeliveryPresetV1 | None = None
    stereo_mode: StereoModeV1 | None = None
    pro_repair: dict[str, bool] | None = None
    hpf_enabled: bool | None = None
    hpf_cutoff_hz: float | None = Field(default=None, ge=20.0, le=250.0, allow_inf_nan=False)

    @field_validator("hpf_enabled", mode="before")
    @classmethod
    def _validate_hpf_enabled(cls, value: Any) -> Any:
        return _require_bool_or_none(value)

    @field_validator("strength", "ambience_floor_db", "hpf_cutoff_hz", mode="before")
    @classmethod
    def _validate_numeric_not_bool(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @field_validator("pro_repair", mode="before")
    @classmethod
    def _validate_pro_repair(cls, value: Any) -> Any:
        return _validate_pro_repair_dict(value)

    @field_validator("time_range", mode="before")
    @classmethod
    def _reject_bool_time_range(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and any(isinstance(item, bool) for item in value):
            raise ValueError("timeRange values must be numbers, not booleans")
        return value

    @field_validator("time_range")
    @classmethod
    def _check_time_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        return _validate_time_range(value)

    @model_validator(mode="after")
    def validate_route(self) -> ManualStrategyV1:
        is_restore = self.route in {"restore_source", "restore_enrolled"}
        if is_restore and not self.allow_generative_reconstruction:
            raise ValueError("Restore routes require explicit generative reconstruction consent")
        if not is_restore and self.allow_generative_reconstruction:
            raise ValueError("generative reconstruction consent is invalid for a Natural route")
        if self.route == "restore_enrolled" and self.speaker_profile_id is None:
            raise ValueError("restore_enrolled requires speakerProfileId")
        if self.route != "restore_enrolled" and self.speaker_profile_id is not None:
            raise ValueError("speakerProfileId is valid only for restore_enrolled")
        if not is_restore and self.expert_cutoff_hz is not None:
            raise ValueError("expertCutoffHz is valid only for Restore routes")
        return self


class SeparationStrategyV1(ContractModel):
    """Explicit, experimental sound separation; never an automatic fallback."""

    kind: Literal["separate"]
    operation: Literal["extract", "remove"]
    prompt: str = Field(min_length=1, max_length=512)
    model_size: Literal["small", "base", "large"] = "large"
    quality: Literal["fast", "balanced", "high"] = "high"
    seed: int = Field(default=0, ge=0, le=2**31 - 1, strict=True)
    cleanup_profile: Literal["production", "studio", "studio_pro", "dereverb", "lowband"] | None = (
        None
    )
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    strength: float = Field(default=1.0, ge=0.1, le=1.0, allow_inf_nan=False)
    time_range: tuple[float, float] | None = None

    @field_validator("strength", mode="before")
    @classmethod
    def _validate_strength_type(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @field_validator("time_range", mode="before")
    @classmethod
    def _reject_bool_time_range(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and any(isinstance(item, bool) for item in value):
            raise ValueError("timeRange values must be numbers, not booleans")
        return value

    @field_validator("time_range")
    @classmethod
    def _check_time_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        return _validate_time_range(value)

    @model_validator(mode="after")
    def validate_prompt(self) -> SeparationStrategyV1:
        if not self.prompt.strip() or any(ord(c) < 32 for c in self.prompt):
            raise ValueError("Describe a sound using printable text")
        return self


ProcessingStrategyV1 = Annotated[
    SmartSafeStrategyV1 | ManualStrategyV1 | SeparationStrategyV1,
    Field(discriminator="kind"),
]


class ProcessingRequestV1(ContractModel):
    schema_version: Literal[1]
    source_ids: list[str] = Field(min_length=1, max_length=100)
    strategy: ProcessingStrategyV1
    execution_policy: Literal["offline_only", "prefer_offline", "cloud_allowed"]
    cloud_consent_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[\x21-\x7e]+$"
    )
    conflict_policy: Literal["unique", "fail", "replace"] = "unique"
    record_bundle: bool = False
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[\x21-\x7e]+$")

    @model_validator(mode="after")
    def validate_execution(self) -> ProcessingRequestV1:
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("sourceIds must not contain duplicates")
        if any(not source_id or len(source_id) > 128 for source_id in self.source_ids):
            raise ValueError("every sourceId must contain 1-128 characters")
        if self.execution_policy == "cloud_allowed" and self.cloud_consent_id is None:
            raise ValueError("cloud_allowed requires a per-request cloudConsentId")
        if self.execution_policy != "cloud_allowed" and self.cloud_consent_id is not None:
            raise ValueError("cloudConsentId is valid only with cloud_allowed")
        return self


class CapabilityStatusV1(ContractModel):
    capability_id: str
    available: bool
    maturity: Literal["qualified", "experimental", "blocked"]
    reason: str | None = None
    manifest_sha256: str | None = None
    providers: list[str] = Field(default_factory=list)


class CapabilitiesResponseV1(ContractModel):
    schema_version: Literal[1] = 1
    capabilities: list[CapabilityStatusV1]


class SmartAnalysisRequestV1(ContractModel):
    """Analyze one engine-managed source without accepting a client path."""

    schema_version: Literal[1]
    source_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class ProbabilityEstimateV1(ContractModel):
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    conservative: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    direction: Literal["lower", "upper"]
    rationale: str


class SmartAnalysisResponseV1(ContractModel):
    """Closed wire shape for the unqualified streaming analyzer."""

    schema_version: Literal[1] = 1
    qualification: Literal["experimental_unqualified"]
    valid: bool
    sample_rate: int | None
    channels: int | None
    samples: int = Field(ge=0)
    analyzed_frames: int = Field(ge=0)
    duration_s: float = Field(ge=0.0, allow_inf_nan=False)
    speech_dominance: ProbabilityEstimateV1
    music_risk: ProbabilityEstimateV1
    crosstalk_risk: ProbabilityEstimateV1
    band_limited_confidence: ProbabilityEstimateV1
    estimated_cutoff_hz: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    cutoff_confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    bandwidth_shape: Literal["unknown", "silence", "fullband", "steep_lowpass"]
    noise_risk: ProbabilityEstimateV1
    hum_confidence: ProbabilityEstimateV1
    reverberation_risk: ProbabilityEstimateV1
    clipping_risk: ProbabilityEstimateV1
    clipping_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    codec_damage_risk: ProbabilityEstimateV1
    channel_coherence: ProbabilityEstimateV1
    rumble_confidence: ProbabilityEstimateV1
    recorded_high_frequency_speech_confidence: ProbabilityEstimateV1
    uncertainty: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    uncertainty_reasons: list[str]
    state_bound_bytes: int = Field(ge=0)


class ProcessingRecordEvidenceV1(ContractModel):
    """Verification evidence for the portable Full Processing Record ZIP."""

    path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    master_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_uncompressed_bytes: int = Field(ge=1)
    internal_hashes_verified: Literal[True]
    authenticated_publisher: bool
    key_id: str | None = None
    signature_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class JobStatusResponseV1(ContractModel):
    """Versioned lifecycle independent of the legacy UI's running/done names."""

    schema_version: Literal[1] = 1
    job_id: str
    state: JobLifecycleStateV1
    stage: str
    progress: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    message: str
    output_path: str
    report_path: str
    record_bundle: bool = False
    bundle_path: str | None = None
    bundle: ProcessingRecordEvidenceV1 | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: dict[str, str] | None = None
    report: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_bundle_state(self) -> JobStatusResponseV1:
        if self.record_bundle and self.bundle_path is None:
            raise ValueError("recordBundle jobs require bundlePath")
        if not self.record_bundle and (self.bundle_path is not None or self.bundle is not None):
            raise ValueError("bundle evidence is invalid when recordBundle is false")
        if self.bundle is not None and self.bundle.path != self.bundle_path:
            raise ValueError("bundle evidence path must equal bundlePath")
        if self.state == "completed" and self.record_bundle and self.bundle is None:
            raise ValueError("completed recordBundle jobs require verified bundle evidence")
        return self


class UnitOverrideRequestV1(ContractModel):
    """Request to manually override a guard decision on a single unit."""

    unit_index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based unit index in the report's units list",
    )
    unit_id: int | None = Field(
        default=None,
        ge=0,
        description="Unit ID in the report's units list",
    )
    decision: Literal["force_original", "force_enhanced", "auto", "custom_strength"] = Field(
        description="Override the guard decision for this unit"
    )
    strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Custom wet/dry strength in [0.0, 1.0] when decision is 'custom_strength'",
    )

    @field_validator("unit_index", "unit_id", "strength", mode="before")
    @classmethod
    def _reject_bool_fields(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @model_validator(mode="after")
    def validate_override(self) -> UnitOverrideRequestV1:
        if self.unit_index is None and self.unit_id is None:
            raise ValueError("unitIndex or unitId is required")
        if self.decision == "custom_strength" and self.strength is None:
            raise ValueError("strength is required when decision is 'custom_strength'")
        return self


class RegionPreviewRequestV1(ContractModel):
    """Request for a sub-second warm-worker windowed enhancement preview."""

    path: str = Field(min_length=1, max_length=32_768)
    start_s: float = Field(ge=0.0, allow_inf_nan=False)
    end_s: float = Field(gt=0.0, allow_inf_nan=False)
    profile: PreviewProfileV1 = "production"
    strength: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)
    ambience_floor_db: float | None = Field(default=None, ge=-120.0, le=-12.0, allow_inf_nan=False)
    include_perceptual_mos: bool = False
    hpf_enabled: bool | None = None
    hpf_cutoff_hz: float | None = Field(default=None, ge=20.0, le=250.0, allow_inf_nan=False)
    pro_repair: dict[str, bool] | None = None
    stereo_mode: StereoModeV1 | None = None
    delivery_preset: DeliveryPresetV1 | None = None

    @field_validator("hpf_enabled", mode="before")
    @classmethod
    def _validate_hpf_enabled(cls, value: Any) -> Any:
        return _require_bool_or_none(value)

    @field_validator(
        "start_s", "end_s", "strength", "ambience_floor_db", "hpf_cutoff_hz", mode="before"
    )
    @classmethod
    def _reject_bool_fields(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @field_validator("pro_repair", mode="before")
    @classmethod
    def _validate_pro_repair(cls, value: Any) -> Any:
        return _validate_pro_repair_dict(value)

    @model_validator(mode="after")
    def validate_window(self) -> RegionPreviewRequestV1:
        if self.end_s <= self.start_s:
            raise ValueError("endS must be greater than startS")
        return self


class SpectrogramRequestV1(ContractModel):
    """Request for a 2D STFT or delta spectrogram tile."""

    path: str = Field(min_length=1, max_length=32_768)
    cleaned_path: str | None = Field(default=None, min_length=1, max_length=32_768)
    mode: SpectrogramModeV1 = "original"
    start_s: float = Field(ge=0.0, allow_inf_nan=False)
    end_s: float = Field(gt=0.0, allow_inf_nan=False)
    width_cols: int = Field(default=512, ge=1, le=4096)
    n_bins: int = Field(default=128, ge=1, le=1024)
    n_fft: int = Field(default=2048, ge=64, le=16384)
    scale: SpectrogramScaleV1 = "log"
    db_floor: float | None = Field(default=None, allow_inf_nan=False)
    db_ceil: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator(
        "start_s",
        "end_s",
        "width_cols",
        "n_bins",
        "n_fft",
        "db_floor",
        "db_ceil",
        mode="before",
    )
    @classmethod
    def _reject_bool_fields(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @model_validator(mode="after")
    def validate_spectrogram(self) -> SpectrogramRequestV1:
        if self.end_s <= self.start_s:
            raise ValueError("endS must be greater than startS")
        if self.end_s - self.start_s > MAX_SPECTROGRAM_WINDOW_S:
            raise ValueError(
                f"Spectrogram window duration ({self.end_s - self.start_s:.2f}s) exceeds maximum "
                f"{MAX_SPECTROGRAM_WINDOW_S:.1f}s"
            )
        if self.mode == "delta" and self.cleaned_path is None:
            raise ValueError("cleanedPath is required when mode is 'delta'")
        if self.db_floor is not None and self.db_ceil is not None and self.db_floor >= self.db_ceil:
            raise ValueError("dbFloor must be less than dbCeil")
        return self


class SpectralRepairRequestV1(ContractModel):
    """Request for 2D time×frequency surgical spectral box attenuation."""

    path: str = Field(min_length=1, max_length=32_768)
    start_s: float = Field(ge=0.0, allow_inf_nan=False)
    end_s: float = Field(gt=0.0, allow_inf_nan=False)
    f_low_hz: float = Field(ge=0.0, allow_inf_nan=False)
    f_high_hz: float = Field(gt=0.0, allow_inf_nan=False)
    attenuation_db: float = Field(default=24.0, ge=3.0, le=60.0, allow_inf_nan=False)

    @field_validator(
        "start_s",
        "end_s",
        "f_low_hz",
        "f_high_hz",
        "attenuation_db",
        mode="before",
    )
    @classmethod
    def _reject_bool_fields(cls, value: Any) -> Any:
        return _reject_bool_number(value)

    @model_validator(mode="after")
    def validate_box(self) -> SpectralRepairRequestV1:
        if self.end_s <= self.start_s:
            raise ValueError("endS must be greater than startS")
        if self.f_high_hz <= self.f_low_hz:
            raise ValueError("fHighHz must be greater than fLowHz")
        return self


class DeliveryExportRequestV1(ContractModel):
    """Request to export a master WAV with optional broadcast loudness and metadata preservation."""

    master_path: str = Field(min_length=1, max_length=32_768)
    dest_path: str | None = Field(default=None, min_length=1, max_length=32_768)
    format: ExportFormatV1 = "wav24"
    delivery_preset: DeliveryPresetV1 | None = None
    source_media_path: str | None = Field(default=None, min_length=1, max_length=32_768)
    preserve_metadata: bool = True


class TrackInspectRequestV1(ContractModel):
    """Request to inspect tracks inside a multi-channel Poly-WAV/BWF file."""

    path: str = Field(min_length=1, max_length=32_768)


class TrackExtractRequestV1(ContractModel):
    """Request to extract selected 0-based tracks from a Poly-WAV/BWF file."""

    path: str = Field(min_length=1, max_length=32_768)
    track_indices: list[int] = Field(min_length=1, max_length=16)
    out_dir: str | None = Field(default=None, min_length=1, max_length=32_768)

    @field_validator("track_indices", mode="before")
    @classmethod
    def _validate_track_indices(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ValueError("trackIndices must contain non-negative integers")
        return value


__all__ = [
    "CapabilitiesResponseV1",
    "CapabilityStatusV1",
    "DeliveryExportRequestV1",
    "DeliveryPresetV1",
    "ExportFormatV1",
    "JobLifecycleStateV1",
    "JobStatusResponseV1",
    "MAX_SPECTROGRAM_WINDOW_S",
    "ManualStrategyV1",
    "PreviewProfileV1",
    "ProcessingRecordEvidenceV1",
    "ProcessingRequestV1",
    "ProcessingStrategyV1",
    "ProbabilityEstimateV1",
    "RegionPreviewRequestV1",
    "SeparationStrategyV1",
    "SmartAnalysisRequestV1",
    "SmartAnalysisResponseV1",
    "SmartSafeStrategyV1",
    "SpectralRepairRequestV1",
    "SpectrogramModeV1",
    "SpectrogramRequestV1",
    "SpectrogramScaleV1",
    "TrackExtractRequestV1",
    "TrackInspectRequestV1",
    "UnitOverrideRequestV1",
]
