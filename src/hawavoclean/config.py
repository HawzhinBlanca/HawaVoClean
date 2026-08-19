"""Configuration schema, validation, and loading for HawaVoClean."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hawavoclean.errors import ConfigError
from hawavoclean.hashing import hash_json_canonical


class BaseFrozenModel(BaseModel):
    """Base model with strict forbidden extra fields and frozen immutability."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(BaseFrozenModel):
    """Runtime execution settings."""

    device: Literal["auto", "cuda", "cpu"] = "auto"
    isolated_worker: bool = True
    worker_timeout_s: float = Field(default=120.0, ge=5.0, le=600.0)
    worker_memory_limit_mb: int = Field(default=8192, ge=1024)
    development: bool = False
    num_threads: int = Field(default=1, ge=1, le=64)


class InputConfig(BaseFrozenModel):
    """Input validation and channel handling."""

    max_sample_rate: int = Field(default=48000, le=48000)
    supported_sample_rates: list[int] = Field(
        default_factory=lambda: [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    )
    channel_mode: Literal[
        "auto", "mono", "dual_mono_same", "split_speakers", "ambiguous_stereo"
    ] = "auto"
    output_bit_depth: Literal["pcm24", "float32"] = "pcm24"


class SegmentationConfig(BaseFrozenModel):
    """Utterance grouping and VAD."""

    vad_mode: Literal["silero", "energy"] = "energy"
    min_speech_ms: int = Field(default=150, ge=50, le=1000)
    target_speech_group_s: float = Field(default=16.0, ge=5.0, le=30.0)
    hard_max_group_s: float = Field(default=30.0, ge=10.0, le=60.0)
    pause_merge_threshold_ms: int = Field(default=250, ge=50, le=1000)
    context_duration_s: float = Field(default=1.0, ge=0.2, le=5.0)
    min_boundary_non_speech_ms: int = Field(default=200, ge=50, le=1000)


class EnhancementConfig(BaseFrozenModel):
    """Enhancement core configuration (deterministic DSP core)."""

    core_id: str = "wiener-dd-48k-v1"
    model_sample_rate: int = Field(default=48000, ge=8000, le=48000)
    phase_coherent: bool = True
    warmup_cycles: int = Field(default=1, ge=0, le=10)


class AlignmentConfig(BaseFrozenModel):
    """Delay and phase alignment configuration."""

    max_delay_ms: float = Field(default=50.0, ge=0.0, le=250.0)
    min_coherence: float = Field(default=0.70, ge=0.0, le=1.0)
    fractional_delay: bool = True


class GuardConfig(BaseFrozenModel):
    """Spectral fidelity guard parameters.

    The guard compares spectral signatures; it does not verify linguistic
    content. See hawavoclean.guard.spectral_probe.
    """

    guard_id: str = "spectral-guard-v1"
    probe_id: str = "spectral-signature-v1"
    calibration_file: str = "guard-calibration.json"
    # strict_spectral: output must stay spectrally near-identical (gentle
    #   cleanup; anchors and peak divergence enforced).
    # integrity: output may differ spectrally (restoration removes noise and
    #   reverb BY DESIGN); timing, envelope, artifact, and collapse
    #   protections stay enforced, spectral identity does not.
    mode: Literal["strict_spectral", "integrity"] = "strict_spectral"
    max_peak_js_div: float = Field(default=0.60, ge=0.0, le=2.0)
    min_anchor_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    max_posterior_js_div: float = Field(default=0.25, ge=0.0, le=1.0)
    max_timing_drift_ms: float = Field(default=40.0, ge=5.0, le=200.0)
    enforce_signal_integrity: bool = True
    spectral_hole_thresh: float = Field(default=0.10, ge=0.0, le=1.0)
    musical_noise_thresh: float = Field(default=0.20, ge=0.0, le=1.0)
    min_hf_preservation_ratio: float = Field(default=0.60, ge=0.0, le=1.0)


class PolicyConfig(BaseFrozenModel):
    """Enhancement selection and continuity policy."""

    strength_ladder: list[float] = Field(default_factory=lambda: [1.0, 0.75, 0.50, 0.25])
    clean_audio_bypass: bool = False
    clean_bypass_snr_db: float = Field(default=35.0, ge=20.0, le=60.0)
    enforce_continuity: bool = True

    @field_validator("strength_ladder")
    @classmethod
    def validate_strength_ladder(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Strength ladder must not be empty.")
        for s in v:
            if not 0.0 < s <= 1.0:
                raise ValueError(f"Strength value {s} must be in (0.0, 1.0].")
        return sorted(v, reverse=True)


class FinishingConfig(BaseFrozenModel):
    """Deterministic local finishing chain settings."""

    enabled: bool = True
    preset: Literal["gentle", "minimal", "bypass"] = "gentle"
    dc_subsonic_cutoff_hz: float = Field(default=20.0, ge=10.0, le=100.0)
    dehum_50_60hz: bool = True
    declick: bool = True
    plosive_attenuation: bool = True
    dynamic_eq: bool = True
    deess_band: bool = True
    max_deess_gr_db: float = Field(default=4.0, ge=0.0, le=12.0)
    level_rider: bool = True
    max_compression_gr_db: float = Field(default=3.0, ge=0.0, le=12.0)


class LoudnessConfig(BaseFrozenModel):
    """ITU-R BS.1770-4 loudness and peak limiting."""

    target_lufs_stereo: float = Field(default=-16.0, ge=-30.0, le=-10.0)
    target_lufs_mono: float = Field(default=-19.0, ge=-30.0, le=-10.0)
    true_peak_ceiling_dbtp: float = Field(default=-1.0, ge=-6.0, le=0.0)
    max_limiter_reduction_db: float = Field(default=2.5, ge=0.5, le=10.0)
    dither: bool = True


class ReportingConfig(BaseFrozenModel):
    """Audit report generation parameters."""

    schema_version: int = 1
    include_review_timecodes: bool = True


class HawaVoCleanConfig(BaseFrozenModel):
    """Master configuration root model."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    finishing: FinishingConfig = Field(default_factory=FinishingConfig)
    loudness: LoudnessConfig = Field(default_factory=LoudnessConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    def canonical_dict(self) -> dict[str, Any]:
        """Convert to canonical dictionary representation."""
        return self.model_dump(mode="json")

    def compute_hash(self) -> str:
        """Compute deterministic SHA-256 hash of the canonical configuration."""
        return hash_json_canonical(self.canonical_dict())


def load_config(path: Path | str | None = None, is_production: bool = True) -> HawaVoCleanConfig:
    """Load configuration from TOML file or return defaults with production constraints."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Configuration file not found: {p}")
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse TOML config from {p}: {e}") from e

    try:
        config = HawaVoCleanConfig.model_validate(data)
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e

    if is_production and config.runtime.development:
        raise ConfigError("Production mode refuses to run with runtime.development = true")

    return config
