"""Configuration schema, validation, and loading for HawaVoClean."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hawavoclean.errors import ConfigError
from hawavoclean.hashing import hash_json_canonical
from hawavoclean.runtime import activate_runtime, threads_per_worker, worker_pool_size


class BaseFrozenModel(BaseModel):
    """Base model with strict forbidden extra fields and frozen immutability."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(BaseFrozenModel):
    """Runtime execution settings.

    Every key here is enforced. ``device`` and ``num_threads`` are applied by
    :func:`hawavoclean.runtime.activate_runtime`, which :func:`load_config`
    calls; ``isolated_worker``, ``worker_timeout_s`` and ``development`` are
    read by the pipeline and by :func:`load_config` respectively.

    A note on field *names*: this schema is hashed into ``config_hash``, which
    is hashed into the job id, which seeds the master's deterministic dither.
    Adding, removing, or renaming a key here therefore changes the last bits
    of every published sample. Keys are made to mean something rather than
    deleted for that reason; the deletion of a genuinely dead key is a
    deliberate act that reissues the reference hashes.
    """

    #: Compute device for the enhancement core. ``auto`` resolves through
    #: :data:`hawavoclean.runtime.AUTO_DEVICE_PREFERENCE` (today: cpu only —
    #: a GPU backend changes the samples a run produces, so it is opt-in).
    #: An explicit device this machine cannot provide is a hard error, not a
    #: silent fallback. The device that actually ran is recorded in the
    #: report as ``environment.compute_device``.
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    isolated_worker: bool = True
    worker_timeout_s: float = Field(default=120.0, ge=5.0, le=600.0)
    #: Resident-set budget for the process that runs the enhancement core.
    #: Enforced by the core itself (:func:`hawavoclean.runtime.check_memory_budget`),
    #: checked before it accepts each unit: a worker that has already blown
    #: the budget refuses further work, the parent recycles it, and the unit
    #: it refused falls back to ORIGINAL audio — the same fail-closed path a
    #: crash or a timeout takes. Self-policing rather than ``RLIMIT_AS``
    #: because every memory rlimit is unsettable on this project's reference
    #: platform (macOS reports ``RLIM_INFINITY`` and rejects the setrlimit),
    #: and where it can be set it caps reserved *virtual* address space, which
    #: torch reserves in gigabytes without touching — killing healthy runs
    #: rather than runaway ones.
    worker_memory_limit_mb: int = Field(default=8192, ge=1024)
    development: bool = False
    #: Size of the enhancement worker pool: how many speech units may be
    #: enhanced concurrently. NOT an intra-op/BLAS thread count — conflating
    #: the two oversubscribes the CPU with ``num_threads`` squared threads.
    #: When it asks for more than one worker, each worker is given a
    #: proportional share of the machine via ``OMP_NUM_THREADS``
    #: (:func:`hawavoclean.runtime.threads_per_worker`); at the default of 1
    #: nothing is set and the numeric stack keeps its own defaults.
    num_threads: int = Field(default=1, ge=1, le=64)

    def pool_size(self) -> int:
        """Effective worker-pool size on this machine (clamped to CPU count)."""
        return worker_pool_size(self.num_threads)

    def threads_per_worker(self) -> int:
        """CPU threads each pooled worker may use without oversubscribing."""
        return threads_per_worker(self.num_threads)


class InputConfig(BaseFrozenModel):
    """Input validation and channel handling."""

    max_sample_rate: int = Field(default=48000, le=48000)
    #: The input rates this build is declared to handle. It is enforced as the
    #: accepted *envelope* — ``min(...)`` is the floor the media probe refuses
    #: below, ``max_sample_rate`` the ceiling it refuses above — and NOT as a
    #: membership test, because membership would be a lie in the other
    #: direction: every rate between the endpoints is resampled to the core's
    #: 48 kHz and processed correctly today, so rejecting 11.025 kHz for being
    #: absent from the list would refuse material the engine handles. The
    #: endpoints are what the UI's advisory rate warning mirrors.
    supported_sample_rates: list[int] = Field(
        default_factory=lambda: [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    )
    channel_mode: Literal[
        "auto", "mono", "dual_mono_same", "split_speakers", "ambiguous_stereo"
    ] = "auto"
    output_bit_depth: Literal["pcm24", "float32"] = "pcm24"

    @field_validator("supported_sample_rates")
    @classmethod
    def validate_supported_sample_rates(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError(
                "input.supported_sample_rates must not be empty: it is the accepted "
                "rate envelope, and an empty envelope accepts nothing."
            )
        if any(r <= 0 for r in v):
            raise ValueError("input.supported_sample_rates entries must be positive Hz values.")
        return sorted(set(v))

    @model_validator(mode="after")
    def validate_sample_rate_envelope(self) -> "InputConfig":
        if self.min_sample_rate > self.max_sample_rate:
            raise ValueError(
                f"input.supported_sample_rates starts at {self.min_sample_rate} Hz but "
                f"input.max_sample_rate is {self.max_sample_rate} Hz: no input rate "
                "could ever be accepted."
            )
        return self

    @property
    def min_sample_rate(self) -> int:
        """Floor of the accepted rate envelope."""
        return min(self.supported_sample_rates)


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
    # Measured tonal restoration: band levels against a speech-intelligibility
    # target, corrected by the bounded difference. Off means a muffled
    # recording is published as muffled as it arrived. See
    # hawavoclean.finishing.detect.measure_speech_tilt for the target curve and
    # how it was calibrated.
    tonal_restoration: bool = True
    # Ceiling on the whole tonal move, in dB of filter gain. This is a hard
    # bound on the filter bank, verified against its own measured response on
    # every call, not a target: the correction is whatever the file's measured
    # deficit earns, up to this.
    max_tonal_gain_db: float = Field(default=12.0, ge=0.0, le=18.0)
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
    """Load configuration from TOML file or return defaults with production constraints.

    Loading a configuration also *arms* its ``[runtime]`` section — the device
    is resolved and published, and a per-worker thread budget is set when the
    config asks for a worker pool. That is deliberate: it is the one moment a
    configuration stops being data and starts being the run, and it is what
    lets the spawned enhancement worker and the report agree on the compute
    device. An explicitly requested device this machine cannot provide fails
    here, before any audio is touched.
    """
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

    activate_runtime(
        config.runtime.device,
        core_id=config.enhancement.core_id,
        num_threads=config.runtime.num_threads,
        memory_limit_mb=config.runtime.worker_memory_limit_mb,
    )
    return config
