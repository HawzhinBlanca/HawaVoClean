"""Configuration schema for HawaVoClean spectral restoration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseFrozenModel(BaseModel):
    """Base model with strict forbidden extra fields and frozen immutability."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RestorationGuardConfig(BaseFrozenModel):
    """Fidelity guard thresholds for Restoration Guard R."""

    protected_band_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    ctc_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    highband_event_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    harmonic_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    speaker_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class RestorationConfig(BaseFrozenModel):
    """Master configuration for HawaRestore-KD spectral restoration."""

    enabled: bool = False
    mode: Literal["explicit", "disabled"] = "explicit"
    target_sample_rate: int = Field(default=48000, ge=48000, le=48000)
    model: str = "hawarestore-kd"
    model_sha256: str = ""
    solver: Literal["midpoint", "euler"] = "midpoint"
    steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    strengths: list[float] = Field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0])
    cutoff_mode: Literal["auto", "explicit"] = "auto"
    cutoff_confidence_min: float = Field(default=0.80, ge=0.0, le=1.0)
    transition_hz: float = Field(default=500.0, ge=50.0, le=2000.0)
    process_non_speech: bool = False
    deterministic: bool = True
    guard: RestorationGuardConfig = Field(default_factory=RestorationGuardConfig)

    @field_validator("strengths")
    @classmethod
    def validate_strengths(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Restoration strengths ladder must not be empty.")
        for s in v:
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"Strength value {s} must be in [0.0, 1.0].")
        if 0.0 not in v:
            raise ValueError("Strengths ladder must include 0.0 as safe fallback candidate.")
        return sorted(v, reverse=True)
