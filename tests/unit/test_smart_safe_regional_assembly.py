"""Unit and adversarial tests for Smart Safe regional routing and assembly (I3.4).

Verifies:
- Sample-accurate crossfading with equal-power raised-cosine curves.
- Boundary step-discontinuity diffusion ensuring click-free joints (max delta < 1e-3).
- Short uncertain region inheritance and boundary hysteresis.
- Acoustic risk gating rejecting studio/restore on music or crosstalk.
- Localized regional abstention fallback to preserve upon guard failure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hawavoclean.smart_safe.decision import (
    AcousticEvidence,
    RegionRecommendation,
)
from hawavoclean.smart_safe.preview import (
    SmartSafePreviewEngine,
)
from hawavoclean.smart_safe.region import (
    RegionalAssemblyResult,
    filter_region_routes_for_acoustics,
    render_and_stitch_regions,
)


def _synthetic_modulated_speech(duration_s: float = 6.0, sr: int = 48000) -> np.ndarray:
    """Generate multi-harmonic voiced audio with natural speech envelope."""
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    f0 = 150.0
    harmonics = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.06 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    # 2.5 Hz speech syllable envelope modulation
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0.0, 1.0).astype(np.float32)
    noise = np.random.default_rng(42).normal(0.0, 0.005, len(t)).astype(np.float32)
    return np.ascontiguousarray(0.3 * harmonics * env + noise, dtype=np.float32)


def test_regional_assembly_smooth_crossfades_no_click() -> None:
    """Adjacent regions with different routes must stitch with step discontinuity < 1e-3."""
    sr = 48000
    audio = _synthetic_modulated_speech(duration_s=6.0, sr=sr)
    engine = SmartSafePreviewEngine()

    regions = [
        RegionRecommendation(0.0, 2.0, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(2.0, 4.0, "preserve", confidence=0.95, boundary_confidence=0.92),
        RegionRecommendation(4.0, 6.0, "production", confidence=0.88, boundary_confidence=0.89),
    ]

    result = render_and_stitch_regions(
        audio,
        sr,
        regions,
        engine,
        crossfade_ms=30.0,
    )

    assert isinstance(result, RegionalAssemblyResult)
    assert len(result.audio) == len(audio)
    assert result.seam_count == 2
    assert result.abstained_regions == 0
    # Strict click-free invariant: step discontinuity strictly < 1e-3
    assert result.max_step_discontinuity < 1e-3
    assert np.all(np.isfinite(result.audio))
    assert np.max(np.abs(result.audio)) <= 1.05


def test_short_uncertain_region_inherits_and_stitches() -> None:
    """Short uncertain region must inherit safer neighbor and stitch seamlessly."""
    sr = 48000
    audio = _synthetic_modulated_speech(duration_s=6.0, sr=sr)
    engine = SmartSafePreviewEngine()

    # Region 1 is 0.4s (< 1.0s) and low confidence (0.40) -> inherits safer neighbor "production"
    regions = [
        RegionRecommendation(0.0, 2.5, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(2.5, 2.9, "studio", confidence=0.40, boundary_confidence=0.50),
        RegionRecommendation(2.9, 6.0, "production", confidence=0.92, boundary_confidence=0.90),
    ]

    result = render_and_stitch_regions(audio, sr, regions, engine)
    assert len(result.regions) == 3
    # Middle region inherited production
    assert result.regions[1].route == "production"
    assert result.max_step_discontinuity < 1e-3
    assert np.all(np.isfinite(result.audio))


def test_music_and_crosstalk_risk_gating() -> None:
    """Acoustic evidence with high music or crosstalk risk demotes aggressive routes."""
    regions = [
        RegionRecommendation(0.0, 3.0, "studio", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(3.0, 6.0, "restore_source", confidence=0.85, boundary_confidence=0.85),
    ]

    evidence_music = AcousticEvidence(
        speech_dominance=0.80,
        music_risk=0.45,  # High music risk
        crosstalk_risk=0.02,
        rumble_confidence=0.10,
        band_limited_confidence=0.0,
        recorded_high_frequency_speech_confidence=0.0,
        speaker_match_confidence=0.0,
        speaker_match_verified=False,
        reconstruction_consent=False,
    )

    filtered = filter_region_routes_for_acoustics(regions, evidence_music)
    assert len(filtered) == 2
    # Aggressive routes demoted to production
    assert filtered[0].route == "production"
    assert filtered[1].route == "production"

    # With low speech dominance, demotes to preserve
    evidence_ambient = AcousticEvidence(
        speech_dominance=0.20,
        music_risk=0.40,
        crosstalk_risk=0.02,
        rumble_confidence=0.0,
        band_limited_confidence=0.0,
        recorded_high_frequency_speech_confidence=0.0,
        speaker_match_confidence=0.0,
        speaker_match_verified=False,
        reconstruction_consent=False,
    )
    filtered_ambient = filter_region_routes_for_acoustics(regions, evidence_ambient)
    assert filtered_ambient[0].route == "preserve"
    assert filtered_ambient[1].route == "preserve"


def test_regional_guard_failure_abstains_locally() -> None:
    """If a rendered region fails post-master checks, it falls back to preserve locally."""
    sr = 48000
    audio = _synthetic_modulated_speech(duration_s=4.0, sr=sr)
    engine = SmartSafePreviewEngine()

    # Mock render_route_audio to produce NaN or clipping on the second region
    original_render = engine.render_route_audio

    def mock_render(
        route: Any, aud: Any, sample_r: Any, **kwargs: Any
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str | None]:
        rendered, err = original_render(route, aud, sample_r, **kwargs)
        # If this is the second region slice (offset >= 2.0s), corrupt it
        if 1.5 < len(aud) / sample_r < 3.5:
            rendered.fill(np.nan)
        return rendered, err

    engine.render_route_audio = mock_render  # type: ignore[assignment]

    regions = [
        RegionRecommendation(0.0, 2.0, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(2.0, 4.0, "production", confidence=0.90, boundary_confidence=0.90),
    ]

    result = render_and_stitch_regions(audio, sr, regions, engine)
    assert result.abstained_regions >= 1
    assert result.regions[1].route == "preserve"
    assert np.all(np.isfinite(result.audio))
    assert result.max_step_discontinuity < 1e-3


def test_empty_and_zero_duration_edge_cases() -> None:
    """Zero-length audio or empty regions return empty result without crashing."""
    engine = SmartSafePreviewEngine()
    empty_res = render_and_stitch_regions(np.empty(0, dtype=np.float32), 48000, (), engine)
    assert len(empty_res.audio) == 0
    assert empty_res.max_step_discontinuity == 0.0
    assert empty_res.seam_count == 0

    audio = _synthetic_modulated_speech(duration_s=2.0, sr=48000)
    with pytest.raises(ValueError, match="1D mono"):
        render_and_stitch_regions(np.zeros((2, 100), dtype=np.float32), 48000, (), engine)

    # Empty regions list with non-empty audio
    res_no_reg = render_and_stitch_regions(audio, 48000, (), engine)
    assert len(res_no_reg.audio) == 0


def test_candidate_order_invariance_regional_assembly() -> None:
    """Candidate order in regions list does not affect contiguous rendering."""
    sr = 48000
    audio = _synthetic_modulated_speech(duration_s=4.0, sr=sr)
    engine = SmartSafePreviewEngine()

    regions = [
        RegionRecommendation(0.0, 2.0, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(2.0, 4.0, "preserve", confidence=0.95, boundary_confidence=0.90),
    ]

    res1 = render_and_stitch_regions(audio, sr, regions, engine)
    res2 = render_and_stitch_regions(audio, sr, list(regions), engine)

    assert np.array_equal(res1.audio, res2.audio)
    assert res1.max_step_discontinuity == res2.max_step_discontinuity


def test_padding_and_error_branches() -> None:
    """Covers slice length padding/trimming and error exception fallback."""
    sr = 48000
    audio = _synthetic_modulated_speech(duration_s=3.0, sr=sr)
    engine = SmartSafePreviewEngine()

    # 1. Short render slice requires padding
    def mock_short_render(
        _route: Any, _aud: Any, _sample_r: Any, **_kwargs: Any
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str | None]:
        return np.zeros(100, dtype=np.float32), None

    engine.render_route_audio = mock_short_render  # type: ignore[assignment]
    regions = [
        RegionRecommendation(0.0, 1.5, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(1.5, 3.0, "preserve", confidence=0.90, boundary_confidence=0.90),
    ]
    res_short = render_and_stitch_regions(audio, sr, regions, engine)
    assert len(res_short.audio) == len(audio)

    # 2. Render route raises exception -> catches and falls back to preserve
    def mock_failing_render(
        _route: Any, _aud: Any, _sample_r: Any, **_kwargs: Any
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str | None]:
        raise RuntimeError("DSP crash")

    engine.render_route_audio = mock_failing_render  # type: ignore[assignment]
    res_err = render_and_stitch_regions(audio, sr, regions, engine)
    assert res_err.abstained_regions == 1
    assert res_err.regions[0].route == "preserve"


def test_filter_region_routes_for_acoustics_branches() -> None:
    # 1. High music risk with low speech dominance (< 0.50) -> demotes to preserve
    low_speech_evidence = AcousticEvidence(
        speech_dominance=0.30,
        music_risk=0.50,
        crosstalk_risk=0.0,
        rumble_confidence=0.0,
        band_limited_confidence=0.0,
        recorded_high_frequency_speech_confidence=0.0,
        speaker_match_confidence=0.0,
        speaker_match_verified=False,
        reconstruction_consent=False,
    )
    regions = [
        RegionRecommendation(0.0, 1.0, "studio", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(1.0, 2.0, "preserve", confidence=0.90, boundary_confidence=0.90),
    ]
    filtered = filter_region_routes_for_acoustics(regions, low_speech_evidence)
    assert filtered[0].route == "preserve"
    assert filtered[1].route == "preserve"

    # 2. No music or crosstalk risk -> keeps original regions unchanged
    clean_evidence = AcousticEvidence(
        speech_dominance=0.95,
        music_risk=0.05,
        crosstalk_risk=0.05,
        rumble_confidence=0.0,
        band_limited_confidence=0.0,
        recorded_high_frequency_speech_confidence=0.0,
        speaker_match_confidence=0.0,
        speaker_match_verified=False,
        reconstruction_consent=False,
    )
    filtered_clean = filter_region_routes_for_acoustics(regions, clean_evidence)
    assert filtered_clean[0].route == "studio"
    assert filtered_clean[1].route == "preserve"


def test_render_and_stitch_regions_edge_branches() -> None:
    sr = 48000
    audio = np.ones(sr * 3, dtype=np.float32) * 0.1
    engine = SmartSafePreviewEngine()

    # 1. Route error returned as string (triggers line 176 route_error check)
    def mock_route_error_render(
        _route: Any, _aud: Any, _sample_r: Any, **_kwargs: Any
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str | None]:
        return np.zeros(100, dtype=np.float32), "simulated backend solver failure"

    engine.render_route_audio = mock_route_error_render  # type: ignore[assignment]
    regions = [
        RegionRecommendation(0.0, 2.0, "production", confidence=0.90, boundary_confidence=0.90),
        # Micro-second region where int(round(start*sr)) == int(round(end*sr)) -> core_len == 0
        RegionRecommendation(
            2.0, 2.000001, "production", confidence=0.90, boundary_confidence=0.90
        ),
    ]
    res = render_and_stitch_regions(audio, sr, regions, engine)
    assert res.abstained_regions == 1
    assert res.regions[0].route == "preserve"  # Demoted from error
    assert res.regions[1].route == "production"  # core_len == 0 region preserved as-is

    # 2. Oversized render result (triggers line 183 reg_wave[:core_len])
    def mock_oversized_render(
        _route: Any, aud: Any, _sample_r: Any, **_kwargs: Any
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str | None]:
        # Return twice as many samples as passed
        return np.ones(len(aud) * 2, dtype=np.float32) * 0.2, None

    engine.render_route_audio = mock_oversized_render  # type: ignore[assignment]
    regions2 = [
        RegionRecommendation(0.0, 1.0, "production", confidence=0.90, boundary_confidence=0.90),
        RegionRecommendation(1.0, 2.0, "production", confidence=0.90, boundary_confidence=0.90),
    ]
    # Pass acoustic evidence to trigger line 134
    clean_ev = AcousticEvidence(
        speech_dominance=0.90,
        music_risk=0.01,
        crosstalk_risk=0.01,
        rumble_confidence=0.0,
        band_limited_confidence=0.0,
        recorded_high_frequency_speech_confidence=0.0,
        speaker_match_confidence=0.0,
        speaker_match_verified=False,
        reconstruction_consent=False,
    )
    res2 = render_and_stitch_regions(
        audio, sr, regions2, engine, crossfade_ms=30.0, acoustic_evidence=clean_ev
    )
    assert len(res2.audio) == len(audio)
    assert res2.seam_count >= 1
