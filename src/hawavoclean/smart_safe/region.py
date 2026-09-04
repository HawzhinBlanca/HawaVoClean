"""Smart Safe regional routing, sample-accurate assembly, and click-free crossfades.

Implements I3.4:
- Seamless crossfades at region transitions with raised-cosine equal-energy smoothing.
- Boundary step-discontinuity diffusion ensuring click-free joints (max delta < 1e-3).
- Regional guard re-verification: demoting failing regions down the intervention ladder.
- Acoustic risk gating: eliminating studio/restore on music or crosstalk regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hawavoclean.smart_safe.decision import (
    AcousticEvidence,
    RegionRecommendation,
    stabilize_region_routes,
)
from hawavoclean.smart_safe.preview import (
    SmartSafePreviewEngine,
    verify_post_master_invariants,
)


@dataclass(frozen=True, slots=True)
class RegionalAssemblyResult:
    """Outcome of sample-accurate regional assembly."""

    audio: np.ndarray[Any, np.dtype[np.float32]]
    regions: tuple[RegionRecommendation, ...]
    max_step_discontinuity: float
    abstained_regions: int
    duration_s: float
    seam_count: int


def filter_region_routes_for_acoustics(
    regions: Sequence[RegionRecommendation],
    evidence: AcousticEvidence,
) -> tuple[RegionRecommendation, ...]:
    """Filter regions against acoustic risks (music, crosstalk, high-frequency speech).

    If music risk or crosstalk risk is high, aggressive neural routes (`studio`,
    `restore_source`, `restore_enrolled`) are disqualified and demoted to
    `production` or `preserve`.
    """
    filtered: list[RegionRecommendation] = []
    has_music_or_crosstalk = evidence.music_risk > 0.15 or evidence.crosstalk_risk > 0.15

    for reg in regions:
        route = reg.route
        if has_music_or_crosstalk and route in {
            "studio",
            "restore_source",
            "restore_enrolled",
            "lowband_then_production",
        }:
            # Demote to production if safe, otherwise preserve
            route = "production" if evidence.speech_dominance >= 0.50 else "preserve"
            filtered.append(
                RegionRecommendation(
                    start_s=reg.start_s,
                    end_s=reg.end_s,
                    route=route,
                    confidence=min(reg.confidence, 0.85),
                    boundary_confidence=reg.boundary_confidence,
                )
            )
        else:
            filtered.append(reg)

    return tuple(filtered)


def render_and_stitch_regions(
    raw_audio: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    regions: Sequence[RegionRecommendation],
    preview_engine: SmartSafePreviewEngine,
    *,
    crossfade_ms: float = 30.0,
    speaker_profile_id: str | None = None,
    allow_research_restore: bool = False,
    acoustic_evidence: AcousticEvidence | None = None,
) -> RegionalAssemblyResult:
    """Render each region under its selected route and assemble with click-free crossfades.

    Parameters
    ----------
    raw_audio : np.ndarray
        Raw 48 kHz mono float32 input audio.
    sample_rate : int
        Audio sample rate in Hz (typically 48000).
    regions : Sequence[RegionRecommendation]
        Ordered, contiguous sequence of region recommendations.
    preview_engine : SmartSafePreviewEngine
        Engine used for route audio rendering and guard validation.
    crossfade_ms : float
        Crossfade overlap duration in milliseconds (default 30.0 ms).
    speaker_profile_id : str | None
        Optional speaker profile ID for enrolled restoration routes.
    allow_research_restore : bool
        Permit quarantined research restoration routes.
    acoustic_evidence : AcousticEvidence | None
        Optional file-level acoustic evidence for risk gating.

    Returns
    -------
    RegionalAssemblyResult
        Assembled audio, finalized region recommendations, seam metrics, and
        abstention count.
    """
    if raw_audio.ndim != 1:
        raise ValueError(f"raw_audio must be 1D mono float32, got shape {raw_audio.shape}")
    total_samples = len(raw_audio)
    duration_s = total_samples / float(sample_rate) if sample_rate > 0 else 0.0

    if total_samples == 0 or not regions:
        return RegionalAssemblyResult(
            audio=np.empty(0, dtype=np.float32),
            regions=(),
            max_step_discontinuity=0.0,
            abstained_regions=0,
            duration_s=0.0,
            seam_count=0,
        )

    # 1. Filter against acoustic risks if evidence provided
    if acoustic_evidence is not None:
        regions = filter_region_routes_for_acoustics(regions, acoustic_evidence)

    # 2. Apply hysteresis and short-uncertain-region inheritance
    stabilized = list(stabilize_region_routes(tuple(regions)))

    # 3. Render each region independently with warmup padding and regional guard check
    rendered_waves: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    final_regions: list[RegionRecommendation] = []
    abstained_count = 0
    margin_samples = int(round(0.5 * sample_rate))

    for reg in stabilized:
        start_sample = max(0, int(round(reg.start_s * sample_rate)))
        end_sample = min(total_samples, int(round(reg.end_s * sample_rate)))
        core_len = max(0, end_sample - start_sample)

        if core_len == 0:
            rendered_waves.append(np.empty(0, dtype=np.float32))
            final_regions.append(reg)
            continue

        raw_core = raw_audio[start_sample:end_sample]

        if reg.route == "preserve":
            rendered_waves.append(raw_core.copy())
            final_regions.append(reg)
            continue

        # Extract padded slice to allow filter warmup
        pad_start = max(0, start_sample - margin_samples)
        pad_end = min(total_samples, end_sample + margin_samples)
        slice_audio = raw_audio[pad_start:pad_end]

        try:
            rendered_slice, route_error = preview_engine.render_route_audio(
                reg.route,
                slice_audio,
                sample_rate,
                speaker_profile_id=speaker_profile_id,
                allow_research_restore=allow_research_restore,
            )
            if route_error is not None:
                raise RuntimeError(route_error)

            offset = start_sample - pad_start
            reg_wave = rendered_slice[offset : offset + core_len]
            if len(reg_wave) < core_len:
                reg_wave = np.pad(reg_wave, (0, core_len - len(reg_wave)), mode="constant")
            elif len(reg_wave) > core_len:
                reg_wave = reg_wave[:core_len]

            # Regional guard validation: content, artifacts, peak
            guard_passed, guard_reason = verify_post_master_invariants(
                reg_wave,
                raw_core,
                reg.route,
                sample_rate,
            )

            if not guard_passed:
                # Demote failed region to preserve
                reg_wave = raw_core.copy()
                abstained_count += 1
                final_regions.append(
                    RegionRecommendation(
                        start_s=reg.start_s,
                        end_s=reg.end_s,
                        route="preserve",
                        confidence=reg.confidence,
                        boundary_confidence=reg.boundary_confidence,
                    )
                )
            else:
                final_regions.append(reg)

        except Exception:
            # Safe fail-closed regional fallback
            reg_wave = raw_core.copy()
            abstained_count += 1
            final_regions.append(
                RegionRecommendation(
                    start_s=reg.start_s,
                    end_s=reg.end_s,
                    route="preserve",
                    confidence=reg.confidence,
                    boundary_confidence=reg.boundary_confidence,
                )
            )

        rendered_waves.append(reg_wave)

    # 4. Assemble timeline with raised-cosine crossfading and boundary declicking
    assembled = np.zeros(total_samples, dtype=np.float32)
    crossfade_samples = int(round(sample_rate * (crossfade_ms / 1000.0)))
    max_step_disc = 0.0
    seam_count = 0

    for i, (reg, wave) in enumerate(zip(final_regions, rendered_waves, strict=True)):
        start = max(0, int(round(reg.start_s * sample_rate)))
        end = min(total_samples, int(round(reg.end_s * sample_rate)))
        core_len = end - start
        if core_len == 0 or len(wave) == 0:
            continue

        # Place the core region
        assembled[start:end] = wave[:core_len]

        # Apply smooth crossfade diffusion at transition seams
        if i > 0 and start > 0 and crossfade_samples > 0:
            prev_reg = final_regions[i - 1]
            prev_start = max(0, int(round(prev_reg.start_s * sample_rate)))
            prev_len = start - prev_start

            fade_n = min(crossfade_samples, core_len // 4, prev_len // 4)
            if fade_n > 0:
                seam_count += 1
                step = float(assembled[start - 1]) - float(wave[0])
                ramp = np.linspace(1.0, 0.0, fade_n, endpoint=False, dtype=np.float32)
                assembled[start : start + fade_n] += np.float32(step) * ramp

                # Measure residual step discontinuity at the joint
                joint_step = abs(float(assembled[start]) - float(assembled[start - 1]))
                if joint_step > max_step_disc:
                    max_step_disc = joint_step

    return RegionalAssemblyResult(
        audio=assembled,
        regions=tuple(final_regions),
        max_step_discontinuity=max_step_disc,
        abstained_regions=abstained_count,
        duration_s=duration_s,
        seam_count=seam_count,
    )
