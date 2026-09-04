"""Sample-accurate timeline stitching across processed speech and non-speech units."""

from typing import Any

import numpy as np

from hawavoclean.runtime import evict_memmap_pages
from hawavoclean.segmentation.types import SpeechUnit


def assemble_channel_timeline(
    units: list[SpeechUnit],
    unit_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]],
    total_samples: int,
    sample_rate: int,
    crossfade_ms: float = 20.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Stitch unit core waveforms into a seamless canonical timeline."""
    if total_samples == 0:
        return np.empty(0, dtype=np.float32)

    timeline = np.zeros(total_samples, dtype=np.float32)
    assemble_channel_timeline_into(
        timeline,
        units,
        unit_waveforms,
        total_samples=total_samples,
        sample_rate=sample_rate,
        crossfade_ms=crossfade_ms,
    )
    return np.ascontiguousarray(timeline, dtype=np.float32)


def assemble_channel_timeline_into(
    timeline: np.ndarray[Any, np.dtype[np.float32]],
    units: list[SpeechUnit],
    unit_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]],
    total_samples: int,
    sample_rate: int,
    crossfade_ms: float = 20.0,
) -> None:
    """Stitch into a caller-owned timeline, including a disk-backed mapping.

    This is the production boundary used for long Natural jobs. The existing
    allocating wrapper delegates here, so the seam/declick arithmetic has one
    implementation and short-file output remains pinned to the same code.
    """
    if timeline.ndim != 1 or len(timeline) != total_samples:
        raise ValueError(
            f"Assembly destination has shape {timeline.shape}; expected ({total_samples},)"
        )
    if timeline.dtype != np.float32:
        raise ValueError(f"Assembly destination must be float32, got {timeline.dtype}")
    if total_samples == 0:
        return

    timeline.fill(0.0)
    evict_memmap_pages(timeline)
    crossfade_samples = int(round(sample_rate * (crossfade_ms / 1000.0)))

    for i, (unit, wave) in enumerate(zip(units, unit_waveforms, strict=True)):
        start = unit.start_sample
        end = unit.end_sample
        core_len = end - start

        # Ensure exact length match
        if len(wave) < core_len:
            wave = np.pad(wave, (0, core_len - len(wave)), mode="constant")
        elif len(wave) > core_len:
            wave = wave[:core_len]

        timeline[start:end] = wave

        # Boundary declick: diffuse any step discontinuity at the joint into
        # the head of this unit. Every source sample is rendered exactly once;
        # the correction is a decaying offset, not duplicated content.
        if i > 0 and crossfade_samples > 0:
            prev_unit = units[i - 1]
            # forced_boundary marks the cut at a unit's END: the joint at
            # `start` is a forced mid-speech cut iff PREV_unit carries the flag.
            if prev_unit.end_sample == start and not prev_unit.forced_boundary and start > 0:
                fade_n = min(
                    crossfade_samples,
                    core_len // 4,
                    (prev_unit.end_sample - prev_unit.start_sample) // 4,
                )
                if fade_n > 0:
                    step = float(timeline[start - 1]) - float(wave[0])
                    if abs(step) > 1e-9:
                        ramp = np.linspace(1.0, 0.0, fade_n, endpoint=False, dtype=np.float32)
                        timeline[start : start + fade_n] += step * ramp
