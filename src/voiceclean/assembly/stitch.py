"""Sample-accurate timeline stitching across processed speech and non-speech units."""

from typing import Any

import numpy as np

from voiceclean.assembly.overlap import compute_equal_power_crossfade
from voiceclean.segmentation.types import SpeechUnit


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

        # Check if crossfade with previous unit is warranted
        if i > 0 and crossfade_samples > 0:
            prev_unit = units[i - 1]
            # Crossfade if both units meet at exact boundary inside non-speech
            if prev_unit.end_sample == start and not unit.forced_boundary:
                fade_n = min(
                    crossfade_samples,
                    core_len // 4,
                    (prev_unit.end_sample - prev_unit.start_sample) // 4,
                    start,  # prevent negative overlap_start
                )
                if fade_n > 0:
                    fade_out, fade_in = compute_equal_power_crossfade(fade_n)
                    overlap_start = start - fade_n
                    # Apply crossfade in overlap region
                    timeline[overlap_start:start] = (
                        timeline[overlap_start:start] * fade_out + wave[:fade_n] * fade_in
                    )
                    timeline[start:end] = wave
                    continue

        timeline[start:end] = wave

    return np.ascontiguousarray(timeline, dtype=np.float32)
