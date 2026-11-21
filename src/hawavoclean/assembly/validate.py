"""Post-assembly invariant validation enforcing exact structural continuity."""

import numpy as np

from hawavoclean.audio.types import AudioBuffer
from hawavoclean.errors import OutputValidationError
from hawavoclean.runtime import evict_memmap_pages
from hawavoclean.segmentation.types import SpeechUnit

VALIDATION_CHUNK_SAMPLES = 1 << 20


def validate_assembled_timeline(
    assembled_buffer: AudioBuffer,
    expected_channels: int,
    expected_samples: int,
    expected_sample_rate: int,
    units: list[SpeechUnit],
) -> None:
    """Verify all 6 non-negotiable post-assembly invariants."""
    data = assembled_buffer.data
    channels, samples = data.shape

    # 1. Sample count match
    if samples != expected_samples:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )

    # 2. Channel count match
    if channels != expected_channels:
        raise OutputValidationError(
            f"Assembled output channels {channels} != expected input channels {expected_channels}"
        )

    # 3. Sample rate match
    if assembled_buffer.sample_rate != expected_sample_rate:
        raise OutputValidationError(
            f"Assembled output sample rate {assembled_buffer.sample_rate} != expected {expected_sample_rate}"
        )

    # 4. All samples finite
    for start in range(0, samples, VALIDATION_CHUNK_SAMPLES):
        end = min(samples, start + VALIDATION_CHUNK_SAMPLES)
        if not np.all(np.isfinite(data[:, start:end])):
            raise OutputValidationError("Assembled output audio contains NaN or Infinite values.")
        evict_memmap_pages(data, start, end)

    # 5 & 6. Timeline coverage and duplication checks
    # Group units by channel and check coverage
    ch_units_map: dict[int, list[SpeechUnit]] = {}
    for u in units:
        ch_units_map.setdefault(u.channel_id, []).append(u)

    for ch_id, ch_u_list in ch_units_map.items():
        sorted_units = sorted(ch_u_list, key=lambda x: x.start_sample)
        curr = 0
        for u in sorted_units:
            if u.start_sample > curr:
                raise OutputValidationError(
                    f"Timeline gap detected on channel {ch_id}: uncovered span [{curr}, {u.start_sample}]"
                )
            if u.start_sample < curr:
                raise OutputValidationError(
                    f"Timeline overlap detected on channel {ch_id}: duplicated span [{u.start_sample}, {curr}]"
                )
            curr = u.end_sample

        if curr != expected_samples:
            raise OutputValidationError(
                f"Timeline coverage incomplete on channel {ch_id}: covered {curr} != expected {expected_samples}"
            )
