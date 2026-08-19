"""Utterance grouping, context window expansion, and zero-crossing boundary optimization."""

from typing import Any

import numpy as np

from hawavoclean.config import SegmentationConfig
from hawavoclean.hashing import hash_numpy
from hawavoclean.segmentation.types import SpeechInterval, SpeechUnit
from hawavoclean.segmentation.vad import detect_speech_energy


def find_lowest_energy_zero_crossing(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    target_sample: int,
    search_window_samples: int,
) -> int:
    """Find the zero crossing with lowest local RMS energy within a search window."""
    total_len = len(waveform)
    start_search = max(0, target_sample - search_window_samples)
    end_search = min(total_len - 1, target_sample + search_window_samples)

    if end_search <= start_search + 1:
        return min(total_len, max(0, target_sample))

    region = waveform[start_search:end_search]
    zero_crossings = np.where(np.diff(np.signbit(region)))[0] + start_search

    if len(zero_crossings) == 0:
        # Fallback to sample with absolute minimum amplitude
        min_idx = int(np.argmin(np.abs(region)))
        return start_search + min_idx

    # Evaluate local RMS in 50ms window around each zero crossing
    window_50ms = min(2400, search_window_samples // 4)
    best_idx = zero_crossings[0]
    min_rms = float("inf")

    for zx in zero_crossings:
        w_start = max(0, zx - window_50ms)
        w_end = min(total_len, zx + window_50ms)
        rms = float(np.sqrt(np.mean(waveform[w_start:w_end] ** 2)))
        if rms < min_rms:
            min_rms = rms
            best_idx = zx

    return int(best_idx)


def build_speech_units(
    channel_waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    channel_id: int,
    config: SegmentationConfig,
    start_unit_id: int = 0,
) -> list[SpeechUnit]:
    """Segment a single channel into seamless, contiguous SpeechUnits with context windows."""
    total_samples = len(channel_waveform)
    if total_samples == 0:
        return []

    # Step 1: Detect speech intervals
    speech_intervals = detect_speech_energy(
        waveform=channel_waveform,
        sample_rate=sample_rate,
        min_speech_ms=config.min_speech_ms,
        pause_merge_ms=config.pause_merge_threshold_ms,
    )

    context_samples = int(round(sample_rate * config.context_duration_s))
    target_group_samples = int(round(sample_rate * config.target_speech_group_s))
    hard_max_samples = int(round(sample_rate * config.hard_max_group_s))

    if not speech_intervals:
        # All non-speech unit
        u_hash = hash_numpy(channel_waveform)
        return [
            SpeechUnit(
                unit_id=start_unit_id,
                channel_id=channel_id,
                start_sample=0,
                end_sample=total_samples,
                context_start_sample=0,
                context_end_sample=total_samples,
                is_speech=False,
                forced_boundary=False,
                speech_mask=np.zeros(total_samples, dtype=np.bool_),
                input_sha256=u_hash,
            )
        ]

    # Step 2: Form contiguous units covering [0, total_samples]
    # We create units by partitioning the timeline at safe non-speech boundaries.
    units: list[SpeechUnit] = []
    unit_counter = start_unit_id

    # Create a full timeline speech mask
    full_mask = np.zeros(total_samples, dtype=np.bool_)
    for iv in speech_intervals:
        full_mask[iv.start_sample : iv.end_sample] = True

    current_start = 0
    i = 0
    num_intervals = len(speech_intervals)

    while current_start < total_samples:
        # If current_start is before the first/next speech interval, handle leading non-speech
        if i < num_intervals and current_start < speech_intervals[i].start_sample:
            next_speech_start = speech_intervals[i].start_sample
            # If leading non-speech is long (> 1.0s), make it its own non-speech unit
            if (next_speech_start - current_start) > sample_rate:
                end_sample = next_speech_start
                core = channel_waveform[current_start:end_sample]
                units.append(
                    SpeechUnit(
                        unit_id=unit_counter,
                        channel_id=channel_id,
                        start_sample=current_start,
                        end_sample=end_sample,
                        context_start_sample=max(0, current_start - context_samples),
                        context_end_sample=min(total_samples, end_sample + context_samples),
                        is_speech=False,
                        forced_boundary=False,
                        speech_mask=full_mask[current_start:end_sample],
                        input_sha256=hash_numpy(core),
                    )
                )
                unit_counter += 1
                current_start = end_sample
                continue

        # Accumulate speech intervals until target group length is reached or all consumed
        group_end = current_start
        forced = False

        if i >= num_intervals:
            # Trailing non-speech to end of file
            end_sample = total_samples
            core = channel_waveform[current_start:end_sample]
            units.append(
                SpeechUnit(
                    unit_id=unit_counter,
                    channel_id=channel_id,
                    start_sample=current_start,
                    end_sample=end_sample,
                    context_start_sample=max(0, current_start - context_samples),
                    context_end_sample=total_samples,
                    is_speech=False,
                    forced_boundary=False,
                    speech_mask=full_mask[current_start:end_sample],
                    input_sha256=hash_numpy(core),
                )
            )
            unit_counter += 1
            break

        # Grouping loop
        while i < num_intervals:
            iv = speech_intervals[i]
            proposed_end = iv.end_sample
            proposed_len = proposed_end - current_start

            if proposed_len <= target_group_samples:
                group_end = proposed_end
                i += 1
            elif proposed_len <= hard_max_samples:
                # Still within hard max, include this interval and stop group
                group_end = proposed_end
                i += 1
                break
            else:
                # Interval itself or group exceeds hard max
                if group_end > current_start:
                    # We already have accumulated speech, close current unit before this large interval
                    break
                else:
                    # Single interval exceeds hard max -> force cut at lowest energy zero crossing
                    cut_target = current_start + target_group_samples
                    search_win = int(round(sample_rate * 1.0))
                    cut_point = find_lowest_energy_zero_crossing(
                        channel_waveform, cut_target, search_win
                    )
                    group_end = cut_point
                    forced = True
                    # Adjust the current speech interval start in place
                    speech_intervals[i] = SpeechInterval(cut_point, iv.end_sample)
                    break

        # Check if there is non-speech buffer after group_end before next speech interval
        if not forced and i < num_intervals:
            next_start = speech_intervals[i].start_sample
            gap = next_start - group_end
            if gap > 0:
                # Split the silence in the middle
                group_end = group_end + (gap // 2)

        end_sample = min(total_samples, group_end)
        if end_sample <= current_start:
            end_sample = min(total_samples, current_start + target_group_samples)

        core = channel_waveform[current_start:end_sample]
        mask = full_mask[current_start:end_sample]
        has_speech = bool(np.any(mask))

        units.append(
            SpeechUnit(
                unit_id=unit_counter,
                channel_id=channel_id,
                start_sample=current_start,
                end_sample=end_sample,
                context_start_sample=max(0, current_start - context_samples),
                context_end_sample=min(total_samples, end_sample + context_samples),
                is_speech=has_speech,
                forced_boundary=forced,
                speech_mask=mask,
                input_sha256=hash_numpy(core),
            )
        )
        unit_counter += 1
        current_start = end_sample

    return units
