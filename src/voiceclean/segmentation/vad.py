"""Voice Activity Detection (VAD) algorithms and interval extraction."""

from typing import Any

import numpy as np

from voiceclean.segmentation.types import SpeechInterval


def detect_speech_energy(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
    min_speech_ms: int = 150,
    pause_merge_ms: int = 250,
    energy_threshold_rel_db: float = -38.0,
) -> list[SpeechInterval]:
    """Robust adaptive energy-based VAD for speech activity extraction."""
    if len(waveform) == 0:
        return []

    frame_size = int(round(sample_rate * (frame_ms / 1000.0)))
    hop_size = int(round(sample_rate * (hop_ms / 1000.0)))

    if len(waveform) < frame_size:
        # Too short, check if non-zero
        rms = float(np.sqrt(np.mean(waveform**2)))
        if rms > 1e-4:
            return [SpeechInterval(0, len(waveform))]
        return []

    # Compute short-term frame energy (RMS in dB)
    num_frames = max(1, (len(waveform) - frame_size) // hop_size + 1)
    frame_rms = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        chunk = waveform[start : start + frame_size]
        frame_rms[i] = np.sqrt(np.mean(chunk**2) + 1e-12)

    max_rms = float(np.max(frame_rms))
    if max_rms < 1e-5:
        # Essentially digital silence
        return []

    threshold = max(max_rms * (10.0 ** (energy_threshold_rel_db / 20.0)), 1e-4)
    is_speech_frame = frame_rms >= threshold

    # Extract raw contiguous speech intervals
    raw_intervals: list[SpeechInterval] = []
    in_speech = False
    start_sample = 0

    for i, active in enumerate(is_speech_frame):
        frame_start = i * hop_size
        frame_end = min(len(waveform), frame_start + frame_size)

        if active and not in_speech:
            in_speech = True
            start_sample = frame_start
        elif not active and in_speech:
            in_speech = False
            raw_intervals.append(SpeechInterval(start_sample, frame_end))

    if in_speech:
        raw_intervals.append(SpeechInterval(start_sample, len(waveform)))

    if not raw_intervals:
        return []

    # Step 1: Merge intervals separated by less than pause_merge_ms
    merge_gap_samples = int(round(sample_rate * (pause_merge_ms / 1000.0)))
    merged: list[SpeechInterval] = []
    curr_start = raw_intervals[0].start_sample
    curr_end = raw_intervals[0].end_sample

    for iv in raw_intervals[1:]:
        if iv.start_sample - curr_end <= merge_gap_samples:
            curr_end = max(curr_end, iv.end_sample)
        else:
            merged.append(SpeechInterval(curr_start, curr_end))
            curr_start = iv.start_sample
            curr_end = iv.end_sample
    merged.append(SpeechInterval(curr_start, curr_end))

    # Step 2: Discard intervals shorter than min_speech_ms
    min_speech_samples = int(round(sample_rate * (min_speech_ms / 1000.0)))
    filtered = [iv for iv in merged if iv.length_samples >= min_speech_samples]

    return filtered
