"""Controlled linguistic and acoustic corruption suite generator for Guard validation."""

from typing import Any

import numpy as np
import scipy.signal


def corrupt_consonant_splice(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    start_time_s: float = 1.0,
    cut_duration_ms: float = 80.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Simulate consonant deletion/splice artifact."""
    start_sample = int(round(sample_rate * start_time_s))
    cut_samples = int(round(sample_rate * (cut_duration_ms / 1000.0)))
    end_sample = min(len(waveform), start_sample + cut_samples)

    if start_sample >= len(waveform) or end_sample <= start_sample:
        return waveform.copy()

    # Splice out the consonant chunk
    return np.concatenate([waveform[:start_sample], waveform[end_sample:]]).astype(np.float32)


def corrupt_syllable_deletion(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    start_time_s: float = 1.5,
    deletion_ms: float = 200.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Simulate syllable deletion in a voiced span."""
    return corrupt_consonant_splice(waveform, sample_rate, start_time_s, deletion_ms)


def corrupt_repeated_span(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    start_time_s: float = 1.0,
    span_ms: float = 350.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Duplicate / stutter a speech span."""
    start = int(round(sample_rate * start_time_s))
    span = int(round(sample_rate * (span_ms / 1000.0)))
    end = min(len(waveform), start + span)

    if start >= len(waveform) or end <= start:
        return waveform.copy()

    snippet = waveform[start:end]
    return np.concatenate([waveform[:end], snippet, waveform[end:]]).astype(np.float32)


def corrupt_hf_consonant_removal(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    cutoff_hz: float = 1500.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Muffle and wipe high-frequency consonants with aggressive lowpass filter."""
    sos = scipy.signal.butter(6, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return np.ascontiguousarray(scipy.signal.sosfiltfilt(sos, waveform), dtype=np.float32)


def corrupt_spectral_holes(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    band_low_hz: float = 2000.0,
    band_high_hz: float = 4000.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Zero out critical speech frequency subband."""
    sos = scipy.signal.butter(
        4, [band_low_hz, band_high_hz], btype="bandstop", fs=sample_rate, output="sos"
    )
    return np.ascontiguousarray(scipy.signal.sosfiltfilt(sos, waveform), dtype=np.float32)


def corrupt_dropout(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    start_time_s: float = 1.0,
    duration_ms: float = 150.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Introduce zeroed-out dropout gap in active speech."""
    start = int(round(sample_rate * start_time_s))
    dur = int(round(sample_rate * (duration_ms / 1000.0)))
    end = min(len(waveform), start + dur)

    corrupted = waveform.copy()
    corrupted[start:end] = 0.0
    return corrupted
