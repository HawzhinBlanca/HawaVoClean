"""DC subsonic filtering, narrow de-hum notch filters, and transient click repair."""

from typing import Any

import numpy as np
import scipy.signal


def remove_dc_subsonic(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    cutoff_hz: float = 20.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply highpass Butterworth filter to eliminate subsonic thump and DC offset."""
    if len(waveform) < 64:
        return waveform.copy()

    sos = scipy.signal.butter(4, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, waveform)
    return np.ascontiguousarray(filtered, dtype=np.float32)


def remove_electrical_hum(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    hum_freq_hz: float = 50.0,
    num_harmonics: int = 8,
    q_factor: float = 30.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply narrow zero-phase IIR notch filters at hum fundamental and harmonics."""
    if len(waveform) < 64 or hum_freq_hz <= 0.0:
        return waveform.copy()

    # Snap near-standard mains fundamentals (within ±0.5 Hz of 50/60 Hz) so
    # higher harmonics (3rd-8th) remain dead-center inside Q=30 notches.
    effective_freq = float(hum_freq_hz)
    for nominal in (50.0, 60.0):
        if abs(effective_freq - nominal) <= 0.5:
            effective_freq = nominal
            break

    current = waveform.astype(np.float64, copy=True)
    for h in range(1, num_harmonics + 1):
        f_notch = effective_freq * h
        if f_notch >= (sample_rate / 2.0) - 50.0:
            break
        b, a = scipy.signal.iirnotch(f_notch, q_factor, fs=sample_rate)
        current = scipy.signal.filtfilt(b, a, current)

    return np.ascontiguousarray(current, dtype=np.float32)


def repair_transient_clicks(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    threshold_sigma: float = 6.0,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int]:
    """Detect and linear-interpolate micro-transient click samples."""
    n = len(waveform)
    if n < 32:
        return waveform.copy(), 0

    diff = np.diff(waveform)
    thresh = float(np.mean(np.abs(diff)) + threshold_sigma * np.std(np.abs(diff)))
    click_indices = np.where(np.abs(diff) > max(0.25, thresh))[0] + 1

    repaired = waveform.copy()
    repaired_count = 0

    for idx in click_indices:
        if 2 <= idx < n - 2:
            # Replace click sample with cubic interpolation of neighbors
            repaired[idx] = 0.5 * (repaired[idx - 1] + repaired[idx + 1])
            repaired_count += 1

    return np.ascontiguousarray(repaired, dtype=np.float32), repaired_count
