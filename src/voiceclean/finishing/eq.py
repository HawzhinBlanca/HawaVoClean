"""Gentle dynamic parametric equalization for speech presence and mud reduction."""

from typing import Any

import numpy as np
import scipy.signal


def apply_speech_eq(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    mud_cut_db: float = -1.5,
    presence_boost_db: float = 1.0,
    air_shelf_db: float = 0.5,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply conservative 3-band parametric EQ to improve dialogue clarity without harshness."""
    if len(waveform) < 128:
        return waveform.copy()

    # Band 1: Low-mid dip around 350Hz (mud reduction)
    # Band 2: Presence peak around 3.2kHz (consonant articulation)
    # Band 3: High shelf above 10kHz (air/smoothness)
    current = waveform.copy()

    # 350 Hz Bell filter
    f0 = 350.0
    q = 1.2
    gain_linear = 10.0 ** (mud_cut_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * gain_linear
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * gain_linear
    a0 = 1.0 + alpha / gain_linear
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / gain_linear
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    current = scipy.signal.filtfilt(b, a, current)

    # 3200 Hz Presence Bell
    f_pres = min(3200.0, sample_rate * 0.45)
    gain_pres = 10.0 ** (presence_boost_db / 40.0)
    w_p = 2.0 * np.pi * f_pres / sample_rate
    alpha_p = np.sin(w_p) / (2.0 * 1.5)
    b0 = 1.0 + alpha_p * gain_pres
    b1 = -2.0 * np.cos(w_p)
    b2 = 1.0 - alpha_p * gain_pres
    a0 = 1.0 + alpha_p / gain_pres
    a1 = -2.0 * np.cos(w_p)
    a2 = 1.0 - alpha_p / gain_pres
    b_p = np.array([b0, b1, b2]) / a0
    a_p = np.array([a0, a1, a2]) / a0
    current = scipy.signal.filtfilt(b_p, a_p, current)

    # 10 kHz High Shelf (Air)
    if air_shelf_db != 0.0 and sample_rate > 22000:
        f_air = 10000.0
        gain_air = 10.0 ** (air_shelf_db / 40.0)
        w_a = 2.0 * np.pi * f_air / sample_rate
        alpha_a = np.sin(w_a) / 2.0
        b0_a = gain_air * (
            (gain_air + 1.0) + (gain_air - 1.0) * np.cos(w_a) + 2.0 * np.sqrt(gain_air) * alpha_a
        )
        b1_a = -2.0 * gain_air * ((gain_air - 1.0) + (gain_air + 1.0) * np.cos(w_a))
        b2_a = gain_air * (
            (gain_air + 1.0) + (gain_air - 1.0) * np.cos(w_a) - 2.0 * np.sqrt(gain_air) * alpha_a
        )
        a0_a = (gain_air + 1.0) - (gain_air - 1.0) * np.cos(w_a) + 2.0 * np.sqrt(gain_air) * alpha_a
        a1_a = 2.0 * ((gain_air - 1.0) - (gain_air + 1.0) * np.cos(w_a))
        a2_a = (gain_air + 1.0) - (gain_air - 1.0) * np.cos(w_a) - 2.0 * np.sqrt(gain_air) * alpha_a
        b_air = np.array([b0_a, b1_a, b2_a]) / a0_a
        a_air = np.array([a0_a, a1_a, a2_a]) / a0_a
        current = scipy.signal.filtfilt(b_air, a_air, current)

    return np.ascontiguousarray(current, dtype=np.float32)
