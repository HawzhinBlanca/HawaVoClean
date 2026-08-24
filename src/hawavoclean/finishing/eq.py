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


# ---------------------------------------------------------------------------
# Tonal restoration filter bank
# ---------------------------------------------------------------------------
#
# THE TRAP IN `apply_speech_eq` ABOVE, STATED SO IT IS NOT REPEATED. That
# function applies each biquad with `filtfilt`, which runs the filter forwards
# and then backwards. The magnitude response is therefore SQUARED and the
# achieved gain is twice the dB that was asked for: -2.0 dB requested measures
# -3.97 dB, -6.0 dB measures -11.92 dB. Its callers are calibrated around that
# (see the 3.1.1 comment about "~1.5x its nominal setting") and are left alone.
# The functions below design for the gain they are ASKED for: each biquad is
# built for half the dB, `filtfilt` doubles it back, and `tonal_filter_response`
# then measures the finished cascade so the bound is verified rather than
# assumed.
#
# Zero-phase (`filtfilt`) is kept deliberately: a restorative EQ of up to 12 dB
# built from minimum-phase biquads smears consonant transients across the
# crossover, and Guard B's envelope-correlation and timing checks are the
# things that would notice. Forwards-and-backwards has no group delay at all.

TONAL_LOW_SHELF_HZ = 300.0
TONAL_PRESENCE_HZ = 2100.0
TONAL_PRESENCE_Q = 1.30
TONAL_BRILLIANCE_HZ = 4000.0
TONAL_BRILLIANCE_Q = 1.70

# The analysis bands the correction is MEASURED in (detect.TILT_*_BAND_HZ),
# repeated here so the filter bank can check what it actually delivers into
# them. Body first: every level is relative to it.
# WHY THE BELLS ARE LIFT-ONLY, AND WHAT THAT COSTS.
# The two bells overlap, so a lift in one lands partly in the other: a +8 dB
# brilliance move adds +2.4 dB to the presence band. Letting the solver pull a
# bell BELOW flat would cancel that — and it was tried. It is worse. Asked for
# a pure bass cut, a solver with that freedom pulled BOTH bells down 2.7 dB to
# hold their relative numbers at zero, which is a bass cut plus a treble cut:
# dull, mid-forward, and precisely the 3.1.1 regression this feature must not
# become. It is also the wrong shape. A restorative curve that lifts 3-6 kHz
# has to lift the region just below it; carving a notch at 2.1 kHz to make a
# band-average read 0.0 would trade a smooth tilt for a lumpy one, and lumpy
# is what a listener hears. So the bells never cut, the spill is left in, and
# `test_solver_delivers_the_requested_band_move` bounds it: an untargeted band
# never receives more than half of what the targeted band earned, and never
# receives a cut.

_TONAL_BODY_BAND = (300.0, 1000.0)
_TONAL_TARGET_BANDS = ((90.0, 300.0), (1500.0, 3000.0), (3000.0, 6000.0))


def _peaking_biquad(
    sample_rate: float, freq_hz: float, q: float, gain_db: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """RBJ peaking biquad for HALF the requested gain (filtfilt doubles it)."""
    amp = 10.0 ** (gain_db / 80.0)  # /40 for RBJ, /2 again for the two passes
    w0 = 2.0 * np.pi * freq_hz / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b = np.array([1.0 + alpha * amp, -2.0 * cos_w0, 1.0 - alpha * amp])
    a = np.array([1.0 + alpha / amp, -2.0 * cos_w0, 1.0 - alpha / amp])
    return b / a[0], a / a[0]


def _low_shelf_biquad(
    sample_rate: float, freq_hz: float, gain_db: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """RBJ low-shelf biquad for HALF the requested gain (filtfilt doubles it)."""
    amp = 10.0 ** (gain_db / 80.0)
    w0 = 2.0 * np.pi * freq_hz / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(amp) * alpha
    b = np.array(
        [
            amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha),
            2.0 * amp * ((amp - 1.0) - (amp + 1.0) * cos_w0),
            amp * ((amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha),
        ]
    )
    a = np.array(
        [
            (amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha,
            -2.0 * ((amp - 1.0) + (amp + 1.0) * cos_w0),
            (amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha,
        ]
    )
    return b / a[0], a / a[0]


def _tonal_sections(
    sample_rate: int,
    low_shelf_db: float,
    presence_db: float,
    brilliance_db: float,
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    """Biquads for the requested move, skipping bands at or above Nyquist."""
    # Below Nyquist with margin: a biquad designed at or above it is not a
    # filter, and low sample rates are a real input here (8 kHz is supported).
    highest_usable_hz = sample_rate * 0.45
    sections: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    if abs(low_shelf_db) >= 0.05 and highest_usable_hz > TONAL_LOW_SHELF_HZ:
        sections.append(_low_shelf_biquad(sample_rate, TONAL_LOW_SHELF_HZ, low_shelf_db))
    if presence_db >= 0.05 and highest_usable_hz > TONAL_PRESENCE_HZ:
        sections.append(
            _peaking_biquad(sample_rate, TONAL_PRESENCE_HZ, TONAL_PRESENCE_Q, presence_db)
        )
    if brilliance_db >= 0.05 and highest_usable_hz > TONAL_BRILLIANCE_HZ:
        sections.append(
            _peaking_biquad(sample_rate, TONAL_BRILLIANCE_HZ, TONAL_BRILLIANCE_Q, brilliance_db)
        )
    return sections


def tonal_filter_response(
    sample_rate: int,
    low_shelf_db: float,
    presence_db: float,
    brilliance_db: float,
    num_points: int = 1024,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Achieved magnitude response (Hz, dB) of the finished two-pass cascade.

    This is the bound, measured. `apply_tonal_restoration` calls it on every
    invocation and refuses to apply a cascade whose peak exceeds what was
    asked for — so a future edit to a filter constant cannot quietly turn a
    +8 dB restoration into a +16 dB one the way `filtfilt` already did once.
    """
    sections = _tonal_sections(sample_rate, low_shelf_db, presence_db, brilliance_db)
    freqs = np.linspace(0.0, sample_rate / 2.0, num_points)
    total_db = np.zeros(num_points)
    for b, a in sections:
        _, h = scipy.signal.freqz(b, a, worN=freqs, fs=sample_rate)
        # Two passes (forwards + backwards): the magnitude is squared, so the
        # dB doubles. That is the whole reason each section was designed for
        # half its gain.
        total_db += 2.0 * 20.0 * np.log10(np.abs(h) + 1e-12)
    return freqs, total_db


def achieved_band_gains_db(
    sample_rate: int,
    low_shelf_db: float,
    presence_db: float,
    brilliance_db: float,
) -> tuple[float, float, float]:
    """What this cascade actually delivers into each analysis band, relative to body.

    The detector measures every band against the 300-1000 Hz body, so what
    matters is not a filter's peak gain but the difference between what it does
    to its own band and what it does to the body underneath. A low shelf at
    300 Hz asked for -6 dB moves the 90-300 band -4.7 dB and the body -0.5 dB:
    a RELATIVE -4.2 dB, two thirds of the request. Band levels are averaged as
    power and then converted, exactly as `detect._band_stats` does.
    """
    freqs, response_db = tonal_filter_response(
        sample_rate, low_shelf_db, presence_db, brilliance_db
    )
    power = 10.0 ** (response_db / 10.0)

    def mean_band_db(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs < min(high_hz, sample_rate / 2.0))
        if not np.any(mask):
            return 0.0
        return float(10.0 * np.log10(float(np.mean(power[mask])) + 1e-30))

    body = mean_band_db(*_TONAL_BODY_BAND)
    return (
        mean_band_db(*_TONAL_TARGET_BANDS[0]) - body,
        mean_band_db(*_TONAL_TARGET_BANDS[1]) - body,
        mean_band_db(*_TONAL_TARGET_BANDS[2]) - body,
    )


def solve_tonal_gains(
    sample_rate: int,
    want_low_db: float,
    want_presence_db: float,
    want_brilliance_db: float,
    max_low_cut_db: float,
    max_low_lift_db: float,
    max_presence_db: float,
    max_brilliance_db: float,
) -> tuple[float, float, float]:
    """Filter gains that deliver the requested per-band move, clamped to the caps.

    Three overlapping sections cannot each act on one band alone: a +12 dB
    brilliance bell at 4 kHz also adds ~2.9 dB to 1.5-3 kHz, and a presence
    bell returns the favour. Asking each filter for its band's raw deficit
    therefore over-delivers on whichever band sits between the two. This runs a
    few damped fixed-point steps against the ANALYTIC response — no audio, no
    randomness, identical on every machine — so the shipped move matches the
    measured deficit instead of the sum of two spills.

    Every step is clamped to the caps, so the solve can neither run away nor
    talk its way past a bound; if the caps bind, the result under-corrects on
    purpose.
    """
    # Every cap is a MAGNITUDE, and the cut one is negated below. Passing it
    # already-negative -- the obvious reading of "max cut" -- makes ``lower``
    # +max and ``upper`` +max, and ``np.clip`` against a collapsed range
    # returns that single value without complaint: the low shelf then pins to
    # full lift on every input, including audio measured as needing a cut, and
    # nothing anywhere says so. Found by making the mistake.
    caps = {
        "max_low_cut_db": max_low_cut_db,
        "max_low_lift_db": max_low_lift_db,
        "max_presence_db": max_presence_db,
        "max_brilliance_db": max_brilliance_db,
    }
    negative = sorted(name for name, value in caps.items() if value < 0.0)
    if negative:
        raise ValueError(
            f"tonal gain caps are magnitudes and must be >= 0; got negative {', '.join(negative)}. "
            "max_low_cut_db=6.0 means 'cut by at most 6 dB'."
        )

    want = np.array([want_low_db, want_presence_db, want_brilliance_db], dtype=np.float64)
    gains = want.copy()
    lower = np.array([-max_low_cut_db, 0.0, 0.0])
    upper = np.array([max_low_lift_db, max_presence_db, max_brilliance_db])
    gains = np.clip(gains, lower, upper)
    for _ in range(6):
        achieved = np.array(achieved_band_gains_db(sample_rate, *gains), dtype=np.float64)
        residual = want - achieved
        if float(np.max(np.abs(residual))) < 0.05:
            break
        gains = np.clip(gains + 0.9 * residual, lower, upper)
    return float(gains[0]), float(gains[1]), float(gains[2])


def apply_tonal_restoration(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    low_shelf_db: float,
    presence_db: float,
    brilliance_db: float,
    max_abs_gain_db: float = 14.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply the measured tonal correction: low shelf, presence bell, brilliance bell.

    Returns the input unchanged when the correction is trivial, when the
    designed cascade would exceed `max_abs_gain_db` anywhere (a design-time
    bug, refused rather than shipped), or when the filtered result is not
    finite. A lift that would push an already-hot unit into new clipping is
    scaled back so the unit's peak is preserved: Guard B counts newly clipped
    samples as damage, and it would be right to.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim > 1:
        raise ValueError("apply_tonal_restoration expects a single channel")
    if len(x) < 128:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    sections = _tonal_sections(sample_rate, low_shelf_db, presence_db, brilliance_db)
    if not sections:
        return np.ascontiguousarray(waveform, dtype=np.float32)

    _, response_db = tonal_filter_response(sample_rate, low_shelf_db, presence_db, brilliance_db)
    if float(np.max(np.abs(response_db))) > max_abs_gain_db + 0.5:
        # The cascade does not do what it was asked to do. Ship the input.
        return np.ascontiguousarray(waveform, dtype=np.float32)

    current = x
    for b, a in sections:
        current = scipy.signal.filtfilt(b, a, current)
    if not np.all(np.isfinite(current)):
        return np.ascontiguousarray(waveform, dtype=np.float32)

    in_peak = float(np.max(np.abs(x)))
    out_peak = float(np.max(np.abs(current)))
    ceiling = max(in_peak, 0.999)
    if out_peak > ceiling:
        current = current * (ceiling / out_peak)
    return np.ascontiguousarray(current, dtype=np.float32)
