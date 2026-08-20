"""Reference corpus for the finishing chain's tonal restoration.

Two kinds of reference live here.

CONTROLS THAT MUST NOT MOVE. `natural_voice` is a byte-for-byte copy of the
generator pinned by the 3.1.1 tonal-transparency gate
(`test_finishing_tonal_transparency.py`), reproduced rather than imported so
that gate stays independent of this one. `speech_like` is a more honest
speech proxy: voiced harmonics with a realistic rolloff, fricative bursts,
and REAL pauses — the pinned fixture has none, and a signal that never goes
quiet has no measurable noise floor.

DEFECTS THAT MUST BE CORRECTED, each derived from the same `speech_like`
base so that a before/after comparison isolates the defect: brick-wall
lowpassed (nothing above the cut — must be REFUSED, not amplified), softly
muffled (steep tilt, content survives — must be lifted), boomy (a low-end
resonance — must be cut), thin/harsh (bass gone, presence in surplus — must
never be thinned further), and near-silent.
"""

from typing import Any

import numpy as np
import scipy.signal

SR = 48000


def natural_voice(seconds: float = 6.0, f0: float = 130.0) -> np.ndarray[Any, np.dtype[np.float32]]:
    """The pinned 3.1.1 transparency fixture: a plausible male voice spectrum."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    for h in range(1, 60):
        f = f0 * h
        if f > 12000:
            break
        amp = 1.0 / h if f < 500 else (500.0 / f) ** 1.0 / h
        x += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * t) ** 2
    x = x * env
    x = x / np.max(np.abs(x)) * 0.3
    x += 0.002 * rng.standard_normal(len(t))
    return np.asarray(x, dtype=np.float32)


def speech_like(
    seconds: float = 8.0,
    f0: float = 120.0,
    seed: int = 1,
    harmonic_tilt_db_per_octave: float = -9.0,
    fricative_gain: float = 0.11,
    noise_floor: float = 1e-4,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Voiced harmonics + fricative bursts + real pauses, deterministic per seed."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR

    syllables = np.zeros(n)
    i = 0
    while i < n:
        length = int(SR * rng.uniform(0.12, 0.30))
        gap = int(SR * rng.uniform(0.05, 0.45))
        env = np.hanning(max(2, length))
        k = max(0, min(len(env), n - i))
        syllables[i : i + k] += env[:k]
        i += length + gap
    syllables = np.clip(syllables, 0.0, 1.0)

    voiced = np.zeros(n)
    for h in range(1, 140):
        f = f0 * h
        if f > 18000:
            break
        amp = (f / f0) ** (harmonic_tilt_db_per_octave / 6.0206)
        voiced += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    voiced /= np.max(np.abs(voiced))

    sos = scipy.signal.butter(4, [2500, 11000], btype="bandpass", fs=SR, output="sos")
    fricative = scipy.signal.sosfilt(sos, rng.standard_normal(n))
    fricative = np.asarray(fricative) / (float(np.std(fricative)) + 1e-12)
    bursts = np.zeros(n)
    i = int(SR * 0.05)
    while i < n:
        length = int(SR * rng.uniform(0.05, 0.10))
        env = np.hanning(max(2, length))
        k = max(0, min(len(env), n - i))
        bursts[i : i + k] += env[:k]
        i += length + int(SR * rng.uniform(0.25, 0.70))

    x = voiced * syllables + fricative_gain * fricative * np.clip(bursts, 0.0, 1.0)
    x = x / np.max(np.abs(x)) * 0.35
    x = x + noise_floor * rng.standard_normal(n)
    return np.asarray(x, dtype=np.float32)


def _f32(x: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.ascontiguousarray(np.asarray(x), dtype=np.float32)


def brickwall_lowpassed(
    base: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """6th-order lowpass at 1.1 kHz: above the cut there is only dither."""
    src = speech_like() if base is None else base
    sos = scipy.signal.butter(6, 1100, btype="lowpass", fs=SR, output="sos")
    cut = scipy.signal.sosfiltfilt(sos, src)
    cut = cut + 3e-4 * np.random.default_rng(9).standard_normal(len(src))
    return _f32(cut)


def softly_muffled(
    base: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """A steep HF tilt with no brick wall: consonants survive, weakly.

    The distinction from `brickwall_lowpassed` is the whole point of gate 1.
    Both are muffled; only this one still has speech dynamics above 1.5 kHz,
    so only this one may be lifted.
    """
    src = speech_like() if base is None else base
    sos = scipy.signal.butter(2, 700, btype="lowpass", fs=SR, output="sos")
    return _f32(0.06 * src + 3.0 * scipy.signal.sosfiltfilt(sos, src))


def boomy(
    base: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Proximity effect / resonant room: a big 100-300 Hz bump."""
    src = speech_like() if base is None else base
    sos = scipy.signal.butter(2, [100, 300], btype="bandpass", fs=SR, output="sos")
    return _f32(src + 2.6 * scipy.signal.sosfiltfilt(sos, src))


def thin_harsh() -> np.ndarray[Any, np.dtype[np.float32]]:
    """Bass stripped, presence in surplus — the 3.1.1 failure mode, as input."""
    src = speech_like(harmonic_tilt_db_per_octave=-5.0, fricative_gain=0.5, seed=4)
    sos = scipy.signal.butter(2, 220, btype="highpass", fs=SR, output="sos")
    thin = scipy.signal.sosfiltfilt(sos, src)
    return _f32(thin / np.max(np.abs(thin)) * 0.35)


def near_silent(
    base: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Below any sane speech level: nothing to balance."""
    src = speech_like() if base is None else base
    return _f32(src * 2e-5)


def synthetic_corpus() -> list[tuple[str, np.ndarray[Any, np.dtype[np.float32]]]]:
    """Every synthetic control, in report order."""
    base = speech_like()
    return [
        ("fixture natural_voice", natural_voice()),
        ("synth flat speech", base),
        ("synth brickwall lowpass", brickwall_lowpassed(base)),
        ("synth softly muffled", softly_muffled(base)),
        ("synth boomy", boomy(base)),
        ("synth thin/harsh", thin_harsh()),
        ("synth near-silent", near_silent(base)),
    ]


def approved_recording_profile() -> np.ndarray[Any, np.dtype[np.float32]]:
    """A committed stand-in for the real recording the user approved the sound of.

    "Flute 09" is 3 MB of user audio under a gitignored directory, so it cannot
    be the permanent gate — but it is the reference that decides whether this
    feature is a fix or a second regression. What matters about it is its
    MEASURED PROFILE, and that is reproducible: shaping `speech_like` with this
    fixed three-filter chain lands within 0.2 dB of the real recording in every
    analysis band.

        band        this fixture   Flute 09 (real)
        90-300 Hz         +6.8            +7.0
        1.5-3 kHz        -27.8           -27.7
        3-6 kHz          -33.3           -33.4

    That profile is presence-shy by any textbook speech spectrum and it still
    must receive 0.0 dB, because a real listener signed off on how it sounds.
    If a change to the target curve starts correcting this fixture, it would
    have re-voiced a recording the user already accepted.
    """
    from hawavoclean.finishing.eq import _low_shelf_biquad, _peaking_biquad

    y = np.asarray(speech_like(), dtype=np.float64)
    for b, a in (
        _low_shelf_biquad(SR, 300.0, -15.5),
        _peaking_biquad(SR, 2100.0, 1.30, -17.0),
        _peaking_biquad(SR, 4000.0, 1.70, -11.7),
    ):
        y = scipy.signal.filtfilt(b, a, y)
    return _f32(y / np.max(np.abs(y)) * 0.35)


def presence_starved_profile() -> np.ndarray[Any, np.dtype[np.float32]]:
    """The approved profile with the measured deficit of the reported file applied.

    The two real recordings behind this feature are within 1.4 dB of each other
    from 90 Hz to 3 kHz and 11 dB apart above it. So this fixture is literally
    `approved_recording_profile` minus that gap: same source, same syllables,
    same everything below 3 kHz, an 11 dB high-shelf cut above it. It lands at
    3-6 kHz = -44.3 dB against the real file's -44.7.

    The pair is the whole argument. A correction driven by anything below 3 kHz
    cannot tell them apart and would move both; only the band where they
    actually differ can fix one and leave the other alone.
    """
    base = np.asarray(approved_recording_profile(), dtype=np.float64)
    sos = scipy.signal.butter(2, 3000, btype="highpass", fs=SR, output="sos")
    shelved = base - 0.90 * scipy.signal.sosfiltfilt(sos, base)
    return _f32(shelved / np.max(np.abs(shelved)) * 0.35)
