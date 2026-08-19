"""Decay-gated late-reverb suppression: tightens the decay after phrases,
leaves the voice itself alone, and never acts on dry signals."""

from typing import Any

import numpy as np
import scipy.signal

from hawavoclean.finishing.dereverb import suppress_late_reverb

SR = 48000


PHRASE_PERIOD_S = 1.0
PHRASE_ON_S = 0.45


def _dry_phrases(seconds: float = 6.0) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Phrases with a 20 ms raised-cosine on/off ramp (no clicks), known
    offsets at k*PHRASE_PERIOD_S + PHRASE_ON_S."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    for h in range(1, 30):
        x += (0.3 / h) * np.sin(2 * np.pi * 160 * h * t)
    phase = t % PHRASE_PERIOD_S
    ramp = 0.02
    gate = np.clip(phase / ramp, 0, 1) * np.clip((PHRASE_ON_S - phase) / ramp, 0, 1)
    x = x * gate * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t))
    return np.asarray(x + 0.0005 * rng.standard_normal(len(t)), dtype=np.float32)


def _add_reverb(
    x: np.ndarray[Any, Any], rt60: float = 0.5
) -> np.ndarray[Any, np.dtype[np.float32]]:
    rng = np.random.default_rng(1)
    n = int(SR * rt60 * 1.2)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir**2))
    y = scipy.signal.fftconvolve(x, ir)[: len(x)]
    return np.asarray(y / np.max(np.abs(y)) * np.max(np.abs(x)), dtype=np.float32)


def _decay_after_offsets(x: np.ndarray[Any, Any]) -> tuple[float, float]:
    """Median level 50 ms and 100 ms after each KNOWN phrase offset, relative
    to the level just before the offset."""
    n = SR // 100
    m = len(x) // n
    fr = 20 * np.log10(np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1)) + 1e-9)
    curves = []
    k = 0
    while True:
        off = int((k * PHRASE_PERIOD_S + PHRASE_ON_S) * 100)  # frame index of offset
        if off + 12 >= m:
            break
        ref = fr[off - 2]
        curves.append(fr[off : off + 12] - ref)
        k += 1
    c = np.median(np.array(curves), axis=0)
    return float(c[5]), float(c[10])


def test_tightens_decay_after_phrases_on_reverberant_speech() -> None:
    dry = _dry_phrases()
    wet = _add_reverb(dry)
    out = suppress_late_reverb(wet, SR, rt60_s=0.4, floor_db=-10.0, onset_protect_db=3.0)
    w50, w100 = _decay_after_offsets(wet)
    o50, o100 = _decay_after_offsets(out)
    assert o50 < w50 - 1.5, f"50 ms decay not tightened: {w50:+.1f} -> {o50:+.1f} dB"
    assert o100 < w100 - 1.5, f"100 ms decay not tightened: {w100:+.1f} -> {o100:+.1f} dB"


def test_voice_frames_are_protected() -> None:
    dry = _dry_phrases()
    wet = _add_reverb(dry)
    out = suppress_late_reverb(wet, SR, rt60_s=0.4, floor_db=-10.0, onset_protect_db=3.0)
    n = SR // 20
    m = len(wet) // n
    fw = np.sqrt(np.mean(wet[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12)
    loud = fw > np.percentile(fw, 75)
    fo = np.sqrt(np.mean(out[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12)
    change = 20 * np.log10(np.mean(fo[loud]) / np.mean(fw[loud]))
    assert change > -1.0, f"voice frames were attenuated {change:+.1f} dB"


def test_dry_voice_frames_pass_untouched() -> None:
    """On a dry signal the VOICE (frames at/near their local peak) must be
    untouched. The fixture's own 20 ms ramp-out is a decay and may be
    dimmed — that is the intended behaviour, not damage."""
    dry = _dry_phrases()
    out = suppress_late_reverb(dry, SR)
    n = SR // 100
    m = len(dry) // n
    fd = np.sqrt(np.mean(dry[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12)
    fo = np.sqrt(np.mean(out[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12)
    voice = fd > np.percentile(fd, 60)
    change_db = 20 * np.log10(np.mean(fo[voice]) / np.mean(fd[voice]))
    assert change_db > -0.5, f"dry voice frames changed {change_db:+.2f} dB"


def test_short_input_returned_unchanged() -> None:
    x = np.ones(500, np.float32)
    assert np.array_equal(suppress_late_reverb(x, SR), x)
