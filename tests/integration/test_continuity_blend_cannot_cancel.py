"""The continuity fade's safety argument, pinned against the real cores.

:mod:`hawavoclean.policy.continuity` fades an enhanced unit back to its own
original audio at a forced cut. That is safe for a phase-coherent core because
the blend is the same signal with its enhancement *residual* scaled — nothing
can cancel. It is NOT strictly safe for an incoherent core, where the blend is
a real crossfade of two renderings, and the module says so with numbers.

Those numbers are the argument. This test is what keeps them true: it runs both
shipped cores on real speech and measures how far the ``w = 0.5`` blend can
fall below ``min(original, enhanced)`` in any 1/6-octave band of any 30 ms
window — a floor a coherent blend cannot cross at all.

A core swap that made the fade meaningfully destructive would land here.
"""

import warnings
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.alignment.delay import estimate_gcc_phat_delay
from hawavoclean.enhancement.factory import resolve_core

FIXTURE = "tests/fixtures/sample_sorani_podcast.wav"
SR = 48000
WIN = 1440  # the 30 ms fade window itself
HOP = 720
#: Bands more than this far below the window's peak are noise floor, not
#: content: a 60 dB-down band can "cancel" by any amount and no one hears it.
BAND_FLOOR_DB = 25.0
#: A window where one rendering is far louder than the other is a level change,
#: not cancellation, and would swamp the measurement.
LEVEL_MATCH_DB = 3.0


def _sixth_octave_edges(
    lo: float = 100.0, hi: float = 12000.0
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ratio = 2 ** (1 / 6)
    edges = [lo]
    while edges[-1] * ratio < hi:
        edges.append(edges[-1] * ratio)
    return np.asarray(edges, dtype=np.float64)


def _band_energy(
    x: np.ndarray[Any, np.dtype[np.float32]],
    freqs: np.ndarray[Any, np.dtype[np.float64]],
    edges: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)).astype(np.float32))) ** 2
    return np.asarray(
        [
            mag[(freqs >= lo) & (freqs < hi)].sum()
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ],
        dtype=np.float64,
    )


def _blend_dips_db(
    core_id: str, phase_coherent: bool, wave: np.ndarray[Any, np.dtype[np.float32]]
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Per-window worst band dip of the midpoint blend below min(orig, enh),
    and each window's level in dBFS."""
    registration = resolve_core(core_id)
    core = registration.enhancer_class(
        core_id=core_id, sample_rate=SR, phase_coherent=phase_coherent
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enhanced = core.enhance(wave, SR).waveform
    finally:
        if hasattr(core, "close"):
            core.close()

    n = min(len(enhanced), len(wave))
    # Align exactly as the pipeline does before the guard sees the candidate.
    enhanced = estimate_gcc_phat_delay(
        wave[:n], enhanced[:n], SR, max_delay_ms=20.0
    ).aligned_candidate
    n = min(len(enhanced), len(wave))
    original, enhanced = wave[:n], enhanced[:n]

    edges = _sixth_octave_edges()
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    dips: list[float] = []
    levels: list[float] = []
    for start in range(0, n - WIN, HOP):
        o = original[start : start + WIN]
        e = enhanced[start : start + WIN]
        rms_o = float(np.sqrt(np.mean(o.astype(np.float64) ** 2)))
        rms_e = float(np.sqrt(np.mean(e.astype(np.float64) ** 2)))
        if rms_o < 1e-5 or rms_e < 1e-5:
            continue
        if abs(20 * np.log10(rms_o / rms_e)) > LEVEL_MATCH_DB:
            continue
        mid = (0.5 * o + 0.5 * e).astype(np.float32)
        b_o, b_e, b_mid = (_band_energy(v, freqs, edges) for v in (o, e, mid))
        floor = np.minimum(b_o, b_e)
        peak = max(b_o.max(), b_e.max())
        keep = (floor > 0) & (np.maximum(b_o, b_e) > peak * 10 ** (-BAND_FLOOR_DB / 10))
        if not keep.any():
            continue
        dips.append(float(np.min(10 * np.log10(b_mid[keep] / floor[keep]))))
        levels.append(20 * np.log10(rms_o))
    assert len(dips) > 100, f"only {len(dips)} usable windows; the fixture changed"
    return np.asarray(dips), np.asarray(levels)


@pytest.fixture(scope="module")
def speech() -> np.ndarray[Any, np.dtype[np.float32]]:
    data, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    assert sr == SR, f"fixture is {sr} Hz"
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)


@pytest.mark.integration
def test_a_coherent_core_blend_cannot_cancel(
    speech: np.ndarray[Any, np.dtype[np.float32]],
) -> None:
    """The production core's blend is the original plus a scaled residual, so
    it cannot fall below both renderings. Measured worst dip: -0.02 dB."""
    dips, _ = _blend_dips_db("wiener-dd-48k-v1", True, speech)
    assert dips.min() > -0.5, (
        f"the coherent blend fell {dips.min():.2f} dB below min(original, enhanced); "
        "the continuity fade's safety argument for this core rests on it not doing that"
    )


@pytest.mark.integration
def test_an_incoherent_core_blend_cancels_but_stays_bounded(
    speech: np.ndarray[Any, np.dtype[np.float32]],
) -> None:
    """The studio core declares phase_coherent = false, so its blend IS a
    crossfade and does cancel. The fade is applied anyway because the
    alternative is losing the whole unit — but only while the cost stays this
    small. Measured: median +0.07 dB, worst -1.87 dB above -40 dBFS."""
    dips, levels = _blend_dips_db("studio-dfn3-48k-v1", False, speech)
    audible = dips[levels >= -40.0]
    assert len(audible) > 100, "too few audible windows to judge"

    assert np.median(dips) > -0.5, (
        f"the typical studio window now loses {np.median(dips):.2f} dB to the blend; "
        "the fade is only justified while the median is ~0"
    )
    assert audible.min() > -4.0, (
        f"the studio blend now cancels by {audible.min():.2f} dB in an audible window. "
        "A core this uncorrelated with its input makes the fade destructive: gate the "
        "taper on config.enhancement.phase_coherent and fall back to the revert"
    )
