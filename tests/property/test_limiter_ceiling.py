"""True-peak limiter must enforce its ceiling exactly, without flat-topping.

Measurement here is independent: 8x polyphase oversampling computed in the
test. The module under test must never supply its own grade.
"""

from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import resample_poly

from hawavoclean.finishing.limiter import apply_lookahead_limiter

SR = 48000
CEILING_DBTP = -1.0
CEILING_LINEAR = 10.0 ** (CEILING_DBTP / 20.0)


def _independent_true_peak(waveform: np.ndarray) -> float:
    """8x oversampled true peak, computed here, in float64."""
    over = resample_poly(waveform.astype(np.float64), up=8, down=1, axis=-1)
    return float(np.max(np.abs(over)))


def _make_signal(
    base_gain: float, n_transients: int, transient_gain: float, freq: float, seed: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    rng = np.random.default_rng(seed)
    n = SR  # 1 second
    t = np.arange(n) / SR
    x = base_gain * np.sin(2 * np.pi * freq * t)
    x += 0.02 * rng.standard_normal(n)
    positions = rng.integers(SR // 20, n - 100, size=n_transients)
    for p in positions:
        x[p : p + 60] += transient_gain * np.hanning(60)
    return x.astype(np.float32)


@settings(max_examples=50, deadline=None)
@given(
    base_gain=st.floats(0.05, 0.9),
    n_transients=st.integers(1, 24),
    transient_gain=st.floats(0.5, 4.0),  # up to ~12 dB over ceiling
    freq=st.floats(60.0, 20000.0),
    seed=st.integers(0, 2**31 - 1),
)
def test_limiter_enforces_ceiling_without_flat_tops(
    base_gain: float, n_transients: int, transient_gain: float, freq: float, seed: int
) -> None:
    x = _make_signal(base_gain, n_transients, transient_gain, freq, seed)[None, :]

    res = apply_lookahead_limiter(x, SR, ceiling_dbtp=CEILING_DBTP)
    y = res.limited_waveform

    # 1. True peak at or under the ceiling — no tolerance, independent measure.
    tp = _independent_true_peak(y[0])
    assert tp <= CEILING_LINEAR, (
        f"true peak {20 * np.log10(tp):+.3f} dBTP exceeds ceiling {CEILING_DBTP:.2f} dBTP"
    )

    # 2. No flat-top runs: >=3 consecutive samples pinned at +/- ceiling is the
    # signature of hard clipping, not limiting.
    at_ceiling = np.abs(np.abs(y[0]) - CEILING_LINEAR) < 1e-6
    run = 0
    max_run = 0
    for flag in at_ceiling:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)
    assert max_run < 3, f"hard-clip signature: {max_run} consecutive samples pinned at ceiling"


def test_gain_reduction_anticipates_the_peak() -> None:
    """Reduction must be underway BEFORE the transient arrives (lookahead),
    and must reach its minimum by the peak — not after it.

    Without the anticipating envelope, the ceiling could still be met by a
    crude global trim; this pins the mechanism, not just the outcome.
    """
    lookahead_ms = 5.0
    lookahead = int(SR * lookahead_ms / 1000.0)
    n = SR
    p = n // 2
    base = 0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)
    base[p : p + 60] += 2.0 * np.hanning(60)  # one isolated transient, ~7 dB over
    x = base.astype(np.float32)[None, :]

    res = apply_lookahead_limiter(x, SR, ceiling_dbtp=CEILING_DBTP, lookahead_ms=lookahead_ms)
    g = res.gain_envelope
    assert g.size == n, "limiter must expose its gain envelope"

    crest = p + 30  # the Hanning transient crests 30 samples after its onset

    # 1. On time: the gain minimum is reached AT or BEFORE the crest, never
    # after it (a shifted — rather than windowed-min — gain arrives late).
    g_min = float(np.min(g))
    assert float(np.min(g[: crest + 1])) <= g_min * 1.001, (
        "gain reached its minimum only after the transient crest"
    )
    # 2. Anticipation: reduction is well underway before the crest arrives.
    assert g[crest - 60] < 0.9, (
        f"no anticipation: gain 60 samples before the crest is {g[crest - 60]:.4f}"
    )
    # 3. Monotone non-increasing while approaching the crest.
    approach = g[crest - 100 : crest + 1]
    assert np.all(np.diff(approach) <= 1e-6), "gain rose while approaching the transient"
    assert lookahead > 0  # documents the fixture premise
