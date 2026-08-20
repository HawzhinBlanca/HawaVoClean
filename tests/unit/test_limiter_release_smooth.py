"""The run-walking release smoother must equal the per-sample loop, bit for bit.

``_release_smooth`` skips whole runs of the gain envelope that the scalar
recurrence would have left untouched. That shortcut is only legitimate if the
result is *identical* — not close — to running the recurrence over every
sample, including on the ugly inputs: NaN, infinities, zeros, signed zeros,
values above unity, and envelopes that are one unbroken run.
"""

from typing import Any

import numpy as np
import pytest

from hawavoclean.finishing import limiter
from hawavoclean.finishing.limiter import _release_smooth

FloatArray = np.ndarray[Any, np.dtype[np.float32]]


def _scalar_reference(gain: FloatArray, release_coeff: float) -> FloatArray:
    """The original per-sample loop, kept verbatim as the oracle."""
    out = gain.copy()
    current_g = 1.0
    for i in range(out.shape[0]):
        target = float(out[i])
        current_g = target if target < current_g else target + release_coeff * (current_g - target)
        out[i] = current_g
    return out


def _assert_bit_identical(gain: FloatArray, release_coeff: float, label: str) -> None:
    expected = _scalar_reference(gain, release_coeff)
    got = gain.copy()
    _release_smooth(got, release_coeff)
    assert np.array_equal(expected.view(np.uint32), got.view(np.uint32)), label


COEFF = float(np.exp(-1.0 / (48000 * 0.05)))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "gain"),
    [
        ("empty", np.zeros(0, dtype=np.float32)),
        ("single", np.array([0.5], dtype=np.float32)),
        ("all unity", np.ones(50_000, dtype=np.float32)),
        ("all zero", np.zeros(1_000, dtype=np.float32)),
        ("signed zeros", np.array([1.0, -0.0, 0.0, -0.0, 1.0, 1.0], dtype=np.float32)),
        ("nan", np.array([1.0, np.nan, 1.0, 0.5, np.nan, np.nan, 1.0], dtype=np.float32)),
        ("inf", np.array([1.0, np.inf, 1.0, 0.5], dtype=np.float32)),
        ("above unity", np.array([0.5, 2.0, 2.0, 2.0, 1.0], dtype=np.float32)),
        ("denormal", np.full(5_000, 1e-40, dtype=np.float32)),
        ("descending ramp", np.linspace(1.0, 0.1, 20_000).astype(np.float32)),
        ("ascending ramp", np.linspace(0.1, 1.0, 20_000).astype(np.float32)),
    ],
)
def test_release_smooth_matches_the_scalar_loop(label: str, gain: FloatArray) -> None:
    _assert_bit_identical(gain, COEFF, label)


@pytest.mark.unit
def test_release_smooth_matches_on_a_realistic_envelope() -> None:
    """Long unity run punctuated by dips — the shape a real limiter produces."""
    gain = np.ones(200_000, dtype=np.float32)
    for at, depth in ((1_000, 0.62), (50_000, 0.85), (120_000, 0.75), (199_999, 0.9)):
        gain[at : at + 240] = depth
    _assert_bit_identical(gain, COEFF, "realistic")


@pytest.mark.unit
def test_release_smooth_matches_on_random_envelopes() -> None:
    """Randomised shapes and release coefficients, including dense run changes."""
    rng = np.random.default_rng(1234)
    for trial in range(40):
        n = int(rng.integers(1, 4_000))
        gain = (rng.random(n) * float(rng.choice([1.0, 0.01, 2.0]))).astype(np.float32)
        if trial % 3 == 0:  # blocky: long identical runs
            gain = np.repeat(gain[: max(1, n // 40)], 40)[:n].astype(np.float32)
        _assert_bit_identical(
            np.ascontiguousarray(gain), float(rng.random()), f"random trial {trial}"
        )


@pytest.mark.unit
def test_release_smooth_is_applied_in_place() -> None:
    gain = np.array([1.0, 0.5, 1.0, 1.0], dtype=np.float32)
    before = gain.copy()
    _release_smooth(gain, COEFF)
    assert not np.array_equal(before, gain)
    assert np.array_equal(gain, _scalar_reference(before, COEFF))


@pytest.mark.unit
def test_both_run_walk_and_dense_fallback_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dense-envelope fallback and the run walk must produce the same array.

    A gain envelope with almost no repeated values costs more to walk by runs
    than by samples, so the smoother switches to the plain loop. Forcing each
    path over the same input proves the switch is a speed decision only.
    """
    rng = np.random.default_rng(24)
    for label, gain in (
        ("dense", (0.2 + 0.8 * rng.random(20_000)).astype(np.float32)),
        ("blocky", np.repeat((0.2 + 0.8 * rng.random(400)).astype(np.float32), 50)),
    ):
        expected = _scalar_reference(gain, COEFF)
        for threshold in (0, 5, 10**9):  # never / measured / always fall back
            monkeypatch.setattr(limiter, "MIN_RUN_LENGTH_FOR_RUN_WALK", threshold)
            got = gain.copy()
            _release_smooth(got, COEFF)
            assert np.array_equal(expected.view(np.uint32), got.view(np.uint32)), (
                label,
                threshold,
            )
