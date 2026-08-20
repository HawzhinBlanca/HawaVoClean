"""The continuity remedy is a fade, not a revert — and the fade has to be exact.

The rule these tests defend: enhanced audio must never butt against original
audio at a forced mid-speech cut. What changed is the price. Reverting the
enhanced unit satisfied the rule and cascaded — on Flute 09 one failing unit
discarded five passing ones and cost 7.23 dB of speech/floor separation. Fading
the unit's own enhancement back to the original before the joint satisfies the
same rule for 30 ms of audio.
"""

from typing import Any

import numpy as np
import pytest

from hawavoclean.config import load_config
from hawavoclean.finishing.safe_finish import safe_finish_speech_unit
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.guard.verdict import GuardVerdict
from hawavoclean.paths import profile_config_path
from hawavoclean.policy.continuity import (
    CONTINUITY_TAPER_MS,
    MAX_TAPER_FRACTION,
    apply_continuity_taper,
    resolve_source_continuity,
)
from hawavoclean.policy.decision import UnitPolicyDecision
from hawavoclean.segmentation.types import SpeechUnit

SR = 48000
TAPER_N = int(round(SR * CONTINUITY_TAPER_MS / 1000.0))
#: Comfortably longer than the fade can demand; a real unit is seconds long.
UNIT_N = SR  # 1 s


def _unit(uid: int, ch: int, start: int, end: int, forced: bool) -> SpeechUnit:
    return SpeechUnit(
        unit_id=uid,
        channel_id=ch,
        start_sample=start,
        end_sample=end,
        context_start_sample=start,
        context_end_sample=end,
        is_speech=True,
        forced_boundary=forced,
    )


def _dec(enhanced: bool, n: int) -> UnitPolicyDecision:
    return UnitPolicyDecision(
        selected_waveform=np.full(n, 0.5, dtype=np.float32),
        is_enhanced=enhanced,
        chosen_strength=1.0 if enhanced else 0.0,
        guard_verdict=GuardVerdict.PASS if enhanced else GuardVerdict.REVERT,
    )


def _chain(
    n_units: int, length: int = UNIT_N, last_enhanced: bool = False
) -> tuple[list[SpeechUnit], list[UnitPolicyDecision], list[np.ndarray[Any, np.dtype[np.float32]]]]:
    """``n_units`` same-channel units, EVERY boundary forced — the shape of a
    continuous-speech recording with no pauses to cut at, which is what made
    the old rule cascade through a whole file."""
    units = [_unit(i, 0, i * length, (i + 1) * length, forced=True) for i in range(n_units)]
    decs = [_dec(True, length) for _ in range(n_units - 1)] + [_dec(last_enhanced, length)]
    waves = [np.zeros(length, dtype=np.float32) for _ in range(n_units)]
    return units, decs, waves


# --------------------------------------------------------------- the cascade


@pytest.mark.unit
def test_one_failing_unit_no_longer_reverts_the_whole_file() -> None:
    """The regression this change exists for. Six units, every boundary forced,
    the last one rejected by the guard. Before: 0 of 6 enhanced. After: 5."""
    units, decs, waves = _chain(6)
    res = resolve_source_continuity(units, decs, waves, SR)

    assert [d.is_enhanced for d in res.decisions] == [True] * 5 + [False]
    assert res.reverted_ids == set(), "no unit should be reverted; they can all afford a fade"
    # Only the unit that actually touches original audio pays anything.
    assert res.fade_out_samples == [0, 0, 0, 0, TAPER_N, 0]
    assert res.fade_in_samples == [0] * 6


@pytest.mark.unit
def test_a_unit_between_two_originals_fades_on_both_sides() -> None:
    units = [
        _unit(0, 0, 0, UNIT_N, forced=True),
        _unit(1, 0, UNIT_N, 2 * UNIT_N, forced=True),
        _unit(2, 0, 2 * UNIT_N, 3 * UNIT_N, forced=False),
    ]
    decs = [_dec(False, UNIT_N), _dec(True, UNIT_N), _dec(False, UNIT_N)]
    waves = [np.zeros(UNIT_N, dtype=np.float32)] * 3

    res = resolve_source_continuity(units, decs, waves, SR)
    assert res.decisions[1].is_enhanced
    assert res.fade_in_samples[1] == TAPER_N
    assert res.fade_out_samples[1] == TAPER_N


@pytest.mark.unit
def test_no_fade_where_there_is_no_seam() -> None:
    """Enhanced meeting enhanced across a forced cut is not a seam, and a
    natural (pause) boundary is not a forced cut."""
    units, decs, waves = _chain(3, last_enhanced=True)
    res = resolve_source_continuity(units, decs, waves, SR)
    assert res.fade_in_samples == [0, 0, 0]
    assert res.fade_out_samples == [0, 0, 0]

    natural = [
        _unit(0, 0, 0, UNIT_N, forced=False),  # reverted, NATURAL boundary after it
        _unit(1, 0, UNIT_N, 2 * UNIT_N, forced=False),
    ]
    res2 = resolve_source_continuity(
        natural,
        [_dec(False, UNIT_N), _dec(True, UNIT_N)],
        [np.zeros(UNIT_N, dtype=np.float32)] * 2,
        SR,
    )
    assert res2.decisions[1].is_enhanced
    assert res2.fade_in_samples == [0, 0] and res2.fade_out_samples == [0, 0]


@pytest.mark.unit
def test_units_on_different_channels_are_never_neighbours() -> None:
    units = [
        _unit(0, 0, 0, UNIT_N, forced=True),  # ch0: enhanced, forced boundary
        _unit(1, 1, 0, UNIT_N, forced=False),  # ch1: reverted — a DIFFERENT channel
    ]
    res = resolve_source_continuity(
        units,
        [_dec(True, UNIT_N), _dec(False, UNIT_N)],
        [np.zeros(UNIT_N, dtype=np.float32)] * 2,
        SR,
    )
    assert res.decisions[0].is_enhanced
    assert res.fade_out_samples == [0, 0], "a unit on another channel is not adjacent in time"


# ------------------------------------------------------- fail-closed fallback


@pytest.mark.unit
def test_a_unit_too_short_to_fade_still_reverts() -> None:
    """The fade may not eat a unit. Below 4x the fade length the old remedy is
    still the right one, and the fail-closed path must survive."""
    short = int(TAPER_N / MAX_TAPER_FRACTION) - 1
    units = [
        _unit(0, 0, 0, short, forced=True),
        _unit(1, 0, short, short + UNIT_N, forced=False),
    ]
    res = resolve_source_continuity(
        units,
        [_dec(True, short), _dec(False, UNIT_N)],
        [np.zeros(short, dtype=np.float32), np.zeros(UNIT_N, dtype=np.float32)],
        SR,
    )
    assert not res.decisions[0].is_enhanced
    assert res.reverted_ids == {0}
    assert res.fade_out_samples[0] == 0
    assert "too short to carry a fade" in res.decisions[0].decision_reason


@pytest.mark.unit
def test_a_cascade_of_short_units_terminates() -> None:
    """Reverts are monotone, so the fixed point converges even when every unit
    is too short to fade and the old cascade is the only remedy available."""
    short = 8
    units = [_unit(i, 0, i * short, (i + 1) * short, forced=True) for i in range(12)]
    decs = [_dec(True, short) for _ in range(11)] + [_dec(False, short)]
    waves = [np.zeros(short, dtype=np.float32) for _ in range(12)]

    res = resolve_source_continuity(units, decs, waves, SR)
    assert not any(d.is_enhanced for d in res.decisions)
    assert res.reverted_ids == set(range(11))


@pytest.mark.unit
def test_empty_and_single_unit_runs_are_handled() -> None:
    empty = resolve_source_continuity([], [], [], SR)
    assert empty.decisions == [] and empty.fade_in_samples == []

    one = resolve_source_continuity(
        [_unit(0, 0, 0, UNIT_N, forced=True)],
        [_dec(True, UNIT_N)],
        [np.zeros(UNIT_N, dtype=np.float32)],
        SR,
    )
    assert one.decisions[0].is_enhanced
    assert one.fade_in_samples == [0] and one.fade_out_samples == [0]


# ------------------------------------------------------------- the fade itself


@pytest.mark.unit
def test_the_joint_is_the_original_recording_to_the_bit() -> None:
    """The entire point. If the outer sample is not *exactly* the original, the
    step this fade exists to remove is still there, just smaller."""
    rng = np.random.default_rng(7)
    original = rng.standard_normal(UNIT_N).astype(np.float32) * 0.1
    finished = original + rng.standard_normal(UNIT_N).astype(np.float32) * 0.02

    out = apply_continuity_taper(finished, original, TAPER_N, TAPER_N)
    assert out[0] == original[0]
    assert out[-1] == original[-1]


@pytest.mark.unit
def test_audio_outside_the_fade_windows_is_untouched() -> None:
    rng = np.random.default_rng(11)
    original = rng.standard_normal(UNIT_N).astype(np.float32) * 0.1
    finished = original + rng.standard_normal(UNIT_N).astype(np.float32) * 0.02

    out = apply_continuity_taper(finished, original, TAPER_N, TAPER_N)
    interior = slice(TAPER_N, UNIT_N - TAPER_N)
    assert np.array_equal(out[interior], finished[interior])
    assert not np.array_equal(out[:TAPER_N], finished[:TAPER_N])
    assert not np.array_equal(out[-TAPER_N:], finished[-TAPER_N:])


@pytest.mark.unit
def test_the_fade_only_scales_the_enhancement_residual() -> None:
    """Because the two renderings are phase-coherent, the blend is the original
    plus a scaled residual — it can never invert or overshoot, so no sample may
    land outside the interval the two renderings bracket."""
    rng = np.random.default_rng(13)
    original = rng.standard_normal(UNIT_N).astype(np.float32) * 0.1
    finished = original + rng.standard_normal(UNIT_N).astype(np.float32) * 0.05

    out = apply_continuity_taper(finished, original, TAPER_N, TAPER_N)
    residual_in = (finished - original).astype(np.float64)
    residual_out = (out - original).astype(np.float64)

    assert np.all(np.abs(residual_out) <= np.abs(residual_in) + 1e-9)
    same_sign = np.sign(residual_out) == np.sign(residual_in)
    assert np.all(same_sign | (residual_out == 0.0)), "the residual changed sign somewhere"


@pytest.mark.unit
def test_the_fade_weight_is_monotone_across_the_window() -> None:
    """A non-monotone weight would move the timbre back and forth inside the
    fade — worse than the step it replaces."""
    original = np.zeros(UNIT_N, dtype=np.float32)
    finished = np.ones(UNIT_N, dtype=np.float32)

    out = apply_continuity_taper(finished, original, TAPER_N, TAPER_N)
    assert np.all(np.diff(out[:TAPER_N]) >= 0.0), "fade-in is not monotone"
    assert np.all(np.diff(out[-TAPER_N:]) <= 0.0), "fade-out is not monotone"
    assert out.min() >= 0.0 and out.max() <= 1.0

    # Monotone is not enough: a hard step is monotone too, and it is WORSE than
    # the seam it replaces — it moves the whole discontinuity 15 ms upstream,
    # off the low-energy zero crossing the segmenter chose, and measured 7.6x
    # the residual of a plain hard cut. Bound the slope: a raised cosine peaks
    # at pi/2n and a linear ramp at 1/n, so 4/n admits either and refuses any
    # step or corner.
    max_slope = 4.0 / TAPER_N
    assert np.max(np.abs(np.diff(out[:TAPER_N]))) <= max_slope, "fade-in has a step in it"
    assert np.max(np.abs(np.diff(out[-TAPER_N:]))) <= max_slope, "fade-out has a step in it"


@pytest.mark.unit
def test_a_left_edge_fade_alone_touches_only_the_head() -> None:
    """A unit whose seam is on its LEFT — its predecessor across the forced cut
    shipped original — fades in and leaves its tail alone."""
    original = np.zeros(UNIT_N, dtype=np.float32)
    finished = np.ones(UNIT_N, dtype=np.float32)

    out = apply_continuity_taper(finished, original, TAPER_N, 0)
    assert out[0] == original[0]
    assert out[-1] == finished[-1], "the tail was faded, but the seam is on the head"
    assert np.array_equal(out[TAPER_N:], finished[TAPER_N:])


@pytest.mark.unit
def test_a_degenerate_one_sample_fade_ships_the_original() -> None:
    """A one-sample fade has no ramp to speak of. It must resolve toward the
    original — the safe side — rather than toward a half-weighted blend."""
    original = np.zeros(8, dtype=np.float32)
    finished = np.ones(8, dtype=np.float32)

    head = apply_continuity_taper(finished, original, 1, 0)
    assert head[0] == original[0]
    assert np.array_equal(head[1:], finished[1:])

    tail = apply_continuity_taper(finished, original, 0, 1)
    assert tail[-1] == original[-1]
    assert np.array_equal(tail[:-1], finished[:-1])


@pytest.mark.unit
def test_no_fade_requested_returns_the_input_unchanged() -> None:
    finished = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    original = np.zeros(128, dtype=np.float32)
    assert apply_continuity_taper(finished, original, 0, 0) is finished


@pytest.mark.unit
def test_overlapping_fades_are_clamped_rather_than_blended_twice() -> None:
    n = 100
    finished = np.ones(n, dtype=np.float32)
    original = np.zeros(n, dtype=np.float32)
    out = apply_continuity_taper(finished, original, 80, 80)
    assert len(out) == n
    assert np.all(np.isfinite(out))
    assert out[0] == original[0]
    assert np.all(out <= 1.0) and np.all(out >= 0.0)


@pytest.mark.unit
def test_mismatched_lengths_do_not_read_past_the_shorter_array() -> None:
    finished = np.ones(200, dtype=np.float32)
    original = np.zeros(120, dtype=np.float32)
    out = apply_continuity_taper(finished, original, 40, 40)
    assert len(out) == 200
    assert out[0] == 0.0
    assert np.array_equal(out[120:], finished[120:]), "audio past the original was rewritten"


@pytest.mark.unit
def test_the_fade_is_deterministic() -> None:
    rng = np.random.default_rng(17)
    original = rng.standard_normal(UNIT_N).astype(np.float32) * 0.1
    finished = original + rng.standard_normal(UNIT_N).astype(np.float32) * 0.02
    a = apply_continuity_taper(finished, original, TAPER_N, TAPER_N)
    b = apply_continuity_taper(finished, original, TAPER_N, TAPER_N)
    assert a.tobytes() == b.tobytes()


@pytest.mark.unit
def test_the_fade_removes_the_step_a_hard_cut_leaves() -> None:
    """A synthetic seam far larger than any measured on real material: the two
    renderings differ by a constant offset. A hard cut leaves the whole offset
    at the joint; the fade leaves nothing."""
    original = np.full(UNIT_N, 0.20, dtype=np.float32)
    finished = np.full(UNIT_N, 0.35, dtype=np.float32)
    neighbour_first_sample = float(original[0])  # what ships across the cut

    hard_step = abs(float(finished[-1]) - neighbour_first_sample)
    tapered = apply_continuity_taper(finished, original, 0, TAPER_N)
    soft_step = abs(float(tapered[-1]) - neighbour_first_sample)

    assert hard_step == pytest.approx(0.15, abs=1e-6)
    assert soft_step == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("profile", ["production", "studio", "development"])
@pytest.mark.parametrize("n", [977, SR // 2, SR, 3 * SR + 137])
def test_finishing_preserves_length_so_the_fade_stays_at_the_joint(profile: str, n: int) -> None:
    """A load-bearing invariant for the fade, pinned here because nothing else
    pins it.

    The fade-out is written to the LAST ``fade`` samples of the finished
    waveform, and :func:`~hawavoclean.assembly.stitch.assemble_channel_timeline`
    pads or truncates that waveform to the unit's core length. Truncation is
    harmless — the fade is already at the core length. Padding is not: a
    finished waveform SHORTER than the core would have its fade zero-padded
    away from the joint, and the seam would come back silently. So finishing
    must never shorten a unit.
    """
    cfg = load_config(profile_config_path(profile), is_production=profile == "production")
    rng = np.random.default_rng(3)
    t = np.arange(n) / SR
    wave = (
        0.2 * np.sin(2 * np.pi * 140 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
        + 0.02 * rng.standard_normal(n)
    ).astype(np.float32)

    res, _ = safe_finish_speech_unit(
        pre_finish_waveform=wave,
        sample_rate=SR,
        is_speech=True,
        probe=FixedProbe(),
        finishing_config=cfg.finishing,
        guard_config=cfg.guard,
    )
    assert len(res.finished_waveform) == n, (
        f"{profile} finishing returned {len(res.finished_waveform)} samples for a "
        f"{n}-sample unit; the continuity fade would no longer land at the joint"
    )
