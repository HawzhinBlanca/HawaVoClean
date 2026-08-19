"""Continuity enforcement must be channel-aware and visible in the audit trail."""

import numpy as np

from hawavoclean.guard.verdict import GuardVerdict
from hawavoclean.policy.continuity import enforce_source_continuity
from hawavoclean.policy.decision import UnitPolicyDecision
from hawavoclean.segmentation.types import SpeechUnit

SR = 48000


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
        selected_waveform=np.ones(n, dtype=np.float32),
        is_enhanced=enhanced,
        chosen_strength=1.0 if enhanced else 0.0,
        guard_verdict=GuardVerdict.PASS if enhanced else GuardVerdict.REVERT,
    )


def test_units_on_different_channels_are_never_neighbours() -> None:
    n = SR // 2
    units = [
        _unit(0, 0, 0, n, forced=True),  # ch0: enhanced, forced boundary
        _unit(1, 1, 0, n, forced=False),  # ch1: reverted — a DIFFERENT channel
    ]
    decisions = [_dec(True, n), _dec(False, n)]
    waves = [np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]

    adjusted = enforce_source_continuity(units, decisions, waves)

    assert adjusted[0].is_enhanced, (
        "the ch0 unit was reverted because of a ch1 'neighbour'; units on "
        "different channels are not adjacent in time"
    )


def test_continuity_only_fires_across_a_forced_boundary() -> None:
    """forced_boundary marks the cut at a unit's END. A reverted LEFT
    neighbour across a natural (pause) boundary must not revert an enhanced
    unit — speech was never split there.

    Measured cost on a real field recording, 2026-08-19: the only unit that
    passed the guard was reverted by this rule because of a reverted left
    neighbour across a normal pause boundary.
    """
    n = SR // 2
    units = [
        _unit(0, 0, 0, n, forced=False),  # reverted, natural boundary after it
        _unit(1, 0, n, 2 * n, forced=True),  # enhanced; ITS end is the forced cut
        _unit(2, 0, 2 * n, 3 * n, forced=False),  # enhanced on the far side of the cut
    ]
    decisions = [_dec(False, n), _dec(True, n), _dec(True, n)]
    waves = [np.zeros(n, dtype=np.float32)] * 3

    adjusted = enforce_source_continuity(units, decisions, waves)

    assert adjusted[1].is_enhanced, (
        "unit 1 was reverted because of its LEFT neighbour, but the boundary "
        "between them was a natural pause, not a forced mid-speech cut"
    )


def test_continuity_fires_when_the_forced_cut_separates_enhanced_from_original() -> None:
    """The real hazard: a forced cut through speech with enhanced audio on
    one side and original on the other -> audible timbre seam. Revert."""
    n = SR // 2
    units = [
        _unit(0, 0, 0, n, forced=True),  # enhanced; forced cut at its end
        _unit(1, 0, n, 2 * n, forced=False),  # reverted, right across that cut
    ]
    decisions = [_dec(True, n), _dec(False, n)]
    waves = [np.zeros(n, dtype=np.float32)] * 2

    adjusted = enforce_source_continuity(units, decisions, waves)
    assert not adjusted[0].is_enhanced, "enhanced/original seam across a forced cut must revert"
