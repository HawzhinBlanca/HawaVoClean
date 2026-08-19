"""Continuity enforcement must be channel-aware and visible in the audit trail."""

import numpy as np

from voiceclean.guard.verdict import GuardVerdict
from voiceclean.policy.continuity import enforce_source_continuity
from voiceclean.policy.decision import UnitPolicyDecision
from voiceclean.segmentation.types import SpeechUnit

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
