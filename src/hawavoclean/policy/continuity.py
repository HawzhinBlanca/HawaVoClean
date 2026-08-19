"""Source continuity enforcement across adjacent speech units."""

from typing import Any

import numpy as np

from hawavoclean.policy.decision import UnitPolicyDecision
from hawavoclean.segmentation.types import SpeechUnit


def enforce_source_continuity(
    units: list[SpeechUnit],
    decisions: list[UnitPolicyDecision],
    orig_core_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]],
) -> list[UnitPolicyDecision]:
    """Revert enhanced units that would seam against original audio across a
    forced mid-speech cut.

    ``forced_boundary`` marks the cut at a unit's END (speech was split
    there). The boundary between unit i and i+1 is a forced cut iff
    ``units[i]`` carries the flag, and only same-channel neighbours are
    adjacent in time. An enhanced unit meeting original audio across such a
    cut is an audible timbre seam inside continuous speech; a natural pause
    boundary never is. Because a revert can create a new seam on its other
    side, the rule iterates to a fixed point.
    """
    if len(units) <= 1:
        return decisions

    adjusted = list(decisions)
    n = len(units)

    def same_channel(a: int, b: int) -> bool:
        return units[a].channel_id == units[b].channel_id

    def forced_cut_between(left: int, right: int) -> bool:
        return units[left].forced_boundary and same_channel(left, right)

    def has_seam(i: int) -> bool:
        if i < n - 1 and forced_cut_between(i, i + 1) and not adjusted[i + 1].is_enhanced:
            return True
        return i > 0 and forced_cut_between(i - 1, i) and not adjusted[i - 1].is_enhanced

    def reverted(i: int, prior: UnitPolicyDecision) -> UnitPolicyDecision:
        return UnitPolicyDecision(
            selected_waveform=orig_core_waveforms[i].copy(),
            is_enhanced=False,
            chosen_strength=0.0,
            guard_verdict=prior.guard_verdict,
            guard_scores=prior.guard_scores,
            decision_reason=(
                "Reverted by continuity rule: forced mid-speech cut would seam "
                "enhanced audio against original audio."
            ),
        )

    changed = True
    passes = 0
    while changed and passes <= n:  # each pass reverts >= 1 unit or terminates
        changed = False
        passes += 1
        for i in range(n):
            if adjusted[i].is_enhanced and has_seam(i):
                adjusted[i] = reverted(i, adjusted[i])
                changed = True

    return adjusted
