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
    """Enforce that source transitions do not cut through continuous speech runs.

    If an enhanced unit is flanked by a forced boundary connected to a reverted unit,
    revert the enhanced unit to original audio to preserve phonetic continuity.
    """
    if len(units) <= 1:
        return decisions

    adjusted_decisions = list(decisions)
    num_units = len(units)

    def _same_channel(a: int, b: int) -> bool:
        return units[a].channel_id == units[b].channel_id

    # forced_boundary marks the cut at a unit's END (speech was split there).
    # The boundary between unit i and i+1 is a forced cut iff units[i] carries
    # the flag. Continuity is only at stake across such cuts: an enhanced unit
    # meeting original audio across a forced cut is an audible timbre seam
    # inside continuous speech. A natural pause boundary is never a seam.
    def _forced_cut_between(left: int, right: int) -> bool:
        return units[left].forced_boundary and _same_channel(left, right)

    for i in range(num_units):
        curr_dec = adjusted_decisions[i]
        if not curr_dec.is_enhanced:
            continue

        seam = False
        # Cut at this unit's end, original audio on the right
        if i < num_units - 1 and _forced_cut_between(i, i + 1):
            seam = not adjusted_decisions[i + 1].is_enhanced
        # Cut at the previous unit's end, original audio on the left
        if not seam and i > 0 and _forced_cut_between(i - 1, i):
            seam = not adjusted_decisions[i - 1].is_enhanced

        if seam:
            orig_wave = orig_core_waveforms[i]
            adjusted_decisions[i] = UnitPolicyDecision(
                selected_waveform=orig_wave.copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=curr_dec.guard_verdict,
                guard_scores=curr_dec.guard_scores,
                decision_reason=(
                    "Reverted by continuity rule: forced mid-speech cut would seam "
                    "enhanced audio against original audio."
                ),
            )

    return adjusted_decisions
