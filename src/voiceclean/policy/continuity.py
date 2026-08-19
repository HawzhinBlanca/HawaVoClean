"""Source continuity enforcement across adjacent speech units."""

from typing import Any

import numpy as np

from voiceclean.policy.decision import UnitPolicyDecision
from voiceclean.segmentation.types import SpeechUnit


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

    for i in range(num_units):
        curr_unit = units[i]
        curr_dec = adjusted_decisions[i]

        if not curr_dec.is_enhanced:
            continue

        # If current unit had a forced boundary (i.e. cut through speech)
        if curr_unit.forced_boundary:
            # Check left neighbor
            left_reverted = i > 0 and not adjusted_decisions[i - 1].is_enhanced
            # Check right neighbor
            right_reverted = i < num_units - 1 and not adjusted_decisions[i + 1].is_enhanced

            if left_reverted or right_reverted:
                # Revert current unit to original to maintain smooth speech continuity
                orig_wave = orig_core_waveforms[i]
                adjusted_decisions[i] = UnitPolicyDecision(
                    selected_waveform=orig_wave.copy(),
                    is_enhanced=False,
                    chosen_strength=0.0,
                    guard_verdict=curr_dec.guard_verdict,
                    guard_scores=curr_dec.guard_scores,
                    decision_reason="Reverted by continuity rule: adjacent forced boundary connected to original audio.",
                )

    return adjusted_decisions
