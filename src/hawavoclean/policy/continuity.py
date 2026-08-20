"""Source continuity across adjacent speech units.

A ``forced_boundary`` is a cut the segmenter made *inside* continuous speech,
because a single speech interval outran the maximum unit length and had to be
split somewhere. Enhanced audio meeting original audio across such a cut is a
seam: the two sides are different renderings of the same voice, joined at one
sample.

What the seam actually measures
-------------------------------
Measured on Flute 09 (94.6 s, production profile, 2026-08-21) at its one
enhanced/original forced cut, comparing the two renderings where they meet:

- Sample step at the joint, hard cut: **4.7% of local RMS**. Tapered: **0** —
  the last sample is the original recording, bit for bit.
- Spectral difference between the renderings over the final 30 ms:
  **3.05 dB** mean, at a local level of **-51 dBFS**.
- In the assembled timeline the cut is not an outlier at all: mean
  frame-to-frame spectral change **6.1 dB** across the joint against a
  file-wide mean of **7.3 dB**. By that measure a hard cut is invisible.

The seam is small here because it is *placed* small:
:func:`~hawavoclean.segmentation.utterances.find_lowest_energy_zero_crossing`
hunts a +-1 s window for the quietest zero crossing, and on this file the five
forced cuts landed **13.6 to 22.8 dB below the file's median frame RMS**. But
that search is bounded, 13.6 dB is not silence, and the rule has to hold for
material that gives it nowhere quiet to cut. So the seam is guarded — cheaply.

What the rule used to cost
--------------------------
The remedy used to be a revert, and it cascaded. The seam test is symmetric,
so reverting one unit created a seam on its far side; iterated to a fixed
point on a recording whose every boundary is forced -- continuous speech, no
pauses to cut at -- one failing unit reverted the whole file. On Flute 09 five
of six units passed the guard and the listener received ``enhanced: 0/6``.
Measured cost: speech/floor separation **27.65 dB against 34.88 dB** once the
five passing units keep their enhancement. **7.23 dB**, to hide a 3 dB
difference in 30 ms of near-silence.

The remedy now
--------------
An enhanced unit that meets original audio across a forced cut fades its own
enhancement back to the original recording over the last
:data:`CONTINUITY_TAPER_MS` before the joint (and/or the first, when the cut is
on its left). At the joint both sides are then bit-for-bit the original, so the
step is exactly zero and the difference between the renderings is spread over
30 ms instead of one sample. The unit keeps its enhancement everywhere else:
the fade gives back 38% of the enhancement inside its window, which on a 16 s
unit is 0.19% of the unit. A unit too short to afford the fade still reverts --
the fail-closed path is not removed, only made the exception instead of the
rule.

Because the two renderings are phase-coherent (the candidate is GCC-PHAT
aligned to the original, and the strength ladder already blends them linearly),
``(1 - w) * original + w * enhanced`` is not a crossfade between two signals at
all. It is the same signal with its enhancement *residual* scaled by ``w``, so
nothing can cancel; the only artefact available is the amplitude modulation of
the residual itself, whose rate at a 30 ms fade is ~17 Hz -- below the ~20 Hz
where modulation starts to be heard as texture rather than as a transition.
That, and not a spectral measurement, is why the fade is 30 ms: a sweep of
5-150 ms moved none of the metrics above, so the length is chosen by the one
thing that does constrain it.

Why the fade material is this unit's own original audio, and not the enhanced
context the pipeline computes and throws away: the neighbour across the cut is
original *because the guard rejected its candidate*. Extending this unit's
enhanced right context into that neighbour's territory would ship audio no
guard ever scored for that time range, which is the one thing the fail-closed
contract forbids. Fading toward the original is a move toward known-good audio.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hawavoclean.policy.decision import UnitPolicyDecision
from hawavoclean.segmentation.types import SpeechUnit

#: Length of the fade from enhanced back to original at a forced cut. See the
#: module docstring: a 5-150 ms sweep moved no spectral or separation metric,
#: so the length is set by the residual's modulation rate (~17 Hz at 30 ms).
CONTINUITY_TAPER_MS = 30.0

#: The fade may not spend more than this share of a unit at either edge, so a
#: unit tapered on both sides always keeps at least half of itself enhanced.
#: A unit too short to afford the fade reverts instead — at 48 kHz that is
#: anything under 120 ms, which this segmenter does not produce.
MAX_TAPER_FRACTION = 0.25

#: Marker written into ``UnitDecisionRecord.finish_actions`` when a taper ran.
#: The report counts these; the string is the audit trail's own vocabulary.
CONTINUITY_TAPER_ACTION = "continuity_taper"

REVERT_REASON = (
    "Reverted by continuity rule: forced mid-speech cut would seam enhanced "
    "audio against original audio, and the unit is too short to carry a fade."
)


@dataclass(frozen=True)
class ContinuityResolution:
    """What continuity enforcement decided for every unit of a run.

    ``fade_in_samples``/``fade_out_samples`` are per-unit fade lengths at the
    unit's left and right edge, ``0`` where no fade is needed. They are applied
    *after* finishing (:func:`apply_continuity_taper`), because the seam is
    between the **finished** enhanced audio and the original — fading any
    earlier would leave the finishing EQ's own step sitting at the joint.
    """

    decisions: list[UnitPolicyDecision]
    reverted_ids: set[int] = field(default_factory=set)
    fade_in_samples: list[int] = field(default_factory=list)
    fade_out_samples: list[int] = field(default_factory=list)


def resolve_source_continuity(
    units: list[SpeechUnit],
    decisions: list[UnitPolicyDecision],
    orig_core_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]],
    sample_rate: int,
) -> ContinuityResolution:
    """Plan the fades — and, where a unit is too short to fade, the reverts —
    that keep enhanced audio from butting against original audio at a forced cut.

    ``forced_boundary`` marks the cut at a unit's END, so the boundary between
    unit ``i`` and ``i+1`` is a forced cut iff ``units[i]`` carries the flag.
    Only same-channel neighbours are adjacent in time.
    """
    n = len(units)
    if n == 0:
        return ContinuityResolution(list(decisions), set(), [], [])

    adjusted = list(decisions)
    taper = int(round(sample_rate * CONTINUITY_TAPER_MS / 1000.0))

    def same_channel(a: int, b: int) -> bool:
        return units[a].channel_id == units[b].channel_id

    def forced_cut_between(left: int, right: int) -> bool:
        return units[left].forced_boundary and same_channel(left, right)

    def seams(i: int) -> tuple[bool, bool]:
        """(left edge seams, right edge seams) for unit ``i`` as things stand."""
        left = i > 0 and forced_cut_between(i - 1, i) and not adjusted[i - 1].is_enhanced
        right = i < n - 1 and forced_cut_between(i, i + 1) and not adjusted[i + 1].is_enhanced
        return left, right

    def can_afford_fade(i: int) -> bool:
        return taper <= len(orig_core_waveforms[i]) * MAX_TAPER_FRACTION

    def reverted(i: int, prior: UnitPolicyDecision) -> UnitPolicyDecision:
        return UnitPolicyDecision(
            selected_waveform=orig_core_waveforms[i].copy(),
            is_enhanced=False,
            chosen_strength=0.0,
            guard_verdict=prior.guard_verdict,
            guard_scores=prior.guard_scores,
            decision_reason=REVERT_REASON,
        )

    # Reverts are monotone — a unit never becomes enhanced again — and each
    # pass reverts at least one unit or stops, so this terminates in <= n
    # passes. A revert can open a seam for a neighbour, but that neighbour now
    # gets a fade rather than a revert of its own: the cascade cannot propagate
    # past a unit long enough to fade, which on real material is every unit.
    reverted_ids: set[int] = set()
    changed = True
    passes = 0
    while changed and passes <= n:
        changed = False
        passes += 1
        for i in range(n):
            if not adjusted[i].is_enhanced or can_afford_fade(i):
                continue
            if any(seams(i)):
                adjusted[i] = reverted(i, adjusted[i])
                reverted_ids.add(units[i].unit_id)
                changed = True

    fade_in = [0] * n
    fade_out = [0] * n
    for i in range(n):
        if not adjusted[i].is_enhanced or not can_afford_fade(i):
            continue
        left, right = seams(i)
        fade_in[i] = taper if left else 0
        fade_out[i] = taper if right else 0

    return ContinuityResolution(adjusted, reverted_ids, fade_in, fade_out)


def _rising_weight(n: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Raised-cosine ramp from exactly 0.0 to exactly 1.0 over ``n`` samples.

    Raised cosine rather than linear because its slope is zero at both ends, so
    the blend weight joins the untouched audio on either side without a corner.
    The endpoints are pinned rather than trusted to ``cos``: the whole point of
    the fade is that its outer sample is *exactly* the original recording.
    """
    if n <= 1:
        return np.zeros(max(n, 0), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
    w[0] = 0.0
    w[-1] = 1.0
    return w


def apply_continuity_taper(
    finished: np.ndarray[Any, np.dtype[np.float32]],
    original: np.ndarray[Any, np.dtype[np.float32]],
    fade_in_samples: int,
    fade_out_samples: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Fade ``finished`` back to ``original`` at the edges that seam.

    The blend is written ``(1 - w) * original + w * finished`` so that ``w = 0``
    reproduces the original sample exactly and ``w = 1`` the finished sample
    exactly. No rounding creeps into the audio outside the fade windows, and
    the sample at the joint is the original recording to the bit.
    """
    if fade_in_samples <= 0 and fade_out_samples <= 0:
        return finished

    n = min(len(finished), len(original))
    out = np.array(finished, dtype=np.float32, copy=True)
    fade_in = max(0, min(fade_in_samples, n))
    # The two fades never overlap: a unit that seams on both sides spends at
    # most MAX_TAPER_FRACTION of itself on each. Clamp anyway rather than let a
    # caller blend one window twice.
    fade_out = max(0, min(fade_out_samples, n - fade_in))

    if fade_in > 0:
        w = _rising_weight(fade_in)
        out[:fade_in] = (1.0 - w) * original[:fade_in] + w * finished[:fade_in]
    if fade_out > 0:
        w = _rising_weight(fade_out)[::-1]
        start = n - fade_out
        out[start:n] = (1.0 - w) * original[start:n] + w * finished[start:n]

    return out
