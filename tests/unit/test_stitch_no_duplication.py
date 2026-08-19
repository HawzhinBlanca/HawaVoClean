"""Timeline stitching must place every source sample exactly once.

Each unit waveform is a globally unique strictly-increasing ramp, so every
sample value identifies its origin. Rendering any span twice is detectable
as a repeated value outside the declared fade region.
"""

import numpy as np

from hawavoclean.assembly.stitch import assemble_channel_timeline
from hawavoclean.segmentation.types import SpeechUnit

SR = 48000


def _unit(uid: int, start: int, end: int, forced: bool = False) -> SpeechUnit:
    return SpeechUnit(
        unit_id=uid,
        channel_id=0,
        start_sample=start,
        end_sample=end,
        context_start_sample=start,
        context_end_sample=end,
        is_speech=True,
        forced_boundary=forced,
    )


def test_two_unit_butt_joint_duplicates_nothing() -> None:
    n = SR
    b = n // 2
    u0, u1 = _unit(0, 0, b), _unit(1, b, n)
    w0 = np.full(b, 1.0, dtype=np.float32)
    # unique ramp far above w0's value range: value == 10 + index
    w1 = 10.0 + np.arange(n - b, dtype=np.float32)

    tl = assemble_channel_timeline([u0, u1], [w0, w1], n, SR, crossfade_ms=20.0)
    fade = int(SR * 0.02)

    # Unit 1 content must never appear before its own start sample.
    pre_boundary = tl[b - fade : b]
    leaked = pre_boundary[pre_boundary >= 10.0]
    assert leaked.size == 0, (
        f"{leaked.size} samples of unit 1 rendered inside unit 0's span "
        f"(first leaked value {leaked[0] if leaked.size else '-'}); "
        "the unit head is being rendered twice"
    )

    # And unit 1 must still be intact, on time, at its own span.
    np.testing.assert_array_equal(tl[b + fade : n], w1[fade:])


def test_four_unit_chain_with_forced_boundary_duplicates_nothing() -> None:
    n = SR * 2
    bounds = [0, n // 4, n // 2, 3 * n // 4, n]
    units = [_unit(i, bounds[i], bounds[i + 1], forced=(i == 2)) for i in range(4)]
    # Globally unique, DISJOINT value ranges: unit i occupies
    # [1e6*(i+1), 1e6*(i+1) + len), so no unit's values can be mistaken
    # for another's.
    waves = [
        (1e6 * (i + 1) + np.arange(bounds[i + 1] - bounds[i], dtype=np.float64)).astype(np.float32)
        for i in range(4)
    ]
    tl = assemble_channel_timeline(units, waves, n, SR, crossfade_ms=20.0)

    for i in range(1, 4):
        start = bounds[i]
        length = bounds[i + 1] - bounds[i]
        fade = int(SR * 0.02)
        lo, hi = 1e6 * (i + 1), 1e6 * (i + 1) + length
        pre = tl[max(0, start - fade) : start]
        leaked = pre[(pre >= lo) & (pre < hi)]
        assert leaked.size == 0, f"unit {i} content leaked before its start sample"
