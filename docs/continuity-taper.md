# The continuity seam: what it costs, and what it is worth

Evidence for the change from *revert* to *fade* in
`hawavoclean/policy/continuity.py`. Every number here was measured on
2026-08-21 on this tree; the harnesses are described inline so they can be
re-run.

## The problem

A `forced_boundary` is a cut the segmenter made **inside** continuous speech,
because a single speech interval outran `hard_max_group_s` and had to be split
somewhere. `find_lowest_energy_zero_crossing` picks the quietest zero crossing
in a ±1 s window, so the cut lands in the quietest place available — but it
still lands inside a word.

If the unit on one side of that cut is enhanced and the unit on the other is
original, the two sides are different renderings of the same voice joined at
one sample. The old remedy was to revert the enhanced unit.

Reverting is **symmetric**: the reverted unit now presents original audio to
*its* other neighbour, which — if that boundary is also forced — must revert in
turn. The rule iterates to a fixed point. On a recording of continuous speech
there are no pauses to cut at, so *every* boundary is forced, and one failing
unit takes the file with it.

### Flute 09, production profile (94.6 s, 6 units)

| | before | after |
|---|---|---|
| `enhanced` | **0 / 6** | **5 / 6** |
| `continuity_reverted` | 5 | 0 |
| `continuity_crossfaded` | 0 | 1 |
| `finish_bypassed` | 6 | 1 |
| speech/floor separation | **27.65 dB** | **34.88 dB** |

Separation is `hawavoclean.multipass.speech_floor_separation_db` — the spread
between the 90th and 10th percentile of frame RMS (2048-sample frames, 1024
hop) in dB. Five of six units had passed the guard; the listener received none
of them.

## How big is the seam the rule was protecting?

Measured by intercepting `apply_continuity_taper` during a real run, which
hands it the exact two renderings that meet at Flute 09's one enhanced/original
forced cut: the finished enhanced tail of unit 4, and the original audio unit 5
ships.

| window before the cut | enhanced RMS | original RMS | level step | mean spectral difference |
|---|---|---|---|---|
| 10 ms | −54.61 dB | −51.05 dB | −3.56 dB | 3.06 dB |
| 30 ms | −54.59 dB | −51.07 dB | −3.52 dB | 3.05 dB |
| 100 ms | −53.23 dB | −50.02 dB | −3.21 dB | 2.39 dB |
| 500 ms | −36.51 dB | −38.01 dB | +1.50 dB | 1.94 dB |

Spectral difference is the mean |ΔdB| per bin over 100 Hz – 8 kHz of a Hann-
windowed rFFT of the window.

At the joint itself:

| | value |
|---|---|
| local RMS at the cut | 0.002795 (−51 dBFS) |
| last-sample step, hard cut | 0.000131 — **4.7% of local RMS** |
| last-sample step, faded | 0.000000 — **exactly the original sample** |

And in the assembled timeline the cut does not stand out at all. Frame-to-frame
mean spectral change (30 ms frames, 15 ms hop, 100 Hz – 8 kHz):

| | at the cut | file-wide mean | file-wide p99 | file-wide max |
|---|---|---|---|---|
| hard cut | 6.13 dB | 7.34 dB | 18.75 dB | 35.38 dB |
| 30 ms fade | 5.88 dB | 7.34 dB | 18.75 dB | 35.38 dB |
| all-original (old master) | 6.07 dB | 7.15 dB | 17.95 dB | 46.97 dB |

**By this measure a hard cut is invisible** — the change across the joint is
*below* the file's typical frame transition. This refuted the working
assumption going in (a 12.1 dB step against a 6.2 dB control), and it is why
the change is justified by what the old remedy *cost*, not by what the fade
*fixes*.

### Why the seam is small: it is placed small

Local RMS in a ±30 ms window at each of Flute 09's five forced cuts, against
the file's median frame RMS (−29.33 dBFS):

| cut | time | local RMS | vs median |
|---|---|---|---|
| 0\|1 | 16.80 s | −48.96 dBFS | −19.63 dB |
| 1\|2 | 32.13 s | −51.18 dBFS | −21.85 dB |
| 2\|3 | 47.14 s | −48.24 dBFS | −18.92 dB |
| 3\|4 | 62.98 s | −42.90 dBFS | −13.57 dB |
| 4\|5 | 79.72 s | −52.13 dBFS | −22.80 dB |

The zero-crossing search is doing its job. But its window is bounded, −13.6 dB
is not silence, and material that offers nowhere quiet to cut will produce a
louder seam — so the seam stays guarded. The fade makes guarding it cheap
enough that the question stops mattering.

## Why 30 ms

A sweep of the fade length, each condition a full pipeline run, plus a
hard-cut control (the fade disabled but the unit kept enhanced):

| condition | enhanced | seam peak | at cut | separation |
|---|---|---|---|---|
| hard cut | 5/6 | 6.34 dB | 6.13 dB | 34.88 dB |
| 5 ms | 5/6 | 6.34 dB | 6.04 dB | 34.88 dB |
| 10 ms | 5/6 | 6.35 dB | 5.97 dB | 34.88 dB |
| 20 ms | 5/6 | 6.31 dB | 5.92 dB | 34.88 dB |
| **30 ms** | 5/6 | 6.33 dB | 5.88 dB | 34.88 dB |
| 50 ms | 5/6 | 6.26 dB | 5.97 dB | 34.88 dB |
| 80 ms | 5/6 | 6.18 dB | 6.03 dB | 34.88 dB |
| 150 ms | 5/6 | 6.16 dB | 6.06 dB | 34.88 dB |
| reverted (old master) | 0/6 | 6.16 dB | 6.07 dB | **27.65 dB** |

**No spectral or separation metric distinguishes any fade length, or even
distinguishes a fade from a hard cut.** Reported rather than hidden: the length
cannot be chosen by these measurements.

What does constrain it: because the candidate is GCC-PHAT aligned to the
original and the strength ladder already blends the two linearly,
`(1 − w)·original + w·enhanced` is not a crossfade between two signals. It is
the same signal with its enhancement *residual* scaled by `w`, so nothing can
cancel and no comb filtering is available. The only artefact on offer is the
amplitude modulation of the residual itself, at a rate of roughly `1/(2T)` —
**~17 Hz at 30 ms**, below the ~20 Hz where modulation is heard as texture
rather than as a transition. A 5 ms fade would modulate at ~100 Hz, squarely
in the roughness band.

## What the fade costs

Over its 30 ms window the fade gives back 38% of the enhancement (residual RMS
0.000964 → 0.000601, **62% retained**). On the 16.7 s unit that carried it,
that is **0.19% of the unit**, in its quietest 30 ms.

## Blast radius

Of the four committed reference masters (`test_output/perf-ref-hashes.txt`),
**three are byte-identical** after this change:

```
75f6a600…  Flute 09_production.wav   ← changed: the file the rule was damaging
a5d418f2…  Flute 09_studio.wav       ← unchanged
7fc036a8…  src_production.wav        ← unchanged
94c5665b…  src_studio.wav            ← unchanged
```

No configuration key was added, renamed or removed, so `config_hash` — and
with it the job id, the dither seed, and the last bit of every sample — is
untouched. Every hash that moved, moved because the audio did.
