# ADR 0006: Band-Split Restoration Core (`studio-dfn3-lowband-48k-v1`)

## Context

A muffled recording whose noise is low-frequency tonal rumble — an air
handler, traffic, a room mode, a preamp hum with harmonics — defeats both
shipped cores, and it defeats them in opposite ways.

Measured on `test_output/teat1vo-lab/src.mp3` (24 s, dual-mono, 48 kHz;
speech-to-floor separation 15.1 dB):

| profile | separation | 60–300 Hz pause rumble | guard outcome |
| --- | --- | --- | --- |
| source | 15.1 dB | −32.3 dB rel. speech | — |
| production (Wiener) | 19.8 dB | −34.6 dB | enhanced |
| studio (full-band DFN3) | 15.1 dB | −30.5 dB | **all speech units reverted** |

The Wiener filter cannot get under the rumble: its gain floor (0.05) bounds
how far any bin can be attenuated, which is the property that makes it safe
and also the property that makes it useless here. Full-band DeepFilterNet3
does remove the rumble, but on this material it retains only **0.22 of the
original 2–8 kHz energy** — it takes the consonants with it. The fidelity
guard sees that and reverts, so the studio profile publishes the original
audio and the user gets nothing.

## Decision

Give DeepFilterNet3 only the band the problem is in, by running it over the
full band and keeping only its low output:

```
enhanced = DFN3(x)                        # unlimited attenuation
out      = lowpass(enhanced) + (x - lowpass(x))
```

Three choices in that expression are load-bearing.

**DFN3 sees the full band.** The obvious alternative — lowpass first, then
denoise the low band — is worse, not better: a 1.5 kHz-limited signal is out
of distribution for a model trained on full-band speech, and it responds by
classifying most of it as noise. Measured spectral-hole score **0.585**
against the guard's 0.100 threshold, versus **0.066** for the construction
above. The guard reverts the first and accepts the second.

**One filter, subtracted.** Both bands come from the same lowpass, the high
one as the arithmetic complement of the low one. If DFN3 ever returned its
input unchanged, `lowpass(x) + (x - lowpass(x))` is `x` to a float rounding
error, at every frequency. Two independently designed filters — a lowpass
for one path, a highpass for the other — sum to a magnitude ripple and a
phase step at the crossover, which is exactly where the voice's first
formant lives.

**The crossover is a locked parameter, not a setting.** Measured against the
guard's own spectral-hole score (threshold 0.100) and consonant retention:

| crossover | hole score | consonant retention | guard |
| --- | --- | --- | --- |
| 700 Hz | 0.050 | 1.000 | accepts |
| 900 Hz | 0.060 | 0.999 | accepts |
| **1000 Hz** | **0.066** | **~0.999** | **accepts — shipped** |
| 1300 Hz | 0.088 | 0.988 | accepts |
| 1500 Hz | 0.103 | 0.964 | **reverts** |
| 2500 Hz | 0.187 | 0.497 | **reverts** |

Above roughly 1.1 kHz, DFN3 is reaching into the consonant band, the score
crosses the threshold, and every unit comes back as original. So
`crossover_hz` lives inside `params_hash`: moving it is a new core and a
relock, enforced at preflight, by `hawavoclean audit-models`, and by
mutation **M13**. The complementary split is enforced by mutation **M14**.

## Consequences

- Consonants are preserved **by construction** rather than by the model's
  good judgement. Measured retention 0.999 on the lab fixture and 1.000 on
  every unit of the unrelated Flute 09 recording.
- The core is not phase-coherent (DFN3 rewrites phase below the crossover),
  so the `lowband` profile offers the guard only the full-strength
  candidate (`strength_ladder = [1.0]`) and runs it in `integrity` mode.
- Shipped end to end: separation 15.1 → **29.4 dB**, and **35.2 dB** with a
  production pass after it, the guard judging every unit in both runs.
- On material without the rumble problem the core is quiet rather than
  harmful: Flute 09 scores 0.002–0.011 on the hole detector and keeps all
  five units.

## History

The prototype chain this core productizes reported 40.0 dB separation. It
reached that number by running the band split as an **unguarded script** and
feeding the raw, unmastered result to a production pass — so the one
aggressive step in the chain was never judged by the guard, and the
intermediate had not yet spent its dynamic range on a master. Productized,
the split is accountable to the guard like everything else and the
intermediate is a real master; the same chain lands at 35.2 dB. The number
is lower because the step that earned it is now checked.

The prototype's filter was also nominally labelled "1.5 kHz" while its
effective −3 dB point sat near 1 kHz. This core is parameterized by its
actual −3 dB point, so the shipped `crossover_hz = 1000.0` and the
prototype's "1.5 kHz" describe the same filter — they agree on the measured
guard score to three decimals (0.066 vs 0.065).

## Reproducing the measurements

```bash
uv sync --extra studio
hawavoclean process test_output/teat1vo-lab/src.mp3 -o /tmp/lowband.wav --profile lowband
hawavoclean process /tmp/lowband.wav -o /tmp/final.wav --profile production
```

Per-unit guard scores (`spectral_hole_score`, `consonant_retention`) are in
the `.hawavoclean.json` report published beside each output; the separation
figures are the 90th-percentile minus 10th-percentile 20 ms frame level.
