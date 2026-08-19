# Spectral Fidelity Guard

## What the guard is

A two-pass safety perimeter that compares the *spectral signature* of
processed audio against the original and reverts any unit whose signature
diverges. Guard A validates the enhancer's candidate against the original;
Guard B validates the locally-finished audio against the pre-finish
accepted rendering.

## What the guard is not

It is **not** a speech recognizer and does not verify linguistic content.
The probe (`voiceclean.guard.spectral_probe`) has no acoustic model: it
maps the shape of the 0–2.5 kHz spectrum to a symbol distribution and
compares distributions. A change that preserves spectral shape — including
a hypothetical word substitution with similar spectral content — passes it.
Claims about protecting "against word substitutions and phonetic drift"
appeared in earlier revisions of this document and were not true.

`tests/unit/test_probe_is_not_asr.py` pins both sides of this boundary:
different content with the same spectral envelope looks the same to the
probe; the same content with a shifted spectrum looks different.

## Guard checks

1. **Sustained-state token anchors** — collapsed spectral states compared
   by edit distance; insufficient anchors yields `UNVERIFIED` (fail-closed).
2. **Frame distribution divergence** — Jensen-Shannon divergence between
   per-frame symbol distributions; rejects `mean_js_div > 0.25`.
3. **Timing and envelope integrity** — envelope correlation and drift
   bounds (75 ms production threshold).
4. **Acoustic signal integrity** — consonant-band retention, spectral hole,
   musical noise, and new-clipping detectors.

## Guard modes

- **strict_spectral** (production default): the output must stay spectrally
  near-identical to the input. Anchor deletions/substitutions, timestamp
  drift, and peak distribution divergence all gate the verdict. Right for
  gentle cleanup where any spectral change is suspect.
- **integrity** (studio profile): restoration removes noise and reverb BY
  DESIGN, so spectral identity is not enforced. Still enforced: envelope
  correlation and timing drift, bounded mean/peak distribution divergence,
  spectral-hole / musical-noise / consonant-retention / clipping detectors,
  and the output-collapse validation. Anchor statistics are recorded in the
  report but do not gate. The mode is declared in the profile's calibration
  artifact and visible in every report.

## Verdicts

`PASS` (candidate accepted), `REVERT` (original audio used), `UNVERIFIED`
(cannot judge — original audio used), `ERROR` (guard fault — original audio
used), `NO_SPEECH` (guard bypassed for non-speech units).

## Calibration

Thresholds ship as engineering defaults and say so in the artifact.
`voiceclean calibrate` measures real accept/revert rates over a corpus and
corruption profile, writing metrics with measurement provenance attached.
