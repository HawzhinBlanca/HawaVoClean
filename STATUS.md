# Implementation Status

All current gate numbers below were measured on commit `a387b80` on
2026-08-21 (Python 3.14.5). To re-measure the Phase 2 set:
`bash scripts/run_release_checks.sh`, `pytest -m fuzz`,
`python scripts/mutation_gate.py`, and
`python scripts/audio_regression_gate.py --runs 2`.

## What this system is

An offline dialogue cleanup tool: an enhancement core + spectral-change
guard + deterministic finishing + BS.1770 mastering. Three cores ship: the
default Wiener spectral denoiser (classical DSP), and two optional neural
cores built on vendored, hash-locked DeepFilterNet3 weights — full-band
(`studio`) and band-split (`lowband`). There is no speech recognition
anywhere in the system; see README "What it is not".

## Measured verification results (2026-08-21)

| Gate | Measured result |
|---|---|
| Test suite | 869 passed, 0 failed (41 fuzz cases deselected) |
| Fuzz gate (`pytest -m fuzz`) | 41 passed, 0 failed |
| Branch coverage (`--cov-fail-under=90`) | 92.13% |
| Mutation gate | 23/23 mutations caught by their own owning tests |
| Ruff format / lint | exit 0 / exit 0 |
| Mypy `--strict` (199 files) | 0 issues |
| UI (`pnpm typecheck` / `test:run` / `build`) | exit 0 / 342 passed / exit 0 |
| Doctor from repo dir | exit 0; all 4 profiles and all 3 core locks valid |
| Real profile regressions | 6 cases × 2 runs; deterministic v3.3 hashes; zero unexplained semantic drift |
| Publication fault matrix | 57/57 on macOS/APFS and Linux/overlayfs; real SIGINT/SIGTERM/SIGKILL included |
| Evidence ledger | 6 entries; schema, baseline, external hashes and chain verify |
| `audit-models` (3 cores, clean tree) | exit 0 |
| `audit-models` tamper checks | params tamper → exit 2; bad license → exit 2; calibration tamper → exit 2 |
| Acceptance gates (`hawavoclean eval`) | PASSED 4/4 items; capable of FAILED, enforced under `python -O` |
| Real-time factor (production profile, CPU; carried from 2026-08-20) | 0.146 |

The v3.3 version bump intentionally changes the deterministic PCM24 dither
seed, so the six candidate WAV hashes differ from their pre-integration
references. The regression gate proves the difference is confined to at
most 2 least-significant bits (RMS 0.708–0.709 LSB), with identical sample
structure and audio/decision report semantics. This is recorded explicitly;
it is not presented as byte-identical DSP output.

The formal True 10/10 plan requires at least 92.49% branch coverage. The
integrated tree is currently at 92.13%, so Phase 3 must add direct coverage
before its unified release gate can pass. The ordinary 90% gate passing is
not yet the final release claim.

## Measured on the band-split core (`lowband`, reverified 2026-08-21)

Lab fixture `test_output/teat1vo-lab/src.mp3` — a muffled recording sitting
on low-frequency rumble. Separation is the 90th minus the 10th percentile of
20 ms frame level; rumble is the 60–300 Hz level over the quietest fifth of
frames, relative to the file's own speech level. See
[ADR 0006](docs/adr/0006-band-split-restoration-core.md).

| chain | separation | pause rumble | guard |
|---|---|---|---|
| source | 15.1 dB | −32.3 dB | — |
| production (Wiener) | 19.8 dB | −34.6 dB | enhanced |
| studio (full-band DFN3) | 15.1 dB | −30.5 dB | all speech units reverted |
| `lowband` | 29.4 dB | −71.6 dB | enhanced, hole 0.066, consonants 0.999 |
| `lowband` → `production` | 35.2 dB | −83.3 dB | enhanced in both runs |

On unrelated material (Flute 09, 5 units) the `lowband` core keeps every
unit at consonant retention 1.000 and hole scores 0.002–0.011, and the
`studio` profile's per-unit output hashes are unchanged by this work.

## Honest caveats that remain

- The guard is very conservative on the bundled synthetic corpus: most
  units come back UNVERIFIED (featureless tones give the probe too few
  anchors, so it fails closed). Measured on the calibration corpus:
  0/16 corrupted renderings accepted, 8/8 benign renderings rejected
  (dominated by UNVERIFIED-on-identity). On real speech, richer spectral
  structure yields more anchors; no claim is made until that is measured.
- There is no adequately licensed, transcribed, speaker-disjoint Sorani
  corpus with dual human review. The private recordings above are regression
  fixtures, not linguistic acceptance evidence.
- The `lowband` prototype reported 40.0 dB separation on the lab fixture;
  the shipped chain measures 35.2 dB. The prototype reached the higher
  number by running the band split as an unguarded script and handing a
  raw, unmastered intermediate to the second pass. Productized, the
  aggressive step is judged by the guard and the intermediate is a real
  master. The lower figure is the one with the checks in it.
- Both the `lowband` and `studio` figures come from a single 24 s recording
  and a single 94.6 s recording respectively. They are real measurements on
  real audio, not a corpus result.

## History note

A previous revision of this file claimed "PRODUCTION READY", "100%
blueprint-compliant", and a measured zero false-accept rate. Those claims
were false: the metrics were hardcoded, the model registry was fabricated,
and the audit trail misreported cached re-runs. The 2026-08-19 repair
(v2.0.0) removed the fabrications; BLUEPRINT.md is retained as a historical
design document only.
