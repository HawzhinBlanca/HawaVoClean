# Implementation Status

All numbers below were measured on 2026-08-19 in a fresh clone with a fresh
`uv sync --locked` environment (Python 3.14.5) and no prior state. To
re-measure: `bash scripts/run_release_checks.sh` and
`python scripts/mutation_gate.py`.

## What this system is

An offline dialogue cleanup tool: Wiener spectral denoiser + spectral-change
guard + deterministic finishing + BS.1770 mastering. No neural models, no
speech recognition — see README "What it is not".

## Measured verification results (fresh clone, 2026-08-19)

| Gate | Measured result |
|---|---|
| Test suite | 176 passed, 0 failed |
| Branch coverage (cold, `--cov-fail-under=90`) | 90.82% (91.22% on Py 3.13) |
| Cold vs warm coverage delta | 0.0 pp (identical) |
| Mutation gate | 12/12 mutations each break the suite |
| Ruff format / lint | exit 0 / exit 0 |
| Mypy `--strict` (126 files) | 0 issues |
| Doctor from clone dir and from `/` | exit 0 / exit 0 |
| Re-run determinism (same file twice) | per-unit verdicts identical |
| `verify` over all 18 produced outputs | 18/18 pass |
| `audit-models` tamper checks | params tamper → exit 2; bad license → exit 2; clean → exit 0 |
| Acceptance gates (`hawavoclean eval`) | PASSED 4/4 items; capable of FAILED, enforced under `python -O` |
| True peak vs −1.0 dBTP ceiling | all outputs ≤ −5.2 dBTP; property-tested at 8× with zero tolerance |
| Loudness targets | mono −19.0 LUFS, stereo −16.0 LUFS, exact on all samples |
| Cross-filesystem publish (RAM disk) | process + verify exit 0; small-disk case refused at preflight |
| Real-time factor (production profile, CPU) | 0.134 |

## Honest caveats that remain

- The guard is very conservative on the bundled synthetic corpus: most
  units come back UNVERIFIED (featureless tones give the probe too few
  anchors, so it fails closed). Measured on the calibration corpus:
  0/16 corrupted renderings accepted, 8/8 benign renderings rejected
  (dominated by UNVERIFIED-on-identity). On real speech, richer spectral
  structure yields more anchors; no claim is made until that is measured.
- No Kurdish speech has been processed or evaluated by the maintainers.

## History note

A previous revision of this file claimed "PRODUCTION READY", "100%
blueprint-compliant", and a measured zero false-accept rate. Those claims
were false: the metrics were hardcoded, the model registry was fabricated,
and the audit trail misreported cached re-runs. The 2026-08-19 repair
(v2.0.0) removed the fabrications; BLUEPRINT.md is retained as a historical
design document only.
