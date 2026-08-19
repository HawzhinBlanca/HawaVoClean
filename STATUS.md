# Implementation Status

Numbers below are measured, not asserted. Re-measure with
`bash scripts/run_release_checks.sh` and `python scripts/mutation_gate.py`.

## What this system is

An offline dialogue cleanup tool: Wiener spectral denoiser + spectral-change
guard + deterministic finishing + BS.1770 mastering. No neural models, no
speech recognition — see README "What it is not".

## Verification results (2026-08-19, cold state)

| Gate | Result |
|---|---|
| Test suite | see CI output — run `pytest` |
| Mutation gate | run `python scripts/mutation_gate.py` — must report 12/12 |
| Ruff format / lint | clean |
| Mypy `--strict` | clean |
| Doctor (from any CWD) | exit 0 |
| Acceptance gates | can genuinely FAIL; enforced under `python -O` |

## History note

A previous revision of this file claimed "PRODUCTION READY", "100%
blueprint-compliant", and a measured zero false-accept rate. Those claims
were false: the metrics were hardcoded, the model registry was fabricated,
and the audit trail misreported cached re-runs. The 2026-08-19 repair
removed the fabrications; BLUEPRINT.md is retained as a historical design
document only.
