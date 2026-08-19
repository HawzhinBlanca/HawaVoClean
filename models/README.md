# Model Artifacts

Runtime artifacts moved into the package: `src/hawavoclean/resources/models/`
(overridable via `HAWAVOCLEAN_MODEL_DIR`). This directory intentionally holds
no artifacts.

Status, honestly stated:

- The production enhancement core is deterministic DSP (a decision-directed
  Wiener filter). It has no weights. Its provenance is its parameter set,
  hashed in `production-core.lock.toml`.
- **No external neural models have been evaluated.** A previous revision of
  this directory contained a registry of "evaluated" candidates with
  fabricated commit hashes and license claims; it was removed rather than
  corrected, because none of the evaluations had happened.
- The guard calibration artifact carries engineering-default thresholds and
  says so. Measured metrics only appear after a real `hawavoclean calibrate`
  run, with measurement provenance attached.
