# Guard Calibration Protocol

## Split Discipline

- **Calibration Set**: Exclusively used to fit guard thresholds ensuring 0.0 false accepts on counterexamples.
- **Development Set**: Used for candidate comparison and engineering tuning.
- **Acceptance Set**: Hash-locked prior to final selection. Never used to fit thresholds or code.
- **Corruption Set**: Realistic phonetic counterexamples (consonant splicing, word deletion, timing warp, spectral holes).

## Threshold Locking

Production refuses to start without a valid `models/guard-calibration.json` matching the locked calibration ID in `models/production-core.lock.toml`.
