# Hawzhin VoiceClean - Model Artifacts & Registry

This directory contains the production model lockfile, the registered candidate models catalog, and the signed/calibrated guard thresholds.

## Files

- `production-core.lock.toml`: The authoritative, immutable lockfile for the frozen neural enhancement core used in production runtime.
- `model-registry.toml`: Registry of all benchmarked candidates and external baselines.
- `guard-calibration.json`: Hash-locked thresholds and parameters fitted during the calibration phase on the calibration dataset.
