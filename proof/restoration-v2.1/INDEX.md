# Proof Package: HawaVoClean Personalized Kurdish Spectral Restoration (HawaRestore-KD)

**Directive:** `hawavoclean-v2.1-kurdish-restoration-agent-directive.md`  
**Architecture:** HawaRestore-KD (Protected-Band Kurdish Bandwidth-Restoration)  
**Target Sample Rate:** 48,000 Hz  
**Base Architecture:** UniverSR (pinned commit `26dc21c44e11f9f19e823f02b0d4641dd5ea5af2`)  
**Speaker Profiles:** 10 synthetic development voice fixtures (`character_01` to `character_10`); real consented speaker enrollment is pending user checkpoint U3  
**Date:** 2026-08-22

---

## Evidence Status — read this first

This pack documents the Restore-mode engineering scaffolding exactly as it exists in this
repository. Three limits apply to every page below:

1. **The committed checkpoint (`models/hawarestore-kd/hawarestore_kd.pt`) is an engineering
   artifact, not a speech model.** It was produced by
   `research/restoration/train/train_hawarestore.py`, which trains on synthetic sine-harmonic
   simulation signals (`KurdishSimulationDataset`) for a small number of epochs. No recorded
   Kurdish speech — consented or otherwise — has been used for training. Real-corpus training
   is scheduled behind user checkpoint U3 (`docs/true-10-plan.md`).
2. **The 10 speaker profiles are synthetic development fixtures** generated deterministically by
   `research/restoration/profiles_builder.py`. Their consent records are structural placeholders
   exercising the enrollment schema; no real speaker has been enrolled or consented.
3. **No human listening evidence exists.** Subjective evaluation is specified by the locked
   Sorani protocol (`docs/sorani-evaluation-protocol.md`) and runs only after U3 approval.
   No ablation study has been run, and no ablation results are claimed anywhere in this pack.

---

## Directory Index

- [`01-source-and-license/`](01-source-and-license/README.md): Upstream code/weights license matrix, pinned commit provenance, and truthful training-data status of the committed checkpoint.
- [`02-architecture-and-adr/`](02-architecture-and-adr/README.md): Architecture summary aligned with ADR 0008 (mode separation, protected band, candidate ladder, Guard R, fail-closed fallback).
- [`03-speaker-profiles-and-consent/`](03-speaker-profiles-and-consent/README.md): The 10 synthetic development voice fixtures, their manifests and hashes, and the enrollment/consent schema they exercise.
- [`04-training-and-splits/`](04-training-and-splits/README.md): Training pipeline description and split strategy for the synthetic simulation data.
- [`05-model-artifacts-and-locks/`](05-model-artifacts-and-locks/README.md): Committed checkpoint hashes and lockfile verification.
- [`06-pipeline-and-guard-integration/`](06-pipeline-and-guard-integration/README.md): How Restore mode and Guard R integrate with the pipeline and report schema.
- [`07-benchmarks/`](07-benchmarks/README.md): Objective benchmark methodology and results (LSD, protected-band RMS, speaker similarity), computed on the synthetic fixtures via `research/restoration/benchmark.py`.
- [`08-listening-tests/`](08-listening-tests/README.md): Listening-test status: none conducted; planned under the locked Sorani protocol after U3.
- [`09-doctor-and-diagnostics/`](09-doctor-and-diagnostics/README.md): `hawavoclean restore-doctor` preflight and profile diagnostics.
- [`10-verification-logs/`](10-verification-logs/README.md): Test suite execution evidence (lint, types, unit, property, integration, chaos).
- [`11-user-guide/`](11-user-guide/README.md): User and operator guide for Restore mode.

The raw output of `research/restoration/benchmark.py` lives at
`07-benchmarks/benchmark_results.json`, next to its interpretation and limits; the
former root-level copy was a stale duplicate from an unrecorded DSP-fallback run
and has been removed.
