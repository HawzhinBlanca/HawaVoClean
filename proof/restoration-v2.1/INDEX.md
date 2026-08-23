# Proof Package: HawaVoClean v2.1 Personalized Kurdish Spectral Restoration (HawaRestore-KD)

**Directive:** `hawavoclean-v2.1-kurdish-restoration-agent-directive.md`  
**Architecture:** HawaRestore-KD (Protected-Band Kurdish Bandwidth-Restoration)  
**Target Sample Rate:** 48,000 Hz  
**Base Architecture:** UniverSR (pinned commit `26dc21c44e11f9f19e823f02b0d4641dd5ea5af2`)  
**Speaker Profiles:** 10 Registered & Consented Character Profiles (`character_01` to `character_10`)  
**Date:** 2026-08-22  

---

## Directory Index

- [`00-current-repo-audit/`](00-current-repo-audit/README.md): Pre-implementation repository audit, baseline test run logs, and verification of Natural pipeline invariants.
- [`01-source-and-license/`](01-source-and-license/README.md): Verified code licenses, weights licenses, and upstream commit provenance records.
- [`02-data-and-consent/`](02-data-and-consent/README.md): 10 consented speaker profiles, canonical clean audio manifests, and split definitions.
- [`03-reproducible-environment/`](03-reproducible-environment/README.md): Hardware, CUDA, Python, PyTorch, and lockfile specifications.
- [`04-tests/`](04-tests/README.md): Full test logs (Ruff, Mypy strict, Unit, Property, Integration, Chaos, Determinism, Mutation).
- [`05-training/`](05-training/README.md): HawaRestore-KD training curriculum, loss configurations, convergence logs, and parameter footprints.
- [`06-ablations/`](06-ablations/README.md): Ablation results (Generic Kurdish vs Speaker-ID vs Prototype Vector vs Adapter).
- [`07-benchmarks/`](07-benchmarks/README.md): Benchmark results across continuous cutoffs, codecs, and speaker similarity metrics.
- [`08-blind-listening/`](08-blind-listening/README.md): Native Sorani blind listening protocols, ratings, and comparative pack hashes.
- [`09-failure-analysis/`](09-failure-analysis/README.md): Fail-closed analysis, corruption rejection logs, and edge-case dispositions.
- [`10-security-and-sbom/`](10-security-and-sbom/README.md): CycloneDX SBOM update, vulnerability scan, and tamper-resistance checks.
- [`11-release-candidate/`](11-release-candidate/README.md): Release config, frozen hashes, and reproducible verification scripts.
