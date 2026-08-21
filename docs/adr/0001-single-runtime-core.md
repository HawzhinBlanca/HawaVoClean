# ADR 0001: Single Frozen Enhancement Core in Production Runtime

**Status:** Superseded for the shipped inventory by ADR 0006; retained for the one-core-per-pass
invariant and historical rationale.

## Context
Multiple enhancement paths in production create timbre inconsistencies,
non-deterministic failure modes, and dependency sprawl.

## Original decision
Freeze exactly one production-profile enhancement core, locked in
`src/hawavoclean/resources/models/production-core.lock.toml`. That core is
deterministic DSP (a decision-directed Wiener filter); its provenance is its
parameter set, hash-verified against the implementation at preflight and by
`hawavoclean audit-models`.

## Consequences
- One selected runtime path per pass with verifiable provenance.
- Predictable acoustic behavior across long-form dialogue.
- Later neural profiles received separate core IDs/locks instead of replacing
  production. The surviving invariant is one selected, hash-locked,
  audit-verified core per pass—never an implicit ensemble.

## History
The original revision described production as "one verified neural enhancement
core" even though its lock pointed at weights that did not exist. The 2026-08-19
honesty rebuild corrected production to real DSP. Studio and lowband were later
integrated under separate verified locks; ADR 0006 owns that multi-profile
inventory.
