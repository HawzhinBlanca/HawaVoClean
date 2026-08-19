# ADR 0001: Single Frozen Enhancement Core in Production Runtime

## Context
Multiple enhancement paths in production create timbre inconsistencies,
non-deterministic failure modes, and dependency sprawl.

## Decision
Freeze exactly one enhancement core, locked in
`src/hawavoclean/resources/models/production-core.lock.toml`. The shipped
core is deterministic DSP (a decision-directed Wiener filter); its
provenance is its parameter set, hash-verified against the implementation
at preflight and by `hawavoclean audit-models`.

## Consequences
- Single runtime path with verifiable provenance.
- Predictable acoustic behavior across long-form dialogue.
- A future neural core replaces the lockfile contents (weights + digests)
  without changing the invariant: one core, hash-locked, audit-verified.

## History
The original revision of this ADR described the core as "one verified
neural enhancement core". No neural core was ever integrated; the lockfile
pointed at weights that did not exist. The invariant survives; the
description now matches the implementation.
