# ADR 0001: Single Frozen Enhancement Core in Production Runtime

## Context
Multiple enhancement models in production create timbre inconsistencies, non-deterministic failure modes, and massive dependency sprawl.

## Decision
Freeze exactly one verified neural enhancement core in `models/production-core.lock.toml`. Candidate comparison exists only in the offline research harness.

## Consequences
- Single runtime path.
- Minimal VRAM footprint.
- Predictable acoustic behavior across long-form dialogues.
