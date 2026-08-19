# ADR 0002: Two-Pass Guard Architecture (Guard A & Guard B)

## Context
Post-enhancement local finishing (EQ, de-essing, level riding) can degrade
the signal if unconstrained, independently of the enhancer.

## Decision
Guard twice, with the same spectral-change instrument:
- **Guard A** validates the enhancer candidate against the original audio.
- **Guard B** validates the locally finished unit against the pre-finish
  accepted rendering.

## Consequences
- Finishing cannot silently introduce spectral degradation; a Guard B
  rejection bypasses finishing for that unit.
- The guard's scope limit applies to both passes: it detects spectral
  change, not linguistic change (see docs/fidelity-guard.md).

## History
The original revision described Guard A as validating "against reference
ASR". There is no ASR in this system; both passes compare spectral
signatures.
