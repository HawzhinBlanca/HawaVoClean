# ADR 0003: Fail-Closed Passthrough for Speech Units

## Context
Worker crashes, OOMs, model timeouts, or ambiguous speech units must never abort long-form dialogue processing or cause digital silence.

## Decision
Errors or unverified verdicts produce unit-level original audio passthrough with transparent reporting.

## Consequences
- Long-form jobs always complete safely.
- No partial files or digital black silence.
- Full auditability via flagged review timecodes.
