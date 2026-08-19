# ADR 0002: Two-Pass Guard Architecture (Guard A & Guard B)

## Context
Post-enhancement local finishing (EQ, de-essing, level riding) can inadvertently degrade consonants or weaken intelligibility if unconstrained.

## Decision
Guard twice:
- **Guard A**: Validates the neural enhancer output against reference ASR.
- **Guard B**: Validates the locally finished unit against the pre-finish accepted timeline.

## Consequences
- Finishing cannot silently introduce phonetic degradation.
- Unit reverts to pre-finish accepted audio if Guard B rejects finishing.
