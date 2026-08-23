# 08 - Listening Tests

## Status: no human listening evidence exists

**No human listening test — blind, sighted, formal, or informal — has been conducted on
Restore-mode output.** No listener ratings, ABX trials, MUSHRA scores, or panel results exist
anywhere in this repository, and none are claimed. Any release decision made today rests solely
on automated objective metrics.

## What automated evidence exists

Objective verification runs via `research/restoration/benchmark.py` (log-spectral distance,
protected-band RMS deviation, speaker cosine similarity against profile prototype vectors) and
`hawavoclean restore-doctor` (preflight and profile diagnostics). Two limits apply:

1. These metrics are currently computed on the **10 synthetic development voice fixtures**, not
   on recorded Kurdish speech.
2. Objective metrics cannot verify linguistic content. They do not and cannot substitute for
   human evaluation of Sorani intelligibility, timbre naturalness, or speaker identity.

## Planned human evaluation

Subjective evaluation is governed by the locked Sorani human acceptance protocol:

- **Protocol document**: `docs/sorani-evaluation-protocol.md`
  (`hawavoclean-sorani-acceptance-v1`, revision `1.0.0`; machine lock at
  `evidence/release/sorani-evaluation-protocol.json`).
- **Precondition**: user checkpoint **U3** (`docs/true-10-plan.md`) must approve every corpus
  source and its rights, and identify qualified Sorani reviewers, before any candidate material
  is generated or heard.
- **Shape of the evaluation** (defined in full by the protocol, which is authoritative over this
  summary): held-out Sorani source units from held-out speakers, two Sorani-capable reviewers
  independently comparing every source/candidate pair, a third qualified reviewer adjudicating
  disagreements blind to profile identity, and hard stop conditions including any single
  confirmed content alteration.

Until that protocol has been executed and its results recorded, Restore mode has **zero**
subjective quality or intelligibility evidence, and no claim to the contrary should appear in
any release material.
