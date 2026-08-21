# Sorani Human Acceptance Protocol

Status: **valid draft, pending explicit user approval; no held-out results examined**  
Protocol: `hawavoclean-sorani-acceptance-v1` revision `1.0.0`  
Machine lock: `evidence/release/sorani-evaluation-protocol.json`

This protocol is the boundary between an extensively tested audio program and a product whose
strongest promise—preserving Sorani spoken content—has actually been tested by qualified people.
The four bundled synthetic files and the automated `hawavoclean eval` gate do not provide that proof.

## Release decision in one page

The exact 3.3.0 release artifacts will render `production`, `studio`, and `lowband` outputs. Each
compatible profile must have at least **450 held-out Sorani source units**, from at least **45 held-out
speakers**, with no speaker contributing more than ten units. Calibration, acceptance, and reserve
speakers are disjoint. Two Sorani-capable reviewers independently compare every source/candidate pair;
a third qualified reviewer adjudicates every disagreement while the profile identity remains hidden.
ASR may only direct attention to a timecode. It can never clear linguistic content.

Release is stopped by any of the following:

- one confirmed introduced, deleted, substituted, repeated, or meaningfully reordered word or content
  element in shipped audio;
- one guard false accept, meaning changed content passed through to the shipped output;
- one processing-induced severe or unusable artifact;
- failure of the locked intelligibility or P.835 SIG/BAK/OVRL non-inferiority gates;
- failure to demonstrate at least one locked value improvement for an intended use of each profile;
- leakage, unblinding, post-output exclusion, replacement of a failed unit, or a render that is not the
  exact candidate proposed for release.

If code, profiles, thresholds, models, or output-affecting dependencies change after evaluation, the
exposed holdout cannot be reused for the final claim. The final run moves to the untouched reserve or a
newly rights-cleared holdout.

## Why 450 units per profile

For zero observed events in `n` independent units, the exact one-sided binomial 95% upper bound is
`1 - 0.05^(1/n)`. At `n = 450`, that is 0.6636% for one profile. Conservatively assigning alpha
`0.05 / 3` to each of the three shipped profiles gives a simultaneous upper bound of 0.9058%, below
1%. This is stronger than the original 300-unit floor.

That calculation does **not** make repeated speech from one speaker independent or turn a convenience
sample into the Sorani population. The corpus therefore caps speaker concentration, reports
speaker-block sensitivity and effective sample size, and must add units if clustering weakens the
bound. No release document may quote the sub-1% figure without those qualifications.

## Corpus and split lock

Before generating candidates, checkpoint U3 must approve every source and its rights for commercial
product validation and derivative outputs, plus the named reviewer roles. A data custodian then freezes
four hash-locked, speaker-disjoint roles: reviewer training, calibration, held-out acceptance, and
untouched reserve. The lock includes audio hashes, pseudonymous speaker IDs, transcripts, dialect,
capture, environment, degradation, critical-content tags, rights class, and access location.

Coverage must include Slemani, Hawleri, Garmiani, and another documented Sorani variety where rights-
cleared inventory permits; varied speakers and recording chains; clean, noise, reverb, codec, clipping-
risk, music-bleed, and low-bandwidth material; and meaning-sensitive cases such as negation, numbers,
names, borrowed words, short function words, and word-final consonants. Each predefined primary
stratum needs at least 30 units or must be enlarged and relocked before rendering. An underfilled
stratum is reported as underpowered, never silently pooled after results are visible.

Raw audio, identity documents, consent records, and the reviewer-identity mapping remain in controlled
storage outside Git. Only hashes, anonymous IDs, redacted rights classifications, verdicts, analysis,
and evidence are eligible for the repository.

## Human content review

The source transcript is resolved and locked before candidate generation. Reviewers must demonstrate
Sorani fluency, record dialect competency, pass an alteration-detection qualification set outside the
evaluation splits, and have no role in developing or tuning the candidate they review.

For each anonymous source/candidate comparison, two reviewers independently record:

1. whether spoken content changed, with exact token and timecode;
2. keyword/content intelligibility against the locked transcript;
3. processing artifacts on the locked 0–4 severity scale;
4. whether the guard shipped enhanced audio or reverted to the original.

For every unit the guard ultimately reverts, the evaluation harness also captures the final discarded
processed candidate whose rejection caused original selection. That diagnostic capture must be proved
bitwise non-interfering with the exact release path. Two reviewers compare it to the source under the
same profile-blind rules; a content-safe discarded candidate counts as a guard false revert. The
diagnostic candidate is never presented as shipped audio or silently substituted into quality results.

Disagreement or suspicion triggers profile-blind third review with a reasoned adjudication. The system
mapping opens only after all verdicts, exclusions, QC decisions, analysis inputs, and table shells are
hash-locked. Report raw agreement, Cohen's kappa for binary content verdicts, Krippendorff's alpha for
ordinal artifact ratings, and the adjudication rate by profile and stratum.

## Blinded quality and intelligibility test

At least 32 valid Sorani listeners provide at least 16 valid ratings per item/condition through balanced
incomplete randomized blocks. Original and candidate order is counter-balanced. The P.835 sequence
rates speech signal (SIG), background intrusiveness (BAK), and overall quality (OVRL), with
`SIG–BAK–OVRL` and `BAK–SIG–OVRL` scale orders counter-balanced. Separate blinded pairs record
candidate, original, or tie preference.

Qualification, headphone/playback checks, quiet-environment or laboratory-chain rules, anchors,
repeat trials, attention rules, completion-time limits, session-length limits, and breaks are frozen
from training/calibration data. The pilot may increase listener or rating counts to achieve 90% power
at the locked margins; it cannot reduce the floors. P.807 is only a design reference for a Sorani-
adapted critical-token task, not a claim that an English word-pair test transfers without validation.

## Locked analysis and thresholds

Effects are paired candidate-minus-original estimates per profile. A hierarchical mixed-effects model
accounts for speaker, source unit, and listener; speaker-block and listener-block bootstrap is the
sensitivity check, and the more conservative release conclusion wins. Release gates use one-sided 95%
intervals, descriptive results use two-sided 95% intervals, and Holm controls the 5% primary family
within each profile. Missing values are not imputed.

Every profile must meet all of these:

- zero adjudicated content changes and zero guard false accepts;
- zero processing-induced artifacts rated severe or unusable;
- lower confidence bound above -0.02 for intelligibility accuracy difference;
- lower confidence bounds above -0.25 MOS for SIG, BAK, and OVRL;
- on intended noisy strata, BAK superiority lower bound above zero;
- after all safety and non-inferiority gates pass, either OVRL or intended-stratum BAK improvement has
  a Holm-adjusted lower bound above zero, or blinded preference (ties count one half) has a lower bound
  above 0.5.

Every predefined stratum and worst case is reported. Aggregate quality cannot hide a content change,
severe artifact, or underpowered dialect/noise condition. Content-safe candidates discarded by Guard A
or Guard B are reported with uncertainty as the guard false-revert efficiency/no-op outcome, but they
are not treated as evidence that shipped speech changed.

## Exclusion and stopping rules

Allowed source exclusions are inadequate rights, corruption before product processing, no audible
speech, duplication/split leakage, or a source transcript that blinded adjudication cannot resolve.
They are applied before candidate generation. Their IDs and reasons remain in the audit trail.
Technical processing failure is a failed gate. A failed item is never replaced.

One confirmed content change immediately freezes and hashes the evidence state, quarantines the
candidate, and blocks the release claim. Investigation may determine scope on already exposed material,
but fixes and tuning use only development/calibration data. Final acceptance then uses untouched
evidence. Threshold tuning on held-out labels, dropping a failed profile/stratum, and reporting only
aggregate means are prohibited.

## Standards and version reconciliation

The design uses ITU-T P.800 for general subjective-test controls, P.835 for separate SIG/BAK/OVRL
judgments, and P.808 if remote listeners are used. The ITU database marks P.835 (07/2026) in force but
not yet published at this protocol freeze; the accessible 11/2003 procedure is the execution baseline.
Before listener training, the test owner must compare the published 07/2026 text. A material procedural
change requires protocol revision, a new design digest, and fresh approval—never a silent update.

The zero-event rationale uses the exact binomial form associated with Hanley and Lippman-Hand's
zero-numerator discussion, not the rounded `3/n` shortcut. The machine lock contains direct source URLs
and the exact formulas and numbers.

## Approval and execution boundary

Structural validation is intentionally different from approval:

```bash
uv run python -m scripts.validate_sorani_protocol --print-digest
uv run python -m scripts.validate_sorani_protocol --require-approved
```

The first command must pass for this draft. The second must fail until the user explicitly approves the
design digest. Approval records the approver, UTC time, and exact digest without changing the design.
T5.1 remains open until that happens; T5.2 cannot start until U3 separately approves source rights and
identifies the two-reviewer/adjudicator capacity. No held-out sample may be selected, rendered, or
scored before those boundaries are satisfied.
