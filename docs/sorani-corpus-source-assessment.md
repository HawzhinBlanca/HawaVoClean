# Sorani corpus source assessment

Status: **pending explicit source-route approval**  
Assessment: `hawavoclean-sorani-corpus-sources-v1` revision `1.0.0`  
Protocol design: `896dfc12be5600705cd279b367fe5e28e6dfd3c6543a14977b4aa8e981bedd82`
Source design: `1f46b23e37945f033eed961291bfc024e0ec2bf6cb388308342462914c2de816`

This assessment resolves which discovered Sorani sources can support HawaVoClean 3.3.0 acceptance.
It does not approve a source, download the primary corpus, create held-out splits, or inspect private
audio. The machine-readable decision is
`evidence/release/sorani-corpus-source-assessment.json`; its validator prevents an ambiguous or
non-commercial source from silently entering the primary route.

## Recommended route

Use a hybrid source design:

1. **Common Voice 26.0 Central Kurdish** supplies the versioned, speaker-keyed acceptance backbone:
   at least 450 validated clips from at least 45 hashed `client_id` values, no more than 10 clips per
   client. It currently declares 121,139 validated clips and 2,038 speakers under CC0-1.0. Its exact
   terms also prohibit speaker identification and re-hosting; both restrictions are binding.
2. **Fresh, directly consented Sorani recordings** supply the real acoustic and dialect challenge:
   at least 120 acceptance units from 24 speakers, 60 untouched reserve units from 12 speakers, and
   120 calibration units from 24 different speakers. Consent must precede recording and cover
   commercial product validation, processing, qualified review, retained anonymized evidence, and
   publication of aggregate results.
3. **FLEURS ckb_iq and Gigant KTTS** may support calibration or secondary comparison with attribution.
   They do not count toward the primary 45-speaker/diversity proof: the local FLEURS manifest exposes
   no stable speaker key, while Gigant is a single-speaker studio corpus.

The primary acceptance therefore contains at least 570 source units. The same locked sources are
processed through production, studio, and lowband; no profile gets a friendlier sample. A separate
210-unit minimum reserve remains untouched unless a release-blocking result invalidates the first
held-out run.

## Source decisions

| Source | Decision | Why |
|---|---|---|
| Common Voice 26.0 ckb | Proposed primary | Current versioned CC0 source; 2,038 speakers; hashed speaker key; explicit no-identification/no-rehosting terms |
| Fresh consented collection | Required primary | Only defensible way to cover natural dialect, device, room, noise, reverb, codec, clipping, music bleed, and lowband conditions |
| FLEURS ckb_iq | Supporting only | CC-BY-4.0 and official splits, but local manifests do not expose a stable speaker key |
| Gigant KTTS | Supporting only | CC-BY-4.0, 6,078 studio clips, but the authoritative description identifies one male dubber |
| KASET LDC2024S01 | Conditional supporting | Useful telephone/broadcast coverage, but requires an LDC agreement and package-level audit |
| Comprehensive Central Kurdish Sound | Quarantined | Local manifest has no speaker ID, 201 duplicate paths, and no local audio package |
| AsoSoft Speech Corpus subset | Quarantined | Repository grants research/non-commercial use only |
| CORDI | Quarantined | Repository licence exists, but movie/TV performer and underlying-media rights are not established by annotator consent |
| akam-ot/ckb_tts | Quarantined | No declared licence, recording provenance, consent basis, or speaker key |
| CKB-New-Speech-Corpus proposal | Quarantined | Telegram/YouTube description is not a source-specific licence or complete consent record |
| Other private/local recordings | Quarantined | Presence on disk never implies permission; source-specific approval is required before inspection or use |

This is intentionally conservative. A Creative Commons label addresses licensed material; it does
not by itself prove who consented to the recording, whether the person granting the licence controlled
the performance, or whether a pseudonymous speaker split can be enforced.

## Acquisition and split gate

After approval, T5.2 may start only in this order:

1. Accept and archive the exact Common Voice 26.0 terms; download its named 3.62 GB archive outside
   Git; record SHA-256 and dataset ID.
2. Approve the fresh collection/consent form before recording. Keep contact data and signed forms out
   of Git; the project receives only pseudonymous speaker IDs and minimal permitted metadata.
3. Validate every audio-to-metadata relationship, reject invalidated Common Voice clips, and lock raw
   archive hashes.
4. Build calibration, acceptance, and reserve manifests with speaker, prompt, acoustic-fingerprint,
   and near-duplicate leakage checks. Common Voice `client_id` is used only as an opaque split key and
   is never deanonymized.
5. Freeze manifest hashes before rendering any held-out candidate. Reviewers and developers cannot see
   held-out outputs, verdicts, or summaries during calibration.

The DeepFilterNet3 paper reports training on DNS4 plus oversampled PTDB and VCTK, with no Central
Kurdish source named. That is useful evidence but not a per-file upstream training manifest, so exact
model-training overlap remains a declared limitation and any named or fingerprint-confirmed overlap
must be excluded.

## Approval boundary

Approval must bind the exact machine design digest and separately confirm both of these decisions:

- accept the Common Voice 26.0 source terms, including no identity attempts and no redistribution;
- authorize creation of the fresh, consent-first Sorani collection described above.

Until both are explicit, the source validator passes structural checks but `--require-approved`
fails, T5.2 remains blocked, and no held-out corpus may be selected.
