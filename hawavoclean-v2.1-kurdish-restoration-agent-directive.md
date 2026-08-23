# HawaVoClean v2.1 — Personalized Kurdish Spectral Restoration Directive

**Status:** implementation-authoritative amendment  
**Date:** 2026-08-22  
**Audience:** the AI coding agent that already implemented the current HawaVoClean application  
**Applies to:** the existing HawaVoClean repository and the `hawzhin-voiceclean-master-blueprint-v2.md` specification  
**Goal:** add a Kurdish- and speaker-aware restoration capability without weakening the existing Natural mode, fidelity guards, fail-safe behavior, reproducibility, or auditability.

---

## 0. Read this before changing code

You are updating an existing production-oriented audio application. Do not replace, rewrite, or destabilize the working Natural pipeline.

The existing Master Blueprint v2.0 remains authoritative except where this amendment explicitly adds a new **Personalized Restore** mode. All existing requirements concerning source preservation, deterministic execution, failure passthrough, immutable reports, model/checkpoint hashing, testing, security, and evidence-based release gates remain in force.

This amendment does **not** authorize:

- whole-voice regeneration;
- TTS or voice-conversion output in the production path;
- automatic replacement of spoken content;
- a runtime model zoo or model router;
- ten separate full restoration models;
- automatic restoration of healthy full-band audio;
- hidden cloud calls;
- silent fallback;
- deleting, weakening, or bypassing existing Sorani fidelity checks.

A feature is not complete because code exists or a demo sounds impressive. It is complete only when the Definitions of Done and locked acceptance gates below pass with reproducible evidence.

---

# 1. Final research decision

## 1.1 Production architecture

Build a new model and subsystem called:

# **HawaRestore-KD**

HawaRestore-KD must be a **protected-band, speaker-conditioned Kurdish bandwidth-restoration model**.

Use the official **UniverSR** implementation as the practical production foundation, pinned initially to:

```text
repository: woongzip1/UniverSR
commit: 26dc21c44e11f9f19e823f02b0d4641dd5ea5af2
code license: MIT
pretrained weights currently identified as: CC-BY-4.0
```

Do not ship the generic checkpoint without a provenance record and attribution. Verify the license directly from the exact downloaded model revision and store the evidence in the repository.

HawaRestore-KD must borrow the strongest useful ideas from the August 2026 **AnyBand** paper, without claiming to reproduce AnyBand:

- continuously variable cutoff conditioning;
- observed spectrum as the acoustic prompt;
- training loss only on the missing band;
- F0 and voiced/unvoiced conditioning;
- explicit frequency-axis modeling;
- an easy-to-balanced cutoff curriculum;
- cross-band envelope and harmonic-consistency objectives;
- final output constructed from the trusted observed band plus only the generated missing band.

As of this directive, no public, reproducible AnyBand implementation was located. AnyBand is therefore a **design reference**, not a dependency and not a benchmark executable.

## 1.2 Why this is the production choice

UniverSR is the best lean starting point because it already:

- works directly in the complex-STFT domain;
- generates only high-frequency spectral content;
- copies the known low-frequency region into the final spectrum;
- avoids a separate neural vocoder;
- supports 8/12/16/24 kHz inputs to 48 kHz;
- exposes training and inference code;
- is small enough to train and iterate on the existing dual-RTX-3090-Ti workstation;
- is structurally safer for exact Sorani speech than a model that reconstructs the full waveform from semantic or codec tokens.

The production model must improve this foundation for continuous real-world cutoffs, Kurdish acoustic structure, and the ten known speakers.

## 1.3 AnyEnhance decision

Do **not** use AnyEnhance as the production restoration core.

Reality check:

- The lightweight public `viewfinder-annn/AnyEnhance-v1` repository is not the complete prompt-guided/self-critic system described in the paper.
- A fuller prompt-guided implementation was historically merged into Amphion at commit `c0277229f83ea685db15611fd81a5396c571e264`, including prompt input and self-critic code.
- That fuller path reconstructs speech through acoustic-code generation, operates at 44.1 kHz, and has more freedom to alter phonetic detail.
- The commercial rights for the exact full pretrained weights are not sufficiently clear from the currently located sources.

Therefore:

- integrate full AnyEnhance only as an **isolated research comparator** if the exact code, weight provenance, and license are verified;
- never make it the default;
- never use it automatically;
- do not add a user-facing Rescue mode in this phase;
- if it changes a Sorani phoneme or fails licensing review, remove it from the release package entirely.

## 1.4 Other model decisions

Use these only as offline baselines:

- stock UniverSR speech and audio checkpoints;
- MossFormer2_SR_48K;
- Adobe Podcast exports;
- iZotope RX exports;
- the current Natural HawaVoClean result;
- full AnyEnhance only if reproducibly available and legally usable for internal evaluation.

CodecFlow and AnyBand are research references unless official reproducible code and acceptable weights appear. Do not reimplement every paper.

---

# 2. Product behavior

HawaVoClean must expose two separate modes.

## 2.1 Natural mode — unchanged

```text
voiceclean process INPUT --mode natural
```

This remains the default and must preserve current behavior:

```text
input
→ existing conservative enhancement core
→ existing Guard A
→ deterministic finishing
→ existing Guard B
→ output/report
```

No restoration model is loaded in Natural mode. Existing golden outputs must not drift unless an explicitly reviewed bug fix requires it.

## 2.2 Personalized Restore mode — new, explicit, opt-in

```text
voiceclean process INPUT \
  --mode restore \
  --speaker-id CHARACTER_ID \
  --cutoff auto
```

Restore mode is allowed only when:

- the speaker is one of the registered, consented profiles;
- the source is genuinely bandwidth-limited or the user supplies an explicit cutoff;
- the Natural candidate has already passed the existing guard;
- the restoration guard can execute successfully.

The path is:

```text
original source
      │
      ▼
existing Natural pipeline through Guard A
      │
      ▼
Natural-safe candidate
      │
      ▼
bandwidth/cutoff analysis
      │
      ├── healthy or uncertain ───────────────► Natural candidate
      │
      ▼
HawaRestore-KD
(generate missing band only)
      │
      ▼
protected-band merge
      │
      ▼
Restoration Guard R
      │
      ├── fail / error / uncertainty ─────────► Natural candidate
      │
      ▼
existing deterministic finishing
      │
      ▼
existing Guard B
      │
      ├── fail ───────────────────────────────► Natural candidate or original,
      │                                         according to existing policy
      ▼
48-kHz restored output + complete report
```

Restore mode must never be silently selected. Automatic speaker recognition must not choose the profile. Require an explicit `speaker-id`; incorrect profile selection must be treated as user/configuration error and reported.

## 2.3 Output format

- Natural mode keeps the current delivery format.
- Restore mode outputs **48 kHz** because high-frequency reconstruction cannot be represented in a low-rate file.
- Preserve duration, channel mapping, timecode, and synchronization exactly.
- For a 48-kHz input, output sample count must equal input sample count.
- For a lower-rate input, output duration must be sample-exact after the documented rational resampling to 48 kHz.
- Record original sample rate, effective cutoff, output sample rate, resampler, delay compensation, and expected sample-count transformation in the report.

---

# 3. Speaker personalization design

## 3.1 One shared model, not ten models

Train one HawaRestore-KD backbone for all ten characters.

Condition it using:

1. an explicit learned `speaker_id` embedding;
2. a precomputed clean-reference prototype vector;
3. optional tiny per-speaker adapters only if a locked ablation proves a material improvement.

Do not build ten independent full checkpoints. They waste data, complicate deployment, and are more likely to overfit.

## 3.2 Speaker profile schema

Add a versioned, hash-validated profile:

```json
{
  "schema_version": "1.0",
  "speaker_id": "character_01",
  "display_name": "Character 01",
  "consent_record": "consent/character_01.json",
  "canonical_audio_manifest": "profiles/character_01/canonical.jsonl",
  "canonical_audio_sha256": ["..."],
  "profile_embedding_path": "profiles/character_01/profile.safetensors",
  "profile_embedding_sha256": "...",
  "f0_statistics": {
    "median_hz": 0.0,
    "p05_hz": 0.0,
    "p95_hz": 0.0
  },
  "training_split_id": "restore-corpus-v1",
  "adapter": null,
  "created_by_commit": "...",
  "notes": ""
}
```

The loader must reject:

- missing consent metadata;
- unknown schema versions;
- hash mismatches;
- profile/model incompatibility;
- a profile built from synthetic-only audio;
- a profile whose canonical clips overlap the locked acceptance set.

## 3.3 Canonical voice source

The canonical identity must come from **real, consented, clean recordings** of the speaker.

Existing TTS/voice-clone models may be used only for clearly labeled augmentation experiments. They must never be:

- the clean restoration target;
- the canonical speaker identity;
- the acceptance reference;
- mixed into the real acceptance corpus;
- treated as evidence that a restoration is faithful.

A cloned voice contains the clone model’s errors and biases. Training against it would teach HawaRestore-KD to reproduce those errors.

## 3.4 Speaker embedding

For the first ablation:

- use a learned ten-way speaker-ID embedding as the primary conditioning;
- precompute reference prototypes from real clean clips;
- evaluate ERes2NetV2 from `modelscope/3D-Speaker`, initially pinned to commit `065629c313eaf1a01c65c640c46d77e61e9607b4`, as a frozen speaker-similarity metric and optional profile encoder;
- verify the exact pretrained-weight license separately from the Apache-2.0 code license;
- if the external embedding is not stable on Sorani, train a small internal ten-speaker metric head on owned data and use the external model only as a secondary metric.

Do not load a large speaker encoder on every production request if the profile can be precomputed offline.

---

# 4. Restoration model specification

## 4.1 Representation and protected-band invariant

Keep the UniverSR complex-STFT foundation.

The model receives:

- a 48-kHz representation of the Natural-safe waveform;
- a continuous cutoff/missing-band mask;
- F0 trajectory;
- voiced/unvoiced trajectory;
- speaker ID/profile conditioning;
- optional degradation metadata.

The model predicts **only the missing high-frequency region**.

The trusted observed region must not be predicted and then “encouraged” to remain similar. It must be copied or mathematically excluded from the model output.

Final construction:

```text
output spectrum =
    trusted observed spectrum
    +
    accepted generated missing spectrum
```

Use a narrow, complementary transition mask around the detected cutoff to avoid a discontinuity. The transition width must be configurable and calibrated. The protected region below the transition must remain invariant within a strict numerical tolerance.

## 4.2 Continuous cutoff handling

Replace the stock four-rate-only assumption with continuous cutoff support.

Training must sample real cutoff frequencies over the configured range, initially:

```text
2 kHz ≤ cutoff ≤ 22 kHz
target sample rate = 48 kHz
```

Support:

- standard telephony cutoffs;
- 8/12/16/24-kHz source-rate equivalents;
- irregular low-pass filters;
- codec-shaped roll-offs;
- already-48-kHz files containing only limited-band content.

The production detector must return:

```json
{
  "effective_cutoff_hz": 7800.0,
  "confidence": 0.97,
  "shape": "codec_lowpass",
  "restore_recommended": true,
  "evidence": {
    "spectral_rolloff": 0.0,
    "above_cutoff_snr_db": 0.0,
    "stationarity": 0.0
  }
}
```

If confidence is below the calibrated release threshold, bypass restoration unless the user provides an explicit cutoff.

## 4.3 Conditioning

Add only conditions that prove useful:

- continuous cutoff/frequency mask;
- F0 and voiced/unvoiced;
- learned speaker-ID embedding;
- optional precomputed profile vector.

Test, but do not assume, that a reference vector improves restoration. The low-frequency input already carries content, prosody, and much speaker identity.

Do **not** add WavLM, w2v-BERT, an LLM, a TTS text encoder, or a semantic decoder to the production generator in this phase. The existing Hawzhin Sorani CTC model belongs in the loss/guard layer, not as a waveform-generation prior.

## 4.4 Frequency-aware modeling

Implement a focused AnyBand-inspired upgrade rather than copying F5-TTS wholesale:

- preserve the existing efficient backbone where possible;
- add a shallow frequency-axis attention/aggregation block before temporal modeling;
- add a corresponding frequency-aware decoder for frequency-specific prediction;
- condition temporal blocks with cutoff, F0/voicing, and speaker profile through FiLM or AdaLN;
- keep the total trainable model within a size that is reproducibly trainable on the current workstation;
- document parameter count, peak VRAM, throughput, and inference steps.

The generic UniverSR architecture must remain available as an ablation, not as a second production runtime model.

## 4.5 Generation and determinism

- Production uses a deterministic seed derived from input hash, segment index, model hash, profile hash, and config hash.
- Start evaluation with guidance disabled or at the lowest useful value.
- Higher guidance is allowed only when the locked corpus proves it improves quality without increasing fidelity failures.
- Run research over several seeds to quantify variance.
- Production freezes one seed policy and one solver/step configuration.

## 4.6 Safe high-band strength

After generating the missing band once, construct candidate high-band strengths:

```text
1.00, 0.75, 0.50, 0.25, 0.00
```

Only the generated high-band residual changes. The protected band remains identical for every candidate.

Guard candidates from strongest to weakest. Accept the strongest passing candidate. If none passes, use the Natural-safe candidate.

Do not blend full reconstructed waveforms.

---

# 5. Training corpus and simulation

## 5.1 Real clean recordings

Record or curate clean, full-band, consented audio at:

```text
48 kHz
24-bit PCM or float32
mono per speaker
stable microphone placement
low room noise
no enhancement baked into the canonical master
```

Engineering targets:

- minimum serious pilot: approximately 30–60 clean minutes per speaker;
- preferred production target: about 2 clean hours per speaker;
- additional diverse Sorani speakers may be used for general Kurdish pretraining if rights permit.

These are planning targets, not magical minimums. Actual adequacy must be decided by held-out performance.

Ensure coverage of:

- Sorani phoneme and consonant contrasts;
- normal, soft, loud, emotional, and fast speech;
- questions, narration, dialogue, names, numbers, loanwords, and code-switches;
- studio and realistic microphone distances;
- multiple recording sessions;
- the actual content style used by the characters.

## 5.2 Leakage-proof splits

Split by recording session and utterance before chunking:

```text
train
development
calibration
locked acceptance
locked corruption
```

No phrase, take, near-duplicate, or synthetic derivative may cross splits.

The locked acceptance set must contain phrases never seen during training or profile construction.

Store immutable JSONL manifests and SHA-256 hashes for every file.

## 5.3 Paired degradation generation

Use real clean speech as the target and generate the degraded input on the fly.

Primary restoration degradations:

- continuous low-pass cutoffs;
- irregular filter slopes and transition widths;
- downsample/upsample chains;
- MP3, AAC, Opus, and telephony codec roll-offs;
- microphone and recorder bandwidth coloration.

Secondary robustness degradations:

- clipping;
- packet loss;
- modest residual noise;
- modest residual reverb;
- combinations representative of actual production audio.

Do not turn HawaRestore-KD into another universal denoiser. The existing Natural core removes noise/reverb. HawaRestore-KD’s primary job is recovering missing spectral content from the already-safe Natural candidate.

## 5.4 Curriculum

Train in this order:

1. high cutoffs and mild missing bands;
2. gradually wider missing bands;
3. balanced continuous cutoff distribution;
4. irregular filters and codecs;
5. combined realistic degradations.

Do not start adversarial refinement until the non-adversarial model passes content and protected-band tests.

## 5.5 Losses

The initial objective must include:

```text
L_total =
    L_missing_band_flow
  + λ_stft       · L_high_band_multires_complex_stft
  + λ_phase      · L_high_band_phase
  + λ_cross      · L_cross_band_envelope
  + λ_harmonic   · L_f0_harmonic_consistency
  + λ_speaker    · L_speaker_identity
  + λ_ctc        · L_sorani_ctc_consistency
  + λ_protected  · L_protected_band_invariance
```

Rules:

- `L_missing_band_flow` is computed only on missing bins.
- The protected-band term must be effectively zero by construction and tested as an invariant, not merely optimized.
- Freeze the Hawzhin CTC and speaker-evaluation networks.
- Keep the CTC weight small enough that it guards phonetics without turning restoration into ASR-conditioned resynthesis.
- Apply speaker identity loss to real reference targets and restored outputs, never synthetic-clone targets.
- Adversarial losses may inspect only spectral realism, cross-band coherence, and harmonics; they may not rewrite the observed band.

Every loss term requires an ablation. Remove any term that adds complexity without statistically meaningful held-out benefit.

---

# 6. Restoration Guard R

The existing Sorani guard is necessary but insufficient for generated high-frequency content.

A typical 16-kHz ASR backend cannot directly evaluate energy above 8 kHz. Do not claim that unchanged ASR text proves the generated high band is linguistically correct.

Implement Guard R with the following independent checks.

## 6.1 Structural integrity

Reject on:

- wrong duration;
- wrong sample count;
- channel mismatch;
- NaN/Inf;
- unexpected delay or drift;
- clipping;
- discontinuity at chunk or band boundaries;
- nondeterministic output under the frozen environment.

## 6.2 Protected-band invariance

Compare the Natural-safe candidate and restored candidate below the protected boundary.

Measure at least:

- low-passed waveform error;
- complex-STFT error;
- magnitude and phase deviation;
- transition-band leakage.

Calibrate a strict float32 tolerance using known-good round trips. No release is permitted if the restorer materially changes trusted speech below the protected region.

## 6.3 Sorani token and CTC consistency

Run the existing token, timing, confidence, and posterior checks between the Natural-safe candidate and restored output.

This remains a sanity barrier for resampling, crossover leakage, timing changes, and unexpected lower-band mutation.

Any anchor-token substitution/deletion remains a hard failure.

## 6.4 High-frequency consonant/event consistency

Add a dedicated high-band check because ASR alone cannot see the entire restored band.

Using the verified transcript/alignment and the Natural signal:

- derive speech and consonant-event windows;
- derive expected high-frequency activity from the observed 3–8-kHz envelope, onsets, voicing, and phonetic timing;
- reject new strong fricative/transient events outside allowed speech windows;
- reject missing or grossly shifted high-frequency events;
- reject high-band energy during non-speech unless a measured ambience policy permits it;
- reject segment-boundary sibilant bursts;
- validate the detector using deliberately injected false sibilants, deleted fricatives, shifted transients, and harmonic corruption.

Thresholds must be calibrated for zero false accepts on the locked corruption set, then the false-reject rate must be reported.

## 6.5 F0 and harmonic consistency

On voiced regions:

- generated harmonic energy must follow the measured F0 trajectory;
- reject octave errors, unrelated harmonic ladders, or voiced high-band energy during unvoiced spans;
- compare cross-band temporal envelopes;
- verify that high-frequency harmonics remain phase/time coherent enough to avoid metallic doubling.

## 6.6 Speaker identity

Compare restored output against the speaker’s real canonical profile and against the Natural-safe candidate.

Use multiple measurements:

- frozen speaker embedding similarity;
- internal ten-speaker classifier margin;
- F0/formant/timbre statistics where reliable;
- native blind identity ratings.

Do not accept a result merely because one embedding model scores highly.

## 6.7 Guard policy

The descent through the strength ladder happens during candidate selection, per
4.6: candidates are guarded strongest to weakest and the first one to clear every
layer is accepted. The verdict is emitted once, after selection has concluded. It
reports what was decided; it is never an instruction to retry.

Per segment:

```text
PASS        → accepted; strength ≥ 0.75 cleared every layer
WARN        → accepted at a reduced strength below 0.75; the ladder already descended
FAIL        → every active candidate rejected → Natural-safe candidate
ERROR       → restorer or guard raised, or input was degenerate → Natural-safe candidate + report flag
NO_RESTORE  → nothing was judged: bandwidth healthy, cutoff confidence too low,
              or no active candidate offered → Natural-safe candidate
```

PASS and WARN both ship generated content and differ only in how much of it
survived. FAIL, ERROR and NO_RESTORE all ship the Natural-safe candidate and
differ only in why: FAIL means the guard refused what the model produced, ERROR
means the attempt did not complete, NO_RESTORE means no attempt was made.

The zero-strength entry of the ladder is the Natural-safe candidate. It is never
submitted to the guard: scored against itself it clears every layer trivially and
returns first, so a segment whose every real candidate was rejected would be
recorded as having passed. It is what the guard falls back to, never something the
guard approves.

A FAIL verdict must carry the rejection reason and the failing layer's
measurements. "Reverted" without the evidence for it is not an audit trail.

Never output silence, truncated audio, or an unchecked generated segment.

---

# 7. Runtime integration

## 7.1 Repository additions

Add a focused subsystem:

```text
src/voiceclean/restoration/
├── __init__.py
├── bandwidth.py              # cutoff estimation + evidence
├── config.py                 # restoration pydantic schemas
├── profiles.py               # profile loading/hash/consent validation
├── f0.py                     # deterministic F0 + V/UV extraction
├── base.py                   # Restorer protocol
├── universr_upstream.py      # pinned upstream adapter
├── hawarestore_kd.py         # production model adapter
├── protected_band.py         # masks, merge, residual-strength candidates
├── guard.py                  # Guard R orchestration
├── highband_events.py        # consonant/event integrity
├── policy.py                 # accept/reduce/revert
└── report.py                 # restoration report schema

research/restoration/
├── train/
├── simulation/
├── ablations/
├── baselines/
├── evaluation/
└── configs/

profiles/
├── schema.json
└── <speaker_id>/

vendors/
└── universr/                 # pinned exact commit + license

proof/restoration-v2.1/
docs/adr/
```

Keep training/research dependencies isolated from the production environment where practical.

## 7.2 Interfaces

Add:

```text
voiceclean restore-doctor
voiceclean speaker-profile validate PROFILE
voiceclean restoration-benchmark MANIFEST
voiceclean process INPUT --mode restore --speaker-id ID --cutoff auto
voiceclean process INPUT --mode restore --speaker-id ID --cutoff-hz 7800
```

`restore-doctor` must verify:

- exact upstream commit;
- code and weight hashes;
- profile schema and hashes;
- licenses/provenance records;
- 48-kHz path;
- F0 extractor;
- restoration model;
- Guard R;
- deterministic smoke test;
- fail-closed behavior.

Training commands belong under the research package, not the normal end-user CLI.

## 7.3 Configuration

Add a versioned section without changing existing defaults:

```toml
[restoration]
enabled = false
mode = "explicit"
target_sample_rate = 48000
model = "hawarestore-kd"
model_sha256 = ""
solver = "midpoint"
steps = 4
guidance_scale = 0.0
strengths = [1.0, 0.75, 0.5, 0.25, 0.0]
cutoff_mode = "auto"
cutoff_confidence_min = 0.0
transition_hz = 0.0
process_non_speech = false
deterministic = true

[restoration.guard]
protected_band_threshold = 0.0
ctc_threshold = 0.0
highband_event_threshold = 0.0
harmonic_threshold = 0.0
speaker_threshold = 0.0
```

Zero placeholders are forbidden in a release config. `voiceclean restore-doctor` must fail until calibrated values and their calibration artifact hashes are present.

## 7.4 Reporting

Extend the immutable report:

```json
{
  "mode": "restore",
  "speaker_id": "character_01",
  "profile_hash": "...",
  "natural_output_hash": "...",
  "bandwidth": {
    "cutoff_hz": 7800.0,
    "confidence": 0.97,
    "source": "detector"
  },
  "restorer": {
    "name": "hawarestore-kd",
    "commit": "...",
    "weights_sha256": "...",
    "seed_policy": "...",
    "solver": "midpoint",
    "steps": 4,
    "guidance_scale": 0.0
  },
  "segments": {
    "restored": 0,
    "reduced": 0,
    "reverted": 0,
    "bypassed": 0,
    "errors": 0
  },
  "guard_r": {
    "protected_band": {},
    "ctc": {},
    "highband_events": {},
    "harmonic": {},
    "speaker": {}
  },
  "review_timecodes": []
}
```

Record every candidate strength tried and why it passed or failed.

---

# 8. Implementation phases and Definitions of Done

Do not begin a later phase until the prior phase’s evidence is committed.

## Phase 0 — Audit and amendment

Tasks:

- inspect the current repository and map this amendment onto actual modules;
- write an ADR explaining why Restore mode is separate from Natural mode;
- identify every Master Blueprint clause affected;
- pin exact upstream sources;
- complete a license/provenance matrix;
- create the proof directory and evidence index.

Definition of Done:

- no production code changed yet;
- ADR reviewed;
- current Natural tests pass unchanged;
- exact source commits and licenses recorded;
- unresolved weight rights are blockers, not assumptions.

## Phase 1 — Reproduce generic UniverSR

Tasks:

- vendor or isolate the pinned upstream;
- reproduce official inference;
- run official/generic checkpoints;
- verify low-band-copy behavior;
- benchmark memory, runtime, determinism, and long-form chunking.

Definition of Done:

- exact command logs;
- checkpoint hashes;
- reproducible output hashes;
- output at 48 kHz;
- protected-band numerical test passes;
- known limitations documented.

## Phase 2 — Product-safe restoration spine

Tasks:

- implement bandwidth detector;
- implement continuous cutoff masks;
- implement protected-band merge;
- implement strength candidates;
- add Restore CLI and reporting;
- use a stub/generic restorer first;
- fail closed to Natural.

Definition of Done:

- no healthy clip is restored in unit/golden tests;
- strength zero equals Natural output;
- exceptions/timeouts/OOM/invalid profiles revert to Natural;
- duration and synchronization tests pass;
- Natural mode golden outputs remain unchanged.

## Phase 3 — Kurdish generic HawaRestore-KD

Tasks:

- build the real clean corpus;
- implement paired degradation;
- fine-tune continuous-cutoff missing-band model on Kurdish;
- implement F0/voicing;
- add frequency-aware modules only after the plain fine-tune baseline is reproducible;
- implement Guard R except speaker-specific checks.

Definition of Done:

- no train/accept leakage;
- all manifests hashed;
- training reproducible from config/checkpoint;
- generic Kurdish model beats generic UniverSR on locked Kurdish spectral metrics and native listening without a fidelity regression;
- protected-band and corruption gates pass.

## Phase 4 — Speaker personalization ablation

Evaluate, in this order:

1. generic Kurdish model;
2. learned ten-way speaker-ID embedding;
3. speaker ID plus precomputed real-reference prototype;
4. optional tiny per-speaker adapter.

Definition of Done:

- same backbone/training data/evaluation protocol across variants;
- held-out phrases and sessions;
- confidence intervals reported;
- select the simplest variant that materially improves identity and naturalness without content failures;
- if personalization does not help, ship the generic Kurdish model and explicitly report the negative result.

## Phase 5 — AnyBand-inspired refinement

Tasks:

- continuous cutoff curriculum;
- frequency-axis encoder/decoder ablation;
- cross-band and harmonic losses;
- optional endpoint adversarial refinement only after safe baseline;
- solver/guidance/step ablation.

Definition of Done:

- every added component has measured benefit;
- no release gate worsens;
- remove components that do not earn their complexity;
- freeze one architecture and one inference configuration.

## Phase 6 — Full acceptance and release candidate

Tasks:

- run all baselines;
- produce blinded sample packs;
- conduct native Sorani listening;
- run chaos/security/provenance checks;
- freeze release config, profile hashes, model card, SBOM, and reproducible container.

Definition of Done:

- every hard gate below passes;
- complete evidence package exists;
- no unresolved license or provenance issue;
- no hidden TODO, mocked production component, or untested fallback.

---

# 9. Hard release gates

The feature is **NOT COMPLETE** unless every applicable gate passes.

## 9.1 Content fidelity

- Zero confirmed system-caused Sorani word substitutions, deletions, or insertions on the locked acceptance corpus.
- Zero accepted deliberately corrupted examples in the locked corruption corpus.
- No anchor-token degradation relative to Natural.
- No meaningful CTC-posterior divergence beyond calibrated limits.
- Every uncertain segment reverts.

## 9.2 Protected-band fidelity

- The region below the protected boundary passes the calibrated waveform, complex-STFT, magnitude, and phase invariance thresholds.
- No crossover comb filtering, holes, or time smear.
- Strength `0.0` is numerically equivalent to the Natural candidate under the documented format conversion.
- No restored candidate can bypass this check.

## 9.3 Speaker fidelity

- Personalized HawaRestore-KD is not worse than Natural on the locked speaker-similarity suite.
- Native listeners identify the intended character at least as reliably as with Natural.
- Personalization must beat generic Kurdish HawaRestore-KD on held-out material for at least 8 of 10 speakers and show a positive aggregate result with a reported confidence interval.
- Otherwise, remove speaker personalization from the production release.

## 9.4 Restoration quality

On genuinely bandwidth-limited Sorani clips:

- HawaRestore-KD must beat stock UniverSR on the locked aggregate;
- it must beat or tie the best legally usable open baseline;
- native listeners must prefer it to Natural-only at a statistically supported aggregate level;
- compare against Adobe Podcast and RX pre-renders, but do not trade word fidelity for preference scores;
- report every degradation class separately—do not hide failures inside one average.

## 9.5 Healthy-audio safety

- At least 99% of locked healthy full-band clips bypass restoration.
- Zero confirmed false restoration that audibly worsens a healthy acceptance clip.
- Manual override remains possible but is prominently reported.

## 9.6 Robustness

Pass:

- unit tests;
- strict typing/linting;
- property tests;
- integration tests with real models;
- deterministic reruns;
- OOM/timeout/crash tests;
- corrupted checkpoint/profile/config tests;
- interrupted-job recovery;
- malformed media tests;
- long-form podcast tests;
- multichannel policy tests;
- mutation tests for guard logic.

## 9.7 Legal and reproducibility

- Exact source commits recorded.
- Exact model revisions and SHA-256 hashes recorded.
- Code and weight licenses recorded separately.
- Training-data rights and speaker consent documented.
- Container/lockfile/SBOM generated.
- One command reproduces the acceptance evaluation from frozen artifacts.
- No external model or dataset with unresolved production rights is included.

## 9.8 Claim discipline

Do not write “number one,” “best,” “SOTA,” “word-safe,” or “cannot change words” in product copy merely because implementation is complete.

Such a claim is permitted only if:

- the locked Sorani benchmark is published internally with the comparison protocol;
- all hard gates pass;
- sample and metric evidence is retained;
- limitations are stated;
- the claim is scoped to the tested Sorani domain and date.

---

# 10. Required proof package

Create:

```text
proof/restoration-v2.1/
├── INDEX.md
├── 00-current-repo-audit/
├── 01-source-and-license/
├── 02-data-and-consent/
├── 03-reproducible-environment/
├── 04-tests/
├── 05-training/
├── 06-ablations/
├── 07-benchmarks/
├── 08-blind-listening/
├── 09-failure-analysis/
├── 10-security-and-sbom/
└── 11-release-candidate/
```

The evidence must include:

## Source and provenance

- Git commit of the HawaVoClean implementation.
- Clean `git status`.
- Full diff/stat.
- Upstream commit hashes.
- Download commands and SHA-256 hashes.
- Code-license and weight-license evidence.
- Dataset provenance and consent manifests.
- SBOM and vulnerability scan.

## Reproduction

- OS/GPU/driver/CUDA/Python/PyTorch versions.
- Container digest.
- Lockfile hash.
- Exact training and inference commands.
- Random seeds.
- Peak GPU memory.
- Runtime/RTF.
- Checkpoint selection rule.
- Training curves and failure logs.

## Test proof

Store raw command output, not summaries:

```text
ruff
mypy --strict
pytest unit
pytest property
pytest integration
pytest chaos
pytest mutation
voiceclean restore-doctor
voiceclean restoration-benchmark ...
```

Screenshots alone are not proof.

## Benchmark proof

Provide machine-readable CSV/JSON plus a concise Markdown report containing:

- per-speaker results;
- per-degradation results;
- all baseline results;
- confidence intervals;
- rejected/reverted segment counts;
- fidelity failures;
- protected-band errors;
- speaker metrics;
- spectral metrics;
- runtime and memory;
- negative results and known limitations.

## Audio proof

Provide a blinded, lossless sample pack:

```text
original clean reference
degraded input
Natural output
stock UniverSR
generic Kurdish HawaRestore-KD
personalized HawaRestore-KD
Adobe export
RX export
other legal baseline
```

Use held-out phrases. Do not expose filenames that reveal the method during listening.

## Failure proof

Include the worst cases, not only showcase samples:

- every word/phoneme failure;
- every speaker-identity failure;
- every high-band artifact;
- every detector false positive/negative;
- every reverted segment;
- root cause and disposition.

---

# 11. Forbidden shortcuts

You must not:

- claim the full AnyEnhance paper system is the same as the lightweight public v1 repo;
- claim AnyBand was implemented because its concepts were borrowed;
- use a TTS clone as clean ground truth;
- train/test on the same phrases or sessions;
- weaken a threshold to make a result pass;
- remove a failed speaker from the aggregate;
- hide reverted segments;
- use ASR text equality as the only high-band guard;
- select the best seed per test clip;
- tune on the locked acceptance set;
- ship external weights with unclear rights;
- silently resample or alter timing;
- add a runtime model router;
- keep a personalization component that fails ablation;
- report only DNSMOS/NISQA or other model-based MOS;
- mark the task complete with mocked data, stub profiles, or synthetic-only evidence.

---

# 12. Completion response required from the coding agent

When all work is finished, respond using exactly this evidence-oriented structure:

```text
STATUS: PASS | NOT COMPLETE

IMPLEMENTATION
- Repository:
- Branch:
- Commit:
- Clean working tree:
- Changed files:
- ADR:

FROZEN PRODUCTION DESIGN
- Natural core:
- Restoration core:
- Upstream base + commit:
- HawaRestore-KD checkpoint SHA-256:
- Speaker-profile schema/version:
- Output sample rate:
- Solver/steps/guidance:
- Deterministic seed policy:

PROVENANCE
- Code licenses:
- Weight licenses:
- Training-data provenance:
- Speaker consent:
- SBOM:
- Unresolved legal issues:

TEST EVIDENCE
- Ruff:
- Mypy:
- Unit:
- Property:
- Integration:
- Chaos:
- Mutation:
- Determinism:
- Long-form:
- Restore doctor:
- Raw log locations:

ACCEPTANCE GATES
- Sorani word/phoneme failures:
- Corruption false accepts:
- Protected-band failures:
- Healthy-audio false restores:
- Speaker-personalization result:
- Generic-versus-personalized result:
- Blind listening result:
- Baseline comparisons:
- Reverted segments:
- Gates passed/failed:

PERFORMANCE
- Parameter count:
- Peak VRAM:
- RTF:
- Ten-speaker profile load time:
- Long-form runtime:

PROOF PACKAGE
- Index path:
- Benchmark CSV/JSON:
- Blinded WAV pack:
- Failure pack:
- Reproduction command:
- Release artifact hashes:

KNOWN LIMITS
- ...

FINAL VERDICT
- Production-ready: YES | NO
- Authorized to claim “best tested Sorani restoration system”: YES | NO
- Exact reason:
```

If any hard gate is unpassed, use `STATUS: NOT COMPLETE`. Do not soften the wording.

---

# 13. Source anchors to verify and pin

Before implementation, independently re-open and verify these primary sources:

```text
UniverSR
- paper: arXiv 2510.00771
- repository: woongzip1/UniverSR
- initial pinned commit: 26dc21c44e11f9f19e823f02b0d4641dd5ea5af2
- verify code license and each exact model-weight license

AnyBand
- paper: arXiv 2608.00572
- design reference only unless official code/weights become available
- do not claim implementation parity

AnyEnhance
- paper: arXiv 2501.15417 / IEEE TASLP publication
- lightweight repository: viewfinder-annn/AnyEnhance-v1
- historical fuller Amphion merge commit:
  c0277229f83ea685db15611fd81a5396c571e264
- research comparator only after exact weight-license verification

3D-Speaker / ERes2NetV2
- repository: modelscope/3D-Speaker
- initial pinned commit: 065629c313eaf1a01c65c640c46d77e61e9607b4
- code is Apache-2.0; verify exact pretrained-weight terms separately

MossFormer2_SR_48K
- repository: modelscope/ClearerVoice-Studio
- baseline only; pin exact tested revision and verify weights
```

---

# 14. Final engineering rule

The purpose of personalization is not to make a more convincing synthetic voice. It is to use known real-speaker acoustics to choose a more faithful continuation of frequencies that are genuinely absent.

The production hierarchy is:

```text
spoken-content fidelity
> trusted-band preservation
> correct speaker identity
> naturalness
> noise suppression
> brightness
```

Whenever those goals conflict, preserve the higher-ranked property and revert.

Build the simplest system that passes all gates. A smaller generic Kurdish restorer that is demonstrably faithful is superior to a more elaborate personalized generator that merely sounds impressive.
