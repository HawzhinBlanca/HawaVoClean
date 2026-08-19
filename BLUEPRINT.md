# HAWZHIN VOICECLEAN — Master Implementation Blueprint v2.0

**Audience:** autonomous AI coding agent with repository, shell, GPU, and test access  
**Status:** implementation-authoritative; supersedes `hawzhin-voiceclean-v1-blueprint.md`  
**Date:** 2026-08-18  
**Primary platform:** Linux workstation with NVIDIA GPUs; CPU is the correctness fallback  
**Product principle:** highest achievable Sorani dialogue quality with fail-closed linguistic preservation

---

## 0. Read this before writing code

You are implementing a production audio system, not a demo and not a research notebook.

Follow this specification literally unless a requirement is technically impossible or contradicted by verified upstream behavior. When that happens:

1. Stop only the affected implementation step—not the whole project.
2. Reproduce and document the contradiction with commands, logs, and source links.
3. Add an Architecture Decision Record under `docs/adr/`.
4. Choose the smallest safer alternative that preserves the invariants below.
5. Never weaken a safety gate, delete a failing test, or silently change scope to obtain a green build.

A feature is **not complete** until its Definition of Done has passed with actual command output. Code that merely imports, compiles, or looks plausible is not complete.

### Agent operating contract

- Inspect every upstream repository and model card before integrating it. Do not guess APIs, sample rates, licenses, checkpoint names, or tensor shapes.
- Pin every dependency, repository commit, container image digest, and model artifact hash.
- Preserve the source audio unchanged and permanently recoverable.
- Default to the original signal whenever evidence is insufficient.
- Do not use an LLM to make audio decisions.
- Do not place multiple enhancement models in the production path.
- Do not claim “no hallucination” or “never changes words.” No available model can mathematically guarantee that. The system must instead detect risk, revert, report, and quantify residual uncertainty.
- Do not add GUI, streaming, cloud orchestration, queues, databases, accounts, telemetry, or unrelated features.
- Keep `STATUS.md`, `RISKS.md`, and `docs/adr/` current throughout implementation.

---

## 1. Mission, truth boundary, and non-negotiable invariants

Build an offline, single-machine tool that transforms long-form Kurdish Sorani dialogue and podcast recordings into clean, clear, natural, professionally mastered audio while minimizing any change to linguistic content, speaker identity, emotion, timing, and accent.

The system processes audio through **exactly one frozen neural enhancement core**, verifies the result with the **Hawzhin Sorani Fidelity Guard**, safely accepts or reverts each speech unit, applies conservative deterministic finishing, verifies the finished result again, and emits:

- a mastered WAV;
- an immutable JSON report;
- a human-readable review summary;
- timecodes for every reverted, uncertain, or failed unit.

### Invariants, ordered by priority

1. **Source preservation** — the input file is never modified, overwritten, or deleted.
2. **Linguistic preservation** — any detected substitution, deletion, severe confidence loss, phonetic drift, timing warp, or unverifiable speech unit is reverted.
3. **Fail closed** — errors produce original-audio passthrough for the affected unit, never silence, synthetic speech, partial data, or job-wide corruption.
4. **One runtime core** — the released product contains one selected enhancer. Candidate comparison exists only in the research/benchmark harness.
5. **No generative repair by default** — packet-loss concealment, bandwidth extension, vocoder regeneration, and speech inpainting are disabled unless the selected core intrinsically uses them and has passed all Sorani release gates. V1 has no user-facing Rescue mode.
6. **Continuity** — output duration, sample count, channel count, and timeline match the input exactly unless the user explicitly requests another delivery format.
7. **Transparent decisions** — every unit records model, hashes, alignment, guard scores, finish behavior, decision, and reason.
8. **Reference-platform reproducibility** — the same input, immutable environment, GPU model, exact driver/runtime stack, configuration, and weights must reproduce identical policy decisions and numerically stable audio. Cross-platform bit identity is not promised.
9. **Atomic publication** — a final output is exposed only after complete validation and checksum generation.
10. **No silent degradation** — no NaN, Inf, clipping, channel swap, sample-rate mismatch, drift, missing interval, or unreported fallback may reach the published output.

### Explicit anti-goals

Do not build:

- a model router in production;
- enhancer chaining;
- text-conditioned speech generation;
- voice cloning or speaker regeneration;
- automatic translation or transcription editing;
- real-time or streaming mode;
- cloud services or remote uploads;
- GUI or browser UI;
- VST/AU/AAX plug-ins;
- automatic processing of ambiguous stereo masters;
- automatic bandwidth extension on healthy full-band speech;
- a custom training platform in V1.

---

## 2. Corrections that supersede the uploaded V1 draft

The uploaded V1 draft provides the correct foundational ideas—fail-safe processing, a Sorani CTC guard, deterministic finishing, structured reporting, and phased implementation—but the following are now authoritative:

1. **Do not lock GAP-URGENet before testing Sorani.** Its public inference stack is now substantially available, but its learned reconstruction path remains a linguistic-risk candidate. Model selection must be empirical.
2. **Do not keep a runtime fallback enhancer.** If the frozen core fails, the affected unit reverts to original. A second runtime model creates timbre inconsistency, dependency sprawl, and new failure modes.
3. **Guard twice.** The enhancer output is Guard A; the locally finished output is Guard B. Finishing can also weaken consonants or alter intelligibility.
4. **Do not promise bit-identical CUDA output across platforms.** PyTorch itself does not guarantee reproducibility across releases, devices, or platforms. Define and test a reference environment.
5. **Do not switch sources mid-word.** Segment decisions are made at utterance-group level, and source changes occur only at verified low-energy/non-speech boundaries.
6. **Do not hand-wave model sample-rate behavior.** Every resampling step is explicit, high quality, delay-compensated, and reported.
7. **Do not treat ASR transcript equality as sufficient.** Use token anchors, frame-level CTC posterior comparison, timing checks, and signal integrity checks.
8. **Do not calibrate and evaluate on the same corpus.** Calibration, development, and locked acceptance sets are separate and hash-locked.
9. **Do not write final output directly.** Use a resumable job journal, segment cache, validation pass, and atomic rename.
10. **Do not treat repository license as checkpoint/data clearance.** Code, weights, and training data provenance are separate release gates.

---

## 3. Frozen product behavior

### V1 user experience

```bash
voiceclean doctor
voiceclean process INPUT.wav --output OUTPUT.wav --profile production
voiceclean verify OUTPUT.wav --report OUTPUT.voiceclean.json
```

Development-only commands:

```bash
voiceclean calibrate data/calibration/manifest.jsonl
voiceclean benchmark data/development/manifest.jsonl
voiceclean acceptance data/acceptance/manifest.jsonl
voiceclean audit-models
```

### Inputs

- WAV, FLAC, AIFF, MP3, AAC/M4A, Opus, and common video containers through a pinned FFmpeg build.
- Mono dialogue.
- Dual-mono or split-speaker stereo, when explicitly identified or safely detected.
- Common speech-production sample rates: 8, 16, 22.05, 24, 32, 44.1, and 48 kHz; model-specific conversion is explicit. Inputs above 48 kHz are rejected in V1 rather than silently discarding ultrasonic content.
- PCM integer or floating-point audio.

### Outputs

Default:

- 24-bit PCM WAV;
- same sample rate, channel count, sample count, and duration as input;
- `OUTPUT.voiceclean.json`;
- `OUTPUT.voiceclean.txt`;
- SHA-256 checksums.

Optional delivery format:

- 32-bit float WAV, selected through config.

### Exit semantics

- `0`: completed and validated, including jobs containing safe reverts.
- `2`: preflight/config/model/provenance failure; no processing began.
- `3`: final validation or atomic publication failed; no final output published.
- `4`: user input invalid or unsupported.
- Segment-level model, guard, or DSP failures never terminate the job; they revert and are reported.

---

## 4. Lean system architecture

```text
IMMUTABLE INPUT
      │
      ▼
Preflight + media probe + hashes + disk/resource checks
      │
      ▼
Safe decode → explicit channel classification → canonical timeline
      │
      ▼
Speech activity detection → utterance groups with context
      │
      ▼
ONE frozen enhancer in an isolated worker process
      │
      ▼
Length/timing/alignment validation
      │
      ▼
GUARD A: original vs enhanced
      │
      ├── PASS ───────► enhanced candidate
      └── FAIL/ERROR/UNVERIFIED ─► original
      │
      ▼
Sample-accurate timeline assembly at safe boundaries
      │
      ▼
Detection-gated local finishing
      │
      ▼
GUARD B: accepted timeline vs locally finished timeline
      │
      ├── PASS ───────► locally finished unit
      └── FAIL/ERROR/UNVERIFIED ─► pre-finish accepted unit
      │
      ▼
Global static loudness gain + bounded true-peak limiting
      │
      ▼
Final structural/signal validation
      │
      ▼
Atomic WAV + JSON/TXT report publication
```

### Why this is lean

Production contains:

- one enhancer;
- one ASR backend;
- one guard implementation;
- one deterministic finishing path;
- one parent orchestrator and two local worker processes at most;
- file-based state, not a database;
- one CLI, not an application platform.

Candidate complexity is isolated to a one-time benchmark before the production core is frozen.

---

## 5. Core model selection: benchmark first, freeze once

### 5.1 Candidate policy

The agent must not assume that the globally best challenge model is the best Sorani model.

Evaluate these candidate classes using isolated runners:

| Candidate | Role | Runtime eligibility |
|---|---|---|
| Official URGENT 2026 discriminative BSRNN checkpoint | Universal predictive baseline | Eligible after provenance review |
| `MossFormer2_SE_48K` | Full-band phase-sensitive predictive candidate | Eligible after weight/license audit |
| GAP-URGENet official checkpoints | Hybrid quality-ceiling candidate | Eligible only if Sorani gates pass |
| NVIDIA RE-USE | Research quality/fidelity reference | Benchmark only unless commercial permission is obtained |
| Adobe Podcast and RX Dialogue Isolate exports | Commercial external baselines | Pre-rendered comparison only |

FastEnhancer 48 kHz may be added only if the three open candidates all fail quality or operational gates. Do not add candidates for curiosity.

### 5.2 Candidate runner contract

Candidate repositories must not share one Python environment. Each candidate gets a pinned OCI image or isolated `uv` environment and exposes the same file contract:

```text
runner INPUT.wav OUTPUT.wav RESULT.json
```

`RESULT.json` must contain:

- candidate name and version;
- repository URL and commit;
- weight filenames and SHA-256;
- code, weight, and known data licenses;
- input/output sample rate and sample count;
- runtime and peak VRAM;
- warnings/errors;
- whether output is expected to be phase coherent.

The benchmark harness treats each runner as an opaque executable. This prevents vendor dependency conflicts from contaminating production architecture.

### 5.3 Hard selection procedure

1. **Disqualify** any candidate with a confirmed word/phoneme substitution on the locked acceptance set.
2. **Disqualify** any candidate that fails output integrity, provenance, or repeatability requirements.
3. **Disqualify** candidates whose Guard A false-accept rate is nonzero on the locked corruption suite.
4. Among survivors, select the model with the highest native-Sorani blind preference for naturalness and clarity.
5. Tie-break by lower guard-revert rate, then lower runtime/VRAM, then simpler licensing.
6. Freeze exactly one winner in `production-core.lock.toml`.
7. Remove unused model dependencies from the production image.

No weighted “overall score” may override a linguistic-fidelity disqualification.

### 5.4 Production core lock

`production-core.lock.toml` is mandatory and contains:

```toml
schema_version = 1
core_id = "..."
runner_version = "..."
repo_url = "..."
commit = "40-character git SHA"
code_license = "..."
weight_license = "..."
weight_sha256 = { "file" = "sha256" }
expected_sample_rates = [48000]
phase_coherent = true
calibration_id = "sha256"
acceptance_run_id = "sha256"
```

Production startup fails if any field is absent, any hash differs, or the core does not match the calibration artifact.

### 5.5 Runtime failure rule

There is no second enhancer in production.

```text
core succeeds + guard passes → use candidate
anything else                → use original
```

---

## 6. Repository structure

```text
hawzhin-voiceclean/
├── README.md
├── CHANGELOG.md
├── STATUS.md
├── RISKS.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── production.toml
│   ├── development.toml
│   └── schemas/
│       ├── config.schema.json
│       ├── report.schema.json
│       └── corpus.schema.json
├── models/
│   ├── README.md
│   ├── model-registry.toml
│   └── production-core.lock.toml
├── src/voiceclean/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── errors.py
│   ├── logging.py
│   ├── hashing.py
│   ├── job.py
│   ├── journal.py
│   ├── audio/
│   │   ├── probe.py
│   │   ├── decode.py
│   │   ├── encode.py
│   │   ├── resample.py
│   │   ├── channels.py
│   │   └── types.py
│   ├── segmentation/
│   │   ├── vad.py
│   │   ├── utterances.py
│   │   └── types.py
│   ├── enhancement/
│   │   ├── protocol.py
│   │   ├── production.py
│   │   ├── worker.py
│   │   └── validate.py
│   ├── alignment/
│   │   ├── delay.py
│   │   ├── drift.py
│   │   └── coherence.py
│   ├── guard/
│   │   ├── protocol.py
│   │   ├── sorani_normalize.py
│   │   ├── hawzhin_ctc.py
│   │   ├── token_anchor.py
│   │   ├── posterior.py
│   │   ├── timing.py
│   │   ├── signal.py
│   │   ├── calibration.py
│   │   └── verdict.py
│   ├── policy/
│   │   ├── strength.py
│   │   ├── decision.py
│   │   └── continuity.py
│   ├── finishing/
│   │   ├── detect.py
│   │   ├── repair.py
│   │   ├── eq.py
│   │   ├── deess.py
│   │   ├── dynamics.py
│   │   ├── loudness.py
│   │   ├── limiter.py
│   │   └── safe_finish.py
│   ├── assembly/
│   │   ├── overlap.py
│   │   ├── stitch.py
│   │   └── validate.py
│   └── report/
│       ├── schema.py
│       ├── writer.py
│       └── summary.py
├── research/
│   ├── candidates/
│   │   ├── urgent_bsrnn/
│   │   ├── mossformer2/
│   │   ├── gap_urgenet/
│   │   └── reuse/
│   ├── benchmark.py
│   └── ingest_external.py
├── eval/
│   ├── corpus.py
│   ├── corruption.py
│   ├── metrics.py
│   ├── blind_abx.py
│   ├── calibrate.py
│   ├── acceptance.py
│   └── statistics.py
├── tests/
│   ├── unit/
│   ├── property/
│   ├── mutation/
│   ├── integration/
│   ├── chaos/
│   ├── golden/
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── fidelity-guard.md
│   ├── calibration.md
│   ├── operations.md
│   ├── model-provenance.md
│   └── adr/
└── scripts/
    ├── fetch_models.py
    ├── build_reference_image.sh
    ├── generate_sbom.sh
    └── run_release_checks.sh
```

---

## 7. Environment, dependencies, and reproducibility

### 7.1 Version policy

- No production dependency ranges.
- Exact Python, PyTorch, CUDA runtime, FFmpeg, model commits, and package versions are recorded after compatibility testing.
- Commit `uv.lock` and run production commands with `uv run --locked`.
- Pin the Docker base image and the `uv` binary by immutable SHA-256 digest.
- Generate a CycloneDX SBOM for every release.
- Record the host NVIDIA driver, GPU model, CUDA runtime, cuDNN, OS kernel, and CPU in every report.

### 7.2 Reference environment

The release target is one immutable Linux/NVIDIA container. CPU is a correctness fallback and need not match GPU throughput.

MPS/macOS is not a V1 release requirement. It may be added only after the Linux reference build passes all gates.

### 7.3 Determinism controls

At worker startup:

```text
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

In Python:

- seed Python, NumPy, and PyTorch;
- `torch.use_deterministic_algorithms(True)`;
- `torch.backends.cudnn.benchmark = False`;
- `torch.backends.cudnn.deterministic = True`;
- disable TF32;
- no AMP/autocast;
- no `torch.compile` in V1;
- use `torch.inference_mode()`;
- fill/check uninitialized tensors where supported;
- use fixed thread counts for DSP.

If an upstream kernel cannot run deterministically:

1. prove it with a minimal reproduction;
2. attempt a deterministic implementation or CPU fallback;
3. document it in an ADR;
4. require identical policy decisions and bounded numeric difference under the reference platform;
5. never silently switch to nondeterministic execution.

### 7.4 Precision policy

- Neural inference: float32.
- Audio timeline storage: float32.
- Filter design, loudness integration, statistics, and checksum-normalized metrics: float64 accumulation.
- No quantized model in V1.
- No automatic mixed precision.

---

## 8. Configuration and immutable artifacts

Use Pydantic v2 models and one user-facing `production.toml`.

Configuration sections:

```text
runtime
input
segmentation
enhancement
alignment
guard
policy
finishing
loudness
reporting
```

### Rules

- Generate and commit JSON Schemas.
- Unknown fields are errors.
- All thresholds are typed, range-checked, and unit-labelled.
- Production guard thresholds are loaded only from a signed/hash-locked calibration artifact, not hand-entered defaults.
- Only device, cache directory, and log verbosity may be overridden by environment variables.
- Hash the canonicalized effective config and store it in every report.
- Production refuses to run with `development = true`, missing calibration, or unapproved model locks.

Required immutable artifacts:

```text
production-core.lock.toml
guard-calibration.json
acceptance-result.json
environment.lock.json
model-registry.toml
SBOM.cdx.json
```

---

## 9. Media probe, decoding, and channel safety

### 9.1 Probe

Use `ffprobe` JSON without shell interpolation. Record:

- codec/container;
- sample rate;
- sample format/bit depth;
- channel count/layout;
- duration and sample count where available;
- start time and timestamp anomalies;
- stream index;
- metadata needed to reproduce decode.

Reject files with inconsistent duration/timestamps unless a controlled repair path is implemented and reported.

### 9.2 Decode

- Invoke FFmpeg with an argument array, never `shell=True`.
- Decode to little-endian float32 PCM in a private job workspace.
- Preserve the original sample rate and channel count in the canonical timeline.
- Validate decoded byte length exactly.
- Reject NaN/Inf and impossible amplitude values.
- Enforce subprocess timeouts and resource limits.

### 9.3 Channel classification

`channel_mode = auto` may produce only these results:

1. `mono`;
2. `dual_mono_same` — correlation and level evidence show duplicated mono;
3. `split_speakers` — explicitly declared or strongly evidenced isolated microphones;
4. `ambiguous_stereo`.

Policy:

- `mono`: process normally.
- `dual_mono_same`: process one channel and duplicate exactly.
- `split_speakers`: process independently but preserve sample alignment; use linked final loudness control.
- `ambiguous_stereo`: preflight error requiring an explicit user decision. Never silently downmix or independently process a stereo master.

### 9.4 Resampling

- Resample only at model boundaries.
- Use a pinned high-quality band-limited resampler such as libsoxr VHQ.
- Record source rate, model rate, filter mode, expected delay, and output rate.
- Compensate resampler delay before alignment.
- Return to the canonical input rate before guard comparison and assembly.
- Resampling must preserve the exact canonical sample count after final trim/pad validation.

---

## 10. Segmentation: utterance-first, not arbitrary chunks

Use a pinned Silero VAD or a demonstrably stronger offline VAD after evaluation.

### Speech-unit construction

1. Detect speech intervals at the canonical sample rate.
2. Merge intervals separated by less than 250 ms.
3. Group complete utterances toward 12–20 seconds.
4. Prefer boundaries inside at least 200 ms of non-speech.
5. Allow groups up to 30 seconds to avoid cutting continuous speech.
6. If a forced cut is unavoidable, place it at the lowest-energy zero crossing and add at least 1 second of context on both sides.
7. Context is fed to the model but trimmed before policy/assembly.
8. A source decision applies to the whole utterance group, not an arbitrary center window.

Each `SpeechUnit` contains:

```text
unit_id
start_sample
end_sample
context_start_sample
context_end_sample
speech_mask
forced_boundary
channel_id
input_sha256
```

### Non-speech

- Pass non-speech through unchanged by the neural core.
- Never create digital black silence unless it existed in the source.
- Preserve room tone and breaths unless a local repair detector explicitly acts.
- The final limiter and static gain may affect non-speech uniformly.

---

## 11. Enhancement core contract and worker isolation

### 11.1 Protocol

```python
class Enhancer(Protocol):
    metadata: EnhancerMetadata

    def warmup(self) -> None: ...

    def enhance(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int,
    ) -> EnhancementResult: ...
```

`EnhancementResult` contains:

```text
waveform
sample_rate
model_runtime_ms
input_samples
output_samples
peak_vram_bytes
warnings
```

### 11.2 Worker design

Use one long-lived enhancer subprocess controlled by the parent process.

- Communicate through local IPC with explicit message schemas.
- Heartbeat while processing.
- Parent enforces a hard deadline.
- If the worker hangs, crashes, OOMs, or returns invalid data, kill and restart it; the unit becomes error-passthrough.
- Never attempt to kill a hung CUDA call from a Python thread.
- Cache successful unit results by input/config/model hash.
- Resume completed units after process or machine interruption.

GPU allocation on the reference workstation:

- GPU 0: enhancement worker.
- GPU 1: Sorani ASR/guard worker.
- Single-GPU systems run sequentially.

### 11.3 Model feature policy

Disabled by default:

- packet-loss concealment;
- bandwidth extension;
- explicit inpainting;
- speaker conditioning;
- stochastic sampling;
- random chunk offsets.

A feature may be enabled only if it is intrinsic to the selected frozen core, deterministic, explicitly represented in the model lock, and passed the Sorani acceptance gates.

### 11.4 Immediate output validation

Before any guard runs, require:

- finite float32 samples;
- expected channel count;
- correct sample rate;
- output length within the model’s documented deterministic padding tolerance;
- no newly introduced hard clipping;
- no all-zero or near-zero speech result;
- no gross RMS/peak anomaly;
- no unexplained timestamp drift.

Any violation becomes original passthrough.

---

## 12. Alignment and phase-coherence analysis

### 12.1 Alignment

For original and enhanced unit:

1. estimate coarse delay with GCC-PHAT;
2. refine integer delay over multiple voiced windows;
3. estimate fractional delay from phase slope/parabolic peak interpolation;
4. apply only a bounded delay correction;
5. compare local landmark drift across the unit.

Do not time-stretch, warp, or resynthesize enhanced audio to force alignment.

### 12.2 Rejection conditions

Thresholds are calibrated, but implementation must support:

- excessive global delay;
- length mismatch beyond deterministic model padding;
- local drift inconsistent with a constant delay;
- low waveform/STFT coherence for a core expected to be phase coherent;
- inconsistent delay estimates across windows.

### 12.3 Phase-coherence declaration

Do not trust a README label alone. Measure it during candidate evaluation.

- Phase-coherent core: residual-strength candidates are permitted after alignment.
- Phase-incoherent/reconstructed core: no residual mixing; accept or revert only.

Store measured coherence evidence in the production core lock.

---

## 13. Hawzhin Sorani Fidelity Guard

The guard is the product’s safety boundary. It must be implemented before candidate selection is finalized.

### 13.1 ASR backend contract

```python
class SoraniASR(Protocol):
    metadata: ASRMetadata

    def infer(
        self,
        waveform: NDArray[np.float32],
        sample_rate: int,
    ) -> ASRResult: ...
```

`ASRResult` contains:

- raw transcript;
- normalized transcript;
- token IDs and text;
- token timestamps;
- calibrated token confidence;
- frame timestamps;
- frame-level CTC log-posteriors;
- speech/no-speech posterior if available;
- model/version/hash.

Original-unit ASR is computed once and cached. Candidate and finished variants are compared against that same cached reference.

### 13.2 Sorani normalization

Implement a deterministic, reversible normalization layer that records every transformation.

Normalize only orthographic equivalents such as:

- Arabic/Persian code-point variants where semantically identical;
- Unicode presentation forms;
- optional diacritics under an explicit rule;
- whitespace and punctuation;
- numeral forms under explicit configuration.

Do not normalize away lexical differences, consonant distinctions, dialectal forms, names, or code-switching. Keep raw and normalized streams in memory; reports omit transcript text by default for privacy.

### 13.3 Guard A checks

#### A. High-confidence token anchors

- Align original and candidate token sequences with timestamp-aware weighted edit distance.
- Anchors are original tokens above the calibrated high-confidence threshold.
- Fail on anchor deletion or substitution.
- Fail on excessive timestamp movement.
- Fail on excessive confidence loss.
- Treat candidate insertions near weak original regions as uncertainty, not automatic improvement.
- If the unit contains speech but has insufficient reliable anchors, return `UNVERIFIED`, not `PASS`.

#### B. Frame-level CTC posterior preservation

- Delay-align frame sequences.
- Use monotonic DTW only within a tightly bounded residual warp.
- Compare posterior distributions with Jensen–Shannon divergence and token-specific margins.
- Aggregate robust statistics over voiced and consonant-heavy frames.
- Detect local spikes, not only the mean.
- Fail when calibrated limits are exceeded.

#### C. Timing and duration integrity

- Compare token/phonetic landmark timing.
- Detect systematic local drift, missing spans, duplicated spans, and compression/expansion.
- Fail when a content-bearing region cannot be mapped monotonically.

#### D. Signal integrity

At minimum:

- consonant/presence-band retention;
- transient/onset envelope correlation;
- voiced harmonic continuity;
- spectral-hole detector;
- musical-noise detector;
- clipping introduced;
- unexplained high-frequency synthesis;
- speech attenuation/dropout;
- energy discontinuity at unit edges.

All detectors return raw scores, calibrated thresholds, and reasons.

### 13.4 Verdicts

```text
PASS          all required checks passed
REVERT        one or more checks failed
UNVERIFIED    speech exists but evidence is insufficient
ERROR         guard execution or data contract failed
NO_SPEECH     no speech; neural enhancement is bypassed
```

Only `PASS` may select enhanced or locally finished speech.

### 13.5 Guard B

After local finishing, compare:

```text
reference = accepted pre-finish unit
candidate = locally finished unit
```

Run the same token, posterior, timing, and signal checks with finishing-specific calibration. On failure, use the accepted pre-finish unit.

After global gain/limiting, run a final ASR decision-stability check over every speech unit. If any unit changes verdict, retry once with lower static gain and less limiter action. If the retry still fails, publish the assembled pre-master timeline with only downward peak-safe gain and report the fallback.

### 13.6 Guard isolation

Run the guard in its own worker process. A guard crash or timeout never implies acceptance; it implies original/pre-finish passthrough and a report flag.

---

## 14. Decision policy

### 14.1 Phase-incoherent core

```text
Guard A PASS                         → enhanced unit
Guard A REVERT/UNVERIFIED/ERROR      → original unit
```

Never mix the reconstructed output with the original.

### 14.2 Phase-coherent core

Run the enhancer once, align it, and derive strength candidates:

```text
candidate(s) = original + s × (enhanced − original)
s ∈ [1.00, 0.75, 0.50, 0.25]
```

Evaluate strongest to weakest. First Guard A `PASS` wins; otherwise original.

Before residual blending is enabled for a production core, prove phase coherence on the development corpus and record it in the model lock.

### 14.3 Clean-audio bypass

A calibrated input-quality classifier may bypass the enhancer only when it has a conservative high-confidence clean verdict. This classifier cannot force enhancement; it can only skip unnecessary processing.

If not implemented reliably, process normally and let the guard choose.

### 14.4 Continuity rule

If adjacent units would switch source inside continuous speech and no safe boundary exists:

- merge and re-evaluate as one larger unit; or
- revert the connected speech run to original.

Never crossfade different source types through a word merely to preserve a local quality gain.

---

## 15. Sample-accurate assembly

### Rules

- Operate on integer sample indices in the canonical timeline.
- Context samples are never duplicated in the final output.
- Same-source overlapping model units use deterministic overlap-add.
- Source switches occur inside verified non-speech/low-energy regions.
- Use the shortest artifact-free equal-power crossfade supported by calibration, normally 10–30 ms in non-speech.
- If a safe switch point is unavailable, apply the continuity rule above.
- Maintain per-channel alignment exactly.

### Mandatory postconditions

```text
output_samples == input_samples
output_channels == input_channels
output_sample_rate == input_sample_rate
all samples finite
no uncovered timeline interval
no interval rendered more than once
```

Any violation is a release-blocking bug and a job finalization failure.

---

## 16. Safe deterministic finishing

Do not attempt to imitate RX by building a broad destructive restoration chain. The neural core performs the primary cleanup. Finishing is conservative, detection-gated, bounded, and reversible per unit.

### 16.1 Local finishing stages

Fixed order:

1. DC/subsonic correction when detected.
2. Narrow de-hum for detected 50/60 Hz fundamentals and harmonics.
3. Short click repair only on confidently detected transient defects.
4. Plosive attenuation only on detected low-frequency bursts.
5. Gentle dynamic EQ for verified mud/presence imbalance.
6. Conservative split-band de-essing.
7. Level riding/light compression.

### Safety bounds

- No blanket spectral subtraction.
- No automatic dereverberation after the neural core.
- No exciter, saturation, harmonic synthesis, or super-resolution.
- EQ boost/cut limits are calibrated and conservative.
- De-esser maximum gain reduction is bounded.
- Compression maximum gain reduction is bounded.
- Every stage is bypassed if its detector is not confident.
- Every stage records detection score and applied gain.

### 16.2 Safe-finish retry ladder

For each accepted speech unit:

1. Apply `gentle` preset and run Guard B.
2. If it fails, apply `minimal` preset from the unprocessed accepted unit and run Guard B.
3. If it fails, bypass local finishing.

Do not search arbitrary parameter grids at runtime.

### 16.3 Global loudness and true peak

After unit assembly:

1. measure integrated loudness using a pinned BS.1770/R128 implementation;
2. compute one static gain toward the target, default `−16 LUFS` for stereo and `−19 LUFS` for mono unless product requirements set one target explicitly;
3. apply a transparent look-ahead true-peak limiter with ceiling `−1.0 dBTP`;
4. cap limiter gain reduction; if the cap would be exceeded, lower the applied static gain instead of crushing peaks;
5. measure again and report achieved LUFS, LRA, sample peak, true peak, and limiter reduction.

The limiter is always last. Output is dithered only when converting to integer PCM, using deterministic TPDF dither whose seed is derived from the input hash, config hash, and output channel so it is reproducible without repeating an identical noise sequence across unrelated files.

---

## 17. Job journal, caching, recovery, and atomic output

### 17.1 Workspace

Each job creates:

```text
.voiceclean-work/<job_id>/
├── job.json
├── journal.jsonl
├── decoded/
├── units/
├── cache/
├── reports/
└── candidate-output.wav.tmp
```

`job_id` derives from input hash, effective config hash, core hash, guard hash, and tool version.

### 17.2 Journal

Append-only events:

```text
JOB_STARTED
PREFLIGHT_PASSED
UNIT_DECODED
UNIT_ENHANCED
GUARD_A_COMPLETE
UNIT_SELECTED
FINISH_COMPLETE
GUARD_B_COMPLETE
UNIT_COMMITTED
ASSEMBLY_COMPLETE
FINAL_VALIDATION_PASSED
OUTPUT_PUBLISHED
JOB_COMPLETE
```

Flush and fsync critical state. Resume skips only artifacts whose hashes and schemas validate.

### 17.3 Cache key

```text
SHA256(
  unit_pcm_bytes +
  canonical_sample_rate +
  model_hashes +
  guard_hash +
  effective_config_hash +
  tool_version
)
```

Never reuse cache across a changed guard, model, or config.

### 17.4 Atomic publication

1. Write output and reports inside the private workspace.
2. Validate sample structure, peaks, loudness, hashes, and report consistency.
3. `fsync` files and directory.
4. Atomically rename into the destination.
5. Never overwrite an existing output unless `--overwrite` is explicit; even then, use a backup/replace transaction.

### 17.5 Resource failures

- Preflight checks free disk with safety margin.
- On CUDA OOM: restart worker and retry once with a smaller legal unit; on repeat, passthrough.
- On disk-full during processing: stop safely, preserve journal, publish nothing.
- On model/guard timeout: kill worker, passthrough, continue.
- On final validation failure: publish nothing and preserve workspace for inspection.

---

## 18. Reporting and auditability

### 18.1 JSON report

The report is schema-versioned and includes:

```json
{
  "schema_version": 1,
  "job_id": "...",
  "input": {
    "path": "episode.wav",
    "sha256": "...",
    "sample_rate": 48000,
    "channels": 2,
    "samples": 168595200
  },
  "output": {
    "path": "episode_clean.wav",
    "sha256": "...",
    "sample_rate": 48000,
    "channels": 2,
    "samples": 168595200,
    "integrated_lufs": -16.1,
    "true_peak_dbtp": -1.0
  },
  "core": {
    "id": "...",
    "commit": "...",
    "weight_sha256": {}
  },
  "guard": {
    "id": "hawzhin-ctc",
    "model_sha256": "...",
    "calibration_id": "..."
  },
  "environment": {
    "image_digest": "...",
    "gpu": "...",
    "driver": "...",
    "torch": "...",
    "cuda": "..."
  },
  "summary": {
    "units_total": 156,
    "enhanced": 131,
    "reverted": 18,
    "unverified": 4,
    "error_passthrough": 3,
    "finish_bypassed": 6
  },
  "review_timecodes": [],
  "units": []
}
```

Each unit records:

- exact sample/time range;
- input and candidate hashes;
- enhancer result;
- alignment result;
- Guard A and Guard B raw scores and thresholds;
- chosen strength/preset;
- decision and reason;
- worker restarts/errors;
- finish stage actions.

### 18.2 Privacy

By default, reports do not contain transcript text or audio embeddings. Store token IDs, edit operation classes, confidence statistics, and hashes. An explicit debug profile may store text only in the private workspace.

### 18.3 Human summary

The `.txt` report contains:

- job success/failure;
- output path and hash;
- model/guard versions;
- counts of accepted/reverted/error units;
- review timecodes with concise reasons;
- achieved loudness/peak;
- provenance warnings.

---

## 19. Sorani datasets and split discipline

### 19.1 Minimum corpus

Create a consented, rights-cleared corpus of at least 240 short clips, ideally 8–20 seconds each, from at least 24 speakers.

Cover:

- Slemani and Erbil/Hewlêr varieties, plus other relevant Sorani variation;
- male and female speakers and varied ages;
- quiet studio, untreated room, office, street/traffic, air conditioner, fan, electrical hum;
- close, moderate, and distant microphone placement;
- clipping, codec loss, bandwidth limitation, plosives, clicks, reverberation, intermittent noise;
- fast, slow, emotional, whispered, overlapping, and code-switched speech;
- names, numbers, loanwords, and difficult Kurdish consonant contrasts;
- clean recordings that should remain essentially untouched.

Every clip requires a human-verified Sorani transcript and degradation labels.

### 19.2 Immutable splits

- **Calibration set:** threshold fitting only.
- **Development set:** engineering and candidate comparison.
- **Acceptance set:** locked before final selection; never used to tune thresholds or code.
- **Corruption set:** deliberately content-altered counterexamples used to measure guard false accepts.

Write canonical manifests, sort deterministically, hash them, and record hashes in all calibration/acceptance artifacts.

### 19.3 Corruption suite

Create controlled counterexamples from consented audio:

- consonant substitution by cut/splice;
- syllable deletion;
- word deletion;
- repeated word/span;
- timing shift;
- local dropout;
- high-frequency consonant removal;
- vocoder-like smoothing;
- false word insertion;
- speaker/timbre perturbation where feasible.

Corruptions must remain realistic enough to challenge the guard, not merely produce obvious waveform damage.

---

## 20. Calibration

`voiceclean calibrate` must:

1. validate corpus and split hashes;
2. compute all guard features for intact and corrupted pairs;
3. fit token confidence calibration if needed;
4. select thresholds that produce zero false accepts on the calibration corruption set;
5. evaluate false-revert rate on intact calibration audio;
6. test unchanged thresholds on the development set;
7. emit ROC/PR data and stratified results by speaker, dialect, degradation, and sample rate;
8. write `guard-calibration.json` with all hashes and software versions;
9. refuse to overwrite an existing production calibration without a new calibration ID.

Thresholds may be conservative. False reverts reduce cleanup; false accepts can change speech.

Do not tune thresholds on commercial baseline outcomes or to increase the apparent enhanced percentage.

---

## 21. Benchmark and human listening

### 21.1 Automated benchmark

For each candidate and baseline, collect:

#### Fidelity

- reference CER/WER where transcript support is reliable;
- high-confidence anchor substitutions/deletions;
- CTC posterior divergence;
- timing drift;
- speaker similarity as a supporting metric only;
- guard pass/revert/unverified rates;
- corruption false-accept rate.

#### Quality

- native-listener clarity and naturalness;
- DNSMOS, NISQA, UTMOS as supporting metrics;
- PESQ/ESTOI/SI-SDR only for paired synthetic data;
- consonant-band and transient retention;
- residual noise and reverberation measures where valid;
- loudness/peak compliance.

#### Operations

- runtime factor;
- peak VRAM/RAM;
- crash/OOM/timeout rate;
- reproducibility and output integrity.

No perceptual metric is allowed to override a confirmed content change.

### 21.2 Blind listening

Implement a local randomized ABX/pairwise tool:

- hide system identity and order;
- loudness-match samples for comparison without rewriting benchmark files;
- collect separate ratings for intelligibility/fidelity, naturalness, clarity, artifacts, and preference;
- require native Sorani listeners;
- store anonymized votes and confidence intervals;
- prevent the same listener from seeing system labels.

Aim for at least three independent native ratings per acceptance clip. Report uncertainty; do not present tiny differences as decisive.

---

## 22. Hard release gates

A release is forbidden unless every gate passes on the locked acceptance set.

### Linguistic gates

- Zero human-confirmed word substitutions or deletions caused by the system.
- Zero guard false accepts on the locked corruption set.
- High-confidence anchor accuracy is not worse than original within each degradation class.
- No unresolved `UNVERIFIED` unit may use enhanced audio.

### Audio-integrity gates

- Exact input/output sample count, rate, and channel count.
- No NaN/Inf, missing interval, duplication, channel swap, or unexplained drift.
- No introduced hard clipping.
- True peak at or below the configured ceiling.
- Clean studio clips are normally bypassed or produce no audible degradation.

### Quality gates

- Native-Sorani fidelity rating is non-inferior to original and each external baseline.
- Clarity/naturalness preference is at least as good as Adobe Podcast and RX Dialogue Isolate, with uncertainty reported.
- Consonant and transient retention is better than or equal to RX at matched noise reduction.
- Any model that sounds better but changes speech is disqualified.

### Reliability gates

- All segment worker exceptions, hangs, invalid outputs, and OOMs revert safely.
- Interrupted jobs resume without duplicate or missing work.
- Atomic publication tests pass under simulated crash and disk-full conditions.
- Reference-platform policy decisions are repeatable.
- No high-severity dependency vulnerability without an approved documented exception.
- Code, weights, and known dataset provenance are approved for the intended use.

### Performance target

Quality takes priority, but the production reference should process at least in real time on the dual-3090-Ti workstation. If it cannot, document the measured result; do not reduce quality or disable guards merely to meet speed.

---

## 23. Testing strategy

Tests are implemented alongside each module.

### 23.1 Unit tests

Cover every branch of:

- Sorani normalization;
- token alignment and anchor logic;
- posterior divergence;
- signal detectors;
- policy decisions;
- alignment rejection;
- source continuity;
- output validation;
- report schema;
- finish detectors and bounded actions.

### 23.2 Property tests with Hypothesis

For arbitrary valid audio arrays/configs:

- output length equals input length;
- no NaN/Inf;
- timeline coverage is exactly once;
- passthrough is sample-exact before global delivery gain;
- disabled stages do not mutate audio;
- cache keys change when any relevant artifact changes;
- reports validate against schema;
- invalid model output can never be accepted;
- `UNVERIFIED` can never select enhanced audio.

### 23.3 Golden regression

Maintain rights-clear short fixtures for:

- clean speech-like audio;
- noise;
- hum;
- click/plosive;
- sample-rate conversions;
- source switching;
- worker failure.

Compare:

- deterministic report excluding runtime fields;
- waveform hashes on the reference platform where achievable;
- numeric tolerances elsewhere;
- exact policy decisions;
- loudness and true-peak statistics.

### 23.4 Integration tests

- real selected enhancer checkpoint;
- real Hawzhin ASR;
- 5 s, 30 s, and long-form files;
- mono, duplicated dual-mono, split-speaker stereo;
- all supported sample rates;
- CUDA and CPU correctness paths.

Mark GPU/model tests separately; normal CI uses fakes.

### 23.5 Chaos/fail-safe tests

Inject:

- enhancer exception;
- enhancer hang;
- guard exception/hang;
- CUDA OOM;
- worker kill;
- malformed model output;
- wrong length/rate;
- NaN/Inf;
- disk full;
- interrupted atomic rename;
- corrupted cache;
- corrupted checkpoint;
- FFmpeg timeout;
- invalid media timestamps.

Expected behavior must be explicitly asserted.

### 23.6 Mutation testing

Run mutation tests on `guard/`, `policy/`, `alignment/`, and `assembly/`. Critical safety mutations must be killed. A surviving mutation that can alter acceptance behavior blocks release.

### 23.7 Coverage

- 100% branch coverage for policy/verdict/assembly/output-validation code.
- At least 90% branch coverage overall, excluding vendored upstream code.
- Coverage numbers do not substitute for property, chaos, or acceptance tests.

---

## 24. Code quality, security, and supply chain

### Required tools

- `uv` for environment and lockfile;
- `ruff` for formatting and linting;
- `mypy --strict` for blocking type checks;
- `pytest` and `hypothesis`;
- `coverage.py` and `pytest-cov`;
- `pre-commit`;
- `pip-audit` or equivalent vulnerability scan;
- CycloneDX SBOM export;
- secret scanning;
- license/provenance manifest generation.

Do not add a second formatter, linter, or blocking type checker without a proven gap.

### Security rules

- Never load untrusted pickled checkpoints.
- Prefer `safetensors`.
- If upstream `.pt` files are unavoidable, require source/hash verification and use `weights_only=True` where compatible.
- Disable remote code execution/trust flags.
- No model downloads during normal processing.
- No network access in the production container after provisioning.
- No telemetry.
- Private workspaces use restrictive permissions.
- Sanitize filenames and prevent path traversal.
- Run subprocesses without a shell.
- Keep FFmpeg current within the pinned release and scan it in the SBOM.
- Reports exclude transcript text by default.

### CI gates

Every pull request:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src tests eval research
uv run pytest -m "not integration and not gpu" --cov=voiceclean --cov-branch --cov-report=term-missing --cov-fail-under=90
uv run pip-audit
```

Release/nightly on the reference GPU:

```bash
uv run pytest -m "integration or gpu or chaos"
uv run voiceclean doctor --strict
uv run voiceclean acceptance data/acceptance/manifest.jsonl
scripts/generate_sbom.sh
scripts/run_release_checks.sh
```

---

## 25. Implementation phases and Definitions of Done

Do not start phase N+1 until phase N passes its Definition of Done.

### Phase 0 — Evidence, licensing, and compatibility spike

Tasks:

- inspect official candidate repos/model cards;
- record commits, weights, licenses, sample rates, dependencies, and inference commands;
- build isolated candidate runners;
- verify each runner on one short file;
- determine exact reference Python/PyTorch/CUDA/FFmpeg versions;
- create initial model registry and provenance report.

Definition of Done:

- every listed candidate either runs end-to-end or has a reproducible blocker;
- exact working environments are captured;
- no unsupported license assumption remains;
- `docs/model-provenance.md` and ADRs exist.

### Phase 1 — Repository, schemas, environment, and doctor

Tasks:

- create repository layout;
- configure `uv`, Ruff, mypy, pytest, Hypothesis, pre-commit, CI;
- implement config/report/corpus schemas;
- implement hash/model-lock validation;
- implement `voiceclean doctor` preflight skeleton;
- build pinned reference container.

Definition of Done:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict src tests eval research
uv run pytest
uv run voiceclean doctor
```

All pass in the reference container.

### Phase 2 — Audio spine and crash-safe job engine

Tasks:

- probe/decode/encode;
- channel classification;
- canonical timeline;
- segmentation;
- no-op enhancer;
- assembly;
- journal/cache/resume;
- atomic publication;
- initial JSON/TXT reports.

Definition of Done:

- all supported fixture formats round-trip with exact duration/sample count;
- ambiguous stereo fails safely;
- interrupted job resumes;
- crash before rename publishes no final file;
- property and chaos tests pass.

### Phase 3 — Guard before real enhancement

Tasks:

- ASR protocol;
- `FakeSoraniASR`;
- Hawzhin CTC adapter;
- Sorani normalization;
- token anchors;
- CTC posterior comparison;
- timing and signal checks;
- calibration artifact format;
- verdict/policy tests.

Definition of Done:

- every deliberate synthetic token corruption is rejected by unit tests;
- `UNVERIFIED` and guard errors always passthrough;
- original ASR result caching works;
- guard schemas and reports are complete;
- normal CI does not require GPU or real ASR.

### Phase 4 — Production worker and alignment

Tasks:

- isolated enhancer worker;
- heartbeat, timeout, restart, OOM handling;
- output validation;
- delay/fractional alignment;
- local drift/coherence analysis;
- phase-coherent strength policy;
- phase-incoherent accept/revert policy.

Definition of Done:

- valid candidate output is processed;
- every malformed/hung/crashed response reverts;
- exact timeline invariants pass;
- deterministic policy decisions repeat on the reference platform.

### Phase 5 — Safe finishing and Guard B

Tasks:

- detection-gated local stages;
- gentle/minimal/bypass ladder;
- Guard B;
- global loudness measurement/static gain;
- bounded true-peak limiting and dither;
- stage-level reports.

Definition of Done:

- synthetic detector tests meet attenuation/preservation specifications;
- final output obeys loudness/peak bounds;
- finishing-induced corruption fixtures revert to pre-finish audio;
- clean fixture produces no unnecessary processing.

### Phase 6 — Corpus, calibration, and benchmark harness

Tasks:

- validate immutable manifests;
- corruption generator;
- calibration command;
- candidate/external baseline ingestion;
- metrics and statistics;
- blind ABX tool;
- immutable run artifacts.

Definition of Done:

- starter corpus runs end-to-end;
- thresholds are fitted without acceptance-set leakage;
- CSV/JSON/Markdown summaries agree;
- blind tool randomizes and records reproducibly.

### Phase 7 — Sorani model selection and production freeze

Human/data requirement:

- complete the calibration, development, acceptance, and corruption corpora;
- run all candidates and external baselines;
- complete native-listener review.

Definition of Done:

- hard gates evaluated;
- exactly one eligible core selected;
- `production-core.lock.toml` finalized;
- unused runtime candidate dependencies removed;
- calibration rerun with frozen production core;
- final acceptance set passes without tuning.

### Phase 8 — Reliability, security, and release hardening

Tasks:

- full chaos suite;
- long-form and multi-channel tests;
- vulnerability/license/SBOM checks;
- deterministic reference runs;
- operational docs;
- performance profiling without quality shortcuts;
- final provenance review.

Definition of Done:

- every hard release gate passes;
- no unresolved critical/high defect;
- no `TODO`, `FIXME`, placeholder, or stub in the production path;
- clean build from repository + locked artifacts succeeds;
- a second clean reference machine reproduces policy decisions;
- release package contains README, operations guide, model provenance, SBOM, locks, schemas, and tests.

---

## 26. Agent progress protocol

At the end of each phase, update `STATUS.md` with:

```text
Phase:
Implemented:
Commands run:
Tests passed:
Artifacts produced:
Known limitations:
Open blockers:
Next phase:
```

Rules:

- Include actual command output summaries and paths.
- A failed command remains visible until fixed.
- Every deviation gets an ADR.
- Never lower thresholds or reduce tests to obtain completion.
- Never replace real integration tests with mocks and call the phase complete.
- Never report benchmark superiority before the locked acceptance and blind-listening results exist.

---

## 27. Final acceptance checklist

The agent may declare V1 complete only when all are true:

- [ ] One and only one production enhancer is frozen.
- [ ] Production model, weights, environment, calibration, and acceptance artifacts are hash-locked.
- [ ] Source audio is never modified.
- [ ] Input/output sample rate, channels, sample count, and duration match exactly.
- [ ] Every model/guard/finish error safely reverts.
- [ ] Guard A and Guard B are active and calibrated.
- [ ] No `UNVERIFIED` unit uses processed speech.
- [ ] Locked corruption set has zero false accepts.
- [ ] Locked acceptance set has zero confirmed system-caused word substitutions/deletions.
- [ ] Clean speech is not audibly degraded.
- [ ] Native-Sorani blind tests meet the quality gates.
- [ ] Adobe and RX comparisons are complete and honestly reported.
- [ ] Long-form resume, crash, OOM, timeout, corrupted-cache, and disk-full tests pass.
- [ ] Atomic publication and output validation pass.
- [ ] Reference-platform policy decisions are reproducible.
- [ ] CI, GPU integration, mutation, security, provenance, license, and SBOM gates pass.
- [ ] README states limitations without claiming perfect fidelity or guaranteed recovery.
- [ ] No placeholder, disabled safety check, hidden network call, or unresolved critical defect remains.

---

## 28. Known limits that must appear in the README

1. No current enhancement model can guarantee that speech content is never altered.
2. The guard is bounded by Hawzhin ASR accuracy and calibration coverage.
3. A genuinely missing phoneme cannot be recovered with certainty; V1 reverts instead of inventing one.
4. Challenge rankings do not establish Sorani superiority because Sorani was not an evaluation language.
5. Objective quality metrics can reward pleasant but linguistically incorrect reconstruction; native listening and fidelity gates remain decisive.
6. Enhanced audio must not be represented as untouched forensic evidence.
7. Commercial distribution requires separate approval of code, model weights, and training-data provenance.
8. Cross-platform bit-identical output is not guaranteed; the immutable Linux/NVIDIA reference environment defines reproducibility.

---

## 29. Verification ledger as of 2026-08-18

These sources informed the frozen decisions above. The implementation agent must re-check them before pinning artifacts.

1. **URGENT 2026 Track 1 official task, metrics, and ranking design**  
   https://urgent-challenge.github.io/urgent2026/track1/

2. **Official URGENT 2026 BSRNN/BSRNN-Flow baseline code and checkpoints**  
   https://github.com/urgent-challenge/urgent2026_challenge_track1

3. **GAP-URGENet official implementation and listed five challenge checkpoints**  
   https://github.com/Xiaobin-Rong/gap-urgenet

4. **ClearerVoice-Studio and `MossFormer2_SE_48K` phase-sensitive-mask implementation**  
   https://github.com/modelscope/ClearerVoice-Studio

5. **FastEnhancer 48 kHz release, if an additional predictive candidate becomes necessary**  
   https://github.com/aask1357/fastenhancer

6. **NVIDIA RE-USE model card and noncommercial terms**  
   https://huggingface.co/nvidia/RE-USE

7. **PyTorch reproducibility limitations and deterministic controls**  
   https://docs.pytorch.org/docs/stable/notes/randomness

8. **uv lockfiles and immutable Docker pinning guidance**  
   https://docs.astral.sh/uv/concepts/projects/sync/  
   https://docs.astral.sh/uv/guides/integration/docker/

9. **FFmpeg official documentation**  
   https://ffmpeg.org/documentation.html

---

## 30. Final implementation directive

Build the smallest system that satisfies every invariant and release gate above.

Do not optimize for the percentage of audio enhanced. Optimize for:

1. zero accepted linguistic corruption in the tested domain;
2. natural, clear, non-muddy Sorani speech;
3. safe recovery from every foreseeable operational failure;
4. reproducible and auditable decisions;
5. one-command long-form processing on the reference workstation.

When quality and certainty conflict, preserve the original and flag the timecode.
