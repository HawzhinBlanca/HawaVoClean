# Conversational audio editing — development implementation

Implemented 19 November 2026. This is a source-checkout feature, not a new signed
production release or a claim of universally best sound quality.

## Open the app

Double-click **Open HawaVoClean.command** in the repository. The launcher opens
the native desktop app and starts its private local chat service if necessary.
It uses installed assets; it never downloads models or enables cloud chat.

Load a recording and try:

- “Clean the voice and reduce room noise.”
- “Keep the speech and remove the music.”
- “Remove the dog barking.”
- “Actually, keep just the music instead.”

Read the proposed edit, choose **Apply to recording**, then compare Original and
Cleaned before saving. Asking the assistant does not start an audio job. Changing
the source retires the previous plan. Unsupported or unresolved requests ask for
clarification instead of silently running another operation.

## What runs where

The local Ollama model interprets text only. Qwen3:8b is installed in this
checkout's ignored `.local-ai` directory and served on a private loopback port.
The existing Ollama service on port 11434 is unchanged. Cloud chat is disabled.

Dialogue cleanup runs through the existing guarded Wiener/DFN3/WPE profiles.
Explicit Extract/Remove jobs run a separately installed, pinned AudioGhost
SAM-Audio Lite worker before optional dialogue cleanup. Extract selects the
model's target; Remove selects its generated residual. Other sound effects do
not automatically pass through a speech enhancer.

The queue, cancellation, A/B listening, existing export controls and immutable
audio/report generations remain shared. Requests go to workers through stdin,
not prompt-bearing process arguments. Source digests, selected stems, model and
runtime identities, settings and cleanup stage boundaries are retained in the
processing report. Signal checks do not prove that words or speaker identity
were preserved.

## AudioGhost model setup still required

The isolated AudioGhost code/runtime is installed locally. Meta's gated
SAM-Audio model weights are not installed, and Hugging Face authentication is
not configured. Extract/Remove therefore stay unavailable and report setup
needed. Dialogue cleanup and local chat work independently.

The account owner needs access to the intended
[SAM-Audio model](https://huggingface.co/facebook/sam-audio-large) and must
authenticate locally before downloading it. Never paste a token into chat.
See [runtime setup](audioghost-runtime.md) for the isolated environment and
read-only installation diagnostic.

Large/High is the highest exposed separation preset. It is not a comparison
against every model, precision variant or reranker. Current separation supports
one selected output, mono/stereo, at most 120 seconds and 256 MiB of input.
Streaming decode enforces the actual duration bound even when container metadata
is wrong. Long-file separation and real Sorani listening quality remain
unqualified.

## Verification

- Real local-model prompts test speech extraction, noise removal, non-speech
  extraction, dialogue cleanup, follow-up changes and unsupported requests.
  `.local-ai/planner-live-verification.json` records the initial live protocol;
  additional paraphrase checks and earlier failures remain beside it. These
  generated instructions contain no user audio or personal data.
- A real browser check uploaded a synthetic two-second WAV, planned cleanup,
  applied it through the job API, and reached a completed master with A/B and
  report downloads. This proves plumbing, not real speech quality.
- Native Electron shell smoke test and all 45 desktop tests pass.
- UI production build, TypeScript checking and all 574 UI tests pass.
- Focused backend checks cover request contracts, model readiness, source
  lineage/publication, bounded decoding, symlink rejection, durable jobs,
  cancellation and chat response validation. Final counts are recorded below
  after all implementation changes.
- The broad unit run encountered sandbox-denied subprocess/shared-memory
  operations; those files are rerun with OS permissions. A pre-existing Resolve
  test expects nine trusted-sender checks while committed code has ten. This
  integration changes neither Resolve code nor that assertion.
- No real AudioGhost inference, listening comparison, enrolled-speaker
  restoration, signed packaging or final release qualification is claimed.

Final verification: **157 focused backend tests**, **574 UI tests**, **45 desktop
tests**, the native shell smoke test, Ruff, strict mypy and schema generation
checks pass. **13/13 live local instruction cases** pass. All 16 training/checkpoint
tests and all 11 server lifecycle tests pass when rerun with OS permissions;
process-supervision tests also pass outside the sandbox. The unrelated existing
Resolve nine-versus-ten assertion remains failing. The initial broad run's
obsolete doctor-wording assertion was updated to test the corrected diagnostic
and passes in the focused suite.

## Engineering references

- [Local chat setup](local-audio-chat.md)
- [AudioGhost runtime setup](audioghost-runtime.md)
- [Original dated assessment](audioghost-integration-assessment.md)
- `src/hawavoclean/assistant.py`: bounded local text planning and validated plans
- `src/hawavoclean/audio_edit.py`: separation, cleanup and publication
- `src/hawavoclean/separation/`: isolated worker and installation diagnostics
- `ui/src/components/AudioAssistant.tsx`: review-before-apply chat workflow

