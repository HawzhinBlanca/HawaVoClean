# AudioGhost integration assessment

Update, 19 November 2026: the local chat workflow, isolated separation adapter,
job dispatch and UI are now implemented. The pinned runtime is installed;
gated model weights and real separation-quality qualification remain outstanding.
See [implementation status](audio-assistant-implementation.md). The assessment
below preserves the findings and proposal as they stood on 18 November.

Assessed: 18 November 2026. HawaVoClean source: `2ff0a07` plus the existing
uncommitted health-signals changes. AudioGhost source: `79653d1a789f4fc73893d10733e2324ed088cecf`
(v1.2, 17 November 2026). This is an integration proposal; no separation
implementation or model-quality qualification was performed.

## Recommendation

Add an optional, explicit **Extract sound / Remove sound** operation powered by
AudioGhost's SAM-Audio Lite engine. Keep HawaVoClean's UI, durable queue, local
security boundary, mastering and verified publication. Run separation in its
own Python environment and supervised process before speech segmentation.

Begin with one prompt and a 10–30 second preview on SAM-Audio Small. Measure
quality and speed on this Mac before building the full application integration.
Use explicit English prompts initially; Sorani audio retention needs its own
listening evaluation. Text prompting does not establish Sorani prompt support,
speaker identity verification, or missing-band restoration.

## Current application

| Layer | Verified implementation | Integration consequence |
| --- | --- | --- |
| Enhancement | Wiener production, DFN3/WPE studio, DFN3 lowband, registered in `src/hawavoclean/enhancement/factory.py` | Useful finishing engines after dialogue extraction |
| Core protocol | `enhance(waveform, sample_rate)` in `enhancement/protocol.py` | No prompt, anchor, target/residual, progress or cancellation fields; separation needs a distinct stage contract |
| Pipeline | Decode → speech segmentation → enhancement → guards → finishing/mastering → publication | Separate before speech segmentation; the existing speech-only worker loop would omit non-speech targets and repeatedly load a large model |
| API | Closed `ProcessingRequestV1`, discriminated strategies, capability reporting | New operation requires an explicit schema extension and matching job dispatch; a new profile name alone cannot implement it |
| Desktop/UI | React/Vite UI in an Electron shell | Reuse the application; AudioGhost's Next.js frontend is unnecessary |
| Queue/output | Subprocess jobs, durable state, immutable generations and reports | Reuse lifecycle and publication; bind the new stage's provenance to its artifacts |
| Restore/cloud | Capabilities currently block both Restore routes and cloud | AudioGhost does not qualify either feature |

The repository already had modified operations/readiness documents, server app
and jobs, plus new health-signals source/tests. Those files were not changed by
this assessment.

## What to reuse upstream

[AudioGhost](https://github.com/0x0funky/audioghost-ai/tree/79653d1a789f4fc73893d10733e2324ed088cecf)
wraps SAM-Audio for text-guided sound separation. The reusable implementation is
[`backend/workers/sam_lite.py`](https://github.com/0x0funky/audioghost-ai/blob/79653d1a789f4fc73893d10733e2324ed088cecf/backend/workers/sam_lite.py),
particularly `load_lite_model()` and `separate()`. It supports temporal anchors,
overlapping chunks, seeded inference, progress and cooperative cancellation.

Its output contains a target stem and a separately generated residual stem.
For extract, select the target; for remove, select the residual. Do not assume
that subtracting the target from the source reproduces the model's residual,
or that the two generated stems sum exactly to the original.

The separate
[`sam_audio_lite.py` CLI](https://github.com/0x0funky/audioghost-ai/blob/79653d1a789f4fc73893d10733e2324ed088cecf/sam_audio_lite.py)
provides a simple isolated-environment benchmark path without starting Redis,
Celery, its API or its UI. A production wrapper should call the engine through
structured input/output rather than parse CLI progress text.

## Proposed flow and contract

```text
HawaVoClean source ID + explicit separation request
  → existing authenticated job queue
  → decode/resample to the installed model's declared sample rate
  → one supervised SAM-Audio worker in a separate environment
  → target + residual in private temporary storage
  → validate selected stem and restore the declared export timeline/rate
  → optional HawaVoClean speech cleanup, if the selected material is dialogue
  → existing mastering and verified generation publication
  → level-matched preview and explicit export
```

Do not hardcode the model rate as 48 kHz: upstream reads it from the codec
configuration. Handle resampler delay, exact output length, channel order and
codec padding explicitly. Keep the original source as the report's source;
the selected stem is a derived intermediate, not a new recording with lost
lineage. Compare final output against the original in A/B listening.

A proposed strategy is `kind: "separate"` with `operation: "extract" | "remove"`,
`prompt`, `modelSize`, `quality`, `seed`, optional signed temporal anchors, and
an optional dialogue cleanup profile. Introduce it through the existing API's
versioning rules, update Python and TypeScript contracts together, and preserve
idempotency across every setting. Keep it out of automatic Smart Safe selection
until separately qualified. An alternative is a staged v2 request if v1 is
contractually frozen; existing clients should continue to work.

Expose a dedicated separation capability. Availability must depend on a
verified model pack, exact runtime dependency probe, supported provider and
successful local installation. Report experimental maturity while quality or
provider qualification is incomplete. Missing assets must not trigger downloads
inside an offline processing job.

## Constraints discovered

1. **Runtime separation is necessary.** This checkout uses Python 3.14.5,
   NumPy 2.5.2, torch 2.13.0 and torchaudio 2.11.0. AudioGhost's
   [tested constraints](https://github.com/0x0funky/audioghost-ai/blob/79653d1a789f4fc73893d10733e2324ed088cecf/backend/constraints.txt)
   instead pin torch 2.9.0, torchaudio 2.9.0, torchcodec 0.9.1, NumPy below 2,
   transformers below 5 and huggingface_hub below 1. Use its Python 3.11 setup
   in a separate environment first; do not replace HawaVoClean's dependencies.
2. **Mac performance is unproven here.** The host is arm64. Upstream defaults
   to CPU on Apple Silicon; MPS is an opt-in experimental provider. Its README
   reports roughly 5.1 GB for Small weights plus a 0.9 GB text encoder, and
   approximately 4.2 GB minimum CUDA VRAM in its tested configuration. Those
   CUDA figures are not measured Apple unified-memory requirements or local
   speed results. Start Small/Fast, measure cold and warm runs, and explicitly
   use FP32 for the initial MPS experiment as upstream's task wrapper does.
3. **Chunked inference is not bounded host memory.** `separate()` pads the
   complete input, creates noise for the whole duration, and retains chunk
   results until assembly. Do not inherit HawaVoClean's six-hour/8 GB media
   qualification. Initially cap duration, channels and concurrency; long-media
   support needs measured streaming/staging work.
4. **Guards need operation-specific meaning.** Removing a sound intentionally
   changes the spectrum. Existing strict spectral identity and speech-only
   guards cannot certify arbitrary extraction. Keep shape, finite-sample,
   delay, channel and clipping checks; add task-specific quality evaluation.
   For dialogue outputs, test word/consonant retention and speaker identity
   through real listening. A quiet intended residual must not automatically
   fail merely because the target occupied most of the source.
5. **Faults must remain visible.** Cancellation, timeout, model mismatch and
   OOM should terminate/clean the supervised worker and fail the separation
   operation truthfully. An optional original-audio fallback must be reported
   as bypassed, never as successful extraction/removal. Use one loaded worker
   per device initially; the speech worker pool must not multiply model memory.
6. **Pin provenance.** Record source and model digests, AudioGhost/SAM commits,
   prompt/settings, chosen stem, seed, precision, provider, stage timings and
   guard decisions in the processing record. A seed is not a cross-device
   bit-identical-output guarantee. Keep prompts and paths out of health
   telemetry; any prompt retained locally belongs in the user-owned report.
7. **Separate licenses/access.** AudioGhost's wrapper is
   [MIT-licensed](https://github.com/0x0funky/audioghost-ai/blob/79653d1a789f4fc73893d10733e2324ed088cecf/LICENSE).
   SAM-Audio code/weights use Meta's separate
   [SAM License](https://huggingface.co/facebook/sam-audio-small/blob/main/LICENSE),
   and [model access is gated](https://huggingface.co/facebook/sam-audio-small).
   Preserve the applicable notices and install/access terms in the optional
   pack workflow. The wrapper's MIT license does not cover Meta's materials.

## Application issue to reconcile

`hawavoclean doctor` prints a fixed statement that Smart Safe is BLOCKED
(`src/hawavoclean/cli.py:173`), while `capabilities_v1()` advertises
`smart_safe` as available/qualified (`server/app.py`). The adjacent `preserve`
reason also says qualified Smart Safe routing is unavailable. Existing Smart
Safe pipeline tests pass, but these statements disagree. Derive diagnostics
and API readiness from one contract before adding another engine. This
assessment does not decide which qualification claim is correct.

## Delivery order and acceptance

1. **Measure feasibility:** isolated runtime; existing model-access approval;
   fixed-seed Small/Fast previews of real Sorani speech with music, traffic,
   competing speech, clean dialogue, and at least one non-speech target. Record
   cold/warm latency, peak RAM, CPU/MPS behavior and paired listening judgments.
2. **Implement the stage:** structured worker protocol, cancellation/deadlines,
   duration limits, validated stems and explicit truthful failure reporting.
   Add request/capability/job dispatch and full provenance before UI controls.
3. **Add the small UI:** Extract/Remove, prompt, optional time hints, preview and
   export. Default to one prompt and one selected output. Add independent stem
   export only when publication/report contracts cover both artifacts.
4. **Qualify:** test unsupported packs/providers, offline operation, crash/OOM,
   cancellation, chunk boundaries, silence and no-target cases, mono/stereo,
   44.1/48 kHz timelines and output/source lineage. Require real Sorani listening
   evidence before calling it production speech-safe. Assess long-media support
   separately.

## Checks actually run

- `python -m hawavoclean.cli doctor`: exit 0, all reported checks pass, with
  the contradictory Smart Safe text described above.
- Direct `capabilities_v1()` inspection: production/studio/lowband available;
  Smart Safe advertised qualified; Restore and cloud blocked.
- Focused Python tests: **52 passed in 18.66 seconds**, covering
  `test_server_contracts_v1.py`, `test_natural_capabilities.py`,
  `test_smart_safe_pipeline_integration.py`, and `test_health_signals.py`.
- UI TypeScript build check (`tsc -b`): completed without diagnostics.
- UI tests: **548 passed across 32 files**. The first sandboxed run passed
  539 tests but could not run the nine mock-engine cases because loopback
  access was denied; rerunning the suite with loopback access passed all 548.
- No AudioGhost weights downloaded or inference run; no subjective audio
  quality, Mac speed, installed desktop workflow or Resolve-host validation
  is claimed.
