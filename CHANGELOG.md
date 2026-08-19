# Changelog

All notable changes to the HawaVoClean system will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — engine bridge for the first UI screen (`docs/ui-contract.md`)
- `hawavoclean serve --port 0 --token TOKEN [--ui-dir DIR]`: a loopback-only
  FastAPI/uvicorn server (`hawavoclean/server/`). Binds 127.0.0.1 only (any
  other `--host` is refused), requires the token on every `/api/*` request
  (header `X-Hawa-Token` or `?token=`), prints exactly one
  `{"event":"ready","port":…,"pid":…,"version":…}` line on stdout once it is
  listening, and then re-points stdout at stderr so nothing else can ever
  appear there. Routes: `GET /api/health`, `POST /api/analyze` (waveform
  min/max/RMS buckets, sine-calibrated 1/12-octave long-term spectrum,
  BS.1770 loudness, noise floor), `POST /api/jobs` + `GET /api/jobs/{id}` +
  `GET /api/jobs/{id}/events` (SSE, ≥50 ms throttle, `: ping` keep-alive) +
  `POST /api/jobs/{id}/cancel`, Range-capable `GET /api/audio`,
  `POST /api/upload`, `POST /api/shutdown` (answers, then exits within 1 s),
  optional static UI mount at `/`. Every error is `{"error","message"}` JSON.
  Client paths must resolve under home, `/Volumes` or the work dir (403
  otherwise): the engine never reads arbitrary files for a web page.
- `hawavoclean/server/jobs.py`: child-process job manager. Each job is
  `python -m hawavoclean.cli process IN -o OUT --profile P [--overwrite]
  --progress-json` (the batch command's isolation pattern), one at a time,
  FIFO queue, stdout JSON lines → status snapshots, stderr tail → failure
  message, cancel = SIGTERM then SIGKILL after 5 s, asyncio-friendly
  subscriptions for SSE. `JobStatus` carries two additive fields beyond the
  contract: `created_at` and a monotonically increasing `seq`.
- `hawavoclean process --progress-json`: one JSON object per line on stdout
  (`progress` events, then `done` or `error`; logs stay on stderr). The
  real stdout is kept on a private descriptor and fd 1 is pointed at stderr
  for the rest of the run, so a library banner or the inherited enhancement
  worker cannot corrupt the stream.
- `run_pipeline(..., on_progress=)` with `hawavoclean/progress.py`
  (`ProgressEvent`, `ProgressCallback`): preflight 0.02, decode 0.05,
  segment 0.08, enhance/guard 0.08→0.80 linearly over units (one `enhance`
  before and one `guard` after every unit, `NO_SPEECH` units included),
  finish 0.80/0.95, publish 0.98. Callback exceptions are logged and
  swallowed; the pipeline never fails because of a progress sink.
- Python extra `ui = [fastapi, uvicorn, python-multipart]`; `httpx2` added to
  `dev` for the FastAPI test client. Tests: `tests/unit/test_server_*.py`,
  `tests/unit/test_progress_*.py` (≈80 tests, ~12 s; new modules ≥98 %
  branch coverage).

### Fixed — engine bridge review pass
- `GET /api/audio` 416 responses now carry `Content-Range: bytes */<size>`
  (RFC 9110; Chromium's media stack reads it to recover the resource length
  when seeking), and a reversed explicit range (`bytes=5-2`) is treated as
  an invalid spec — header ignored, whole file served — instead of a 416.
- `/api/analyze` spectrum: 1/12-octave bands narrower than the FFT main
  lobe (below ~400 Hz at 48 kHz) widen their integration window to it, so
  the contract calibration rule (full-scale sine at a band centre ≈ 0 dB)
  now holds at every band; a 40 Hz sine previously read ≈ −6 dB.
- `POST /api/upload` with a bare `..`/`.` filename no longer targets the
  upload directory itself (saved as `upload.bin`).
- SSE subscriptions are registered inside the response generator, so a
  connection aborted before the body starts can no longer leak a subscriber.
- Job failure mapping tolerates a stderr drain thread kept alive past the
  child by an orphaned grandchild (no more `deque mutated during iteration`).

## [3.2.0] - 2026-08-19

### Added — decay-gated late-reverb suppression (studio core v1.1.0)
User feedback: "the reverb is still there". Measured on the real recording:
single-channel WPE (taps 10) cut the post-phrase tail only 1.6 dB; pushing
WPE harder (taps >= 40) collapsed the VOICE broadband by 7-8 dB and the
guard correctly refused it. A different tool was needed.

- `finishing/dereverb.py`: Lebart/Habets late-reverb estimate subtracted
  only inside decays — frames more than 6 dB below their recent (look-back)
  peak — scaled by decay depth; the voice itself (within 6 dB of its local
  peak) is untouched by construction. Runs after DeepFilterNet3 in the
  studio core. Params hash-locked in `studio-core.lock.toml` (v1.1.0).
- Measured on Flute 09 vs original: room tail -6.2 dB @50 ms, -9.8 dB
  @100 ms after each phrase; voice tonality within ±0.3 dB in every band;
  all 5 units clear the spectral-hole guard with margin (0.05-0.08 < 0.10).
- Setting chosen as the strongest that clears the guard on EVERY unit:
  hotter settings scored 0.14 and the continuity rule cascaded the whole
  take back to original (correct behaviour; wrong setting).
- Found and fixed during development: the decay gate used a look-AHEAD
  maximum (wrong sign, copied from the limiter), so it never engaged on
  smooth decays; and a 1.5 dB voice-protect band dimmed dry speech 1.7 dB
  (now 6 dB, swept on synthetic dry/wet speech). Regression tests pin both.
- Scope, stated honestly: sustained content BETWEEN phrases on this
  recording (a held instrument tone, spectral flatness 0.02, 29% of the
  file, ±17 Hz waver = played content) is not reverb and is not removed.

## [3.1.1] - 2026-08-19

### Fixed — finishing EQ was re-voicing every recording thin and bright
Found by ear ("harsh, treble, bass removed") and confirmed by measurement
on dialogue frames: low-mids -5.7 dB, bass -2.2 dB, presence +2.7 dB vs the
original. DeepFilterNet3 was tonally flat (±0.4 dB); the cause was the
finishing chain's `parametric_speech_eq`. Its "mud" detector used a +2 dB
low-mid/presence threshold that fired on 100% of real voices (measured
+11 to +41 dB — natural speech simply carries that much more low-mid
energy), then applied a -3 dB low-mid cut and +2.5 dB presence boost to
every unit.
- Mud is now EXCESS over a measured normal-voice reference (+36 dB) by more
  than 6 dB; the correction scales with the excess, caps at ~3 dB audible,
  and the blanket presence/air boost is gone.
- After the fix, Flute 09 dialogue bands sit within ±0.2 dB of the original
  (sub-bass -0.8 from the deliberate 75 Hz rumble filter).
- New permanent gate: finishing and the full pipeline must be tonally
  transparent (±1.5 dB per band) on a natural-voice spectrum; a genuine
  +12 dB boom is still corrected, gently.

## [3.1.0] - 2026-08-19

### Fixed — 36 bugs from an adversarial hunt (fuzz harness + 3 parallel reviews)

Every fix landed red-test-first; every repro is a permanent regression test.
A 42-input adversarial fuzz gate (`pytest -m fuzz`) now runs the real CLI.

**Would have hurt users directly**
- `process X -o X` silently destroyed the source; refused at preflight now,
  including report-sidecar collisions. Destination existence and writability
  are checked BEFORE decoding; no workspace leaks on user-error paths.
- A keypress during ffmpeg decode truncated the file and published a
  half-length master as success (ffmpeg inherited the terminal). `-nostdin`
  + `stdin=DEVNULL`.
- MP4 with a video stream first was rejected as "rate=0, channels=0"
  (probe read streams[0]). First AUDIO stream selected; decode pins it.
- Batch: no per-file deadline (a hung file hung the batch) — each file runs
  in a child with a hard timeout; stem collisions (`a.wav` + `a.m4a`)
  silently overwrote — refused up front.
- Interrupts: SIGTERM/SIGKILL of the parent orphaned the worker child
  (holding the model); child now runs a parent-death watchdog; SIGTERM
  unwinds cleanly (exit 130); no partial outputs ever.
- Mastering peaked at 5.5 GB RSS on an 8-minute file (full-file 8x float64
  oversampling) — chunked true-peak, in-place envelope: 895 MB.
- Worker: interpreter HUNG at exit after a child died mid-request (queue
  feeder thread blocked) — queues released on kill; a dead child is now
  noticed in <1 s instead of after the full timeout.
- Limiter crashed at 11025 Hz and other rate/lookahead parities, and on
  1-sample input.

**Guard / DSP correctness**
- GCC-PHAT delay sign was inverted: alignment DOUBLED the delay. Fixed; flat
  correlation and oversized search windows handled.
- Guard failed OPEN on NaN candidates and on empty candidates (unit became
  silence). Fail closed.
- Clipping check rejected any peak-normalised input; musical-noise score
  rejected identical candidates. Both now relative to the original.
- 50/60 Hz hum detection was mathematically impossible at 22.05-48 kHz (3
  FFT bins in the band); dedicated 16384-point check — de-hum now actually
  runs.
- Spectral-hole detector false-rejected clean denoises (scored the lowered
  floor between phrases); continuity rule fired on the wrong side and did
  not cascade; stitch declick keyed off the wrong unit's flag.
- Short (<400 ms) files: sample peak used as LUFS (9 dB gain jump at the
  boundary); ungated mean-square estimate now.
- VAD: DC offset made pauses "speech" and forced cuts land inside words;
  one transient hid quiet speech (threshold anchored to the max frame).
  DC-removed frames, 98th-percentile anchor, local-mean zero crossings.
- Segmentation glued half of arbitrarily long silence gaps onto speech
  units (300 s "speech"); capped to the context window.
- Declared channel_mode never validated against the file (failed after full
  processing); streamed WebM with no duration rejected; <8 kHz rates
  crashed; dual-mono output lost L/R bit-identity to per-channel dither.

**Honesty / operations**
- Pipeline now verifies calibration-artifact integrity (not just audit).
- `verify` honours the CONFIGURED true-peak ceiling; eval gate fails on an
  empty manifest; eval/benchmark/calibrate exit with documented codes.
- `phase_coherent` / `model_sample_rate` validated against the core at
  preflight (config error, exit 2) and passed through to the worker.

## [3.0.1] - 2026-08-19

### Fixed — guard precision, found by a DJI field recording
- Spectral-hole detector scored a lowered noise floor in the GAPS between
  phrases as "holes" (measured 0.66 on a clean denoise), rejecting good
  restoration. It now evaluates only active frames and only bins that
  carried signal in the original; the score is the fraction of signal bins
  wiped. Thresholds rescaled to 0.10 (all profiles); calibration artifacts
  re-derived. Red-first tests: a clean floor drop scores ~0; a real 1–3 kHz
  wipe inside the signal is still caught.
- Continuity rule fired on the wrong side: `forced_boundary` marks the cut
  at a unit's END, but the rule also reverted for a reverted LEFT neighbour
  across a natural pause. It now fires only across an actual forced cut
  (enhanced audio meeting original across a mid-speech split). The
  pre-existing test that encoded the old behaviour was corrected.

## [3.0.0] - 2026-08-19

### Renamed — HawaVoClean (formerly Hawzhin VoiceClean)
- Project, package, and CLI renamed: `voiceclean` -> `hawavoclean`
  (breaking: imports, the console command, `HAWAVOCLEAN_*` environment
  variables, and new report suffixes `.hawavoclean.json` / `.txt`).
  Existing reports with the old suffix remain readable via explicit paths.
- No behavior changes; full verification battery re-run after the rename.

## [2.1.0] - 2026-08-19

### Added — a real neural restoration core
- `StudioVoiceCore` (`studio-dfn3-48k-v1`): WPE dereverberation +
  DeepFilterNet3 speech enhancement. Weights vendored and hash-locked in
  `studio-core.lock.toml`; digests verified at preflight and by
  `audit-models`. Optional install: `uv sync --extra studio`.
- `--profile studio`: integrity-mode guarding, neural core, same mastering
  chain. Measured on a real recording: noise floor −27 dB, SNR +26.7 dB,
  signal preserved within 0.3 dB.
- Guard modes: `strict_spectral` (unchanged default) vs `integrity`
  (timing/envelope/artifact/collapse protections without spectral-identity
  gating). Studio thresholds measured against real guard scores; the
  calibration artifact records the measurement provenance.
- Core registry (`enhancement/factory.py`): every registered core carries
  its lockfile and an implementation-hash callable; preflight and audit
  verify weights digests and that lock tables reconstruct `params_hash`.

## [2.0.0] - 2026-08-19

### The honesty release

An audit on 2026-08-19 found that this codebase misrepresented itself:
the "neural enhancement core" was a classical Wiener filter with a
fabricated weights digest; the "Sorani CTC ASR" fidelity guard contained no
acoustic model; calibration metrics (including the headline 0.0
false-accept rate) were hardcoded literals; the model registry listed
evaluations that never happened; and the audit report falsified verdicts on
cached re-runs. This release removes every fabrication and fixes every
reproduced defect. It is a breaking release: names, config keys, report
schema, and artifact locations all changed to match reality.

### Changed (honesty)
- `ProductionEnhancerCore` -> `WienerSpectralEnhancer` (`wiener-dd-48k-v1`):
  named for the algorithm it implements. Provenance is now the parameter
  set, hash-locked and verified at preflight and by `audit-models`.
- `HawzhinSoraniASR` -> `SpectralSignatureProbe`; `SoraniASR` protocol ->
  `SpectralProbe`; `ASRResult` -> `ProbeResult` with `raw_signature` /
  `frame_distributions` fields. Module docstrings state plainly that the
  probe detects spectral change and is not a speech recognizer;
  `test_probe_is_not_asr.py` pins the boundary.
- `eval/calibrate.py` now MEASURES accept/revert rates over corruption
  profiles (mild/standard/severe); hardcoded metrics deleted. Artifacts
  carry measurement provenance or no metrics at all.
- `research/benchmark.py` now benchmarks the real pipeline; fabricated
  candidate scores deleted. Model registry deleted (nothing was evaluated).
- Datasets regenerated as declared-synthetic: dialect "synthetic",
  verified_by_human false, no transcripts claimed.
- torch/torchaudio removed (they were imported to print a version string).
- README, STATUS, RISKS, docs/ rewritten to describe the implemented
  system; BLUEPRINT.md marked historical.

### Fixed (audited defects, each with a red-first regression test)
- Audit falsification: resume cache deleted — every run recomputes and
  reports its own verdicts; verdicts are identical across re-runs.
- Workspace leak: scratch space removed on success; test suite fails loudly
  if pre-existing workspace state could serve cached results.
- Limiter: true peak now provably at or under the ceiling (sliding-minimum
  lookahead + slope-limited attack + verified trim); hard-clip fallback
  removed; property-tested at 8x oversampling with no tolerance.
- Continuity rule: enforced before records are built, channel-aware, and
  visible as `original_continuity` in reports.
- Stitch: boundary declick no longer renders unit heads twice
  (content-conservation tested).
- Report: real `probe_hash` and per-unit `output_sha256`; Guard B bypass
  reports the scores of the attempt it describes; placeholder strings
  banned by test.
- `audit-models` verifies params hash, license allowlist, and calibration
  integrity, and exits non-zero on tampering.
- Acceptance gates restructured: explicit conditionals (survive `python
  -O`), structured failures, can return FAILED, plus a did-something floor.
- CLI works from any directory (packaged resources + env overrides);
  publication stages on the destination filesystem with rollback.
- Chaos tests now inject real faults: SIGKILLed worker, hung worker,
  NaN/wrong-length/silent model output, ENOSPC at publish.
- `scripts/mutation_gate.py`: 12 behavior mutations must each break the
  suite.

## [1.0.0] - 2026-08-19

### Added
- Complete Master Implementation Blueprint v2.0 execution.
- High-performance audio spine with FFprobe media probing, float32 PCM decoding, and TPDF dithered encoding.
- Auto-channel classification supporting mono, dual-mono identical, and split-speaker stereo.
- Speech activity detection and speech-unit utterance grouping with context windows.
- Hawzhin Sorani Fidelity Guard with Unicode normalization, token anchors, frame-level CTC log-posterior JS divergence, timing preservation, and signal integrity detectors.
- Isolated enhancer worker architecture with crash/timeout recovery and heartbeat protocol.
- Multi-stage deterministic finishing chain (de-hum, click repair, plosive attenuation, dynamic EQ, de-esser, level riding) guarded by Guard B.
- BS.1770-4 integrated loudness normalization and look-ahead true-peak limiter with ceiling enforcement.
- Resumable job journal and atomic workspace publishing.
- Schema-validated immutable JSON reports and human-readable TXT review summaries.
- Comprehensive CLI suite: `doctor`, `process`, `verify`, `calibrate`, `benchmark`, `acceptance`, `audit-models`.
