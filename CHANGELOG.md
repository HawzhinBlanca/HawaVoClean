# Changelog

All notable changes to the HawaVoClean system will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — one failing unit no longer discards a whole file of passing ones

The continuity rule forbids enhanced audio from butting against original audio
across a *forced* cut — a boundary the segmenter made inside continuous speech
because a speech interval outran the maximum unit length. The remedy was to
revert the enhanced unit, and reverting is symmetric: it creates a seam on the
unit's far side. Iterated to a fixed point on a recording whose every boundary
is forced — continuous speech, no pauses to cut at — **one failing unit
reverted the entire file**.

Measured on Flute 09 (94.6 s, production profile): five of six units passed the
guard, and the report read `enhanced: 0/6`, `continuity_reverted: 5`. The
listener received original audio with loudness normalisation and nothing else.

The remedy is now a fade. An enhanced unit that meets original audio across a
forced cut fades its own enhancement back to the original recording over the
30 ms before the joint (and/or after it, when the cut is on its left). At the
joint both sides are then the original recording **bit for bit**, so the step
is exactly zero rather than merely small, and the difference between the two
renderings is spread over 30 ms instead of one sample.

- **Flute 09, production: `enhanced: 0/6` → `5/6`.** Speech/floor separation
  **27.65 dB → 34.88 dB (+7.23 dB)**; `continuity_reverted: 5 → 0`,
  `continuity_crossfaded: 0 → 1`; `finish_bypassed: 6 → 1`.
- **Nothing else moved.** Of the four committed reference masters
  (`test_output/perf-ref-hashes.txt`), three reproduce **byte for byte** —
  Flute 09 studio and both teat1vo profiles have no continuity reverts, so
  they have no seams to fade. Only the one file the rule was damaging changed.
  No configuration key was added, renamed or removed, so `config_hash` — and
  with it the dither seed and the last bit of every sample — is untouched.

What the fade costs, and what the seam was actually worth
(`docs/continuity-taper.md` carries the full evidence):

- Sample step at the joint, hard cut: **4.7% of local RMS**. Faded: **0**.
- Spectral difference between the two renderings over the final 30 ms:
  **3.05 dB**, at a local level of **-51 dBFS**. The old remedy spent 7.23 dB
  of separation to hide it.
- In the assembled timeline the cut is not even an outlier: mean
  frame-to-frame spectral change **6.1 dB** across the joint against a
  file-wide mean of **7.3 dB**. A 5–150 ms sweep of the fade length moved no
  spectral or separation metric, so 30 ms is set by the one thing that does
  constrain it — the enhancement residual's fade is an amplitude modulation
  at ~17 Hz, below where modulation reads as texture rather than transition.
- The seam is small because it is *placed* small: the segmenter hunts a ±1 s
  window for the quietest zero crossing, and Flute 09's five forced cuts landed
  **13.6 to 22.8 dB below the file's median frame RMS**. That search is
  bounded and 13.6 dB is not silence, so the seam is still guarded — cheaply.

Fail-closed is intact. A unit too short to afford the fade (under 4× its
length, i.e. 120 ms at 48 kHz) still reverts, and the fixed-point iteration
still converges. The fade material is the unit's **own** original audio, never
the enhanced context the pipeline computes and discards: the neighbour across
the cut ships original *because the guard rejected its candidate*, so
extending this unit's enhanced context into that neighbour's territory would
publish audio no guard ever scored for that time range.

- `hawavoclean/policy/continuity.py`: `enforce_source_continuity` becomes
  `resolve_source_continuity`, returning a `ContinuityResolution` — the
  decisions, the reverts it still had to make, and a per-unit fade plan. The
  new `apply_continuity_taper` blends as `(1 - w) * original + w * finished`
  with a raised-cosine `w` whose endpoints are pinned to exactly 0.0 and 1.0,
  so the outer sample is the original to the bit and the audio outside the
  fade windows is untouched.
- The fade is applied **after** finishing, because the seam is between the
  *finished* enhanced audio and the original — fading any earlier would leave
  the finishing EQ's own step sitting at the joint.
- Report: new `summary.continuity_crossfaded`, and a
  `continuity_taper(in=…,out=…)` entry in the unit's `finish_actions`. A faded
  unit's `final_decision` stays `enhanced`, because it is.
- Mutation gate: **M14–M21** — fade on the wrong edge, left-edge seams
  ignored, a too-short unit faded instead of reverted, the pipeline planning a
  fade it never applies, the ramp becoming a hard step, the fade resolving to
  the candidate instead of the original, the fade planned at the wrong sample
  rate, and a continuity revert filed under the guard's own REVERT. 21/21
  caught, every mutation owner-credited.

**Adversarial audit, and what it broke.** Seven independent auditors attacked
the change and every finding was handed to a second agent whose job was to
refute it; 4 of 16 survived. All four are now closed, and each was re-derived
here before being acted on rather than taken on report:

- **The "nothing can cancel" justification was false for the `studio`
  profile.** The fade was defended on the grounds that the two renderings are
  phase-coherent, so the blend is the original with its residual scaled and
  cannot cancel. That holds for `wiener-dd-48k-v1` (measured worst dip below
  `min(original, enhanced)`: **−0.02 dB** over 435 windows) and not for
  `studio-dfn3-48k-v1`, which ships `phase_coherent = false` — the very reason
  `policy/strength.py` refuses to residual-blend it — where the measured worst
  is **−3.63 dB** (**−1.87 dB** restricted to windows above −40 dBFS; median
  **+0.07 dB**, whole-file correlation with the input +0.9896). The fade is
  still applied to both, deliberately: studio's `strength_ladder = [1.0]`
  leaves the guard no partial rung, so reverting costs the unit's whole
  enhancement — 7.23 dB — against at most 1.87 dB in one band for part of
  30 ms. The docstring and `docs/continuity-taper.md` now carry the measured
  distribution instead of the false premise, the 30 ms rationale that rested on
  it has been rewritten as the reasoned choice it is, and
  `test_continuity_blend_cannot_cancel.py` pins the bound so a core that made
  the fade destructive could not arrive unnoticed.
- **A hard step passed the whole suite.** Replacing the raised cosine with
  `w = (t >= 0.5)` kept the joint sample original, stayed monotone and bounded,
  and shipped a discontinuity 15 ms upstream of the low-energy zero crossing
  the segmenter chose — worse than the seam the fade exists to remove. The
  monotonicity test now bounds the ramp's slope (M18).
- **The joint sample was never asserted where it can be violated.** Fading
  toward `dec.selected_waveform` instead of the original, and applying the fade
  before finishing rather than after, both passed everything. Neither could be
  caught by a fixture with no forced cuts, and no shipped fixture can have one
  — every fixture is 8 s while `hard_max_group_s` has a schema floor of 10.0.
  `test_the_joint_of_a_real_forced_cut_is_the_original_recording` tiles a real
  fixture into an unbroken 24 s speech interval so the **segmenter** makes the
  cut and the **policy** plans the fade, then asserts the joint sample of the
  assembled pre-master timeline is the decoded original bit for bit (M19). It
  also pins the fade's absolute length, so passing a wrong sample rate at the
  call site — which silently ships a 10 ms fade — now fails (M20).
- **Pre-existing: a continuity revert could be filed under the guard's own
  REVERT** and nothing noticed. They are different events and only the second
  is a cost this rule is accountable for; conflating them is how the cascade
  stayed invisible as long as it did (M21).

Refuted and recorded as such: the reference-hash reproducibility complaint (all
four reproduced first try by two independent auditors, and again here), the
report schema/UI compatibility complaints, the `MAX_TAPER_FRACTION` bound
complaint (unreachable — the shortest real unit adjacent to a forced cut in a
full segmenter sweep was 11.1 s, 93× the threshold), and three documentation
arithmetic complaints.

Still unproven rather than disproven, and stated as such in
`docs/continuity-taper.md`: **the fade's audible benefit**. Every spectral and
separation metric is identical for a hard cut, 5 ms, 30 ms, 150 ms, and for
deleting the continuity rule outright. All measured benefit belongs to removing
the cascade. The fade is justified by argument; only the cascade removal is
justified by measurement.

### Fixed — four config keys that were declared and never read

`runtime.device`, `runtime.num_threads`, `runtime.worker_memory_limit_mb` and
`input.supported_sample_rates` had zero uses outside `config.py`: four promises
the configuration made and the code did not keep. All four now do something,
and making them real moved **no published sample** — the four committed
reference masters (`test_output/perf-ref-hashes.txt`, Flute 09 and teat1vo
across the production and studio profiles) reproduce byte for byte, with every
unit decision and guard verdict unchanged.

- New `hawavoclean/runtime.py` enforces the `[runtime]` section.
  `load_config()` arms it: loading a profile resolves its device, publishes a
  per-worker CPU budget when it asks for a pool, and publishes its memory
  budget. The channel is the process environment because the enhancement core
  lives in a `spawn`-ed subprocess whose only other inbound channel is a fixed
  argument tuple — and because `OMP_NUM_THREADS` has to be set before the
  child imports torch to have any effect at all.
- **`device`** — resolved for real, `mps` added to the accepted values, and
  plumbed into the studio core, which pins DeepFilterNet's own device lookup
  (in memory; the vendored `config.ini` is never rewritten, its digest stays
  locked) so model and feature tensors cannot land on different devices. An
  explicitly requested device this machine cannot provide is a **designed
  error before any audio is touched**, never a silent fall back to the CPU.
  A classical-DSP core reports `cpu` whatever was asked for, because that is
  what ran.
  - `auto` resolves to **cpu**, and that is a measurement, not caution:
    on this reference machine (torch 2.13, DeepFilterNet3, 20 s unit @ 48 kHz)
    MPS took 731 ms against the CPU's 339 ms — **2.2x slower** — and its
    output is not bit-identical (max |Δ| 1.8e-08 at the core, 7.2e-07 through
    the finishing chain). End to end on Flute 09 (94.6 s, studio profile):
    CPU 8.20 s (RTF 0.087) vs MPS 12.87 s (RTF 0.136). Guard verdicts, chosen
    strengths, LUFS and true peak were identical across the two devices;
    guard scores drifted in the sixth decimal. **MPS is not worth adopting on
    this hardware.** `AUTO_DEVICE_PREFERENCE` is the single constant a future
    backend has to earn its way into.
  - Provenance: `HawaVoCleanReport.environment` gains **`compute_device`** —
    the device that actually ran. A GPU computes different samples, and two
    machines running the same config can differ here while sharing a config
    hash, so a result must never be attributable to the wrong compute path.
  - The device is deliberately **not** in any core's `params_hash` or
    lockfile. A lockfile pins *what the model is*, and that is the same model
    on every device; the device belongs to the environment, which is exactly
    where BLUEPRINT invariant 8 already scopes reproducibility. Folding it in
    would fail `audit-models` on a machine merely for owning a GPU and would
    need one locked hash per device to mean anything. `audit-models` and
    `doctor` still pass unchanged.
- **`num_threads`** — defined as what it will actually be used for: the size
  of the enhancement worker pool (units enhanced concurrently), explicitly
  *not* an intra-op/BLAS thread count, since conflating the two oversubscribes
  the machine by `num_threads` squared. `worker_pool_size()` and
  `threads_per_worker()` are the shared arithmetic; above a pool of one, each
  worker is given `cores // pool` via `OMP_NUM_THREADS`/`MKL_NUM_THREADS`,
  never overwriting a value an operator set themselves. At the default of 1
  nothing is set at all — which is why the reference masters did not move.
  (Measured separately: a thread budget is numerically safe here — all four
  reference runs are byte-identical under `OMP_NUM_THREADS=7`.)
- **`worker_memory_limit_mb`** — enforced, by the core policing itself before
  it accepts each unit (`runtime.check_memory_budget`, raising the previously
  unused `WorkerOOMError`): a process that has already blown the budget stops
  taking work, the parent recycles it, and the refused unit takes the existing
  fail-closed path to ORIGINAL audio. Not `RLIMIT_AS`, and that is measured
  too: every memory rlimit is *unsettable* on this project's reference
  platform (macOS returns `RLIM_INFINITY` and rejects the `setrlimit`, while
  `RLIMIT_NOFILE` on the same process succeeds), and where it can be set it
  caps reserved virtual address space — which torch reserves in gigabytes
  without touching — so it would kill healthy runs rather than runaway ones.
  Headroom check: the studio worker peaks at ~1.3 GB on a 25 s unit against
  the shipped 8192 MB budget.
- **`input.supported_sample_rates`** — enforced as the accepted *envelope*:
  `min(...)` is the floor the media probe refuses below and `max_sample_rate`
  the ceiling it refuses above, and `probe_audio` now takes the envelope as a
  parameter with the schema's declaration as its default, so
  `audio/probe.py`'s floor is derived from the configuration instead of being
  a constant that agreed with it by luck. Deliberately not a membership test:
  every rate between the endpoints is resampled to the core's 48 kHz and
  processed correctly today, so rejecting 11.025 kHz for being absent from the
  list would refuse material the engine handles — a capability regression
  dressed up as rigour. The endpoints are what the UI's advisory rate warning
  mirrors. The schema also now rejects an empty envelope, non-positive rates,
  and an envelope whose floor is above `max_sample_rate`.
- Nothing was deleted, and that was a decision with evidence behind it: the
  config schema is hashed into `config_hash`, which is hashed into the job id,
  which seeds the master's deterministic dither. Removing a dead key rewrites
  the last two LSBs of every published sample (measured: max |Δ| 2.4e-07
  across all four reference masters). Keys were therefore made to mean
  something instead. Retiring one is a deliberate act that reissues the
  reference hashes, and `test_making_these_keys_real_did_not_move_the_config_hash`
  pins the production profile's hash so it cannot happen by accident.

### Changed — speech units are enhanced concurrently, and a batch stops reloading the model

Two scheduling changes, no arithmetic changes. **All four committed reference
masters reproduce byte for byte** (`test_output/perf-ref-hashes.txt`, Flute 09
and teat1vo across production and studio), every unit decision in the four
committed reference reports is identical, and ten consecutive runs of the same
file produce one SHA-256.

- **A worker pool instead of one worker.** `pipeline.py` dispatched units one
  at a time through a single subprocess, though speech units are independent
  by construction — that independence is what the segmentation architecture
  is *for*. `EnhancementWorkerPool` now enhances several at once and hands the
  answers back **indexed by unit**, never by completion order, so scheduling
  cannot reach the report or a sample. The pipeline consumes them in unit
  order and guards unit *i* while the pool is still enhancing *i+1*, so
  guarding and enhancing overlap instead of queueing.
  Measured, median of 3 interleaved A/B runs against the same tree:
  Flute 09 (94.6 s, 6 units) **3.29 s -> 2.87 s** on production and
  (5 units) **6.51 s -> 4.62 s** on studio (RTF 0.069 -> 0.049). A file with a
  single speech unit is unchanged (1.88 / 2.77 s), which is the point: with
  one unit the pool starts no threads at all and stays in the caller's thread.
- **The pool's licence is that both cores are stateless across calls**, which
  is measured rather than assumed: the same unit run first, second, and alone
  hashes identically on the Wiener core and on DeepFilterNet3
  (`tests/unit/test_enhancement_pool.py`). If DFN3 had carried STFT or
  normalisation state between calls, a pool would have changed the audio.
- **Sizing.** N = min(configured, cores - 2, memory, units). `runtime.num_threads`
  is honoured through `runtime.worker_pool_size` when an operator raises it;
  its default of 1 is read as "unset" rather than "one worker", because that
  value is hashed into `config_hash` -> job id -> dither seed, so raising it in
  a shipped profile would move published samples. The memory cap is drawn
  first from a deliberately pessimistic 1 GB/worker (which is what makes an
  unmeasured concurrent start safe) and then from the first worker's measured
  RSS: **500 MB for a warm DeepFilterNet3 worker, 133 MB for a Wiener worker**
  on the reference machine. `HAWAVOCLEAN_ENHANCE_WORKERS` overrides both.
- **Fail-closed is per unit, and now tested through the pool.** A SIGKILLed
  worker fails only the unit it was holding: that unit publishes ORIGINAL
  audio and records `original_error`, the units queued behind it are
  untouched, the slot is restaffed, and the job completes
  (`tests/chaos/test_pool_fail_closed.py`). A slot that cannot start at all
  hands its unit back to a slot that can, rather than costing it.
- **`batch` keeps one warm child across files.** The per-file child process
  existed for isolation — a stuck decoder or a wedged model must cost one
  file, not the batch — and every file still runs under a hard deadline in a
  process the parent can kill, now with its own process group so a breach
  takes the decoder and the workers with it. What is dropped is the part
  isolation never needed: a fresh interpreter and a fresh model load per file.
  Measured on 6 files: **11.81 s -> 6.76 s** (production) and
  **16.99 s -> 8.94 s** (studio), every master byte-identical to the
  single-file reference.
- **`close()` stopped pretending.** It put `STOP` on the queue and terminated
  in the next statement, so no child ever read the message. Both paths are now
  real, and the default is the signal — because that is what the measurement
  says: straight to SIGTERM is **2.6 ms** (Wiener) / **8.0 ms** (DFN3) against
  **43.9 ms / 182.2 ms** for a graceful exit, which has to unwind torch. The
  worker owns a model and two queues, all of which die with the process, so
  there is nothing for politeness to buy. `close(grace_s=...)` still asks.
  (Teardown was never the 0.95 s the perf brief estimated: it measures 2.6-8 ms
  before this change and after it.)
- **Guard A was left in the parent, on evidence.** Threads make it *slower*,
  not faster — 6 units measured 0.585 s sequential against 1.251 s across 6
  threads (0.47x), because it is GIL-bound. The only route left is another
  process, and the parent is where the judge belongs: moving it into the
  process that produced the candidate would put the verdict in the hands of
  the thing it is judging.

### Added — multi-pass enhancement: `process --passes N|auto`
- `hawavoclean process IN -o OUT --passes N` (N = 1..4, default 1) runs the
  full pipeline N times, each pass re-enhancing the previous pass's mastered
  output; `--passes auto` adds passes (max 4) only while the guard does not
  regress (enhanced-unit count >= previous pass) AND measured speech/floor
  separation improves by >= 0.5 dB — a pass that fails is DISCARDED, recorded
  with its reason, and the previous pass ships. New module
  `hawavoclean/multipass.py`; the default `--passes 1` path is byte-identical
  to a run before the flag existed (same code path, same audio bytes, same
  progress stream). `batch` does not take `--passes` — the flag is refused.
- Intermediate passes keep FULL finishing including mastering — the measured
  recipe on the muffled teat1vo lab source (production profile): pass 1
  clears the guard only at strength 0.50 and its tonal restoration lifts the
  presence band; that restored output lets pass 2 run at strength 1.00,
  deepening speech/floor separation 14.9 -> 19.7 -> 23.6 dB (source -> pass 1
  -> pass 2) with zero musical-noise inflation. A third pass converges
  (23.6 -> 23.2 dB), so auto stops at 2 kept passes there and records pass 3
  as discarded.
- Separation metric: frame RMS (2048/1024), p90 minus p10 in dB, on the mono
  mix — `multipass.speech_floor_separation_db`, implemented over a cumulative
  energy sum (memory linear in the signal) and unit-tested identical to the
  naive framed computation.
- Provenance: `HawaVoCleanReport` gains `passes: list[PassRecord]`
  (default `[]`; schema_version stays 1, every pre-existing report still
  validates and `verify` round-trips). Each `PassRecord` carries pass_index,
  input/output SHA-256 (chained pass to pass), unit counts, the distinct
  chosen strengths, separation_db, integrated_lufs, and the discarded flag +
  reason. The report's `units` remain the FINAL (shipped) pass's records; the
  `.txt` summary gains a MULTI-PASS AUDIT TRAIL section when more than one
  pass is on record.
- Fail-closed: destination preflighted before pass 1 decodes a sample;
  intermediate outputs live in a `multipass-*` temp dir under the work root,
  removed on success, error, and SIGINT/SIGTERM (chaos-tested with the
  SIGSTOP freeze protocol); any pass raising fails the whole run — auto's
  discard applies only to a completed pass that failed the criteria. The
  final master + amended report publish through the same atomic staging as a
  single-pass run (`JobWorkspace.publish_atomically`, now a staticmethod).
- `--progress-json` with multiple passes: each pass's events are rescaled
  into its share ([k-1, k]/N; auto treats the current pass as the last until
  another starts) and carry `"pass":{"index":k,"total":N|null}`. The
  single-pass stream is unchanged, byte for byte. Mutation gate: M13 deletes
  the auto discard and must be caught by its owning test.

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

### Added — `POST /api/peaks`: windowed waveform peaks (ui-contract addendum 1)
- New route for the waveform's zoom/pan re-query (goal box E3, and E1 in
  part). Request `{"path", "start_s", "end_s", "buckets"}`, response
  `PeaksWindow`: mono-mix `peaks.min`/`peaks.max`/`rms_db` per bucket over
  the requested span only, plus `samples_per_bucket` so a client knows when
  it has reached one sample per bucket and cannot zoom further. `end_s` is
  clamped to the duration, `buckets` (1..8000, default 1200) is clamped down
  to the sample count so every bucket still covers ≥ 1 sample. Path policy,
  auth and error shape are identical to `/api/analyze`: 400 `bad_request`
  for a start at/after the end of the file, an empty or reversed window, a
  negative or non-finite bound (`NaN`/`Infinity` are valid JSON to
  `json.loads`, so the model rejects them explicitly) or an out-of-range
  bucket count; 403/404 for the path.
- `hawavoclean.audio.decode.decode_audio_window(probe, start_s, end_s)`:
  decodes a span, not a file. ffmpeg gets `-ss` **before** `-i` (input seek)
  plus `-t`; the soundfile fallback reads a frame range. `decode_audio` is
  untouched. Measured: a 5 s window out of a 3-hour, 2.07 GB recording costs
  **+4.8 MB peak RSS** and 27 ms warm (812 ms cold, all of it the probe's
  whole-file SHA-256), against ~2 GB for a full decode.
- Two seek traps the windowed decode has to defuse, both found by testing a
  window against a full decode of the same file: the first frame after a
  seek into a lossy stream has no MDCT overlap partner (a quarter-second
  pre-roll is decoded and discarded), and an explicit `-ss 0` hands back an
  mp4's encoder-priming samples that a plain decode trims (so the head of a
  file is not seeked at all). Each was worth 0.3–0.7 full scale of error.
- A window longer than 4 Mi samples (87 s at 48 kHz — far longer than
  anything on screen) is bucketed by streaming reduction instead of one
  decode: peak RSS for bucketing a whole file is then constant in its
  length. Asking for a 3-hour file as one window measured 8.5 GB before and
  155 MB after.
- The last 8 probes are cached by path + mtime + size: probing SHA-256s the
  whole file, which is noise next to a full decode but was the entire cost
  of serving a window (0.8 s per gesture on a 2 GB file, now 27 ms).
- Tests: `tests/unit/test_server_peaks.py`, `tests/unit/test_decode_window.py`
  (86 tests). The load-bearing ones assert a window's buckets equal a
  full-file analysis of the same span — including windows aligned to no
  bucket boundary — that the chunked reduction is numerically identical to
  the single decode, and that at `samples_per_bucket == 1` the response *is*
  the raw samples. The contract's memory rule is proved, not assumed: a
  generated 30-minute/346 MB file is served from a fresh subprocess whose
  peak RSS is measured (+5.7 MB for a 5 s window, +154 MB for the whole file
  as one window) and then deleted.

### Changed — `POST /api/analyze` streams the file instead of decoding it (goal box E1)
- Analyze used to call `decode_audio` on the whole file. Measured before this
  change: a 3 h / 2073.6 MB recording cost **12,756.8 MB of peak RSS** and
  36.11 s; a 30 min / 345.6 MB one cost 3039.2 MB and 5.56 s. It is now a
  single streaming decode pass with four accumulators, and peak RSS is
  **flat in file length**: 222.3 MB for the 3-hour file (32.94 s) and 227.0 MB
  for the 30-minute one (5.65 s) — 57x less memory and 9 % less wall time on
  the long file, with every number it returns unchanged (-14.71 LUFS,
  -3.71 dBTP, -17.15 dB noise floor, identical spectrum). Nothing about the
  response shape changed.
- New `iter_decode_audio(probe, chunk_samples, timeout_s)` in
  `hawavoclean/audio/decode.py` (additive; `decode_audio` and
  `decode_audio_window` are untouched): one ffmpeg process, no seek, stdout
  read in fixed-size blocks, stderr to a temp file so a chatty decoder cannot
  deadlock on a pipe nobody drains. The sample stream is **bit-identical** to
  `decode_audio`, verified on PCM, FLAC, 44.1 kHz and the project's AAC-in-mp4
  test media at four different chunk sizes. Chunk default 512 Ki frames
  (~11 s at 48 kHz), chosen from a sweep on the 3-hour file: 256 Ki costs
  88 MB / 33.6 s, 512 Ki 118 MB / 32.1 s, 2 Mi 269 MB / 31.8 s, 4 Mi 366 MB /
  31.5 s.
- The four reductions, each written to land on exactly the grid its whole-file
  counterpart used, and each proved against it in
  `tests/unit/test_server_analyze_streaming.py` (which keeps the old
  whole-file implementation as its oracle):
  - **overview buckets** — running min/max/sum-of-squares per bucket, the same
    machinery `/api/peaks` already used (`_BucketReducer` is now shared by
    both). Bit-equal to the whole-file buckets on PCM.
  - **1/12-octave long-term average spectrum** — a running sum of per-frame
    power over a running frame count. Identical by construction (same frames,
    same hop grid, same divisor); measured worst case **1.4e-14 dB**.
  - **BS.1770 integrated loudness** — the K-weighting biquads keep their
    filter state across chunks (an IIR split with `lfilter`'s `zi` is exact),
    one mean square is accumulated per 400 ms gating block per channel, and
    the absolute (-70 LUFS) and relative (-10 LU) gates are applied at the end
    exactly as pyloudnorm applies them, from pyloudnorm's own coefficients.
    Measured worst case over mono / stereo / 6-channel / gate-heavy /
    sub-400 ms / near-silent / 44.1 kHz / real AAC fixtures at four chunk
    sizes: **1.2e-7 LU** (contract: 0.01 LU). The residue is float64 block
    sums against pyloudnorm's float32 ones and does not grow with chunk count.
  - **true peak** — `oversampled_peak_envelope` fed a rolling buffer that
    always has `EDGE` (4096) samples of real audio on both sides of every
    finalised region, against a polyphase FIR whose half-length is ten input
    samples. Measured difference: **exactly 0.00 dB**, as is sample peak.
- One behaviour change, deliberate and measured: the overview grid and
  `duration_s` are now laid on the *container* sample count — the timeline the
  playhead, `/api/peaks` and the `<audio>` element already use — instead of
  the decoder's. For PCM/FLAC the two are the same number. For a lossy
  container they are not: the project's AAC test file decodes 71 samples
  (1.5 ms) past the length its container declares, so `/api/analyze` and
  `/api/peaks` used to report durations 1.5 ms apart. They now agree, which
  closes the discrepancy carried forward from web iteration 1. Spectrum,
  loudness and true peak still see every decoded sample.
- `analyze_audio` now goes through the bounded probe cache, so an analyze
  immediately followed by a burst of zoom queries no longer re-SHA-256s the
  file (0.8 s on a 2 GB input).
- `tests/unit/test_server_analyze_streaming.py`: 16 tests. The `@pytest.mark.slow`
  memory proof generates a 10-minute and a 30-minute file under `test_output/`,
  measures each in a fresh subprocess (peak RSS is a process-lifetime
  high-water mark, so an in-process delta would be contaminated), asserts the
  growth is under 400 MB *and* that the two sizes differ by less than 64 MB —
  flat, not merely smaller — then deletes them. Observed: 115 MB file →
  +114.8 MB, 346 MB file → +116.9 MB.

### Added — an upload size cap, and the evidence that uploads never buffer (goal box E2)
- Verified what Starlette actually does rather than assuming it: each *file*
  part of a multipart body goes into a `SpooledTemporaryFile(max_size=1 MiB)`,
  so anything past a megabyte is on disk before the route runs.
  `MultiPartParser.max_part_size` (also 1 MiB) looks like a cap but applies
  only to non-file fields. Measured on a live engine with a 1.07 GB file:
  peak RSS **133.8 MB against an idle 127.8 MB — 6.0 MB of growth**, 1.26 s,
  and the saved file is byte-exact.
- `POST /api/upload` now copies with `while chunk := await file.read(
  UPLOAD_CHUNK_BYTES)` (module constant, 1 MiB) and deletes the partial
  destination and its directory if anything fails part way through, so a
  half-written file can never masquerade as audio to the other endpoints.
- New configurable cap: `DEFAULT_MAX_UPLOAD_BYTES` = 8 GiB, overridable with
  `HAWAVOCLEAN_MAX_UPLOAD_BYTES` (0 disables it) or
  `create_app(max_upload_bytes=…)`. A malformed or negative value falls back
  to the default rather than silently uncapping the endpoint.
  `UploadSizeLimitMiddleware` refuses an over-sized body twice: from the
  declared `Content-Length` before a byte is read (measured: 413 in 0.14 s
  with 1.2 MB of RSS growth and nothing written), and from a running byte
  count on the receive channel for a client that sends
  `Transfer-Encoding: chunked` and declares no length (measured: 413 after
  17.0 MB against a 16 MiB cap, nothing left on disk). It sits inside the
  token check, so an unauthenticated flood is still 401.
- `tests/unit/test_server_upload_streaming.py`: 13 tests — the spool
  threshold, that the part reaches the route already on disk, that the copy
  loop asks for exactly the configured chunk as many times as it takes, the
  partial-file cleanup, both 413 paths, the boundary case, the zero-disables
  case, and the environment parsing including junk values.

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

## [3.3.0] - 2026-08-20

### Fixed — a muffled recording came back muffled, 15.7 dB louder
Reported on a real 24 s take ("embarrassingly bad"), and the measurement said
the same thing: EVERY band of the output had moved by exactly the same
+15.7 dB. The chain had applied a flat loudness gain and nothing else.

Two reasons, both real. The `mud_imbalance_db` detector scored the file at
38.7 dB against a 42.0 dB gate and missed by 3.3 dB — but even a hit would not
have helped, because the only correction behind that gate is a 2 dB low-mid
trim, and this file's deficit was ~24 dB of missing presence, not an excess of
low-mids. A one-number low-mid/presence ratio cannot see a recording whose low
end is NORMAL and whose consonant region was never captured.

**New: measured tonal restoration** (`detect.measure_speech_tilt`,
`eq.apply_tonal_restoration`, `finishing.tonal_restoration` /
`max_tonal_gain_db`). Band levels are measured against a speech-intelligibility
target — 90-300 Hz, 1.5-3 kHz and 3-6 kHz, each relative to the 300-1000 Hz
body of the voice — and the correction is the bounded difference: a low shelf
(cut ≤ 6 dB, lift ≤ 4 dB), a presence bell (≤ 10 dB) and a brilliance bell
(≤ 12 dB). Nothing is corrected above 6 kHz at all; air that was never recorded
cannot be restored, only imitated with hiss.

**The target was calibrated on the two recordings this is judged by**, not on a
textbook spectrum. Measured (p75 of speech-active frames, dB relative to body):

| | 90-300 | 1.5-3k | 3-6k |
|---|---|---|---|
| pinned 3.1.1 natural-voice fixture | +13.0 | -22.0 | -33.4 |
| Flute 09 — user approved this sound | +7.0 | -27.8 | -33.4 |
| the reported file | +3.0 | -26.4 | -44.7 |

The approved recording and the reported one are within 1.4 dB of each other
from 90 Hz to 3 kHz and 11 dB apart above it. Any correction driven by the low
end or the presence band would have moved both — which is 3.1.1 ("harsh and
treble sounding, dialogs bass removed lot") happening a second time. So the
targets sit below both acceptable references, a 7 dB deadband sits on top of
that, and the correction stops AT the deadband edge. A voice inside the band
gets exactly 0.0 dB and no voice can be pushed past it: over-brightening is
impossible by construction rather than by tuning.

**Two gates keep it off bands that have nothing in them.** A band must carry
DYNAMICS — its loud level minus its own quiet level ≥ 12 dB, ramped, because
speech swings tens of dB across syllables and hiss, buzz, codec noise and
dither do not (approved recording +42 dB at 3-6 kHz, reported file +18 dB, a
brick-wall-lowpassed control +1.4 dB, correctly refused). And it must have been
CAPTURED at all: a backstop below -48 dB relative to the body, also ramped.
Both are ramps, not thresholds — with a cliff, adjacent units of the reported
file landed on opposite sides and swapped 10 dB of EQ mid-file.

**Measured, end to end, on the user's file (production profile):**
2-4 kHz +3.5 dB, 4-8 kHz +8.9 dB, every band below 1 kHz within 0.5 dB, 8-16 kHz
+0.2 dB (the dead region, untouched). -16.0 LUFS, -1.5 dBTP. Guard B PASS:
consonant retention 3.69, spectral hole 0.001, musical noise 0.000.
**On Flute 09, both profiles: no `tonal_restore` action at all**, and the output
band table is identical to the pre-change run to within 0.01 dB in every band.

Also found and fixed on the way:
- Tonal balance is a property of the RECORDING, not of a 20 s block of it. The
  pipeline now measures every unit it will finish, combines by median, and
  applies one filter to all of them. Per unit it drifted 2.8 dB of 3-6 kHz
  between two adjacent blocks of the reported file — an audible pump at every
  boundary.
- `apply_speech_eq` applies each biquad with `filtfilt`, which squares the
  magnitude response: it has always delivered TWICE the dB it was asked for
  (-2.0 requested measures -3.97; -6.0 measures -11.92). Its existing callers
  are calibrated around that and are untouched. The new bank designs each
  section for half its gain and then MEASURES the finished cascade on every
  call, refusing to apply one that exceeds its declared bound.
- Three overlapping sections cannot each act on one band alone, so the gains
  are solved against the analytic response rather than requested naively. A
  variant that was free to pull the bells below flat turned a pure bass cut
  into a bass-and-treble cut — dull and mid-forward, the 3.1.1 failure again —
  so the bells stay lift-only and the leftover spill is bounded by test
  instead: an untargeted band never receives a cut, nor more than half of what
  the band that earned the move received.
- A brilliance lift can create sibilance that was not harsh before it, and
  de-essing was decided on the unlifted signal. Sibilance is now re-detected
  after a lift.

New permanent gate `tests/unit/test_finishing_tonal_restoration.py` (60 tests)
with a committed corpus (`tests/support/tonal_corpus.py`). `test_output/` is
gitignored, so the load-bearing pair is synthetic: `approved_recording_profile`
reproduces the approved recording's measured profile to within 0.2 dB in every
band and must receive 0.0 dB, and `presence_starved_profile` is that same
signal with the reported file's 11 dB deficit above 3 kHz and must be restored.
The real audio is also tested when it is present on the machine.

**What this does not fix, stated plainly.** The reported file is 57 dB down at
4-8 kHz and 74 dB down above 8 kHz. That was never captured and is not
recoverable; the correction refuses to touch it and the file still sounds like
what it is — a dull, distant recording, now intelligible rather than muffled.
The dynamics gate can tell a constant floor from modulated content, but it
cannot tell attenuated speech from signal-following codec noise; the reported
file's upper bands track its body almost exactly (envelope correlation 0.95),
which is what both a lowpassed voice and coder noise look like. And on the
STUDIO profile this file gets no correction at all: DeepFilterNet3 removes 79%
of its consonant band (retention 0.206, spectral hole 0.387), Guard A correctly
reverts the unit — and the pipeline only finishes units that were enhanced, so
finishing is skipped with it. Production is the profile that carries this fix
on this file.

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
