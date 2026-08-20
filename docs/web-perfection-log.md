# Web perfection log

Evidence trail for `docs/web-perfection-goal.md`. Newest iteration last.
Each entry: what was attempted, what was measured, what the gates said.

## Baseline (2026-08-20, commit 086b1c4)
First screen shipped and verified end-to-end in web mode against the real engine on
`test_output/ui-smoke/Flute 09.m4a.mp4`: analyze -> PROCESS -> 5/5 units enhanced,
noise floor -48.5 dB -> -84.7 dB, master + report written, A/B decks time-locked,
zero console errors, all requests to 127.0.0.1. Gates: ruff/format/mypy --strict clean,
397 passed + 41 fuzz, mutation gate 12/12, `pnpm typecheck`/`pnpm build` green.
Nothing in the goal checklist is claimed yet — the baseline is the starting point.

## Iteration 1 — 2026-08-20 — waveform becomes a real instrument (E3, B3, B4, B1)

**E3 windowed peaks.** New `POST /api/peaks` (contract Addendum 1) + `decode_audio_window`
(ffmpeg `-ss` before `-i`, soundfile `start=/stop=` fallback). Two seek traps found and fixed by
testing a window against a full decode: lossy streams need a 0.25 s discarded pre-roll (no MDCT
overlap partner on the first frame after a seek), and an explicit `-ss 0` makes ffmpeg hand back
mp4 encoder-priming samples — both were worth 0.3–0.7 full scale of error. AAC-in-mp4 windows are
now bit-exact against the full decode.
Memory (the normative rule): **5 s window out of a 346 MB / 30 min file = +5.7 MB peak RSS, 172 ms**;
out of a 2.07 GB / 3 h file = +4.8 MB, 812 ms cold / 27 ms warm. Whole 2 GB file as one window went
8.5 GB -> 155 MB via streaming reduction. A bounded probe cache removed a 0.8 s/gesture SHA-256.
Committed proof: `test_a_window_out_of_a_huge_file_costs_a_window` (generates 346 MB, asserts
+5.7 MB < 32 MB, deletes it). Correctness proof: `test_window_buckets_equal_a_full_file_analysis_of_
the_same_span` over 5 windows including one aligned to no bucket boundary, plus
`test_deep_zoom_returns_the_raw_samples` (samples_per_bucket == 1 -> min == max == raw samples).
32 new tests.

**B3 zoom/pan.** Imperative `waveView` controller owns the visible window; the store mirrors it on a
120 ms trailing timer so no zustand write happens per wheel event. Wheel zooms about the cursor,
ctrl+wheel pinch, shift+wheel/ruler-drag pans, double-click or FIT resets. Detail is re-fetched per
view (debounced 120 ms, AbortController, 24-entry LRU) for both decks, with an immediate redraw from
data in hand so it never feels laggy. Ruler is range-aware (100 us -> 6 h ladder, major/minor,
`h:mm:ss` -> `m:ss.dddd`). Overview scrubber with a draggable window rect. Verdict strip is now
view-linked via two CSS custom properties (`--vs`/`--vd`) — a zoom is two setProperty calls, zero
React renders.

**B4 unit inspection + B1 keyboard.** Clicking a verdict segment selects the unit: highlights its
range, seeks the transport, pans it into view keeping the zoom span, and opens a real inspector
(decision pill, channel/range/duration/speech/runtime/strength meter, finishing preset + action
chips, reason, and Guard A/Guard B score tables with per-score bars). Designed empty state with run
summary tiles. Full keyboard map in one hook (Space/A/B/arrows/Shift+arrows/P/Esc/[/]/?) with a
bevelled shortcut overlay (focus trap, Esc, restore focus).

**My own verification (orchestrator, real engine + real browser, 1440x900):**
- Zoom: 6 wheel-ups at canvas centre took 0.0–94.6 s -> 46.014–48.599 s, pivot preserved; ruler
  relabelled to `0:46.x`; overview rect tracked; `POST /api/peaks` 200 fired on each view change.
- Real job through the UI: 5/5 units enhanced, -24.9 -> -21.7 LUFS, noise floor -48.5 -> -84.7 dB.
- Click verdict segment 3 -> `UNIT 02 · 3/5`, ENHANCED, 00:40.251 -> 00:59.550, runtime 872 ms,
  strength 1.00, preset `gentle`, Guard A PASS (env corr 0.969, cons keep 0.678, JS div 0.242),
  Guard B PASS (0.996 / 1.518 / 0.006); both decks seeked to 40.251.
- Keyboard (with proper flush delays): `]` 4->5, `[` 5->4, arrows +5/-5 and Shift +/-1 s exact,
  Cmd/Ctrl/Alt combos fully inert, `?` opens the overlay, Esc closes it, and while the overlay is up
  it correctly owns the keyboard (other bindings inert — this looked like a bug in a first probe that
  read the DOM synchronously before React flushed; a delayed re-test showed correct behaviour).
- Zero console errors. All requests to 127.0.0.1.

**Gates:** ruff clean, ruff format 158 files, `mypy --strict` 78 files clean, **463 passed** (was 397),
fuzz 41 passed, `pnpm typecheck` + `pnpm build` green (396.6 kB / 125.5 kB gz). Built worker still a
classic script (0 import/export) so it will load from `file://` in Resolve.

**Open, carried forward:** (a) `/api/analyze` still decodes whole files, so E1 is only half done;
(b) `<audio>` range requests log `net::ERR_ABORTED` on deck swap — a normal media-element abort, but
C4's "zero failed requests" needs it addressed or explicitly struck; (c) `/api/analyze` and
`/api/peaks` differ by up to 1.5 ms on lossy containers (container vs decoder timeline) — peaks is
the correct one for the playhead; (d) no vitest yet, so D4's UI-unit-test clause is unmet;
(e) `/api/health` polls very frequently — worth a look during the perf pass.

## Iteration 2 — 2026-08-20 — visual grade (A1, A2, A4, A6, A7, D3 ticked; A3, A5 deliberately not)

Five agents: waveform renderer, spectrum renderer, chrome/typography, micro-interaction, then an
independent design critique that judged the result against Waves / FabFilter Pro-Q 4 / Gullfoss /
RX 11 and found **21 defects, fixed 21**.

**A2 waveform (ticked).** WebGL2 renderer rewritten around *analytic coverage in the fragment
shader* (`antialias:false` — AA is computed, not sampled), 7 instanced programs, zero per-frame
allocations. Vertical gradient fill core->edge, RMS body as a denser core, half-res two-pass bloom
fed only by the focused deck, playhead with a 56 px leading ramp + gaussian halo + pixel-snapped
1 px core. The non-focused deck now draws as a contour *on top* of the filled focused deck, so the
amber original no longer vanishes under cyan. **Headline: at 1:1 zoom it switches from min/max bars
to a continuous anti-aliased sample trace** — verified by me at 68.08 s, both decks overlaid.
Palette is read from CSS custom properties through a probe element and re-posted on theme change.
Measured: 120 wheel events = 0.134 ms each on the main thread, rAF median 8.3 ms / p95 10.4 ms.

**A1 panel depth + A4 typography (ticked).** One lighting model (top-lit): raised panels get inner
top highlight + hairline + outer shadow, inset displays get the inverse plus vignette; elevation and
radii are now tokens. My audit: **2 font stacks total, 0 elements falling back to a browser default,
14 numeric readouts and 0 without tabular figures.** Contrast audit over 89 leaf text nodes: 1 hit,
investigated and confirmed a false positive (active A/B button is filled by a pseudo-element, so my
walker read the panel behind it; it is dark-on-cyan and clearly legible). The critique's own audit
found 15 real failures before its fix (worst 2.66:1) and 0 after — the `--fg-3`/`--fg-4` ramp was
recut for it.

**A6 empty states + A7 A/B/verdict (ticked).** Verified myself: the pre-file screen is a designed
product (drop well, placeholder tiles with em-dashes, spectrum "NO SIGNAL" grid); drag-over toggles
`.dropzone.over` and reverts cleanly on leave; the verdict tooltip carries real data
(`Unit 0 · ch 0 / ENHANCED / 00:00.000 -> 00:20.569 / GUARDS A PASS B PASS / STRENGTH 1.00`).
The verdict strip was the worst thing on screen (full-saturation cyan slabs for a *status* band, and
selection only 1.2x brighter than neighbours) — recut as a recessed hue-over-black recipe with the
selected segment the single lit bar (~3x luminance separation).

**D3 reduced motion (ticked).** Global `animation/transition: none !important` under the media
query, plus static restatement of the states that were carried by animation alone (busy LED, error
LED, analyzing scan, drop glyph, skeleton, running plate) so nothing becomes unreadable.

**Not ticked, on the critique's own honest assessment:**
- **A3 spectrum** — the frequency ladder, Hz placement, difference shading and legend were all
  fixed, but the fill is still "a faint blue-grey wash" with a halo rather than real bloom; beside
  Pro-Q 4 it reads thin. Carried to iteration 3.
- **A5 micro-interaction** — hover/active/focus states, LED cadences and press animation are in, but
  the PROCESS plate's interior is mostly empty and its ring is a flat 2D stroke rather than a
  modelled meter: "the one control that would not pass as a Waves component". Carried to iteration 3.

Other critique fixes worth recording: `WebGL2 · worker` debug text removed from the panel head;
spectrum axis had a hole (10k lost by 1.2 px) now an all-or-decades rule; `Hz` moved off the dB axis
to the frequency baseline; metric sub-row `-24.9 -> +3.2` (read as before->after) changed to a delta
operator; transport's three different control heights unified at 28 px; the timecode's stripped
leading space fixed; playhead clock and deck legend given smoked-glass plates instead of floating on
the waveform; right column widened to 400 px at >=1600 so the full 1-2-5 ladder fits; 960x640
inspector scrollbar restored (`scrollbar-width: thin` was making Chromium ignore the styled bar).

**Gates:** `pnpm typecheck` + `pnpm build` green (413.89 kB / 131.11 kB gz JS, 58.94 kB / 11.88 kB gz
CSS). Built worker still a classic script (0 import/export). No engine code touched, so the Python
gates are unchanged from iteration 1. Zero console errors across every flow I exercised.
Iteration-1 features re-verified on this build: zoom 14 steps to 1:1, ruler drag, FIT, overview
click, verdict click -> unit selection, `[`/`]`, arrows, `?` overlay.

## Iteration 3 — 2026-08-20 — A3, A5, B2, B5, B6, B7, B8, C4, E1, E2 (20/27)

**E1 streaming analyze — the biggest number in the project so far.**
`POST /api/analyze` no longer decodes the file. Peak RSS on a 3 h / 2073.6 MB input:
**12,756.8 MB -> 222.3 MB (57x)**, wall 36.1 s -> 32.9 s, and every returned value unchanged.
Peak RSS is now flat in file length (115 MB file -> +114.8 MB; 346 MB file -> +116.9 MB).
New additive `iter_decode_audio` streams one ffmpeg pass (stderr to a temp file — a pipe nobody
drains deadlocks a chatty decoder) and is bit-identical to `decode_audio`. Four accumulators run
over that single pass, each **proved against the old whole-file implementation** (kept in the test
as the oracle) over 9 fixtures x 4 chunk sizes: LTAS **1.4e-14 dB**, BS.1770 integrated loudness
**1.2e-7 LU** (contract was 0.01), true peak and sample peak **exactly 0**. The loudness residue is
float64-vs-float32 block sums, not a chunking artefact — identical from 977 to 1 Mi chunk sizes.
Gate-edge cases replicated: <400 ms ungated branch, near-silence, all-blocks-below-gate, >5 channels.

**E2 streaming upload.** 1.07 GB uploaded over real HTTP: **133.8 MB peak RSS vs 127.8 MB idle**.
Chunked write, configurable cap, 413 on both the Content-Length and chunked paths.

**A3 spectrum (was "the weakest panel").** Rebuilt on three surfaces: a cached opaque base (grid,
labels, fills, bloom, curve cores) blitted once per frame, a scratch layer that punches the cleaned
deck's area out of the original's with `destination-out` so the overlap is never two washes stacked
into mud, and a half-res bloom (4 widening additive passes through a blur filter). The surviving
amber is exactly `{original > cleaned}` — the REMOVED band is *value-driven, not name-driven* — and
carries a device-resolution 45 degree hatch with a matching key swatch. Two legends merged into one
key row; `LTAS` moved into the panel title. Live overlay given real ballistics (attack 22->12 ms,
release 340->170 ms, peak-hold 0.35 s then full scale in 1.5 s). Measured **0.038 ms/frame**, 100 %
of available animation frames. The latent trap is fixed: sizing now uses offsetWidth/offsetHeight, so
an ancestor `transform: scale(2.5)` leaves the backing store at 668x534 instead of inflating it 6.25x.

**A5 process plate (was "the one control that would not pass as a Waves component").** The ring is
now a modelled meter: opaque channel bed with a top-lit groove, inner/outer rim hairlines, a 48-tick
bezel scale, an arc with a gradient along its sweep and a glow riding the arc head. The interior is a
sunken readout — a 7-lamp stage rail plus three cells that change per phase (READY/LENGTH/FORMAT
idle, STAGE/UNIT/ELAPSED running, RESULT/UNITS/TOOK done, RESULT/REASON/TOOK failed). One
`@property`-registered colour cross-fades amber -> cyan -> green/red in a single interpolation. Bugs
caught on the way: the plate claimed "Armed" for a clip whose analysis had failed, and the elapsed
clock carried the previous run's start into a second run.

**B2/B5/B7 (flow).** Upload has real XHR progress + cancel (measured: 880 MB drop tracked to 100 %;
cancel at +300 ms on 600 MB left no partial file). Designed rejection sheet, multi-file drop takes
the first audio file and says so, folder drop has its own wording. Session history keeps 8 runs with
both analyses cached. Report access: master WAV / JSON / txt / copy summary.

**B6/B8 (resilience).** Offline is a designed row, not a dead LED; health poll backs off
400->5000 ms while offline; a returning engine reconciles via `GET /api/jobs/{id}` and treats a 404
from a live engine as "the run died with the process" (`ENGINE_RESTARTED`), which is the case that
would otherwise hang forever. Controls switched from `disabled` to `aria-disabled` + `title`,
because a disabled element receives no pointer events and so never shows the explanation.

**My own verification (real engine, real browser):**
- Two real jobs (studio 5/5, production 0/6). Switching between them in the run list: **`/api/analyze`
  call count stayed at 3 across the switch** — a pure state restore — while the header retargeted
  from `UNITS 0/6 LUFS +2.7` to `UNITS 5/5 LUFS +3.2` and the ON SCREEN badge moved.
- All three artefacts served correctly: `.wav` 200 audio/wav 13,624,364 B, `.hawavoclean.json` 200
  application/json 11,907 B, `.hawavoclean.txt` 200 text/plain 2,130 B.
- Copy summary payload: `Flute 09.m4a · studio · 5/5 units enhanced · -24.9 -> -21.7 LUFS · noise
  floor -48.5 -> -84.7 dB`. Drop rejection: `CANNOT OPEN meeting notes.txt / text/plain is not an
  audio or video format this tool opens. Accepted: wav · aiff · mp3 · flac · m4a · mp4 · mov.`
- **Killed the engine with the app fully loaded**: banner appeared with a live retry countdown and
  the sentence "Nothing on screen was lost — Flute 09.m4a.mp4, its report, 2 runs are still loaded";
  waveform, spectrum, metrics, history and zoom all preserved; downloads greyed with explanations.
  Restarted it: banner cleared to "Engine back · v3.2.0", controls re-armed, no reload needed.
- C4 on a clean tab: **17 requests, all 127.0.0.1, statuses {200, 206, 202}, zero >=400, zero
  non-loopback.** (Honest caveat: deliberately killing the engine produces browser-level
  ERR_CONNECTION_REFUSED lines in the console that JavaScript cannot suppress — that is Chromium
  logging a refused socket, and the UI handles it as designed. One `blob:` abort also remains on
  large in-memory decks; it is an in-process buffer read with no socket and does not appear in
  resource timing.)
- B8 at 960x640, 1440x900 and 2560x1440: no page overflow, no real clipping (I chased an apparent
  cut on the A/B "CLEANED" label — DOM math showed 50 px of text in a 131 px button with nothing
  overlapping, so it was screenshot downscaling, not a bug). At 2560 the layout reflows to three
  columns rather than stretching one panel.

**Orchestrator decision:** the resilience agent had raised the *healthy* health-poll cadence to 5 s
(the pre-existing value was 10 s). Since `probeSoon()` now reacts instantly to any failed call, I set
it back to **10 s** — quieter log, no loss of responsiveness.

**Gates:** ruff clean, format 160 files, `mypy --strict` clean, **492 passed** (was 463), fuzz 41
passed, `pnpm typecheck` + `pnpm build` green (321.29 kB / 100.03 kB gz JS — *smaller* than iteration
2 despite everything added — 79.76 kB / 15.27 kB gz CSS). Worker still a classic script.

**Known trade-off recorded:** on lossy containers the analyze overview bucket grid is now laid on the
container timeline (matching `/api/peaks`, the playhead and `<audio>`) instead of the decoder
timeline. On Flute 09 that shifts buckets by median 0.029 dB / p95 0.29 dB / worst 4.4 dB in one
quiet bucket. Spectrum, loudness and true peak are unaffected (they see every decoded sample). The
only exact alternative is spilling the mono signal to disk for a second pass (~2 GB on a 3 h file).

## Iteration 4 — 2026-08-20 — C1, C2, C3, C5, D4 (25/27; D1/D2 in progress)

**Harness honesty, recorded because it changes how to read every fps claim in this log.**
The Claude Browser pane runs its tab with `visibilityState: "hidden"` and `innerWidth: 0`. rAF never
fires there — and because Chrome gates a DedicatedWorker's rAF on the same document, the waveform
worker is throttled too (3 renders in 25 s). `tabs_select` does not fix it. So C1 was measured in a
separate headful Chrome (own profile, `--expose-gc`, killed afterwards) reporting
`visibilityState: "visible"` and real vsync: **median rAF interval 8.3 ms** — this Mac is 120 Hz
ProMotion, so "60 fps" was actually measured against an **8.3 ms** budget, not 16.7 ms.

**C1 — 90-minute file (1.037 GB, generated then deleted).**
| phase | events | main-thread ms/event (med/p95/max) | rAF gap (med/p95/max) |
|---|---|---|---|
| zoom burst 2/frame, in->out->in through 1:1 | 360 | 0.2 / 0.4 / 0.6 | 8.3 / 10.0 / 10.3 |
| zoom 8/frame (pathological ~960 Hz) | 960 | 0.0 / 0.1 / 0.4 | 8.4 / 10.3 / 10.4 |
| ruler pan drag | 240 | 0.2 / 0.4 / 0.4 | 8.3 / 9.6 / 10.3 |
| seek drag (transport + playhead per event) | 240 | 0.2 / 0.4 / 0.5 | 8.3 / 9.9 / 10.3 |

**Long tasks (>50 ms) across all 1800 events: 0. Zero dropped frames** (max gap 10.4 ms vs 8.3 ms
vsync). Worker's own frame time: mean **0.107 ms**, p95 0.20 ms. The goal's actual requirement —
"worker never blocks main thread >16 ms" — is proved structurally: the renderer is a DedicatedWorker
on an OffscreenCanvas, so the main thread never enters it; its only main-thread cost is one
structured-clone postMessage per event, p95 **0.4 ms**. Request discipline: a 360-event zoom burst
produced **2** `/api/peaks` calls (120 ms debounce + AbortController + 24-entry LRU).

*Real bug found and fixed by this measurement:* `/api/analyze` always asked for 1200 buckets while
`WaveformDisplay` compared against a hard-coded `BASE_BUCKETS = 1200`, so on any display wider than
1200 device columns the fit view instantly failed its detail test and issued a **whole-file**
`/api/peaks` — a second full decode on every clip load, **2500 ms on the 90-minute file**. The view
now records the display's real bucket demand. After: `/api/peaks` at fit = **0 calls**, and the fit
view is sharper 2.5 s sooner. Asking for more buckets is free (1200 -> 2400 buckets on a 1 GB file:
15.98 s vs 16.00 s — decoding is the entire cost).

**C2 — paint and bundle.** Cold load: first-paint **76 ms**, FCP/LCP **168 ms**, DCL 53.6 ms, **0
long tasks**. `/api/health` starts at 82.5 ms, i.e. *after* first paint — the chassis does not wait
on the engine. Bundle measured with gzip -9 on the emitted files: JS 102.6 kB gz + CSS 15.3 kB gz +
worker 7.9 kB gz + html 0.33 kB = **126.1 kB gz against a 500 kB budget**.

**C3 — 10 analyze+process cycles**, forced GC x5 before each sample. Heap **11,859 -> 13,973 KB
= +2.06 MB** against a 10 MB budget; per-cycle deltas decay 700 -> 26 KB (the 8-entry history ring
filling, then ~30 KB/cycle marginal — flat, not linear). Live-object census constant at every
sample: `<audio>` 2, `<canvas>` 4, `Worker` 0 created/0 live, `EventSource` 10 created / **0 live**,
`blob:` URLs 20 created / **exactly 2 live**, DOM nodes plateau at 478 from cycle 8. **No leaks.**
Correction to the goal's wording: audio elements and workers are not created per file — `DualPlayer`
is a singleton that swaps `src` and revokes the previous blob URL, and the waveform `Worker` is
created once at mount and `terminate()`d on unmount. Better than per-file disposal, and verified at
zero growth over 10 switches.

**C5 — 20 adversarial fixtures** driven through the real UI. Four real failures found and fixed:
- **corrupt container** showed a 358-character raw exception (`ffprobe failed to probe /Users/.../
  work/uploads/<uuid>/random.wav: Command '['/opt/homebrew/bin/ffprobe', ...]' returned non-zero
  exit status 1.`), 3013 px of text ellipsised mid-path. Now: *"random.wav" is not readable audio —
  the container is empty, truncated or corrupt. Re-export it, or try the original file.*
- **truncated m4a** — same class, same fix.
- **video with no audio track** — leaked the work-dir path; now *"noaudio.mp4" carries no audio track
  — it is a video-only (or data-only) container. There is nothing here to clean.*
- **192 kHz file** — analysed fine then failed at PROCESS with a bare `INVALID_USER_INPUT:`. Now the
  RATE cell warns pre-flight and the refusal reads *"hi192k.wav" is 192 kHz. This tool works up to
  48 kHz — resample it down and load it again.*
Passing untouched: 0-byte, 1-sample, 50 ms, `it's a "take" — 01.wav`, `テスト音声.wav`,
`🎙️ take.wav`, `-rf take.wav`, embedded newline, 196-character name (clamped to 260 px, no page
overflow at 1440 or 960), `q&a=1?x#hash%20 take.wav` (all three artefacts HEAD 200), and a file
deleted between analyze and process.

**D4 — vitest.** 222 tests in 9 files, 556 ms: store transitions, SSE client (full back-off ladder
measured tick-by-tick, `onGone` 404 semantics), API client (token header vs query, error shapes,
abort, upload progress against a fake XHR), the keyboard map (every binding, modifier and
editable-target inertness, Esc priority chain), selection/stepping, and the pure render logic
(`ticks`, `viewWindow` pivot preservation to 1e-9, `peaksCache` LRU + capability latch). The suite
was mutation-checked: breaking the zoom pivot, the tick minPx, the modifier guard, the SSE back-off
formula, the history slice, the token header name, the key rounding and the `ENGINE_RESTARTED` code
each produced failures in exactly the owning file. One production change to enable it: the keyboard
map was lifted verbatim out of `App.tsx` into `state/keymap.ts` (a pure move).

**Four items the agents flagged but could not own — fixed by the orchestrator:**
1. **Esc did not cancel an analysis.** Added the missing rung to the Esc ladder in `state/keymap.ts`
   (upload -> *analysis* -> job -> rejection -> selection), updated the test mock, and added a test
   pinning the ordering. 222 tests green.
2. **`stepUnit` wrapped asymmetrically.** With nothing selected and the playhead outside every unit,
   `]` past the last unit came round to the **first** and `[` before the first took the **last** —
   silently throwing the user back to the top of a take. Both now clamp; the two tests that
   documented the old behaviour were rewritten to assert clamping.
3. **`formatTimeShort` printed `1:60`** for 119.6 s (it rounded the seconds remainder without the
   minute renormalisation). Now rounds to whole seconds first, then splits.
4. **`clearPeaksCache()` did not reset the capability latch**, so once an engine without
   `/api/peaks` had been seen the route was never probed again for the life of the page — including
   after pointing at a different, capable engine. Now re-arms.

**Gates:** ruff clean, format 160 files, `mypy --strict` clean, **492 passed**, fuzz **41 passed**,
`pnpm typecheck` + `pnpm build` green, `pnpm test:run` **222 passed**, worker grep **0**.

**Still open:** D1/D2 — the accessibility agent died mid-run (connection lost) and its work is being
redone. Also recorded, not fixed: at 1:1 zoom on a 90-minute file there is a ~150 ms transient during
a pan where the display draws from an interpolated base band rather than real samples (frame rate
unaffected; a fix was attempted, could not be shown to be an improvement, and was reverted rather
than churn the ticked renderer on a hypothesis).

## Iteration 5 — 2026-08-20 — D1 + D2 (27/27)

**Harness.** The Claude Browser pane cannot do this box. Its tab runs unfocused, so
`Input.dispatchKeyEvent` never reaches the renderer's focus machinery: 40 dispatched `Tab`
presses left `document.activeElement` on `<body>`, and `?` did not open the overlay either. Every
keyboard result below therefore comes from a real headful Chrome driven over CDP from a small
script (`launch --remote-debugging-port`, own profile, killed afterwards) — the same escape hatch
iteration 4 used for rAF.

**Inherited state.** The tree already carried a substantial, unfinished D1 pass from the agent that
died. It was audited rather than reverted: most of it is good and is now verified. One thing it had
broken is recorded below.

### The bug the accessibility pass itself introduced

`.sr-only` and `.panel > *` have **the same specificity** (0,1,0), and `.panel > * { position:
relative }` is written later in `app.css`. So inside a panel the visually-hidden recipe lost, and
the `<p class="sr-only" id="wave-view-help">` that D1 had added to `WaveformDisplay` stopped being
out of flow. As a real 1px grid item it added a **third row** to `.wavepanel`'s two-row grid
(measured: `grid-template-rows: 30px 0.148438px 177.844px`), which pushed the panel head out of its
own row and down onto the ruler. At 960×640 the head's whole contents — the panel title, the clip
name, the `0.0 s – 94.6 s of 94.6 s` readout, the magnification and the **FIT button, a focus stop**
— were painted *underneath* the opaque ruler canvas. `elementsFromPoint` on the title returned
`CANVAS.ruler-canvas` first, and a differential screenshot proved the text contributed **zero**
pixels to the shipped frame. Fixed with `.panel > .sr-only { position: absolute }`; the panel is
back to `30px 177.992px` and the head is back at the top of the panel (screenshot re-taken).
This is also the reason B8 and D2 had both been "verified" over it: a walker that reads computed
styles cannot see that an element is buried under a canvas.

### D1 — keyboard operability

**Focus path, real Tab dispatch, loaded+done state, identical at 1440×900 and 960×640** — 18 stops,
cycling cleanly with no trap:

1 Open file… · 2 the drop well (the file input fills it) · 3 FIT · 4 waveform display
(`role=group`) · 5 overview (`role=scrollbar`) · 6 Previous unit · 7 Next unit · 8 ? KEYS ·
9 unit detail (`role=group`, the scrolling region) · 10 the run row · 11 STUDIO (the profile
radiogroup's single stop) · 12 PROCESS AGAIN · 13 Play · 14 CLEANED (the A/B radiogroup's single
stop) · 15 Master WAV · 16 JSON report · 17 Summary .txt · 18 Copy summary → back to 1.

With a unit selected it is 19 stops (`Clear selection` arms at position 8). While the engine is
offline it is 18 with `Retry now` first and the artefact buttons still focusable, carrying their own
"unavailable while the engine is offline" wording. Shift+Tab walks the exact mirror (checked
stop-by-stop; the only diff in the automated compare was the history row's own label changing
between the two passes).

Deliberately **not** stops, each with its reason in the source: the verdict segments (`tabindex=-1`
— a report can carry hundreds of units and they would bury the rest of the screen; they keep
`button` semantics, a full name and `aria-pressed`, and `[`/`]` is a better sequential route because
it also seeks and pans), and the non-checked radio in each segmented control (WAI-ARIA roving
tabindex).

**Focus rings.** One recipe, `outline: 2px solid var(--ix-focus)` with a token offset, drawn only on
`:focus-visible`. Proved on pixels, not on stylesheets: freeze animations, screenshot with nothing
focused, Tab to each stop, screenshot again, diff. Every one of the 18 stops matched
`:focus-visible`, changed pixels, and the **worst indicator contrast across the whole path is
10.19:1** (WCAG 2.4.11 asks 3:1). Two were fixed to get there:

- the lit A/B segment measured **3.72:1** — a cyan ring on a cyan-filled key. It now takes a dark
  ring (`rgba(4,20,28,.95)`), the move a real console makes on an illuminated switch: **15.02:1**.
- the drop well's whole indicator was a 1px border-colour change (the file input that fills it is
  `opacity: 0`, so its own outline is invisible). It now wears the same 2px ring inside its border:
  8.31 → **12.11:1**.

**The shortcut dialog still traps deliberately and restores.** Opened with `?` from `Master WAV`:
six Tabs and three Shift+Tabs all stayed inside the dialog; Esc closed it; focus went back to
`Master WAV`. One real defect found here by axe and fixed: at 960×640 the shortcut list overflows
and scrolls, and a scrolling region with no focusable content cannot be scrolled from the keyboard
(`scrollable-region-focusable`, serious). It is now a named tab stop, so Tab alternates between the
close key and the list.

**Live regions — what a screen reader is actually told.** Measured with a MutationObserver that
computes accessible text (aria-hidden subtrees excluded) and only records a region when its text
changes, across a real 8-second job:

```
   5 ms [polite] Processing started
8070 ms [polite] Processing finished. 5 of 5 units enhanced.
```

Two announcements for the whole run — no per-tick spam, although the header readout, the plate
face, the footer and the source strip all re-render on every SSE tick. One region was removed to
get there: the header's engine lamp was `aria-live="polite"` and was narrating ENGINE BUSY →
ENGINE READY, i.e. the same two events in words that say less. Killing the engine with the app
loaded produces **exactly one** assertive announcement — "Engine offline. Nothing on screen was
lost, and the app is reconnecting on its own." — while the retry countdown beside it ticks about
24 times in the same six seconds without being announced once. `role="alert"` is the right level
and this is the justification: engine loss changes what every control on the screen can do, so it
earns the interruption, and it only earns it once; the countdown and the outage clock stay outside
any live region, readable when you go and look.

**ARIA, read off Chrome's own accessibility tree** (`Accessibility.getFullAXTree`), not off the
markup: `heading "HAWAVOCLEAN v3.2" level=1`; landmarks `banner / region "Source clip" / region
"Waveform" / region "Unit inspector" / region "Session runs" / complementary "Analysis and
controls" / region "Spectrum" / region "Processing controls" / contentinfo`; `group "Waveform,
showing 0.0 s to 94.6 s of 94.6 seconds" describedby=wave-view-help`; `scrollbar "Visible waveform
window" orientation=horizontal`; five `button "Unit 0, channel 0, ENHANCED, 00:00.000 to 00:20.569,
guard A PASS, guard B PASS, strength 1.00, …" pressed=false`; `button "PROCESS AGAIN"
describedby=process-readout`; `radiogroup "Profile"` / `radiogroup "A/B source"` with
`radio … checked=true|false`; `list "1 run, newest first"` → `button "Flute 09.m4a.mp4, currently on
screen, studio, 5/5 units enhanced, LUFS change +3.2, took 8.0 s, at 08:23:47"`;
`group "Integrated: -21.7 LUFS, from -24.9 LUFS, change +3.2 LUFS"` (and the other two tiles — named
as the sentence they draw, interior hidden, so they are not read as a pile of loose numbers);
`image "Long-term average spectrum: original and cleaned, with the removed energy between them"`.
The guard score tables are real table semantics; the label cell is now `rowheader`, not another
`cell`.

**Live values really update.** `aria-valuenow` / `aria-valuetext` on the scrubber and the waveform
region's own name are written imperatively (a zoom must not cost a React render), so they were
tested by driving them: `0.000` / `"0.0 s to 94.6 s"` → after two `+` zooms `28.827` /
`"28.83 s to 65.78 s"` → after PageDown `57.654` / `"57.65 s to 94.61 s"`, with the region's label
tracking every step. Arrow keys on the scrubber correctly do nothing at fit (there is nothing to
scroll when the whole clip is in view).

**The tooltip is not the only route to a unit's data.** Keyboard only, no pointer: `]` `]` `]` moved
the selection 0 → 1 → 2 with `aria-pressed` following it, `[` went back to 1, and the inspector
carried strictly more than the hover card does — `UNIT 01 · 2/5 ENHANCED · CHANNEL 0 · RANGE
00:20.569 → 00:40.251 · DURATION 19.68 s · SPEECH YES · RUNTIME 940 ms · STRENGTH 1.00 · FINISHING
gentle, dc_subsonic_removed(cutoff=75.0Hz), dialogue_leveler(gr_max=2.0dB) · REASON Passed Guard A
with strength s=1.00` — plus both guard score tables below the fold of that region.

**axe-core 4.13.0** (dev dependency, read from `node_modules` and injected as source at run time —
never imported by the app, `grep -c axe dist/assets/*.js` = 0), run over eight states:
idle / loaded+done / done with the inspector scrolled to its guard tables / shortcut overlay open,
each at 1440×900 and 960×640.

| | violations before | violations after |
|---|---|---|
| all eight states | **1** — `scrollable-region-focusable`, *serious*, `.sc-body` (keys@960×640) | **0** |

`color-contrast` shows as *incomplete* (1 node) in every state, in both runs: that is axe declining
to judge the wordmark, whose glyphs are a gradient (`background-clip: text`). It is measured by hand
below at 10.7:1.

### D2 — contrast on the pixels actually shipped

**Method, stated so the numbers are defensible.** Two independent measurements per text node,
run over every text node in the document at each state and size.

*(a) CSS layer compositing.* Text colour is the computed `color`, composited over the backdrop if it
carries alpha. The backdrop is built by walking from `html` down to the element and stacking every
paint layer in paint order: `background-color`; every `background-image` layer that is a gradient —
parsed and **evaluated at this text's own position inside that ancestor's box** by projecting the
text-box point onto the gradient line, finding the bracketing stops and interpolating in
premultiplied sRGB (linear, radial and repeating-linear; `url()` layers are counted as *unresolved*
rather than ignored — there were none); `::before`/`::after` fills that cover the box; and
**covering siblings that paint below the text**, ordered by z-index, then positioned-over-static,
then document order. Five points per text run (four corners + centre), worst kept. Nodes clipped
out of a scrolling ancestor, or behind a modal scrim, are counted and excluded — they are not
shipped pixels.

*(b) Differential pixel truth.* Capture the frame as shipped (A), then capture it again with every
glyph's fill emptied via `-webkit-text-fill-color: transparent` (B) — which blanks letters without
disturbing anything that resolves `currentColor` — with animations pinned to their end frame so A
and B differ *only* by glyphs. A pixel that changed is a pixel a glyph painted; read those same
pixels **from B** and you have the true backdrop behind that element's own letters, including canvas
pixels, inset shadows, blurs and gradients, and with no bleed from a bright neighbour that merely
shares the box. Ratio = the element's specified colour against the worst such pixel.

Threshold per node: **4.5:1**, and 3:1 only where WCAG says large — ≥24px, or ≥18.66px at
weight ≥700. Nothing on this screen qualified as large: every measured node took the 4.5:1 rule.

**The two "failures" the earlier audits reported were walker artefacts, and both are now handled
rather than excused.** `.wordmark .name` at 1.08:1 — it uses `background-clip: text` with a
transparent fill, so the gradient *is* the glyphs; measured properly it is 10.7:1. `.seg.ab
button.on` at 1.10:1 — the lit A/B label is dark-on-amber/cyan, but the fill comes from a *sibling*
`.thumb`, which no ancestor-chain walker can see; with covering siblings composited it is 7.9:1, and
the differential pixel pass independently reads 8.11:1.
That is the same false positive iteration 2 recorded and could not resolve.

**Real failures, before → after** (differential pixel truth; the CSS method agreed within 0.4):

| node | before | after | rule |
|---|---|---|---|
| `.pc-sub` "DFN3 restoration core" @1440×900 | **4.49:1** | 4.86:1 | 4.5 |
| `.pc-sub` "DFN3 restoration core" @960×640 | **4.42:1** | 4.78:1 | 4.5 |

It is still the worst text on the screen after the fix, which is the right shape for a bottom-of-ramp
token: the whole-screen worst case is now 4.78:1 (done @960×640), then `1.0×` at 4.97:1 and the clip
`Duration`/`Rate`/`Channels` keys at 5.12:1.

One text on the whole screen, and the fix is a token, not a font size: `--fg-4` #808a99 → **#86909f**.
The old value was cut against the flat panel tokens (5.1:1 on `--panel`), but the raised controls
panel paints its interior around rgb(30 36 40) on the real pixels, and there the bottom step of the
ramp fell 0.08 short. The new value measures 5.6:1 on `--panel`, 5.2:1 on `--panel-2` and 4.8:1 on
the controls panel, and the four greys are still four visibly separate greys.

**Final numbers — 0 failures in every state, by both methods:**

| state | nodes measured | CSS-composited fails | pixel fails | notes |
|---|---|---|---|---|
| idle @1440×900 | 87 | 0 | 0 | 3 over a canvas |
| idle @960×640 | 82 | 0 | 0 | 3 over a canvas |
| done @1440×900 | 139 | 0 | 0 | 12 clipped out of the inspector's scroll |
| done @960×640 | 114 | 0 | 0 | 10 clipped |
| done, inspector scrolled @1440×900 | 162 | 0 | 0 | the guard tables |
| done, inspector scrolled @960×640 | 115 | 0 | 0 | |
| shortcut overlay @1440×900 | 115 | 0 | 0 | 187 nodes behind the modal, excluded |
| shortcut overlay @960×640 | 65 | 0 | 0 | 176 behind the modal |
| engine offline @1440×900 | 131 | 0 | 0 | the banner, its clock and its refusals |

The 122 text nodes the previous audit had to skip because they sit on gradients are all measured
here; **0 unresolved layers** in every run. The one node the pixel method declines is the wordmark
(its glyphs are its background, so blanking them changes nothing to diff) — the CSS method covers
it at 10.7:1. Elements over a `<canvas>` are the case that only the pixel method can judge, and
they pass on it: e.g. the "Overview" hint at 5.95:1 read off the drawn overview canvas, and the whole
wave panel head, which is only measurable at all since the grid bug above was fixed (its title reads
7.21:1 on pixels at 960×640, against 7.14:1 predicted from CSS).

**Also fixed while in here:** the run list said "1 runs".

**Gates:** `pnpm typecheck` clean, `pnpm build` green (334.57 kB / 104.22 kB gz JS, 82.32 kB /
15.68 kB gz CSS), `pnpm test:run` **222 passed in 9 files**, worker still a classic script
(`grep -cE '\bimport\b|\bexport\b'` = **0**). No Python touched, so the engine gates are unchanged
from iteration 4. Re-verified on the final build: zoom/pan, the view-linked verdict strip, the unit
inspector, the keyboard map and the `?` overlay, the analyser, the modelled process meter, and the
offline banner. History restore re-measured rather than eyeballed: two real runs (studio 5/5 and
production 0/6), then switching between them twice left the `/api/analyze` call count at **2** while
the header retargeted `UNITS 5/5 LUFS +3.2` ↔ `UNITS 0/6 LUFS +2.7` and the verdict strip repopulated
to 6 segments. Zero console errors through all of it.

**Honest gaps:** (a) the *running* phase was not put through the contrast walker as its own state —
the job is ~8 s and one audit pass takes longer than that; its text (`STAGE Enhancing`, `UNIT 4 / 5`,
`ELAPSED`) uses tokens measured in the other states, and `aria-busy` and the readout were verified
live instead. (b) 13 shortcut-list rows below the fold at 960×640 are counted as clipped rather than
measured; they are `.sc-row`s identical in class to the visible ones, which pass. (c) The modal marks
itself with `aria-modal="true"` and traps Tab, but does not set `inert` on the rest of the app, so a
screen reader in browse mode can still walk the background. (d) No real screen reader was run —
VoiceOver cannot be driven from here; the announcements above are what the DOM would hand one, not
a recording of one speaking.

## Iteration 5 — 2026-08-20 — D1, D2 (27/27)

The first accessibility agent lost its connection mid-run and left a large partial pass in the tree.
That partial work was **committed in `dedd59e` and pushed** before anyone noticed what it contained.
The retry audited it rather than reverting it — and found a real, shipped regression:

**The accessibility pass had broken the layout.** `.sr-only` and `.panel > *` have identical
specificity (0,1,0) and `.panel > * { position: relative }` is written later in app.css, so inside a
panel the visually-hidden recipe lost. The `<p class="sr-only" id="wave-view-help">` added to
WaveformDisplay stopped being out of flow, became a real 1 px grid item, and added a third row to
`.wavepanel`'s two-row grid (measured `grid-template-rows: 30px 0.148438px 177.844px`). That pushed
the panel head out of its row and **under the opaque ruler canvas**: at 960x640 the panel title, the
clip name, the `0.0 s – 94.6 s of 94.6 s` readout, the magnification and the FIT button (a focus
stop) painted behind the ruler — `elementsFromPoint` returned `CANVAS.ruler-canvas` first and a
differential screenshot proved the text contributed zero pixels. Fixed with
`.panel > .sr-only { position: absolute }`.
**Orchestrator re-verified after the fix:** `grid-template-rows: 30px 177.992px` (two rows again),
head at y 119–149 against a ruler starting at y 157 — no overlap — and the top element at the head's
own position is the text SPAN, not the canvas. Screenshot at 960x640 confirms.
Worth recording plainly: `ruff`, `mypy`, `pytest`, `tsc` and 222 vitest tests were all green while
this was broken. A CSS layout regression is invisible to every gate in this project.

**D1 — keyboard and ARIA.** The Claude Browser pane cannot test this: its tab runs unfocused, so
`Input.dispatchKeyEvent` never reaches the renderer's focus machinery (40 Tab presses left
`activeElement` on `<body>`). Measured instead in a real headful Chrome over CDP.
- **Focus path: 18 stops**, clean cycle, no trap, Shift+Tab an exact mirror, identical at 1440x900
  and 960x640. 19 with a unit selected (Clear selection arms), 18 while offline with Retry now first.
  Deliberate non-stops, justified in source: verdict segments (`tabindex=-1` — there can be hundreds;
  they keep button semantics, `aria-pressed` and full names, and `[`/`]` is the better sequential
  route) and the unchecked radio of each segmented control (roving tabindex).
- **Focus rings proved on pixels**, not on CSS: animations frozen, screenshot unfocused, Tab to each
  stop, screenshot, diff. All 18 stops matched `:focus-visible` and changed pixels; worst indicator
  contrast **10.19:1** against WCAG 2.4.11's 3:1. Two were fixed to get there — the lit A/B segment
  was a cyan ring on a cyan key (3.72:1 -> 15.02:1), and the drop well's whole indicator was a 1 px
  border-colour change because its file input is `opacity: 0` (8.31 -> 12.11:1).
- **axe-core 4.13.0** run over 8 states (idle / done / done+guard tables scrolled / shortcut overlay,
  each at 1440x900 and 960x640). Before: **1 serious violation** — `scrollable-region-focusable` on
  the shortcut list at 960x640, which scrolls but had no focusable content, so a keyboard user could
  not scroll it. After: **0 violations in all 8 states.** axe is a devDependency injected at run
  time; `grep -c axe dist/assets/*.js` = **0**.
- **Live regions, measured over a real 8 s job** with a MutationObserver on accessible text: exactly
  **two** announcements — "Processing started" at 5 ms, "Processing finished. 5 of 5 units enhanced."
  at 8070 ms — even though the header, plate, footer and source strip all re-render on every SSE
  tick. One region was removed to achieve that: the header lamp was `aria-live="polite"` and narrated
  the same two events in words that said less. Killing the engine produced exactly **one** assertive
  announcement while the retry countdown ticked ~24 times in the same 6 s without being announced.

**D2 — contrast measured two independent ways**, because the previous audits skipped 122
gradient-backed nodes:
(a) *CSS layer compositing* — backdrop built from `html` down, stacking every background-color and
every gradient `background-image` layer **evaluated at that text's own position** (projected onto the
gradient line, stops bracketed, interpolated in premultiplied sRGB), plus ::before/::after fills and
covering siblings resolved by paint order.
(b) *Differential pixel truth* — capture as shipped, then capture again with every glyph emptied via
`-webkit-text-fill-color: transparent` and animations pinned; the changed pixels are glyph pixels,
and reading those same pixels from the second capture gives the true backdrop behind the letters —
canvas pixels, inset shadows, blurs and all.
- The two "failures" earlier audits reported were **artefacts, now explained rather than excused**:
  the wordmark at 1.08:1 uses `background-clip: text`, so the gradient *is* the glyphs (really
  10.7:1), and the A/B lit label at 1.10:1 is filled by a sibling `.thumb` no ancestor walker can see
  (7.9:1 by CSS, 8.11:1 by pixels — the same false positive I recorded in iteration 2 and could not
  resolve then).
- **One real failure, findable only by the pixel method:** `.pc-sub` "DFN3 restoration core" at
  **4.49:1** (1440x900) and **4.42:1** (960x640) — the raised controls panel paints its interior
  near rgb(30 36 40), which no flat-token audit models. Fixed **by token, not by font size**:
  `--fg-4` #808a99 -> #86909f. After: 4.86 / 4.78:1.
- **Final: 0 failures by both methods across 9 states** — idle, done, done+scrolled and overlay at
  both sizes, plus engine-offline (87/82/139/114/162/115/115/65/131 nodes). 0 unresolved layers.
  Whole-screen worst case now **4.78:1**. Text over canvases passes on pixels (waveform panel head
  7.21:1 measured vs 7.14:1 predicted — only measurable at all because of the grid fix).

**Gates:** ruff clean, format clean, `mypy --strict` clean, 492 pytest, 41 fuzz, `pnpm typecheck` +
`pnpm build` green, **222 vitest passed**, worker grep 0, axe absent from the bundle.

**Honest gaps carried out of the loop** (see the closing report): the running phase was not put
through the contrast walker as its own frame (its tokens are measured elsewhere); the modal traps Tab
but does not set `inert`, so a screen reader in browse mode can still walk background content; no
real screen reader was run — the announcement log is what the DOM would hand an AT, not a recording;
contrast was measured at 1440x900 and 960x640 but not re-measured at ultrawide where the layout
reflows to three columns; and 13 below-the-fold shortcut rows at 960x640 were measured by class
equivalence rather than directly.

## Adversarial audit — 2026-08-20 — "27/27" was premature; 6 boxes unticked

Three independent skeptics were told to REFUTE the completion claim rather than confirm it. Most of
the work survived independent re-measurement — several claims came back stronger than the log's own
evidence (the verdict strip is pixel-exact against the waveform at every zoom, worst 0.05 px at 1:1;
history re-selection really costs 0 `/api/analyze` calls even across 10 rapid switches; the three
artefacts are SHA-256 identical to disk; streaming loudness re-derived from outside the codebase
agrees to 1.08e-7 LU and ffmpeg's own `ebur128` — a separate implementation — agrees too; 0 contrast
failures reproduce under a different pixel method at five widths including two the loop never
measured). But it refuted enough to matter, so **A6, B5, B6, C4, C5 and D4 are unticked (21/27)**.

### REFUTED — real defects, being fixed

1. **The A/B control lies about which deck you are hearing.** Delete the master from the work dir and
   restore that run: the cleaned `<audio>` fails with `MEDIA_ELEMENT_ERROR code 4`, the player's
   fallback calls `setActive('original')` — but never mirrors it into the store. The A/B control
   still reads CLEANED, lit, `aria-checked=true`, and pressing Play plays the ORIGINAL element
   unmuted (currentTime advanced 6.15 -> 46.75 s over ~30 s) while the UI insists you are hearing the
   cleaned master. No error anywhere. **This is the exact failure class the whole project exists to
   eliminate**, and it reproduces on ANY cleaned-deck load failure, not just a deleted file.
2. **A run whose master is gone restores as a success.** `selectRun` only re-reads when the cached
   analysis is missing, so a missing FILE is never noticed: header `UNITS 5/5`, `RESULT Complete`,
   and an enabled Master WAV link that returns 404.
3. **Two runs of the same profile overwrite each other.** Same output path, so the older history row
   shows its own cached report on screen while its download links hand over the NEWER run's bytes.
4. **A clip stranded by a mid-analyze kill never recovers.** The failure state is designed, but after
   the engine restarts the badge stays NO ANALYSIS forever, PROCESS stays disabled, `/api/analyze` is
   never retried, there is no retry affordance, and the error row still promises "this comes back on
   its own when it reconnects" — which is now false.
5. **Layout broken at 1280x800 and 2560x1440** (idle AND done). The inspector's hint row collapses:
   each `<kbd>` goes from clientWidth 15 px at 1440/1920 to **8 px with scrollWidth 11 px** — 37 %
   glyph overflow, so `[`, `]` and `?` spill out of crushed pills, and "step units" is squeezed
   52.2 -> 25.9 px and wraps into the keycaps. **1280 was never tested by the loop at all.** Second
   gate-invisible layout regression this project has shipped.
6. **A favicon 404** on a real cold load. The browser pane never issues that request, so the
   environment the loop measured C4 in is the one environment that hides it.

### The gates themselves were weaker than they looked

7. **The mutation gate can pass vacuously.** `run_suite` treats ANY non-zero pytest exit as "caught",
   and pytest runs with `-x`. In the audit's run, **7 of the 12 mutations were credited to an
   unrelated flaky chaos test** that short-circuited the suite before the mutated code ever ran. The
   12/12 I reported is true (the 7 were re-run in isolation and genuinely caught) but **the evidence
   that produced it was worthless**.
8. **The project's own release gate fails on a clean tree.** `scripts/run_release_checks.sh` step 3/5
   runs `mypy --strict src tests scripts data` -> **10 errors in 5 files**, every one in a test file
   this web effort authored. The loop only ever ran `mypy --strict src`, so it never saw them.
9. **Two of the four bugs I "fixed and verified" have no test pinning them** — `formatTimeShort`
   printing `1:60` and `clearPeaksCache` not re-arming the capability latch can both be reintroduced
   verbatim with 222/222 still green.

### Numbers corrected

10. **E1's headline is mono-only.** 12,756 -> 222 MB / 32.9 s reproduces exactly — for a mono float32
    file, which is the shape the loop's own fixture generator writes. The same 2,073.6 MB as **stereo
    16-bit costs 252-283 MB and 55-57 s: 1.7x the wall time.** The work is CPU-bound (self_cpu 54.17 s
    of 55.51 s wall) in the per-channel loudness biquads and 4x true-peak oversampling. Flatness in
    file length is upheld and is stronger than claimed (6x the file, 2 % *less* peak RSS).
11. **The streaming rewrite is not faster.** On a like-for-like 30-minute pair: whole-file oracle
    5.49 s / 3038.8 MB vs streaming 5.72 s / 225.8 MB. The memory win (13.5x here) is real; the
    "36.1 -> 32.9 s" speed win is not reproducible at that size.
12. **E3 "bit-exact on AAC" holds only after a position-dependent shift.** The container's stts
    timeline and the decoder's diverge linearly at 15.6 ppm: max |diff| = 0 at lag +1 sample at t=1 s
    but **+70 samples (1.46 ms) at t=94 s**. Over a [40,45) window, 430 of 1200 buckets differ from a
    full decode by up to 0.313 full scale, and at 1:1 zoom the "raw samples" are the wrong 48 samples.
    No committed test covers a mid-file window on a lossy container. (Related to, and larger than, the
    overview-grid trade-off already recorded in iteration 3.)
13. **C2's "126.1 kB gz" is a property of the file on disk, not of the product.** There is no
    GZip middleware in the server, so **439.8 kB actually crosses the socket** on every cold load. The
    box's bar is still met (439.8 kB raw < 500 kB) but the number in the log described a compression
    step that does not exist.

### Also weaker, queued behind the above

14. Engine death mid-job takes **~11 s** to notice (the SSE stream dying does not trigger a probe;
    only the 10 s health poll does), so the UI shows a phantom "Enhancing unit 2/5" meanwhile.
15. An **8-channel WAV** gets no pre-flight warning and fails with the raw engine string naming an
    internal knob (`split_speakers`), truncated mid-word — the same defect class C5 claimed to have
    eliminated.
16. The spectrum's REMOVED **key and accessible name are name-driven** while the fill is value-driven:
    with zero removed pixels the key still reads "Removed" lit and the aria-label still promises it.
17. The **overview scrubber has no hover state** (0 pixels change) despite being a draggable Tab stop.
18. `Footer.tsx` hardcodes `<b>Engine error</b>` and reuses it for non-engine failures — a clipboard
    permission denial surfaced as "ENGINE ERROR".

### Orchestrator follow-up on the audit's E3 finding (measured, 2026-08-20)

The audit reported AAC windows drifting up to 70 samples (1.46 ms) at t=94 s **against the unseeked
full decode**. I checked the question that actually decides whether E3 is honestly ticked — whether
the *display* is self-consistent and playhead-aligned — and it is:

- `/api/analyze` overview vs `/api/peaks` over the identical whole-file span, 1200 buckets:
  **mean |diff| 0.00008, max 0.025, zero buckets over 0.05.** Same timeline.
- The same 2 s span near **t=89-91 s** (the audit's worst-drift region) requested two different ways —
  once directly at 1143 buckets, once as a slice of an 80-94 s window at 8000 buckets — cross-correlates
  to **best alignment at lag = 0 buckets** (bucket = 1.750 ms), mean |diff| 0.00637, 34/1143 buckets
  over 0.05 (bucket-grid phase, not position).

So `/api/peaks`, the playhead and `<audio>` share one timeline and agree with each other: what you see
at any zoom is where the playhead is. The outlier is the **unseeked full decode**, which is what the
*pipeline* runs on — so the report's unit times sit on a timeline that diverges from the display's by
up to 1.46 ms at the end of a 95 s file. That is sub-pixel at normal zoom and about 35 px at 1:1, and
it only matters if you zoom to 1:1 exactly on a unit boundary near the end of a long lossy file.

**E3 stays ticked** — its requirement (deep zoom shows true samples, not interpolated buckets) is met
and the display is coherent. The cross-timeline gap is recorded here as a known, bounded limitation
rather than claimed away, and the log's earlier "bit-exact against the full decode" wording was
overstated: it is bit-exact *after* a position-dependent shift.

## Iteration 6 — 2026-08-20 — the refuted A6 layout defect, and four honesty gaps

Ports and browsers for this pass: engine on 8797, the Claude browser pane for layout/ARIA
measurement (its tab is hidden — fine for layout, useless for focus and for frame timing) and a
real headful Chrome over CDP for anything that needs a live renderer's own clock or a real
network log.

### 1. The keycap row the loop shipped broken (A6, refuted)

Measured before the fix, and it reproduces exactly as the audit reported: the unit inspector's
hint row (`[ ] step units · ? all shortcuts`) is laid out by `.insp-empty`, which was
`grid-template-columns: minmax(0, 1fr) auto`. The stat tiles take their whole min-content width
and the lead column gets what is left — **114px at 1280x800, 102px at 2560x1440** against a hint
that needs ~190. Every `<kbd>` was shrunk to `clientWidth 8` around an `scrollWidth 11` glyph (a
37% overflow: "[", "]" and "?" spilled out of crushed pills) and "step units" was squeezed from
52.2px to 25.9 and wrapped into them. 1440 and 1920 were fine, which is exactly how a shrink bug
hides from a sweep that only looks at two widths.

The fix is at the cause, not on the keycap: `.insp-empty` is now a wrapping flex row whose lead
half has `flex: 1 1 200px` (the hint's natural width), so the stat row drops to its own line at the
point where the pair stops fitting, and nothing inside either half can shrink below its content. A
keycap and the words it means are also now one `<span class="chunk">` — an inline-flex,
`white-space: nowrap`, `flex: none` unit — so the row can only ever wrap *between* chunks.
After, at both bad widths: lead 546px / 534px, `keys` scrollWidth == clientWidth, every kbd
`15/15`, both labels `52/52` and `66/66`, row height 15px (one line).

### 2. The audit that says it stays fixed

A programmatic sweep over **7 widths x 5 states = 35 screens**. For every element that owns text:

- `scrollWidth <= clientWidth + 1` and `scrollHeight <= clientHeight + 1`. Clipped overflow
  (`hidden`/`clip`) and *visible* overflow are counted separately — the second bucket is the one
  that catches the keycap bug, whose glyphs spilled rather than being cut. Text cut by a clipping
  *ancestor* is a third bucket, measured from the text's own client rects, and `text-overflow:
  ellipsis` (designed truncation, always paired with a `title`) is counted apart from failure.
- Occlusion at the text's own centre via `elementsFromPoint`, with two corrections the naive
  version needs: pointer-events are forced on for the probe (a `pointer-events: none` overlay must
  not be able to hide from the test what it hides on screen), and a candidate blocker only counts
  if it really paints above — decided by z-index, then positioned-over-static, then document order
  between the two branches under the common ancestor. Blockers with an opacity-0 or
  visibility-hidden *ancestor* are not blockers (this is what the plate's four stacked state
  glyphs are).
- Skipped and counted rather than judged: `.sr-only` subtrees, text scrolled out of a scrolling
  ancestor, text off-viewport, and nodes behind the modal scrim.

**Falsification first**: with one keycap forced back to `width: 8px`, the sweep reports
`spill: 1 — kbd "[" cw 8 sw 11`; restored, 0. The instrument catches the defect it is here for.

| state | 960x640 | 1280x800 | 1440x900 | 1680x1050 | 1920x1080 | 2560x1440 | 3440x1440 |
|---|---|---|---|---|---|---|---|
| idle | 82 / **0** | 87 / **0** | 87 / **0** | 88 / **0** | 88 / **0** | 88 / **0** | 88 / **0** |
| analyzing | 84 / **0** | 91 / **0** | 91 / **0** | 92 / **0** | 92 / **0** | 92 / **0** | 92 / **0** |
| done | 104 / **0** | 115 / **0** | 115 / **0** | 116 / **0** | 116 / **0** | 116 / **0** | 116 / **0** |
| offline | 105 / **0** | 121 / **0** | 121 / **0** | 122 / **0** | 122 / **0** | 122 / **0** | 122 / **0** |
| overlay open | 141 / **0** | 189 / **0** | 189 / **0** | 190 / **0** | 190 / **0** | 190 / **0** | 190 / **0** |

(text nodes checked / failures, all buckets. The overlay rows also carry 84-88 nodes behind the
scrim, excluded. `analyzing` needed a 90-minute synthetic file — 16.8 s of real decode — to hold
the state open long enough to resize into it; it was captured live in three passes, never faked.)

**One state the matrix did not include found a sixth failure, so it is recorded here rather than
being left for the next audit**: with the error bar up at 960x640, the desk gives back the strip it
reserves (`--errbar-h` + 14) and the wave display lands at **33.5px**. The playhead readout and the
deck legend are positioned against that floor and are cut by it — 9.5px and 8px of clipped glyphs.
The panel's furniture gives up first, which is the rule this stylesheet already applies to the
scrubber and the verdict gutter at short heights, and nothing becomes unknowable: the transport
shows the same playhead time and the A/B switch names the same two decks. After: 0 failures with
the bar up, and both readouts return the moment it is dismissed (display back to 85.5px).

### 3. An 8-channel file no longer leaks a knob that does not exist here (C5)

Built a real 7.1 WAV (`ffmpeg … pan=7.1`, 6 s, 48 kHz), loaded it, ran it, deleted it.

Before: CHANNELS `8 ch` on a plain cell, PROCESS armed, and ten seconds later the engine's own
sentence — *"Multi-channel audio with 8 channels is not supported without explicit split_speakers
declaration."* `split_speakers` is a `channel_mode` value in the engine's config file, and the web
API's `JobRequest` carries `input_path`, `profile` and `overwrite` and nothing else: there is no
control on that screen, and no request the page could send, that would satisfy it. It was also long
enough to be cut mid-word in both the status line and the run list.

After, measured on the running engine:

- the CHANNELS cell warns the moment the analysis lands, exactly as the sample-rate cell does:
  `class="kv warn"`, a 10px glyph, and *"8 channels — this tool reads the file, but a run will be
  refused. It cleans one voice at a time; fold the file down to mono or stereo first."*
- the refusal is a designed kind (`channels`): status line **"Processing failed · 8-channel audio
  is more than this tool takes"** (a whole thought, no ellipsis), error bar **CLIP REFUSED** +
  *"“fixture-71.wav” carries 8 channels. This tool cleans one voice at a time — fold the file down
  to mono or stereo and load that."*, run list **"8-channel audio is more than this tool takes"**.
  The sentence fits uncut even at 960x640 (`.errbar .text` scrollWidth == clientWidth == 856).
  The engine's words stay in `raw` for a bug report and nowhere else.
- the ambiguous-stereo refusal, which leaks the same knob (`'dual_mono_same'`, `channel_mode`), is
  answered the same way rather than left for the next audit to find.
- PROCESS is **not** blocked, deliberately: the flag is advisory, like the rate flag, so a future
  engine that folds 7.1 itself makes this a stale note rather than a wall.

**A contrast failure this uncovered**, and it is the honest kind — the warn treatment was never on
screen in any state the D2 sweeps captured, because every one of them had a 48 kHz mono clip in it.
Measured on the shipped pixels: `.clipinfo .kv.warn .k` (the word "Channels") **2.29:1**, and the
same shape on `.kv.bad .k` at **2.57:1**. Both keys now use the ramp colour every other key in that
row already uses — **5.56:1** measured, against values at 12.69:1 (amber) and 5.88:1 (red) — and the
state keeps being carried by the value's colour and the glyph beside it.

### 4. A dead engine mid-job is now noticed in 9 ms, not 11 s (B6)

The stream's death was evidence nobody was reading: `followJob` reported the disconnect, the store
dimmed `streamConnected`, and then nothing happened until the 10 s heartbeat came round — a job in
flight makes no other request, so there is no `TypeError` for `probeSoon` to classify. The SSE error
path now calls a `probeNow()` that re-measures liveness on the spot. The healthy cadence is
untouched at 10 s.

Measured in a real headful Chrome (the pane's hidden tab aligns timers to ~1 s and cannot answer
this question), instrumenting `EventSource`, `fetch` and the DOM, with the engine **SIGKILL**ed 3 s
into a studio run:

```
+8 ms   EventSource error, readyState 0
+8 ms   GET /api/health   (the probe this fix adds)
+9 ms   data-offline=true · offline banner painted · "Engine offline — Engine unreachable"
```

The previous health probe had gone out at **-4120 ms**, so the old path's next look would have been
at +5880 ms — and the audit's ~11 s was the same mechanism with worse luck on the phase. (With a
graceful `SIGTERM` the same run takes ~1.1 s end to end, because uvicorn keeps the stream open while
it shuts down; that is the engine's exit taking the time, not the UI's noticing.)

Terminal reconciliation is unchanged and was re-verified: bringing the engine back reconciles the
phantom run to `failed` — plate reads **RETRY / Failed / Engine gone**, run list keeps it — and the
error bar now reads **ENGINE GONE**, *"The engine stopped while this run was in flight, so it never
finished. Nothing was written; press PROCESS to run it again."* That last string had no designed
`kind`, so it fell to the generic branch that builds a headline by cutting the detail at 89
characters: the status line and the run row both used to end on *"Nothing was writte…"*.

### 5. The three smaller honesty fixes

**The spectrum's REMOVED key was name-driven while the fill is value-driven.** The amber band is
whatever survives a `destination-out` punch — the frequencies where the cleaned curve sits below the
original — so a run that took nothing out paints no amber at all, while the key stayed lit and the
canvas's accessible name kept promising "the removed energy between them". The renderer now answers
`removedCoverage()` (measured in dB on the curves, so it is answerable before the panel is laid out;
0.32 dB is the 0.6 px the hatch uses at the shipped plot height), and both the key and the name
follow it. Measured on two real runs of the same clip:

| run | coverage | key | canvas name |
|---|---|---|---|
| studio, 5/5 enhanced | 0.08 | `item removed on` | "…with the removed energy between them across 8% of the band" |
| production, 0/6 enhanced | **0** | `item removed off` | "…Nothing was removed — the cleaned curve does not sit below the original anywhere…" |

**The overview scrubber had no hover state at all** — `.dragging` and the focus ring, and zero
pixels at rest, so its only affordance was `cursor: pointer`, which a keyboard or touch user never
sees. Rule 2 at the top of `interaction.css` calls that a bug. It now takes the same answer as the
verdict track directly beneath it (same kind of object: an inset well you drag along): one hairline
of extra light around the glass, and the window you grab lifts. Measured in a real Chrome, hover on
vs off — `.wave-overview` box-shadow gains `rgba(255,255,255,0.07) 0 0 0 1px`; the fit window's fill
goes 0.03 → 0.06 and its ring 0.08 → 0.16; the zoomed (cyan) window goes fill 0.10 → 0.16, ring 0.45
→ 0.62, glow 0.22/10px → 0.30/12px.

**The error bar called everything an engine error.** `<b>Engine error</b>` was hardcoded, and the
audit caught it publishing a *clipboard* permission failure as ENGINE ERROR. Every failure now
arrives with its own source (`failureSource`, driven by the classification that already knew):
Engine offline / Engine gone / Engine error / Engine refused / Clip refused / Version mismatch /
Access refused / Clipboard blocked / Busy / Waveform renderer / App error. Verified live: the copy
button in an unfocused tab now reads **CLIPBOARD BLOCKED** *"The browser refused clipboard access."*;
a corrupt WAV reads **CLIP REFUSED**; an 8-channel run reads **CLIP REFUSED**; an interrupted run
reads **ENGINE GONE**.

**The favicon 404 was real, and the browser pane is why C4 missed it.** Proved rather than assumed,
with two static servers on two origins serving the same page, one with the icon link stripped:

```
no icon declared →  "GET /favicon.ico HTTP/1.1" 404      (real Chrome, cold load)
icon declared    →  zero favicon requests
```

The icon is an inline SVG data URI in `index.html` — five bars in the product's two colours, amber
ORIGINAL running into cyan CLEANED — so the build stays one folder of self-contained assets with no
extra request to make. It decodes (`new Image()` resolves, 150x150 intrinsic).

### Gates and non-regression

`pnpm typecheck` clean, `pnpm build` green (343.42 kB / 107.06 kB gz JS, 83.87 kB / 15.96 kB gz CSS),
`pnpm test:run` **292 passed in 12 files** (13 of them new, in `src/state/preflight.test.ts`, pinning
the channel pre-flight, the two channel refusals, the interrupted-run headline and the error-bar
source labels; with the `channels` branch removed 4 of them fail, so they bite), worker still a
classic script (`grep -cE '\bimport\b|\bexport\b'` = **0**), axe-core 4.13.0 injected at run time
over idle / overlay / done / done+scrolled / offline / 8-channel-warned / 8-channel-refused at 1440x900
and 960x640: **0 violations**, the one `color-contrast` *incomplete* being the gradient wordmark as
before, and axe absent from the bundle.

Re-verified by measurement, not by eye: the view-linked verdict strip is still pixel-exact — the
track shares the canvas's x and width to 0.000 px, segment geometry at 1:1 is within **0.04 px** of
`(t - view.start)/(view.end - view.start)` and within **0.089 px** at 18x zoom after a wheel zoom;
zoom/pan, FIT, the unit inspector, the `?` overlay, the analyser, the process meter, streaming
analyze and the offline banner all still work; the focus ring set is unchanged (19 stops in a
two-run done state = the audit's 18 plus the second history row, no positive `tabindex`).

**Honest gaps from this pass.** (a) The focus *order* was re-counted structurally, not re-walked with
real Tab presses — nothing in this pass adds or removes a focusable element, but the 18-stop cycle
itself was last proved in iteration 5. (b) The occlusion test cannot see `::before`/`::after` fills,
which `elementsFromPoint` never returns; a decorative pseudo painting over text would be missed by
it (the differential-pixel method in iteration 5 is the one that catches those). (c) `analyzing` at
1440 and 1920 needed a second and third pass because the 90-minute analyse finished mid-sweep once
the file was in the OS cache; every cell in the table was captured with `.analyzing-row` on screen.
(d) The contrast fix above was measured with the CSS-compositing method on the two nodes it changed,
not with the full differential-pixel walker over every state.

## Second adversarial audit — 2026-08-20 — 6 more boxes unticked (21/27)

Round one's fixes were re-attacked from angles other than the ones that found them, the fix round's
own 17-file surface was swept for regressions, and a third skeptic hunted ground neither round had
touched. **A7, B4, B5, B6, B8 and D4 are unticked.** Three round-one fixes were confirmed to hold
(run-restore artefact truth with zero re-analyze, stranded-clip recovery incl. double-kill and
32 ms pid-swap variants, the mutation gate's credit rules incl. both negative controls).

### REFUTED

1. **A killed engine leaves its job running to completion — and the UI states the opposite.**
   Engine SIGKILLed 3.5 s into a run: the job child was reparented to init (ppid 1) and **finished
   13 s later**, writing a full 13,624,364 B master + report, SHA-256 identical to a normal 5/5 run.
   The UI meanwhile reconciled to `failed` with *"The engine stopped while this run was in flight, so
   it never finished. Nothing was written."* — false, with no route to the file that exists.
   `jobs.py` kills children only from the graceful `shutdown()`; the CLI child has no parent-death
   watchdog (only `enhancement/worker.py` has one, a level deeper).
2. **Stereo reports are half-invisible — and every fixture in this entire project is mono.**
   On a genuine stereo clip the engine emits per-channel units that **overlap in time** (ch0
   0.000-20.570 / 20.570-40.001, ch1 0.000-20.250 / 20.250-40.001). All four segments are
   `position:absolute` at the same y in **one lane**; scanning `elementFromPoint` across the 689 px
   track, channel 1 is topmost on **687 px, channel 0 on 1 px**. Half the report has no tooltip and
   **cannot be clicked or selected at all**. The waveform is one lane too, so overlapping units of
   different channels produce an identical highlight.
3. **The deck-fault plate collapses the spectrum panel** — the fourth gate-invisible layout
   regression, in a state the fix round *invented and never swept* (iteration 6's 35-screen matrix
   contains no deck-fault state). The plate takes 69.3 px from the spectrum panel at every size: at
   960x640 the canvas is **0 px** (was 72) and the legend is **10.9 %** visible, with
   `elementsFromPoint` returning a metrics tile where each label should be. With the offline banner
   also up the collapse reaches 1280x800.
4. **A run can permanently lose its cached analysis.** `onJobStatus`'s done branch guards
   `setCleaned` against a stale job — correctly — but the same guard also skips the **history**
   patch, so a run whose analysis returns after the next action reads "— LUFS Δ" for ever and costs
   a full `POST /api/analyze` on restore. Hit 1 of 12 runs.
5. **The release gate does not pass, and the mutation gate cannot score.**
   `tests/chaos/test_interrupt_cleanup.py::test_interrupt_leaves_no_partials_and_no_orphans[2]`
   still fails — reproduced **five times**, including bare `pytest` on an idle machine
   (*"SIGINT: partial outputs at destination"*). `run_release_checks.sh` exits 1;
   `scripts/mutation_gate.py` **aborts with exit 2** on its red baseline without scoring a single
   mutation. The 12/12 I reported is not currently reproducible on a clean tree. It passes in
   isolation (3 passed in 4.12 s), so it is ordering/load dependent.

### WEAKER

6. **Two deck-fault holes.** A master truncated to 100 bytes is **not faulted at all** — Chrome
   accepts the RIFF prefix (duration 0.000396 s), CLEANED stays lit, the transport reads
   00:00.0/00:00.0, Play does nothing, no message anywhere, run still "Done". And an engine outage
   during a deck load is **misclassified**: Chromium reports code 4, so the fault records
   `unreadable` and says "nothing here could decode it" about a healthy file — and since
   `retryFaultedDecks()` only retries `network`, the deck **stays dead for the rest of the session**.
   No test in the tree ever produces a real `network` fault from the player.
7. **Same-profile de-collision is session memory only.** Within a page session studio x3 gives
   plain/-2/-3, all coexisting,each row's links SHA-256-verified against its own bytes. But
   `overwrite: true` is always sent and `bumpOutputPath` only avoids paths *this session* used, so
   after a reload the next run silently destroys the earlier master (measured with a planted
   sentinel). The SUPERSEDED net is unreachable through the web UI.
8. **The live run's artefacts are never re-verified** — delete the on-screen run's master and the
   Master WAV control stays an enabled link whose own href answers 404 (the cleaned deck plays from
   an in-memory blob, so nothing probes).
9. **Restoring a run does not restore its profile** — a PRODUCTION row restores with the radiogroup
   on STUDIO, so the screen asserts both at once and **"PROCESS AGAIN" runs studio, not again**.
10. **The verdict strip reads "1 UNITS"** where its sibling run list gets the singular right.
11. **A NUL byte escapes the path policy as a 500** with a raw Python exception
    (`ValueError: lstat: embedded null character in path`), reachable through the documented
    `?file=` autoload, though `resolve_client_path` documents 400/403/404 as its only refusals.
12. **The axe evidence was overstated in this log**: axe declines to judge color-contrast on
    **55-193 nodes per state** ("background color could not be determined due to a pseudo element").
    The "1 incomplete" figure I recorded was a count of *rules*, not nodes. The 0-violations result
    is real; its contrast coverage is much thinner than the log implied. The differential-pixel
    measurements remain the load-bearing contrast evidence, and an independent re-measurement put
    `.kv.warn` at 5.17:1 / 12.2:1 against the 5.56:1 / 12.69:1 recorded here — ~7 % optimistic.
13. **Iteration 6's log entry documents none of that commit's five largest claims** (the deck fault,
    the three-HEAD artefact verification, the output-path de-collision, the stranded-clip retry and
    the gate-integrity rewrite). The code was verified behaviourally by the auditor; the log simply
    did not carry the measurements. Recorded here so the gap is not silent.

## Iteration 7 — 2026-08-20 — A7/B4: the half of a stereo report that was not on screen

Engine on 8933 (own port, own process), a real headless Chrome over CDP for every measurement
below (the Claude pane's tab does not composite, so it cannot screenshot, and its dispatched
clicks did not reach the page). Fixtures built for this pass and deleted after it:
`test_output/stereo/stereo-voice.wav` (L = the Flute 09 voice 0–60 s, R = the same voice from
30 s — genuinely different content per channel, Pearson **r = 0.0015**, max |L−R| 0.86),
`stereo-tone.wav` (L = 220 Hz over 0–40 s, R = 660 Hz over 20–60 s, r = −0.00002) and
`mono-voice.wav` (the same voice, 1 channel, 60 s).

### 1. What was actually wrong

The engine classifies both fixtures as `split_speakers` (`classify_channels`: correlation < 0.40)
and then decides **per channel**. The real report from `stereo-voice.wav`:

```
ch0 u0 0.000–20.570   ch0 u1 20.570–40.251   ch0 u2 40.251–60.001
ch1 u3 0.000–20.474   ch1 u4 20.474–40.528   ch1 u5 40.528–60.001
```

Two full sets of decisions over the same seconds, emitted one channel after the other — so a
real stereo report is *never* sorted by time to begin with. The strip laid all of them out in
one lane, `position: absolute` at the same y. Falsified here rather than taken on trust: with
the new lanes forced back into one (`.verdict-lane{top:0;height:100%}` injected at run time)
the same `elementFromPoint` scan the audit used reports **one** distinct segment centre-y and
**ch1 topmost on 1525 of 1526 px, ch0 on 0** — units 0, 1 and 2 unreachable by any pointer.
The instrument catches the defect it is here for.

### 2. One lane per channel, and the track's box unchanged

The strip now stacks a sub-track per channel *inside the same track element*. That matters:
the track's x and width are the waveform canvas's x and width to 0.000 px, and that is what the
view-linked alignment rests on, so the track may only grow **taller**. Lanes are
`left: 0; right: 0` bands at `top: lane/lanes*100%`, and the segment's own
`left`/`width` formula is untouched — a lane is exactly as wide as the track, so the percentage
resolves against the same number either way.

Measured on the running engine, 1440x900, `stereo-voice.wav`, studio:

| | before (audit) | after |
|---|---|---|
| distinct segment centre-y | 1 | **2** |
| topmost px at ch0's centre-y | 1 | **ch0 1015, ch1 0** (of 1018) |
| topmost px at ch1's centre-y | 687 (ch1) | **ch1 1016, ch0 0** (of 1018) |
| units with zero reachable px | 2 of 4 | **0 of 6** |

Per unit, at its own lane's centre-y: 348 / 333 / 334 px (ch0) and 347 / 340 / 329 px (ch1) —
every unit gets its whole width, because the `L`/`R` tag is `pointer-events: none` and takes
nothing from the segment under it. Track alignment after the change: `track.x − canvas.x =
0.000 px`, `track.width − canvas.width = 0.000 px`; segment geometry against
`(t − view.start)/view.span` is within **0.008 px** at FIT.

Widths swept the same way: **960x640** — track 30px (the `max-height: 720px` rule takes the lane
to 15px), lanes 15px, segments 9px, alignment 0.000/0.000, tags `13/13` client/scroll width and
`15/15` height (no overflow), legend hidden by the existing 1120px rule, scan 585/586 px per lane.
**2560x1440** — track 36px, lanes 18px, alignment 0.000/0.000, all six units reachable
(495–523 px each). The stereo strip costs the waveform 14px of height at 18px lanes and 8px at
15px lanes; that is the only thing on the panel that changed size.

Mono is untouched, measured rather than assumed: `mono-voice.wav` gives `data-lanes="1"`, **no**
lane elements, **no** tags, track height **22**, segment height **14** — the audit's own numbers —
and alignment 0.000/0.000. A dual-mono file lands here too, because the lane count is read from
the units' own channels (the engine processes ch0 and duplicates it), not from `input.channels`.

### 3. Selection that says which channel it is

The decks are one **mixed-to-mono** envelope: `/api/peaks` calls `.to_mono()` before it buckets,
and both decks play the mixed file. So the display was *not* split into per-channel waveforms —
drawing the same curve twice under an L and an R label would be a fiction, and the endpoint that
would make it true is engine surface this pass does not own. The channel is carried by the
**selection band** instead: it fills only its channel's horizontal slice of the display, closed
top and bottom with its own hairlines, over a much fainter full-height wash and faint full-height
edges that keep the *time* extent readable. A DOM tag inside the band says the words.

Measured on the two units that overlap in time (ch0 u1 20.570–40.251 vs ch1 u4 20.474–40.528),
same clip, same zoom, mean luminance inside the band's x-range vs outside it:

| selection | display top half | display bottom half |
|---|---|---|
| ch0 (L) | **67.04** (out-of-band 44.73) | 34.86 (out-of-band 31.04) |
| ch1 (R) | 46.89 | **54.75** |

The lane's own fill is ~6x the context wash in the other lane; 81 593 px of the 1018x300 crop
differ between the two selections, and the out-of-band columns are pixel-identical. The tag reads
`LEFT CHANNEL` at 25% of the display height and `RIGHT CHANNEL` at 75%, positioned by one style
write per view change (no React render on a pan). The inspector agrees: an `L`/`R` badge at the
head of the record beside the unit number, `· unit 2 of 6 · L` in the panel title, and the Channel
row promoted from `0` to `L · left`. On a mono report none of those three appear at all.

### 4. `[` / `]`, and what it walks

Changed deliberately, and the two vitest cases that pinned the old order were rewritten to pin
the new one (they are marked CHANGED in `selection.test.ts` with the reason). Reading order was
time-major with a channel tie-break; on a report whose channels overlap that flips lane on nearly
every press — ch1 20.474 then ch0 20.570 is a 96 ms move and a lane change — and visits two units
that both start at 0.000 back to back without the playhead moving at all. It is now
**channel-major**: each lane in time order, lane after lane, which is the order the lanes are
drawn in. With nothing selected the bootstrap is confined to the first lane, because a playhead
at 30 s sits inside a unit of *every* channel at once.

Walked live with real key events, 7 x `]` then 3 x `[`, reading the lit segment, the inspector's
index and badge, and the waveform tag after each press:

```
]  L u0 (1/6 L)  L u1 (2/6 L)  L u2 (3/6 L)  R u3 (4/6 R)  R u4 (5/6 R)  R u5 (6/6 R)  R u5 (clamped)
[  R u4 (5/6 R)  R u3 (4/6 R)  L u2 (3/6 L)
```

All ten steps agree across all four readouts. Bootstrap: selection cleared, playhead parked at
30 s, `]` selects **L u1** (20.105–40.030, the *left* unit the playhead is in) and seeks to 20.1.

### 5. A second, harder fixture

`stereo-tone.wav` is the audit's own shape and it lands a *mixed* verdict set — `original_reverted`
x2 and `original_no_speech` on L, `original_no_speech`, `original_reverted` and `original_error`
on R — so the lanes were also read with four different segment colours in them. Scan: ch0 1017/1018
at lane 0's centre-y, ch1 1017/1018 at lane 1's, six of six units reachable.

### 6. Contrast, a11y and gates

Measured on the shipped pixels rather than from the stylesheet: the lane tag's glyph is **9.83:1**
over its scrim on a cyan segment and **9.81:1** over an amber one, the waveform's channel tag
**14.57:1**, the inspector's channel badge **11.37:1**. axe-core 4.13.0 injected at run time over
the stereo done state: **0 violations**, the single `color-contrast` *incomplete* being the same
whole-app one as before (164 nodes, every text node over this stylesheet's gradients, the gradient
wordmark included). `pnpm typecheck` clean, `pnpm build` green, `pnpm test:run` **328 passed in 12
files** (13 of them new here: `reportChannels` / `channelName` / `highlightFor`, the channel-major
order, the out-of-time-order report, the first-lane bootstrap, and the mono no-channel case), and
the emitted worker is still a classic script (`grep -cE '\bimport\b|\bexport\b'` = **0**).

### 7. Honest gaps

(a) **A report with more than two channels is unit-tested only.** `channelName` numbers them
`C0, C1, …` and the lane CSS is written for any count, but the web API refuses a >2-channel file
before a job starts (`AmbiguousStereoError`, and the CHANNELS pre-flight warns about it), so no
run through this UI can produce one; only a CLI run with `channel_mode: split_speakers` declared
could, and there is no path that hands such a report to the page. Nothing on screen has been seen
with three lanes.
(b) **The waveform is still one mixed envelope.** The band and its tag say which channel a
decision belongs to; they do not show that channel's audio. A true per-channel display needs
`/api/peaks` to stop folding to mono, which is an engine change.
(c) The `L`/`R` tag paints over the first ~13 px of whatever segment starts at t=0. It takes no
pointer (measured: unit 0 keeps all 348 px of its width as a hit target) and a segment carries no
information in its first 13 px that it does not carry in the rest, but it *is* an overlay, and it
is there because the track cannot give up width without losing its 0.000 px alignment.
(d) Screenshots were taken in headless Chrome with SwiftShader (`--use-angle=swiftshader`), not on
the GPU; the WebGL2 path is the same code, but the exact anti-aliasing of the band edges on a real
GPU was not compared.

## Iteration 7b — 2026-08-20 — tonal restoration (user-reported: "the output is embarrasingly bad")

The user tested /Users/hawzhin/Desktop/Teat1vo.mp3 and was right to call the output embarrassing:
every band moved by exactly +15.7 dB — a flat loudness gain on a muffled, boomy source, no tonal
work at all. Cause: `mud_imbalance_db` scored 38.7 against a 42.0 gate (missed by 3.3 dB), and even
a hit would only have bought a 2 dB low cut against a ~24 dB presence deficit, with the presence and
air moves hardcoded to zero since the 3.1.1 thinning complaint.

**Fix: bounded tonal restoration in finishing.** Three bands measured against a target calibrated on
the references we actually have — the pinned 3.1.1 natural-voice fixture and Flute 09 (the file the
user approved). The suggested "broadcast" target was rejected by measurement: it would have given the
approved file a ~20 dB boost. Deadband-edge semantics: correction stops at the edge of acceptable, so
over-brightening is impossible by construction. Two gates keep it off empty bands (syllabic dynamics
>= 12 dB, capture backstop at -48 dB), both ramped after a hard threshold made adjacent units swap
10 dB of EQ. Caps: shelf +-6/+4, presence <= +10, brilliance <= +12, nothing above 6 kHz.
One filter per RECORDING (median across units) — per-unit it pumped 2.8 dB at boundaries.

Found on the way: `apply_speech_eq` has always delivered TWICE the requested dB (`filtfilt` squares
the response; -6.0 requested = -11.92 measured). Existing callers are calibrated around it; the new
bank designs for half and measures its finished cascade on every call, refusing one over bound.

**Results (end-to-end, real pipeline):** Teat1vo production: 2-4 kHz +3.5 dB, 4-8 kHz +8.9 dB, low
end within 0.5 dB, 8-16 kHz +0.2 dB (dead region honestly left alone), -16.0 LUFS / -1.5 dBTP,
Guard B PASS (consonant retention 3.69, hole 0.001). **Flute 09: zero correction, both profiles,
digit-for-digit identical to the pre-change run** — the regression canary the 3.1.1 complaint
demanded. Synthetic controls: flat voice untouched, brick-wall lowpass refused (`no-dynamics`),
boomy cut, thin/harsh gets bass back and no further thinning, near-silent declined.

**Open policy question, deliberately not decided by an agent:** the studio profile still ships the
flat-gain result on this file, because Guard A reverts DFN3's output (consonant retention 0.206) and
reverted units skip per-unit finishing entirely. Running finishing on reverted units would fix it but
changes the fail-closed contract ("REVERT = you get the original back"). Recorded for the user.

Gates: ruff/format/mypy clean, **574 passed** (was 514), fuzz 41, audit-models + doctor green. The
two new config keys change profile hashes (provenance, not pinned locks); core lockfiles verify
unchanged.

## Audit 3 follow-up — 2026-08-20 — the watchdog in every topology

The third audit left two watchdog defects; both are engine-side (`watchdog.py` / `cli.py`),
outside the goal checklist's lettered boxes but inside its evidence discipline.

**1. Background-spawn topology (the REFUTED item).** A batch started with `&` from a
non-interactive shell — also nohup pipelines and supervisors — runs with `SIGINT=SIG_IGN`,
which persists across exec into the per-file child, so the watchdog's `os.kill(self, SIGINT)`
was a silent no-op and only the 5 s `os._exit` backstop remained. The audit measured a child
that PUBLISHED a full master 4.5 s after its SIGKILLed parent died. Fix: the self-interrupt
signal is chosen at fire time — `signal.getsignal(SIGINT) is SIG_IGN` escalates to SIGTERM,
which `cli._install_signal_handlers` (installed before the watchdog arms) maps onto the same
KeyboardInterrupt unwind: worker torn down, staging removed, nothing published.
`cli.main()` also gained an outermost KeyboardInterrupt boundary, because a watchdog interrupt
that lands during startup (spawner died before arming completed) used to escape as a raw
traceback. Measured with the audit's own repro, both topologies: shell `&` batch SIGKILLed
mid-file → per-file child AND its worker gone **0.183 s** after the kill; plain Popen with
SIGINT ignored → **0.077 s**; destination still EMPTY 8 s later in both. Foreground unchanged.
Pinned by `tests/chaos/test_background_spawn.py`: the SIG_IGN premise probed on this machine,
then both topologies under the procwatch freeze protocol, with a 3 s interrupt-path deadline so
the 5 s backstop can never masquerade as the fix. In-process, the escalation and the SIG_IGN
unwind-with-cleanup are pinned in `tests/unit/test_watchdog.py` (red first: the escalation test
failed with `SIGINT != SIGTERM` before the one-line disposition check went in).

**2. Stale/exported HAWAVOCLEAN_PARENT_PID (the WEAKER item).** `HAWAVOCLEAN_PARENT_PID=99999`
made even `--version` exit 130 with a raw KeyboardInterrupt out of `watchdog.py`. The variable
is a private contract between a spawner and its DIRECT child, so it is honored only when
`getppid()` names the declared pid at arm time (death is then the ppid *changing*), or when the
child is provably a pre-arm orphan (ppid 1 and the declared pid gone) — anything else is
ignored with a logged warning. Measured: dead pid, never-alive pid, and a live stranger's pid
all exported → warning + `hawavoclean 3.2.0` + exit 0; the real contract still tears an orphan
down (chaos suite: 15/15, including the SIGSTOP-safe engine-topology tests audit 3 upheld).

Gates: ruff clean, format clean (191 files), `mypy --strict src` clean (79 files, and the
release script's wider `src tests scripts data` scope clean), full pytest **583 passed x3
consecutive** (574 + 9 new watchdog/topology tests), fuzz **41 passed**, mutation gate
**12/12 by owners**, `bash scripts/run_release_checks.sh` exit 0 (coverage 92.57 percent,
doctor all green). UI untouched by this work; `pnpm vitest run` 340 passed on the current tree.

Honest gaps: (a) if a spawner outside `cli.main()` arms the watchdog with BOTH SIGINT and
SIGTERM inherited ignored, only the 5 s backstop remains — in-repo callers cannot reach that
state because the CLI installs its SIGTERM handler before arming, but the library function
alone does not restore dispositions by design (an inherited ignore is the spawner's stated
wish). (b) `HAWAVOCLEAN_PARENT_PID=1` is refused silently by the pre-existing `<= 1` guard,
without the new warning. (c) The measured teardown times are one machine's; the chaos deadline
is 3 s against a 0.25 s poll precisely so load cannot decide the verdict.

## Iteration 7 addendum — 2026-08-20 — the five fix families commit 1488a33 carried without log evidence

Audit 3's bookkeeping finding, settled: iteration 7's entry documented only the stereo work,
while its commit also carried five fix families whose evidence lived in agent reports and was
then independently re-measured by the third audit. Recorded here, after the fact and marked as
such — these are iteration-7 measurements plus audit-3's re-verification, not new work.

**1. Orphan watchdog (the round-2 REFUTED B6).** New `src/hawavoclean/watchdog.py`: a spawning
process stamps its pid into `HAWAVOCLEAN_PARENT_PID`; the child arms a daemon thread before
argument parsing that polls `kill(pid, 0)` and `getppid()` every 0.25 s and raises SIGINT on
itself when the parent is gone (ordinary interrupt path: worker torn down, staging removed,
nothing published), `os._exit(130)` backstop after 5 s. Iteration-7 measurement on the audit's
own scenario (real engine, Flute 09, SIGKILL at "Enhancing unit 3/5"): before, the orphan ran
4.64 s longer and published a 13,624,364 B master + JSON + txt; after, the job child is gone in
0.25 s, workers gone, destination empty. Audit 3 re-measured the server path and upheld it:
child dead **0.139 s** (kill at +3 s), **0.16 s** (+8 s, deep in enhancement), **0.246 s**
(+0.2 s, the import/warmup window); SIGSTOP does not kill it (child unharmed through a 10 s
freeze, job reconciled to done after SIGCONT). The same audit REFUTED the background-spawn
batch topology (inherited `SIGINT=SIG_IGN`) — fixed and measured in the engine-side
"Audit 3 follow-up" entry above.

**2. Display-column floor (the round-2 REFUTED B8).** `.right` was `minmax(0, 1fr) auto`, so
the deck-fault plate starved the spectrum panel to 0 px at 960x640. The display track now has a
floor (`--display-floor`, 214 px / 193 px compact, stated as its parts in the stylesheet) and
the column scrolls with a scroll-shadow cue when both panels genuinely exceed it. Instrument
falsified first (floor forced to 0 reproduces plot 0 px / key 7.7 % visible / labels behind a
metrics tile), then 7 widths x 7 states = 49 screens: 0 spill, 0 occlusion, 0 hard clipping,
0 page overflow. Audit 3's own 16-cell sweep (4 viewports x deck-fault / deck-fault+offline /
FILE GONE / 8-channel-refused): spectrum canvas **>= 55.6 px in every cell**, legend 8/8 and
metric labels 3/3 visible by 5-point elementsFromPoint. Its one finding — `.hist-body` still
shipped `scrollbar-width: thin` — is fixed in the web entry below.

**3. Deck-fault truth (the round-2 WEAKER A7).** Three holes closed in iteration 7: a new
`truncated` fault kind checks the deck's decoded length against the length the run reports
(live: a 100-byte master reads readyState 4 / MediaError null / duration 0.000396 s and is
faulted, CLEANED disabled, row flagged FILE BROKEN); an engine outage during a deck load is
classified by evidence — the probe's own failure plus an injected liveness check — instead of
by Chromium's `MediaError` code 4, and `retryFaultedDecks` brings the deck back on reconnect;
every condemning fault kind marks the master unavailable with per-kind wording, so chmod-000
and PNG masters lose their enabled `<a download>`. The ask-twice discipline dates from here:
the engine answers 200 with a full Content-Length for a chmod-000 file and then delivers
nothing, so an answered-but-undelivered probe is asked once more — a second answer means the
file, no answer means the engine. Audit 3 confirmed each detection on a re-fetch and found the
four residual holes it reported (blind HEAD re-verification of the on-screen run; A/B state on
a condemned deck; the mid-load stale window; a valid master at 50.06 % of expected duration
raising no fault). The first and third are closed in the web entry below.

**4. History patch (the round-2 REFUTED B5).** `onJobStatus`'s done-branch history patch moved
in front of the "is this still the current job?" guard, so a run whose cleaned analysis lands
after the user has moved on keeps its cached numbers (before: the row read "— LUFS Δ" for the
rest of the session and restoring it cost a full `POST /api/analyze`; audit 2 hit it on 1 of 12
runs). Pinned by vitest ("the row that ran keeps its numbers").

**5. Profile restore (the round-2 WEAKER B5).** `selectRun` restores the run's profile, so a
PRODUCTION row no longer restores under a radiogroup reading STUDIO, and PROCESS AGAIN sends
the restored run's own profile (vitest-pinned; audit 3 watched the radiogroup follow
Studio <-> Production in passing and called B5 upheld).

Also in that commit, smaller and previously unlogged: same-profile de-collision within a
session (`bumpOutputPath`; the post-reload sentinel overwrite audit 2 measured remains open),
the mutation gate's owner-scoped credit rules with negative controls, NUL-byte and surrogate
paths as designed 400s, and honest queue positions. Gates at the commit: 574 pytest, 328
vitest, release script exit 0, mutation gate 12/12 by owners — all re-run by audit 3 on the
clean tree, exit 0 everywhere.

## Audit 3 follow-up — 2026-08-20 — web: the on-screen run can no longer be lied about, and the run swap is atomic

Engine on 8977 (own port, this tree's build), attacks driven in a real Chrome for Testing over
Playwright (headless, SwiftShader — the pane's hidden tab cannot composite; every number below
is from the real browser against the real engine). Fixtures confined to `test_output/iter8`.

### 1. A7 · re-verification now reads the file instead of asking about it

The audit's finding: the on-screen run's re-verification was three HEADs, and the engine
answers HEAD 200 for a chmod-000 master (the open fails only when a body is produced) and for
one truncated to 100 bytes (the stat is happy) — so attacking the file of the run being LOOKED
AT left CLEANED lit and Master WAV an enabled link delivering junk. Server truth, measured
first: `HEAD` on a chmod-000 master → **200**; ranged GET → **206 committed, then the body dies**
(curl: `size_download=0`, exit 18 "partial file"); on a 100-byte truncation → 206 with
`Content-Range: bytes 0-0/100` against the healthy file's `/13624364`.

The fix, in `EngineClient.verify` + `verifyArtifacts`: every artefact costs **one ranged byte**
(`Range: bytes=0-0`, consumed to completion — the same discipline the player's deck probe uses,
so no `net::ERR_ABORTED` lands in the log), `delivered` is the load-bearing answer, and the
master's full length from `Content-Range` is measured against the size the run recorded when
its master first loaded (`HistoryEntry.masterBytes`, written the moment the run lands and
re-recorded on every healthy sighting — never overwritten by a condemned one, because the
recorded size is the fact the condemnation rests on). An answered-but-undelivered byte is asked
once more; a second answer condemns the file, silence is an outage and no verdict (vitest: "an
undelivered byte followed by silence is an outage, not a condemnation").

Reproduced against the CURRENT on-screen run (94.6 s job, 13,624,364 B master), both "look
again" gestures:

- **chmod 000 + re-pick row**: Master WAV becomes an `aria-disabled` button with *"the engine
  cannot read a byte of it — the file is unreadable where it stands"*, row flagged **NO
  ACCESS**, plate **CLEANED DECK UNAVAILABLE** with matching wording, A/B flips to ORIGINAL
  with CLEANED disabled; JSON and txt stay downloadable. `chmod 644` + re-pick heals all of it.
- **truncate to 100 B + re-pick row**: row flagged **FILE BROKEN**, refusal names the numbers —
  *"The cleaned master on disk is 100 B where this run wrote 13,624,364 B — it has been
  truncated or rewritten, so it is not this run's master any more"* — same A/B flip, same
  disabled link; restore + re-pick heals.
- **engine coming back**: master chmod-000'd while the engine was SIGKILLed; on restart the
  condemnation arrived **409 ms after the offline banner cleared, with no user gesture**.

Vitest pins the client (`verify` sends exactly `bytes=0-0`, reads `Content-Range`'s
denominator, treats a dying body as undelivered, re-raises when nobody answers) and the flows
(truncation condemned by recorded length; chmod-000 condemned as NO ACCESS; 404 still FILE
GONE; a healthy sighting records the yardstick; outage stays an outage).

### 2. A7 · the audit's fourth hole: the run swap is now atomic-or-labelled

Reproduced first, in its current shape: `selectRun` switched the run's identity synchronously
(source, job chip, report) but the deck swap sat behind engine round-trips. With run B (12.0 s)
on screen and the engine **SIGSTOPped** (a hung engine: every request stalls — kill -STOP, a
real system state), clicking run A's (94.6 s) row left the job chip and clip name reading run A
while **both `<audio>` elements kept run B's 12 s blobs, the transport read `00:00.0 / 00:12.0`,
the status line still said "Done · 1/1 unit enhanced · short_studio.wav", and the A/B stayed
lit CLEANED — indefinitely, for as long as the hang lasted.** (With a SIGKILLed engine the
refused sockets fail fast and both decks end in designed `network` faults with reconnect retry
— iteration 6's classification, re-verified — so the hang is the reproducible remnant of the
audit's window.)

The fix settles the swap before the first await, in two halves. *Atomic*: the run's own cached
facts — analyses, report, status line — switch with the name (they are session memory; the
post-verification code re-states them and amends the cleaned half if the master fails). 
*Labelled*: a new `DualPlayer.claimOnly(deck, url)` retires any deck not already holding this
run's file, synchronously — `retire` keeps the element's `src`, so a same-file restore still
revives with zero requests. Measured after, same attack: transport reads **`00:00.0 / 00:00.0`**,
the status line carries run A's own summary, pressing Play produces **nothing** (no element
advances), and when the engine resumes both decks load run A's real audio (94.611854 s /
94.613333 s) with the A/B landing on CLEANED. The raw elements still privately hold the old
bytes during the window (the src-never-removed rule; nothing on screen or in the sound path
claims them). Vitest pins the ordering: `claimOnly` is called with both of the run's file URLs
*before* the first `verify` reaches the engine, the cached facts are already swapped
synchronously, and a failed run claims no cleaned deck at all.

### 3. The plural fix reaches the whole string family

Audit 3: "the fix was applied to the one surface the audit named, not to the string family" —
a 1-unit run still shipped "1 of 1 units enhanced" to the screen reader and "1/1 units
enhanced" to the footer and run list. One helper now owns the rule (`state/plural.ts`: the noun
agrees with the count it follows — `unitNoun`, `unitsEnhanced`, `unitsEnhancedSpoken`), and
every surface goes through it: the live region (`App.tsx`), the footer status line and the
copy-summary line (`actions.ts`), the run list's spoken sentence and visible cell
(`JobHistory.tsx`), and the verdict strip's count and lane labels (`VerdictStrip.tsx`).
`grep -rn "units" ui/src` now finds no hand-built plural outside comments. Six vitest cases pin
the family, including `0 of 1 unit` and `0/0 units`.

### 4. `.hist-body` joins the visible-scrollbar rule

The exact pattern `interaction.css`'s own comment warns about: `scrollbar-width: thin` makes
Chromium ignore the `::-webkit-scrollbar` rules beneath it and fall back to an overlay bar
invisible until already scrolling — and the run list really scrolls in the 960x640 fault states
(audit 3: scrollHeight 119 vs clientHeight 80/64), so rows below the fold read as missing.
Fixed to match `.insp-body`/`.right`: `scrollbar-width: auto` and the styled 8 px bar with a
visible track at rest. Verified on pixels in the exact state (960x640, offline banner up,
2 runs, scrollHeight 119 vs clientHeight 64): the edge strip now carries **30 thumb + 30 track
pixels at rest** over the body's 60 px span, against `.insp-body`'s 69 + 23 over 92 px — the
same recipe painting the same bar. A harness trap recorded for the next audit: Playwright's
default launch args include `--hide-scrollbars`, under which BOTH bodies read zero bar pixels
and zero layout gutter — any scrollbar claim measured in a default headless run is vacuous;
this measurement stripped that flag (`ignore_default_args`).

### 5. D2 · the differential-pixel contrast sweep, re-run on the current build

Audit 3's second bookkeeping finding: the full two-method pass predated two iterations of CSS
change. Re-run of the pixel-truth method (capture as shipped; blank every measured element's
glyphs via `-webkit-text-fill-color: transparent` with animations pinned; changed pixels are
glyph pixels; read those pixels from the blanked frame for the true backdrop; the element's
specified colour — alpha composited over that backdrop — against the worst such pixel) over
every visible text element, ten states on the shipped bundle:

| state | 1440x900 nodes / worst | 960x640 nodes / worst |
|---|---|---|
| idle | 84 / **4.95:1** | 79 / **4.95:1** |
| done | 114 / **4.85:1** | 103 / **4.56:1** |
| shortcut overlay | 102 / **4.98:1** | 59 / **4.98:1** |
| deck-fault (FILE BROKEN) | 107 / **4.64:1** | 96 / **4.56:1** |
| 8-channel refused | 102 / **4.88:1** | 90 / **4.78:1** |

**0 failures in all ten states.** The whole-screen worst is still `.pc-sub` "DFN3 restoration
core" — the bottom of the `--fg-4` ramp iteration 5 recut — at 4.56:1 in the done and
deck-fault states at 960x640, then `1.0x` at 4.97–4.99:1 and the clip `Duration`/`Rate`/
`Channels` keys at 5.08–5.12:1; the warn/refusal surfaces this sweep exists for (errbar text,
CLIP REFUSED label, warn cells) all clear 4.5. Falsification first: `.pc-sub` forced to
`rgb(70,78,90)` reports exactly one FAIL at 1.86:1; restored, 0. Declined rather than judged:
the wordmark (its glyphs are its background — background-clip:text; the CSS method carried it
at 10.7:1 in iteration 5) and the plate's running-phase "100 %" readout, which sits under an
opacity-0 ancestor in these states and paints no pixels — invisible in truth, so no verdict.
Skipped and counted per state: nodes behind the modal scrim (119), scroll-clipped shortcut
rows at 960x640 (43), hidden/zero-size (0–13).

### Gates and non-regression

`pnpm typecheck` clean, `pnpm build` green (**351.78 kB / 109.52 kB gz** JS, 86.39 kB /
16.52 kB gz CSS — still under the 500 KB budget), `pnpm test:run` **342 passed in 13 files**
(14 new here: 5 client `verify`, 6 plural, 2 atomic-swap, plus the outage/condemnation flow
cases), emitted worker still a classic script (`grep -cE '\bimport\b|\bexport\b'` = **0**).
No Python touched by this pass (the tree's engine-side changes are the watchdog entry above).
Re-verified on the final build in the same sessions: restore-with-healthy-files re-enables all
three links and the CLEANED deck; the mono flow (analyze → PROCESS → 5/5 → A/B) is unchanged;
the stereo lanes, panel floor and history behaviours were not touched by any of these diffs
(`VerdictStrip` changed only its two plural templates, CSS only the `.hist-body` scrollbar
block).

### Honest gaps

(a) Audit 3's two other deck-fault holes stand: the A/B on a condemned deck discovered through
a *deck load* (PNG-as-wav) was not re-attacked here — the `setAbMode(activeDeck)` mirror and
today's chmod/truncate runs suggest it is dead, but that is inference, not measurement — and a
valid master at just over half its expected duration (`DECK_MIN_DURATION_RATIO = 0.5`,
exclusive) still raises no fault. (b) `masterBytes` is session memory: after a reload the first
healthy sighting re-records it, so a truncation that happens *between* reload and that sighting
is caught by the deck's duration check, not by length. (c) During the labelled swap window the
A/B still shows the *intent* (CLEANED lit on an empty, silent deck) rather than a loading
state. (d) The contrast sweep ran in headless SwiftShader Chrome at deviceScaleFactor 1 —
same code path, not the GPU's exact antialiasing — and the running phase is still measured only
through its tokens, as iteration 5 recorded. (e) The condemnation attacks leave one incomplete
ranged response in the network log per attack (the probe reading a body the engine cannot
produce) — that is the attack path observing the attack, not a happy-flow regression.

## Loop close — 2026-08-20 — 27/27, third-audit-backed

A7 is the last box ticked, and unlike the two premature closes before it, this one is earned:
audit 3 upheld the stereo-lane half with independent pixel scans, and iteration 8 closed the
deck-truth holes with the audit's own attacks run against the CURRENT on-screen run — chmod 000
condemned via a 1-byte ranged GET ("the engine cannot read a byte of it"), truncation condemned by
byte count ("100 B where this run wrote 13,624,364 B"), both healing on restore, and the
hung-engine deck-swap hole fixed atomic-or-labelled (a deck can never again claim audio it does
not hold). The watchdog now also covers background-spawn topologies (SIGINT-ignored: escalates to
SIGTERM; child + worker gone in 0.077-0.183 s with an empty destination where a full master used
to publish), and a stale exported HAWAVOCLEAN_PARENT_PID warns instead of killing every CLI run.

Final state, verified by the orchestrator on the closing tree: ruff + format clean,
mypy --strict clean, **583 pytest**, 41 fuzz, **342 vitest**, worker classic (0 import/export),
D2 contrast re-run current (0 failures across 10 states, worst 4.56:1), release script exit 0,
mutation gate 12/12 by owning tests on the clean tree.

Three adversarial audit rounds were run against this goal; rounds one and two each refuted six
boxes that had been ticked in good faith, and round three upheld 16 of 18 fresh-eyes claims with
the remainder fixed and re-attacked above. Known, recorded limits that are NOT failures of the
goal: the studio profile ships a flat-gain (guard-reverted) result on sources DFN3 damages —
finishing-on-reverted-units is a fail-closed-contract policy decision deliberately left to the
user; 4-8 kHz that a source never captured is not synthesized; the analyze overview grid on lossy
containers follows the container timeline (bounded 1.5 ms divergence from the pipeline's decode);
and a >2-channel report has never been seen on screen because the web API refuses those files
before a job starts.
