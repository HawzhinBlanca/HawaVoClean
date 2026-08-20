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
