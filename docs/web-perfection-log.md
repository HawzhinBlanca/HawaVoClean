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
