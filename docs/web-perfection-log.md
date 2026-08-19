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
