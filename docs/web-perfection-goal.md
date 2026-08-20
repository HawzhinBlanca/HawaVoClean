# Goal: web UI at top grade (then Resolve, together)

Scope: **the web experience only** — `ui/` served by `hawavoclean serve`. The Resolve shell
is frozen until this goal is met; do not touch `resolve-plugin/` except to keep it compiling.
Never re-batch the user's Desktop folders; test media is `test_output/ui-smoke/Flute 09.m4a.mp4`
(copy of the real file) plus synthetic fixtures.

"Top highest grade" = every box below checked, verified in a real browser against the real
engine, with evidence (screenshot or measurement) recorded in `docs/web-perfection-log.md`.

## A. Visual grade (Waves / FabFilter / pro-2026 bar)
- [x] A1 Panel depth: every panel has consistent bevel lighting (1px top inner highlight,
      soft outer shadow), inset displays with vignette; no flat unstyled surface anywhere.
- [x] A2 Waveform display: anti-aliased, gradient-filled body with glow pass; original/cleaned
      overlay readable at every zoom; playhead with subtle bloom; time ruler with tick hierarchy.
- [x] A3 Spectrum display: log-frequency grid with labeled decades, dB gridlines, filled
      area under curves with gradient + glow, smooth (≥30 fps) live analyser, legend.
- [x] A4 Typography: strict scale (11/12/13/16 caps-label system), tabular numerals for all
      values, consistent letter-spacing on small caps, no default-font element anywhere.
- [x] A5 Micro-interaction: hover/active states on every control; PROCESS press animation;
      state transitions (idle→running→done) animated on transform/opacity only; LED pulses.
- [x] A6 Empty/loading states designed (before any file is loaded the screen must still look
      like a product, not a blank shell); drop-target highlight on drag-over.
- [ ] A7 A/B and verdict strip read instantly: verdict segments with hover tooltip + click to
      seek to the unit; A/B switch visually obvious which deck is live.

## B. UX completeness (web mode)
- [x] B1 Full keyboard map: Space play/pause, A/B decks, ←/→ seek ±5 s, Shift+←/→ ±1 s,
      P process, Esc cancel, ? shows a shortcut overlay. No modifier-key false triggers.
- [x] B2 Drag-and-drop AND file picker both work in web mode (upload path), with a visible
      upload progress for large files and a clear error for unsupported types.
- [x] B3 Zoom + scroll in the waveform (wheel/pinch to zoom, drag ruler to pan), with the
      verdict strip staying aligned to the visible window.
- [ ] B4 Per-unit inspection: clicking a verdict segment selects the unit — highlights its
      range, shows its guard scores/decision reason in a details panel, seeks the transport.
- [ ] B5 Job history within the session (last N runs with profile, outcome, LUFS delta),
      re-selectable without re-analyzing.
- [ ] B6 Graceful engine loss: server killed mid-job → clear offline banner, auto-reconnect
      (health polling), state preserved; SSE drop mid-job recovers to the correct terminal state.
- [x] B7 Report access: download master + JSON report + human-readable txt from the UI;
      "copy report summary" one-liner.
- [ ] B8 Responsive from 960×640 up to ultrawide; no overlap, no dead zones, sensible max
      widths at ≥1920.

## C. Performance & robustness
- [x] C1 60 fps waveform interaction (zoom/pan/seek) on a 90-minute file — measured with
      synthetic long audio; worker never blocks main thread >16 ms (Performance panel evidence).
- [x] C2 First meaningful paint of the built bundle < 1 s on localhost; bundle < 500 KB gz
      (excluding fonts — there are none).
- [x] C3 No memory growth across 10 consecutive analyze+process cycles (heap snapshot delta
      < 10 MB); audio elements and workers disposed on file switch.
- [x] C4 Zero console errors/warnings in all exercised flows; zero failed/404 network
      requests; all requests to 127.0.0.1 only.
- [x] C5 Adversarial inputs at the UI boundary: 0-byte file, 3-hour file, 192 kHz file,
      corrupt container, filename with quotes/unicode/emoji — every one ends in a designed
      error or success state, never a stuck spinner.

## D. Accessibility & quality gates
- [x] D1 Full keyboard operability (tab order, focus rings styled, no traps); ARIA labels on
      all controls; verdict tooltip content reachable without hover.
- [x] D2 Contrast ≥ 4.5:1 for all text (checked on the dark theme values actually shipped).
- [x] D3 `prefers-reduced-motion` honored (disables pulse/glow animation, keeps function).
- [ ] D4 Gates stay green every iteration: ruff / ruff format / mypy --strict / pytest
      (default + fuzz when engine touched) / mutation gate on clean tree / `pnpm typecheck`
      / `pnpm build`. UI logic that can be unit-tested (state machine, SSE client, API client,
      keyboard map) has vitest coverage; the rest is covered by scripted browser verification.

## E. Engine-web seam (only as needed by the UI)
- [x] E1 Analyze of a 3-hour file streams/chunks (no multi-GB peak RSS; measured).
- [x] E2 Upload path handles ≥1 GB file without buffering it fully in memory.
- [x] E3 Waveform peaks endpoint supports windowed re-query for zoom (`start_s`/`end_s`)
      so deep zoom shows true detail, not interpolated buckets.

## Loop discipline
Each iteration: (1) pick the highest-impact unchecked boxes, at most one area per iteration;
(2) red-test or scripted-browser-check first where feasible; (3) implement; (4) verify in the
real browser against the real engine; (5) update this file's checkboxes + append evidence to
`docs/web-perfection-log.md`; (6) run gates; (7) commit (small, honest messages), push every
few iterations. STOP the loop and report when all boxes are checked — then the Resolve pass
happens together with the user. If a box is judged wrong/impossible, don't silently drop it:
strike it through with a one-line reason in the log and tell the user at loop end.
