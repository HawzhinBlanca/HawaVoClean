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
