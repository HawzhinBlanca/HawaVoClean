# HawaVoClean v1 - System Architecture

## Architectural Flow

```text
IMMUTABLE INPUT
      │
      ▼
Preflight + media probe + hashes + disk/resource checks
      │
      ▼
Safe decode → explicit channel classification → canonical timeline
      │
      ▼
Speech activity detection → utterance groups with context
      │
      ▼
ONE frozen DSP enhancer (Wiener filter) in an isolated worker process
      │
      ▼
Length/timing/alignment validation
      │
      ▼
GUARD A: original vs enhanced
      │
      ├── PASS ───────► enhanced candidate
      └── FAIL/ERROR/UNVERIFIED ─► original
      │
      ▼
Sample-accurate timeline assembly at safe boundaries
      │
      ▼
Detection-gated local finishing
      │
      ▼
GUARD B: accepted timeline vs locally finished timeline
      │
      ├── PASS ───────► locally finished unit
      └── FAIL/ERROR/UNVERIFIED ─► pre-finish accepted unit
      │
      ▼
Global static loudness gain + bounded true-peak limiting
      │
      ▼
Final structural/signal validation
      │
      ▼
Atomic WAV + JSON/TXT report publication
```

## Architectural Invariants

1. **Source Preservation**: The input audio is opened strictly read-only and never modified.
2. **Fail Closed**: Any failure or uncertainty reverts to original audio at the speech unit level.
3. **Single Core**: One frozen core in production runtime; candidate exploration is isolated to benchmark tooling.
4. **Two-Pass Guard**: Guard A evaluates the enhancer's candidate; Guard B evaluates deterministic finishing output. Both compare spectral signatures (see docs/fidelity-guard.md for the scope limit).
5. **Sample-Accurate Timeline**: Integer sample math preserves exact input length, channels, and timing.
