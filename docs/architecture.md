# HawaVoClean 3.3 system architecture

HawaVoClean is an offline dialogue-restoration engine with three product surfaces: CLI/batch,
loopback web UI, and a DaVinci Resolve workflow plugin. All three call the same Python processing
engine and publish the same schema-v2 report/output contract.

## Processing flow

```text
read-only input
    │
    ▼
preflight ── media probe, release/model/config hashes, disk/resource checks
    │
    ▼
safe decode ── explicit channel classification ── canonical 48 kHz timeline
    │
    ▼
speech activity detection ── bounded units with context and safe/forced boundaries
    │
    ▼
one selected enhancement core in isolated worker process(es)
    ├── production: phase-coherent Wiener DSP
    ├── studio: WPE + full-band DeepFilterNet3 + late-tail suppression
    └── lowband: full-band DFN3, retained only below 1 kHz + original high band
    │
    ▼
length/timing/alignment validation
    │
    ▼
Guard A: source vs enhanced candidate
    ├── PASS ───────────────► candidate
    └── FAIL/ERROR/UNKNOWN ─► original speech unit
    │
    ▼
local deterministic finishing ── Guard B ── pre-finish fallback on rejection
    │
    ▼
continuity resolution/taper ── sample-exact timeline assembly
    │
    ▼
global BS.1770 gain ── bounded 8×-measured true-peak limiter ── PCM24 dither
    │
    ▼
structural/signal/report validation
    │
    ▼
immutable generation (WAV + JSON + TXT + manifest)
    │
    ▼
single atomically replaced `current` pointer = publication commit
```

`process --passes N|auto` repeats the complete guarded pipeline over the previous committed master.
An automatic extra pass is retained only while guard behavior does not regress and measured
speech/floor separation improves by at least 0.5 dB. `batch` intentionally remains single-pass and
isolates each input behind a long-lived child protocol.

## Core and guard boundary

Exactly one registered core is selected per pass. Multiple registered cores do not create an ensemble
or an unreported model choice. The chosen core ID, lock digest, parameters, device and runtime versions
are written into the report.

Guard A decides whether each enhanced speech unit may ship. Guard B evaluates deterministic local
finishing against Guard A's accepted unit. The probe is a spectral/integrity detector—not ASR and not
a linguistic oracle. Production uses strict spectral comparison; restoration profiles use integrity
mode so intended spectral repair is possible while timing, envelope, collapse and artifact defenses
remain. See [the fidelity-guard contract](fidelity-guard.md).

Guard-reverted units return to the original decoded unit and bypass per-unit finishing. Global static
mastering still applies to the assembled file. Changing this policy requires a new approved ADR and
Sorani evidence; it cannot drift through refactoring.

## Publication model

Three flat renames cannot commit a WAV and two reports as one filesystem transaction. Each public
output therefore owns an adjacent hidden bundle:

```text
.<output>.hawavoclean/
├── current -> generations/<generation-id>
├── transaction.json
└── generations/<generation-id>/
    ├── master.wav
    ├── report.json
    ├── summary.txt
    └── manifest.json
```

Files and directories are flushed, hashed and validated before one relative `current` pointer is
atomically replaced and the bundle directory is flushed. The prior complete generation remains.
First-party readers resolve the pointer once and use immutable paths from that generation; resolving
the public WAV and report independently across an overwrite is prohibited. The complete durable-state,
fault and recovery contract is [the publication state machine](publication-state-machine.md).

## Product surfaces and process boundaries

| Surface | Boundary | Important behavior |
|---|---|---|
| CLI | Current process + isolated enhancement workers | `process`, `batch`, `verify`, diagnostics and evaluation tooling |
| Batch | Long-lived child plus per-file deadlines | One failed/hung input cannot abort later files; non-zero summary on any failure |
| Web UI | Token-authenticated engine on exactly `127.0.0.1` | Bounded FIFO jobs, bounded uploads/retention, path policy, range audio, SSE progress |
| Resolve plugin | Sandboxed Electron renderer → validated preload IPC → main → loopback engine | Transactional install/rollback; local checksum-covered UI; all foreign navigation/popups denied |
| CPU container | Non-root UID/GID 10001, read-only root-compatible | Production profile only; explicit work/cache mounts; no studio/GPU claim |

The UI is one React bundle used by browser, the controlled Electron test shell and Resolve's embedded
runtime. The Resolve shell owns lifecycle, token generation, bridge validation and engine shutdown.
The engine never binds a non-loopback address. Audio, filenames and reports remain local unless the
user explicitly exports them.

## Trust boundaries

1. **Input media is hostile.** Probe/decode run with explicit limits; paths and channel ambiguity fail
   closed; the input is never opened for writing.
2. **Enhancement can fail.** Workers are killable, timed, memory-policed and disposable. Bad length,
   NaN, silence, crash, timeout or ambiguous guard state returns the original unit.
3. **Reports are claims.** Schema-v2 release/build/model/runtime identities recompute and mismatch is
   rejected. Schema v1 remains readable but cannot claim modern provenance.
4. **Renderer content is untrusted by default.** Only the checksum-covered `hawa://app` UI and one
   authenticated engine port are allowed. IPC validates web contents, frame and URL.
5. **Resolve's Electron is vendor-owned.** Application controls reduce reachability but do not patch
   the embedded binary; [the residual risk](resolve-runtime-risk.md) remains an explicit release gate.
6. **Evaluation data is separate from release code.** Raw audio, identities, source client keys and
   consent records stay out of Git. Held-out splits are hash-locked and unavailable during calibration.

## Architectural invariants

1. Source files are read-only and output sample count/channel layout/timeline are conserved.
2. Any uncertain unit returns to original audio; no enhancer fault may create silence or abort a
   long-form file after safe per-unit fallback is possible.
3. One selected, hash-locked core runs per pass and is named truthfully in the report.
4. Guard A owns enhancement selection; Guard B owns local finishing selection.
5. A forced-boundary transition either returns exactly to source at the joint or reverts safely.
6. The true-peak ceiling is measured independently at 8× oversampling and hard clipping is forbidden.
7. A visible publication resolves to one complete verified generation; a mixed generation is never
   authoritative.
8. Network processing is loopback-only and every API request requires a fresh token.
9. The tested artifact identity and the reported artifact identity must match before release.

## Related decisions

- [ADR 0004: release support and safety contract](adr/0004-release-support-and-safety-contract.md)
- [ADR 0005: committed output generations](adr/0005-committed-output-generations.md)
- [ADR 0006: band-split restoration core](adr/0006-band-split-restoration-core.md)
- [ADR 0007: canonical release identity](adr/0007-canonical-release-identity.md)
- [UI and Resolve contract](ui-contract.md)
