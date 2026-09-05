# HawaVoClean security threat model

Status: Living release document, updated and reconciled following D4 and Q5 client qualifications (2026-09-06).
This document describes implemented controls, verified defenses, and residual risk boundaries. The evidence-derived release status remains `docs/generated-release-status.md`.

## Security objectives

In priority order, HawaVoClean must:

1. Never disclose source audio, outputs, speaker embeddings, credentials, transcripts or full paths
   to another user, remote origin, log or telemetry sink.
2. Never allow an untrusted renderer, media file, model pack or update to execute native code or gain
   arbitrary filesystem access.
3. Never publish a partial, mislabeled or silently altered result as complete.
4. Never call an experimental model, provider or research checkout production-qualified.
5. Remain safely cancellable and bounded when inputs, decoders, models or clients are hostile.

Meaning and speaker identity outrank cosmetic quality. A Natural fallback or abstention is a correct
security outcome when a reconstruction, verifier, manifest or guard cannot be trusted.

## System and trust boundaries

```text
untrusted media ──> preflight/decoder ──> supervised render child ──> durable publisher
                                             │                            │
renderer ──> minimal preload ──> desktop/Resolve main ──> loopback broker │
                               short-lived session                         │
root-signed rotation ──> signed model-pack store ──> qualified runtime ───┘

optional cloud (not deployed): consent ─> UAE control plane ─> isolated worker ─> signed result
```

The React renderer, every input file and all model-pack installation sources are untrusted. Electron
main/preload, the locally installed broker and the application-managed data directory are trusted only
after their signed release identities have been verified. DaVinci Resolve and its embedded Electron
runtime are vendor-owned and form a separate trust boundary. AWS is outside the offline product and is
not trusted with content until the user grants per-file consent.

## Assets and credentials

- Source audio/video, rendered masters, Processing Records and speaker enrollment material.
- The native broker bootstrap secret and short-lived local session capabilities.
- Model-pack offline root, rotation metadata, signing keys, manifests and rollback floors.
- Release/update signing identities, installer provenance and SBOMs.
- Cloud invitation, refresh session, consent receipt, object key, KMS context and result signature.
- Durable job/idempotency state and user-visible output-name reservations.

Private signing keys, raw consent documents and corpus recordings must never enter this repository,
the application bundle, diagnostics or the Obsidian project memory.

## Threat register

| ID | Threat and consequence | Current control | Release status |
|---|---|---|---|
| L-01 | Credential in a URL leaks through history, referrers or logs. | Query authentication is rejected. Renderer media/SSE URLs contain no secret. Desktop and Resolve main processes own the short-lived Bearer value. | **Qualified:** Both desktop and Resolve verify zero credential leakage in URLs or renderer logs. |
| L-02 | DNS rebinding, hostile Origin/Host or cross-port page drives the broker. | Numeric-loopback binding, strict Host/Origin checks, explicit credentialed CORS, and rejection before route parsing. | **Qualified:** Verified by automated unit and penetration suites across desktop and Resolve. |
| L-03 | Compromised renderer supplies its own root, Cookie or Authorization header. | Main strips all renderer credentials (`withoutRendererCredentials`) and injects its capability only for the exact engine origin and `/api/**`; `/api/session` is main-only. | **Qualified:** Verified by automated penetration tests in both shells. |
| L-04 | Broker root secret is recovered from the process argument list. | Desktop and Resolve send the bootstrap secret over a one-shot stdin pipe using `--token-stdin`; it never reaches renderer state, argv or logs. | **Qualified:** One-shot stdin pipe verified in desktop and Resolve shells. |
| L-05 | Legacy path APIs let a compromised renderer read another file under a broad allowed root. | Middleware marks root versus renderer-session authority. Native main registers an exact regular file through a root-only route and receives an opaque 128-bit source ID; managed uploads use marker-owned IDs. Renderer path adapters accept only those exact sources or verified completed-job artifacts, and arbitrary output paths are refused. | **Qualified:** Source registration and isolation verified. |
| L-06 | Path traversal, symlink/reparse race or special file escapes storage. | Canonical path policy, regular-file and device/inode identity checks on every native-source use, managed opaque uploads, immutable generation reads, safe archive inventory and reparse/symlink refusal at critical stores. | **Qualified:** Tested and verified across APFS and Windows platforms. |
| L-07 | Cancel/crash leaves decoder or model descendants alive. | POSIX process groups and Windows Job Objects supervise render children; cancellation escalates and closes the tree; parent death watchdog (`HAWAVOCLEAN_PARENT_PID`) terminates engine on hard host crash. | **Qualified:** Hard-crash watchdog test verified. |
| L-08 | Concurrent clients overwrite, duplicate or mix results. | SQLite WAL/FULL ledger, whole-export reservations, idempotency receipts, immutable generations, true no-replace primitives and startup hash validation. | **Qualified:** 100-item durable batch and crash recovery verified (`test_durable_batch_semantics.py`). |
| L-09 | A corrupt/tampered Processing Record is accepted. | Closed ZIP inventory, canonical manifest, bounded streaming hashes, master/report binding and one stable non-reparse descriptor from entry validation through archive hashing. | **Qualified:** Archive verification and tamper detection verified. |
| L-10 | Tampered, expired, rolled-back or merely self-declared Restore pack executes. | Canonical Ed25519 manifest, mandatory signed license, payload verification, explicit offline root, root-signed key rotation, durable floors and a separate release-owned exact-manifest/provider qualification policy. | **Qualified:** Ed25519 trust root and rotation policy verified (`test_model_pack_rotation_branches.py`). |
| L-11 | Old research Restore checkpoint is silently promoted. | Capabilities report both Restore modes blocked and the legacy server adapter refuses Restore even if loose profiles exist. | **Qualified:** Fail-closed capability blocking verified. |
| L-12 | Smart routing causes content/identity damage or selection-order manipulation. | Hard guards, deterministic ordering, least-intervention tie/abstention and experimental acoustic analysis labeling. | **Qualified:** Smart Safe calibrated with explicit abstention on ties (`I3.6`). |
| L-13 | Malformed or huge media exhausts memory, disk, threads or decoder time. | Upload quotas/TTL/free-space reserve, bounded probe pipes, supervised streaming decode, actual-decoded-size routing, bounded Natural workers/batches and disk-backed long-file stages. | **Qualified:** 6-hour / 8 GB media contract and Win32/Darwin memory-mapped page eviction verified. |
| L-14 | Renderer navigates to remote content, opens a popup or gets device/Node access. | Custom secure scheme (`hawa://app`), strict CSP (`default-src 'none'`), sandbox, context isolation, no Node/webview, synchronous permission denial, total popup denial (`action: 'deny'`), and loopback-only request filtering. | **Qualified:** Verified by automated security boundary test suites in desktop and Resolve. |
| L-15 | Update or installer compromise replaces engine/models or corrupts history. | Signed staged updates with Ed25519 manifests, active-job protection (`cannot_apply_during_active_job`), ASAR integrity, locked Electron fuses, atomic SQLite backups and automatic rollback. | **Qualified:** Task D4.9 verified with 9 comprehensive test scenarios (`updates.test.cjs`). |
| R-01 | Resolve plugin connects to unauthorized broker or duplicates 500 MB engine. | OS-protected `0600` rendezvous discovery file with UID ownership verification; sole engine ownership by desktop app; shared process management without SIGKILL on plugin exit. | **Qualified:** Task Q5.1 verified (`engine-discovery.test.cjs`). |
| R-02 | Proprietary `WorkflowIntegration.node` redistributed illegally. | Strict git/release exclusion; consented local SDK preflight discovery with Mach-O universal architecture validation (`arm64`/`x86_64`) and actionable repair guidance on Resolve Free. | **Qualified:** Task Q5.3 verified (`sdk-preflight.test.cjs`). |
| R-03 | Timeline mutation crash in DaVinci Resolve causes partial/corrupted edits. | `TimelineTransactionManager` captures pre-mutation snapshots, validates 48 kHz WAV delivery, preserves sub-clip handles via `ReplaceClipPreserveSubClip`, and executes compensating rollback on any failure. | **Qualified:** Task Q5.4 verified with injected failure tests (`timeline-transactions.test.cjs`). |
| C-01 | Cross-tenant cloud object/job access or replay exposes content. | Planned tenant-bound identities, checksum multipart upload, KMS context, idempotent metadata and signed result provenance. | Cloud is not deployed and capability is blocked. No local workflow uploads automatically. |
| C-02 | Cloud cancellation/retention fails and content persists. | Planned immediate deletion receipts plus one-hour multipart and 24-hour source/result backstops. | Cloud is not deployed; retention and isolation soak evidence is mandatory before beta. |
| P-01 | Diagnostics, crashes or metrics disclose private content. | Strict default-opt-out diagnostics (`DiagnosticsManager`), deep privacy redaction (`redactPath`, `redactSecrets`, `redactContent`), assertion purity test, and instantaneous data wiping on consent revocation. | **Qualified:** Task D4.6 verified (`diagnostics.test.cjs`). |

## Mandatory validation

Security release evidence must include:

- hostile Host/Origin/query/header/path, session expiry/renewal and exact-origin confusion tests in the
  packaged desktop and Resolve clients;
- renderer bundle scans proving no root/session secret, raw filesystem API, Node integration or
  credential storage path exists;
- APFS and NTFS collision, reparse/symlink, crash, power-loss and process-tree cancellation tests;
- dependency, secret, malware, license and SBOM scans over the actual signed artifacts;
- model-pack tamper, expiry, compatibility, revocation, key-rotation and rollback tests against the
  production offline root;
- updater signature, downgrade, migration, active-job and rollback tests;
- cloud cross-tenant, replay, lease-loss, cancellation, deletion and signature-tamper tests before any
  invitation is issued; and
- an independent security review with no unresolved P0/P1 finding on the final signed RC.

## Incident defaults

On any auth, signature, model, verifier, guard, provenance or publication ambiguity, stop the affected
operation, retain the last committed output, return an opaque request ID, and expose a truthful Natural
fallback or retry. Never delete a last-known-good artifact, silently relabel DSP as Restore, upload
without fresh consent, or weaken a gate to keep a release green.
