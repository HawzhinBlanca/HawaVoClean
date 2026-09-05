# High-end production implementation

Status: foundation implementation in progress (2026-08-27). This is the architectural delta for the
standalone macOS/Windows, macOS Resolve and optional UAE-cloud product. It does not replace the
evidence-derived count in `generated-release-status.md`, and it must not be used to call a build
“10/10”.

## Locked product shape

HawaVoClean is one offline-first engine with three clients:

- a hardened Electron application on macOS 14+ Apple Silicon and Windows 11 x64;
- a thin macOS DaVinci Resolve client that discovers or launches the installed broker; and
- optional, per-file-consent cloud acceleration in AWS `me-central-1`, initially invite-only.

Natural processing remains the shipping safety baseline. Smart Safe is the default only after its
locked Sorani selection gates pass. The retired HawaRestore research checkpoint is not eligible for a
production capability; source and enrolled Restore remain blocked until a separately signed pack and
its independent evaluation both pass.

The critical path is:

```text
release truth
  → durable cross-platform engine
  → real source/enrolled Restore pack
  → qualified Smart Safe
  → signed desktop/Resolve/cloud clients
  → independent listening, security and host qualification
  → two reproducible signed RCs
```

## Foundation now present

These are implemented code foundations, not final acceptance evidence:

| Boundary | Implemented foundation | Deliberately not claimed |
|---|---|---|
| CI truth | Every shell pipeline uses explicit Bash `pipefail`; the aggregate names and checks every leaf; missing evidence uploads fail; a local injected UI failure proves both leaf and aggregate semantics. Mutable governance tasks are bound to the exact current contract hash instead of inheriting old proof. | The deliberate UI failure has not yet run through a disposable remote PR, so current GitHub leaf/aggregate proof remains open. |
| Job durability | Application-data SQLite WAL/FULL ledger, idempotency binding, transactional whole-export reservation, `unique`/`fail`/`replace`, startup artifact validation, restart interruption, bounded/paged history, stable terminal TTLs and compact idempotency receipts. Historical jobs retain all master/report/summary hashes, resolve their exact immutable generation even when two deterministic renders share the same master bytes, and remain directly addressable after pruning without re-entering bounded list history. | The 1,000-case real APFS/NTFS acceptance run is not complete. Internal storage still uses compatibility lifecycle names translated at the v1 boundary. |
| Filesystem publication | Platform abstraction for locks, atomic/write-through replacement and directory flushing; immutable generations remain authoritative while public WAV/JSON/TXT exports are ordinary files. | Native NTFS fault/power-loss qualification remains. Three independent public files cannot be one atomic filesystem object; the portable ZIP is the closed record. |
| Process ownership | POSIX process groups and Windows Job Objects, with Windows children created suspended, assigned before execution and then resumed; cancellation escalates across the full tree. Desktop and Resolve launch the broker with a parent-death lease, and hard-kill host tests prove the real child exits without graceful cleanup. | Real Windows timing and nested-job behavior remain unqualified. |
| Managed input | Bounded disk-backed uploads plus root-only registration of exact native selections; both yield opaque source IDs. Before durable request acceptance and probing, each job pins from one identity-checked, no-follow regular-file handle into an immutable, frozen, bounded-disk snapshot (`PinnedSource`); probe, whole decode and streaming decode consume that exact snapshot, and the request hash cryptographically binds source SHA-256 and byte size. Renderer path adapters accept only registered sources, marker-owned uploads or verified job-bound immutable artifacts, and explicit outputs must be derived siblings. | The snapshot adds one input-sized disk copy and has not been qualified at the 8 GiB/6-hour limit or on NTFS. Root-auth path compatibility and renderer path-form adapters remain for one release. |
| Natural long audio | Every Natural decode is disk-backed and independently capped. At 64 MiB decoded PCM the production path switches to bounded analysis/batching, disk stages, memmap assembly, streaming mastering and deterministic PCM24/float/RF64 encoding; forced short/long fixture outputs are byte-identical. | The real three-hour `<2 GB RSS`, scratch, cancellation and performance gates on macOS and Windows remain unmeasured. Restore is still whole-file. |
| Smart analysis | Fixed-state streaming acoustic analysis, bounded metadata capture/decoder process trees, a bounded shared analysis pool, conservative uncertainty and deterministic guard/ranking primitives. | It is explicitly `experimental_unqualified`: no calibrated Sorani classifier, 10,000-comparison ranker or locked Smart quality result exists. Smart Safe job routing stays blocked. |
| Model packs | Strict canonical manifests, Ed25519 signatures, mandatory signed licenses, payload/provenance binding, compatibility/expiry checks, atomic install, rollback floors and root-signed key rotation. Authenticity cannot self-promote readiness: a release-owned policy must match exact pack ID, version, manifest SHA and canonical CPU-inclusive provider set. | No production offline root, release qualification policy, Restore pack, provider matrix or genuine source-conditioned model exists, so Restore remains blocked. |
| Restore safety | Capabilities report source/enrolled Restore blocked; the legacy server path also refuses a loose research profile/checkpoint. | Corpus collection, training, enrollment verifier, provider inference, per-segment guards and independent Restore qualification are external/unfinished critical-path work. |
| Processing Record | Deterministic bounded ZIP with master, JSON report, summary and canonical manifest; verification uses one stable non-reparse descriptor. CLI and durable jobs create/verify it before completion and bind it to the exact generation. No-overwrite publication uses an atomic no-replace commit, so a racing writer is preserved rather than overwritten. | Publisher authentication and desktop Save-As UX remain open. Master and ZIP are two honest atomic boundaries, so a hard kill may leave a valid master with the job interrupted before its ZIP. A deliberately replaced shared ZIP may be unavailable to an older job, but the broker returns an explicit conflict and never substitutes newer evidence. |
| Broker security | Strict loopback Host/Origin checks, no wildcard CORS, no URL token, root/session authority tagging, bounded in-memory sessions, stdin bootstrap transport, opaque request IDs and sanitized internal failures. | Packaged-host inspection and an independent review are still required; root-auth legacy compatibility remains intentionally privileged for one release. |
| Desktop shell | Sandboxed/context-isolated renderer, minimal preload, native dialogs, exact-origin main-owned session injection, process lifecycle, fuses/config validation, source-shell/hard-crash tests, and a source-bound unsigned macOS app proof. The validator recomputes Electron's exact ASAR-header integrity digest, validates a deep ad-hoc seal and checksum-complete engine, then the gate launches the packaged app through its ASAR, preload, renderer, broker, authentication and network sandbox before separate engine smoke and post-smoke integrity checks. Candidate schema 2 marks that app non-distributable and reruns the packaged-app test after reconstructing the archived candidate. | The new app proof still needs its exact two-pass schema-2 release-gate run. Icons, updater, diagnostics, Developer ID/notarization/stapling, final DMG/ZIP, Authenticode/NSIS and real Windows QA are not complete. |
| Resolve client | Thin shell session design and transactional installer foundations. | Real Resolve two-version workflow/timeline/accessibility qualification, licensed SDK handling and notarized PKG evidence remain. |
| Cloud | Capability is blocked and local operation never uploads. | AWS control/data plane, invitations, consent/retention receipts, tenant isolation and soak/SLO evidence are not deployed. |
| Security | A maintained threat model names local, model, updater, Resolve and cloud boundaries and their open P0/P1 controls. | Independent review is required on the final signed candidate; implementing-agent review is not independent evidence. |

## Public contract

The closed v1 request and lifecycle models live in `src/hawavoclean/server/contracts.py`. The boundary
enforces:

- schema version 1 and forbidden unknown fields;
- explicit Smart Safe versus manual strategy;
- explicit reconstruction consent for every Restore-capable strategy;
- speaker profile requirements for enrolled Restore;
- per-request cloud consent only with `cloud_allowed`;
- visible-ASCII bounded idempotency keys and no duplicate source IDs;
- fail-closed unavailable routes; and
- durable public lifecycle names from `queued` through `completed`, `cancelled`, `interrupted` or
  `failed`.

`GET /api/v1/capabilities` is the only eligibility authority. Each Natural route is revalidated,
without loading a model or mutating broker runtime state, against its effective configuration,
guard calibration, optional dependencies, implementation parameters, core lock, weights,
phase/sample-rate contract and provider. A qualified route carries a composite manifest hash and an
exact reason; a missing or tampered input blocks it, and v1 submission repeats the same check. A
Restore route or pack may likewise be `qualified`, `experimental` or `blocked`, but loose profiles,
checkpoints or files never imply readiness. The legacy `/api/jobs` surface is a one-release adapter
and may not bypass those decisions.

## Work that requires people, hardware or external authority

The following cannot be manufactured by more unit tests or declared complete from this workstation:

1. A governed 300-hour, 250-speaker consented Sorani corpus with at least 45 locked speakers and
   disjoint source/session/speaker splits.
2. Training and frozen evaluation of a genuine observation-conditioned missing-band model and an
   enrolled-speaker verifier against licensed baselines.
3. At least 10,000 blinded three-rating Sorani pairwise comparisons for the monotonic Smart ranker,
   plus the locked 450-unit-per-condition safety matrices.
4. Native Windows 11/NTFS/DirectML/CUDA tests and signed, timestamped NSIS installation on clean
   network-disabled hosts.
5. Developer ID/notarization/stapling, updater signing and isolated native signing runners.
6. Real DaVinci Resolve tests on the latest two supported major versions and a decision on SDK
   redistribution rights.
7. AWS account deployment in UAE, KMS/Cognito/Batch/S3/SQS/DynamoDB policy review and tenant/retention
   soak tests.
8. Independent audio-science and security reviewers, Kurdish-speaking usability participants and a
   second release approver.

Until those inputs exist, the correct implementation behavior is to keep the associated capability
blocked and expose the reason—not substitute a mock, synthetic corpus, unqualified provider or
administrator bypass.

## Release claim rule

“True 10/10” is allowed only when every non-negotiable acceptance row in the approved plan points to
reproducible evidence on the same final linear commit/tag, two native signed RC runs reproduce their
artifact identities, no P0/P1 remains, and the independent reviewers approve after the last change.
A green local suite, a model file, an installer that opens, or a subjective listening impression is
not equivalent evidence.
