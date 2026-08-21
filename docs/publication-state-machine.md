# Publication Transaction State Machine

This is the executable recovery contract for ADR 0005. `current` is the only authority. A journal,
generation directory or public alias without a valid `current` pointer is recovery material, not a
published output.

## Invariants

1. `current` is absent or is a relative symlink of the form `generations/<64 lowercase hex>`.
2. A target generation is accepted only after its manifest, sizes, artifact SHA-256 values and the
   report's claimed WAV SHA-256 all verify.
3. Every public alias has one owned relative target through the shared `current` pointer. No alias
   names a generation directly.
4. A pre-commit failure leaves the previous generation authoritative. A post-commit failure is
   completed forward only when the new generation verifies; otherwise the caller receives failure.
5. Recovery never removes or rewrites the previous generation.
6. The per-output advisory lock serializes publishers and recovery. Readers resolve `current` once
   under that lock and retain paths inside the selected immutable generation.

## Durable states

| State | Durable filesystem facts | Authority visible to readers | Recovery transition |
|---|---|---|---|
| S0 empty | No `current`; public names absent | None | Initialize owned bundle, then prepare S1–S5 |
| S1 bundle | Owned bundle and `generations/` directory durable | None, or the pre-existing legacy triplet | Validate ownership; continue preparation |
| S2 partial staging | One or more temporary generation artifacts may exist | Prior generation, legacy triplet, or none | Remove/ignore staging; retry from candidate |
| S3 staged generation | WAV, JSON, TXT and manifest are file-fsynced; staging directory fsynced | Prior generation, legacy triplet, or none | Verify; rename generation or discard invalid staging |
| S4 generation renamed | Content-addressed generation name exists; parent-directory fsync may not have completed | Prior generation, legacy triplet, or none | If present and valid, reuse; otherwise recreate |
| S5 generation durable | Generation rename and `generations/` directory fsync completed | Prior generation, legacy triplet, or none | Write prepared journal; repair/create owned aliases |
| S6 prepared | Prepared journal may name old/new generation; owned aliases exist | Prior generation if `current` exists; otherwise none because aliases dangle uniformly | Ignore journal for authority; repair aliases; continue |
| S7 pointer replaced | `current` atomically names new generation; bundle-directory fsync may not have completed | Exactly old or new after crash; never a mixed triplet | Re-read `current`; verify selected generation; retry/finish forward |
| S8 pointer durable | `current` replacement and bundle-directory fsync completed | New complete generation | Finish committed journal and verification; a recoverable interruption returns success only after both finish |
| S9 committed | Committed journal durable and public aliases verified against manifest | New complete generation | Idempotently verify/reuse |

The S7 old-or-new outcome is deliberate: a process kill or power loss between atomic rename and the
directory durability barrier may preserve either directory entry. Both outcomes are safe because the
old generation was retained and the new generation was already fully durable before the rename.

## Legacy migration

| State | Public view | Recovery |
|---|---|---|
| L0 complete flat triplet | Three regular files containing one matching generation | Copy, fsync and verify all three into a content-addressed generation |
| L1 legacy generation durable | Original regular triplet remains unchanged | Point `current` at legacy generation |
| L2 alias conversion in progress | Each converted alias and each remaining regular file reads the same legacy bytes | Resume conversion after hash comparison; refuse any differing file |
| L3 migrated | All public aliases traverse `current` to the legacy generation | Prepare and commit requested overwrite normally |
| LX incomplete/mismatched | One or two regular files, unexpected symlink, or bytes that differ from committed manifest | Fail closed for manual recovery; never overwrite |

## Failure and interruption matrix

`tests/unit/test_publication_transaction.py` owns the contract:

- It discovers every call made during overwrite to the copy wrapper, write wrapper, directory fsync,
  `rename`, `replace`, symlink creation and staging cleanup. It injects a failure immediately before
  and after every discovered call. Every case must leave either the old byte-identical generation or
  the complete new generation, then pass an idempotent retry.
- It separately injects a partial file write followed by permission loss.
- It sends real `SIGINT`, `SIGTERM` and `SIGKILL` subprocess interruptions at generation-files-durable,
  generation-committed, each first-publish alias replacement, before-pointer, pointer-replaced and
  pointer-durable checkpoints.
- It exercises first publish, overwrite, legacy migration, repeated recovery, identical content,
  disk full, cancellation, post-commit recovery failure, concurrent publishers, stable readers and
  malicious/unexpected symlinks.
- The same suite must pass on macOS/APFS and Linux/overlayfs using a read-only source mount.
- Mutation M9 changes the authoritative overwrite pointer back to the prior generation. Its declared
  owning test must fail, proving the pointer-ordering assertion is live.

## Reader rule

A consumer that needs more than one artifact must call `resolve_committed_publication()` once and use
the returned immutable paths. Resolving public WAV and JSON aliases independently across an overwrite
is prohibited because those two opens could intentionally observe different complete generations.

