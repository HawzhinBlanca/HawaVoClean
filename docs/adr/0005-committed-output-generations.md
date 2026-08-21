# ADR 0005: Commit Output Triplets Through One Generation Pointer

**Status:** Accepted  
**Date:** 2026-08-21  
**Deciders:** Hawzhin Mahmood and the HawaVoClean release implementation

## Context

`JobWorkspace.publish_atomically()` stages a WAV, JSON report and TXT summary, then calls
`os.replace()` three times. Those renames are individually atomic but not atomic as a set. Executed
fault injection proved that a hard kill can expose a WAV without reports and that an overwrite failure
can delete the prior WAV while retaining stale reports.

No portable POSIX primitive atomically replaces three independent flat paths. Continuing to use that
API while calling it atomic would preserve a false safety claim.

## Decision

Each requested output owns an adjacent hidden bundle:

```text
.<output-name>.hawavoclean/
├── current -> generations/<generation-id>
├── transaction.json
└── generations/
    └── <generation-id>/
        ├── master.wav
        ├── report.json
        ├── summary.txt
        └── manifest.json
```

The public filenames remain:

```text
output.wav                 -> .output.wav.hawavoclean/current/master.wav
output.hawavoclean.json    -> .output.wav.hawavoclean/current/report.json
output.hawavoclean.txt     -> .output.wav.hawavoclean/current/summary.txt
```

All three aliases traverse the same `current` symlink. A new immutable generation is copied, flushed,
hashed and validated completely before a temporary `current` symlink is atomically replaced and the
bundle directory is flushed. That **single pointer replacement** is the commit point for all three
artifacts.

Additional rules:

1. Generation IDs are content-derived from the generation manifest. Repeating identical content is
   idempotent; an existing generation is reused only after every hash verifies.
2. The prior generation remains immutable until the new pointer commit is durable. Cleanup is a later,
   bounded-retention operation and never part of commit correctness.
3. First publication creates the public aliases before `current`. Until `current` commits, the aliases
   are uniformly unresolved rather than exposing a partial generation.
4. Overwriting a legacy regular-file triplet first copies and flushes the complete old triplet into a
   legacy generation, points `current` to it, then replaces each public file with an alias. During that
   migration, every replaced alias and every not-yet-replaced regular file resolves to the same old
   bytes. An incomplete legacy triplet is rejected for manual recovery.
5. `transaction.json` records the intended old and new generation and is atomically rewritten. Recovery
   is idempotent and derives authority from `current`, not from a possibly stale journal state.
6. Unexpected symlinks, bundle ownership mismatches, path traversal and generation hash mismatches fail
   closed. HawaVoClean never follows an alias outside the expected adjacent bundle.
7. All first-party readers resolve committed bundles. Legacy complete triplets remain readable, while
   partial or hash-mismatched sets produce an explicit verification error.
8. Once the pointer commit is durable, an interrupt cannot truthfully roll the generation back. The run
   reports success/recovery of that committed generation rather than claiming cancellation destroyed it.

## Options Considered

### Three renames plus best-effort rollback

| Dimension | Assessment |
|---|---|
| Compatibility | High |
| Crash safety | Fails under `SIGKILL`/power loss |
| Overwrite safety | Can lose the prior file |
| Complexity | Superficially low |

Rejected because the existing implementation has already failed executable probes.

### Backup journal around three flat paths

| Dimension | Assessment |
|---|---|
| Compatibility | High |
| Recoverability | Good after restart |
| Concurrent visibility | Partial states remain visible |
| Complexity | High and stateful |

Rejected because recovery after the fact is weaker than one authoritative commit point.

### Immutable generations plus one shared pointer

| Dimension | Assessment |
|---|---|
| Compatibility | Public names retained through relative aliases |
| Crash safety | One atomic authority transition |
| Recovery | Idempotent; old generations retained |
| Complexity | Medium, explicit and testable |

Accepted because all public artifacts change generation through one filesystem operation.

## Consequences

- Public output paths become HawaVoClean-owned symlinks on supported platforms.
- Copying only a public symlink without its adjacent bundle is not a valid export; export tooling must
  dereference it or package the committed generation.
- The bundle consumes additional space until retention runs, intentionally preferring recoverability
  over premature deletion.
- Existing consumers that open the familiar paths continue to work because ordinary file APIs follow
  the relative aliases.
- Windows requires a separate design and remains outside the `v3.3.0` contract.

## Verification

- Inject failure before and after every copy, flush, alias replacement, pointer replacement and cleanup.
- Send `SIGINT`, `SIGTERM` and `SIGKILL` at each durable state and run recovery repeatedly.
- Assert that `current` names either the prior complete generation or the new complete generation.
- Assert all three public paths resolve through the same `current` pointer and their hashes equal the
  committed manifest.
- Exercise first publish, legacy migration, overwrite, identical republish, disk full, permission loss,
  concurrent readers and malicious/unexpected symlinks.

