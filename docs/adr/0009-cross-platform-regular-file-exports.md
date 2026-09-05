# ADR 0009: Cross-Platform Immutable Generations with Regular-File Exports

**Status:** Accepted; supersedes ADR 0005's public-symlink mechanism  
**Date:** 2026-08-27  
**Deciders:** Hawzhin Mahmood and the HawaVoClean high-end production implementation

## Context

ADR 0005 correctly established immutable generations and one authoritative commit point, but exposed
the familiar WAV/JSON/TXT names as POSIX symlinks. That made a copied WAV non-portable, excluded NTFS,
and encouraged clients to treat three filesystem aliases as though they were an ordinary export.

No APFS or NTFS primitive atomically replaces three independent visible files. The design therefore
needs to distinguish the atomic internal authority from recoverable user-facing convenience copies.

## Decision

Each output retains an adjacent application-owned bundle:

```text
.<output-name>.hawavoclean/
├── current                  # canonical regular JSON pointer
├── transaction.json
└── generations/
    └── <content-derived-generation-id>/
        ├── master.wav
        ├── report.json
        ├── summary.txt
        └── manifest.json
```

The rules are:

1. A generation is copied, flushed, hashed and closed before it is eligible. A true no-replace
   primitive commits a new content-derived generation directory without overwriting a concurrent
   winner.
2. Atomically replacing and durably flushing the regular `current` pointer is the single authority
   transition. Recovery trusts only a fully verified pointed generation.
3. The familiar WAV/JSON/TXT paths are ordinary self-contained files. After the authority transition,
   they are refreshed one at a time from the committed generation. They are portable exports, not the
   commit authority.
4. A hard kill can leave those three convenience files temporarily mixed. First-party readers acquire
   the publication lock, resolve the immutable authority and repair the copies before returning them.
   Job artifact endpoints resolve the exact generation from durable audio/report/summary hashes and
   never serve the mutable copies.
5. A Full Processing Record ZIP is the closed portable unit when the master and both reports must stay
   cryptographically bound outside HawaVoClean storage. It is verified before job completion.
6. Platform primitives cover write-through replacement, true no-replace creation, file/directory
   flushing, exclusive locking, reparse/symlink refusal and identity revalidation on APFS/POSIX and
   NTFS/Win32 seams.
7. Legacy ADR 0005 symlink bundles migrate only from a complete, verified state. Unexpected links,
   partial triplets, ambiguous same-audio generations or manifest mismatches fail closed.

## Consequences

- Copying the visible WAV produces a playable self-contained master without hidden storage.
- External software that bypasses the broker can observe convenience files between refresh operations;
  no documentation or API may call the three visible files one atomic object.
- Immutable generations and the Processing Record consume additional disk by design.
- Windows now has a real publication foundation, but Windows support remains unclaimed until native
  NTFS fault injection, installers, process-tree, update and real-host acceptance pass.
- Publisher authentication is separate from Processing Record integrity and remains an open gate.

## Verification

- Fault injection surrounds generation creation, pointer replacement, every convenience copy and
  recovery step.
- Concurrent publishers prove no-replace behavior and one complete winning generation.
- Startup and job-artifact tests delete/tamper visible files and require exact immutable recovery or an
  explicit `ARTIFACT_INVALID` failure.
- Same-master/different-sidecar generations are disambiguated by all retained digests or rejected.
- APFS and emulated Win32 seams are covered locally; the 1,000-case native APFS/NTFS acceptance matrix
  remains required release evidence.
