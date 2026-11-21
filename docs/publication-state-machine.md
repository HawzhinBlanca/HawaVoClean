# Publication Transaction State Machine

This is the executable recovery contract for ADR 0009. The regular `current` JSON record is the only
authority. A journal, generation directory or visible WAV/JSON/TXT file without a valid pointer is
recovery material or a convenience export, not proof of a completed publication.

## Invariants

1. `current` is absent or canonical JSON naming exactly one 64-lowercase-hex generation ID.
2. A generation is accepted only after its manifest, sizes, artifact SHA-256 values and the report's
   claimed WAV SHA-256 all verify.
3. The visible WAV/JSON/TXT paths are ordinary regular files. They may be repaired only when their
   bytes still match a known generation; unexpected links, devices or user-edited bytes fail closed.
4. A pre-pointer failure leaves the previous generation authoritative. Once the pointer commits, a
   recoverable failure completes forward only from the verified new generation.
5. Recovery never rewrites an immutable generation or removes the previous one as part of correctness.
6. The per-output lock serializes publication/recovery. First-party readers resolve one immutable
   generation under that lock; job readers additionally bind audio/report/summary hashes.

## Durable states

| State | Durable filesystem facts | Authority | Recovery |
|---|---|---|---|
| S0 empty | No valid bundle/pointer | None | Initialize the owned bundle |
| S1 bundle | Owner record and `generations/` are durable | Prior pointer, legacy triplet, or none | Validate ownership; prepare candidate |
| S2 staging | Temporary generation files may exist | Prior pointer or none | Ignore/remove staging and retry |
| S3 generation durable | WAV/JSON/TXT/manifest are flushed; content-derived directory committed with true no-replace | Prior pointer or none | Verify/reuse the generation |
| S4 prepared | Journal names old/new generation | Prior pointer or none | Journal is not authority; continue |
| S5 pointer replaced | `current` names the new generation; parent flush may be pending | Old or new after a crash, never a partially authored generation | Re-read, verify and finish the observed authority forward |
| S6 pointer durable | Pointer replacement and bundle flush completed | New immutable generation | Refresh regular exports and committed journal |
| S7 exports refreshing | Zero to three visible files may reflect the new generation | New immutable generation | Repair remaining known exports; never trust a mixed triplet |
| S8 committed | Journal and all convenience exports verify | New immutable generation | Idempotently verify/reuse |

The old-or-new outcome around S5 is deliberate. The old generation remains available and the new one
was fully durable before the single authority transition. The three regular convenience exports are
not claimed to transition atomically.

## Legacy migration

| State | Public view | Recovery |
|---|---|---|
| L0 complete flat triplet | Three matching regular files | Copy, flush and verify into a legacy generation |
| L1 legacy generation durable | Original files unchanged | Commit the regular pointer to that generation |
| L2 ADR-0005 symlink bundle | Complete owned symlinks through one valid legacy pointer | Verify, migrate pointer format, then materialize regular exports |
| L3 migrated | Regular exports match the pointed generation | Publish normally |
| LX incomplete/ambiguous | Missing/mismatched files, unexpected link/reparse point, unsafe owner or bad manifest | Fail closed for manual recovery |

## Failure and interruption matrix

`tests/unit/test_publication_transaction.py` owns this contract:

- It discovers copy, write, directory-flush, no-replace, replace and cleanup primitives and injects a
  failure before/after each operation.
- It sends real `SIGINT`, `SIGTERM` and `SIGKILL` at generation/pointer/export checkpoints.
- It exercises first publish, overwrite, legacy migration, repeated recovery, identical content,
  disk full, cancellation, concurrent publishers, stable immutable readers, malicious links and
  user-edited exports.
- It proves job-bound lookup ignores mixed visible files and rejects same-audio/different-sidecar
  ambiguity unless all retained digests select exactly one generation.
- Local seams cover APFS/POSIX and Win32 primitives; native APFS/NTFS fault and power-loss runs remain
  release evidence.

## Reader rule

A consumer needing a coherent master/report/summary set calls `resolve_committed_publication()` once
and uses the returned immutable paths. A durable job calls
`resolve_immutable_publication_generation()` with its retained artifact hashes. Independently opening
the three visible convenience paths across an overwrite is prohibited. For portable external
archival, use and verify the Full Processing Record ZIP.
