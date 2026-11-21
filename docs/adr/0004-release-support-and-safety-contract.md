# ADR 0004: Release Support and Safety Contract

**Status:** Accepted  
**Date:** 2026-08-21  
**Deciders:** Hawzhin Mahmood and the HawaVoClean release implementation

> 2026-08-27 amendment: the current `v3.3.0` support boundary below remains historical release scope.
> ADR 0009 supersedes its POSIX-only publication mechanism for the high-end macOS/Windows product;
> that implementation foundation does not itself qualify Windows as supported.

## Context

HawaVoClean currently declares Python `>=3.11`, carries a stale CUDA Dockerfile, runs its Resolve
shell only on the user's Mac, and writes schema-v1 reports. A release cannot be called reproducible
or supported while those surfaces have no explicit boundary. The fail-closed promise also depends on
whether a guard-reverted unit is allowed through the finishing chain.

## Decision

The `v3.3.0` candidate has the following support contract:

- The offline CLI targets CPython 3.11–3.14 on macOS and Linux. Every claimed combination must pass
  required CI before release; an unproven combination is removed from the claim rather than waived.
- The primary product/Resolve qualification target is Apple silicon macOS with DaVinci Resolve Studio
  21.0.3. Resolve versions or operating systems outside the recorded matrix are unqualified.
- Windows is not a `v3.3.0` target. The crash-safe output design relies on POSIX relative symlinks and
  atomic replacement semantics that must not be implied to work on Windows without a separate port.
- A CPU Linux reference image is required. Studio/CUDA container support is advertised only if a
  separate GPU image passes a real NVIDIA run; otherwise it is explicitly absent from the release.
- Guard-reverted units remain byte-for-byte original at the per-unit decision boundary. Finishing is
  bypassed unless a later, human-approved ADR replaces this rule with Sorani listening evidence.
- Report schema v1 remains readable. The release will write schema v2 with complete build/runtime
  provenance; readers must reject unknown future major schemas.
- Existing complete flat WAV/JSON/TXT triplets remain importable. New publication uses the committed
  generation contract in ADR 0005 while preserving the familiar public filenames through stable
  relative aliases.
- The release is offline-first: no processing data, filenames, reports or audio may leave loopback or
  the local filesystem unless the user explicitly exports them.

## Options Considered

### Keep the current implicit support surface

| Dimension | Assessment |
|---|---|
| Complexity | Low initially |
| Release honesty | Unacceptable |
| Maintenance | Unbounded |
| Verification | Cannot define a complete matrix |

This was rejected because `>=3.11`, a CUDA image, and a Resolve plugin otherwise read as claims the
project has not proved.

### Support only the current machine

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Reproducibility | Weak |
| User fit | High |
| Portability | Needlessly narrow for the core CLI |

This was rejected for the core CLI but retained as the primary Resolve qualification target.

### Tiered CLI, container and Resolve contracts

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Release honesty | High |
| Verification | Explicit and automatable |
| Maintenance | Bounded by declared matrices |

This option is accepted.

## Consequences

- CI and documentation must distinguish core CLI support from host-specific Resolve qualification.
- The CPU container must be repaired; CUDA becomes a separate proof-bearing artifact or no claim.
- The publication design may use POSIX symlinks and atomic pointer replacement without pretending to
  support Windows.
- `REVERT = original` stays the safe default and cannot drift through an implementation refactor.
- Support can narrow if evidence fails, but cannot expand without new proof.

## Verification

- Matrix jobs install the built wheel and process/verify representative audio outside the repository.
- Contract tests pin schema compatibility, fail-closed passthrough and offline/loopback boundaries.
- Documentation consistency tests compare the published support table to the release manifest.
