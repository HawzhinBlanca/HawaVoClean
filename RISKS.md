# Risk Management & Mitigation Register

| Risk ID | Category | Description | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R-01 | Audio Integrity | Processing artifact alters speech spectrum. | Medium | High | Spectral-change guard reverts units whose signature diverges; fail-closed passthrough on any fault. **Scope limit: the guard detects spectral change, not linguistic change.** |
| R-02 | Operational | Worker crash or hang during long-form batch. | Medium | High | Isolated worker subprocess with hard deadlines, restart, and unit-level fallback — verified by chaos tests that SIGKILL and hang the worker. |
| R-03 | Audio Integrity | Timeline drift, duplication, or sample count mismatch. | Low | Critical | Integer sample indexing, content-conservation stitch tests, post-assembly invariants, sample-exact encode verified against the published file. |
| R-04 | Provenance | Core drift or unlicensed components entering the runtime. | Low | High | Params-hash lockfile verified against the implementation by `audit-models` (which exits non-zero on mismatch), license allowlist, provenance integrity tests that reject unverifiable digests. |
| R-05 | Data Loss | Interrupted job corrupting or overwriting source audio. | Low | Critical | Source never opened for writing; destination-filesystem staging with rollback; workspace removed only after successful publication. |
| R-06 | Honesty | Documentation or reports overstating capability. | Medium | High | Placeholder-free report tests, measured-only calibration metrics, provenance tests that fail on fabricated digests, and this register naming the guard's scope limit explicitly. |
