# Risk Management & Mitigation Register

| Risk ID | Category | Description | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|---|---|
| R-01 | Linguistic Integrity | Neural enhancer hallucinating or altering Sorani phonemes/words. | Medium | Critical | Guard A & B reject unverified units; fail-closed revert to original audio; zero tolerance on locked acceptance tests. |
| R-02 | Operational | Enhancement worker crash, hang, or CUDA OOM during long-form batch. | Medium | High | Isolated worker subprocess with IPC heartbeat, deadline enforcement, restart mechanism, and unit-level fallback. |
| R-03 | Audio Integrity | Timeline drift, phase cancellation, or sample count mismatch. | Low | Critical | Integer sample indexing, context trimming, GCC-PHAT delay compensation, postcondition validation. |
| R-04 | Provenance | Unauthorized model weights or non-commercial licensing contamination. | Low | High | Pinned `production-core.lock.toml` with SHA-256 weight digests, license audit tool, and strict SBOM generation. |
| R-05 | Data Loss / Corruption | Interrupted job leaving partial files or overwriting source audio. | Low | Critical | Source file read-only isolation, private scratch workspace `.voiceclean-work/`, atomic rename on publish. |
| R-06 | Privacy | Dialogue transcripts leaking in public log files or reports. | Low | Medium | Reports omit plain text transcripts by default; store token IDs and edit distances only unless debug mode requested. |
