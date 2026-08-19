# Model Provenance & Licensing Register

## Production core: `wiener-dd-48k-v1`

- **Implementation**: `voiceclean.enhancement.production.WienerSpectralEnhancer`
- **Algorithm**: decision-directed spectral Wiener filter (Ephraim–Malah
  a-priori SNR tracking), exact phase preservation.
- **Weights**: none — this is deterministic DSP. Provenance is the
  parameter set, hash-locked in
  `src/voiceclean/resources/models/production-core.lock.toml` and verified
  against the implementation by `voiceclean audit-models` (non-zero exit on
  mismatch).
- **License**: Proprietary / All Rights Reserved (this repository).

## External candidates

None have been evaluated. An earlier revision of this document listed
"evaluated" and "disqualified" external models with commit hashes and
license verdicts; none of those evaluations had occurred and the entries
were fabricated. They were removed rather than corrected. When a candidate
is genuinely benchmarked (`voiceclean benchmark` over a labelled corpus),
it earns an entry here with its measured numbers and verifiable digests.
