# Model Provenance & Licensing Register

## Production core: `wiener-dd-48k-v1`

- **Implementation**: `hawavoclean.enhancement.production.WienerSpectralEnhancer`
- **Algorithm**: decision-directed spectral Wiener filter (Ephraim–Malah
  a-priori SNR tracking), exact phase preservation.
- **Weights**: none — this is deterministic DSP. Provenance is the
  parameter set, hash-locked in
  `src/hawavoclean/resources/models/production-core.lock.toml` and verified
  against the implementation by `hawavoclean audit-models` (non-zero exit on
  mismatch).
- **License**: Proprietary / All Rights Reserved (this repository).

## Studio core: `studio-dfn3-48k-v1`

- **Implementation**: `hawavoclean.enhancement.studio.StudioVoiceCore`
- **Model**: DeepFilterNet3 (https://github.com/Rikorose/DeepFilterNet),
  license MIT — weights vendored at
  `src/hawavoclean/resources/models/deepfilternet3/` and hash-locked in
  `studio-core.lock.toml`; digests verified by `audit-models` and preflight.
- **Dereverberation**: single-channel WPE (nara_wpe, MIT).
- **Requires**: `uv sync --extra studio` (torch; deepfilternet's stale
  numpy<2 pin is overridden — its runtime is verified on numpy 2 by the
  studio test suite).
- **Measured** (real 94.6 s recording, 2026-08-19): noise floor −49.9 →
  −76.9 dBFS, SNR proxy +26.7 dB, signal level within 0.3 dB, reverb tail
  −1.0 dB at 100–300 ms after offsets (single-channel WPE is modest).

## External candidates

None have been evaluated. An earlier revision of this document listed
"evaluated" and "disqualified" external models with commit hashes and
license verdicts; none of those evaluations had occurred and the entries
were fabricated. They were removed rather than corrected. When a candidate
is genuinely benchmarked (`hawavoclean benchmark` over a labelled corpus),
it earns an entry here with its measured numbers and verifiable digests.
