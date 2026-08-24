# Model and enhancement-core provenance

HawaVoClean 3.3 has three registered cores. Production is classical DSP; studio and lowband share one
vendored DeepFilterNet3 checkpoint but apply it differently. `hawavoclean audit-models` and processing
preflight verify every core lock, parameter digest, weight digest, calibration digest and licence
allowlist entry before audio can be published.

## Locked inventory

| Core | Implementation | Model/algorithm | Phase coherent | Lock |
|---|---|---|---|---|
| `wiener-dd-48k-v1` | `hawavoclean.enhancement.production.WienerSpectralEnhancer` | Decision-directed spectral Wiener filter; no weights | Yes | `production-core.lock.toml` |
| `studio-dfn3-48k-v1` | `hawavoclean.enhancement.studio.StudioVoiceCore` | Single-channel WPE + DeepFilterNet3 + decay-gated late-tail suppression | No | `studio-core.lock.toml` |
| `studio-dfn3-lowband-48k-v1` | `hawavoclean.enhancement.studio_lowband.StudioLowBandCore` | Full-band DeepFilterNet3, retained below 1 kHz; original complementary high band | No | `studio-lowband-core.lock.toml` |

All locks live in `src/hawavoclean/resources/models/`. A parameter change is a new locked core
revision, not a hidden taste adjustment.

## Production core

- Algorithm: Ephraim–Malah-style a-priori SNR tracking in a phase-preserving Wiener filter.
- Weights: none.
- Parameter SHA-256: `e4eab6048ccbd3fcde5729385c5f72dfa1e87bce786f182c3962d30419960c64`.
- Expected rate: 48 kHz internally; supported inputs are converted through the declared decode path.
- Code licence: proprietary / all rights reserved.

Because this core has no learned weights, its provenance is the implementation plus the canonical
parameter digest and source/build identity in the schema-v2 report.

## Shared DeepFilterNet3 resource

- Upstream: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet).
- Selected licence: MIT; the upstream code is dual MIT/Apache-2.0.
- Runtime dependency: `deepfilternet`/DeepFilterLib 0.5.6 with PyTorch.
- Config SHA-256: `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290`.
- Checkpoint SHA-256: `23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003`.
- Vendored path: `src/hawavoclean/resources/models/deepfilternet3/`.
- Third-party notices: repository `THIRD_PARTY_LICENSES.md` and the vendored licence files.

The [DeepFilterNet3 paper](https://arxiv.org/html/2305.08227v1) reports training on the full
multilingual DNS4 corpus with PTDB-TUG and VCTK oversampled. It does not name a Central Kurdish source.
That is useful provenance but not a per-file upstream training manifest; exact fingerprint exclusion
cannot be claimed. The Sorani held-out protocol therefore records possible unknown upstream overlap
as a limitation and excludes any named or fingerprint-confirmed overlap.

`deepfilternet` 0.5.6 declares a stale `numpy<2` constraint. The project overrides that resolver pin
and directly tests the runtime on the locked NumPy stack. A compatibility shim is installed before
the old package imports the removed `torchaudio.backend.common` path. These are explicit compatibility
controls, not a claim that the upstream package was rebuilt.

## Studio core

- Core version: 1.1.0.
- Parameter SHA-256: `f20eb492ca9d39bc099382efa94f755f1157012a7efdadb165e5fca31859b1ac`.
- WPE implementation: [nara_wpe](https://github.com/fgnt/nara_wpe), MIT.
- DFN3 receives the full 48 kHz signal with unlimited attenuation; WPE and bounded late-tail
  suppression surround the neural stage.
- The core is non-phase-coherent. Accept/revert is per unit; residual blending is prohibited.

The 94.6-second real engineering reference measured noise floor −49.9 to −76.9 dBFS, an SNR proxy
increase of 26.7 dB and source level within 0.3 dB. This single recording is regression/calibration
evidence, not a Sorani population claim.

## Lowband core

- Core version: 1.0.0.
- Parameter SHA-256: `a5a207009987d8943347df930862831cf442c2dfa52d132ccc9fbb6260c2feeb`.
- DFN3 receives the same full-band input as studio. A fourth-order complementary zero-phase
  Butterworth split retains DFN3 below 1,000 Hz and the original above it.
- WPE and late-tail suppression are disabled in this profile.

The crossover is locked because it is a content-safety boundary: the engineering reference scores
0.066 against the 0.100 spectral-hole threshold at 1,000 Hz, while 1,500 Hz scores 0.103 and is
reverted. Feeding DFN3 a pre-lowpassed signal is also rejected because it moves the model out of the
full-band distribution it was trained to receive.

## Licence policy and SBOM

`license-policy.toml` permits only the exact declared licence labels. The deterministic CycloneDX 1.6
SBOM inventories the three core relationships, both weight files, Python/JavaScript/system packages,
container packages and release artifacts with hashes. A licence allowlist is a mechanical gate, not
a legal opinion; ambiguous corpus and performer rights are handled separately by the Sorani source
assessment.

## External candidates and historical correction

No unshipped enhancement model has earned a benchmark entry. A prior revision listed “evaluated” and
“disqualified” external models with fabricated commits and licence verdicts. Those entries were
removed. A future candidate earns an entry only after its exact artifact, licence, source revision,
labelled calibration split and reproducible measurements are recorded without touching held-out data.
