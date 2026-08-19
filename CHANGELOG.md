# Changelog

All notable changes to the Hawzhin VoiceClean system will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-08-19

### Added — a real neural restoration core
- `StudioVoiceCore` (`studio-dfn3-48k-v1`): WPE dereverberation +
  DeepFilterNet3 speech enhancement. Weights vendored and hash-locked in
  `studio-core.lock.toml`; digests verified at preflight and by
  `audit-models`. Optional install: `uv sync --extra studio`.
- `--profile studio`: integrity-mode guarding, neural core, same mastering
  chain. Measured on a real recording: noise floor −27 dB, SNR +26.7 dB,
  signal preserved within 0.3 dB.
- Guard modes: `strict_spectral` (unchanged default) vs `integrity`
  (timing/envelope/artifact/collapse protections without spectral-identity
  gating). Studio thresholds measured against real guard scores; the
  calibration artifact records the measurement provenance.
- Core registry (`enhancement/factory.py`): every registered core carries
  its lockfile and an implementation-hash callable; preflight and audit
  verify weights digests and that lock tables reconstruct `params_hash`.

## [2.0.0] - 2026-08-19

### The honesty release

An audit on 2026-08-19 found that this codebase misrepresented itself:
the "neural enhancement core" was a classical Wiener filter with a
fabricated weights digest; the "Sorani CTC ASR" fidelity guard contained no
acoustic model; calibration metrics (including the headline 0.0
false-accept rate) were hardcoded literals; the model registry listed
evaluations that never happened; and the audit report falsified verdicts on
cached re-runs. This release removes every fabrication and fixes every
reproduced defect. It is a breaking release: names, config keys, report
schema, and artifact locations all changed to match reality.

### Changed (honesty)
- `ProductionEnhancerCore` -> `WienerSpectralEnhancer` (`wiener-dd-48k-v1`):
  named for the algorithm it implements. Provenance is now the parameter
  set, hash-locked and verified at preflight and by `audit-models`.
- `HawzhinSoraniASR` -> `SpectralSignatureProbe`; `SoraniASR` protocol ->
  `SpectralProbe`; `ASRResult` -> `ProbeResult` with `raw_signature` /
  `frame_distributions` fields. Module docstrings state plainly that the
  probe detects spectral change and is not a speech recognizer;
  `test_probe_is_not_asr.py` pins the boundary.
- `eval/calibrate.py` now MEASURES accept/revert rates over corruption
  profiles (mild/standard/severe); hardcoded metrics deleted. Artifacts
  carry measurement provenance or no metrics at all.
- `research/benchmark.py` now benchmarks the real pipeline; fabricated
  candidate scores deleted. Model registry deleted (nothing was evaluated).
- Datasets regenerated as declared-synthetic: dialect "synthetic",
  verified_by_human false, no transcripts claimed.
- torch/torchaudio removed (they were imported to print a version string).
- README, STATUS, RISKS, docs/ rewritten to describe the implemented
  system; BLUEPRINT.md marked historical.

### Fixed (audited defects, each with a red-first regression test)
- Audit falsification: resume cache deleted — every run recomputes and
  reports its own verdicts; verdicts are identical across re-runs.
- Workspace leak: scratch space removed on success; test suite fails loudly
  if pre-existing workspace state could serve cached results.
- Limiter: true peak now provably at or under the ceiling (sliding-minimum
  lookahead + slope-limited attack + verified trim); hard-clip fallback
  removed; property-tested at 8x oversampling with no tolerance.
- Continuity rule: enforced before records are built, channel-aware, and
  visible as `original_continuity` in reports.
- Stitch: boundary declick no longer renders unit heads twice
  (content-conservation tested).
- Report: real `probe_hash` and per-unit `output_sha256`; Guard B bypass
  reports the scores of the attempt it describes; placeholder strings
  banned by test.
- `audit-models` verifies params hash, license allowlist, and calibration
  integrity, and exits non-zero on tampering.
- Acceptance gates restructured: explicit conditionals (survive `python
  -O`), structured failures, can return FAILED, plus a did-something floor.
- CLI works from any directory (packaged resources + env overrides);
  publication stages on the destination filesystem with rollback.
- Chaos tests now inject real faults: SIGKILLed worker, hung worker,
  NaN/wrong-length/silent model output, ENOSPC at publish.
- `scripts/mutation_gate.py`: 12 behavior mutations must each break the
  suite.

## [1.0.0] - 2026-08-19

### Added
- Complete Master Implementation Blueprint v2.0 execution.
- High-performance audio spine with FFprobe media probing, float32 PCM decoding, and TPDF dithered encoding.
- Auto-channel classification supporting mono, dual-mono identical, and split-speaker stereo.
- Speech activity detection and speech-unit utterance grouping with context windows.
- Hawzhin Sorani Fidelity Guard with Unicode normalization, token anchors, frame-level CTC log-posterior JS divergence, timing preservation, and signal integrity detectors.
- Isolated enhancer worker architecture with crash/timeout recovery and heartbeat protocol.
- Multi-stage deterministic finishing chain (de-hum, click repair, plosive attenuation, dynamic EQ, de-esser, level riding) guarded by Guard B.
- BS.1770-4 integrated loudness normalization and look-ahead true-peak limiter with ceiling enforcement.
- Resumable job journal and atomic workspace publishing.
- Schema-validated immutable JSON reports and human-readable TXT review summaries.
- Comprehensive CLI suite: `doctor`, `process`, `verify`, `calibrate`, `benchmark`, `acceptance`, `audit-models`.
