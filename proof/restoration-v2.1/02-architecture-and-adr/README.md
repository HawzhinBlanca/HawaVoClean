# 02 - Architecture and Architecture Decision Records (ADR)

## Architectural Blueprint

HawaVoClean introduces **HawaRestore-KD**, a personalized Kurdish spectral bandwidth restoration
subsystem (`src/hawavoclean/restoration/`). This page summarizes the architecture;
[ADR 0008](../../docs/adr/0008-personalized-kurdish-spectral-restoration.md) is authoritative
where the two differ.

### Key Principles

1. **Mode Separation**:
   - `Natural` mode: the default. No restoration models or weights are loaded; zero generative
     components. Existing golden outputs and deterministic behaviors are unchanged.
   - `Restore` mode: explicit opt-in (`--mode restore --speaker-id <ID>`) extending degraded
     bandwidth up to a 48 kHz target rate. Restore mode is single-pass and is refused in
     combination with `--passes`.
2. **Protected-Band Invariance**:
   - The model operates in the complex-STFT domain and predicts only the missing spectral bins
     above the detected cutoff. Below the calibrated narrow transition band the original signal
     is copied identically; numerical tests enforce float32 invariance on the protected band.
3. **Single Conditioned Backbone**:
   - Rather than 10 separate checkpoints, a single shared **flow-matching** backbone (vector-field
     network integrated by a deterministic ODE solver, based on the pinned UniverSR architecture)
     is conditioned on continuous cutoff frequency, F0/voicing trajectories, a learned 10-way
     speaker-ID embedding, and precomputed acoustic prototype vectors.
4. **Candidate High-Band Strength Ladder**:
   - Candidate strengths are `[1.00, 0.75, 0.50, 0.25, 0.00]`. The protected band is identical
     across all candidates. Only the four non-zero strengths are submitted to Guard R; the `0.00`
     entry **is** the Natural-safe candidate and is never scored — evaluating it would compare the
     Natural audio against itself and trivially pass. It is the fail-closed fallback, not a
     proposal.
5. **Multi-Layer Restoration Guard R** — each submitted candidate is evaluated through:
   - Structural integrity (sample/channel/duration conservation, clipping, NaN/Inf,
     discontinuity);
   - Protected-band invariance (waveform and complex-STFT deviation within float32 calibration
     limits);
   - Sorani token and CTC posterior consistency against the Natural-safe candidate;
   - High-frequency consonant/event consistency (generated highs must align with speech windows;
     false sibilants outside speech are rejected);
   - F0 and harmonic consistency (no octave errors in the generated high band);
   - Speaker identity against the profile's canonical prototype vector. This check uses the
     deterministic DSP embedding extractor (`src/hawavoclean/restoration/speaker_embed.py`),
     not a neural speaker model.
6. **Fail-Closed Fallback**:
   - If a candidate fails any Guard R layer, the ladder steps down to the next lower strength.
     If all active strengths fail, or any error occurs, the system falls back to the Natural-safe
     candidate and the report records verdict `FAIL` (or `ERROR`) with the rejection reason and
     the failing layer's metrics.
7. **Reproducible Inference**:
   - The vector field is integrated on CPU by default, with randomness drawn from an explicit
     `torch.Generator` seeded per block from the job ID — never the process-global RNG — so the
     restored master is identical across machines.
8. **Weights Are Mandatory**:
   - Restore mode refuses to start without a loadable checkpoint, and the reported
     `weights_sha256` is computed from the file actually loaded into the network. There is no
     untrained-weights fallback.

## ADR Reference

- [ADR 0008: Personalized Kurdish Spectral Bandwidth Restoration](../../docs/adr/0008-personalized-kurdish-spectral-restoration.md)
