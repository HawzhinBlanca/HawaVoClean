# 02 - Architecture and Architecture Decision Records (ADR)

## Architectural Blueprint
HawaVoClean v2.1 introduces **HawaRestore-KD**, a personalized Kurdish spectral bandwidth restoration subsystem.

### Key Principles
1. **Mode Separation**:
   - `Natural` mode: Zero generative components; non-speech reduction, spectral-change guarded finishing, and BS.1770 mastering. Bit-identical to baseline HawaVoClean.
   - `Restore` mode: Explicit opt-in (`--mode restore --speaker-id <ID>`) extending degraded bandwidths up to 48 kHz.
2. **Protected-Band Invariance**:
   - The observed spectrum below the transition band $f < f_{\text{cutoff}} - \Delta f / 2$ is strictly preserved with zero modification.
3. **Single Backbone Model**:
   - Rather than 10 separate checkpoints, a single shared diffusion/flow backbone model is conditioned on continuous cutoff frequency, F0 trajectory, speaker ID embeddings, and acoustic prototype vectors.
4. **Restoration Guard (Guard R)**:
   - Evaluates a 5-step candidate ladder `[1.0, 0.75, 0.5, 0.25, 0.0]`. If any safety layer (protected band invariance, CTC posterior divergence, highband energy/continuity, pitch divergence, or speaker similarity) fails, the ladder steps down, ultimately failing closed to the Natural-safe candidate (strength 0.0).

## ADR Reference
- [ADR 0008: Personalized Kurdish Spectral Bandwidth Restoration](../../docs/adr/0008-personalized-kurdish-spectral-restoration.md)
