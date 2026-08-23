# ADR 0008: Personalized Kurdish Spectral Restoration (`HawaRestore-KD`)

## Context

Audio recordings originating from legacy mobile networks, VoIP telephony, historic broadcasts, or aggressively compressed codecs suffer from severe high-frequency cutoff (e.g., 4 kHz, 8 kHz, or irregular codec low-pass filters). In these recordings, vocal sibilants, fricatives, and high-frequency harmonics are physically absent.

The existing Natural pipeline cores (`wiener-dd-48k-v1`, `studio-dfn3-48k-v1`, and `studio-dfn3-lowband-48k-v1`) are conservative speech enhancement and denoising systems. By design, they attenuate noise and preserve observed speech energy, but they cannot reconstruct missing spectral bandwidth.

Reconstructing missing speech frequencies requires generative bandwidth extension. However, unconstrained neural waveform generation or whole-voice resynthesis introduces unacceptable risks:
1. Hallucination or alteration of spoken Sorani Kurdish phonemes and words;
2. Mutation of trusted low-frequency vowels and formants;
3. Inconsistent speaker identity;
4. High-frequency artifacts (metallic doubling, false sibilant bursts, unnatural hiss).

## Decision

We introduce an explicit, opt-in **Personalized Restore** mode powered by **HawaRestore-KD** (`src/hawavoclean/restoration/`), governed by the following architectural invariants:

### 1. Two Independent Modes
- **Natural mode (`--mode natural`)**: Remains the default. No restoration models or weights are loaded. Existing golden outputs and deterministic behaviors remain unchanged.
- **Restore mode (`--mode restore`)**: Requires an explicit speaker profile (`--speaker-id <ID>`) and operates at a target sample rate of 48 kHz.

### 2. Protected-Band Invariance
The restoration model operates in the complex-STFT domain and predicts **only the missing spectral bins** above the detected cutoff frequency. The trusted observed spectrum is preserved by construction:
$$\text{Output Spectrum} = \text{Trusted Spectrum} \odot (1 - M_{\text{trans}}) + \text{Generated Spectrum} \odot M_{\text{trans}}$$
Below the calibrated narrow transition band, the original signal is copied identically. Numerical tests enforce float32 invariance on the protected band.

### 3. Single Conditioned Backbone for 10 Speakers
Rather than maintaining ten separate models, we train a single shared HawaRestore-KD backbone based on the pinned UniverSR architecture (MIT license), conditioned with:
- Continuous cutoff frequency / missing band mask;
- F0 and voiced/unvoiced (V/UV) trajectories;
- Learned 10-way speaker-ID embedding;
- Precomputed canonical clean-audio prototype vectors.

### 4. Candidate High-Band Strength Ladder
Restoration generates candidate high-band residual strengths: `[1.00, 0.75, 0.50, 0.25, 0.00]`. The protected band is identical across all candidates.

### 5. Multi-Layer Restoration Guard R
Each candidate is evaluated through Guard R:
- **Structural Integrity**: Sample count conservation, channel count, duration, clipping, NaN/Inf, discontinuity.
- **Protected-Band Invariance**: Waveform and complex-STFT deviation below the cutoff must not exceed float32 calibration limits.
- **Sorani Token & CTC Consistency**: Token anchors and posterior alignments from the Natural-safe candidate must be preserved.
- **High-Frequency Consonant / Event Consistency**: Generated high frequencies must align with speech windows and expected 3–8 kHz envelope energy; false sibilants outside speech are rejected.
- **F0 & Harmonic Consistency**: High-band voiced harmonics must follow measured pitch without octave errors.
- **Speaker Identity**: Restored segments must match the speaker's canonical prototype vector.

### 6. Fail-Closed Fallback
If any candidate fails Guard R, the policy evaluates the next lower strength. If all active strengths fail or an error occurs, the system automatically falls back to the **Natural-safe candidate**.

## Consequences

- Healthy, full-band audio is protected against false restoration.
- Trusted speech below the cutoff frequency cannot be altered.
- Kurdish phonemic integrity is verified prior to and after restoration.
- All decisions, cutoff measurements, model hashes, and guard scores are recorded in the immutable `.hawavoclean.json` report.
