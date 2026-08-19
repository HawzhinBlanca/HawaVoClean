# Hawzhin Sorani Fidelity Guard

## Overview

The Hawzhin Sorani Fidelity Guard serves as the non-negotiable safety perimeter protecting Kurdish Sorani dialogue against generative hallucination, word substitutions, deletions, phonetic drift, and acoustic artifacts.

## Guard Check Pillars

1. **High-Confidence Token Anchors**:
   - Compares token streams via timestamp-weighted Levenshtein distance.
   - Any deletion or substitution of high-confidence tokens (`confidence >= 0.75`) causes immediate unit rejection.
   - Insufficient anchors in speech produces `UNVERIFIED` (fail-closed revert).

2. **Frame-Level CTC Posterior Preservation**:
   - Evaluates Jensen-Shannon (JS) divergence between frame-level acoustic posteriors across voiced frames.
   - Rejects units where `mean_js_div > 0.25` or `peak_js_div > 0.60`.

3. **Timing and Duration Integrity**:
   - Asserts monotonic time mapping and envelopes correlation (`r >= 0.80`).
   - Flags time-stretching or drift greater than 40ms.

4. **Acoustic Signal Integrity Detectors**:
   - **Consonant Band Retention**: Asserts >=60% energy retention in 2kHz - 8kHz band.
   - **Spectral Hole Detector**: Identifies subband wipeouts.
   - **Musical Noise Detector**: Identifies isolated spectral peaks.
   - **Clipping Detector**: Strictly forbids newly introduced hard clipping.
