# Hawzhin VoiceClean v1

> Production-grade, offline dialogue audio enhancement and linguistic fidelity system for Kurdish Sorani dialogue and podcast recordings.

## Overview

Hawzhin VoiceClean transforms raw, noisy, reverberant, or low-quality Sorani speech into clean, clear, natural, mastered audio while preserving linguistic integrity, speaker identity, prosody, and phonetic nuances without generative hallucination.

### Key Architecture Principles & Invariants

1. **Source Preservation**: The input file is never modified or overwritten.
2. **Linguistic Preservation**: Any detected substitution, deletion, severe confidence loss, phonetic drift, or unverifiable speech unit triggers an immediate fail-closed revert to the original audio.
3. **Fail-Closed Passthrough**: Processing errors or uncertain verdicts produce original-audio passthrough for that unit, never silence, synthetic speech, or job corruption.
4. **Single Runtime Core**: Exactly one frozen neural enhancement core in production, isolated in a crash-safe subprocess.
5. **Two-Pass Hawzhin Sorani Fidelity Guard**: Guard A validates enhancer output against cached reference ASR; Guard B validates locally finished audio against the pre-finish accepted timeline.
6. **Sample-Accurate Timeline Continuity**: Exact duration, sample count, channel layout, and timeline are preserved.
7. **Transparent Audit Logging**: Full immutable JSON report (`OUTPUT.voiceclean.json`) and human review summary (`OUTPUT.voiceclean.txt`) recording every unit decision and timecode.
8. **Deterministic Publishing**: Atomic workspace assembly and validation before atomic rename.

## Known Limitations

1. No current enhancement model can guarantee that speech content is never altered under extreme corruption; VoiceClean reverts to original audio whenever uncertainty is detected.
2. The Hawzhin Fidelity Guard accuracy is bounded by acoustic coverage and calibration thresholds.
3. Genuinely missing or obliterated phonemes cannot be recovered with certainty; VoiceClean reverts rather than hallucinating synthetic phonemes.
4. Forensic integrity: processed audio is an enhanced dialogue master, not untouched forensic evidence.
5. Commercial deployment requires independent validation of code, model weight licenses, and dataset provenance.
6. Cross-platform bit-identity across arbitrary GPU architectures is not promised; numerical stability and policy-decision reproducibility are guaranteed under the pinned reference stack.

## Installation

```bash
# Clone the repository
git clone https://github.com/hawzhin-ai/hawzhin-voiceclean.git
cd hawzhin-voiceclean

# Install pinned dependencies via uv
uv sync --locked
```

## Quick Start

```bash
# System and environment preflight
voiceclean doctor

# Process a Sorani podcast file with production profile
voiceclean process interview.wav --output interview_clean.wav --profile production

# Verify an output file against its audit report
voiceclean verify interview_clean.wav --report interview_clean.voiceclean.json
```

## Development & Calibration Commands

```bash
voiceclean calibrate data/calibration/manifest.jsonl
voiceclean benchmark data/development/manifest.jsonl
voiceclean acceptance data/acceptance/manifest.jsonl
voiceclean audit-models
```

## License

All rights reserved. Proprietary and confidential.
