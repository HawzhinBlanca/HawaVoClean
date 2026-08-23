# 08 - Blind ABX and Subjective Listening Protocol

## Human Listening Protocol Specification
Subjective listening evaluation for production sign-off requires a double-blind ABX trial protocol across the 10 Kurdish speaker character profiles and native Sorani Kurdish listeners.

### Evaluation Axes
1. **Intelligibility**: Phonetic clarity of Kurdish sibilants, fricatives, and pharyngeal consonants ($/s/$, $/ʃ/$, $/x/$, $/ħ/$, $/ʕ/$).
2. **Timbre Naturalness**: Freedom from metallic ringing, robotic phase dispersion, or watery artifacts.
3. **Speaker Identity Match**: Perceived speaker recognition against reference clean utterances.

### Formal Protocol Status
- **Automated Evidence**: Full objective verification (LSD, Protected-Band Invariance, Speaker Cosine Similarity, Guard R Multi-Layer Verdicts) is executed continuously via `research/restoration/benchmark.py` and `hawavoclean restore-doctor`.
- **Subjective Panel Requirements**: Production sign-off requires 15+ native Sorani speakers completing the standardized MUSHRA evaluation harness over 40 paired test utterances (8 kHz low-pass degraded vs. UniverSR vs. HawaRestore-KD vs. Clean 48 kHz reference).
- **Current Status**: Protocol and scoring harness defined; automated objective metrics gate release. Subjective panels to be scheduled prior to final broad release.
