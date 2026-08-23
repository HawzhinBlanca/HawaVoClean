# 11 - User and Operator Guide

## Basic Usage

### 1. Natural Mode (Default)
Runs standard non-generative cleanup, safe finishing, and BS.1770 mastering:
```bash
hawavoclean process input.wav --output output_clean.wav
```

### 2. Restore Mode (Opt-in)
Runs personalized Kurdish spectral bandwidth extension up to 48 kHz:
```bash
hawavoclean process degraded_kurdish.wav \
  --output restored_master.wav \
  --mode restore \
  --speaker-id character_01
```

### 3. Explicit Cutoff Frequency
Optionally provide manual cutoff override (e.g. 7.5 kHz telephony cutoff):
```bash
hawavoclean process telephony_input.wav \
  --output restored_master.wav \
  --mode restore \
  --speaker-id character_02 \
  --cutoff-hz 7500.0
```

### 4. Validating Profiles
```bash
hawavoclean speaker-profile validate profiles/character_01/profile.json
```

### 5. Running Preflight Diagnostics
```bash
hawavoclean restore-doctor
```
