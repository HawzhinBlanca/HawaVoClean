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

## Constraints and Failure Behaviour

- **Restore mode requires a speaker ID.** `--mode restore` without `--speaker-id`
  is rejected before any processing starts.
- **Restore mode is single-pass.** It cannot be combined with `--passes`, which has
  no restoration stage; the combination is refused rather than silently producing
  an un-restored master.
- **`--cutoff-hz` implies manual cutoff selection**, and `--cutoff manual` without a
  frequency is rejected. The report records `bandwidth.cutoff_mode` so a reader can
  tell a measured boundary from an asserted one.
- **A missing or unloadable checkpoint is fatal.** Restore mode refuses to run on
  untrained weights rather than publishing synthesised audio. Point
  `HAWAVOCLEAN_RESTORATION_CHECKPOINT` at the model if it lives outside the repo.
- **Restoration resamples the master to 48 kHz.** A 44.1 kHz input is published at
  48 kHz with its duration preserved; the report's input and output stanzas record
  both rates.
- **Long files are processed in blocks.** Memory is flat in duration rather than
  growing with it, so a feature-length file does not exhaust the machine.
