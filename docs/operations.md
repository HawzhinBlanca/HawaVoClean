# Operational Runbook

## Execution Commands

### Diagnostic Health Check
```bash
voiceclean doctor
```

### Production Processing (gentle, strict guard)
```bash
voiceclean process interview.wav --output interview_mastered.wav --profile production
```

### Studio Restoration (neural denoise + dereverb, integrity guard)
```bash
voiceclean process interview.wav --output interview_studio.wav --profile studio
```

### Batch Processing (per-file isolation, summary, non-zero exit on any failure)
```bash
voiceclean batch folder/*.m4a --output-dir cleaned/ --profile studio --suffix _studio --skip-existing
```

### Master Verification
```bash
voiceclean verify interview_mastered.wav --report interview_mastered.voiceclean.json
```

## Exit Codes

- `0`: Success, output and reports published cleanly.
- `2`: Preflight, configuration, or model lock failure.
- `3`: Publication or validation failure; candidate output discarded.
- `4`: Invalid user input format or ambiguous stereo.
