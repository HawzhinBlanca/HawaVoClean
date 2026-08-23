# 03 - Speaker Profiles and Consent Records

## 10 Character Speaker Profiles

The repository provides 10 validated Kurdish character profiles in `profiles/`.
Every figure below is read from the committed `profile.json` and `consent.json`
files, so the table cannot drift from the artifacts it describes.

| Speaker ID | Character Name | Pitch Range (p05 - p95) | Median F0 | Consent |
|---|---|---|---|---|
| `character_01` | Kurdish Speaker 01 (Male Deep) | 85 - 155 Hz | 115.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_02` | Kurdish Speaker 02 (Female Clear) | 175 - 275 Hz | 215.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_03` | Kurdish Speaker 03 (Male Baritone) | 95 - 175 Hz | 130.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_04` | Kurdish Speaker 04 (Female Warm) | 155 - 255 Hz | 195.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_05` | Kurdish Speaker 05 (Male Tenor) | 115 - 205 Hz | 155.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_06` | Kurdish Speaker 06 (Female Mezzo) | 185 - 285 Hz | 225.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_07` | Kurdish Speaker 07 (Male Crisp) | 105 - 190 Hz | 140.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_08` | Kurdish Speaker 08 (Female Bright) | 195 - 305 Hz | 240.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_09` | Kurdish Speaker 09 (Male Resonant) | 90 - 165 Hz | 125.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |
| `character_10` | Kurdish Speaker 10 (Female Dynamic) | 165 - 265 Hz | 205.0 Hz | commercial_dialogue_restoration (KRI, 2026-08-22) |

## Structure of each Profile
Each folder `profiles/character_XX/` contains:
1. `profile.json`: Metadata conforming to schema `profiles/schema.json` v1.0.
2. `consent/consent.json`: Consent record, jurisdiction, and authorization date.
3. `canonical/canonical.jsonl`: Reference manifest of clean Kurdish speech recordings.
4. `embedding/profile.npy`: 192-dimensional acoustic prototype embedding.

Validation is not advisory. `validate_speaker_profile` refuses a profile whose
embedding hash does not recompute, whose consent record is missing or not
granted, whose canonical manifest is empty, or whose embedding is degenerate.

## Validation CLI
```bash
hawavoclean speaker-profile validate profiles/
```
Output (exit code 0):
```text
[OK] Speaker profile valid: character_01 (Kurdish Speaker 01 (Male Deep))
[OK] Speaker profile valid: character_02 (Kurdish Speaker 02 (Female Clear))
[OK] Speaker profile valid: character_03 (Kurdish Speaker 03 (Male Baritone))
[OK] Speaker profile valid: character_04 (Kurdish Speaker 04 (Female Warm))
[OK] Speaker profile valid: character_05 (Kurdish Speaker 05 (Male Tenor))
[OK] Speaker profile valid: character_06 (Kurdish Speaker 06 (Female Mezzo))
[OK] Speaker profile valid: character_07 (Kurdish Speaker 07 (Male Crisp))
[OK] Speaker profile valid: character_08 (Kurdish Speaker 08 (Female Bright))
[OK] Speaker profile valid: character_09 (Kurdish Speaker 09 (Male Resonant))
[OK] Speaker profile valid: character_10 (Kurdish Speaker 10 (Female Dynamic))
[OK] 10 speaker profile(s) validated.
```

A directory containing no profiles exits non-zero rather than reporting success,
so a mistyped path cannot pass this gate silently.
