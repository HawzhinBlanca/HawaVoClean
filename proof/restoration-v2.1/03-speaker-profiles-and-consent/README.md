# 03 - Speaker Profiles and Consent Records

## 10 Character Speaker Profiles
The repository provides 10 validated Kurdish character profiles in `profiles/`:

| Speaker ID | Character Name | Dialect | Gender | Pitch Range | Median F0 | Consent Status |
|---|---|---|---|---|---|---|
| `character_01` | Kurdish Speaker 01 (Male Deep) | Sorani (ckb) | Male | 75 - 165 Hz | 115.0 Hz | Verified & Signed |
| `character_02` | Kurdish Speaker 02 (Female Clear) | Sorani (ckb) | Female | 145 - 295 Hz | 210.0 Hz | Verified & Signed |
| `character_03` | Kurdish Speaker 03 (Male Baritone) | Sorani (ckb) | Male | 85 - 180 Hz | 130.0 Hz | Verified & Signed |
| `character_04` | Kurdish Speaker 04 (Female Warm) | Sorani (ckb) | Female | 135 - 270 Hz | 195.0 Hz | Verified & Signed |
| `character_05` | Kurdish Speaker 05 (Male Tenor) | Sorani (ckb) | Male | 100 - 220 Hz | 155.0 Hz | Verified & Signed |
| `character_06` | Kurdish Speaker 06 (Female Mezzo) | Sorani (ckb) | Female | 150 - 300 Hz | 225.0 Hz | Verified & Signed |
| `character_07` | Kurdish Speaker 07 (Male Crisp) | Kurmanji (kmr) | Male | 90 - 200 Hz | 140.0 Hz | Verified & Signed |
| `character_08` | Kurdish Speaker 08 (Female Bright) | Kurmanji (kmr) | Female | 160 - 320 Hz | 235.0 Hz | Verified & Signed |
| `character_09` | Kurdish Speaker 09 (Male Resonant) | Sorani (ckb) | Male | 80 - 175 Hz | 125.0 Hz | Verified & Signed |
| `character_10` | Kurdish Speaker 10 (Female Dynamic) | Sorani (ckb) | Female | 140 - 280 Hz | 205.0 Hz | Verified & Signed |

## Structure of each Profile
Each folder `profiles/character_XX/` contains:
1. `profile.json`: Metadata conforms to schema `profiles/schema.json` v1.0.
2. `consent/consent.json`: Signed consent record and authorization timestamp.
3. `canonical/canonical.jsonl`: Reference manifest of clean Kurdish speech recordings.
4. `embedding/profile.npy`: 192-dimensional acoustic prototype embedding.

## Validation CLI
```bash
hawavoclean speaker-profile validate profiles/
```
Output:
```text
[OK] Speaker profile valid: character_01 (Kurdish Speaker 01 (Male Deep))
...
[OK] Speaker profile valid: character_10 (Kurdish Speaker 10 (Female Dynamic))
```
