# 05 - Model Artifacts and Lockfiles

## Lockfile Verification
HawaRestore-KD operates under strict frozen parameter and weights digests.

### Checksum Register
- **Backbone**: `hawarestore-kd`
- **Upstream UniverSR Pinned Commit**: `26dc21c44e11f9f19e823f02b0d4641dd5ea5af2`
- **Backbone Weights Location**: `models/hawarestore-kd/hawarestore_kd.pt`
- **Backbone Weights SHA-256**: `c578c870e4201bea5af0ae403f3fa481a6dc4691fcbde86cdd198cbb3559edbb`
- **Speaker Prototypes**: 192-dimensional unit-normalized acoustic prototype vectors with verified SHA-256 hashes per character in `profiles/character_XX/profile.json`.

## Immutable Audit Trail
Every output report generated in `--mode restore` embeds:
- `restoration.speaker_id`
- `restoration.profile_hash`
- `restoration.natural_output_hash`
- `restoration.bandwidth`
- `restoration.restorer` metadata (name, commit, weights_sha256, solver, steps)
- `restoration.segments` counts
- `restoration.guard_r` metrics
