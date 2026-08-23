# 09 - Doctor and Diagnostics

## CLI Diagnostics Command
```bash
hawavoclean restore-doctor
```

The doctor is a gate, not a banner. It exits non-zero if the upstream checkout or
its licence is missing, if any of the ten profiles fails hash/consent validation,
if the F0 extractor drifts more than 5 Hz on a synthetic 150 Hz tone, if the
protected band is not preserved, or if Guard R reverts the smoke-test candidate
to Natural. The protected-band check uses exactly the tolerances Guard R uses, so
the doctor cannot fail audio the production gate would accept.

## Verification Output

Captured from a real run on the committed tree, not transcribed by hand:

```text
================================================================================
                HAWAVOCLEAN - SPECTRAL RESTORATION DOCTOR (v2.1)                
================================================================================
[OK] Upstream foundation: UniverSR (pinned commit: 26dc21c4...)
[OK] Verified licenses: UniverSR (MIT), 3D-Speaker (Apache-2.0)
[OK] Profile verified: character_01 (Kurdish Speaker 01 (Male Deep)) [median F0: 115.0 Hz, hash: 42aa4c80...]
[OK] Profile verified: character_02 (Kurdish Speaker 02 (Female Clear)) [median F0: 215.0 Hz, hash: 2dc5e897...]
[OK] Profile verified: character_03 (Kurdish Speaker 03 (Male Baritone)) [median F0: 130.0 Hz, hash: 475f26c9...]
[OK] Profile verified: character_04 (Kurdish Speaker 04 (Female Warm)) [median F0: 195.0 Hz, hash: 21681406...]
[OK] Profile verified: character_05 (Kurdish Speaker 05 (Male Tenor)) [median F0: 155.0 Hz, hash: 7dbbac52...]
[OK] Profile verified: character_06 (Kurdish Speaker 06 (Female Mezzo)) [median F0: 225.0 Hz, hash: 3212713d...]
[OK] Profile verified: character_07 (Kurdish Speaker 07 (Male Crisp)) [median F0: 140.0 Hz, hash: aab1fb43...]
[OK] Profile verified: character_08 (Kurdish Speaker 08 (Female Bright)) [median F0: 240.0 Hz, hash: 8ce49a28...]
[OK] Profile verified: character_09 (Kurdish Speaker 09 (Male Resonant)) [median F0: 125.0 Hz, hash: ca42cbd6...]
[OK] Profile verified: character_10 (Kurdish Speaker 10 (Female Dynamic)) [median F0: 205.0 Hz, hash: 86423ad1...]
[OK] F0 Extractor verified on 48 kHz (extracted 151.2 Hz)
[OK] HawaRestore-KD & Guard R verified (verdict=PASS, strength=1.00)
================================================================================
Restore-Doctor status: ALL RESTORATION CHECKS PASSED. Ready for restore mode.
```

The verdict line reports `strength=1.00`: an accepted restoration. A run that
reverted to Natural would read `strength=0.00` and the doctor would fail, since a
revert means the model produced nothing the guard was willing to ship.
