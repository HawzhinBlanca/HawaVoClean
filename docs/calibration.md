# Guard Calibration

The shipped thresholds in
`src/voiceclean/resources/models/guard-calibration.json` are engineering
defaults — chosen by inspection, not fitted. The artifact says so in its
`provenance` field, and its `calibration_id` is the canonical hash of the
thresholds themselves (verified by `voiceclean doctor` and the provenance
test suite).

To measure real guard behavior over a corpus:

```bash
voiceclean calibrate \
  --manifest data/calibration/manifest.json \
  --output /tmp/guard-calibration.measured.json \
  --corruption-profile standard
```

The command evaluates the guard against corrupted renderings (which it must
reject) and benign renderings (which it should accept), and writes
*counted* rates with a `measured` block naming the corpus digest,
corruption profile, and evaluation counts. Rates respond to the corruption
profile — `tests/unit/test_calibration_is_measured.py` proves a hardcoded
number cannot pass.

Note the corpus limitation: bundled datasets are synthetic tones. Rates
measured on them exercise the guard's spectral comparisons, not its
behavior on real Kurdish speech.
