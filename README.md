# HawaVoClean v1

> Offline dialogue audio cleanup: Wiener spectral denoising, spectral-change
> guarded finishing, and BS.1770 loudness mastering — engineered to never
> damage the source material.

## What this is

HawaVoClean takes a noisy dialogue or podcast recording and produces a
mastered WAV: steady-state noise reduced, hum and clicks repaired, speech
EQ'd and de-essed, loudness normalized to broadcast targets, true peaks
limited to −1.0 dBTP. Every processing decision is recorded in an audit
report published beside the output.

**Three enhancement cores, honestly labelled.** The default production core
is a decision-directed spectral Wiener filter — classical DSP, gentle,
guarded strictly. The optional **studio profile** uses a real neural model:
WPE dereverberation + DeepFilterNet3 (MIT-licensed, weights vendored and
hash-locked, verified by `hawavoclean audit-models`). Measured on a real
94.6 s recording: noise floor −49.9 → −76.9 dBFS, SNR +26.7 dB, signal
level preserved within 0.3 dB. DeepFilterNet3 is a *speech* enhancer —
sustained musical tones may be attenuated as noise; review the flagged
timecodes in the report for music-heavy material.

The **low-band profile** is for the case where both of those fail: a
muffled recording whose noise is low-frequency tonal rumble. The Wiener
filter cannot get under the rumble without hitting its gain floor, and
full-band DeepFilterNet3 takes the consonants with it (measured on a real
24 s recording: 0.22 of the original 2–8 kHz energy retained, and the guard
reverted the whole take). This core runs DFN3 over the full band but keeps
only its output *below a 1000 Hz crossover*, handing the original back
above it — so consonants survive by construction, not by the model's good
judgement. Measured on that same recording: speech-to-floor separation
15.1 → 29.4 dB, 60–300 Hz pause rumble down 39 dB, consonant retention
0.999, guard spectral-hole score 0.066 against its 0.100 threshold. Follow
it with a production pass (see Usage) for 35.2 dB separation.

**What it still is not.** There is no speech recognition in this system.
The fidelity guard compares *spectral signatures*: in `strict_spectral`
mode (production) it demands the output stay spectrally near-identical; in
`integrity` mode (studio) it enforces timing, envelope, artifact, and
collapse protections while allowing the spectral change that restoration
is. It cannot verify *linguistic* content — that would require a trained
Kurdish Sorani acoustic model and a human-verified corpus, and this
project has neither. Where that limits a guarantee, the guarantee is not
made.

## Invariants that hold (and are tested)

1. **Source preservation** — the input file is never opened for writing.
2. **Fail-closed passthrough** — any enhancer fault, invalid output, or
   guard rejection yields the original audio for that unit; never silence,
   never a corrupted timeline. Chaos tests kill, hang, and corrupt the
   worker to prove it.
3. **Sample-exact timeline** — output duration, sample count, channel
   layout, and unit timing equal the input's, enforced by post-assembly
   invariants and content-conservation tests.
4. **True-peak ceiling** — −1.0 dBTP, verified by 8× oversampled
   measurement inside the limiter and independently in tests. No hard
   clipping anywhere in the chain.
5. **Truthful audit trail** — the report describes the run that produced
   it. There is no resume cache; re-running a job recomputes and re-reports
   every unit identically.
6. **Verifiable provenance** — the core's parameters are hash-locked and
   `hawavoclean audit-models` fails if the lockfile and the implementation
   disagree. No digest in this repository refers to a file that does not
   exist.

## Known limitations

1. The guard detects spectral change, not linguistic change. A processing
   artifact that preserves spectral shape passes it.
2. All bundled corpora are synthesized tones, labelled as such. No Kurdish
   speech has been processed or evaluated by the maintainers.
3. The conservative guard reverts aggressively: on the bundled synthetic
   corpus roughly half of speech units keep their original audio (they
   still receive loudness normalization).
4. Processed audio is an enhanced dialogue master, not forensic evidence.

## Installation

```bash
cd hawavoclean
uv sync --locked                 # base install (Wiener core, no torch)
uv sync --locked --extra studio  # + neural studio core
```

Third-party license texts for the vendored model weights are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Usage

```bash
hawavoclean doctor
hawavoclean process interview.wav --output interview_clean.wav --profile production
hawavoclean process interview.wav --output interview_studio.wav --profile studio
hawavoclean process rumbly.wav --output rumbly_lowband.wav --profile lowband
hawavoclean batch recordings/*.m4a --output-dir cleaned/ --profile studio --suffix _studio
hawavoclean verify interview_clean.wav --report interview_clean.hawavoclean.json
hawavoclean audit-models
```

`batch` isolates failures: one bad file never aborts the rest, the summary
names every failure, and the exit code is non-zero unless every file
succeeded.

For a muffled recording sitting on low-frequency rumble, the measured best
result is the low-band profile followed by a production pass — the band
split gets out from under the rumble, the Wiener pass takes the rest of the
floor down, and the guard judges every unit in both runs:

```bash
hawavoclean process rumbly.wav --output rumbly_lowband.wav --profile lowband
hawavoclean process rumbly_lowband.wav --output rumbly_final.wav --profile production
```

Both the studio and low-band profiles need the optional neural
dependencies:

```bash
uv sync --extra studio
```

Development and evaluation commands:

```bash
hawavoclean calibrate --manifest data/calibration/manifest.json --output /tmp/calib.json
hawavoclean eval --manifest data/acceptance/manifest.json
hawavoclean benchmark --manifest data/acceptance/manifest.json --output /tmp/bench.json
```

Paths are CWD-independent: configs and model artifacts ship inside the
package and can be overridden with `HAWAVOCLEAN_CONFIG_DIR`,
`HAWAVOCLEAN_MODEL_DIR`, and `HAWAVOCLEAN_WORK_DIR`.

## License

All rights reserved. Proprietary and confidential.
