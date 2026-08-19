# Hawzhin VoiceClean v1

> Offline dialogue audio cleanup: Wiener spectral denoising, spectral-change
> guarded finishing, and BS.1770 loudness mastering — engineered to never
> damage the source material.

## What this is

VoiceClean takes a noisy dialogue or podcast recording and produces a
mastered WAV: steady-state noise reduced, hum and clicks repaired, speech
EQ'd and de-essed, loudness normalized to broadcast targets, true peaks
limited to −1.0 dBTP. Every processing decision is recorded in an audit
report published beside the output.

**What it is not.** There is no neural network and no speech recognition in
this system. The enhancement core is a decision-directed spectral Wiener
filter — classical DSP that can only attenuate spectral magnitude, never
synthesize content. The fidelity guard compares *spectral signatures*
between original and processed audio: it reliably detects that the spectrum
changed, and it reverts processing when it does. It cannot verify
*linguistic* content — validating that words survived processing would
require a trained Kurdish Sorani acoustic model and a human-verified
corpus, and this project has neither. Where that limits a guarantee, the
guarantee is not made.

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
   `voiceclean audit-models` fails if the lockfile and the implementation
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
git clone <your-remote-url> hawzhin-voiceclean
cd hawzhin-voiceclean
uv sync --locked
```

## Usage

```bash
voiceclean doctor
voiceclean process interview.wav --output interview_clean.wav --profile production
voiceclean verify interview_clean.wav --report interview_clean.voiceclean.json
voiceclean audit-models
```

Development and evaluation commands:

```bash
voiceclean calibrate --manifest data/calibration/manifest.json --output /tmp/calib.json
voiceclean eval --manifest data/acceptance/manifest.json
voiceclean benchmark --manifest data/acceptance/manifest.json --output /tmp/bench.json
```

Paths are CWD-independent: configs and model artifacts ship inside the
package and can be overridden with `VOICECLEAN_CONFIG_DIR`,
`VOICECLEAN_MODEL_DIR`, and `VOICECLEAN_WORK_DIR`.

## License

All rights reserved. Proprietary and confidential.
