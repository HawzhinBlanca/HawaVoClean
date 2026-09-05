# HawaVoClean v3.3

> Offline dialogue audio cleanup: Wiener spectral denoising, spectral-change
> guarded finishing, and BS.1770 loudness mastering — engineered to fail
> closed and preserve the source timeline.

## What this is

HawaVoClean takes a noisy dialogue or podcast recording and produces a
mastered WAV: steady-state noise reduced, hum and clicks repaired, speech
EQ'd and de-essed, loudness normalized to broadcast targets, true peaks
limited to −1.0 dBTP. Every processing decision is recorded in an audit
report published beside the output.

**Release state:** 3.3.0 is a release candidate in progress, not a published 10/10 release. The exact
evidence counts and open human/vendor gates are generated in
[the release-status snapshot](docs/generated-release-status.md); [STATUS.md](STATUS.md) explains the
current verdict in plain language.

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
is. It cannot verify *linguistic* content. HawaVoClean now has a locked
Sorani human-evaluation protocol and a rights-safe source design, but neither
is approved or executed and no held-out result exists. Where that limits a
guarantee, the guarantee is not made.

## Invariants that hold (and are tested)

1. **Source preservation** — the input file is never opened for writing.
2. **Fail-closed passthrough** — any enhancer fault, invalid output, or
   guard rejection yields the original audio for that unit; never silence,
   never a corrupted timeline. Chaos tests kill, hang, and corrupt the
   worker to prove it.
3. **Sample-exact timeline** — output duration, sample count, channel
   layout, and unit timing equal the input's, enforced by post-assembly
   invariants and content-conservation tests. Restore mode is the one
   documented exception: it publishes at 48 kHz, so a 44.1 kHz input keeps
   its duration and channel layout but not its sample count. The report
   records both rates.
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
7. **One authoritative output generation** — an immutable, verified
   generation and one atomic `current` record decide which WAV/report/summary
   belongs together. The visible files are ordinary, self-contained exports;
   official readers resolve and repair them from that authority after an
   interruption, and the previous generation remains recoverable.

## Known limitations

1. The guard detects spectral change, not linguistic change. A processing
   artifact that preserves spectral shape passes it.
2. The tracked acceptance corpus is synthetic. Private real Sorani recordings
   are used only as non-redistributed engineering regressions; they are not
   licensed, speaker-disjoint human acceptance evidence.
3. The conservative guard reverts aggressively: on the bundled synthetic
   corpus roughly half of speech units keep their original audio (they
   still receive loudness normalization).
4. Processed audio is an enhanced dialogue master, not forensic evidence.
5. A processed output is a three-file export backed by its adjacent hidden
   `.<output-name>.hawavoclean/` generation bundle. The visible WAV is an
   ordinary self-contained file, not a symlink. For a portable, tamper-evident
   master plus both reports, use the Full Processing Record ZIP.
6. The supported container is CPU/production only. Studio/GPU container
   support and Windows are not claimed for 3.3.0.
7. Restore mode is generative and produces audio that was never recorded. It
   is opt-in, per-speaker, consent-gated and guard-checked, but a restored
   master is a plausible reconstruction of the missing band, not a recovery of
   it. The acceptance evidence for it is synthetic; no human listening study
   is claimed.
8. The Resolve plugin's application boundary is hardened, but Resolve 21.0.3
   embeds a vendor-owned Electron 36.3.2 with unaccepted high-severity
   advisories. See [the runtime-risk assessment](docs/resolve-runtime-risk.md).

## Installation

```bash
cd hawavoclean
uv sync --frozen                 # base install (Wiener core, no torch)
uv sync --frozen --extra studio  # + neural studio and lowband cores
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
hawavoclean record create interview_clean.wav --report interview_clean.hawavoclean.json \
  --summary interview_clean.hawavoclean.txt --output interview_clean.record.zip
hawavoclean record verify interview_clean.record.zip --json
hawavoclean audit-models
```

`batch` isolates failures: one bad file never aborts the rest, the summary
names every failure, and the exit code is non-zero unless every file
succeeded.

`record create` publishes a self-contained ZIP atomically and refuses to
replace an existing destination unless `--overwrite` is explicit. `record
verify` checks the closed inventory, every internal hash, and the report/master
binding. Version 1 is an integrity-only record, not a publisher signature; both
human and `--json` output disclose `authenticated_publisher: false`.

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
uv sync --frozen --extra studio
```

### Legacy Restore research prototype (not production-qualified)

Everything above is non-generative. The source-checkout CLI still contains the
retired HawaRestore-KD research path, but it is not a production capability,
is not bundled as qualified Restore, and is refused by the server even when a
loose checkpoint/profile happens to exist. It cannot satisfy the source-based
or enrolled-speaker Restore v2 contract.

```bash
uv sync --frozen --extra restoration
hawavoclean restore-doctor
hawavoclean process telephony.wav --output restored.wav \
  --mode restore --speaker-id character_01
```

Those commands are retained only for controlled research and regression work.
They require an explicit speaker ID and checkpoint, are not eligible through
`/api/v1/capabilities`, and must never be described or distributed as the new
Restore. Production source/enrolled Restore stays blocked until a genuine
source-conditioned pack, exact release-owned qualification policy, provider
matrix, per-segment guards, and independent Sorani evaluation all pass.

Development and evaluation commands:

```bash
hawavoclean calibrate --manifest data/calibration/manifest.json --output /tmp/calib.json
hawavoclean eval --manifest data/acceptance/manifest.json
hawavoclean benchmark --manifest data/acceptance/manifest.json --output /tmp/bench.json
```

Paths are CWD-independent: configs and model artifacts ship inside the
package and can be overridden with `HAWAVOCLEAN_CONFIG_DIR`,
`HAWAVOCLEAN_MODEL_DIR`, and `HAWAVOCLEAN_WORK_DIR`.

## Web and Resolve surfaces

The browser/desktop UI uses the same engine over authenticated loopback only:

```bash
printf '%s\n' REPLACE_WITH_A_RANDOM_SECRET | \
  hawavoclean serve --host 127.0.0.1 --port 0 --token-stdin
```

The server has bounded active/terminal jobs, upload size/total/TTL limits,
startup scavenging and disk-pressure refusal; it never deletes a committed
user output. The Resolve installer consumes a self-contained, manifest-bearing
engine artifact, stages and self-tests the complete plugin, then backs up and
atomically activates it with rollback on failure. Building/installing that
artifact and running the actual in-host matrix are documented in
[the operational runbook](docs/operations.md).

## Release validation

The release claim has one local entry point:

```bash
bash scripts/run_release_checks.sh
```

It runs the complete gate twice in fresh detached checkouts and succeeds only
when the wheel, UI, non-distributable unsigned desktop-app proof, Resolve plugin, container and SBOM
identities reproduce. The pinned host requirements, every included check, retained proof
format and honest external limits are documented in
[docs/release-gate.md](docs/release-gate.md).

The last complete proof is bound to its named commit. Any later source or
documentation change makes a final rerun mandatory; a historical green proof
is never silently promoted to the current HEAD.

## License

All rights reserved. Proprietary and confidential.
