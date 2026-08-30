# Operational runbook

This runbook covers the candidate's supported operating paths, safe publication handling, loopback
server, CPU container, and Resolve installation. The current release verdict and remaining gates live
in [the generated release status](generated-release-status.md) and [STATUS.md](../STATUS.md).

## Qualification boundary

| Surface | Declared v3.3 target | Evidence state |
|---|---|---|
| Offline CLI | CPython 3.11–3.14 on macOS and Linux | Local release gate proves macOS arm64/Python 3.14 plus a fresh Python 3.11 wheel install; the complete OS/Python CI matrix remains a release gate |
| CPU container | Linux arm64, production profile, non-root, read-only root | Built, processed, verified, reproduced and scanned locally; final-commit rebuild is still required |
| Studio/lowband CLI | macOS/Linux with the `studio` extra | Exercised on the release workstation; no GPU-container claim |
| Resolve plugin | Apple-silicon macOS, Resolve Studio 21.0.3 | Installer and staged shell proven; real in-host workflow/accessibility acceptance is still open |
| Windows standalone | Unsupported in v3.3; future Windows 11 x64 target | Cross-platform publication and Job Object foundations exist, but native NTFS fault tests, offline NSIS packaging, Authenticode, DirectML/CUDA qualification and real-host QA are open |

Do not convert a declared target into a release claim until its required CI/in-host gate is green.

## Protected CI and release-runner setup

The committed GitHub workflow has one stable branch-protection context: `required`. It
succeeds only when the source contract, all eight Linux/macOS and Python 3.11–3.14 jobs, the macOS
web/desktop/Resolve-shell job, and the exact Apple-silicon release gate all succeed on the same commit. The
hosted matrix builds a wheel, installs it with hash-locked runtime dependencies in a separate virtual
environment, then runs `doctor`, `process`, and `verify` outside the source environment.

The exact release job is deliberately assigned to `[self-hosted, macOS, ARM64,
hawavoclean-release]` and the protected `release-candidate` environment. Before enabling it:

1. Resolve checkpoint U1 while retaining the private-repository boundary.
2. Create the `release-candidate` environment with at least one named reviewer. Approval must happen
   before the job is scheduled and therefore before private evidence is available to the checkout.
3. Register an ephemeral or single-purpose Apple-silicon runner under a dedicated unprivileged OS
   account. It must contain only the exact tools and Resolve/SDK identities in
   `evidence/release/toolchain-lock.json`; do not leave repository credentials or unrelated working
   copies on it.
4. Keep a read-only private evidence mirror outside the Actions workspace, preserving the relative
   paths in `evidence/release/audio-regressions.json`. Set the repository variable
   `HAWAVOCLEAN_RELEASE_EVIDENCE_ROOT` to that external root. The hydrator copies only manifest-named,
   SHA-256-matching, Git-ignored regular files and refuses symlinks, escapes, conflicting hashes, or
   pre-existing wrong bytes.
5. Require `required` on `main`, one fresh approving review, last-push approval, linear
   history, resolved conversations, administrator enforcement, and no force pushes/deletions. Protect
   `v*` tags against updates and deletion.

Every third-party action is pinned to a full commit SHA, checkout credentials are not persisted, and
the workflow has only `contents: read`. Evidence uploads fail the job when their expected path is
absent; retained artifacts are source-SHA-named and use the immutable service for 30 days (hosted)
or 90 days (full release proof).

Validate the committed design and inspect the exact non-mutating API plan with:

```bash
uv run --frozen python scripts/validate_github_governance.py
uv run --frozen python scripts/validate_github_governance.py --print-api-plan
```

The plan is not an apply command. No repository setting should change until U1 is explicitly
approved and the named environment reviewer and runner are ready. T3.2 closes only after the required
checks pass on the exact candidate; T3.3 additionally requires a disposable pull request whose
intentional failing test is shown to block merge.

## Install and preflight

From the exact release checkout:

```bash
uv sync --frozen
uv run hawavoclean doctor
uv run hawavoclean audit-models
```

Studio and lowband need the neural dependency closure:

```bash
uv sync --frozen --extra studio
```

`doctor` checks the runtime, packaged profiles, calibration integrity, and all core locks. Stop on any
warning that becomes a failure; never work around a model-hash mismatch.

## Process, batch, and verify

```bash
uv run hawavoclean process interview.wav \
  --output interview_clean.wav --profile production

uv run hawavoclean process interview.wav \
  --output interview_studio.wav --profile studio --passes auto

uv run hawavoclean batch recordings/*.m4a \
  --output-dir cleaned --profile production --suffix _clean --skip-existing

uv run hawavoclean verify interview_clean.wav \
  --report interview_clean.hawavoclean.json

uv run hawavoclean record create interview_clean.wav \
  --report interview_clean.hawavoclean.json \
  --summary interview_clean.hawavoclean.txt \
  --output interview_clean.record.zip

uv run hawavoclean record verify interview_clean.record.zip --json

# One supervised process: publish the master, then create and verify the ZIP.
uv run hawavoclean process interview.m4a -o interview_clean.wav \
  --profile production --record-bundle interview_clean.hawavoclean.zip
```

`--passes` is process-only. It accepts `1`–`4` or `auto`; auto keeps a later pass only while measured
speech/floor separation improves by at least 0.5 dB without a guard regression. Batch jobs are always
single-pass, have a 1,800-second per-file deadline by default, isolate failures, and return non-zero
when any input fails.

Exit codes are stable:

| Code | Meaning | Operator action |
|---|---|---|
| 0 | Success | Verify or review the committed output |
| 2 | Preflight, configuration, model, or known runtime failure | Correct the named prerequisite; no successful publication is claimed |
| 3 | Output validation/publication failure | Preserve the destination and retry; recovery will not discard the last complete generation |
| 4 | Invalid/unsupported input or ambiguous stereo | Correct the input or declare a supported channel treatment |

## Output generations, relocation, and recovery

A successful `interview_clean.wav` publication is one logical generation with four adjacent paths:

```text
interview_clean.wav
interview_clean.hawavoclean.json
interview_clean.hawavoclean.txt
.interview_clean.wav.hawavoclean/
```

The three visible paths are ordinary regular-file exports, while the hidden bundle's single
`current` record is the authoritative generation. A hard interruption can occur while those three
exports are being refreshed individually; broker/job artifact readers resolve and repair them from
the immutable authority before serving. The WAV itself remains self-contained, while the Full
Processing Record is the portable way to keep the master and both reports bound together.

Use `hawavoclean record create` for relocation or archival. Its ZIP contains ordinary `master.wav`,
`report.json`, `summary.txt`, and canonical `manifest.json` entries; it never depends on the hidden
generation store. The destination is created under an exclusive lock and published atomically. It
fails if the ZIP already exists unless `--overwrite` is explicit.

Run `hawavoclean record verify RECORD.zip` after copying the archive. This verifies the closed
inventory, every internal hash, and the report/master binding. It does **not** authenticate who made
the archive: version 1 has no publisher signature, and reports `authenticated_publisher: false` in
machine output. Preserve the printed archive SHA-256 in a separately trusted system when an external
identity anchor is required.

When `process --record-bundle ZIP` is used, ZIP construction runs in the same supervised child as
the render. Cancellation therefore kills rendering and record creation as one process tree. The job
does not become complete until the broker independently verifies the ZIP and binds its master,
report, and summary hashes to the authoritative committed generation. The ZIP builder hashes the
exact bytes it copies, verifies the temporary archive before atomic replacement, and preserves a
valid prior ZIP if source mutation or verification fails.

The master generation and portable ZIP are two atomic publication boundaries, not one filesystem
transaction: the master commits first. A hard kill between those boundaries can leave a valid new
master without a new ZIP. Such a job is explicitly `interrupted`/`failed`, never `completed`, and
startup will not promote it merely because the derived ZIP path contains a valid older record from a
replace operation. Retry with the intended conflict policy and a new idempotency key. This is the
strongest safe behavior until publication stores the ZIP inside the same immutable generation before
the single authority-pointer transition.

Renaming a committed publication is not currently an exposed operation: process again to the new
destination, or use a Full Processing Record when the goal is portable archival. The adjacent
`.interview_clean.wav.hawavoclean.lock` is transient coordination state and is not part of a record.

On interruption, retry the same command and destination. Startup publication recovery treats only a
verified `current` target as authoritative, preserves the prior immutable generation, completes a
valid post-commit state forward, and refuses ambiguous legacy or unexpected-symlink states. Do not
manually delete the hidden bundle during recovery. The exact state machine is in
[publication-state-machine.md](publication-state-machine.md).

## Loopback server and web UI

Use a fresh random secret and keep the server on loopback:

```bash
HAWA_TOKEN="$(openssl rand -hex 32)"
printf '%s\n' "$HAWA_TOKEN" | \
  uv run hawavoclean serve --host 127.0.0.1 --port 0 --token-stdin --ui-dir ui/dist
```

The ready line reports the assigned port. Every `/api` request needs the token. Native shells use
the one-shot stdin channel so the bootstrap secret is absent from the process command line. The
legacy `--token` argv form remains for one compatibility release. Non-loopback binds, empty tokens,
arbitrary output paths, and unauthenticated requests fail closed.

Default bounds are finite:

| Resource | Default |
|---|---:|
| Active jobs | 8 |
| Retained terminal jobs | 256 for at most 24 hours |
| One upload | 8 GiB |
| Concurrent uploads | 2 |
| Total managed uploads | 16 GiB |
| Upload retention | 24 hours |
| Free-space reserve | 512 MiB |

`HAWAVOCLEAN_MAX_UPLOAD_BYTES` may change the single-upload cap but cannot disable it. Uploads are
stored only under HawaVoClean's marker-scoped work area. Expired uploads are scavenged at startup,
job-terminal cleanup removes its managed upload, and storage pressure refuses new work. Committed user
outputs are outside this cleanup scope and are never removed by retention.

Stop the server through authenticated `POST /api/shutdown` when possible. If a shell/Resolve host
dies, parent-death watchdogs terminate engine/job descendants; a later start scavenges expired
managed uploads.

## CPU container

The release image is production/CPU only. A representative hardened invocation mirrors the release
gate: non-root image user, read-only root, bounded temporary filesystems, and an explicit input/output
mount.

```bash
docker run --rm --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --tmpfs /cache:rw,uid=10001,gid=10001,mode=0750,size=2g \
  --mount type=bind,source="$PWD/work",target=/work \
  hawavoclean:3.3.0 \
  process /work/input.wav --output /work/output.wav --profile production
```

The mounted `work` directory must exist and be writable by the container user. No studio, CUDA, or
GPU image is supported for v3.3.

## Candidate assembly, signing, and artifact-only smoke

Run the two-pass gate with `--retain-candidate-assets` on the exact eventual candidate commit. Then
assemble from that proof and its sibling `candidate-inputs` directory. A final candidate requires a
user-controlled signing key stored outside the repository and runner workspace:

```bash
uv run --frozen python scripts/release_candidate.py assemble \
  --gate-proof build/release-gate/<session>/release-gate-proof.json \
  --assets build/release-gate/<session>/candidate-inputs \
  --output build/candidates/hawavoclean-3.3.0 \
  --signing-key /secure/offline/path/release-key \
  --signer-identity release-owner
```

The output is a closed inventory: `candidate-manifest.json`, `SHA256SUMS`, its OpenSSH `sshsig`, and
the eight proof-matched files under `assets/`. The signed checksum file covers the manifest and every
asset. The manifest binds the source commit, full-gate file/canonical hashes, toolchain lock, tested
tree/image identities, two-pass release-file identities, signing namespace, and signer identity.
Unexpected files, symlinks, duplicate JSON keys, altered bytes, altered proof, wrong identity, or a
missing signature fail verification.

Create an `allowed_signers` file outside Git in OpenSSH format, restricting the key to the
`hawavoclean-release` namespace, then verify and exercise only candidate runtimes:

```bash
uv run --frozen python scripts/release_candidate.py verify \
  build/candidates/hawavoclean-3.3.0 \
  --gate-proof build/release-gate/<session>/release-gate-proof.json \
  --allowed-signers /secure/offline/path/allowed_signers

uv run --frozen python scripts/release_candidate.py smoke \
  build/candidates/hawavoclean-3.3.0 \
  --gate-proof build/release-gate/<session>/release-gate-proof.json \
  --allowed-signers /secure/offline/path/allowed_signers \
  --input tests/fixtures/sample_sorani_podcast.wav \
  --output build/candidates/hawavoclean-3.3.0-smoke.json
```

The smoke reconstructs the normalized UI, plugin and unsigned macOS app-proof trees and compares them
to the exact tested tree hashes. It directly exercises the app's embedded engine, installs the
candidate wheel with its candidate runtime lock into a fresh managed Python 3.11 environment, and
runs `doctor`, `process`, and `verify`; it also loads the candidate container and requires the exact
tested image ID before repeating the non-root/read-only flow. Its proof is written outside the
immutable candidate. Assembly without signing arguments is permitted only for local rehearsal and
is labeled `unsigned_pending_signing`; verification then requires the explicit `--allow-unsigned`
flag and does not complete T7.2.

Candidate schema 2 marks the macOS app proof as non-distributable qualification evidence. Signing
the outer checksum inventory does not turn that inner ad-hoc app into a release: Developer ID
signing, notarization/stapling and the final DMG/ZIP remain separate native gates.

## Resolve build, install, rollback

Close Resolve, use the exact release revision, and build a self-contained engine from the exact wheel:

```bash
uv build
uv run python scripts/build_resolve_engine.py \
  --wheel "$PWD/dist/hawavoclean-3.3.0-py3-none-any.whl" \
  --output "$PWD/build/resolve-engine"
resolve-plugin/install.sh --engine-bundle "$PWD/build/resolve-engine"
```

Do not prefix that build with `SOURCE_DATE_EPOCH=...`. In a Git checkout the
backend derives both source anchors from Git itself and treats explicit ones as
a cross-check, so supplying one without `HAWAVOCLEAN_SOURCE_REVISION` is refused
outright — `detached release builds require both`. The pair is for building from
an unpacked sdist, where there is no `.git` to read. This page carried the
one-variable form until 2026-08-26; it never worked.

The installer rejects a source checkout, mutable virtual environment, incomplete engine, wrong
manifest, or unlocked dependency tree. It assembles a content-addressed stage, verifies every byte,
runs `doctor`, and proves staged Electron → engine → auth → shutdown before activation.

The system plugin directory is normally root-owned. In that case the unprivileged installer prints
one exact activation command. Review that resolved command, then run only that command with `sudo`.
The activator refuses unknown targets and a running Resolve, verifies the copy, preserves the prior
owned plugin, activates by same-filesystem rename, verifies again, and restores the prior plugin on
failure/interruption. Restart Resolve, then open **Workspace → Workflow Integrations → HawaVoClean**.

For a dry assembly, use:

```bash
resolve-plugin/install.sh --engine-bundle "$PWD/build/resolve-engine" --no-install
```

Full requirements, standalone testing, and troubleshooting are in
[the Resolve plugin runbook](../resolve-plugin/README.md). The Resolve-owned Electron risk remains an
explicit release decision in [resolve-runtime-risk.md](resolve-runtime-risk.md).

## Incident triage

1. Preserve the input, output directory (including hidden bundle), command, stderr, and JSON report.
2. Run `hawavoclean doctor`, `hawavoclean audit-models`, then `hawavoclean verify` if a committed output
   exists.
3. For server failures, record `/api/health` before shutdown without recording the token; inspect the
   terminal job's structured error and confirm the job child is gone.
4. For Resolve failures, keep the installed plugin's `VERSION`, `SHA256SUMS`, `SYMLINKS`, visible error
   page, Resolve version/build, and host/runtime identity. Never include private audio or tokens.
5. Treat hash mismatch, unexpected symlink, mixed/partial legacy output, repeatable crash, missing
   rollback, or any possible content change as release-blocking. Do not repair evidence in place.

Reports intentionally exclude transcripts and speaker embeddings. Logs and support bundles should
contain paths only when needed and must never contain audio, credentials, or the loopback token.
