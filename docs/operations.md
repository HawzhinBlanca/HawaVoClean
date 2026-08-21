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
| Windows | None for v3.3 | Unsupported; publication depends on POSIX symlink and replacement semantics |

Do not convert a declared target into a release claim until its required CI/in-host gate is green.

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

The three visible paths are relative aliases through the hidden bundle's single `current` pointer.
Never copy, rename, or archive a visible alias alone. To relocate without changing the output name,
archive all four paths from their parent directory and extract them together:

```bash
tar -cf interview_clean.publication.tar \
  interview_clean.wav \
  interview_clean.hawavoclean.json \
  interview_clean.hawavoclean.txt \
  .interview_clean.wav.hawavoclean
```

After extraction, run `hawavoclean verify` against the visible WAV and JSON report. Renaming a
publication is not currently an exposed operation: process again to the new destination. The
adjacent `.interview_clean.wav.hawavoclean.lock` is transient coordination state and is not part of
the archived generation.

On interruption, retry the same command and destination. Startup publication recovery treats only a
verified `current` target as authoritative, preserves the prior immutable generation, completes a
valid post-commit state forward, and refuses ambiguous legacy or unexpected-symlink states. Do not
manually delete the hidden bundle during recovery. The exact state machine is in
[publication-state-machine.md](publication-state-machine.md).

## Loopback server and web UI

Use a fresh random secret and keep the server on loopback:

```bash
HAWA_TOKEN="$(openssl rand -hex 32)"
uv run hawavoclean serve --host 127.0.0.1 --port 0 --token "$HAWA_TOKEN" --ui-dir ui/dist
```

The ready line reports the assigned port. Every `/api` request needs the token. Non-loopback binds,
empty tokens, arbitrary output paths, and unauthenticated requests fail closed.

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

## Resolve build, install, rollback

Close Resolve, use the exact release revision, and build a self-contained engine from the exact wheel:

```bash
SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)" uv build
uv run python scripts/build_resolve_engine.py \
  --wheel "$PWD/dist/hawavoclean-3.3.0-py3-none-any.whl" \
  --output "$PWD/build/resolve-engine"
resolve-plugin/install.sh --engine-bundle "$PWD/build/resolve-engine"
```

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
