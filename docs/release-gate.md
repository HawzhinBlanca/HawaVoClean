# Full release gate

`bash scripts/run_release_checks.sh` is the sole local definition of an
automatically valid HawaVoClean v3.3 release candidate. The command refuses a
dirty invoking tree, checks the pinned release host, and then performs two full
passes in separate detached Git worktrees. It does not rewrite tracked source.

Each pass runs all of the following against the same commit:

- exact Python dependency sync; formatting, linting and strict type checks;
- generated release-identity and JSON-schema drift checks, plus semantic validation of the result-free
  Sorani human-evaluation design and corpus-source locks (approval, acquisition and results remain
  separate human gates);
- the default suite with branch coverage at or above 92.49%;
- the separate fuzz suite and all declared owner-scoped mutations;
- two-run frozen real-audio regressions for production, studio and lowband;
- exact UI/plugin installs, typechecking, production build and UI tests;
- Python, UI, plugin and build-tool lock audits;
- deterministic wheel and source-archive builds plus a fresh Python 3.11 wheel
  install, doctor, production-profile process and verify exercise;
- a relocatable Resolve engine build and the real staged Electron-to-engine
  lifecycle self-test using the hash-pinned Resolve SDK bridge;
- the pinned CPU container build, exact package check, non-root read-only
  doctor plus production-profile process/verify exercise, and current high/critical vulnerability and
  configuration scans; and
- generation and validation of the artifact-bound CycloneDX 1.6 SBOM.

The gate compares the two passes' wheel, source archive, UI tree, Resolve
engine, Resolve plugin, container image, SBOM, real-audio regression record and
both independent CLI audio outputs. A difference in any promised identity is a
failure, even when both individual passes otherwise succeed.

## Pinned host contract

Exact tool versions and the Resolve host/SDK identity live in
`evidence/release/toolchain-lock.json`. Python and Node selectors are also
declared in `.python-version` and `.node-version`. Package resolution remains
bound to `uv.lock`, the two pnpm locks, npm's package lock, Docker base digests
and `docker/wolfi-packages.lock`. Security audits intentionally use the current
advisory and vulnerability databases; their tools are pinned, but the security
knowledge is not frozen stale.

The real regression recordings and historical references are local,
non-redistributed evidence. Their paths and SHA-256 identities are committed in
`evidence/release/audio-regressions.json`; the gate verifies each source before
copying only the required files into an isolated checkout.

## GitHub Actions mapping

`.github/workflows/ci.yml` invokes this exact command on a protected, dedicated Apple-silicon runner;
it does not redefine a smaller remote “release gate.” Separate hosted jobs add the complete declared
Python 3.11–3.14 × Linux/macOS support matrix and a macOS web/Resolve-shell check. The stable
`release / required` job fails unless every component succeeds on the same source commit.

Private regression files are never stored in GitHub or uploaded as evidence. After protected
environment approval, `scripts/hydrate_release_evidence.py` validates them from an external
runner-local mirror and copies only the required Git-ignored files into the checkout. The exact gate
then re-verifies the same hashes before creating either isolated pass. Full-gate logs and proof JSON
are uploaded under a source-SHA/run-attempt identity; the private audio itself remains excluded.

The machine-checked design, immutable action revisions, required branch/tag rules, and read-only API
plan live in `evidence/release/github-governance-contract.json` and
`scripts/validate_github_governance.py`. They remain a design—not remote proof—until U1, a green exact
candidate run, activated protections, and the deliberately failing pull-request exercise are
recorded.

## Proof output

Every invocation creates a new ignored directory under `build/release-gate/`.
`release-gate-proof.json` records the source commit, toolchain, external input
hashes, every command, duration, exit status and log hash, the artifact
identities from each pass, and the cross-pass comparison. The proof contains a
canonical SHA-256 over all its other fields. Failed attempts are retained in
their own timestamped directory and never replace a prior proof.

For the eventual T7.2 candidate, retain the actual distributable bytes from both passes:

```console
bash scripts/run_release_checks.sh --retain-candidate-assets
```

This adds no weaker build path. Each isolated pass exports the wheel, source archive, hash-locked
runtime requirements, normalized UI archive, normalized full Resolve-plugin archive, normalized
Linux-arm64 container archive, and CycloneDX SBOM from the artifacts it just tested. The gate compares
those release-file SHA-256 identities across both passes, retains pass 1 under `candidate-inputs/`, and
deletes the redundant passing copy. A mismatch fails the full gate. Failed exports remain in their
timestamped session for diagnosis.

Directory archives use sorted members, source-commit time, zero owner/group identities, preserved
executable modes, and validated relative symlinks. The Docker export normalizes only its outer tar
metadata and order; its content-addressed configuration and layer bytes are unchanged. The retained
container is exported by image ID so a per-pass tag cannot contaminate reproducibility.

The committed compact checkpoint is validated on every generated-status check.
To additionally prove that every committed count, artifact identity and digest
was derived from a retained raw proof and its hash-bound logs, run:

```console
uv run python scripts/validate_release_gate_checkpoint.py \
  --full-proof build/release-gate/<run>/release-gate-proof.json
```

Passing this command establishes the automated local portion of T3.1. It does
not claim the Phase 5 Sorani human evaluation, the Phase 6 in-Resolve workflow
and accessibility exercise, GitHub branch governance, release signing, or
acceptance of Blackmagic's vendor-owned Electron risk.
