# ADR 0007: Canonical Release Identity and Report Schema v2

## Context

The Python package, Python runtime, UI package, Resolve plugin and UI mock each carried an independent
version string. A report had only `schema_version = 1` and no product release identity, so a valid
report could not prove which release format produced it. Changing one string could silently produce
an internally inconsistent release.

## Decision

`src/hawavoclean/release.json` is the single authored identity. It names the product version and the
current report schema. Python reads those exact packaged bytes at runtime. The packaging manifests
are generated mirrors; `scripts/sync_release_identity.py` rejects any drift and `--write`
updates all six mirrors, including the standalone desktop package.

Schema-v2 reports embed the product, release version, report-schema version and SHA-256 of those exact
identity bytes. The strict report model rejects a missing, altered or internally inconsistent v2
identity. The v1 reader remains supported, but a v1 report is explicitly represented without release
identity because inventing the current version for an old report would be false provenance.

## Consequences

- A release bump starts with one JSON edit, followed by generated-mirror synchronization.
- UI and Resolve package versions are the product release version, not independent shell versions.
- Reformatting `release.json` changes its identity digest deliberately; the exact shipped bytes are
  part of the build identity.
- Schema v1 remains readable, but only schema v2 can claim a release identity.
- Source revision, dependency/model inventory and installed-wheel provenance remain separate Phase 4
  work; the release digest must not be described as a source-commit digest.
