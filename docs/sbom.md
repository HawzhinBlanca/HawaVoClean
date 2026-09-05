# Release SBOM

HawaVoClean emits a deterministic CycloneDX 1.6 inventory that binds the exact source commit,
immutable CPU image ID, wheel, source archive, UI bundle, complete source-bound unsigned macOS app
proof (including its embedded engine) and Resolve plugin bundle. The generator scans source components
only from a `git archive` of the named commit; explicitly supplied release artifacts are separately
hashed as complete canonical trees, so ignored caches and historical test output cannot leak into the
source inventory or masquerade as an artifact.

## Inventory contract

- Python hashes come from `uv.lock`; UI, desktop, and Resolve pnpm plus toolchain npm registry integrity values are decoded into
  CycloneDX hashes; installed Wolfi package hashes and licenses come from the exact image.
- Every APK, npm and PyPI component must have a cryptographic hash or generation fails.
- Every component has an explicit license result. `NOASSERTION` means the scanned metadata did not
  assert a license; it is not a claim that the component is permissively licensed.
- Vendored model resources and their core-lock relationships are included directly.
- Directory artifacts use `canonical-jsonl-v1`: relative paths, regular-file contents and sizes,
  executable modes, empty directories and internal symlink targets are hashed, with regular-file and
  symlink counts retained. Absolute, dangling or root-escaping symlinks fail generation.
- A supplied image tag is never trusted as identity. Docker resolves it to `sha256:…`, and its OCI
  source revision, version and creation date must match the clean Git release before scanning.
- Mutable local `RepoTag` aliases reported by Trivy are excluded from the canonical inventory; the
  immutable image ID, repository digest, layer identities and OCI labels remain recorded. Attaching
  another local name to identical image bytes therefore cannot change the SBOM.

## Verification

The committed T4.2 proof snapshot is not the final release artifact. Its checksum can be replayed with:

```bash
cd evidence/release
shasum -a 256 -c hawavoclean-3.3.0.cdx.json.sha256
```

Generation also validates against the hash-pinned upstream CycloneDX 1.6 schemas with
`check-jsonschema==0.35.0`. The proof record at `evidence/release/t4.2-sbom-proof.json` contains the
independently recomputed artifact hashes, inventory counts and retained failed attempts.

## Signing boundary

The generator creates the deterministic SBOM and checksum without possessing a release identity. The
final release gate must sign this exact byte stream using the user's authorized release identity and
verify that signature before publication. An agent-generated or repository-committed private key is
not an acceptable substitute.

The SBOM-bound desktop app is intentionally ad-hoc/unsigned qualification evidence and carries
`distribution_eligible: false`. Its presence proves the assembled app/engine bytes exercised by the
exact gate; it does not prove Developer ID signing, notarization, stapling, updater signing or a DMG.
