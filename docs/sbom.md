# Release SBOM

HawaVoClean emits a deterministic CycloneDX 1.6 inventory that binds the exact source commit,
immutable CPU image ID, wheel, source archive, UI bundle and Resolve plugin bundle. The generator
scans only a `git archive` of the named commit, so ignored caches, build output and historical test
trees cannot enter the release inventory.

## Inventory contract

- Python hashes come from `uv.lock`; pnpm and npm registry integrity values are decoded into
  CycloneDX hashes; installed Wolfi package hashes and licenses come from the exact image.
- Every APK, npm and PyPI component must have a cryptographic hash or generation fails.
- Every component has an explicit license result. `NOASSERTION` means the scanned metadata did not
  assert a license; it is not a claim that the component is permissively licensed.
- Vendored model resources and their core-lock relationships are included directly.
- Directory artifacts use `canonical-jsonl-v1`: relative paths, regular-file contents and sizes,
  executable modes, empty directories and internal symlink targets are hashed. Absolute, dangling or
  root-escaping symlinks fail generation.
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
