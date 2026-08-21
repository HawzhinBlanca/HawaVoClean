# Release Evidence

This directory is the machine-checkable evidence ledger for the true-10 release plan.

- `baseline.json` freezes the audited source topology, toolchain, tracked lock/fixture hashes and local
  external artifacts. Tracked artifacts are read from their recorded Git commits so later legitimate
  edits do not rewrite the baseline. Ignored dependency locks and private audio are named separately;
  they are never misrepresented as reproducible Git content.
- `evidence-entry.schema.json` defines the ledger entry contract.
- `ledger.jsonl` is append-only and SHA-256 chained. Each entry hashes its complete canonical content
  and the prior entry hash. Git history supplies the external anchor: rewriting history and recomputing
  the chain changes the committed file hash and commit.
- `t4.1-container-proof.json` binds the reproducible, non-root CPU image, exact transitive Wolfi lock,
  read-only operational probe and current vulnerability/misconfiguration results.
- `hawavoclean-3.3.0.cdx.json` and its checksum are the deterministic CycloneDX 1.6 T4.2 proof
  snapshot; `t4.2-sbom-proof.json` binds its inventory, artifact hashes and retained failed attempts.
- `t4.5-resolve-installer-proof.json` binds the transactional, relocatable Resolve package proof.
- `t4.6-resolve-runtime-proof.json` separates the audited standalone Electron lock from the exact
  signed Blackmagic-owned runtime, retains every advisory ID, and records the still-unaccepted risk.
- `t3.1-release-gate-proof.json` records the two clean-checkout full-gate passes, all ten matching
  artifact identities, test/audit counts, proof hashes and retained failed attempts.
- `sorani-evaluation-protocol.json` is the result-free, machine-validated T5.1 design lock. Its design
  digest deliberately excludes the later approval record, so explicit approval can bind the exact
  study design without rewriting it. A structurally valid draft is not an approved protocol.

Verify the committed evidence:

```bash
uv run python scripts/release_evidence.py verify
```

Also verify ignored local artifacts on the audited workstation:

```bash
uv run python scripts/release_evidence.py verify --check-external
```

Append only through the tool. It refuses an invalid existing chain and flushes the new line before
reporting success:

```bash
uv run python scripts/release_evidence.py append \
  --task-id T0.1 \
  --source-commit "$(git rev-parse HEAD)" \
  --command "uv run python scripts/release_evidence.py verify" \
  --status passed \
  --summary "Baseline and ledger verified" \
  --input evidence/release/baseline.json \
  --tool "python=$(python --version 2>&1)"
```

Do not name `ledger.jsonl` as an output: its post-append digest would be self-referential. The entry's
`entry_sha256`, printed by the tool and chained into the next entry, is the authoritative new ledger
head.

Failed and blocked results belong in the ledger too. `known_limits` must say what a passing command did
not prove. Never delete or edit an earlier line; append a correction that names the superseded task.
