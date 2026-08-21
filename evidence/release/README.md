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
