# 10 - Verification Logs and Test Runs

## Test Suite Execution Evidence
- **Total test cases executed**: 1,084 passed, 41 deselected (1,125 collected).
- **Dedicated restoration tests**: 35, across unit, integration, chaos, and property suites.
- **Static analysis**: `ruff check` and `ruff format --check` clean; `mypy --strict`
  reports no issues across 254 source files.
- **Generated artifacts**: `scripts/generate_schemas.py --check` reports no drift.

## What the restoration tests pin

Beyond the happy path, the suite fixes the failure modes that a review of this
subsystem found, so they cannot silently return:

- A total revert reports `verdict=FAIL` with the rejecting layer's metrics, not a
  passing verdict produced by scoring the Natural fallback against itself.
- Audio longer than one block is restored in overlapping blocks, and a block seam
  introduces no sample-to-sample jump the source does not already contain.
- A missing or unloadable checkpoint raises `ModelProvenanceError` instead of
  running on the random initialisation.
- The attested `weights_sha256` is the hash of the file actually loaded, and the
  default device is CPU so the master does not depend on the host accelerator.
- The human summary renders the bandwidth keys the estimate really emits.

## Reproducibility Commands
```bash
# 1. Run full test suite
uv run pytest -q

# 2. Run restoration tests specifically
uv run pytest tests/unit/test_restoration* tests/integration/test_restoration* tests/chaos/test_restoration* tests/property/test_restoration* -v

# 3. Run restore doctor
uv run hawavoclean restore-doctor

# 4. Run speaker profile validation
uv run hawavoclean speaker-profile validate profiles/

# 5. Run restoration benchmark
uv run hawavoclean restoration-benchmark --output proof/restoration-v2.1/07-benchmarks/benchmark_results.json
```
