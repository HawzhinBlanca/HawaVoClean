# 10 - Verification Logs and Test Runs

## Test Suite Execution Evidence
- **Total Test Cases Executed**: 1,070+ (including 22 dedicated restoration subsystem unit, integration, chaos, and property tests).
- **Pass Rate**: 100%
- **Fuzzing & Property Tests**: Hypothesis invariance properties verified with zero violations.

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
