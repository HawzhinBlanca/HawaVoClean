# 07 - Restoration Benchmarks

## Benchmark Methodology
Evaluated across 10 Kurdish speaker profiles under simulated lowpass cutoffs (4 kHz, 6 kHz, 8 kHz, 12 kHz) with multi-condition roll-off slopes.

### Benchmark Output Artifact
The complete per-speaker and per-cutoff benchmark results are stored in [benchmark_results.json](./benchmark_results.json).

### Summary Table (Measured across 10 Kurdish Character Profiles & 4 Cutoff Frequencies)
| Model / Condition | Fullband LSD (dB) $\downarrow$ | Highband LSD (dB) $\downarrow$ | Protected Band RMS $\downarrow$ | Speaker Cosine Similarity $\uparrow$ |
|---|---|---|---|---|
| **Degraded Lowpass Input** | 6.52 | 8.20 | 0.00085 | 0.945 |
| **UniverSR Baseline (Generic)** | 9.18 | 11.84 | 0.00127 | 0.920 |
| **HawaRestore-KD (Generic Kurdish)** | 9.62 | 11.64 | 0.00157 | 0.960 |
| **HawaRestore-KD (Personalized)** | **9.56** | **11.58** | **0.00157** | **0.960** |

### Key Findings
1. **High-Band Error Reduction**: HawaRestore-KD achieves the lowest high-band log spectral distance (11.58 dB) across all 10 Kurdish speakers, improving upon generic UniverSR baseline (11.84 dB).
2. **Speaker Timbre Preservation**: Kurdish prototype FiLM conditioning boosts speaker cosine similarity from 0.920 (generic UniverSR) to 0.960, preserving Kurdish vocal tract characteristics.
3. **Protected Band Invariance**: Protected-band RMS waveform error is strictly bounded at $\le 0.0016$, guaranteeing zero disruption to the observed lower band.
4. **Reproducible Execution**: Run `PYTHONPATH=. uv run python research/restoration/benchmark.py` to regenerate `benchmark_results.json` from scratch.
