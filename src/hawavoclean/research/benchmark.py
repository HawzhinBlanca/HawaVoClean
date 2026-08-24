"""Benchmark harness: measured statistics for the production profile.

Runs the actual production pipeline over a corpus and reports counted
outcomes. It is not a three-profile comparison or evidence for unshipped
external candidates.
"""

import json
import time
from pathlib import Path
from typing import Any

from hawavoclean.eval.corpus import load_corpus_manifest
from hawavoclean.logging import get_logger
from hawavoclean.pipeline import run_pipeline

logger = get_logger("benchmark")


def run_benchmark(
    manifest_path: Path | str,
    output_report_path: Path | str = "benchmark_results.json",
    output_audio_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the production pipeline over the corpus and report measured stats."""
    manifest = load_corpus_manifest(manifest_path)
    logger.info(f"Benchmarking over {manifest.items_count} items from {manifest_path}")

    audio_dir = (
        Path(output_audio_dir).resolve()
        if output_audio_dir is not None
        else Path(output_report_path).resolve().parent / "benchmark_audio"
    )
    audio_dir.mkdir(parents=True, exist_ok=True)

    per_item: list[dict[str, Any]] = []
    units_total = 0
    units_enhanced = 0
    audio_seconds = 0.0
    wall_seconds = 0.0

    for item in manifest.items:
        t0 = time.perf_counter()
        report = run_pipeline(
            input_path=Path(item.audio_path),
            output_path=audio_dir / f"{item.id}_bench.wav",
            profile="production",
            overwrite=True,
        )
        elapsed = time.perf_counter() - t0

        units_total += report.summary.units_total
        units_enhanced += report.summary.enhanced
        audio_seconds += report.input.duration_s
        wall_seconds += elapsed

        per_item.append(
            {
                "id": item.id,
                "duration_s": report.input.duration_s,
                "wall_s": elapsed,
                "units_total": report.summary.units_total,
                "enhanced": report.summary.enhanced,
                "reverted": report.summary.reverted,
                "unverified": report.summary.unverified,
                "true_peak_dbtp": report.output.true_peak_dbtp,
                "integrated_lufs": report.output.integrated_lufs,
            }
        )

    benchmark_data = {
        "manifest_sha256": manifest.manifest_sha256,
        "items_evaluated": manifest.items_count,
        "core_id": per_item and "wiener-dd-48k-v1" or None,
        "measured": {
            "units_total": units_total,
            "units_enhanced": units_enhanced,
            "enhanced_fraction": units_enhanced / units_total if units_total else 0.0,
            "real_time_factor": wall_seconds / audio_seconds if audio_seconds else 0.0,
        },
        "per_item": per_item,
    }

    out = Path(output_report_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    return benchmark_data
