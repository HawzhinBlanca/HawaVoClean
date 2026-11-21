"""Benchmark harness: measured statistics for the production profile.

Runs the actual production pipeline over a corpus and reports counted
outcomes. When the corpus manifest includes ``clean_path`` references,
the C1 research metrics (PESQ, ESTOI, SI-SNR, LSD) are also computed
and included in the report.
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
    compute_quality_metrics: bool = True,
) -> dict[str, Any]:
    """Run the production pipeline over the corpus and report measured stats.

    When ``compute_quality_metrics`` is True and the corpus manifest includes
    ``clean_path`` fields, standard SE quality metrics are computed for each
    item (PESQ, ESTOI, SI-SNR, LSD).
    """
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

    # C2: Try to import metrics module for quality measurement
    metrics_fn = None
    if compute_quality_metrics:
        try:
            from hawavoclean.eval.metrics import compute_metrics

            metrics_fn = compute_metrics
        except ImportError:
            logger.warning("Metrics module unavailable; skipping quality metrics")

    metrics_results: list[dict[str, Any]] = []

    for item in manifest.items:
        t0 = time.perf_counter()
        out_wav = audio_dir / f"{item.id}_bench.wav"
        report = run_pipeline(
            input_path=Path(item.audio_path),
            output_path=out_wav,
            profile="production",
            overwrite=True,
        )
        elapsed = time.perf_counter() - t0

        units_total += report.summary.units_total
        units_enhanced += report.summary.enhanced
        audio_seconds += report.input.duration_s
        wall_seconds += elapsed

        item_result: dict[str, Any] = {
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

        # C2: Compute quality metrics when reference is available
        clean_path = getattr(item, "clean_path", None)
        if metrics_fn is not None and clean_path is not None and Path(clean_path).exists():
            try:
                m = metrics_fn(clean_path, out_wav)
                item_result["metrics"] = {
                    "pesq_wb": m.pesq_wb,
                    "estoi": m.estoi,
                    "si_snr_db": m.si_snr_db,
                    "lsd_db": m.lsd_db,
                    "separation_db": m.separation_db,
                }
                metrics_results.append(item_result["metrics"])
            except Exception as e:
                logger.warning(f"Metrics failed for {item.id}: {e}")
                item_result["metrics"] = {"error": str(e)}

        per_item.append(item_result)

    # C2: Aggregate quality metrics
    import numpy as np

    def _aggregate(vals: list[float]) -> dict[str, float] | None:
        if not vals:
            return None
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "n": len(vals),
        }

    quality_aggregate: dict[str, Any] = {}
    if metrics_results:
        for key in ("pesq_wb", "estoi", "si_snr_db", "lsd_db", "separation_db"):
            vals = [m[key] for m in metrics_results if m.get(key) is not None]
            quality_aggregate[key] = _aggregate(vals)

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
        "quality_metrics": quality_aggregate if quality_aggregate else None,
        "per_item": per_item,
    }

    out = Path(output_report_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    return benchmark_data
