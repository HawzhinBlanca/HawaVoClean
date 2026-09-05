"""C4 · Per-release metrics regression gate.

Compares a current benchmark report against a locked baseline report and
fails if any metric regresses beyond its allowed tolerance. The gate is
intended to run in CI/CD on every release candidate.

The baseline is a JSON file with the same structure as the benchmark report's
``quality_metrics`` field. It is locked into the repo and only updated when
the team explicitly approves a new baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hawavoclean.logging import get_logger

logger = get_logger("regression_gate")

#: Per-metric regression tolerance. Metrics where "higher is better" use a
#: negative tolerance (the candidate can be worse by at most this much).
#: Metrics where "lower is better" use a positive tolerance.
REGRESSION_THRESHOLDS: dict[str, dict[str, float]] = {
    # Higher is better — allow at most 0.05 regression
    "pesq_wb": {"direction": 1.0, "tolerance": 0.05},
    "estoi": {"direction": 1.0, "tolerance": 0.02},
    "si_snr_db": {"direction": 1.0, "tolerance": 0.5},
    "separation_db": {"direction": 1.0, "tolerance": 0.5},
    # Lower is better — allow at most 0.05 increase
    "lsd_db": {"direction": -1.0, "tolerance": 0.05},
}


def check_regression(
    baseline_path: Path | str,
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    """Check a candidate benchmark report against a locked baseline.

    Returns a dict with ``passed``, ``failures`` (list of strings), and
    ``comparisons`` (per-metric details).
    """
    baseline_data = json.loads(Path(baseline_path).read_text())
    candidate_metrics = candidate_report.get("quality_metrics", {})

    if not candidate_metrics:
        return {
            "passed": False,
            "failures": ["candidate report has no quality_metrics"],
            "comparisons": {},
        }

    failures: list[str] = []
    comparisons: dict[str, Any] = {}

    for metric, config in REGRESSION_THRESHOLDS.items():
        baseline_stats = baseline_data.get(metric)
        candidate_stats = candidate_metrics.get(metric)

        if baseline_stats is None:
            comparisons[metric] = {"status": "skipped", "reason": "no baseline"}
            continue
        if candidate_stats is None:
            comparisons[metric] = {"status": "skipped", "reason": "no candidate data"}
            continue

        baseline_mean = baseline_stats["mean"]
        candidate_mean = candidate_stats["mean"]
        direction = config["direction"]
        tolerance = config["tolerance"]

        # direction=1: higher is better, delta = candidate - baseline > 0 is good
        # direction=-1: lower is better, delta = baseline - candidate > 0 is good
        delta = (candidate_mean - baseline_mean) * direction

        passed = delta >= -tolerance
        comparisons[metric] = {
            "status": "passed" if passed else "FAILED",
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "delta": candidate_mean - baseline_mean,
            "direction": "higher_is_better" if direction > 0 else "lower_is_better",
            "tolerance": tolerance,
        }

        if not passed:
            failures.append(
                f"{metric} regressed: {baseline_mean:.4f} → {candidate_mean:.4f} "
                f"(delta {candidate_mean - baseline_mean:+.4f}, "
                f"tolerance {tolerance:.4f})"
            )

    gate_passed = len(failures) == 0
    result = {
        "passed": gate_passed,
        "failures": failures,
        "comparisons": comparisons,
    }

    if gate_passed:
        logger.info("Regression gate PASSED — no metric regressions detected")
    else:
        for f in failures:
            logger.warning("Regression gate FAILED: %s", f)

    return result
