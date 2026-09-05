from __future__ import annotations

import json
from pathlib import Path

from hawavoclean.eval.regression_gate import check_regression


def test_check_regression_missing_quality_metrics(tmp_path: Path) -> None:
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text("{}", encoding="utf-8")

    result = check_regression(baseline_file, {})
    assert result["passed"] is False
    assert "candidate report has no quality_metrics" in result["failures"]


def test_check_regression_skipped_metrics(tmp_path: Path) -> None:
    baseline_file = tmp_path / "baseline.json"
    baseline_file.write_text(json.dumps({"pesq_wb": {"mean": 3.5}}), encoding="utf-8")

    # Candidate has estoi but not pesq_wb, baseline has pesq_wb but not estoi
    candidate_report = {
        "quality_metrics": {
            "estoi": {"mean": 0.9},
        }
    }
    result = check_regression(baseline_file, candidate_report)
    assert result["comparisons"]["pesq_wb"]["status"] == "skipped"
    assert result["comparisons"]["pesq_wb"]["reason"] == "no candidate data"
    assert result["comparisons"]["estoi"]["status"] == "skipped"
    assert result["comparisons"]["estoi"]["reason"] == "no baseline"
    assert result["passed"] is True


def test_check_regression_pass_and_fail(tmp_path: Path) -> None:
    baseline_file = tmp_path / "baseline.json"
    baseline_data = {
        "pesq_wb": {"mean": 3.0},
        "lsd_db": {"mean": 2.0},
    }
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    # 1. Candidate within tolerance:
    candidate_pass = {
        "quality_metrics": {
            "pesq_wb": {"mean": 2.98},  # tolerance 0.05 -> pass
            "lsd_db": {"mean": 2.03},  # lower is better; tolerance 0.05 -> pass
        }
    }
    res_pass = check_regression(baseline_file, candidate_pass)
    assert res_pass["passed"] is True
    assert res_pass["comparisons"]["pesq_wb"]["status"] == "passed"
    assert res_pass["comparisons"]["lsd_db"]["status"] == "passed"

    # 2. Candidate regressed:
    candidate_fail = {
        "quality_metrics": {
            "pesq_wb": {"mean": 2.80},  # regressed by 0.2 > 0.05
            "lsd_db": {"mean": 2.50},  # regressed by 0.5 > 0.05
        }
    }
    res_fail = check_regression(baseline_file, candidate_fail)
    assert res_fail["passed"] is False
    assert len(res_fail["failures"]) == 2
    assert res_fail["comparisons"]["pesq_wb"]["status"] == "FAILED"
    assert res_fail["comparisons"]["lsd_db"]["status"] == "FAILED"
