"""Tests for C4 — per-release metrics regression gate."""

import json
from pathlib import Path

import pytest

from hawavoclean.eval.regression_gate import REGRESSION_THRESHOLDS, check_regression


@pytest.fixture()
def baseline_path(tmp_path: Path) -> Path:
    """Write a baseline JSON with realistic metrics."""
    baseline = {
        "pesq_wb": {"mean": 2.8, "std": 0.3, "min": 2.0, "max": 3.5, "n": 10},
        "estoi": {"mean": 0.85, "std": 0.05, "min": 0.7, "max": 0.95, "n": 10},
        "si_snr_db": {"mean": 15.0, "std": 2.0, "min": 10.0, "max": 20.0, "n": 10},
        "lsd_db": {"mean": 0.8, "std": 0.1, "min": 0.5, "max": 1.0, "n": 10},
        "separation_db": {"mean": 20.0, "std": 3.0, "min": 15.0, "max": 25.0, "n": 10},
    }
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(baseline, indent=2))
    return p


class TestRegressionGate:
    """C4 regression gate checks."""

    def test_no_regression_passes(self, baseline_path: Path) -> None:
        """A candidate that improves on every metric should pass."""
        candidate = {
            "quality_metrics": {
                "pesq_wb": {"mean": 3.0},
                "estoi": {"mean": 0.87},
                "si_snr_db": {"mean": 16.0},
                "lsd_db": {"mean": 0.7},
                "separation_db": {"mean": 21.0},
            }
        }
        result = check_regression(baseline_path, candidate)
        assert result["passed"] is True
        assert len(result["failures"]) == 0

    def test_slight_regression_within_tolerance(self, baseline_path: Path) -> None:
        """Regression within tolerance should still pass."""
        candidate = {
            "quality_metrics": {
                "pesq_wb": {"mean": 2.76},  # -0.04, tolerance is 0.05
                "estoi": {"mean": 0.84},  # -0.01, tolerance is 0.02
                "si_snr_db": {"mean": 14.6},  # -0.4, tolerance is 0.5
                "lsd_db": {"mean": 0.84},  # +0.04, tolerance is 0.05
                "separation_db": {"mean": 19.6},  # -0.4, tolerance is 0.5
            }
        }
        result = check_regression(baseline_path, candidate)
        assert result["passed"] is True

    def test_regression_beyond_tolerance_fails(self, baseline_path: Path) -> None:
        """Regression beyond tolerance should fail."""
        candidate = {
            "quality_metrics": {
                "pesq_wb": {"mean": 2.5},  # -0.3, well beyond 0.05 tolerance
                "estoi": {"mean": 0.85},
                "si_snr_db": {"mean": 15.0},
                "lsd_db": {"mean": 0.8},
                "separation_db": {"mean": 20.0},
            }
        }
        result = check_regression(baseline_path, candidate)
        assert result["passed"] is False
        assert any("pesq_wb" in f for f in result["failures"])

    def test_lsd_increase_beyond_tolerance_fails(self, baseline_path: Path) -> None:
        """LSD is lower-is-better, so increase beyond tolerance should fail."""
        candidate = {
            "quality_metrics": {
                "pesq_wb": {"mean": 2.8},
                "estoi": {"mean": 0.85},
                "si_snr_db": {"mean": 15.0},
                "lsd_db": {"mean": 1.0},  # +0.2, beyond 0.05 tolerance
                "separation_db": {"mean": 20.0},
            }
        }
        result = check_regression(baseline_path, candidate)
        assert result["passed"] is False
        assert any("lsd_db" in f for f in result["failures"])

    def test_no_quality_metrics_fails(self, baseline_path: Path) -> None:
        """Missing quality_metrics should fail."""
        result = check_regression(baseline_path, {"quality_metrics": None})
        assert result["passed"] is False

    def test_missing_metric_skipped_not_failed(self, baseline_path: Path) -> None:
        """A missing metric in the candidate should be skipped, not failed."""
        candidate = {
            "quality_metrics": {
                "pesq_wb": {"mean": 3.0},
                # other metrics missing
            }
        }
        result = check_regression(baseline_path, candidate)
        # Only PESQ was compared; others should be skipped
        skipped = [k for k, v in result["comparisons"].items() if v.get("status") == "skipped"]
        assert len(skipped) >= 3

    def test_thresholds_are_positive(self) -> None:
        """All tolerance values should be positive."""
        for metric, config in REGRESSION_THRESHOLDS.items():
            assert config["tolerance"] > 0, f"{metric} has non-positive tolerance"
