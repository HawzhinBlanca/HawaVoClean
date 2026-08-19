"""The audit report must tell the same truth every time the same job runs.

Two consecutive runs of the same input, same config, same working directory
must produce per-unit verdicts that agree. A cached second run that flips
REVERT to PASS is a falsified audit trail.
"""

import shutil
from pathlib import Path

import pytest

from hawavoclean.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_noisy_hum.wav"


@pytest.mark.integration
def test_unit_verdicts_identical_across_reruns(tmp_path: Path) -> None:
    work = REPO / ".hawavoclean-work"
    shutil.rmtree(work, ignore_errors=True)
    try:
        reports = []
        for i in (1, 2):
            reports.append(
                run_pipeline(
                    input_path=FIXTURE,
                    output_path=tmp_path / f"run{i}.wav",
                    profile="production",
                    overwrite=True,
                )
            )
        r1, r2 = reports
        for u1, u2 in zip(r1.units, r2.units, strict=True):
            assert "Resumed from workspace cache." not in (u2.decision_reason or ""), (
                f"unit {u2.unit_id}: run 2 substituted a cache marker for the real verdict"
            )
            assert (u1.guard_a_verdict, u1.final_decision, u1.chosen_strength) == (
                u2.guard_a_verdict,
                u2.final_decision,
                u2.chosen_strength,
            ), (
                f"unit {u1.unit_id} verdict flipped between identical runs: "
                f"run1=({u1.guard_a_verdict}, {u1.final_decision}) "
                f"run2=({u2.guard_a_verdict}, {u2.final_decision})"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
