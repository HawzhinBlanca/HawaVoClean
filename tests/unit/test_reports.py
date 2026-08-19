"""Unit tests for report generation, summary rendering, and schema validation."""

import tempfile
from pathlib import Path

import pytest

from voiceclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    MediaStats,
    UnitSummary,
    VoiceCleanReport,
)
from voiceclean.report.summary import generate_human_summary
from voiceclean.report.writer import load_json_report, write_json_report


@pytest.mark.unit
def test_report_serialization_and_summary() -> None:
    rep = VoiceCleanReport(
        job_id="test_job_123",
        config_hash="a" * 64,
        input=MediaStats(
            path="in.wav",
            sha256="aaa",
            sample_rate=48000,
            channels=1,
            samples=48000,
            duration_s=1.0,
        ),
        output=MediaStats(
            path="out.wav",
            sha256="bbb",
            sample_rate=48000,
            channels=1,
            samples=48000,
            duration_s=1.0,
            true_peak_dbtp=-1.0,
            integrated_lufs=-16.0,
        ),
        core=CoreMetadata(id="wiener-dd-48k-v1", algorithm="wiener-dd", params_hash="a" * 64),
        guard=GuardMetadata(id="spectral-guard", probe_hash="1" * 64, calibration_id="cal_1"),
        environment=EnvironmentMetadata(
            platform="darwin",
            os_version="14.0",
            python_version="3.13.0",
            numpy_version="2.0.0",
            scipy_version="1.14.0",
            soundfile_version="0.13.0",
        ),
        summary=UnitSummary(
            units_total=1,
            enhanced=1,
            reverted=0,
            unverified=0,
            error_passthrough=0,
            no_speech=0,
            finish_applied=1,
            finish_bypassed=0,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        j_path = tmp / "report.json"

        write_json_report(rep, j_path)
        assert j_path.exists()

        loaded = load_json_report(j_path)
        assert loaded.job_id == "test_job_123"

        summary_txt = generate_human_summary(rep)
        assert "HAWZHIN VOICECLEAN - AUDIT SUMMARY" in summary_txt
        assert "Job ID:               test_job_123" in summary_txt
