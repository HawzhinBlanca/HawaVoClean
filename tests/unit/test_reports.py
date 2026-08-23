"""Unit tests for report generation, summary rendering, and schema validation."""

import tempfile
from pathlib import Path

import pytest

from hawavoclean.report.schema import (
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)
from hawavoclean.report.summary import generate_human_summary
from hawavoclean.report.writer import load_json_report, write_json_report
from tests.support.report_provenance import build, core, environment, guard


@pytest.mark.unit
def test_report_serialization_and_summary() -> None:
    rep = HawaVoCleanReport(
        release=current_release_metadata(),
        build=build(),
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
        core=core("wiener-dd-48k-v1", "wiener-dd", "a" * 64),
        guard=guard("spectral-guard", "1" * 64, "cal_1"),
        environment=environment(
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
        assert "HAWAVOCLEAN - AUDIT SUMMARY" in summary_txt
        assert "Job ID:               test_job_123" in summary_txt


@pytest.mark.unit
def test_restoration_summary_reads_real_bandwidth_keys() -> None:
    """The restoration block must render the keys the bandwidth estimate emits.

    An earlier version read ``detected_cutoff_hz`` and ``hf_snr_db``, which
    ``BandwidthEstimate.to_dict()`` never produces, so every published summary
    silently printed 0.0 Hz and 0.0 dB.
    """
    rep = HawaVoCleanReport(
        release=current_release_metadata(),
        build=build(),
        job_id="restore_job",
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
        core=core("wiener-dd-48k-v1", "wiener-dd", "a" * 64),
        guard=guard("spectral-guard", "1" * 64, "cal_1"),
        environment=environment(
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
        restoration={
            "mode": "restore",
            "speaker_id": "character_01",
            "profile_hash": "c" * 64,
            "natural_output_hash": "d" * 64,
            "bandwidth": {
                "effective_cutoff_hz": 7800.0,
                "confidence": 0.93,
                "shape": "codec_lowpass",
                "restore_recommended": True,
                "cutoff_mode": "auto",
                "evidence": {
                    "spectral_rolloff": 22.5,
                    "above_cutoff_snr_db": 61.25,
                    "stationarity": 0.1,
                    "high_band_energy_ratio_db": 61.25,
                },
            },
            "restorer": {
                "name": "hawarestore-kd",
                "commit": "26dc21c4",
                "solver": "midpoint",
                "weights_sha256": "e" * 64,
            },
            "segments": {"restored": 1, "reduced": 0, "reverted": 0, "bypassed": 0, "errors": 0},
            "guard_r": {
                "verdict": "PASS",
                "accepted_strength": 1.0,
                "reason": "Accepted strength 1.00",
            },
        },
    )

    txt = generate_human_summary(rep)
    cutoff_line = next(line for line in txt.splitlines() if line.startswith("Cutoff Frequency:"))

    # Every number on this line must come from a key the estimate really emits;
    # a missing key would silently render as 0.0.
    assert "7800.0 Hz" in cutoff_line
    assert "codec_lowpass" in cutoff_line
    assert "confidence 0.93" in cutoff_line
    assert "61.2 dB" in cutoff_line, "the SNR must come from bandwidth.evidence"
    assert "0.0 Hz" not in cutoff_line.replace("7800.0 Hz", "")
    assert "0.0 dB" not in cutoff_line.replace("61.2 dB", "")

    assert "Guard R Verdict:      PASS" in txt
    assert "eeeeeeeeeeeeeeee" in txt, "the loaded weights hash belongs in the audit summary"
