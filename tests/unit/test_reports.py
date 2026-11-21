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


@pytest.mark.unit
def test_smart_safe_decision_summary_en() -> None:
    """Smart Safe plain-language decision report in English exposes all required I3.8 fields."""
    rep = HawaVoCleanReport(
        release=current_release_metadata(),
        build=build(),
        job_id="smart_safe_job_1",
        config_hash="f" * 64,
        input=MediaStats(
            path="input.wav",
            sha256="111" * 21 + "1",
            sample_rate=48000,
            channels=1,
            samples=480000,
            duration_s=10.0,
        ),
        output=MediaStats(
            path="output.wav",
            sha256="222" * 21 + "2",
            sample_rate=48000,
            channels=1,
            samples=480000,
            duration_s=10.0,
            integrated_lufs=-16.0,
            true_peak_dbtp=-1.0,
        ),
        core=core("wiener-dd-48k-v1", "wiener-dd", "a" * 64),
        guard=guard("spectral-guard", "1" * 64, "cal_1"),
        environment=environment(
            platform="darwin",
            os_version="15.0",
            python_version="3.11.0",
            numpy_version="1.26.0",
            scipy_version="1.12.0",
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
            "mode": "smart_safe",
            "selected_route": "production",
            "confidence": 0.942,
            "abstained": False,
            "reason": "Highest quality predicted among eligible candidates",
            "ranker_version": "smart-safe-baseline-v1",
            "ranker_sha256": "333" * 21 + "3",
            "decision_sha256": "444" * 21 + "4",
            "acoustic_evidence": {
                "speech_probability": 0.98,
                "music_probability": 0.02,
                "crosstalk_probability": 0.01,
                "estimated_cutoff_hz": 18500.0,
                "cutoff_confidence": 0.95,
                "noise_floor_db": -52.4,
                "snr_db": 34.2,
                "rt60_s": 0.28,
                "clipping_ratio": 0.0,
                "channel_coherence": 0.99,
            },
            "candidates": [
                {
                    "route": "production",
                    "eligible": True,
                    "reasons": ["passed_all_guards"],
                    "rank_score": 4.35,
                    "confidence": 0.94,
                    "evidence_sha256": "555" * 21 + "5",
                },
                {
                    "route": "studio",
                    "eligible": False,
                    "reasons": ["hard_guard:linguistic_instability"],
                    "rank_score": 3.10,
                    "confidence": 0.88,
                    "evidence_sha256": "666" * 21 + "6",
                },
                {
                    "route": "restore_source",
                    "eligible": False,
                    "reasons": ["research_quarantine:restore_blocked"],
                    "rank_score": None,
                    "confidence": None,
                    "evidence_sha256": "777" * 21 + "7",
                },
            ],
        },
    )

    txt = generate_human_summary(rep, lang="en")
    assert "--- SMART SAFE DECISION REPORT ---" in txt
    assert "Selected Route:       PRODUCTION" in txt
    assert "Intervention Cost:    1 (Low / Conservative Wiener Filtering)" in txt
    assert "Selection Confidence: 94.2% (0.942)" in txt
    assert "Abstention Status:    NORMAL" in txt
    assert "Reconstruction:       DISCLOSED — Zero generative reconstruction" in txt
    assert "Decision Digest:      4444444444444444..." in txt

    # Acoustic Evidence
    assert "--- ACOUSTIC EVIDENCE & DETECTIONS ---" in txt
    assert "Speech Probability:   98.0%" in txt
    assert "Music Risk:           2.0%" in txt
    assert "Estimated Cutoff:     18500.0 Hz (confidence: 0.95)" in txt
    assert "Noise Floor:          -52.4 dB (Estimated SNR: 34.2 dB)" in txt
    assert "Reverb RT60:          0.28 s" in txt

    # Candidate Matrix
    assert "--- CANDIDATE EVALUATION MATRIX ---" in txt
    assert "[ELIGIBLE] production (Cost: 1) | MOS: 4.350 | Conf: 0.94" in txt
    assert "[REJECTED] studio (Cost: 2) | MOS: 3.100 | Conf: 0.88" in txt
    assert "Reasons: hard_guard:linguistic_instability" in txt
    assert "[REJECTED] restore_source (Cost: 3) | MOS: N/A | Conf: N/A" in txt
    assert "Reasons: research_quarantine:restore_blocked" in txt


@pytest.mark.unit
def test_smart_safe_decision_summary_ckb_and_abstained() -> None:
    """Smart Safe summary in Kurdish ('ckb') formats localized labels and abstention details."""
    rep = HawaVoCleanReport(
        release=current_release_metadata(),
        build=build(),
        job_id="smart_safe_job_ckb",
        config_hash="f" * 64,
        input=MediaStats(
            path="input.wav",
            sha256="111" * 21 + "1",
            sample_rate=48000,
            channels=1,
            samples=48000,
            duration_s=1.0,
        ),
        output=MediaStats(
            path="output.wav",
            sha256="222" * 21 + "2",
            sample_rate=48000,
            channels=1,
            samples=48000,
            duration_s=1.0,
        ),
        core=core("wiener-dd-48k-v1", "wiener-dd", "a" * 64),
        guard=guard("spectral-guard", "1" * 64, "cal_1"),
        environment=environment(
            platform="darwin",
            os_version="15.0",
            python_version="3.11.0",
            numpy_version="1.26.0",
            scipy_version="1.12.0",
            soundfile_version="0.13.0",
        ),
        summary=UnitSummary(
            units_total=1,
            enhanced=0,
            reverted=1,
            unverified=0,
            error_passthrough=0,
            no_speech=0,
            finish_applied=0,
            finish_bypassed=1,
        ),
        restoration={
            "mode": "smart_safe",
            "selected_route": "preserve",
            "confidence": 0.99,
            "abstained": True,
            "reason": "Post-master invariant verification failed: severe clipping detected",
            "ranker_version": "smart-safe-baseline-v1",
            "ranker_sha256": "aaa" * 21 + "a",
            "decision_sha256": "bbb" * 21 + "b",
            "acoustic_evidence": {
                "speech_probability": 0.85,
                "music_probability": 0.35,
                "crosstalk_probability": 0.05,
                "estimated_cutoff_hz": 8000.0,
                "cutoff_confidence": 0.90,
                "noise_floor_db": -40.0,
                "snr_db": 18.0,
                "rt60_s": 0.45,
                "clipping_ratio": 0.002,
                "channel_coherence": 0.95,
            },
            "candidates": [
                {
                    "route": "preserve",
                    "eligible": True,
                    "reasons": ["safe_fallback"],
                    "rank_score": 3.0,
                    "confidence": 1.0,
                    "evidence_sha256": "ccc" * 21 + "c",
                },
            ],
        },
    )

    txt = generate_human_summary(rep, lang="ckb")
    assert "--- ڕاپۆرتی بڕیاری زیرەکی پارێزراو (SMART SAFE DECISION REPORT) ---" in txt
    assert "ڕێگای هەڵبژێردراو:         پاراستن (Preserve)" in txt
    assert "تێچووی دەستێوەردان:        0 (None / Bit-Exact Passthrough)" in txt
    assert "دۆخی پەشیمانبوونەوە:       چالاک کرا (گەڕانەوە بۆ کەمترین دەستێوەردانی پارێزراو)" in txt
    assert "هۆکاری پەشیمانبوونەوە:     Post-master invariant verification failed" in txt
    assert (
        "ئاشکراکردنی بنیاتنانەوە:   ئاشکراکراو — دەنگەکە هیچ بنیاتنانەوەیەکی دەستکردی تێدا نییە"
        in txt
    )
    assert "--- بەڵگە و شیکارییە دەنگییەکان (ACOUSTIC EVIDENCE) ---" in txt
    assert "ئەگەری ئاخاوتن:            85.0%" in txt
    assert "مەترسیی مۆسیقا:            35.0%" in txt
    assert "کاتژمێری دەنگدانەوە (RT60):  0.45 چرکە" in txt
    assert "--- هەڵسەنگاندنی بەربژێرەکان (CANDIDATE EVALUATION MATRIX) ---" in txt
    assert "پاراستن (Preserve)" in txt
