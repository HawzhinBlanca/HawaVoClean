"""The acceptance gate must be able to report failure — structurally, and under -O."""

from pathlib import Path
from typing import Any

import voiceclean.eval.acceptance as acceptance_mod
from voiceclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    MediaStats,
    UnitSummary,
    VoiceCleanReport,
)


def _report_violating_sample_count() -> VoiceCleanReport:
    return VoiceCleanReport(
        job_id="testjob",
        config_hash="c" * 64,
        input=MediaStats(
            path="in.wav",
            sha256="a" * 64,
            samples=48000,
            sample_rate=48000,
            channels=1,
            duration_s=1.0,
        ),
        output=MediaStats(  # one sample short: a hard invariant violation
            path="out.wav",
            sha256="b" * 64,
            samples=47999,
            true_peak_dbtp=-1.2,
            sample_rate=48000,
            channels=1,
            duration_s=1.0,
        ),
        core=CoreMetadata(id="test-core", algorithm="test", params_hash="f" * 64),
        guard=GuardMetadata(id="test-guard", probe_hash="d" * 64, calibration_id="e" * 64),
        environment=EnvironmentMetadata(
            platform="test",
            os_version="test",
            python_version="3",
            numpy_version="0",
            scipy_version="0",
            soundfile_version="0",
        ),
        summary=UnitSummary(units_total=1, enhanced=1),
        units=[],
    )


def test_gate_reports_failed_instead_of_raising(monkeypatch: Any, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version": 1, "manifest_id": "m", "split_name": "acceptance",'
        ' "items_count": 1, "items": [{"id": "item1", "audio_path": "x.wav",'
        ' "audio_sha256": "", "duration_s": 1.0, "speaker_id": "s", "dialect": "synthetic",'
        ' "gender": "unknown", "environment": "synthetic", "degradation_type": "clean",'
        ' "transcript_sorani": "-", "verified_by_human": false, "split": "acceptance"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        acceptance_mod, "run_pipeline", lambda **_kw: _report_violating_sample_count()
    )

    result = acceptance_mod.evaluate_acceptance_gates(manifest, output_dir=tmp_path / "out")

    assert result["release_gate_status"] == "FAILED", (
        "an invariant violation must surface as release_gate_status=FAILED, "
        f"got {result['release_gate_status']!r}"
    )
    assert result["passed_items"] == 0
    failing = [r for r in result["results"] if not r["passed"]]
    assert failing and failing[0]["id"] == "item1", "the failing item must be named"
    assert any("sample" in f.lower() for f in failing[0].get("failures", [])), (
        "the failure list must say WHICH gate failed"
    )
