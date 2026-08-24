"""The acceptance gate must be able to report failure — structurally, and under -O."""

from pathlib import Path
from typing import Any

import hawavoclean.eval.acceptance as acceptance_mod
from hawavoclean.report.schema import (
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)
from tests.support.report_provenance import build, core, environment, guard


def _report_violating_sample_count() -> HawaVoCleanReport:
    return HawaVoCleanReport(
        release=current_release_metadata(),
        build=build(),
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
        core=core("test-core", "test", "f" * 64),
        guard=guard("test-guard", "d" * 64, "e" * 64),
        environment=environment(),
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


def _base_report(**overrides: Any) -> HawaVoCleanReport:
    """A structurally valid report; overrides poke individual invariants."""
    from hawavoclean.report.schema import UnitDecisionRecord

    defaults: dict[str, Any] = {
        "release": current_release_metadata(),
        "build": build(),
        "job_id": "j",
        "config_hash": "c" * 64,
        "input": MediaStats(
            path="i.wav",
            sha256="a" * 64,
            samples=48000,
            sample_rate=48000,
            channels=1,
            duration_s=1.0,
        ),
        "output": MediaStats(
            path="o.wav",
            sha256="b" * 64,
            samples=48000,
            true_peak_dbtp=-1.2,
            sample_rate=48000,
            channels=1,
            duration_s=1.0,
        ),
        "core": core("t", "t", "f" * 64),
        "guard": guard("g", "d" * 64, "e" * 64),
        "environment": environment(platform="t", os_version="t"),
        "summary": UnitSummary(units_total=1, enhanced=1),
        "units": [
            UnitDecisionRecord(
                unit_id=0,
                channel=0,
                start_sample=0,
                end_sample=48000,
                start_time_s=0.0,
                end_time_s=1.0,
                is_speech=True,
                input_sha256="1" * 64,
                output_sha256="2" * 64,
                guard_a_verdict="PASS",
                final_decision="enhanced",
            )
        ],
    }
    defaults.update(overrides)
    return HawaVoCleanReport(**defaults)


def _run_gate_with(monkeypatch: Any, tmp_path: Path, report: HawaVoCleanReport) -> dict[str, Any]:
    manifest = tmp_path / "m.json"
    manifest.write_text(
        '{"schema_version": 1, "manifest_id": "m", "split_name": "acceptance",'
        ' "items_count": 1, "items": [{"id": "item1", "audio_path": "x.wav",'
        ' "audio_sha256": "", "duration_s": 1.0, "speaker_id": "s", "dialect": "synthetic",'
        ' "gender": "unknown", "environment": "synthetic", "degradation_type": "clean",'
        ' "transcript_sorani": "-", "verified_by_human": false, "split": "acceptance"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(acceptance_mod, "run_pipeline", lambda **_kw: report)
    result: dict[str, Any] = acceptance_mod.evaluate_acceptance_gates(
        manifest, output_dir=tmp_path / "o"
    )
    return result


def test_gate_channel_mismatch_fails(monkeypatch: Any, tmp_path: Path) -> None:
    rep = _base_report(
        output=MediaStats(
            path="o.wav",
            sha256="b" * 64,
            samples=48000,
            true_peak_dbtp=-1.2,
            sample_rate=48000,
            channels=2,
            duration_s=1.0,
        )
    )
    res = _run_gate_with(monkeypatch, tmp_path, rep)
    assert res["release_gate_status"] == "FAILED"
    assert any("channel" in f for f in res["results"][0]["failures"])


def test_gate_sample_rate_mismatch_fails(monkeypatch: Any, tmp_path: Path) -> None:
    rep = _base_report(
        output=MediaStats(
            path="o.wav",
            sha256="b" * 64,
            samples=48000,
            true_peak_dbtp=-1.2,
            sample_rate=44100,
            channels=1,
            duration_s=1.0,
        )
    )
    res = _run_gate_with(monkeypatch, tmp_path, rep)
    assert res["release_gate_status"] == "FAILED"


def test_gate_true_peak_violation_fails(monkeypatch: Any, tmp_path: Path) -> None:
    rep = _base_report(
        output=MediaStats(
            path="o.wav",
            sha256="b" * 64,
            samples=48000,
            true_peak_dbtp=-0.4,
            sample_rate=48000,
            channels=1,
            duration_s=1.0,
        )
    )
    res = _run_gate_with(monkeypatch, tmp_path, rep)
    assert res["release_gate_status"] == "FAILED"
    assert any("true peak" in f for f in res["results"][0]["failures"])


def test_gate_unverified_enhanced_unit_fails(monkeypatch: Any, tmp_path: Path) -> None:
    from hawavoclean.report.schema import UnitDecisionRecord

    rep = _base_report(
        units=[
            UnitDecisionRecord(
                unit_id=0,
                channel=0,
                start_sample=0,
                end_sample=48000,
                start_time_s=0.0,
                end_time_s=1.0,
                is_speech=True,
                input_sha256="1" * 64,
                output_sha256="2" * 64,
                guard_a_verdict="UNVERIFIED",
                final_decision="enhanced",
            )
        ]
    )
    res = _run_gate_with(monkeypatch, tmp_path, rep)
    assert res["release_gate_status"] == "FAILED"
    assert any("UNVERIFIED" in f for f in res["results"][0]["failures"])


def test_gate_nothing_enhanced_fails_corpus_floor(monkeypatch: Any, tmp_path: Path) -> None:
    from hawavoclean.report.schema import UnitDecisionRecord

    rep = _base_report(
        units=[
            UnitDecisionRecord(
                unit_id=0,
                channel=0,
                start_sample=0,
                end_sample=48000,
                start_time_s=0.0,
                end_time_s=1.0,
                is_speech=True,
                input_sha256="1" * 64,
                output_sha256="2" * 64,
                guard_a_verdict="REVERT",
                final_decision="original_reverted",
            )
        ],
        summary=UnitSummary(units_total=1, enhanced=0, reverted=1),
    )
    res = _run_gate_with(monkeypatch, tmp_path, rep)
    assert res["release_gate_status"] == "FAILED"
    assert res["corpus_failures"], "the did-nothing floor must trip"


def test_gate_pipeline_exception_is_recorded_not_raised(monkeypatch: Any, tmp_path: Path) -> None:
    def boom(**_kw: Any) -> Any:
        raise RuntimeError("pipeline exploded")

    manifest = tmp_path / "m.json"
    manifest.write_text(
        '{"schema_version": 1, "manifest_id": "m", "split_name": "acceptance",'
        ' "items_count": 1, "items": [{"id": "item1", "audio_path": "x.wav",'
        ' "audio_sha256": "", "duration_s": 1.0, "speaker_id": "s", "dialect": "synthetic",'
        ' "gender": "unknown", "environment": "synthetic", "degradation_type": "clean",'
        ' "transcript_sorani": "-", "verified_by_human": false, "split": "acceptance"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(acceptance_mod, "run_pipeline", boom)
    res = acceptance_mod.evaluate_acceptance_gates(manifest, output_dir=tmp_path / "o")
    assert res["release_gate_status"] == "FAILED"
    assert "RuntimeError" in res["results"][0]["failures"][0]
