"""CLI error-handler and doctor failure-branch coverage."""

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

import hawavoclean.cli as cli
from hawavoclean.errors import ExitCode, HawaVoCleanError
from hawavoclean.paths import models_dir

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def test_doctor_fails_with_empty_model_dir(monkeypatch: Any, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(tmp_path / "empty"))
    assert _run_cli(monkeypatch, "doctor") == int(ExitCode.PREFLIGHT_FAILURE)


def test_doctor_fails_with_empty_config_dir(monkeypatch: Any, tmp_path: Path) -> None:
    (tmp_path / "cfg").mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_CONFIG_DIR", str(tmp_path / "cfg"))
    assert _run_cli(monkeypatch, "doctor") == int(ExitCode.PREFLIGHT_FAILURE)


def test_doctor_detects_calibration_tampering(monkeypatch: Any, tmp_path: Path) -> None:
    override = tmp_path / "models"
    shutil.copytree(models_dir(), override)
    calib = override / "guard-calibration.json"
    calib.write_text(
        calib.read_text().replace('"max_posterior_js_div": 0.25', '"max_posterior_js_div": 0.9')
    )
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(override))
    assert _run_cli(monkeypatch, "doctor") == int(ExitCode.PREFLIGHT_FAILURE)


def test_doctor_warns_without_ffmpeg(monkeypatch: Any) -> None:
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    assert _run_cli(monkeypatch, "doctor") == 0  # warn, not fail


def test_process_preflight_error_exit_code(monkeypatch: Any, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(tmp_path / "empty"))
    rc = _run_cli(
        monkeypatch, "process", str(FIXTURE), "-o", str(tmp_path / "o.wav"), "--overwrite"
    )
    assert rc == int(ExitCode.PREFLIGHT_FAILURE)


def test_process_unhandled_error_exit_code(monkeypatch: Any, tmp_path: Path) -> None:
    def boom(**_kw: Any) -> Any:
        raise RuntimeError("surprise")

    monkeypatch.setattr(cli, "run_pipeline", boom)
    rc = _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(tmp_path / "o.wav"))
    assert rc == int(ExitCode.PUBLICATION_FAILURE)


def test_process_hawavoclean_error_exit_code(monkeypatch: Any, tmp_path: Path) -> None:
    def raise_vc(**_kw: Any) -> Any:
        raise HawaVoCleanError("custom", exit_code=ExitCode.INVALID_USER_INPUT)

    monkeypatch.setattr(cli, "run_pipeline", raise_vc)
    rc = _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(tmp_path / "o.wav"))
    assert rc == int(ExitCode.INVALID_USER_INPUT)


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One processed output shared by the verify-mismatch tests."""
    import os

    tmp = tmp_path_factory.mktemp("pub")
    os.environ["HAWAVOCLEAN_WORK_DIR"] = str(tmp / "w")
    from hawavoclean.pipeline import run_pipeline

    run_pipeline(input_path=FIXTURE, output_path=tmp / "v.wav", overwrite=True)
    return tmp / "v.wav", tmp / "v.hawavoclean.json"


def _tampered_report(report: Path, tmp_path: Path, field: str, value: object) -> Path:
    import json

    data = json.loads(report.read_text())
    data["output"][field] = value
    out = tmp_path / f"tampered_{field}.json"
    out.write_text(json.dumps(data))
    return out


def test_verify_detects_each_structural_mismatch(
    monkeypatch: Any, tmp_path: Path, published: tuple[Path, Path]
) -> None:
    audio, report = published
    for field, value in (("samples", 1), ("sample_rate", 44100), ("channels", 2)):
        import json

        data = json.loads(report.read_text())
        data["output"][field] = value
        bad = tmp_path / f"bad_{field}.json"
        bad.write_text(json.dumps(data))
        rc = _run_cli(monkeypatch, "verify", str(audio), "-r", str(bad))
        assert rc == int(ExitCode.PUBLICATION_FAILURE), f"mismatched {field} not detected"


def test_verify_rejects_malformed_report(
    monkeypatch: Any, tmp_path: Path, published: tuple[Path, Path]
) -> None:
    audio, _ = published
    bad = tmp_path / "malformed.json"
    bad.write_text("{not json")
    rc = _run_cli(monkeypatch, "verify", str(audio), "-r", str(bad))
    assert rc == int(ExitCode.PUBLICATION_FAILURE)


def test_eval_command_prints_failures(monkeypatch: Any, tmp_path: Path) -> None:

    def fake_eval(**_kw: Any) -> dict[str, Any]:
        return {
            "release_gate_status": "FAILED",
            "passed_items": 0,
            "total_items": 1,
            "speech_units_total": 1,
            "speech_units_enhanced": 0,
            "corpus_failures": ["nothing enhanced"],
            "results": [{"id": "x", "passed": False, "failures": ["sample count mismatch"]}],
        }

    monkeypatch.setattr("hawavoclean.eval.acceptance.evaluate_acceptance_gates", fake_eval)
    rc = _run_cli(
        monkeypatch,
        "eval",
        "--manifest",
        "data/acceptance/manifest.json",
        "--output-dir",
        str(tmp_path),
    )
    assert rc == int(ExitCode.PUBLICATION_FAILURE)
