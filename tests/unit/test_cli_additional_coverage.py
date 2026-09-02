from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hawavoclean.cli import cmd_batch_worker, cmd_enroll_speaker, cmd_metrics
from hawavoclean.errors import ExitCode, PreflightError


def test_cmd_metrics_missing_args() -> None:
    args = argparse.Namespace(corpus=None, reference=None, candidate=None, output=None)
    code = cmd_metrics(args)
    assert code == int(ExitCode.PREFLIGHT_FAILURE)


def test_cmd_metrics_single_pair(tmp_path: Path) -> None:
    out_file = tmp_path / "metrics.json"
    args = argparse.Namespace(
        corpus=None,
        reference="tests/fixtures/sample_sorani_podcast.wav",
        candidate="tests/fixtures/sample_sorani_podcast.wav",
        output=str(out_file),
    )

    code = cmd_metrics(args)
    assert code == int(ExitCode.SUCCESS)
    assert out_file.is_file()
    data = json.loads(out_file.read_text())
    assert "si_snr_db" in data
    assert "lsd_db" in data


def test_cmd_metrics_corpus_mode(tmp_path: Path) -> None:
    corpus_json = tmp_path / "corpus.json"
    corpus_json.write_text(
        json.dumps(
            [
                {
                    "reference": "tests/fixtures/sample_sorani_podcast.wav",
                    "candidate": "tests/fixtures/sample_sorani_podcast.wav",
                }
            ]
        ),
        encoding="utf-8",
    )
    out_file = tmp_path / "corpus_out.json"
    args = argparse.Namespace(
        corpus=str(corpus_json),
        reference=None,
        candidate=None,
        output=str(out_file),
    )

    code = cmd_metrics(args)
    assert code == int(ExitCode.SUCCESS)
    assert out_file.is_file()
    data = json.loads(out_file.read_text())
    assert data["total_pairs"] == 1


def test_cmd_enroll_speaker_dir_not_found(tmp_path: Path) -> None:
    args = argparse.Namespace(
        audio_dir=str(tmp_path / "non_existent"),
        output_dir=str(tmp_path / "out"),
        speaker_id="spk_01",
        display_name=None,
        consent_granted=True,
        consent_note=None,
    )
    assert cmd_enroll_speaker(args) == int(ExitCode.INVALID_USER_INPUT)


def test_cmd_enroll_speaker_failure_and_success(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    out_dir = tmp_path / "out"

    args = argparse.Namespace(
        audio_dir=str(audio_dir),
        output_dir=str(out_dir),
        speaker_id="spk_01",
        display_name="Speaker One",
        consent_granted=True,
        consent_note="Owned audio",
    )

    # 1. Failure branch
    with patch(
        "hawavoclean.enrollment.enroll_speaker", side_effect=RuntimeError("enrollment crash")
    ):
        assert cmd_enroll_speaker(args) == int(ExitCode.PREFLIGHT_FAILURE)

    # 2. Success branch
    dummy_res = SimpleNamespace(
        speaker_id="spk_01",
        n_files=2,
        total_duration_s=600.0,
        f0_median_hz=180.0,
        f0_p05_hz=120.0,
        f0_p95_hz=240.0,
        embedding_dim=192,
        profile_dir=out_dir,
    )
    dummy_prof = SimpleNamespace(speaker_id="spk_01")

    with (
        patch("hawavoclean.enrollment.enroll_speaker", return_value=dummy_res),
        patch("hawavoclean.restoration.profiles.validate_speaker_profile", return_value=dummy_prof),
    ):
        assert cmd_enroll_speaker(args) == int(ExitCode.SUCCESS)


def test_cmd_worker_loop() -> None:
    inputs = [
        "not json\n",
        json.dumps({"input": "in.wav", "output": "out.wav", "profile": "production"}) + "\n",
        json.dumps({"input": "fail.wav", "output": "out.wav", "profile": "production"}) + "\n",
        json.dumps({"input": "crash.wav", "output": "out.wav", "profile": "production"}) + "\n",
        json.dumps({"op": "quit"}) + "\n",
    ]
    stdin_mock = io.StringIO("".join(inputs))

    def fake_pipeline(input_path: str, **_kwargs: object) -> None:
        if "fail" in input_path:
            raise PreflightError("synthetic preflight fail")
        if "crash" in input_path:
            raise RuntimeError("synthetic crash")

    emitted: list[dict[str, Any]] = []

    class FakeSink:
        def emit(self, msg: dict[str, Any]) -> None:
            emitted.append(msg)

        def close(self) -> None:
            pass

    with (
        patch.object(sys, "stdin", stdin_mock),
        patch("hawavoclean.cli._JsonLineSink", return_value=FakeSink()),
        patch("hawavoclean.cli.run_pipeline", side_effect=fake_pipeline),
        patch("hawavoclean.enhancement.worker.set_pool_reuse"),
        patch("hawavoclean.enhancement.worker.shutdown_pool_cache"),
    ):
        code = cmd_batch_worker(argparse.Namespace())

    assert code == int(ExitCode.SUCCESS)
    assert emitted[0] == {"event": "ready"}
    # Line 1: invalid json error
    assert emitted[1]["ok"] is False
    assert emitted[1]["code"] == int(ExitCode.INVALID_USER_INPUT)
    # Line 2: ok
    assert emitted[2] == {"ok": True}
    # Line 3: HawaVoCleanError
    assert emitted[3]["ok"] is False
    assert emitted[3]["code"] == int(ExitCode.PREFLIGHT_FAILURE)
    # Line 4: RuntimeError
    assert emitted[4]["ok"] is False
    assert emitted[4]["code"] == int(ExitCode.PUBLICATION_FAILURE)


def test_cmd_doctor_success_and_failures() -> None:
    from hawavoclean.cli import cmd_doctor

    # Normal success run
    res = cmd_doctor(argparse.Namespace())
    assert res in (int(ExitCode.SUCCESS), int(ExitCode.PREFLIGHT_FAILURE))

    # Test failure when profile config is missing or broken
    with patch("hawavoclean.cli.profile_config_path") as mock_cfg:
        mock_cfg.return_value = Path("/nonexistent/cfg.toml")
        assert cmd_doctor(argparse.Namespace()) == int(ExitCode.PREFLIGHT_FAILURE)

    # Test warning when ffmpeg not found
    with patch("shutil.which", return_value=None):
        cmd_doctor(argparse.Namespace())


def test_cmd_restore_doctor_success_and_failures() -> None:
    from hawavoclean.cli import cmd_restore_doctor

    # Normal restore doctor run
    res = cmd_restore_doctor(argparse.Namespace())
    assert res in (int(ExitCode.SUCCESS), int(ExitCode.PREFLIGHT_FAILURE))

    # Profiles root missing
    with patch("hawavoclean.cli.paths_profiles_root", return_value=Path("/nonexistent/profiles")):
        assert cmd_restore_doctor(argparse.Namespace()) == int(ExitCode.PREFLIGHT_FAILURE)
