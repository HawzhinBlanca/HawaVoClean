"""Unit tests for VoiceClean command line interface."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voiceclean.cli import (
    cmd_audit_models,
    cmd_benchmark,
    cmd_blind_abx,
    cmd_calibrate,
    cmd_doctor,
    cmd_eval,
    cmd_process,
    cmd_verify,
)
from voiceclean.errors import ExitCode
from voiceclean.guard.spectral_probe import FixedProbe
from voiceclean.pipeline import run_pipeline


@pytest.mark.unit
def test_cli_doctor() -> None:
    args = argparse.Namespace()
    ret = cmd_doctor(args)
    assert ret == int(ExitCode.SUCCESS)


@pytest.mark.unit
def test_cli_audit_models() -> None:
    args = argparse.Namespace()
    ret = cmd_audit_models(args)
    assert ret == int(ExitCode.SUCCESS)


@pytest.mark.unit
def test_cli_verify_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav_file = tmp / "test.wav"
        sr = 48000
        sig = (0.2 * np.sin(2 * np.pi * 400 * np.linspace(0, 1, sr, endpoint=False))).astype(
            np.float32
        )
        sf.write(str(wav_file), np.column_stack([sig, sig]), sr, subtype="PCM_24")

        # Run process to get authentic report
        _rep = run_pipeline(
            input_path=wav_file,
            output_path=tmp / "out.wav",
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

        args = argparse.Namespace(
            output=tmp / "out.wav",
            report=tmp / "out.voiceclean.json",
        )
        ret = cmd_verify(args)
        assert ret == int(ExitCode.SUCCESS)


@pytest.mark.unit
def test_cli_process_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        args = argparse.Namespace(
            input="tests/fixtures/sample_sorani_podcast.wav",
            output=str(tmp / "out_cli.wav"),
            config=None,
            profile="development",
            overwrite=True,
        )
        ret = cmd_process(args)
        assert ret == int(ExitCode.SUCCESS)
        assert (tmp / "out_cli.wav").exists()


@pytest.mark.unit
def test_cli_calibrate_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        args = argparse.Namespace(
            manifest="data/calibration/manifest.json",
            output=str(tmp / "guard-calib-test.json"),
            corruption_profile="standard",
            fixed_probe=True,
        )
        ret = cmd_calibrate(args)
        assert ret == int(ExitCode.SUCCESS)
        assert (tmp / "guard-calib-test.json").exists()


@pytest.mark.unit
def test_cli_eval_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        args = argparse.Namespace(
            manifest="data/acceptance/manifest.json",
            output_dir=str(tmp / "eval_out"),
        )
        ret = cmd_eval(args)
        assert ret == int(ExitCode.SUCCESS)


@pytest.mark.unit
def test_cli_benchmark_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        args = argparse.Namespace(
            manifest="data/acceptance/manifest.json",
            output=str(tmp / "benchmark.json"),
        )
        ret = cmd_benchmark(args)
        assert ret == int(ExitCode.SUCCESS)
        assert (tmp / "benchmark.json").exists()


@pytest.mark.unit
def test_cli_blind_abx_command() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        args = argparse.Namespace(
            manifest="data/acceptance/manifest.json",
            manifest_b=None,
            output=str(tmp / "blind_trials.json"),
        )
        ret = cmd_blind_abx(args)
        assert ret == int(ExitCode.SUCCESS)
        assert (tmp / "blind_trials.json").exists()
