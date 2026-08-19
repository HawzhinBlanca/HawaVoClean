"""The installed CLI must work from any working directory."""

import argparse
from pathlib import Path
from typing import Any

import pytest

from voiceclean.cli import cmd_doctor
from voiceclean.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"


def test_doctor_passes_outside_repo_root(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rc = cmd_doctor(argparse.Namespace())
    assert rc == 0, "doctor must not depend on the current working directory"


@pytest.mark.integration
def test_pipeline_runs_outside_repo_root(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_pipeline(
        input_path=FIXTURE,
        output_path=tmp_path / "out.wav",
        profile="production",
        overwrite=True,
    )
    assert (tmp_path / "out.wav").exists()
    assert "unknown" not in report.core.id.lower()
