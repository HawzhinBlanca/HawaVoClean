"""The installed CLI must work from any working directory."""

import argparse
from pathlib import Path
from typing import Any

import pytest

from hawavoclean.cli import _resolve_profiles_dir, cmd_doctor
from hawavoclean.errors import InvalidUserInputError
from hawavoclean.pipeline import run_pipeline

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


def test_profiles_dir_defaults_to_the_configured_root_not_the_cwd(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """``--profiles-dir`` unset must resolve the way everything else does.

    Its argparse default was the literal relative string ``"profiles"``, so
    the lookup landed wherever the user happened to be standing and
    ``HAWAVOCLEAN_PROFILES_DIR`` was ignored outright -- a restore run started
    from the folder holding the audio reported "profile not found" on a
    machine where the server and the doctor both found every profile.
    """
    staged = tmp_path / "elsewhere" / "profiles"
    (staged / "character_01").mkdir(parents=True)
    (staged / "character_01" / "profile.json").write_text("{}")
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(staged))
    monkeypatch.chdir(tmp_path)

    assert _resolve_profiles_dir(None) == staged.resolve()
    # An explicit flag still wins, and is not second-guessed.
    assert _resolve_profiles_dir("some/other/dir") == Path("some/other/dir")


def test_an_unknown_speaker_names_the_directory_actually_searched(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """The refusal must name the resolved root, so the user can see where it looked."""
    staged = tmp_path / "profiles"
    staged.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(staged))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InvalidUserInputError, match=str(staged.resolve())):
        run_pipeline(
            input_path=FIXTURE,
            output_path=tmp_path / "out.wav",
            profile="development",
            overwrite=True,
            mode="restore",
            speaker_id="character_01",
            profiles_dir=_resolve_profiles_dir(None),
        )
