"""Batch command: isolation of failures, summary, exit codes, stem cleaning."""

import sys
from pathlib import Path
from typing import Any

import pytest

import hawavoclean.cli as cli
from hawavoclean.cli import _clean_stem
from hawavoclean.errors import ExitCode

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def test_clean_stem_strips_stacked_audio_suffixes() -> None:
    assert _clean_stem(Path("Flute 09.m4a.mp4")) == "Flute 09"
    assert _clean_stem(Path("take.wav")) == "take"
    assert _clean_stem(Path("notes.txt")) == "notes.txt"
    assert _clean_stem(Path(".wav")) == ".wav"  # no empty stems


def test_batch_processes_all_and_exits_zero(monkeypatch: Any, tmp_path: Path) -> None:
    rc = _run_cli(
        monkeypatch,
        "batch",
        str(FIX / "sample_sorani_podcast.wav"),
        str(FIX / "sample_noisy_hum.wav"),
        "-o",
        str(tmp_path),
        "--overwrite",
    )
    assert rc == 0
    assert (tmp_path / "sample_sorani_podcast_clean.wav").exists()
    assert (tmp_path / "sample_noisy_hum_clean.wav").exists()
    assert (tmp_path / "sample_noisy_hum_clean.hawavoclean.json").exists()


def test_batch_isolates_failures_and_exits_nonzero(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """One bad input must not stop the good one, and must fail the exit code."""
    rc = _run_cli(
        monkeypatch,
        "batch",
        str(FIX / "sample_ambiguous_stereo.wav"),  # raises AmbiguousStereoError
        str(FIX / "sample_sorani_podcast.wav"),
        "-o",
        str(tmp_path),
        "--overwrite",
    )
    assert rc == int(ExitCode.PUBLICATION_FAILURE)
    out = capsys.readouterr().out
    assert "1/2 succeeded" in out
    assert "AmbiguousStereoError" in out
    assert (tmp_path / "sample_sorani_podcast_clean.wav").exists()


def test_batch_skip_existing(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    src = FIX / "sample_sorani_podcast.wav"
    assert _run_cli(monkeypatch, "batch", str(src), "-o", str(tmp_path), "--overwrite") == 0
    rc = _run_cli(monkeypatch, "batch", str(src), "-o", str(tmp_path), "--skip-existing")
    assert rc == 0
    assert "SKIP" in capsys.readouterr().out


def test_batch_no_valid_inputs(monkeypatch: Any, tmp_path: Path) -> None:
    rc = _run_cli(monkeypatch, "batch", str(tmp_path / "ghost.wav"), "-o", str(tmp_path))
    assert rc == int(ExitCode.INVALID_USER_INPUT)
