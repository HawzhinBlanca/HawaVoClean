"""Integration tests for CLI restore mode, restore-doctor, speaker-profile validate."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.cli import main


def test_cli_restore_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    """Test hawavoclean restore-doctor CLI command."""
    with pytest.raises(SystemExit) as exc_info:
        import sys

        sys.argv = ["hawavoclean", "restore-doctor"]
        main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "ALL RESTORATION CHECKS PASSED" in captured.out


def test_cli_speaker_profile_validate(capsys: pytest.CaptureFixture[str]) -> None:
    """Test hawavoclean speaker-profile validate command on profiles/ directory."""
    with pytest.raises(SystemExit) as exc_info:
        import sys

        sys.argv = ["hawavoclean", "speaker-profile", "validate", "profiles/"]
        main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "character_01" in captured.out
    assert "character_10" in captured.out


def test_cli_process_restore_mode(tmp_path: Path) -> None:
    """Test end-to-end processing in restore mode via CLI."""
    in_wav = tmp_path / "input.wav"
    out_wav = tmp_path / "output.wav"

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1500 * t)).astype(
        np.float32
    )
    sf.write(in_wav, sig, sr)

    import sys

    sys.argv = [
        "hawavoclean",
        "process",
        str(in_wav),
        "--output",
        str(out_wav),
        "--mode",
        "restore",
        "--speaker-id",
        "character_01",
        "--cutoff-hz",
        "4000.0",
        "--overwrite",
        "--allow-research-restore",
    ]

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert out_wav.exists()

    report_path = tmp_path / "output.hawavoclean.json"
    assert report_path.exists()
    with open(report_path) as f:
        rep_json = json.load(f)
    assert "restoration" in rep_json
    assert rep_json["restoration"]["speaker_id"] == "character_01"
    assert rep_json["restoration"]["mode"] == "restore"


def test_cli_verify_restoration_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that hawavoclean verify validates and prints restoration audit details."""
    in_wav = tmp_path / "input.wav"
    out_wav = tmp_path / "output.wav"

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1500 * t)).astype(
        np.float32
    )
    sf.write(in_wav, sig, sr)

    import sys

    sys.argv = [
        "hawavoclean",
        "process",
        str(in_wav),
        "--output",
        str(out_wav),
        "--mode",
        "restore",
        "--speaker-id",
        "character_02",
        "--cutoff-hz",
        "6000.0",
        "--overwrite",
        "--allow-research-restore",
    ]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    report_path = tmp_path / "output.hawavoclean.json"
    sys.argv = [
        "hawavoclean",
        "verify",
        str(out_wav),
        "--report",
        str(report_path),
    ]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "VERIFICATION PASSED" in captured.out
    assert "Restoration Mode:" in captured.out
    assert "character_02" in captured.out


def test_cli_batch_restore_mode(tmp_path: Path) -> None:
    """Test batch processing with restore mode."""
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()
    out_dir.mkdir()

    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig1 = 0.2 * np.sin(2 * np.pi * 400 * t).astype(np.float32)
    sig2 = 0.2 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    f1 = in_dir / "file1.wav"
    f2 = in_dir / "file2.wav"
    sf.write(f1, sig1, sr)
    sf.write(f2, sig2, sr)

    import sys

    sys.argv = [
        "hawavoclean",
        "batch",
        str(f1),
        str(f2),
        "-o",
        str(out_dir),
        "--mode",
        "restore",
        "--speaker-id",
        "character_03",
        "--overwrite",
        "--allow-research-restore",
    ]

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    out1 = out_dir / "file1_clean.wav"
    out2 = out_dir / "file2_clean.wav"
    assert out1.exists()
    assert out2.exists()
