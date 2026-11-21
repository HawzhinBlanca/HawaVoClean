"""Tests for clean-only / no-sidecars master output publication."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.cli as cli
from hawavoclean.publication import publish_output_generation


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Create a minimal valid test WAV file."""
    wav_path = tmp_path / "test_input.wav"
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    data = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(wav_path, data, sr, subtype="PCM_24")
    return wav_path


def test_publish_output_generation_clean_only(tmp_path: Path, sample_wav: Path) -> None:
    """Verify publish_output_generation with clean_only=True emits only master WAV."""
    dest_wav = tmp_path / "out_clean.wav"
    json_report = '{"output": {"sha256": "fake"}}'
    txt_summary = "test summary"

    publish_output_generation(
        temp_audio_path=sample_wav,
        destination_audio_path=dest_wav,
        json_report_str=json_report,
        txt_summary_str=txt_summary,
        overwrite=True,
        clean_only=True,
    )

    assert dest_wav.exists()
    assert dest_wav.stat().st_size > 0
    # Verify no sidecar files or bundles are present in tmp_path
    sidecar_json = tmp_path / f"{dest_wav.stem}.hawavoclean.json"
    sidecar_txt = tmp_path / f"{dest_wav.stem}.hawavoclean.txt"
    bundle_dir = tmp_path / f".{dest_wav.name}.hawavoclean"
    lock_file = tmp_path / f".{dest_wav.name}.hawavoclean.lock"

    assert not sidecar_json.exists()
    assert not sidecar_txt.exists()
    assert not bundle_dir.exists()
    assert not lock_file.exists()


def test_cli_process_clean_only(monkeypatch: Any, tmp_path: Path, sample_wav: Path) -> None:
    """Verify CLI process --clean-only produces only destination WAV."""
    dest_wav = tmp_path / "cli_clean.wav"

    code = _run_cli(monkeypatch, "process", str(sample_wav), "-o", str(dest_wav), "--clean-only")
    assert code == 0
    assert dest_wav.exists()

    sidecar_json = tmp_path / f"{dest_wav.stem}.hawavoclean.json"
    sidecar_txt = tmp_path / f"{dest_wav.stem}.hawavoclean.txt"
    bundle_dir = tmp_path / f".{dest_wav.name}.hawavoclean"

    assert not sidecar_json.exists()
    assert not sidecar_txt.exists()
    assert not bundle_dir.exists()


def test_cli_batch_directory_clean_only(monkeypatch: Any, tmp_path: Path) -> None:
    """Verify CLI batch with directory input and --clean-only produces only clean WAVs."""
    in_dir = tmp_path / "in_batch"
    in_dir.mkdir()
    # Create two test files in directory
    f1 = in_dir / "01.wav"
    f2 = in_dir / "02.wav"
    sf.write(f1, np.zeros(48000, dtype=np.float32), 48000, subtype="PCM_24")
    sf.write(f2, np.zeros(48000, dtype=np.float32), 48000, subtype="PCM_24")

    out_dir = tmp_path / "out_batch"

    code = _run_cli(monkeypatch, "batch", str(in_dir), "-o", str(out_dir), "--clean-only")
    assert code == 0

    assert (out_dir / "01_clean.wav").exists()
    assert (out_dir / "02_clean.wav").exists()

    # Verify no .json or .txt or hidden bundles exist in out_dir
    all_files = list(out_dir.iterdir())
    assert len(all_files) == 2
    for f in all_files:
        assert f.name.endswith("_clean.wav")
        assert not f.name.startswith(".")
