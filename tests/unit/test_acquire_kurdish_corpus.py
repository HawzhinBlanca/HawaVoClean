"""Unit tests for the Kurdish multi-speaker speech corpus acquisition tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from scripts import acquire_kurdish_corpus


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    """Create a dummy speech-like WAV file."""
    wav_path = tmp_path / "test_input.wav"
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    phase = 2.0 * np.pi * 150.0 * t
    rng = np.random.default_rng(42)
    sig = (
        0.5 * np.sin(phase)
        + 0.35 * np.sin(2 * phase)
        + 0.2 * np.sin(3 * phase)
        + 0.1 * rng.standard_normal(len(t))
    ).astype(np.float32)
    sf.write(str(wav_path), sig, sr)
    return wav_path


def test_convert_to_canonical_wav(dummy_wav: Path, tmp_path: Path) -> None:
    """convert_to_canonical_wav should resample to 48kHz mono 24-bit PCM."""
    out_path = tmp_path / "converted.wav"
    dur, digest = acquire_kurdish_corpus.convert_to_canonical_wav(
        dummy_wav, out_path, target_sr=48000
    )

    assert dur == pytest.approx(1.0, abs=0.05)
    assert len(digest) == 64
    assert out_path.is_file()

    info = sf.info(str(out_path))
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.subtype == "PCM_24"


def test_compute_file_sha256(dummy_wav: Path) -> None:
    """compute_file_sha256 returns valid 64-char hex digest."""
    digest = acquire_kurdish_corpus.compute_file_sha256(dummy_wav)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_fetch_tts4all_speaker_clips_no_files_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fetch_tts4all_speaker_clips raises ValueError if no clips found for speaker."""
    mock_api = MagicMock()
    mock_info = MagicMock()
    mock_info.siblings = []
    mock_api.dataset_info.return_value = mock_info
    monkeypatch.setattr(acquire_kurdish_corpus, "HfApi", lambda: mock_api)

    with pytest.raises(ValueError, match="No WAV files found"):
        acquire_kurdish_corpus.fetch_tts4all_speaker_clips(
            speaker="nonexistent",
            max_clips=2,
            output_speaker_dir=tmp_path / "out",
        )


def test_acquire_corpus_mocked_end_to_end(
    dummy_wav: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mocked end-to-end acquisition and manifest generation."""
    mock_api = MagicMock()
    mock_info = MagicMock()
    mock_sibling = MagicMock()
    mock_sibling.rfilename = "wavs/fatih/clip_01.wav"
    mock_info.siblings = [mock_sibling]
    mock_api.dataset_info.return_value = mock_info

    monkeypatch.setattr(acquire_kurdish_corpus, "HfApi", lambda: mock_api)
    monkeypatch.setattr(acquire_kurdish_corpus, "hf_hub_download", lambda **_kw: str(dummy_wav))

    out_dir = tmp_path / "kurdish_out"
    profiles_dir = tmp_path / "profiles"

    summary = acquire_kurdish_corpus.acquire_corpus(
        speakers=["fatih"],
        max_clips_per_speaker=1,
        output_dir=out_dir,
        enroll=True,
        profiles_dir=profiles_dir,
        min_enroll_duration_s=0.5,
    )

    assert summary.total_clips == 1
    assert summary.total_duration_s > 0
    assert Path(summary.manifest_path).is_file()

    manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
    assert manifest["speakers"] == ["fatih"]
    assert len(manifest["clips"]) == 1

    profile_json = profiles_dir / "kurdish_fatih" / "profile.json"
    assert profile_json.is_file()


def test_main_cli_help() -> None:
    """CLI --help flag exits cleanly with 0."""
    with pytest.raises(SystemExit) as exc:
        acquire_kurdish_corpus.main(["--help"])
    assert exc.value.code == 0
