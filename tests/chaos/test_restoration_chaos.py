"""Chaos and boundary tests for restoration subsystem."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.errors import InvalidUserInputError
from hawavoclean.pipeline import run_pipeline
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD


def test_missing_speaker_id_in_restore_mode_fails(tmp_path: Path) -> None:
    """Test that requesting --mode restore without --speaker-id raises InvalidUserInputError."""
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(in_wav, np.zeros(48000, dtype=np.float32), 48000)

    with pytest.raises(
        InvalidUserInputError, match="Restore mode requires an explicit --speaker-id"
    ):
        run_pipeline(
            input_path=in_wav,
            output_path=out_wav,
            mode="restore",
            speaker_id=None,
        )


def test_nonexistent_speaker_id_fails(tmp_path: Path) -> None:
    """Test that requesting an unknown speaker ID fails cleanly."""
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(in_wav, np.zeros(48000, dtype=np.float32), 48000)

    with pytest.raises(InvalidUserInputError, match="not found"):
        run_pipeline(
            input_path=in_wav,
            output_path=out_wav,
            mode="restore",
            speaker_id="character_99_nonexistent",
        )


def test_restoration_on_all_zero_silent_input() -> None:
    """Test restoration resilience on pure silence input."""
    sr = 48000
    silence = np.zeros(sr, dtype=np.float32)
    restorer = HawaRestoreKD(sample_rate=sr)

    cands = restorer.restore(silence, sample_rate=sr, effective_cutoff_hz=4000.0)
    assert len(cands) == 5
    # Silence restored should remain zero/silent without NaN or Inf
    for c in cands:
        assert np.all(np.isfinite(c.audio))


def test_path_traversal_speaker_id_rejected(tmp_path: Path) -> None:
    """Test that path traversal attempts in speaker_id are rejected immediately."""
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(in_wav, np.zeros(48000, dtype=np.float32), 48000)

    for bad_id in [
        "../../etc/passwd",
        "../character_01",
        "character_01/../..",
        "spk/01",
        "spk\\01",
    ]:
        with pytest.raises(InvalidUserInputError):
            run_pipeline(
                input_path=in_wav,
                output_path=out_wav,
                mode="restore",
                speaker_id=bad_id,
            )


def test_restoration_extremely_short_audio() -> None:
    """Test that extremely short audio buffers fail closed without crashing (R2.2)."""
    sr = 48000
    restorer = HawaRestoreKD(sample_rate=sr)
    for length in [0, 1, 10, 100, 512]:
        short_audio = np.zeros(length, dtype=np.float32)
        cands = restorer.restore(short_audio, sample_rate=sr, effective_cutoff_hz=8000.0)
        assert len(cands) >= 1
        for c in cands:
            assert len(c.audio) == length
            assert np.all(np.isfinite(c.audio))


def test_restoration_multichannel_audio() -> None:
    """Test that stereo audio is restored cleanly across both channels."""
    sr = 48000
    stereo = np.zeros((2, sr), dtype=np.float32)
    stereo[0, :] = 0.1 * np.sin(2 * np.pi * 300 * np.linspace(0, 1, sr))
    stereo[1, :] = 0.1 * np.sin(2 * np.pi * 400 * np.linspace(0, 1, sr))

    restorer = HawaRestoreKD(sample_rate=sr)
    cands = restorer.restore(stereo, sample_rate=sr, effective_cutoff_hz=6000.0)
    assert len(cands) == 5
    for c in cands:
        assert c.audio.shape == (2, sr)
        assert np.all(np.isfinite(c.audio))
