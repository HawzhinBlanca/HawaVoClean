"""ffprobe/ffmpeg-less fallback paths and input rejection branches."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.errors import InvalidUserInputError

SR = 48000


@pytest.fixture()
def wav(tmp_path: Path) -> Path:
    p = tmp_path / "t.wav"
    t = np.arange(SR) / SR
    sf.write(str(p), (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR, subtype="PCM_24")
    return p


def test_probe_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidUserInputError):
        probe_audio(tmp_path / "absent.wav")


def test_probe_soundfile_fallback(monkeypatch: Any, wav: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    result = probe_audio(wav)
    assert result.sample_rate == SR
    assert result.samples == SR
    assert result.channels == 1


def test_probe_rejects_ultrasonic(tmp_path: Path) -> None:
    p = tmp_path / "hi.wav"
    sf.write(str(p), np.zeros(1000, dtype=np.float32), 96000, subtype="PCM_24")
    with pytest.raises(InvalidUserInputError):
        probe_audio(p, max_sample_rate=48000)


def test_probe_rejects_garbage_bytes(tmp_path: Path) -> None:
    p = tmp_path / "junk.wav"
    p.write_bytes(b"not audio at all")
    from hawavoclean.errors import HawaVoCleanError

    with pytest.raises(HawaVoCleanError):
        probe_audio(p)


def test_decode_soundfile_fallback(monkeypatch: Any, wav: Path) -> None:
    media = probe_audio(wav)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    buf = decode_audio(media, timeout_s=30.0)
    assert buf.samples == SR
    assert buf.data.dtype == np.float32
