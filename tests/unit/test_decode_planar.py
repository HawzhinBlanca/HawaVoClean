from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hawavoclean.audio.decode import decode_audio, decode_audio_to_memmap
from hawavoclean.audio.probe import probe_audio
from hawavoclean.errors import InvalidUserInputError, PreflightError

SAMPLE_WAV = Path("tests/fixtures/sample_sorani_podcast.wav")


def test_decode_audio_disk_planar_success(tmp_path: Path) -> None:
    probe = probe_audio(SAMPLE_WAV)
    dest = tmp_path / "planar.pcm"

    buf = decode_audio_to_memmap(probe, dest, chunk_samples=8192)
    assert dest.is_file()
    assert buf.sample_rate == probe.sample_rate
    assert buf.channels == probe.channels
    assert buf.samples == probe.samples

    # Compare values against standard in-memory decode_audio
    expected_buf = decode_audio(probe)
    np.testing.assert_allclose(buf.data, expected_buf.data, atol=1e-5)

    # Ensure intermediate temporary channel files were cleaned up
    temp_files = list(tmp_path.glob(".*channel*.tmp"))
    assert len(temp_files) == 0


def test_decode_audio_disk_planar_destination_already_exists(tmp_path: Path) -> None:
    probe = probe_audio(SAMPLE_WAV)
    dest = tmp_path / "already_exists.pcm"
    dest.write_bytes(b"existing content")

    with pytest.raises(InvalidUserInputError, match="already exists"):
        decode_audio_to_memmap(probe, dest)


def test_decode_audio_disk_planar_insufficient_space(tmp_path: Path) -> None:
    probe = probe_audio(SAMPLE_WAV)
    dest = tmp_path / "no_space.pcm"

    fake_usage = shutil._ntuple_diskusage(total=10**12, used=10**12, free=1024)
    with (
        patch("shutil.disk_usage", return_value=fake_usage),
        pytest.raises(PreflightError, match="Insufficient scratch space"),
    ):
        decode_audio_to_memmap(probe, dest)


def test_decode_audio_disk_planar_os_error_handling(tmp_path: Path) -> None:
    probe = probe_audio(SAMPLE_WAV)
    dest = tmp_path / "os_error.pcm"

    with (
        patch("builtins.open", side_effect=OSError("disk read-only simulation")),
        pytest.raises(PreflightError, match="could not write scratch audio"),
    ):
        decode_audio_to_memmap(probe, dest)
    assert not dest.exists()
