"""Unit tests for probe, decode, encode, resample, and channel layout handling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.channels import handle_channel_layout
from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.encode import encode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioBuffer, AudioProbeResult, ChannelMode
from hawavoclean.errors import InvalidUserInputError


@pytest.mark.unit
def test_encode_and_probe_float32() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "float32.wav"
        data = np.random.default_rng(42).normal(0.0, 0.2, size=(2, 48000)).astype(np.float32)
        buf = AudioBuffer(data=data, sample_rate=48000)

        out_path = encode_audio(buf, dest, output_bit_depth="float32", dither=False)
        assert out_path.exists()

        probe = probe_audio(dest)
        assert probe.sample_rate == 48000
        assert probe.channels == 2
        assert probe.samples == 48000
        assert probe.bit_depth == 32


@pytest.mark.unit
def test_decode_audio_native() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "test.wav"
        sig = np.zeros(24000, dtype=np.float32)
        sf.write(str(wav_path), sig, 48000, subtype="PCM_24")

        probe = probe_audio(wav_path)
        buf = decode_audio(probe)
        assert buf.sample_rate == 48000
        assert buf.channels == 1
        assert buf.samples == 24000


@pytest.mark.unit
def test_decode_invalid_file_raises_error() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_file = Path(tmpdir) / "corrupt.wav"
        with open(bad_file, "wb") as f:
            f.write(b"NOT_A_VALID_WAV_HEADER")

        fake_probe = AudioProbeResult(
            path=bad_file,
            format_name="wav",
            codec_name="pcm_s16le",
            channels=1,
            sample_rate=48000,
            samples=1000,
            duration_s=1.0,
            bit_depth=16,
            sha256="0" * 64,
        )
        with pytest.raises(InvalidUserInputError):
            decode_audio(fake_probe)


@pytest.mark.unit
def test_handle_channel_layout_dual_mono() -> None:
    data = np.ones((2, 1000), dtype=np.float32)
    buf = AudioBuffer(data=data, sample_rate=48000)
    channels, dup = handle_channel_layout(buf, ChannelMode.DUAL_MONO_SAME)
    assert len(channels) == 1
    assert dup is True
