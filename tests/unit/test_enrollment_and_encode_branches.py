from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.encode import (
    _finalize_deterministic_wav,
    encode_audio,
    encode_audio_streaming,
)
from hawavoclean.audio.types import AudioBuffer
from hawavoclean.enrollment import (
    _energy_voiced_segments,
    _load_mono_48k,
    _resample_to_48k,
    enroll_speaker,
)
from hawavoclean.errors import OutputValidationError

# --- 1. Audio Encode Edge and Error Branches ---


def test_clamp_wav_peak_chunk_invalid_headers(tmp_path: Path) -> None:
    # 1. Not a valid WAV (too short)
    short_file = tmp_path / "short.wav"
    short_file.write_bytes(b"RIFF")
    with pytest.raises(OutputValidationError, match="not a valid WAV/RF64 file"):
        _finalize_deterministic_wav(short_file)

    # 2. Valid 12-byte header, but truncated chunk header
    trunc_chunk = tmp_path / "trunc_chunk.wav"
    trunc_chunk.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    # only 4 bytes of chunk header instead of 8
    with pytest.raises(OutputValidationError, match="truncated chunk header"):
        _finalize_deterministic_wav(trunc_chunk)

    # 3. Malformed PEAK chunk with size < 8
    bad_peak = tmp_path / "bad_peak.wav"
    bad_peak.write_bytes(b"RIFF\x20\x00\x00\x00WAVEPEAK\x04\x00\x00\x00\x00\x00\x00\x00")
    with pytest.raises(OutputValidationError, match="malformed PEAK chunk"):
        _finalize_deterministic_wav(bad_peak)


def test_clamp_wav_peak_chunk_success(tmp_path: Path) -> None:
    # Valid WAV with PEAK chunk of size 12
    peak_wav = tmp_path / "peak.wav"
    # Header: RIFF (4), size (4), WAVE (4)
    # Chunk: PEAK (4), size=12 (4), version+timestamp (8), data (4)
    content = (
        b"RIFF\x20\x00\x00\x00WAVE"
        b"PEAK\x0c\x00\x00\x00"
        b"\x01\x00\x00\x00\xff\xff\xff\xff"
        b"\x00\x00\x00\x00"
    )
    peak_wav.write_bytes(content)
    _finalize_deterministic_wav(peak_wav)
    data = peak_wav.read_bytes()
    # Check that timestamp bytes at payload_start+4 (offset 24..28) were zeroed
    assert data[24:28] == b"\x00\x00\x00\x00"


def test_encode_audio_to_wav_write_error(tmp_path: Path) -> None:
    dest = tmp_path / "out.wav"
    buf = AudioBuffer(
        data=np.zeros((1, 1600), dtype=np.float32),
        sample_rate=16000,
    )
    with (
        patch("soundfile.write", side_effect=RuntimeError("disk full")),
        pytest.raises(OutputValidationError, match="Failed to write WAV output"),
    ):
        encode_audio(buf, dest)
    assert not dest.exists()


def test_encode_audio_stream_write_error(tmp_path: Path) -> None:
    dest = tmp_path / "out_stream.wav"
    buf = AudioBuffer(
        data=np.zeros((1, 1600), dtype=np.float32),
        sample_rate=16000,
    )
    with (
        patch("soundfile.SoundFile", side_effect=RuntimeError("cannot open stream")),
        pytest.raises(OutputValidationError, match="Failed to write WAV output"),
    ):
        encode_audio_streaming(buf, dest)
    assert not dest.exists()


# --- 2. Enrollment Edge and Error Branches ---


def test_enrollment_resample_and_mono(tmp_path: Path) -> None:
    # 1. Resample to 48k from 16k
    audio_16k = np.zeros(1600, dtype=np.float32)
    resampled = _resample_to_48k(audio_16k, 16000)
    assert len(resampled) == 4800

    # 2. Load stereo file and verify it converts to mono 48k
    stereo_path = tmp_path / "stereo.wav"
    stereo_data = np.zeros((1600, 2), dtype=np.float32)
    sf.write(str(stereo_path), stereo_data, 16000)

    loaded, duration = _load_mono_48k(stereo_path)
    assert loaded.ndim == 1
    assert len(loaded) == 4800
    assert duration == pytest.approx(0.1, abs=1e-3)

    # 3. Energy voiced segments
    voiced_idx = _energy_voiced_segments(loaded, 48000)
    assert isinstance(voiced_idx, np.ndarray)


def test_enroll_speaker_verbose_and_exceptions(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    f1 = audio_dir / "f1.wav"
    # Write 2 seconds of 48k audio
    sf.write(str(f1), np.sin(np.linspace(0, 440 * 2 * np.pi * 2, 96000)).astype(np.float32), 48000)

    out_dir = tmp_path / "profile_out"

    fake_f0_traj = MagicMock()
    fake_f0_traj.statistics.voiced_fraction = 0.8
    fake_f0_traj.statistics.median_hz = 150.0
    fake_f0_traj.f0_hz = np.array([150.0, 152.0, 148.0])
    fake_f0_traj.vuv_mask = np.array([1.0, 1.0, 1.0])

    fake_embed = np.ones(128, dtype=np.float32)

    # 1. Success with verbose=True
    with (
        patch("hawavoclean.enrollment.SpeakerEmbeddingExtractor.extract", return_value=fake_embed),
        patch("hawavoclean.enrollment.F0Extractor.extract", return_value=fake_f0_traj),
    ):
        res = enroll_speaker(
            speaker_id="test_spk",
            display_name="Test Speaker",
            audio_dir=audio_dir,
            output_dir=out_dir,
            consent_granted=True,
            min_duration_s=1.0,
            verbose=True,
        )
        assert res.speaker_id == "test_spk"
        assert res.n_files == 1

    # 2. No valid speech detected -> ValueError (embedding norm is 0)
    zero_embed = np.zeros(128, dtype=np.float32)
    with (
        patch("hawavoclean.enrollment.SpeakerEmbeddingExtractor.extract", return_value=zero_embed),
        patch("hawavoclean.enrollment.F0Extractor.extract", return_value=fake_f0_traj),
        pytest.raises(ValueError, match="No valid speech detected in any file"),
    ):
        enroll_speaker(
            speaker_id="test_spk",
            display_name="Test Speaker",
            audio_dir=audio_dir,
            output_dir=tmp_path / "out2",
            consent_granted=True,
            min_duration_s=1.0,
        )

    # 3. No voiced frames detected -> ValueError
    fake_unvoiced = MagicMock()
    fake_unvoiced.statistics.voiced_fraction = 0.0
    fake_unvoiced.statistics.median_hz = 0.0
    fake_unvoiced.f0_hz = np.array([0.0, 0.0])
    fake_unvoiced.vuv_mask = np.array([0.0, 0.0])

    with (
        patch("hawavoclean.enrollment.SpeakerEmbeddingExtractor.extract", return_value=fake_embed),
        patch("hawavoclean.enrollment.F0Extractor.extract", return_value=fake_unvoiced),
        pytest.raises(ValueError, match="No voiced frames detected in any file"),
    ):
        enroll_speaker(
            speaker_id="test_spk",
            display_name="Test Speaker",
            audio_dir=audio_dir,
            output_dir=tmp_path / "out3",
            consent_granted=True,
            min_duration_s=1.0,
        )
