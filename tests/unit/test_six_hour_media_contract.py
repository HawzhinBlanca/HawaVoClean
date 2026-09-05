"""Comprehensive qualification suite for Phase E1.4.

Enforces the six-hour/8 GB contract across MP3, M4A, MP4 audio extraction,
WAV, AIFF, and FLAC, mono and stereo, without full-file memory paths or
resource bombs.

Covers:
1. All 6 format families (MP3, M4A, MP4 video audio extraction, WAV, AIFF, FLAC) in mono and stereo.
2. Channel layout boundaries (1, 2 accepted; 0, 3, 5.1, 7.1, 8, 16 rejected).
3. Exact 6-hour boundary (21,600.0s accepted, 21,600.001s rejected, sample ceiling, streaming decode ceiling).
4. Exact 8 GB boundary (8,589,934,592 bytes accepted, 8,589,934,593 bytes rejected before hashing/ffprobe).
5. Malformed container headers for all 6 formats failing closed cleanly.
6. Decompression bombs, resource bombs, and metadata amplification.
7. Disk-full preflight across PinnedSource, decode_audio_to_memmap, and destination volume.
8. Bounded streaming memory paths without full-file heap allocations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.audio.decode as decode_module
import hawavoclean.audio.probe as probe_module
from hawavoclean.audio.decode import (
    decode_audio,
    decode_audio_to_memmap,
    iter_decode_audio,
)
from hawavoclean.audio.probe import (
    MAX_FFPROBE_METADATA_BYTES,
    MAX_STREAM_RECORDS,
    probe_audio,
)
from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.errors import (
    MediaPreflightError,
    MediaPreflightReason,
    PreflightError,
)
from hawavoclean.pipeline import (
    MAX_INPUT_CHANNELS,
    MAX_INPUT_DURATION_S,
    MAX_INPUT_FILE_BYTES,
    _validate_natural_media_contract,
)
from hawavoclean.source_pin import (
    PinnedSource,
)

ffmpeg_bin = shutil.which("ffmpeg") or ""
ffprobe_bin = shutil.which("ffprobe") or ""

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers for Synthetic Media Generation
# ---------------------------------------------------------------------------


def _generate_sine_audio(
    duration_s: float = 0.5,
    sample_rate: int = 48_000,
    channels: int = 1,
    freq: float = 440.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    samples = int(round(duration_s * sample_rate))
    t = np.arange(samples, dtype=np.float32) / sample_rate
    base = 0.25 * np.sin(2.0 * np.pi * freq * t, dtype=np.float32)
    if channels == 1:
        return np.ascontiguousarray(base[np.newaxis, :])
    data = np.zeros((channels, samples), dtype=np.float32)
    for ch in range(channels):
        data[ch] = base * (1.0 - 0.2 * ch)
    return np.ascontiguousarray(data)


def _write_media(
    path: Path,
    data: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int = 48_000,
) -> None:
    suffix = path.suffix.lower()

    if suffix == ".wav":
        sf.write(str(path), data.T, sample_rate, format="WAV", subtype="PCM_16")
    elif suffix in (".aiff", ".aif"):
        sf.write(str(path), data.T, sample_rate, format="AIFF", subtype="PCM_24")
    elif suffix == ".flac":
        sf.write(str(path), data.T, sample_rate, format="FLAC", subtype="PCM_16")
    elif suffix == ".mp3":
        assert ffmpeg_bin, "ffmpeg required for mp3 generation"
        wav_tmp = path.with_suffix(".tmp.wav")
        sf.write(str(wav_tmp), data.T, sample_rate, format="WAV", subtype="PCM_16")
        subprocess.run(
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-y",
                "-i",
                str(wav_tmp),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(path),
            ],
            check=True,
        )
        wav_tmp.unlink(missing_ok=True)
    elif suffix == ".m4a":
        assert ffmpeg_bin, "ffmpeg required for m4a generation"
        wav_tmp = path.with_suffix(".tmp.wav")
        sf.write(str(wav_tmp), data.T, sample_rate, format="WAV", subtype="PCM_16")
        subprocess.run(
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-y",
                "-i",
                str(wav_tmp),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(path),
            ],
            check=True,
        )
        wav_tmp.unlink(missing_ok=True)
    elif suffix == ".mp4":
        assert ffmpeg_bin, "ffmpeg required for mp4 video generation"
        wav_tmp = path.with_suffix(".tmp.wav")
        sf.write(str(wav_tmp), data.T, sample_rate, format="WAV", subtype="PCM_16")
        duration = data.shape[1] / sample_rate
        subprocess.run(
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=navy:s=64x64:d={duration:.3f}",
                "-i",
                str(wav_tmp),
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(path),
            ],
            check=True,
        )
        wav_tmp.unlink(missing_ok=True)
    else:
        raise ValueError(f"Unsupported test suffix {suffix}")


def _mock_ffprobe_response(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration: object = "1.0",
    samples: object = "48000",
    channels: object = 1,
    sample_rate: object = "48000",
    stream_index: object = 0,
    format_name: object = "wav",
    codec_name: object = "pcm_s16le",
    extra_streams: int = 0,
) -> None:
    streams: list[dict[str, object]] = [
        {
            "index": stream_index,
            "codec_type": "audio",
            "codec_name": codec_name,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
            "nb_samples": samples,
            "bits_per_raw_sample": "16",
        }
    ]
    streams.extend({"index": i + 1, "codec_type": "video"} for i in range(extra_streams))
    payload = {
        "streams": streams,
        "format": {"format_name": format_name, "duration": duration},
    }
    encoded = subprocess.CompletedProcess(
        ["ffprobe"], 0, stdout=json.dumps(payload).encode("utf-8"), stderr=b""
    )
    monkeypatch.setattr(shutil, "which", lambda name: name)
    monkeypatch.setattr(probe_module, "_run_bounded_capture", lambda *_args, **_kwargs: encoded)


# ---------------------------------------------------------------------------
# Dimension 1: Format Matrix & MP4 Audio Extraction (Mono & Stereo)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ffmpeg_bin, reason="ffmpeg required for container tests")
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize(
    "suffix",
    [".wav", ".aiff", ".aif", ".flac", ".mp3", ".m4a", ".mp4"],
)
def test_six_format_families_decode_mono_and_stereo_to_memmap(
    tmp_path: Path, suffix: str, channels: int
) -> None:
    """Every declared format in mono and stereo passes probe, contract validation,
    and chunked disk-backed memmap decoding. MP4 extracts the audio stream cleanly.
    """
    sr = 48_000
    audio_data = _generate_sine_audio(duration_s=0.25, sample_rate=sr, channels=channels)
    media_path = tmp_path / f"test_{channels}ch{suffix}"
    _write_media(media_path, audio_data, sample_rate=sr)

    # 1. Probe
    probe = probe_audio(media_path, max_channels=MAX_INPUT_CHANNELS)
    assert probe.channels == channels
    assert probe.sample_rate == sr
    assert probe.samples > 0
    assert probe.duration_s > 0.1

    # 2. Pipeline Contract Validation
    _validate_natural_media_contract(media_path, probe)

    # 3. Disk-Backed Planar Decode
    memmap_path = tmp_path / f"decoded_{channels}ch_{suffix}.f32"
    buf = decode_audio_to_memmap(probe, memmap_path, chunk_samples=2048)

    assert isinstance(buf.data, np.memmap)
    assert buf.channels == channels
    assert buf.sample_rate == sr
    assert buf.data.shape == (channels, buf.samples)
    # Check signal correlation with original sine wave
    for ch in range(channels):
        orig = audio_data[ch, : min(audio_data.shape[1], buf.samples)]
        dec = buf.data[ch, : len(orig)]
        # For lossy codecs (mp3/aac), allow small compression difference; correlation > 0.90
        corr = float(np.corrcoef(orig, dec)[0, 1])
        assert corr > 0.90, f"Channel {ch} correlation {corr:.3f} is too low for {suffix}"


@pytest.mark.skipif(not ffmpeg_bin, reason="ffmpeg required for video container tests")
def test_mp4_video_container_extracts_audio_and_ignores_video_frames(tmp_path: Path) -> None:
    """When an MP4 has a video stream (Stream 0) and an audio stream (Stream 1),
    probe discovers audio_stream_index=1, and decode extracts only the audio stream.
    """
    sr = 48_000
    audio_data = _generate_sine_audio(duration_s=0.5, sample_rate=sr, channels=2)
    mp4_path = tmp_path / "video_podcast.mp4"
    _write_media(mp4_path, audio_data, sample_rate=sr)

    probe = probe_audio(mp4_path)
    assert probe.audio_stream_index == 1
    assert probe.channels == 2

    # Verify decode maps stream 1 and discards video
    buf = decode_audio(probe)
    assert buf.channels == 2
    assert buf.sample_rate == sr
    assert buf.samples > sr * 0.4


@pytest.mark.parametrize(
    ("name", "actual_format"),
    [
        ("fake.wav", "mp3"),
        ("fake.wav", "flac"),
        ("fake.mp3", "wav"),
        ("fake.flac", "aiff"),
        ("fake.aiff", "mp3"),
        ("fake.m4a", "wav"),
        ("fake.ogg", "ogg"),
        ("no_ext", "wav"),
    ],
)
def test_disguised_or_unsupported_containers_fail_contract_validation(
    tmp_path: Path, name: str, actual_format: str
) -> None:
    """Files with renamed extensions or unsupported containers fail closed."""
    path = tmp_path / name
    probe = AudioProbeResult(
        path=path,
        format_name=actual_format,
        codec_name="pcm_s16le",
        sample_rate=48_000,
        channels=1,
        duration_s=1.0,
        samples=48_000,
        bit_depth=16,
        sha256="0" * 64,
        audio_stream_index=0,
    )
    with pytest.raises(MediaPreflightError) as raised:
        _validate_natural_media_contract(path, probe)
    assert raised.value.reason is MediaPreflightReason.UNSUPPORTED_FORMAT


# ---------------------------------------------------------------------------
# Dimension 2: Channel Layout Boundaries (1, 2 vs 0, 3, 5.1, 7.1, 8, 16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [3, 5, 6, 7, 8, 16])
def test_probe_and_contract_reject_multichannel_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: int
) -> None:
    """Multichannel layouts (3ch, 5.1/6ch, 7.1/8ch, 16ch ambisonics) are rejected."""
    path = tmp_path / f"surround_{channels}ch.wav"
    path.write_bytes(b"dummy_riff_data")

    _mock_ffprobe_response(monkeypatch, channels=channels)
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path, max_channels=MAX_INPUT_CHANNELS)
    assert raised.value.reason is MediaPreflightReason.UNSUPPORTED_CHANNEL_LAYOUT

    probe = AudioProbeResult(
        path=path,
        format_name="wav",
        codec_name="pcm_s16le",
        sample_rate=48_000,
        channels=channels,
        duration_s=1.0,
        samples=48_000,
        bit_depth=16,
        sha256="0" * 64,
        audio_stream_index=0,
    )
    with pytest.raises(MediaPreflightError) as raised_contract:
        _validate_natural_media_contract(path, probe)
    assert raised_contract.value.reason is MediaPreflightReason.UNSUPPORTED_CHANNEL_LAYOUT


def test_probe_rejects_zero_channels_as_malformed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "zero_ch.wav"
    path.write_bytes(b"dummy")
    _mock_ffprobe_response(monkeypatch, channels=0)
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.MALFORMED_METADATA


@pytest.mark.skipif(not ffmpeg_bin, reason="ffmpeg required for video container tests")
def test_video_only_mp4_container_has_no_audio_stream(tmp_path: Path) -> None:
    """An MP4 file with only video and zero audio tracks fails with NO_AUDIO_STREAM."""
    silent_mp4 = tmp_path / "silent_video.mp4"
    subprocess.run(
        [
            ffmpeg_bin,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(silent_mp4),
        ],
        check=True,
    )
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(silent_mp4)
    assert raised.value.reason is MediaPreflightReason.NO_AUDIO_STREAM


# ---------------------------------------------------------------------------
# Dimension 3: Exact 6-Hour Boundary (Exact Limit Tests)
# ---------------------------------------------------------------------------


def test_exact_six_hour_duration_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Duration boundary is exact:
    - 21,600.0s is accepted.
    - 21,600.001s is rejected with DURATION_LIMIT.
    - 21,601.0s is rejected with DURATION_LIMIT.
    """
    path = tmp_path / "six_hours.wav"
    path.write_bytes(b"source")

    # 1. Exactly 21,600.0s -> Accepted
    _mock_ffprobe_response(
        monkeypatch,
        duration=str(MAX_INPUT_DURATION_S),
        samples=str(int(MAX_INPUT_DURATION_S * 48_000)),
    )
    result = probe_audio(path, max_duration_s=MAX_INPUT_DURATION_S)
    assert result.duration_s == 21600.0
    _validate_natural_media_contract(path, result)

    # 2. 21,600.001s -> Rejected
    _mock_ffprobe_response(
        monkeypatch,
        duration="21600.001",
        samples=str(int(21600.001 * 48_000)),
    )
    with pytest.raises(MediaPreflightError) as raised_1ms:
        probe_audio(path, max_duration_s=MAX_INPUT_DURATION_S)
    assert raised_1ms.value.reason is MediaPreflightReason.DURATION_LIMIT

    # 3. 21,601.0s -> Rejected
    _mock_ffprobe_response(
        monkeypatch,
        duration="21601.0",
        samples=str(int(21601.0 * 48_000)),
    )
    with pytest.raises(MediaPreflightError) as raised_1s:
        probe_audio(path, max_duration_s=MAX_INPUT_DURATION_S)
    assert raised_1s.value.reason is MediaPreflightReason.DURATION_LIMIT


def test_exact_six_hour_sample_count_boundary(tmp_path: Path) -> None:
    """Sample count ceiling:
    - Exactly 21,600 * 48,000 = 1,036,800,000 samples is accepted.
    - 1,036,800,001 samples is rejected with DURATION_LIMIT.
    """
    path = tmp_path / "boundary_samples.wav"
    sr = 48_000
    exact_max_samples = int(MAX_INPUT_DURATION_S * sr)

    probe_ok = AudioProbeResult(
        path=path,
        format_name="wav",
        codec_name="pcm_s16le",
        sample_rate=sr,
        channels=1,
        duration_s=MAX_INPUT_DURATION_S,
        samples=exact_max_samples,
        bit_depth=16,
        sha256="0" * 64,
        audio_stream_index=0,
    )
    _validate_natural_media_contract(path, probe_ok)

    probe_excess = AudioProbeResult(
        path=path,
        format_name="wav",
        codec_name="pcm_s16le",
        sample_rate=sr,
        channels=1,
        duration_s=MAX_INPUT_DURATION_S,
        samples=exact_max_samples + 1,
        bit_depth=16,
        sha256="0" * 64,
        audio_stream_index=0,
    )
    with pytest.raises(MediaPreflightError) as raised:
        _validate_natural_media_contract(path, probe_excess)
    assert raised.value.reason is MediaPreflightReason.DURATION_LIMIT


def test_streaming_decode_independent_sample_ceiling_terminates_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a container lies about its duration (claiming small duration) but the
    stream emits audio beyond MAX_STREAM_DECODE_DURATION_S, iter_decode_audio
    enforces the independent ceiling and terminates the child process with RESOURCE_BOMB.
    """
    # Scale down ceiling for fast deterministic testing
    monkeypatch.setattr(decode_module, "MAX_STREAM_DECODE_DURATION_S", 2)
    fake_probe = AudioProbeResult(
        path=tmp_path / "streaming_bomb.wav",
        format_name="wav",
        codec_name="pcm_f32le",
        sample_rate=1,
        channels=1,
        duration_s=1.0,
        samples=1,
        bit_depth=32,
        sha256="0" * 64,
        audio_stream_index=0,
    )

    raw_frames = np.ones(5, dtype=np.float32).tobytes()

    class _MockStdout:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self.closed = False

        def read1(self, _size: int) -> bytes:
            d = self._data
            self._data = b""
            return d

        def read(self, size: int) -> bytes:
            return self.read1(size)

        def close(self) -> None:
            self.closed = True

    class _MockProcess:
        def __init__(self) -> None:
            self.stdout = _MockStdout(raw_frames)
            self.stderr = None
            self.returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, _timeout: float = 0) -> int:
            return 0

    class _MockSupervisor:
        def __init__(self) -> None:
            self.process = _MockProcess()
            self.killed = False

        def terminate_tree(self, _grace: float = 0.5) -> None:
            self.killed = True

        def close(self) -> None:
            pass

    monkeypatch.setattr("hawavoclean.audio.decode.shutil.which", lambda _name: "/bin/ffmpeg")
    monkeypatch.setattr(
        "hawavoclean.audio.decode.ProcessSupervisor.spawn", lambda *_a, **_kw: _MockSupervisor()
    )

    with pytest.raises(MediaPreflightError) as raised:
        list(iter_decode_audio(fake_probe, chunk_samples=1, timeout_s=2.0))
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB
    assert "ceiling of 2 samples" in str(raised.value)


# ---------------------------------------------------------------------------
# Dimension 4: Exact 8 GB Boundary (Exact Limit Tests)
# ---------------------------------------------------------------------------


def test_exact_eight_gib_file_size_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File size boundary is exact:
    - 8,589,934,592 bytes (8 GiB) is accepted.
    - 8,589,934,593 bytes is rejected immediately with FILE_TOO_LARGE.
    """
    path = tmp_path / "boundary_size.wav"
    path.write_bytes(b"content")
    real_stat = Path.stat

    # 1. Test probe_audio boundary
    def mock_stat_exact(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        res = real_stat(self, *args, **kwargs)
        if self == path:
            return os.stat_result(
                (
                    res.st_mode,
                    res.st_ino,
                    res.st_dev,
                    res.st_nlink,
                    res.st_uid,
                    res.st_gid,
                    MAX_INPUT_FILE_BYTES,
                    res.st_atime,
                    res.st_mtime,
                    res.st_ctime,
                )
            )
        return res

    monkeypatch.setattr(Path, "stat", mock_stat_exact)
    _mock_ffprobe_response(monkeypatch)
    result = probe_audio(path, max_file_size_bytes=MAX_INPUT_FILE_BYTES)
    assert result is not None

    def mock_stat_exceeded(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        res = real_stat(self, *args, **kwargs)
        if self == path:
            return os.stat_result(
                (
                    res.st_mode,
                    res.st_ino,
                    res.st_dev,
                    res.st_nlink,
                    res.st_uid,
                    res.st_gid,
                    MAX_INPUT_FILE_BYTES + 1,
                    res.st_atime,
                    res.st_mtime,
                    res.st_ctime,
                )
            )
        return res

    monkeypatch.setattr(Path, "stat", mock_stat_exceeded)
    monkeypatch.setattr(
        probe_module, "hash_file", lambda _p: pytest.fail("oversized file was hashed")
    )
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path, max_file_size_bytes=MAX_INPUT_FILE_BYTES)
    assert raised.value.reason is MediaPreflightReason.FILE_TOO_LARGE


def test_pinned_source_rejects_over_eight_gib_before_copy_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PinnedSource refuses files larger than 8 GiB before reading, copying, or hashing."""
    source_path = tmp_path / "oversized_source.wav"
    source_path.write_bytes(b"small_file")

    real_fstat = os.fstat
    real_lstat = os.lstat
    target_name = source_path.name

    class _StatProxy:
        def __init__(self, base: Any, size: int) -> None:
            self._base = base
            self.st_size = size

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

    def fake_fstat(fd: int) -> Any:
        res = real_fstat(fd)
        return _StatProxy(res, MAX_INPUT_FILE_BYTES + 1)

    def fake_lstat(path_obj: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> Any:
        res = real_lstat(path_obj)
        if str(path_obj).endswith(target_name):
            return _StatProxy(res, MAX_INPUT_FILE_BYTES + 1)
        return res

    monkeypatch.setattr(os, "fstat", fake_fstat)
    monkeypatch.setattr(os, "lstat", fake_lstat)

    staging = tmp_path / "staging"
    with pytest.raises(MediaPreflightError) as raised:
        PinnedSource.create(
            source_path,
            staging_root=staging,
            max_file_size_bytes=MAX_INPUT_FILE_BYTES,
        )
    assert raised.value.reason is MediaPreflightReason.FILE_TOO_LARGE
    # Ensure no snapshot directory was created or populated
    assert not list(staging.glob("source-pin-*"))


# ---------------------------------------------------------------------------
# Dimension 5: Malformed Container Headers (All 6 Formats)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suffix", "corrupt_bytes"),
    [
        (".wav", b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00corrupt_data"),
        (".wav", b"RIFF\x24\x00\x00\x00WAVEdata\xff\xff\xff\xfftruncated"),
        (".aiff", b"FORM\x00\x00\x00\x00AIFFCOMM\x00\x00corrupt"),
        (".flac", b"fLaC\x00\x00\x00\x22corrupt_flac_metadata_block"),
        (".mp3", b"\xff\xfb\x00\x00corrupt_mp3_frame_header_without_sync"),
        (".m4a", b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00moov\x00\x00\x00\x04bad_box"),
        (".mp4", b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommoov\xff\xff\xff\xff"),
    ],
)
def test_malformed_container_headers_fail_closed_with_preflight_error(
    tmp_path: Path, suffix: str, corrupt_bytes: bytes
) -> None:
    """Malformed or truncated container headers for all 6 formats fail closed cleanly
    with MediaPreflightError, never hanging or raising unhandled exceptions.
    """
    corrupt_file = tmp_path / f"corrupt_media{suffix}"
    corrupt_file.write_bytes(corrupt_bytes)

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(corrupt_file)

    assert raised.value.reason in (
        MediaPreflightReason.PROBE_FAILED,
        MediaPreflightReason.MALFORMED_METADATA,
        MediaPreflightReason.NO_AUDIO_STREAM,
    )


# ---------------------------------------------------------------------------
# Dimension 6: Decompression Bombs & Resource Bombs
# ---------------------------------------------------------------------------


def test_decompression_bomb_duration_sample_count_amplification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file declaring a 1-second duration but 100 hours of samples is rejected as a resource bomb."""
    path = tmp_path / "sample_bomb.wav"
    path.write_bytes(b"dummy")
    _mock_ffprobe_response(
        monkeypatch,
        duration="1.0",
        samples=str(48_000 * 3600 * 10),
    )
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB


def test_resource_bomb_extreme_stream_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container declaring more than MAX_STREAM_RECORDS (64) streams is rejected as a resource bomb."""
    path = tmp_path / "extreme_streams.mp4"
    path.write_bytes(b"dummy")
    _mock_ffprobe_response(monkeypatch, extra_streams=MAX_STREAM_RECORDS + 1)
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB


def test_resource_bomb_child_pipe_overflow() -> None:
    """Child processes emitting unbounded pipe output (> 1 MiB) are killed and raise RESOURCE_BOMB."""
    started = time.monotonic()
    with pytest.raises(MediaPreflightError) as raised:
        probe_module._run_bounded_capture(
            [
                shutil.which("python3") or "python3",
                "-c",
                "import sys; sys.stdout.buffer.write(b'A' * 2000000)",
            ],
            timeout_s=5.0,
            stdout_limit=MAX_FFPROBE_METADATA_BYTES,
        )
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB
    assert time.monotonic() - started < 3.0


@pytest.mark.parametrize("duration", ["nan", "inf", "-inf", "-1.0"])
def test_probe_rejects_non_finite_or_negative_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: str
) -> None:
    path = tmp_path / "bad_num.wav"
    path.write_bytes(b"dummy")
    _mock_ffprobe_response(monkeypatch, duration=duration, samples="48000")
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.MALFORMED_METADATA


@pytest.mark.parametrize("rate", [4_000, 384_000])
def test_probe_rejects_sub_floor_and_ultrasonic_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: int
) -> None:
    """Sample rates below 8 kHz or above 192 kHz are rejected."""
    path = tmp_path / f"rate_{rate}.wav"
    path.write_bytes(b"dummy")
    _mock_ffprobe_response(monkeypatch, sample_rate=rate, samples=rate)
    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path, max_sample_rate=192_000)
    assert raised.value.reason is MediaPreflightReason.UNSUPPORTED_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Dimension 7: Disk-Full Preflight
# ---------------------------------------------------------------------------


def test_pinned_source_aborts_when_scratch_disk_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PinnedSource validates available disk space before and during snapshot copy."""
    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"hello world")

    staging = tmp_path / "staging"
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10**9, used=10**9, free=0),
    )

    with pytest.raises(PreflightError, match="Insufficient scratch space"):
        PinnedSource.create(
            source_path,
            staging_root=staging,
            max_file_size_bytes=MAX_INPUT_FILE_BYTES,
        )
    assert not list(staging.glob("source-pin-*"))


def test_decode_to_memmap_aborts_and_cleans_up_on_disk_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """decode_audio_to_memmap halts and unlinks temporary channel files when disk space disappears."""
    fake_probe = AudioProbeResult(
        path=tmp_path / "source.wav",
        format_name="wav",
        codec_name="pcm_f32le",
        sample_rate=48_000,
        channels=2,
        duration_s=1.0,
        samples=48_000,
        bit_depth=32,
        sha256="0" * 64,
        audio_stream_index=0,
    )

    def one_chunk(*_a: object, **_kw: object) -> Any:
        yield AudioBuffer(np.zeros((2, 1024), dtype=np.float32), 48_000)

    monkeypatch.setattr(decode_module, "iter_decode_audio", one_chunk)
    monkeypatch.setattr(
        "hawavoclean.audio.decode.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10**9, used=10**9, free=0),
    )

    destination = tmp_path / "output_pcm.f32"
    with pytest.raises(PreflightError, match="Insufficient scratch space"):
        decode_audio_to_memmap(fake_probe, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.channel-*.tmp"))


# ---------------------------------------------------------------------------
# Dimension 8: Memory Boundedness (No Full-File Memory Paths)
# ---------------------------------------------------------------------------


def test_memmap_decode_chunk_granularity_and_planar_shape(tmp_path: Path) -> None:
    """decode_audio_to_memmap yields AudioBuffers with shape (channels, samples)
    backed by np.memmap without loading the full file into heap RAM.
    """
    sr = 48_000
    channels = 2
    duration_s = 0.5
    samples = int(duration_s * sr)
    data = _generate_sine_audio(duration_s=duration_s, sample_rate=sr, channels=channels)
    wav_path = tmp_path / "memmap_test.wav"
    _write_media(wav_path, data, sample_rate=sr)

    probe = probe_audio(wav_path)
    dest = tmp_path / "planar_stage.f32"

    chunk_size = 4096
    buf = decode_audio_to_memmap(probe, dest, chunk_samples=chunk_size)

    assert isinstance(buf.data, np.memmap)
    assert buf.data.shape == (channels, samples)
    assert dest.stat().st_size == channels * samples * np.dtype(np.float32).itemsize
    np.testing.assert_allclose(buf.data, data, atol=1e-4)
