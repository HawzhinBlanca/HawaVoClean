"""Production media boundary tests without allocating long or large media."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.audio.probe as probe_module
from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import MAX_STREAM_RECORDS, probe_audio
from hawavoclean.audio.types import AudioProbeResult
from hawavoclean.errors import MediaPreflightError, MediaPreflightReason
from hawavoclean.pipeline import (
    MAX_INPUT_DURATION_S,
    MAX_INPUT_FILE_BYTES,
    _source_identity,
    _validate_natural_media_contract,
)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_probe_capture_terminates_excessive_child_output(stream: str) -> None:
    descriptor = 1 if stream == "stdout" else 2
    command = [
        sys.executable,
        "-c",
        (f"import os,time; os.write({descriptor}, b'x' * 65536); time.sleep(30)"),
    ]
    started = time.monotonic()
    with pytest.raises(MediaPreflightError) as raised:
        probe_module._run_bounded_capture(
            command,
            timeout_s=10,
            stdout_limit=4096,
            stderr_limit=4096,
        )
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB
    assert time.monotonic() - started < 5


def _media(path: Path, **overrides: object) -> AudioProbeResult:
    format_name = overrides.pop("format_name", "wav")
    assert isinstance(format_name, str)
    values: dict[str, object] = {
        "path": path,
        "format_name": format_name,
        "codec_name": "pcm_s16le",
        "sample_rate": 48_000,
        "channels": 1,
        "duration_s": 1.0,
        "samples": 48_000,
        "bit_depth": 16,
        "sha256": "a" * 64,
        "audio_stream_index": 0,
    }
    values.update(overrides)
    return AudioProbeResult(**values)  # type: ignore[arg-type]


def _ffprobe_result(
    *,
    duration: object = "1.0",
    samples: object = "48000",
    channels: object = 1,
    sample_rate: object = "48000",
    stream_index: object = 0,
    format_name: object = "wav",
    codec_name: object = "pcm_s16le",
    extra_streams: int = 0,
) -> subprocess.CompletedProcess[str]:
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
    streams.extend({"index": index + 1, "codec_type": "video"} for index in range(extra_streams))
    payload = {
        "streams": streams,
        "format": {"format_name": format_name, "duration": duration},
    }
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")


def _mock_ffprobe(
    monkeypatch: pytest.MonkeyPatch, result: subprocess.CompletedProcess[str]
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: name)
    encoded = subprocess.CompletedProcess(
        result.args,
        result.returncode,
        stdout=result.stdout.encode("utf-8"),
        stderr=result.stderr.encode("utf-8"),
    )
    monkeypatch.setattr(probe_module, "_run_bounded_capture", lambda *_args, **_kwargs: encoded)


@pytest.mark.parametrize(
    ("name", "format_name"),
    [
        ("speech.wav", "wav"),
        ("speech.AIFF", "aiff"),
        ("speech.aif", "AIFF"),
        ("speech.flac", "flac"),
        ("speech.mp3", "mp3"),
        ("speech.m4a", "mov,mp4,m4a,3gp,3g2,mj2"),
        ("phone.m4a.mp4", "mov,mp4,m4a,3gp,3g2,mj2"),
    ],
)
def test_natural_contract_accepts_only_declared_container_families(
    tmp_path: Path, name: str, format_name: str
) -> None:
    path = tmp_path / name
    _validate_natural_media_contract(path, _media(path, format_name=format_name))


@pytest.mark.parametrize(
    ("name", "format_name"),
    [("speech.ogg", "ogg"), ("renamed.wav", "mp3"), ("no-extension", "wav")],
)
def test_natural_contract_rejects_unsupported_or_renamed_container(
    tmp_path: Path, name: str, format_name: str
) -> None:
    path = tmp_path / name
    with pytest.raises(MediaPreflightError) as raised:
        _validate_natural_media_contract(path, _media(path, format_name=format_name))
    assert raised.value.reason is MediaPreflightReason.UNSUPPORTED_FORMAT


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"sha256": "not-a-digest"}, MediaPreflightReason.MALFORMED_METADATA),
        ({"audio_stream_index": 2048}, MediaPreflightReason.RESOURCE_BOMB),
        ({"bit_depth": 4096}, MediaPreflightReason.MALFORMED_METADATA),
    ],
)
def test_natural_contract_rejects_malformed_probe_identity_and_extremes(
    tmp_path: Path, overrides: dict[str, object], reason: MediaPreflightReason
) -> None:
    path = tmp_path / "speech.wav"
    with pytest.raises(MediaPreflightError) as raised:
        _validate_natural_media_contract(path, _media(path, **overrides))
    assert raised.value.reason is reason


def test_natural_contract_rejects_probe_for_different_source(tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    with pytest.raises(MediaPreflightError) as raised:
        _validate_natural_media_contract(path, _media(tmp_path / "other.wav"))
    assert raised.value.reason is MediaPreflightReason.SOURCE_CHANGED


def test_probe_refuses_over_eight_gib_before_hash_or_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.wav"
    path.write_bytes(b"tiny")
    real_stat = Path.stat

    def reported_large(candidate: Path, *args: Any, **kwargs: Any) -> object:
        value = real_stat(candidate, *args, **kwargs)
        if candidate == path:
            return SimpleNamespace(st_mode=value.st_mode, st_size=MAX_INPUT_FILE_BYTES + 1)
        return value

    monkeypatch.setattr(Path, "stat", reported_large)
    monkeypatch.setattr(
        probe_module,
        "hash_file",
        lambda _path: pytest.fail("oversized source was hashed"),
    )
    monkeypatch.setattr(
        probe_module,
        "_run_bounded_capture",
        lambda *_args, **_kwargs: pytest.fail("oversized source reached ffprobe"),
    )

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path, max_file_size_bytes=MAX_INPUT_FILE_BYTES)
    assert raised.value.reason is MediaPreflightReason.FILE_TOO_LARGE


def test_probe_rejects_more_than_six_hours_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "long.wav"
    path.write_bytes(b"source")
    duration = MAX_INPUT_DURATION_S + 0.001
    _mock_ffprobe(
        monkeypatch,
        _ffprobe_result(duration=str(duration), samples=str(round(duration * 48_000))),
    )

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path, max_duration_s=MAX_INPUT_DURATION_S)
    assert raised.value.reason is MediaPreflightReason.DURATION_LIMIT


def test_exact_six_hour_boundary_is_accepted_without_allocating_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "six-hours.wav"
    path.write_bytes(b"source")
    _mock_ffprobe(
        monkeypatch,
        _ffprobe_result(
            duration=str(MAX_INPUT_DURATION_S),
            samples=str(round(MAX_INPUT_DURATION_S * 48_000)),
        ),
    )
    result = probe_audio(path, max_duration_s=MAX_INPUT_DURATION_S)
    assert result.duration_s == MAX_INPUT_DURATION_S


def test_probe_rejects_multichannel_layout_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surround.wav"
    path.write_bytes(b"source")
    _mock_ffprobe(monkeypatch, _ffprobe_result(channels=6))

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path, max_channels=2)
    assert raised.value.reason is MediaPreflightReason.UNSUPPORTED_CHANNEL_LAYOUT


@pytest.mark.parametrize("duration", ["nan", "inf", "-inf"])
def test_probe_rejects_non_finite_duration_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: str
) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(b"source")
    _mock_ffprobe(monkeypatch, _ffprobe_result(duration=duration, samples="48000"))

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.MALFORMED_METADATA


def test_probe_rejects_extreme_stream_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "many-streams.mp4"
    path.write_bytes(b"source")
    _mock_ffprobe(
        monkeypatch,
        _ffprobe_result(extra_streams=MAX_STREAM_RECORDS),
    )

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB


def test_probe_rejects_duration_sample_count_amplification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bomb.wav"
    path.write_bytes(b"source")
    _mock_ffprobe(monkeypatch, _ffprobe_result(duration="1", samples=str(48_000 * 3_600)))

    with pytest.raises(MediaPreflightError) as raised:
        probe_audio(path)
    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB


def test_durationless_probe_uses_null_sink_and_bounded_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fragmented.mp4"
    path.write_bytes(b"source")
    ffprobe_result = _ffprobe_result(duration="0", samples="0", format_name="mov,mp4,m4a")
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(
                command,
                ffprobe_result.returncode,
                stdout=ffprobe_result.stdout.encode("utf-8"),
                stderr=ffprobe_result.stderr.encode("utf-8"),
            )
        assert command[0] == "ffmpeg"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"out_time_us=2000000\nprogress=end\n",
            stderr=b"",
        )

    monkeypatch.setattr(shutil, "which", lambda name: name)
    monkeypatch.setattr(probe_module, "_run_bounded_capture", run)
    result = probe_audio(path, max_duration_s=MAX_INPUT_DURATION_S)

    assert result.samples == 96_000
    ffmpeg_command = calls[1]
    assert ffmpeg_command[ffmpeg_command.index("-f") :][:2] == ["-f", "null"]
    assert "-progress" in ffmpeg_command
    assert "-t" in ffmpeg_command
    assert "s16le" not in ffmpeg_command and "f32le" not in ffmpeg_command


def test_whole_file_decode_rejects_non_finite_samples(tmp_path: Path) -> None:
    path = tmp_path / "nan.wav"
    values = np.zeros(1024, dtype=np.float32)
    values[64] = np.nan
    sf.write(str(path), values, 48_000, format="WAV", subtype="FLOAT")
    media = probe_audio(path)

    from hawavoclean.errors import InvalidUserInputError

    with pytest.raises(InvalidUserInputError, match="NaN or Infinite"):
        decode_audio(media)


def test_source_identity_changes_when_file_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    path.write_bytes(b"first")
    first = _source_identity(path)
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(b"second version")
    os.replace(replacement, path)
    assert _source_identity(path) != first
