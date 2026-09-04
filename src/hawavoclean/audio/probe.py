"""Media probe implementation using ffprobe without shell interpolation."""

import json
import math
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, cast

import soundfile as sf

from hawavoclean.audio.types import AudioProbeResult
from hawavoclean.config import InputConfig
from hawavoclean.errors import (
    MediaPreflightError,
    MediaPreflightReason,
    PreflightError,
)
from hawavoclean.hashing import hash_file
from hawavoclean.process_supervisor import ProcessSupervisor


def _declared_supported_rates() -> list[int]:
    """The rate envelope ``[input]`` declares, straight from the schema."""
    factory = InputConfig.model_fields["supported_sample_rates"].default_factory
    assert factory is not None  # declared with a default_factory; see config.py
    return [int(r) for r in factory()]  # type: ignore[call-arg]


#: Floor of the accepted input-rate envelope. Derived from
#: ``input.supported_sample_rates`` rather than restated here, so the
#: configuration is the source of truth and the two cannot drift apart. The
#: ceiling is ``input.max_sample_rate``, passed in by the caller.
MIN_SUPPORTED_SAMPLE_RATE: int = min(_declared_supported_rates())

# Metadata is not media. A probe response larger than this is either an
# unsupported pathological container or a resource-amplification attempt.
MAX_FFPROBE_METADATA_BYTES = 1024 * 1024
MAX_STREAM_RECORDS = 64
MAX_METADATA_TEXT_CHARS = 256
MAX_METADATA_INTEGER = (1 << 63) - 1
_CAPTURE_READ_BYTES = 64 * 1024


def _run_bounded_capture(
    command: Sequence[str],
    *,
    timeout_s: float,
    stdout_limit: int = MAX_FFPROBE_METADATA_BYTES,
    stderr_limit: int = MAX_FFPROBE_METADATA_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    """Run a probe child without ever buffering unbounded pipe output.

    ``subprocess.run(capture_output=True)`` enforces no memory ceiling while a
    child is running.  Probe tools process hostile containers, so both pipes
    are drained concurrently into fixed-size buffers.  Crossing either cap
    terminates the complete supervised process tree immediately.
    """

    if timeout_s <= 0 or stdout_limit < 1 or stderr_limit < 1:
        raise ValueError("probe capture limits and timeout must be positive")
    supervisor = ProcessSupervisor.spawn(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    process = cast(subprocess.Popen[bytes], supervisor.process)
    assert process.stdout is not None and process.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflowed: list[str] = []
    overflow_lock = threading.Lock()

    def drain(name: str, stream: BinaryIO, limit: int) -> None:
        try:
            while True:
                block = stream.read(_CAPTURE_READ_BYTES)
                if not block:
                    return
                if not isinstance(block, bytes):
                    raise TypeError(f"probe {name} returned non-bytes data")
                remaining = limit + 1 - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(block[:remaining])
                if len(buffers[name]) > limit or len(block) > remaining:
                    with overflow_lock:
                        if not overflowed:
                            overflowed.append(name)
                    supervisor.kill_tree()
                    return
        except (OSError, ValueError):
            # Closing a pipe while terminating an overflowing/timed-out child
            # is an expected end state. The main thread owns the verdict.
            return

    readers = [
        threading.Thread(
            target=drain,
            args=("stdout", process.stdout, stdout_limit),
            name="hawavoclean-probe-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=drain,
            args=("stderr", process.stderr, stderr_limit),
            name="hawavoclean-probe-stderr",
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout_s
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                supervisor.terminate_tree(0.5)
                raise subprocess.TimeoutExpired(list(command), timeout_s)
            time.sleep(0.01)
        for reader in readers:
            reader.join(timeout=1.0)
        if overflowed:
            raise MediaPreflightError(
                MediaPreflightReason.RESOURCE_BOMB,
                f"probe {overflowed[0]} exceeded {stdout_limit if overflowed[0] == 'stdout' else stderr_limit:,} bytes",
            )
        return subprocess.CompletedProcess(
            list(command),
            int(process.returncode),
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
        )
    finally:
        supervisor.close()
        for stream in (process.stdout, process.stderr):
            stream.close()
        for reader in readers:
            reader.join(timeout=1.0)


def _metadata_int(value: object, field: str, *, minimum: int = 0) -> int:
    """Parse a bounded metadata integer without accepting bools or floats."""
    if isinstance(value, (bool, float)) or not isinstance(value, (int, str)):
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} is not an integer.",
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} is not a valid integer.",
        ) from exc
    if parsed < minimum or parsed > MAX_METADATA_INTEGER:
        raise MediaPreflightError(
            MediaPreflightReason.RESOURCE_BOMB,
            f"Audio metadata field {field!r} is outside the supported range.",
        )
    return parsed


def _metadata_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} is not a valid number.",
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} is not a valid number.",
        ) from exc
    if not math.isfinite(parsed):
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} must be finite.",
        )
    return parsed


def _metadata_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_METADATA_TEXT_CHARS:
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} is missing or malformed.",
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio metadata field {field!r} contains control characters.",
        )
    return value


def _count_samples_by_decoding(
    file_path: Path,
    ffmpeg_bin: str | None,
    sample_rate: int,
    audio_stream_index: int,
    max_duration_s: float | None,
) -> int:
    """Measure a duration-less stream without capturing its decoded payload.

    FFmpeg writes audio to its null muxer and reports only its progress clock.
    ``-stats_period`` keeps that text bounded for long inputs; ``-t`` makes the
    production duration cap an actual decode-work cap rather than a check
    applied after a hostile stream has expanded indefinitely.
    """
    if not ffmpeg_bin:
        return 0
    duration_args = ["-t", f"{max_duration_s + 1.0:.6f}"] if max_duration_s is not None else []
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-stats_period",
        "60",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        str(file_path),
        "-map",
        f"0:{audio_stream_index}",
        "-vn",
        *duration_args,
        "-f",
        "null",
        "-",
    ]
    try:
        res = _run_bounded_capture(
            cmd,
            timeout_s=300,
        )
    except (MediaPreflightError, OSError, subprocess.SubprocessError):
        return 0
    if res.returncode != 0:
        return 0
    try:
        stdout = res.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return 0
    out_time_us = 0
    for line in stdout.splitlines():
        if line.startswith("out_time_us="):
            try:
                out_time_us = max(out_time_us, int(line.removeprefix("out_time_us=")))
            except ValueError:
                return 0
    if out_time_us <= 0:
        return 0
    duration_s = out_time_us / 1_000_000.0
    return int(round(duration_s * sample_rate))


def probe_audio(
    path: Path | str,
    max_sample_rate: int = 48000,
    supported_sample_rates: Sequence[int] | None = None,
    *,
    max_file_size_bytes: int | None = None,
    max_duration_s: float | None = None,
    max_channels: int | None = None,
) -> AudioProbeResult:
    """Probe an audio file safely and extract structured metadata.

    The accepted rate envelope is enforced here, before a single sample is
    decoded: ``min(supported_sample_rates)`` is the floor and
    ``max_sample_rate`` the ceiling, both straight from ``[input]``. Passing
    ``None`` uses the envelope the configuration schema declares by default.
    """
    rates = (
        _declared_supported_rates()
        if supported_sample_rates is None
        else list(supported_sample_rates)
    )
    if not rates:
        raise PreflightError(
            "No supported sample rates configured: input.supported_sample_rates is empty."
        )
    rate_floor = min(int(r) for r in rates)
    file_path = Path(path).resolve()
    try:
        source_stat = file_path.stat()
    except OSError as exc:
        raise MediaPreflightError(
            MediaPreflightReason.NOT_FOUND,
            f"Input audio file does not exist or cannot be read: {file_path}",
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise MediaPreflightError(
            MediaPreflightReason.NOT_REGULAR_FILE,
            f"Input audio source is not a regular file: {file_path}",
        )
    if source_stat.st_size <= 0:
        raise MediaPreflightError(
            MediaPreflightReason.EMPTY_FILE,
            f"Input audio file is empty: {file_path}",
        )
    if max_file_size_bytes is not None and source_stat.st_size > max_file_size_bytes:
        raise MediaPreflightError(
            MediaPreflightReason.FILE_TOO_LARGE,
            f"Input file is {source_stat.st_size:,} bytes; the maximum is "
            f"{max_file_size_bytes:,} bytes.",
        )

    # Prefer ffprobe if installed
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=format_name,duration,bit_rate:stream=index,codec_type,codec_name,sample_rate,channels,bits_per_raw_sample,duration,nb_samples",
            "-of",
            "json",
            str(file_path),
        ]
        try:
            res = _run_bounded_capture(cmd, timeout_s=30)
            if res.returncode != 0:
                raise subprocess.CalledProcessError(
                    res.returncode,
                    cmd,
                    output=res.stdout,
                    stderr=res.stderr,
                )
            data = json.loads(res.stdout.decode("utf-8", errors="strict"))
            if not isinstance(data, dict):
                raise MediaPreflightError(
                    MediaPreflightReason.MALFORMED_METADATA,
                    "ffprobe returned a non-object metadata document.",
                )
        except MediaPreflightError:
            raise
        except Exception as e:
            raise MediaPreflightError(
                MediaPreflightReason.PROBE_FAILED,
                f"ffprobe failed to probe {file_path}: {e}",
            ) from e

        streams = data.get("streams", [])
        if not isinstance(streams, list):
            raise MediaPreflightError(
                MediaPreflightReason.MALFORMED_METADATA,
                "ffprobe streams metadata is not a list.",
            )
        if len(streams) > MAX_STREAM_RECORDS:
            raise MediaPreflightError(
                MediaPreflightReason.RESOURCE_BOMB,
                f"Container declares {len(streams)} streams; at most {MAX_STREAM_RECORDS} are accepted.",
            )
        if any(not isinstance(stream, dict) for stream in streams):
            raise MediaPreflightError(
                MediaPreflightReason.MALFORMED_METADATA,
                "ffprobe returned a malformed stream record.",
            )
        audio_streams = [st for st in streams if st.get("codec_type") == "audio"]
        if not audio_streams:
            raise MediaPreflightError(
                MediaPreflightReason.NO_AUDIO_STREAM,
                f"No audio stream found in {file_path} ({len(streams)} non-audio stream(s) present)",
            )
        # First AUDIO stream — containers frequently list video first.
        audio_stream = audio_streams[0]
        audio_stream_index = _metadata_int(audio_stream.get("index", 0), "stream.index")
        fmt = data.get("format", {})
        if not isinstance(fmt, dict):
            raise MediaPreflightError(
                MediaPreflightReason.MALFORMED_METADATA,
                "ffprobe format metadata is not an object.",
            )

        sample_rate = _metadata_int(audio_stream.get("sample_rate", 0), "sample_rate")
        channels = _metadata_int(audio_stream.get("channels", 0), "channels")
        codec_name = _metadata_text(audio_stream.get("codec_name"), "codec_name")
        format_name = _metadata_text(fmt.get("format_name"), "format_name")

        bit_depth: int | None = None
        if "bits_per_raw_sample" in audio_stream and audio_stream["bits_per_raw_sample"] != "N/A":
            bit_depth = _metadata_int(audio_stream["bits_per_raw_sample"], "bits_per_raw_sample")
        elif "bits_per_sample" in audio_stream and audio_stream["bits_per_sample"] != "N/A":
            bit_depth = _metadata_int(audio_stream["bits_per_sample"], "bits_per_sample")
        elif codec_name == "pcm_f32le" or codec_name == "pcm_s32le":
            bit_depth = 32
        elif codec_name == "pcm_s24le" or codec_name == "pcm_s24be":
            bit_depth = 24
        elif codec_name == "pcm_s16le" or codec_name == "pcm_s16be":
            bit_depth = 16

        duration_s = _metadata_float(
            audio_stream.get("duration") or fmt.get("duration") or 0.0,
            "duration",
        )
        nb_samples = audio_stream.get("nb_samples")
        if nb_samples and nb_samples != "N/A":
            samples = _metadata_int(nb_samples, "nb_samples")
        else:
            samples = int(round(duration_s * sample_rate))
    else:
        # Fallback to soundfile info
        audio_stream_index = 0
        try:
            info = sf.info(str(file_path))
            sample_rate = info.samplerate
            channels = info.channels
            duration_s = float(info.duration)
            samples = int(info.frames)
            format_name = info.format
            codec_name = info.subtype
            bit_depth = None
        except Exception as e:
            raise MediaPreflightError(
                MediaPreflightReason.PROBE_FAILED,
                f"Neither ffprobe nor soundfile could read {file_path}: {e}",
            ) from e

    if sample_rate <= 0 or channels <= 0:
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Invalid audio stream in {file_path}: rate={sample_rate}, channels={channels}",
        )
    if max_channels is not None and channels > max_channels:
        raise MediaPreflightError(
            MediaPreflightReason.UNSUPPORTED_CHANNEL_LAYOUT,
            f"Input has {channels} channels; only mono and stereo are supported.",
        )
    if samples <= 0:
        # Streamed containers (WebM/Matroska from MediaRecorder, OBS, live
        # captures) carry no duration. Count the samples with a null decode
        # rather than rejecting a perfectly decodable file.
        samples = _count_samples_by_decoding(
            file_path,
            shutil.which("ffmpeg"),
            sample_rate,
            audio_stream_index,
            max_duration_s,
        )
        if samples <= 0:
            raise MediaPreflightError(
                MediaPreflightReason.MALFORMED_METADATA,
                f"Audio stream in {file_path} has no decodable samples",
            )
        duration_s = samples / sample_rate

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise MediaPreflightError(
            MediaPreflightReason.MALFORMED_METADATA,
            f"Audio duration must be a positive finite value, got {duration_s!r}.",
        )
    if samples <= 0 or samples > MAX_METADATA_INTEGER:
        raise MediaPreflightError(
            MediaPreflightReason.RESOURCE_BOMB,
            f"Audio sample count is outside the supported range: {samples!r}.",
        )
    sample_duration_s = samples / sample_rate
    mismatch_s = abs(sample_duration_s - duration_s)
    mismatch_limit_s = max(2.0, min(10.0, duration_s * 0.005))
    if mismatch_s > mismatch_limit_s:
        raise MediaPreflightError(
            MediaPreflightReason.RESOURCE_BOMB,
            "Audio sample count and declared duration disagree by "
            f"{mismatch_s:.3f}s (limit {mismatch_limit_s:.3f}s).",
        )
    if max_duration_s is not None and (
        duration_s > max_duration_s or sample_duration_s > max_duration_s
    ):
        raise MediaPreflightError(
            MediaPreflightReason.DURATION_LIMIT,
            f"Input duration is {max(duration_s, sample_duration_s):.3f}s; the maximum is "
            f"{max_duration_s:.3f}s (six hours).",
        )

    if sample_rate < rate_floor:
        raise MediaPreflightError(
            MediaPreflightReason.UNSUPPORTED_SAMPLE_RATE,
            f"Input sample rate {sample_rate} Hz is below the minimum supported {rate_floor} Hz.",
        )
    if sample_rate > max_sample_rate:
        raise MediaPreflightError(
            MediaPreflightReason.UNSUPPORTED_SAMPLE_RATE,
            f"Input sample rate {sample_rate} Hz exceeds maximum supported {max_sample_rate} Hz. Ultrasonic rates are rejected in V1.",
        )

    # Hash after every metadata/decode probe has finished. The production
    # pipeline supplies a private read-only snapshot, so this digest names the
    # exact bytes all later decode stages consume. Standalone callers also get
    # a digest taken as close as possible to the returned metadata rather than
    # one captured before ffprobe opened the path.
    try:
        file_sha256 = hash_file(file_path)
    except OSError as exc:
        raise MediaPreflightError(
            MediaPreflightReason.SOURCE_CHANGED,
            f"Audio source changed or became unreadable while it was probed: {file_path}",
        ) from exc

    return AudioProbeResult(
        path=file_path,
        format_name=format_name,
        codec_name=codec_name,
        sample_rate=sample_rate,
        channels=channels,
        duration_s=duration_s,
        samples=samples,
        bit_depth=bit_depth,
        sha256=file_sha256,
        audio_stream_index=audio_stream_index,
    )
