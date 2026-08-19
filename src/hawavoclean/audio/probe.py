"""Media probe implementation using ffprobe without shell interpolation."""

import json
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

from hawavoclean.audio.types import AudioProbeResult
from hawavoclean.errors import InvalidUserInputError, PreflightError
from hawavoclean.hashing import hash_file

MIN_SUPPORTED_SAMPLE_RATE = 8000


def _count_samples_by_decoding(file_path: Path, have_ffmpeg: bool) -> int:
    """Decode to a null sink and count output samples (for duration-less containers)."""
    ffmpeg_bin = shutil.which("ffmpeg") if have_ffmpeg else None
    if not ffmpeg_bin:
        return 0
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(file_path),
        "-vn",
        "-f",
        "s16le",
        "-ac",
        "1",
        "pipe:1",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=300, stdin=subprocess.DEVNULL)
    except Exception:
        return 0
    return len(res.stdout) // 2


def probe_audio(path: Path | str, max_sample_rate: int = 48000) -> AudioProbeResult:
    """Probe an audio file safely and extract structured metadata."""
    file_path = Path(path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise InvalidUserInputError(f"Input audio file does not exist: {file_path}")

    file_sha256 = hash_file(file_path)

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
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(res.stdout)
        except Exception as e:
            raise InvalidUserInputError(f"ffprobe failed to probe {file_path}: {e}") from e

        streams = data.get("streams", [])
        audio_streams = [st for st in streams if st.get("codec_type") == "audio"]
        if not audio_streams:
            raise InvalidUserInputError(
                f"No audio stream found in {file_path} ({len(streams)} non-audio stream(s) present)"
            )
        # First AUDIO stream — containers frequently list video first.
        audio_stream = audio_streams[0]
        audio_stream_index = int(audio_stream.get("index", 0))
        fmt = data.get("format", {})

        sample_rate = int(audio_stream.get("sample_rate", 0))
        channels = int(audio_stream.get("channels", 0))
        codec_name = str(audio_stream.get("codec_name", "unknown"))
        format_name = str(fmt.get("format_name", "unknown"))

        bit_depth: int | None = None
        if "bits_per_raw_sample" in audio_stream and audio_stream["bits_per_raw_sample"] != "N/A":
            bit_depth = int(audio_stream["bits_per_raw_sample"])
        elif "bits_per_sample" in audio_stream and audio_stream["bits_per_sample"] != "N/A":
            bit_depth = int(audio_stream["bits_per_sample"])
        elif codec_name == "pcm_f32le" or codec_name == "pcm_s32le":
            bit_depth = 32
        elif codec_name == "pcm_s24le" or codec_name == "pcm_s24be":
            bit_depth = 24
        elif codec_name == "pcm_s16le" or codec_name == "pcm_s16be":
            bit_depth = 16

        duration_s = float(audio_stream.get("duration") or fmt.get("duration") or 0.0)
        nb_samples = audio_stream.get("nb_samples")
        if nb_samples and nb_samples != "N/A":
            samples = int(nb_samples)
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
            raise PreflightError(
                f"Neither ffprobe nor soundfile could read {file_path}: {e}"
            ) from e

    if sample_rate <= 0 or channels <= 0:
        raise InvalidUserInputError(
            f"Invalid audio stream in {file_path}: rate={sample_rate}, channels={channels}"
        )
    if samples <= 0:
        # Streamed containers (WebM/Matroska from MediaRecorder, OBS, live
        # captures) carry no duration. Count the samples with a null decode
        # rather than rejecting a perfectly decodable file.
        samples = _count_samples_by_decoding(file_path, ffprobe_bin is not None)
        if samples <= 0:
            raise InvalidUserInputError(f"Audio stream in {file_path} has no decodable samples")
        duration_s = samples / sample_rate

    if sample_rate < MIN_SUPPORTED_SAMPLE_RATE:
        raise InvalidUserInputError(
            f"Input sample rate {sample_rate} Hz is below the minimum supported "
            f"{MIN_SUPPORTED_SAMPLE_RATE} Hz."
        )
    if sample_rate > max_sample_rate:
        raise InvalidUserInputError(
            f"Input sample rate {sample_rate} Hz exceeds maximum supported {max_sample_rate} Hz. Ultrasonic rates are rejected in V1."
        )

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
