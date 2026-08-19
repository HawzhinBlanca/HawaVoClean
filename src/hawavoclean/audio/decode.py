"""Safe float32 audio decoding using FFmpeg or soundfile."""

import shutil
import subprocess
from typing import Any

import numpy as np
import soundfile as sf

from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.errors import InvalidUserInputError


def decode_audio(
    probe: AudioProbeResult,
    timeout_s: float = 120.0,
) -> AudioBuffer:
    """Decode audio file to float32 AudioBuffer with strict byte and sanity checks."""
    file_path = probe.path
    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-i",
            str(file_path),
            "-map",
            f"0:{probe.audio_stream_index}",
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(probe.sample_rate),
            "-ac",
            str(probe.channels),
            "pipe:1",
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=timeout_s,
            )
            raw_bytes = res.stdout
        except subprocess.TimeoutExpired as e:
            raise InvalidUserInputError(
                f"FFmpeg decoding timed out after {timeout_s}s: {file_path}"
            ) from e
        except Exception as e:
            raise InvalidUserInputError(f"FFmpeg failed to decode {file_path}: {e}") from e

        # If byte length slightly differs due to container headers vs streams, parse actual
        actual_samples = len(raw_bytes) // (probe.channels * 4)
        if actual_samples == 0:
            raise InvalidUserInputError(f"Decoded zero samples from {file_path}")

        flat_arr = np.frombuffer(raw_bytes[: actual_samples * probe.channels * 4], dtype=np.float32)
        # Reshape to (samples, channels) then transpose to (channels, samples)
        arr: np.ndarray[Any, np.dtype[np.float32]] = np.ascontiguousarray(
            flat_arr.reshape((actual_samples, probe.channels)).T, dtype=np.float32
        )
    else:
        try:
            data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
            if sr != probe.sample_rate:
                raise InvalidUserInputError(
                    f"Decoded sample rate {sr} does not match probe {probe.sample_rate}"
                )
            arr = np.ascontiguousarray(data.T, dtype=np.float32)  # shape (channels, samples)
        except Exception as e:
            raise InvalidUserInputError(f"Failed to decode audio file {file_path}: {e}") from e

    # Sanity checks: NaN, Inf, extreme amplitude
    if not np.all(np.isfinite(arr)):
        raise InvalidUserInputError(
            f"Decoded audio from {file_path} contains NaN or Infinite values."
        )

    max_amp = float(np.max(np.abs(arr)))
    if max_amp > 10.0:  # float headroom sanity
        raise InvalidUserInputError(
            f"Decoded audio has abnormal float amplitude ({max_amp:.2f} > 10.0), likely corrupt."
        )

    return AudioBuffer(data=arr, sample_rate=probe.sample_rate)
