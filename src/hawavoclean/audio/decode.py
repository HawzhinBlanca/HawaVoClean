"""Safe float32 audio decoding using FFmpeg or soundfile.

:func:`decode_audio` reads a whole file; :func:`decode_audio_window` reads a
time window only (ffmpeg input seek / ``soundfile`` frame range), so a few
seconds out of a multi-hour file costs a few megabytes instead of gigabytes;
:func:`iter_decode_audio` reads the whole file as a stream of chunks, which is
what a reduction over a three-hour file needs to stay inside a few hundred MB.
"""

import math
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import soundfile as sf

from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.errors import InvalidUserInputError

# One chunk of a streamed whole-file decode: 512 Ki frames is ~11 s at 48 kHz,
# 2 MB of float32 per channel. Swept against a 3-hour file: 256 Ki costs
# 88 MB / 33.6 s, 512 Ki 118 MB / 32.1 s, 2 Mi 269 MB / 31.8 s, 4 Mi 366 MB /
# 31.5 s — so this is the knee, where the reduction is already as fast as it
# gets and the footprint is still small.
DECODE_CHUNK_SAMPLES = 512 * 1024

# Seeking a lossy stream drops the MDCT overlap the first frame needs, so the
# first ~2048 decoded samples after a seek are wrong (measured on AAC: peaks
# off by 0.27 full scale). Start the decode a quarter second early and throw
# the warm-up away — 48 kB of extra decode buys an exact window.
WINDOW_PREROLL_S = 0.25


def _check_decoded(
    arr: np.ndarray[Any, np.dtype[np.float32]], file_path: object
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Reject NaN/Inf and absurd float amplitude in a decoded buffer."""
    if not np.all(np.isfinite(arr)):
        raise InvalidUserInputError(
            f"Decoded audio from {file_path} contains NaN or Infinite values."
        )

    max_amp = float(np.max(np.abs(arr)))
    if max_amp > 10.0:  # float headroom sanity
        raise InvalidUserInputError(
            f"Decoded audio has abnormal float amplitude ({max_amp:.2f} > 10.0), likely corrupt."
        )
    return arr


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
            "-nostdin",  # never read the terminal: a stray 'q' aborted decodes silently
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
                stdin=subprocess.DEVNULL,
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
    _check_decoded(arr, file_path)

    return AudioBuffer(data=arr, sample_rate=probe.sample_rate)


def window_sample_bounds(probe: AudioProbeResult, start_s: float, end_s: float) -> tuple[int, int]:
    """Sample range ``[start, end)`` covered by the time window ``[start_s, end_s)``.

    Both edges are placed on the sample grid with ``round(t * sample_rate)``,
    so the ffmpeg and soundfile paths agree sample for sample. ``end`` is
    clamped to the file length; ``start`` past the end of the file is an
    error, as is a non-finite, negative or empty window.
    """
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise InvalidUserInputError(
            f"Window bounds must be finite numbers, got start_s={start_s}, end_s={end_s}"
        )
    if start_s < 0.0:
        raise InvalidUserInputError(f"Window start_s must be >= 0, got {start_s}")
    if end_s <= start_s:
        raise InvalidUserInputError(
            f"Window end_s must be greater than start_s, got start_s={start_s}, end_s={end_s}"
        )
    sample_rate = probe.sample_rate
    total = int(probe.samples)
    start = int(round(start_s * sample_rate))
    end = int(round(end_s * sample_rate))
    if total > 0:
        if start >= total:
            raise InvalidUserInputError(
                f"Window starts at {start_s}s, at or past the end of {probe.path} "
                f"({total / sample_rate:.3f}s)"
            )
        end = min(end, total)
    end = max(end, start + 1)
    return start, end


def decode_audio_window(
    probe: AudioProbeResult,
    start_s: float,
    end_s: float,
    timeout_s: float = 120.0,
) -> AudioBuffer:
    """Decode only ``[start_s, end_s)`` of a file to a float32 AudioBuffer.

    The whole point is that the cost scales with the window, not the file:
    ffmpeg gets ``-ss`` *before* ``-i`` (input seek, so the demuxer jumps
    instead of decoding everything up to the window) plus ``-t`` for the
    length, and the soundfile fallback reads a frame range. Sanity checks
    and error semantics are the same as :func:`decode_audio`.

    The window is anchored to the *container* timeline — the same timeline the
    player's playhead and ``GET /api/audio`` range requests use. For PCM that
    is identical to the index into a full decode. A lossy container whose
    packet durations do not sum to its decoded sample count (measured: 71 of
    4435 packets declared 1023 instead of 1024 ticks in the project's m4a.mp4
    test file) drifts a little against a whole-file decode — 1.5 ms over 95 s
    there — because the file itself says two different things, not because the
    seek is sloppy.
    """
    file_path = probe.path
    start, end = window_sample_bounds(probe, start_s, end_s)
    want_samples = end - start
    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin:
        seek_start = max(0, start - int(round(WINDOW_PREROLL_S * probe.sample_rate)))
        prefix = start - seek_start
        # From the very beginning, seek nothing at all: an explicit "-ss 0" makes
        # ffmpeg hand back an mp4's encoder-priming samples that a plain decode
        # trims away, which would shift the window by ~1024 samples.
        seek_args = ["-ss", f"{seek_start / probe.sample_rate:.9f}"] if seek_start > 0 else []
        cmd = [
            ffmpeg_bin,
            "-nostdin",  # never read the terminal: a stray 'q' aborted decodes silently
            "-v",
            "error",
            # Fast seek: BEFORE -i so ffmpeg seeks the input instead of
            # decoding and discarding everything ahead of the window.
            *seek_args,
            "-i",
            str(file_path),
            "-t",
            f"{(prefix + want_samples) / probe.sample_rate:.9f}",
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
                stdin=subprocess.DEVNULL,
            )
            raw_bytes = res.stdout
        except subprocess.TimeoutExpired as e:
            raise InvalidUserInputError(
                f"FFmpeg decoding timed out after {timeout_s}s: {file_path}"
            ) from e
        except Exception as e:
            raise InvalidUserInputError(f"FFmpeg failed to decode {file_path}: {e}") from e

        frame_bytes = probe.channels * 4
        # A container whose declared duration overshoots the real stream hands
        # back fewer frames than asked; never more than pre-roll + window.
        decoded = min(len(raw_bytes) // frame_bytes, prefix + want_samples)
        actual_samples = decoded - prefix
        if actual_samples <= 0:
            raise InvalidUserInputError(
                f"Decoded zero samples from {file_path} window [{start_s}, {end_s})"
            )
        flat_arr = np.frombuffer(
            raw_bytes[prefix * frame_bytes : decoded * frame_bytes], dtype=np.float32
        )
        arr: np.ndarray[Any, np.dtype[np.float32]] = np.ascontiguousarray(
            flat_arr.reshape((actual_samples, probe.channels)).T, dtype=np.float32
        )
    else:
        try:
            data, sr = sf.read(
                str(file_path), start=start, stop=end, dtype="float32", always_2d=True
            )
            if sr != probe.sample_rate:
                raise InvalidUserInputError(
                    f"Decoded sample rate {sr} does not match probe {probe.sample_rate}"
                )
            arr = np.ascontiguousarray(data.T, dtype=np.float32)  # shape (channels, samples)
        except Exception as e:
            raise InvalidUserInputError(f"Failed to decode audio file {file_path}: {e}") from e
        if arr.shape[1] == 0:
            raise InvalidUserInputError(
                f"Decoded zero samples from {file_path} window [{start_s}, {end_s})"
            )

    _check_decoded(arr, file_path)

    return AudioBuffer(data=arr, sample_rate=probe.sample_rate)


def iter_decode_audio(
    probe: AudioProbeResult,
    chunk_samples: int = DECODE_CHUNK_SAMPLES,
    timeout_s: float = 1800.0,
) -> Iterator[AudioBuffer]:
    """Decode a whole file as a sequence of contiguous ``AudioBuffer`` chunks.

    The sample stream is *identical* to :func:`decode_audio` — the same ffmpeg
    command, no seek, so no lossy-container pre-roll question arises — but the
    consumer never holds more than one chunk, so a reduction over a three-hour
    file costs a chunk instead of the file. Chunks are back to back and every
    chunk except the last carries exactly ``chunk_samples`` frames.

    Error semantics match :func:`decode_audio`: a decoder failure, a timeout or
    a file that yields no samples all raise :class:`InvalidUserInputError`, and
    every chunk goes through the same NaN/Inf/amplitude sanity check. ffmpeg's
    stderr goes to a temporary file rather than a pipe, because a pipe nobody
    drains until the end would deadlock on a file that logs a lot.
    """
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")
    file_path = probe.path
    ffmpeg_bin = shutil.which("ffmpeg")
    frame_bytes = probe.channels * 4
    total = 0

    def _buffer(raw: bytes) -> AudioBuffer:
        flat = np.frombuffer(raw, dtype=np.float32)
        arr: np.ndarray[Any, np.dtype[np.float32]] = np.ascontiguousarray(
            flat.reshape((len(raw) // frame_bytes, probe.channels)).T, dtype=np.float32
        )
        _check_decoded(arr, file_path)
        return AudioBuffer(data=arr, sample_rate=probe.sample_rate)

    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-nostdin",  # never read the terminal: a stray 'q' aborted decodes silently
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
        deadline = time.monotonic() + timeout_s
        want = chunk_samples * frame_bytes
        with tempfile.TemporaryFile() as errfile:
            proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
                cmd, stdout=subprocess.PIPE, stderr=errfile, stdin=subprocess.DEVNULL
            )
            try:
                assert proc.stdout is not None
                pending = b""
                while True:
                    block = proc.stdout.read(want - len(pending))
                    if time.monotonic() > deadline:
                        raise InvalidUserInputError(
                            f"FFmpeg decoding timed out after {timeout_s}s: {file_path}"
                        )
                    if not block:
                        break
                    pending += block
                    if len(pending) < want:
                        continue  # short read from the pipe, not end of stream
                    total += len(pending) // frame_bytes
                    yield _buffer(pending)
                    pending = b""
                whole = len(pending) - len(pending) % frame_bytes
                if whole:
                    total += whole // frame_bytes
                    yield _buffer(pending[:whole])
                proc.stdout.close()
                code = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
            finally:
                if proc.poll() is None:  # pragma: no cover - abandoned generator
                    proc.kill()
                    proc.wait()
                if proc.stdout is not None and not proc.stdout.closed:
                    proc.stdout.close()
            if code != 0:
                errfile.seek(0)
                detail = errfile.read().decode("utf-8", "replace").strip()[-500:]
                raise InvalidUserInputError(
                    f"FFmpeg failed to decode {file_path}: exit {code} {detail}"
                )
    else:
        try:
            with sf.SoundFile(str(file_path)) as f:
                if f.samplerate != probe.sample_rate:
                    raise InvalidUserInputError(
                        f"Decoded sample rate {f.samplerate} does not match probe "
                        f"{probe.sample_rate}"
                    )
                while True:
                    data = f.read(chunk_samples, dtype="float32", always_2d=True)
                    if data.shape[0] == 0:
                        break
                    total += int(data.shape[0])
                    arr = np.ascontiguousarray(data.T, dtype=np.float32)
                    _check_decoded(arr, file_path)
                    yield AudioBuffer(data=arr, sample_rate=probe.sample_rate)
        except InvalidUserInputError:
            raise
        except Exception as e:
            raise InvalidUserInputError(f"Failed to decode audio file {file_path}: {e}") from e

    if total == 0:
        raise InvalidUserInputError(f"Decoded zero samples from {file_path}")
