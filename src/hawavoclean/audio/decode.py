"""Safe float32 audio decoding using FFmpeg or soundfile.

:func:`decode_audio` reads a whole file; :func:`decode_audio_window` reads a
time window only (ffmpeg input seek / ``soundfile`` frame range), so a few
seconds out of a multi-hour file costs a few megabytes instead of gigabytes;
:func:`iter_decode_audio` reads the whole file as a stream of chunks, which is
what a reduction over a three-hour file needs to stay inside a few hundred MB.
"""

import contextlib
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import soundfile as sf

from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.errors import (
    InvalidUserInputError,
    MediaPreflightError,
    MediaPreflightReason,
    PreflightError,
)
from hawavoclean.process_supervisor import ProcessSupervisor

# One chunk of a streamed whole-file decode: 512 Ki frames is ~11 s at 48 kHz,
# 2 MB of float32 per channel. Swept against a 3-hour file: 256 Ki costs
# 88 MB / 33.6 s, 512 Ki 118 MB / 32.1 s, 2 Mi 269 MB / 31.8 s, 4 Mi 366 MB /
# 31.5 s — so this is the knee, where the reduction is already as fast as it
# gets and the footprint is still small.
DECODE_CHUNK_SAMPLES = 512 * 1024
DECODE_DISK_SAFETY_MARGIN_BYTES = 500 * 1024 * 1024

# ``probe_audio`` rejects production inputs longer than six hours, but decode
# output is an independent trust boundary: a damaged or hostile container can
# lie consistently in every metadata field.  Streaming decode therefore has
# its own absolute frame ceiling and does not use ``probe.samples`` to decide
# when it has consumed too much data.
MAX_STREAM_DECODE_DURATION_S = 6 * 60 * 60

# A healthy local FFmpeg decode produces bytes continuously.  The whole-job
# deadline bounds total work; this shorter deadline prevents a wedged decoder
# from owning the shared analysis slot for the rest of that period.
STREAM_NO_PROGRESS_TIMEOUT_S = 30.0

# The reader may block inside an OS pipe read, so it runs on a daemon thread
# and reports small blocks through a bounded queue.  Four queued blocks plus
# one pending output chunk keep the memory bound effectively constant.
STREAM_READ_BYTES = 64 * 1024
STREAM_EVENT_QUEUE_SIZE = 4
STREAM_TERMINATION_GRACE_S = 0.25

_StreamEventKind = Literal["data", "eof", "error"]
_StreamEvent = tuple[_StreamEventKind, bytes | BaseException | None]

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
        seek_args = ["-ss", f"{start_s:.9f}"] if start_s > 0 else []
        cmd = [
            ffmpeg_bin,
            "-nostdin",  # never read the terminal: a stray 'q' aborted decodes silently
            "-v",
            "error",
            "-i",
            str(file_path),
            *seek_args,
            "-t",
            f"{want_samples / probe.sample_rate:.9f}",
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
        # back fewer frames than asked; never more than window.
        decoded = min(len(raw_bytes) // frame_bytes, want_samples)
        if decoded <= 0:
            raise InvalidUserInputError(
                f"Decoded zero samples from {file_path} window [{start_s}, {end_s})"
            )
        flat_arr = np.frombuffer(raw_bytes[: decoded * frame_bytes], dtype=np.float32)
        arr: np.ndarray[Any, np.dtype[np.float32]] = np.ascontiguousarray(
            flat_arr.reshape((decoded, probe.channels)).T, dtype=np.float32
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
    *,
    no_progress_timeout_s: float = STREAM_NO_PROGRESS_TIMEOUT_S,
    max_decoded_samples: int | None = None,
) -> Iterator[AudioBuffer]:
    """Decode a whole file as a sequence of contiguous ``AudioBuffer`` chunks.

    The sample stream is *identical* to :func:`decode_audio` — the same ffmpeg
    command, no seek, so no lossy-container pre-roll question arises — but the
    consumer never holds more than one chunk, so a reduction over a three-hour
    file costs a chunk instead of the file. Chunks are back to back and every
    chunk except the last carries exactly ``chunk_samples`` frames.

    Error semantics match :func:`decode_audio`: a decoder failure, a timeout or
    a file that yields no samples all raise :class:`InvalidUserInputError`, and
    every chunk goes through the same NaN/Inf/amplitude sanity check. A decoded
    stream that exceeds ``max_decoded_samples`` is a resource-bomb preflight
    error. When the caller does not provide that test/embedding override, the
    ceiling is six hours at the requested output rate and is deliberately
    independent of untrusted duration/sample-count metadata.

    FFmpeg stdout is drained by one bounded reader thread. The iterator itself
    waits on a bounded queue, so it can enforce both the total deadline and a
    shorter no-progress deadline even while the OS pipe read is blocked. FFmpeg
    is owned by :class:`ProcessSupervisor`; timeout, overflow, generator close
    and decoder failure terminate its complete process boundary. stderr goes
    to a temporary file rather than an undrained pipe.
    """
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(f"timeout_s must be a positive finite number, got {timeout_s}")
    if not math.isfinite(no_progress_timeout_s) or no_progress_timeout_s <= 0.0:
        raise ValueError(
            f"no_progress_timeout_s must be a positive finite number, got {no_progress_timeout_s}"
        )
    if max_decoded_samples is None:
        decoded_sample_ceiling = int(math.ceil(MAX_STREAM_DECODE_DURATION_S * probe.sample_rate))
    else:
        if isinstance(max_decoded_samples, bool) or max_decoded_samples < 1:
            raise ValueError(f"max_decoded_samples must be >= 1, got {max_decoded_samples}")
        decoded_sample_ceiling = int(max_decoded_samples)
    file_path = probe.path
    ffmpeg_bin = shutil.which("ffmpeg")
    frame_bytes = probe.channels * 4
    total = 0
    started_at = time.monotonic()
    deadline = started_at + timeout_s

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
            # Bound upstream decode work as well as checking the actual byte
            # stream below. The extra frame ensures a source beyond the limit
            # is observed as an overflow rather than looking like clean EOF.
            "-t",
            f"{(decoded_sample_ceiling + 1) / probe.sample_rate:.9f}",
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
        want = chunk_samples * frame_bytes
        with tempfile.TemporaryFile() as errfile:
            try:
                supervisor = ProcessSupervisor.spawn(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=errfile,
                    stdin=subprocess.DEVNULL,
                )
                # ProcessSupervisor is also used for text-mode workers and its
                # public annotation reflects that path. This spawn is binary.
                proc = cast(subprocess.Popen[bytes], supervisor.process)
            except OSError as e:
                raise InvalidUserInputError(
                    f"FFmpeg failed to start decoding {file_path}: {e}"
                ) from e

            assert proc.stdout is not None
            stdout = proc.stdout
            events: queue.Queue[_StreamEvent] = queue.Queue(maxsize=STREAM_EVENT_QUEUE_SIZE)
            stop_reader = threading.Event()

            def _emit(event: _StreamEvent) -> bool:
                while not stop_reader.is_set():
                    try:
                        events.put(event, timeout=0.05)
                        return True
                    except queue.Full:
                        continue
                return False

            def _drain_stdout() -> None:
                read_once = getattr(stdout, "read1", stdout.read)
                try:
                    while not stop_reader.is_set():
                        block = read_once(STREAM_READ_BYTES)
                        if not block:
                            _emit(("eof", None))
                            return
                        if not isinstance(block, bytes):
                            raise TypeError("FFmpeg stdout returned non-bytes data in binary mode")
                        if not _emit(("data", block)):
                            return
                except BaseException as exc:
                    _emit(("error", exc))

            reader = threading.Thread(
                target=_drain_stdout,
                name="hawavoclean-ffmpeg-stream-reader",
                daemon=True,
            )
            reader.start()
            code: int | None = None
            pending = bytearray()
            last_progress_at = started_at
            clean_exit = False
            try:
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        raise InvalidUserInputError(
                            f"FFmpeg decoding timed out after {timeout_s}s: {file_path}"
                        )
                    no_progress_deadline = last_progress_at + no_progress_timeout_s
                    if now >= no_progress_deadline:
                        raise InvalidUserInputError(
                            "FFmpeg decoding made no progress for "
                            f"{no_progress_timeout_s}s: {file_path}"
                        )
                    try:
                        event_kind, payload = events.get(
                            timeout=min(deadline - now, no_progress_deadline - now)
                        )
                    except queue.Empty:
                        continue

                    if event_kind == "error":
                        assert isinstance(payload, BaseException)
                        raise InvalidUserInputError(
                            f"FFmpeg stream read failed for {file_path}: {payload}"
                        ) from payload
                    if event_kind == "eof":
                        break
                    assert isinstance(payload, bytes)
                    last_progress_at = time.monotonic()
                    pending.extend(payload)

                    # Count complete decoded frames before yielding any part
                    # of this event. No byte beyond the ceiling is accepted.
                    complete_pending = len(pending) // frame_bytes
                    if total + complete_pending > decoded_sample_ceiling:
                        raise MediaPreflightError(
                            MediaPreflightReason.RESOURCE_BOMB,
                            "Decoded audio exceeded the independent streaming "
                            f"ceiling of {decoded_sample_ceiling:,} samples: {file_path}",
                        )

                    while len(pending) >= want:
                        raw = bytes(pending[:want])
                        del pending[:want]
                        total += chunk_samples
                        yield _buffer(raw)
                        # Time spent by the consumer processing a yielded
                        # chunk is not decoder no-progress time.
                        last_progress_at = time.monotonic()

                whole = len(pending) - len(pending) % frame_bytes
                if whole:
                    total += whole // frame_bytes
                    yield _buffer(bytes(pending[:whole]))
                    last_progress_at = time.monotonic()

                now = time.monotonic()
                if now >= deadline:
                    raise InvalidUserInputError(
                        f"FFmpeg decoding timed out after {timeout_s}s: {file_path}"
                    )
                try:
                    code = proc.wait(
                        timeout=min(
                            deadline - now,
                            max(0.001, last_progress_at + no_progress_timeout_s - now),
                        )
                    )
                except subprocess.TimeoutExpired as e:
                    raise InvalidUserInputError(
                        "FFmpeg decoder stopped producing output but did not exit within "
                        f"{no_progress_timeout_s}s: {file_path}"
                    ) from e
                clean_exit = True
            finally:
                stop_reader.set()
                # On every exceptional path (including GeneratorExit), end
                # the complete decoder process tree before releasing handles.
                if not clean_exit:
                    supervisor.terminate_tree(STREAM_TERMINATION_GRACE_S)
                supervisor.close()
                if not stdout.closed:
                    stdout.close()
                reader.join(timeout=1.0)
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
                    read_started_at = time.monotonic()
                    if read_started_at >= deadline:
                        raise InvalidUserInputError(
                            f"Audio decoding timed out after {timeout_s}s: {file_path}"
                        )
                    data = f.read(chunk_samples, dtype="float32", always_2d=True)
                    read_finished_at = time.monotonic()
                    if read_finished_at - read_started_at > no_progress_timeout_s:
                        raise InvalidUserInputError(
                            "Audio decoding made no progress for "
                            f"{no_progress_timeout_s}s: {file_path}"
                        )
                    if data.shape[0] == 0:
                        break
                    decoded = int(data.shape[0])
                    if total + decoded > decoded_sample_ceiling:
                        raise MediaPreflightError(
                            MediaPreflightReason.RESOURCE_BOMB,
                            "Decoded audio exceeded the independent streaming "
                            f"ceiling of {decoded_sample_ceiling:,} samples: {file_path}",
                        )
                    total += decoded
                    arr = np.ascontiguousarray(data.T, dtype=np.float32)
                    _check_decoded(arr, file_path)
                    yield AudioBuffer(data=arr, sample_rate=probe.sample_rate)
        except InvalidUserInputError:
            raise
        except Exception as e:
            raise InvalidUserInputError(f"Failed to decode audio file {file_path}: {e}") from e

    if total == 0:
        raise InvalidUserInputError(f"Decoded zero samples from {file_path}")


def decode_audio_to_memmap(
    probe: AudioProbeResult,
    output_path: Path | str,
    timeout_s: float = 1800.0,
    *,
    chunk_samples: int = DECODE_CHUNK_SAMPLES,
) -> AudioBuffer:
    """Decode ``probe`` into a planar, disk-backed float32 buffer.

    The decoder itself is :func:`iter_decode_audio`, so hostile-stream limits,
    no-progress deadlines and complete-process-tree cleanup are identical to
    the ordinary streaming path.  Each channel is appended to a private
    temporary file while the decoded length is still unknown.  Once EOF is
    proved, those files are concatenated into one planar backing file and
    mapped as ``(channels, samples)``.  At no point is a file-length PCM object
    materialised on the Python heap.

    Planar layout is intentional.  Natural processing repeatedly takes a
    channel slice; an interleaved mapping would make that slice strided and
    several existing DSP functions would quietly copy the entire channel.
    The returned mapping remains owned by the caller and must stay live until
    processing has finished.
    """

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise InvalidUserInputError(f"Disk-backed decode destination already exists: {destination}")

    channel_paths = [
        destination.with_name(f".{destination.name}.channel-{channel}.tmp")
        for channel in range(probe.channels)
    ]
    handles: list[Any] = []
    handle_stack = contextlib.ExitStack()
    total_samples = 0
    try:
        handles = [
            handle_stack.enter_context(open(path, "xb"))  # noqa: SIM115 - ExitStack owns it
            for path in channel_paths
        ]
        for chunk in iter_decode_audio(
            probe,
            chunk_samples=chunk_samples,
            timeout_s=timeout_s,
        ):
            if chunk.channels != probe.channels or chunk.sample_rate != probe.sample_rate:
                raise InvalidUserInputError(
                    "Streaming decoder changed the declared channel count or sample rate."
                )
            # The temporary channel files and the final planar file coexist
            # during the layout conversion. Reserve the complete projected
            # destination before appending this chunk so an underreported
            # container fails cleanly instead of filling the workspace volume.
            chunk_bytes = chunk.samples * probe.channels * np.dtype(np.float32).itemsize
            projected_bytes = (
                (total_samples + chunk.samples) * probe.channels * np.dtype(np.float32).itemsize
            )
            free_bytes = shutil.disk_usage(destination.parent).free
            required_free = chunk_bytes + projected_bytes + DECODE_DISK_SAFETY_MARGIN_BYTES
            if free_bytes < required_free:
                raise PreflightError(
                    "Insufficient scratch space for disk-backed decode: "
                    f"available {free_bytes / (1024 * 1024):.1f} MiB, "
                    f"required {required_free / (1024 * 1024):.1f} MiB."
                )
            for channel, handle in enumerate(handles):
                # ``tobytes`` is bounded by one decode chunk.  The channel is
                # contiguous inside the chunk returned by iter_decode_audio.
                handle.write(chunk.data[channel].tobytes(order="C"))
            total_samples += chunk.samples
            del chunk

        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
        handle_stack.close()
        handles.clear()

        with open(destination, "xb") as planar:
            for path in channel_paths:
                with open(path, "rb") as source:
                    shutil.copyfileobj(source, planar, length=1024 * 1024)
            planar.flush()
            os.fsync(planar.fileno())

        expected_bytes = total_samples * probe.channels * np.dtype(np.float32).itemsize
        if total_samples <= 0 or destination.stat().st_size != expected_bytes:
            raise InvalidUserInputError(
                "Disk-backed decode produced an empty or structurally incomplete PCM stage."
            )
        mapped: np.memmap[Any, np.dtype[np.float32]] = np.memmap(
            destination,
            dtype=np.float32,
            mode="r+",
            shape=(probe.channels, total_samples),
        )
        return AudioBuffer(data=mapped, sample_rate=probe.sample_rate)
    except OSError as exc:
        handle_stack.close()
        with contextlib.suppress(OSError):
            destination.unlink(missing_ok=True)
        raise PreflightError(f"Disk-backed decode could not write scratch audio: {exc}") from exc
    except Exception:
        handle_stack.close()
        with contextlib.suppress(OSError):
            destination.unlink(missing_ok=True)
        raise
    finally:
        handle_stack.close()
        for path in channel_paths:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
