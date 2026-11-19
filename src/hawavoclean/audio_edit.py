"""Bounded, explicit sound separation followed by optional dialogue cleanup.

The language model never runs this module. The authenticated server submits a
closed request through stdin and supervises this process with its normal jobs.

Long clips (> 90 s, up to 60 minutes) stream through disk-backed 60 s chunks
with 2.0 s raised-cosine overlap-add crossfading so RAM stays strictly bounded.
Optional surgical ``time_range`` edits process only the padded context window
and splice back with a 30 ms equal-power raised-cosine crossfade while keeping
100% of samples outside ``[start_s, end_s]`` bit-identical to the original.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import platform
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import scipy
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hawavoclean.alignment.delay import estimate_gcc_phat_delay
from hawavoclean.audio.decode import iter_decode_audio
from hawavoclean.audio.encode import _wav_container_format, encode_audio_streaming
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.resample import resample_audio
from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.config import HawaVoCleanConfig
from hawavoclean.finishing.limiter import apply_lookahead_limiter_to_memmap
from hawavoclean.finishing.loudness import StreamingLoudnessMeter
from hawavoclean.finishing.truepeak import EDGE as TRUE_PEAK_EDGE
from hawavoclean.finishing.truepeak import oversampled_peak_envelope_window
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.pipeline import run_pipeline
from hawavoclean.platform_fs import is_reparse_or_symlink
from hawavoclean.progress import ProgressEvent
from hawavoclean.provenance import deterministic_settings, runtime_versions
from hawavoclean.publication import public_output_path, publish_output_generation
from hawavoclean.record_bundle import create_processing_record
from hawavoclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_build_metadata,
    current_release_metadata,
)
from hawavoclean.report.writer import serialize_json_report
from hawavoclean.runtime import evict_memmap_pages
from hawavoclean.separation import (
    MAX_DURATION_S,
    MAX_FILE_SIZE_BYTES,
    WORKER_MAX_DURATION_S,
    separate_audio,
)
from hawavoclean.server.contracts import SeparationStrategyV1

CHUNK_THRESHOLD_S = 90.0
CHUNK_DURATION_S = 60.0
CHUNK_OVERLAP_S = 2.0
TIME_RANGE_PAD_S = 1.5
TIME_RANGE_FADE_S = 0.030
_STREAM_BLOCK_FRAMES = 65536
_STATS_BLOCK_FRAMES = 1 << 20
_DELAY_ANALYSIS_MAX_S = 10.0


def _validate_time_range(value: tuple[float, float]) -> tuple[float, float]:
    start_s, end_s = float(value[0]), float(value[1])
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise ValueError("time_range bounds must be finite numbers")
    if start_s < 0.0 or end_s <= start_s:
        raise ValueError("time_range must satisfy 0 <= start_s < end_s")
    return (start_s, end_s)


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str
    original_input_path: str
    output_path: str
    overwrite: bool = False
    options: SeparationStrategyV1
    bundle_path: str | None = None
    time_range: tuple[float, float] | None = None
    strength: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_edit_controls(self) -> EditRequest:
        if self.time_range is not None:
            _validate_time_range(self.time_range)
        return self


def _resolve_edit_controls(request: EditRequest) -> tuple[tuple[float, float] | None, float]:
    raw_range: Any = (
        request.time_range
        if request.time_range is not None
        else getattr(request.options, "time_range", None)
    )
    time_range: tuple[float, float] | None = None
    if raw_range is not None:
        if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
            raise ValueError("time_range must be a (start_s, end_s) pair")
        time_range = _validate_time_range((float(raw_range[0]), float(raw_range[1])))

    if "strength" in request.model_fields_set:
        strength = float(request.strength)
    else:
        opt_strength = getattr(request.options, "strength", None)
        strength = float(request.strength if opt_strength is None else opt_strength)
    if not math.isfinite(strength) or not (0.0 < strength <= 1.0):
        raise ValueError("strength must satisfy 0.0 < strength <= 1.0")
    return time_range, strength


def _event(stage: str, progress: float, message: str) -> None:
    print(
        json.dumps({"event": "progress", "stage": stage, "progress": progress, "message": message}),
        flush=True,
    )


def _create_memmap(path: Path, channels: int, samples: int) -> np.memmap[Any, np.dtype[np.float32]]:
    byte_count = channels * samples * np.dtype(np.float32).itemsize
    with open(path, "xb") as handle:
        handle.truncate(byte_count)
    return np.memmap(path, dtype=np.float32, mode="r+", shape=(channels, samples))


def _close_memmap(arr: np.memmap[Any, np.dtype[np.float32]] | None) -> None:
    if arr is None:
        return
    with contextlib.suppress(OSError, ValueError, AttributeError):
        arr.flush()
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        with contextlib.suppress(OSError, ValueError):
            mmap_obj.close()


def _hash_memmap(arr: np.memmap[Any, np.dtype[np.float32]]) -> str:
    digest = hashlib.sha256()
    samples = int(arr.shape[1])
    for start in range(0, samples, _STREAM_BLOCK_FRAMES):
        end = min(samples, start + _STREAM_BLOCK_FRAMES)
        digest.update(np.ascontiguousarray(arr[:, start:end], dtype=np.float32).tobytes())
        evict_memmap_pages(arr, start, end)
    return digest.hexdigest()


def _decode_to_wav(probe: AudioProbeResult, decoded_path: Path) -> tuple[int, int, int]:
    """Stream decoded float32 audio directly to disk while enforcing frame ceiling."""
    max_frames = int(MAX_DURATION_S * probe.sample_rate)
    estimated_frames = max(int(probe.samples), int(math.ceil(probe.duration_s * probe.sample_rate)))
    container = _wav_container_format(probe.channels, max(1, estimated_frames), "FLOAT")
    frames = 0
    stream = iter_decode_audio(probe, chunk_samples=_STREAM_BLOCK_FRAMES)
    try:
        with sf.SoundFile(
            str(decoded_path),
            mode="w",
            samplerate=probe.sample_rate,
            channels=probe.channels,
            subtype="FLOAT",
            format=container,
        ) as sink:
            for block in stream:
                if block.channels != probe.channels or block.channels not in (1, 2):
                    raise ValueError("Decoded recording exceeds the sound-separation limit")
                frames += block.samples
                if frames > max_frames:
                    raise ValueError("Decoded recording exceeds the sound-separation limit")
                sink.write(block.data.T)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if frames < 1:
        raise ValueError("The recording has no audio samples")
    return probe.sample_rate, probe.channels, frames


def _stats_from_wav(path_label: str, digest: str, wav_path: Path) -> MediaStats:
    """Measure loudness and 4x true peak directly from a WAV file with bounded RAM."""
    with sf.SoundFile(str(wav_path)) as audio:
        sr = int(audio.samplerate)
        channels = int(audio.channels)
        samples = int(audio.frames)
        meter = StreamingLoudnessMeter(sr, channels)
        max_tp = 0.0
        for start in range(0, samples, _STATS_BLOCK_FRAMES):
            end = min(samples, start + _STATS_BLOCK_FRAMES)
            halo_lo = max(0, start - TRUE_PEAK_EDGE)
            halo_hi = min(samples, end + TRUE_PEAK_EDGE)
            audio.seek(halo_lo)
            raw = audio.read(halo_hi - halo_lo, dtype="float32", always_2d=True)
            chunk_cf = np.ascontiguousarray(raw.T, dtype=np.float32)
            core_lo = start - halo_lo
            core_hi = core_lo + (end - start)
            meter.push(chunk_cf[:, core_lo:core_hi])
            env = oversampled_peak_envelope_window(chunk_cf, 4, core_lo, core_hi)
            max_tp = max(max_tp, float(np.max(env)))
    measurement = meter.finish(max_tp)
    return MediaStats(
        path=path_label,
        sha256=digest,
        sample_rate=sr,
        channels=channels,
        samples=samples,
        duration_s=samples / sr,
        integrated_lufs=measurement.integrated_lufs,
        true_peak_dbtp=measurement.true_peak_dbtp,
    )


def _extract_wav_slice(decoded_wav: Path, dst_wav: Path, start_frame: int, num_frames: int) -> None:
    """Copy a bounded frame slice from ``decoded_wav`` into a classic float32 WAV."""
    with sf.SoundFile(str(decoded_wav)) as src:
        container = _wav_container_format(src.channels, max(1, num_frames), "FLOAT")
        src.seek(start_frame)
        remaining = num_frames
        with sf.SoundFile(
            str(dst_wav),
            mode="w",
            samplerate=src.samplerate,
            channels=src.channels,
            subtype="FLOAT",
            format=container,
        ) as dst:
            while remaining > 0:
                take = min(remaining, _STREAM_BLOCK_FRAMES)
                block = src.read(take, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    raise ValueError("Decoded recording ended unexpectedly during slice read")
                dst.write(block)
                remaining -= block.shape[0]


def _chunk_spans(
    total_frames: int, chunk_frames: int, overlap_frames: int, max_worker_frames: int
) -> list[tuple[int, int]]:
    """Build overlapping chunk spans covering ``[0, total_frames)``."""
    step = max(1, chunk_frames - overlap_frames)
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < total_frames:
        end = min(total_frames, pos + chunk_frames)
        tail = total_frames - end
        if 0 < tail <= overlap_frames and (total_frames - pos) <= max_worker_frames:
            end = total_frames
        spans.append((pos, end))
        if end >= total_frames:
            break
        pos += step
    return spans


def _stream_wav_into_memmap(
    wav_path: Path,
    target_mm: np.memmap[Any, np.dtype[np.float32]],
    target_sr: int,
    target_frames: int,
) -> None:
    """Load ``wav_path`` into ``target_mm``, streaming when rates/lengths match."""
    channels = int(target_mm.shape[0])
    with sf.SoundFile(str(wav_path)) as src:
        if src.channels != channels:
            raise ValueError("The separated output did not preserve a valid recording timeline")
        if src.samplerate == target_sr and src.frames == target_frames:
            pos = 0
            while pos < target_frames:
                take = min(target_frames - pos, _STREAM_BLOCK_FRAMES)
                block = src.read(take, dtype="float32", always_2d=True)
                if block.shape[0] != take or not np.isfinite(block).all():
                    raise ValueError(
                        "The separated output did not preserve a valid recording timeline"
                    )
                target_mm[:, pos : pos + take] = block.T
                evict_memmap_pages(target_mm, pos, pos + take)
                pos += take
            target_mm.flush()
            return
        data = src.read(dtype="float32", always_2d=True)
        src_sr = int(src.samplerate)
    resampled = resample_audio(data.T, src_sr, target_sr, target_frames)
    if resampled.shape != (channels, target_frames) or not np.isfinite(resampled).all():
        raise ValueError("The separated output did not preserve a valid recording timeline")
    target_mm[:, :] = resampled
    target_mm.flush()


def _separate_region_to_memmap(
    decoded_wav: Path,
    *,
    region_start: int,
    region_frames: int,
    total_frames: int,
    sample_rate: int,
    channels: int,
    workspace: Path,
    options: SeparationStrategyV1,
    on_progress: Callable[[float, str], None],
) -> tuple[np.memmap[Any, np.dtype[np.float32]], dict[str, Any], str, Path | None]:
    """Run single-pass (<= 90 s) or raised-cosine chunked separation into a memmap."""
    stem_mm = _create_memmap(workspace / "stem.f32", channels, region_frames)
    try:
        threshold_frames = max(1, int(round(CHUNK_THRESHOLD_S * sample_rate)))
        max_worker_frames = max(1, int(round(WORKER_MAX_DURATION_S * sample_rate)))

        if region_frames <= min(threshold_frames, max_worker_frames):
            is_full_classic = (
                region_start == 0
                and region_frames == total_frames
                and sf.info(str(decoded_wav)).format != "RF64"
            )
            if is_full_classic:
                source_wav = decoded_wav
            else:
                source_wav = workspace / "region_in.wav"
                _extract_wav_slice(decoded_wav, source_wav, region_start, region_frames)
            stem_wav = workspace / "stem.wav"
            metadata = separate_audio(
                source_wav,
                stem_wav,
                prompt=options.prompt,
                operation=options.operation,
                model_size=options.model_size,
                quality=options.quality,
                seed=options.seed,
                device=options.device,
                progress=on_progress,
            )
            selected_sha = hash_file(stem_wav)
            _stream_wav_into_memmap(stem_wav, stem_mm, sample_rate, region_frames)
            return stem_mm, metadata, selected_sha, stem_wav

        chunk_frames = min(max_worker_frames, max(4, int(round(CHUNK_DURATION_S * sample_rate))))
        overlap_frames = min(
            int(round(CHUNK_OVERLAP_S * sample_rate)),
            max(1, chunk_frames // 4),
        )
        spans = _chunk_spans(region_frames, chunk_frames, overlap_frames, max_worker_frames)
        total_chunks = len(spans)
        last_metadata: dict[str, Any] = {}

        for idx, (c_start, c_end) in enumerate(spans):
            chunk_len = c_end - c_start
            chunk_in = workspace / f"chunk_in_{idx}.wav"
            chunk_out = workspace / f"chunk_out_{idx}.wav"
            try:
                _extract_wav_slice(decoded_wav, chunk_in, region_start + c_start, chunk_len)

                def _chunk_progress(
                    value: float, message: str, *, _idx: int = idx, _total: int = total_chunks
                ) -> None:
                    on_progress((_idx + value) / _total, message)

                last_metadata = separate_audio(
                    chunk_in,
                    chunk_out,
                    prompt=options.prompt,
                    operation=options.operation,
                    model_size=options.model_size,
                    quality=options.quality,
                    seed=options.seed,
                    device=options.device,
                    progress=_chunk_progress,
                )
                chunk_raw, chunk_sr = sf.read(chunk_out, dtype="float32", always_2d=True)
            finally:
                chunk_in.unlink(missing_ok=True)
                chunk_out.unlink(missing_ok=True)

            chunk_arr = resample_audio(chunk_raw.T, int(chunk_sr), sample_rate, chunk_len)
            if chunk_arr.shape != (channels, chunk_len) or not np.isfinite(chunk_arr).all():
                raise ValueError("The separated output did not preserve a valid recording timeline")

            left_ov = spans[idx - 1][1] - c_start if idx > 0 else 0
            right_ov = c_end - spans[idx + 1][0] if idx + 1 < total_chunks else 0
            if left_ov > 0:
                t_in = (np.arange(left_ov, dtype=np.float64) + 0.5) / left_ov
                fade_in = (0.5 * (1.0 - np.cos(np.pi * t_in))).astype(np.float32)
                stem_mm[:, c_start : c_start + left_ov] += chunk_arr[:, :left_ov] * fade_in
            interior_end = chunk_len - right_ov
            if interior_end > left_ov:
                stem_mm[:, c_start + left_ov : c_start + interior_end] = chunk_arr[
                    :, left_ov:interior_end
                ]
            if right_ov > 0:
                t_out = (np.arange(right_ov, dtype=np.float64) + 0.5) / right_ov
                fade_out = (0.5 * (1.0 + np.cos(np.pi * t_out))).astype(np.float32)
                stem_mm[:, c_end - right_ov : c_end] = (
                    chunk_arr[:, chunk_len - right_ov :] * fade_out
                )
            evict_memmap_pages(stem_mm, c_start, c_end)

        stem_mm.flush()
        selected_sha = _hash_memmap(stem_mm)
        last_metadata = {
            **last_metadata,
            "samples": region_frames,
            "chunked": True,
            "chunk_count": total_chunks,
            "chunk_duration_s": CHUNK_DURATION_S,
            "chunk_overlap_s": CHUNK_OVERLAP_S,
        }
        return stem_mm, last_metadata, selected_sha, None
    except Exception:
        _close_memmap(stem_mm)
        raise


def _estimate_blend_delay(
    decoded_wav: Path,
    wet_mm: np.memmap[Any, np.dtype[np.float32]],
    region_start: int,
    region_frames: int,
    sample_rate: int,
) -> tuple[int, float]:
    """Estimate constant wet/dry delay on a bounded active mono slice (<= 10 s)."""
    win_len = min(region_frames, max(512, int(round(_DELAY_ANALYSIS_MAX_S * sample_rate))))
    if region_frames < 512:
        return 0, 0.0

    best_offset = 0
    if region_frames > win_len:
        best_energy = -1.0
        step = max(sample_rate, win_len // 2)
        for offset in range(0, region_frames - win_len + 1, step):
            probe_end = min(region_frames, offset + sample_rate)
            energy = float(np.mean(np.square(wet_mm[:, offset:probe_end], dtype=np.float64)))
            if energy > best_energy:
                best_energy = energy
                best_offset = offset

    with sf.SoundFile(str(decoded_wav)) as src:
        src.seek(region_start + best_offset)
        dry_raw = src.read(win_len, dtype="float32", always_2d=True)
    dry_mono = np.mean(dry_raw, axis=1, dtype=np.float32)
    wet_mono = np.mean(wet_mm[:, best_offset : best_offset + win_len], axis=0, dtype=np.float32)
    res = estimate_gcc_phat_delay(dry_mono, wet_mono, sample_rate, max_delay_ms=50.0)
    if res.passed and res.correlation_peak > 0.05:
        return int(round(res.delay_samples)), float(res.delay_ms)
    return 0, 0.0


def _apply_wet_dry_blend(
    decoded_wav: Path,
    wet_mm: np.memmap[Any, np.dtype[np.float32]],
    out_mm: np.memmap[Any, np.dtype[np.float32]],
    *,
    region_start: int,
    region_frames: int,
    shift_samples: int,
    strength: float,
) -> None:
    """Stream delay-aligned ``strength * wet + (1 - strength) * dry`` into ``out_mm``."""
    channels = int(wet_mm.shape[0])
    wet_gain = np.float32(strength)
    dry_gain = np.float32(1.0 - strength)
    with sf.SoundFile(str(decoded_wav)) as src:
        src.seek(region_start)
        for pos in range(0, region_frames, _STREAM_BLOCK_FRAMES):
            end = min(region_frames, pos + _STREAM_BLOCK_FRAMES)
            n = end - pos
            dry = np.ascontiguousarray(src.read(n, dtype="float32", always_2d=True).T)
            wet_block = np.zeros((channels, n), dtype=np.float32)
            # Positive shift means wet lags dry -> read wet[pos + shift : end + shift].
            src_lo = pos + shift_samples
            src_hi = end + shift_samples
            valid_lo = max(0, src_lo)
            valid_hi = min(region_frames, src_hi)
            if valid_hi > valid_lo:
                dst_lo = valid_lo - src_lo
                dst_hi = dst_lo + (valid_hi - valid_lo)
                wet_block[:, dst_lo:dst_hi] = wet_mm[:, valid_lo:valid_hi]
            out_mm[:, pos:end] = wet_block * wet_gain + dry * dry_gain
            evict_memmap_pages(wet_mm, max(0, valid_lo), max(0, valid_hi))
            evict_memmap_pages(out_mm, pos, end)
    out_mm.flush()


def _surgical_output_bit_depth(source: Path) -> Literal["pcm24", "float32"]:
    """Preserve float32 bit-identity on float WAV sources and pcm24 on integer/lossy sources."""
    with contextlib.suppress(OSError, RuntimeError):
        info = sf.info(str(source))
        if info.format in {"WAV", "WAVEX", "RF64"} and info.subtype in {"FLOAT", "DOUBLE"}:
            return "float32"
    return "pcm24"


def _copy_wav_range_to_memmap(
    decoded_wav: Path,
    target_mm: np.memmap[Any, np.dtype[np.float32]],
    start_frame: int,
    end_frame: int,
) -> None:
    """Stream-copy ``[start_frame, end_frame)`` from ``decoded_wav`` into ``target_mm``."""
    if end_frame <= start_frame:
        return
    with sf.SoundFile(str(decoded_wav)) as src:
        src.seek(start_frame)
        pos = start_frame
        while pos < end_frame:
            take = min(end_frame - pos, _STREAM_BLOCK_FRAMES)
            block = src.read(take, dtype="float32", always_2d=True)
            target_mm[:, pos : pos + take] = block.T
            evict_memmap_pages(target_mm, pos, pos + take)
            pos += take


def _splice_time_range(
    decoded_wav: Path,
    processed_mm: np.memmap[Any, np.dtype[np.float32]],
    *,
    win_start: int,
    start_idx: int,
    end_idx: int,
    total_frames: int,
    sample_rate: int,
    channels: int,
    workspace: Path,
) -> np.memmap[Any, np.dtype[np.float32]]:
    """Limit active edit region and equal-power crossfade back into original timeline."""
    edit_len = end_idx - start_idx
    rel_start = start_idx - win_start
    active_mm = _create_memmap(workspace / "active_in.f32", channels, edit_len)
    limited_active: np.memmap[Any, np.dtype[np.float32]] | None = None
    spliced_mm = _create_memmap(workspace / "spliced.f32", channels, total_frames)
    try:
        for pos in range(0, edit_len, _STREAM_BLOCK_FRAMES):
            stop = min(edit_len, pos + _STREAM_BLOCK_FRAMES)
            active_mm[:, pos:stop] = processed_mm[:, rel_start + pos : rel_start + stop]
            evict_memmap_pages(active_mm, pos, stop)
        active_mm.flush()

        limiter_res = apply_lookahead_limiter_to_memmap(
            active_mm, sample_rate, workspace / "active_limited.f32"
        )
        limited_active = cast(np.memmap[Any, np.dtype[np.float32]], limiter_res.limited_waveform)

        # Samples outside [start_idx, end_idx) are copied verbatim from original.
        _copy_wav_range_to_memmap(decoded_wav, spliced_mm, 0, start_idx)
        for pos in range(0, edit_len, _STREAM_BLOCK_FRAMES):
            stop = min(edit_len, pos + _STREAM_BLOCK_FRAMES)
            spliced_mm[:, start_idx + pos : start_idx + stop] = limited_active[:, pos:stop]
            evict_memmap_pages(limited_active, pos, stop)
            evict_memmap_pages(spliced_mm, start_idx + pos, start_idx + stop)
        _copy_wav_range_to_memmap(decoded_wav, spliced_mm, end_idx, total_frames)

        # 30 ms equal-power raised-cosine crossfade strictly inside [start_idx, end_idx).
        fade = min(int(round(TIME_RANGE_FADE_S * sample_rate)), edit_len // 2)
        if fade > 0:
            t = (np.arange(fade, dtype=np.float64) + 0.5) / fade
            w_in = np.sin(0.5 * np.pi * t).astype(np.float32)
            w_out = np.cos(0.5 * np.pi * t).astype(np.float32)
            with sf.SoundFile(str(decoded_wav)) as src:
                if start_idx > 0:
                    src.seek(start_idx)
                    dry_left = src.read(fade, dtype="float32", always_2d=True).T
                    spliced_mm[:, start_idx : start_idx + fade] = (
                        spliced_mm[:, start_idx : start_idx + fade] * w_in + dry_left * w_out
                    )
                if end_idx < total_frames:
                    src.seek(end_idx - fade)
                    dry_right = src.read(fade, dtype="float32", always_2d=True).T
                    spliced_mm[:, end_idx - fade : end_idx] = (
                        spliced_mm[:, end_idx - fade : end_idx] * w_out + dry_right * w_in
                    )
        spliced_mm.flush()
        return spliced_mm
    except Exception:
        _close_memmap(spliced_mm)
        raise
    finally:
        _close_memmap(active_mm)
        _close_memmap(limited_active)


def run_audio_edit(request: EditRequest) -> HawaVoCleanReport:
    """Publish one selected stem, with original-source lineage and explicit limits."""
    source = Path(request.input_path).resolve()
    destination = public_output_path(request.output_path)
    if is_reparse_or_symlink(destination):
        raise ValueError("Choose a regular output file, not a linked destination")
    if source == destination or Path(request.original_input_path).resolve() == destination:
        raise ValueError("The source recording cannot be replaced")

    time_range, strength = _resolve_edit_controls(request)
    _event("preflight", 0.02, "Checking sound separation")
    probe = probe_audio(
        source,
        max_sample_rate=192000,
        max_duration_s=MAX_DURATION_S,
        max_channels=2,
        max_file_size_bytes=MAX_FILE_SIZE_BYTES,
    )
    if (
        probe.duration_s > MAX_DURATION_S
        or probe.channels not in (1, 2)
        or probe.sample_rate > 192000
    ):
        raise ValueError(
            f"Sound separation currently supports mono/stereo clips up to {MAX_DURATION_S:g} seconds"
        )

    options = request.options
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hawa-separate-", dir=destination.parent) as directory:
        workspace = Path(directory)
        decoded = workspace / "decoded.wav"
        sample_rate, channels, total_frames = _decode_to_wav(probe, decoded)
        total_duration_s = total_frames / sample_rate

        start_idx = 0
        end_idx = total_frames
        win_start = 0
        win_end = total_frames
        if time_range is not None:
            start_s, end_s = time_range
            start_idx = int(round(start_s * sample_rate))
            if start_idx >= total_frames:
                raise ValueError(
                    f"time_range starts at {start_s:.3f}s, at or past end of recording "
                    f"({total_duration_s:.3f}s)"
                )
            end_idx = min(total_frames, int(round(end_s * sample_rate)))
            if end_idx <= start_idx:
                raise ValueError("time_range must span at least one audio sample")
            pad_samples = int(round(TIME_RANGE_PAD_S * sample_rate))
            win_start = max(0, start_idx - pad_samples)
            win_end = min(total_frames, end_idx + pad_samples)

        region_frames = win_end - win_start
        _event("enhance", 0.08, "Separating the requested sound")
        stem_mm: np.memmap[Any, np.dtype[np.float32]] | None = None
        cleaned_mm: np.memmap[Any, np.dtype[np.float32]] | None = None
        blended_mm: np.memmap[Any, np.dtype[np.float32]] | None = None
        final_mm: np.memmap[Any, np.dtype[np.float32]] | None = None
        try:
            stem_mm, metadata, selected_sha, stem_wav = _separate_region_to_memmap(
                decoded,
                region_start=win_start,
                region_frames=region_frames,
                total_frames=total_frames,
                sample_rate=sample_rate,
                channels=channels,
                workspace=workspace,
                options=options,
                on_progress=lambda value, message: _event("enhance", 0.08 + 0.62 * value, message),
            )
            metadata.update(
                {
                    "selected_stem_sha256": selected_sha,
                    "cleanup_profile": options.cleanup_profile,
                    "guard_scope": "signal-integrity-only; sound-selection quality requires listening",
                    "source_sha256": probe.sha256,
                    "strength": strength,
                }
            )
            if time_range is not None:
                metadata.update(
                    {
                        "time_range": [time_range[0], time_range[1]],
                        "time_range_samples": [start_idx, end_idx],
                        "time_range_pad_s": TIME_RANGE_PAD_S,
                        "time_range_fade_ms": TIME_RANGE_FADE_S * 1000.0,
                    }
                )

            processed_mm = stem_mm
            cleanup_report: HawaVoCleanReport | None = None
            if options.cleanup_profile:
                if stem_wav is None:
                    stem_wav = workspace / "stem_for_cleanup.wav"
                    encode_audio_streaming(
                        AudioBuffer(stem_mm, sample_rate),
                        stem_wav,
                        output_bit_depth="float32",
                        dither=False,
                    )
                    metadata["selected_stem_sha256"] = hash_file(stem_wav)

                def cleanup_progress(event: ProgressEvent) -> None:
                    _event(event.stage, 0.70 + 0.20 * event.progress, event.message)

                cleaned = workspace / "dialogue.wav"
                cleanup_report = run_pipeline(
                    stem_wav, cleaned, profile=options.cleanup_profile, on_progress=cleanup_progress
                )
                cleaned_mm = _create_memmap(workspace / "cleaned.f32", channels, region_frames)
                _stream_wav_into_memmap(cleaned, cleaned_mm, sample_rate, region_frames)
                processed_mm = cleaned_mm

            if strength < 1.0:
                shift_samples, delay_ms = _estimate_blend_delay(
                    decoded, processed_mm, win_start, region_frames, sample_rate
                )
                blended_mm = _create_memmap(workspace / "blended.f32", channels, region_frames)
                _apply_wet_dry_blend(
                    decoded,
                    processed_mm,
                    blended_mm,
                    region_start=win_start,
                    region_frames=region_frames,
                    shift_samples=shift_samples,
                    strength=strength,
                )
                metadata["blend_delay_ms"] = round(delay_ms, 3)
                metadata["blend_shift_samples"] = shift_samples
                processed_mm = blended_mm

            config_payload = options.model_dump(mode="json")
            if time_range is not None:
                config_payload["time_range"] = [time_range[0], time_range[1]]
            if strength != 1.0:
                config_payload["strength"] = strength
            config_hash = hash_json_canonical(config_payload)

            temporary_output = workspace / "master.wav"
            if time_range is not None:
                final_mm = _splice_time_range(
                    decoded,
                    processed_mm,
                    win_start=win_start,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    total_frames=total_frames,
                    sample_rate=sample_rate,
                    channels=channels,
                    workspace=workspace,
                )
                encode_audio_streaming(
                    AudioBuffer(final_mm, sample_rate),
                    temporary_output,
                    output_bit_depth=_surgical_output_bit_depth(source),
                    dither=False,
                    seed_context=config_hash,
                )
            else:
                limiter_res = apply_lookahead_limiter_to_memmap(
                    processed_mm, sample_rate, workspace / "limited.f32"
                )
                final_mm = cast(np.memmap[Any, np.dtype[np.float32]], limiter_res.limited_waveform)
                encode_audio_streaming(
                    AudioBuffer(final_mm, sample_rate),
                    temporary_output,
                    output_bit_depth="pcm24",
                    dither=True,
                    seed_context=config_hash,
                )
        finally:
            _close_memmap(final_mm)
            _close_memmap(blended_mm)
            _close_memmap(cleaned_mm)
            _close_memmap(stem_mm)

        output_stats = _stats_from_wav(
            str(destination), hash_file(temporary_output), temporary_output
        )
        input_stats = _stats_from_wav(request.original_input_path, probe.sha256, decoded)

        if cleanup_report is not None:
            # Existing guard decisions apply only to stem->cleanup, not to the
            # original mixture->stem separation. Preserve that stage boundary.
            metadata["cleanup_input_sha256"] = cleanup_report.input.sha256
            metadata["cleanup_output_sha256"] = cleanup_report.output.sha256
            metadata["cleanup_config_hash"] = cleanup_report.config_hash
            payload = cleanup_report.model_dump()
            payload["environment"]["deterministic_settings"]["dither_seed_derivation"] = (
                "sha256(separation_config_hash:channel_index)"
            )
            payload.update(
                input=input_stats.model_dump(),
                output=output_stats.model_dump(),
                config_hash=config_hash,
                separation=metadata,
            )
            report = HawaVoCleanReport.model_validate(payload)
        else:
            contract = {
                "shape": "source-rate/channels/samples",
                "finite": True,
                "peak_ceiling_dbtp": -1.0,
                "semantic_validation": False,
            }
            identity = hash_json_canonical(contract)
            config = HawaVoCleanConfig()
            settings = deterministic_settings(config)
            engine_id = str(metadata.get("engine", "audioghost-sam-audio-lite"))
            provider = str(metadata.get("provider", "cpu"))
            settings.update(
                worker_pool_size=1,
                result_order="source-channel-order",
                dither_seed_derivation="sha256(separation_config_hash:channel_index)",
                compute_device=provider,
            )
            engine_manifest = {
                key: metadata[key]
                for key in (
                    "engine",
                    "model",
                    "license",
                    "upstream_commit",
                    "sam_audio_commit",
                    "model_sha256",
                    "model_config_sha256",
                    "text_encoder_sha256",
                )
                if key in metadata
            }
            report = HawaVoCleanReport(
                release=current_release_metadata(),
                build=current_build_metadata(),
                job_id=f"separate_{uuid.uuid4().hex}",
                config_hash=config_hash,
                input=input_stats,
                output=output_stats,
                core=CoreMetadata(
                    id=engine_id,
                    algorithm="prompt-guided separation",
                    params_hash=config_hash,
                    phase_coherent=False,
                    lock_sha256=hash_json_canonical(engine_manifest),
                    weight_sha256={
                        key: str(metadata[key])
                        for key in ("model_sha256", "model_config_sha256", "text_encoder_sha256")
                        if key in metadata
                    },
                ),
                guard=GuardMetadata(
                    id="separation-signal-only-unqualified",
                    probe_hash=identity,
                    calibration_id="uncalibrated-signal-contract",
                    calibration_sha256=identity,
                ),
                environment=EnvironmentMetadata(
                    platform=platform.platform(),
                    os_version=platform.version(),
                    python_version=platform.python_version(),
                    numpy_version=np.__version__,
                    scipy_version=scipy.__version__,
                    soundfile_version=sf.__version__,
                    runtime_versions=runtime_versions(),
                    deterministic_settings=settings,
                    compute_device=provider,
                ),
                summary=UnitSummary(),
                separation=metadata,
            )

        range_line = (
            f"Time range: {time_range[0]:.3f}s–{time_range[1]:.3f}s\n"
            if time_range is not None
            else ""
        )
        summary = (
            "HAWAVOCLEAN SOUND SEPARATION — EXPERIMENTAL\n"
            f"Engine: {metadata.get('engine', 'audioghost-sam-audio-lite')}\n"
            f"Operation: {options.operation}\nSound: {options.prompt}\n"
            f"Model: {options.model_size}; quality: {options.quality}; "
            f"seed: {options.seed}; strength: {strength:.2f}\n"
            f"{range_line}"
            f"Source SHA256: {report.input.sha256}\nOutput SHA256: {report.output.sha256}\n"
            f"Duration: {report.output.duration_s:.3f} seconds; channels: {report.output.channels}\n"
            f"Dialogue cleanup: {options.cleanup_profile or 'not applied'}\n"
            "Any speech-unit guard decisions cover the selected stem's cleanup only.\n"
            "Signal checks do not establish word retention or sound-selection quality. Listen before export.\n"
        )
        _event("publish", 0.98, "Saving the separated recording")
        paths = publish_output_generation(
            temporary_output,
            destination,
            serialize_json_report(report),
            summary,
            overwrite=request.overwrite,
        )
        if request.bundle_path:
            create_processing_record(
                master_path=paths[0],
                report_path=paths[1],
                summary_path=paths[2],
                destination=request.bundle_path,
                overwrite=request.overwrite,
            )
        return report


def main() -> None:
    from hawavoclean.cli import _install_signal_handlers
    from hawavoclean.watchdog import install_parent_death_watchdog

    _install_signal_handlers()
    install_parent_death_watchdog()
    try:
        raw = sys.stdin.readline(16385)
        if len(raw) > 16384:
            raise ValueError("Sound-separation request is too large")
        request = EditRequest.model_validate_json(raw)
        report = run_audio_edit(request)
        print(
            json.dumps(
                {
                    "event": "done",
                    "report_path": str(Path(request.output_path).with_suffix(".hawavoclean.json")),
                    "output_path": report.output.path,
                }
            ),
            flush=True,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(
            json.dumps({"event": "error", "code": "SEPARATION_FAILED", "message": str(exc)}),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
