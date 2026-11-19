"""True stereo Mid/Side enhancement and multi-track Poly-WAV/BWF track routing.

Provides exact float32 Mid/Side encoding/decoding so true stereo dialogue
recordings clean center speech without introducing inter-channel phase
decorrelation artifacts, alongside streaming inspection and track extraction
for 1-16 channel WAV/BWF polyphonic field-recorder files.
"""

from __future__ import annotations

import math
import os
import re
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.signal
import soundfile as sf

from hawavoclean.audio.encode import _finalize_deterministic_wav, _wav_container_format

MAX_POLY_WAV_CHANNELS = 16
POLY_STREAM_BLOCK_FRAMES = 48000
FLOOR_DBFS = -120.0
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def encode_mid_side(
    left: np.ndarray[Any, Any],
    right: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """Encode stereo ``(left, right)`` 1D arrays into ``(mid, side)`` float32 arrays.

    Uses ``mid = 0.5 * (L + R)`` and ``side = 0.5 * (L - R)`` in float64 before
    casting to contiguous float32 so ``decode_mid_side(*encode_mid_side(L, R))``
    round-trips within ``1e-6``.
    """
    l_arr = np.asarray(left, dtype=np.float64)
    r_arr = np.asarray(right, dtype=np.float64)
    if l_arr.ndim != 1 or r_arr.ndim != 1:
        raise ValueError(
            f"left and right must be 1D arrays, got shapes {l_arr.shape} and {r_arr.shape}"
        )
    if l_arr.shape[0] != r_arr.shape[0]:
        raise ValueError(f"left and right length mismatch: {l_arr.shape[0]} != {r_arr.shape[0]}")
    if not (np.all(np.isfinite(l_arr)) and np.all(np.isfinite(r_arr))):
        raise ValueError("left and right channels must contain only finite values")

    mid = np.ascontiguousarray(0.5 * (l_arr + r_arr), dtype=np.float32)
    side = np.ascontiguousarray(0.5 * (l_arr - r_arr), dtype=np.float32)
    return mid, side


def decode_mid_side(
    mid: np.ndarray[Any, Any],
    side: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """Decode ``(mid, side)`` 1D arrays back into ``(left, right)`` float32 arrays."""
    m_arr = np.asarray(mid, dtype=np.float64)
    s_arr = np.asarray(side, dtype=np.float64)
    if m_arr.ndim != 1 or s_arr.ndim != 1:
        raise ValueError(
            f"mid and side must be 1D arrays, got shapes {m_arr.shape} and {s_arr.shape}"
        )
    if m_arr.shape[0] != s_arr.shape[0]:
        raise ValueError(f"mid and side length mismatch: {m_arr.shape[0]} != {s_arr.shape[0]}")
    if not (np.all(np.isfinite(m_arr)) and np.all(np.isfinite(s_arr))):
        raise ValueError("mid and side channels must contain only finite values")

    left = np.ascontiguousarray(m_arr + s_arr, dtype=np.float32)
    right = np.ascontiguousarray(m_arr - s_arr, dtype=np.float32)
    return left, right


def enhance_true_stereo_ms(
    stereo_2d: np.ndarray[Any, Any],
    enhance_mono_fn: Callable[
        [np.ndarray[Any, np.dtype[np.float32]]], np.ndarray[Any, np.dtype[np.float32]]
    ],
    *,
    side_attenuation: float = 0.5,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Enhance true stereo ``(2, samples)`` audio via phase-coherent Mid/Side processing.

    Enhances the Mid (center dialogue) channel with ``enhance_mono_fn`` and
    applies a constant, phase-preserving linear attenuation ``1 - side_attenuation``
    to the Side channel so stereo imaging stays coherent without L/R phase
    decorrelation.
    """
    arr = np.asarray(stereo_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"stereo_2d must have shape (2, samples), got {arr.shape}")
    if not math.isfinite(side_attenuation) or not 0.0 <= side_attenuation <= 1.0:
        raise ValueError(
            f"side_attenuation must be finite and in [0.0, 1.0], got {side_attenuation!r}"
        )

    n_samples = int(arr.shape[1])
    if n_samples == 0:
        return np.empty((2, 0), dtype=np.float32)

    mid, side = encode_mid_side(arr[0], arr[1])
    enhanced_mid = np.ascontiguousarray(enhance_mono_fn(mid), dtype=np.float32)
    if enhanced_mid.ndim != 1:
        raise ValueError(f"enhance_mono_fn must return a 1D array, got shape {enhanced_mid.shape}")
    if enhanced_mid.shape[0] < n_samples:
        enhanced_mid = np.ascontiguousarray(
            np.pad(enhanced_mid, (0, n_samples - enhanced_mid.shape[0]), mode="constant"),
            dtype=np.float32,
        )
    elif enhanced_mid.shape[0] > n_samples:
        enhanced_mid = np.ascontiguousarray(enhanced_mid[:n_samples], dtype=np.float32)

    if not np.all(np.isfinite(enhanced_mid)):
        raise ValueError("enhance_mono_fn produced NaN or Inf samples")

    side_scale = 1.0 - float(side_attenuation)
    attenuated_side = np.ascontiguousarray(side.astype(np.float64) * side_scale, dtype=np.float32)
    left_out, right_out = decode_mid_side(enhanced_mid, attenuated_side)
    return np.ascontiguousarray(np.vstack((left_out, right_out)), dtype=np.float32)


@dataclass(frozen=True)
class TrackInfo:
    """Per-track audio statistics and dialogue-activity metadata for a Poly-WAV file."""

    index: int
    name: str
    sample_rate: int
    samples: int
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    speech_activity_ratio: float
    likely_dialogue: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "sample_rate": self.sample_rate,
            "samples": self.samples,
            "duration_s": self.duration_s,
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "speech_activity_ratio": self.speech_activity_ratio,
            "likely_dialogue": self.likely_dialogue,
        }


def _read_ixml_track_names(path: Path) -> dict[int, str]:
    """Best-effort extraction of 0-based track names from a BWF ``iXML`` chunk."""
    names: dict[int, str] = {}
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) != 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
                return names
            while True:
                chunk_hdr = handle.read(8)
                if len(chunk_hdr) < 8:
                    break
                chunk_id = chunk_hdr[:4]
                chunk_size = int.from_bytes(chunk_hdr[4:8], "little")
                if chunk_id == b"iXML":
                    # Bound XML payload read to 1 MiB to avoid hostile headers.
                    if 0 < chunk_size <= 1 << 20:
                        raw_xml = handle.read(chunk_size).rstrip(b"\x00").decode("utf-8", "replace")
                        root = ET.fromstring(raw_xml)
                        for track in root.findall(".//TRACK_LIST/TRACK"):
                            idx_text = track.findtext("CHANNEL_INDEX")
                            name_text = track.findtext("NAME")
                            if idx_text and name_text and name_text.strip():
                                idx = int(idx_text.strip()) - 1
                                if idx >= 0:
                                    names[idx] = name_text.strip()
                    break
                # Skip chunk payload (+ RIFF pad byte).
                if chunk_id == b"data" and chunk_size == 0xFFFFFFFF:
                    break
                handle.seek(chunk_size + (chunk_size & 1), os.SEEK_CUR)
    except Exception:
        return {}
    return names


def _build_speech_band_sos(sample_rate: int) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    nyquist = sample_rate / 2.0
    if nyquist <= 400.0:
        return None
    low = min(300.0, nyquist * 0.25) / nyquist
    high = min(3400.0, nyquist * 0.90) / nyquist
    if not 0.0 < low < high < 1.0:
        return None
    sos = scipy.signal.butter(2, [low, high], btype="bandpass", output="sos")
    return np.asarray(sos, dtype=np.float64)


def _linear_to_dbfs(val: float) -> float:
    if val <= 0.0:
        return FLOOR_DBFS
    return max(FLOOR_DBFS, float(20.0 * math.log10(val)))


def inspect_polywav_tracks(path: Path) -> list[TrackInfo]:
    """Inspect a 1-16 channel WAV/BWF file and return per-track audio & speech heuristics."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Poly-WAV file not found: {file_path}")

    ixml_names = _read_ixml_track_names(file_path)

    with sf.SoundFile(str(file_path)) as reader:
        channels = int(reader.channels)
        sample_rate = int(reader.samplerate)
        if not 1 <= channels <= MAX_POLY_WAV_CHANNELS:
            raise ValueError(
                f"Poly-WAV track count {channels} outside supported range 1..{MAX_POLY_WAV_CHANNELS}"
            )
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate {sample_rate} in {file_path}")

        sos = _build_speech_band_sos(sample_rate)
        zi_list: list[np.ndarray[Any, np.dtype[np.float64]]] | None = (
            [np.zeros((sos.shape[0], 2), dtype=np.float64) for _ in range(channels)]
            if sos is not None
            else None
        )

        frame_len = max(16, int(round(sample_rate * 0.020)))
        sum_sq = np.zeros(channels, dtype=np.float64)
        peaks = np.zeros(channels, dtype=np.float64)
        active_frames = np.zeros(channels, dtype=np.int64)
        total_frames_counted = 0
        total_samples = 0

        # Remainder buffer for exact 20 ms framing across streamed blocks.
        rem_full = np.empty((0, channels), dtype=np.float64)
        rem_band = np.empty((0, channels), dtype=np.float64)

        while True:
            block = reader.read(POLY_STREAM_BLOCK_FRAMES, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            if not np.all(np.isfinite(block)):
                raise ValueError(f"Audio data in {file_path} contains NaN or Inf values")

            block64 = block.astype(np.float64, copy=False)
            total_samples += int(block64.shape[0])
            sum_sq += np.sum(np.square(block64), axis=0)
            peaks = np.maximum(peaks, np.max(np.abs(block64), axis=0))

            if sos is not None and zi_list is not None:
                band64 = np.empty_like(block64)
                for ch in range(channels):
                    filtered, zi_list[ch] = scipy.signal.sosfilt(
                        sos, block64[:, ch], zi=zi_list[ch]
                    )
                    band64[:, ch] = filtered
            else:
                band64 = block64

            if rem_full.shape[0] > 0:
                full_cat = np.vstack((rem_full, block64))
                band_cat = np.vstack((rem_band, band64))
            else:
                full_cat = block64
                band_cat = band64

            n_complete = full_cat.shape[0] // frame_len
            if n_complete > 0:
                used = n_complete * frame_len
                full_frames = full_cat[:used].reshape((n_complete, frame_len, channels))
                band_frames = band_cat[:used].reshape((n_complete, frame_len, channels))
                full_rms = np.sqrt(np.mean(np.square(full_frames), axis=1))
                band_rms = np.sqrt(np.mean(np.square(band_frames), axis=1))
                # A frame counts as speech-active when speech-band RMS > -48 dBFS (~0.004)
                # and speech-band accounts for >= 35% of frame RMS (rejects 50/60 Hz hum & hiss).
                speech_ratio = band_rms / (full_rms + 1e-12)
                is_active = (band_rms > 0.004) & (speech_ratio >= 0.35)
                active_frames += np.sum(is_active, axis=0)
                total_frames_counted += n_complete
                rem_full = full_cat[used:]
                rem_band = band_cat[used:]
            else:
                rem_full = full_cat
                rem_band = band_cat

        # Handle very short files (< 20 ms) with one partial frame.
        if total_frames_counted == 0 and rem_full.shape[0] > 0:
            full_rms_short = np.sqrt(np.mean(np.square(rem_full), axis=0))
            band_rms_short = np.sqrt(np.mean(np.square(rem_band), axis=0))
            speech_ratio_short = band_rms_short / (full_rms_short + 1e-12)
            active_frames += ((band_rms_short > 0.004) & (speech_ratio_short >= 0.35)).astype(
                np.int64
            )
            total_frames_counted = 1

    duration_s = round(total_samples / float(sample_rate), 4) if total_samples > 0 else 0.0
    tracks: list[TrackInfo] = []
    for ch in range(channels):
        rms = math.sqrt(float(sum_sq[ch]) / max(total_samples, 1))
        rms_db = round(_linear_to_dbfs(rms), 2)
        peak_db = round(_linear_to_dbfs(float(peaks[ch])), 2)
        activity = (
            round(float(active_frames[ch]) / float(total_frames_counted), 4)
            if total_frames_counted > 0
            else 0.0
        )
        likely = rms_db > -55.0 and activity >= 0.05
        name = ixml_names.get(ch, f"Track {ch + 1}")
        tracks.append(
            TrackInfo(
                index=ch,
                name=name,
                sample_rate=sample_rate,
                samples=total_samples,
                duration_s=duration_s,
                rms_dbfs=rms_db,
                peak_dbfs=peak_db,
                speech_activity_ratio=activity,
                likely_dialogue=likely,
            )
        )
    return tracks


def _sanitize_track_slug(name: str) -> str:
    slug = _SAFE_NAME_RE.sub("_", name.strip()).strip("_")
    return slug[:40] if slug else "track"


def extract_polywav_tracks(
    path: Path,
    track_indices: list[int],
    out_dir: Path,
) -> list[Path]:
    """Stream-extract selected 0-based tracks from a Poly-WAV into mono WAV files in ``out_dir``."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Poly-WAV file not found: {file_path}")
    if not track_indices:
        raise ValueError("track_indices must not be empty")

    ixml_names = _read_ixml_track_names(file_path)
    target_dir = Path(out_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(file_path)) as reader:
        channels = int(reader.channels)
        sample_rate = int(reader.samplerate)
        total_frames = int(reader.frames)
        if not 1 <= channels <= MAX_POLY_WAV_CHANNELS:
            raise ValueError(
                f"Poly-WAV track count {channels} outside supported range 1..{MAX_POLY_WAV_CHANNELS}"
            )
        for idx in track_indices:
            if isinstance(idx, bool) or not 0 <= idx < channels:
                raise ValueError(
                    f"Track index {idx!r} out of range 0..{channels - 1} for {file_path}"
                )

        subtype: Literal["PCM_24", "FLOAT"] = "FLOAT" if reader.subtype == "FLOAT" else "PCM_24"
        container = _wav_container_format(1, total_frames, subtype)

        unique_indices: list[int] = list(dict.fromkeys(track_indices))
        stem = _sanitize_track_slug(file_path.stem)
        final_by_idx: dict[int, Path] = {}
        tmp_by_idx: dict[int, Path] = {}
        writers: dict[int, sf.SoundFile] = {}

        try:
            for idx in unique_indices:
                raw_name = ixml_names.get(idx)
                if raw_name:
                    filename = f"{stem}_track_{idx + 1:02d}_{_sanitize_track_slug(raw_name)}.wav"
                else:
                    filename = f"{stem}_track_{idx + 1:02d}.wav"
                final_path = target_dir / filename
                tmp_path = target_dir / (
                    f".{filename}.{os.getpid()}.{threading.get_ident()}.tmp.wav"
                )
                final_by_idx[idx] = final_path
                tmp_by_idx[idx] = tmp_path
                writers[idx] = sf.SoundFile(
                    str(tmp_path),
                    mode="w",
                    samplerate=sample_rate,
                    channels=1,
                    subtype=subtype,
                    format=container,
                )

            while True:
                block = reader.read(POLY_STREAM_BLOCK_FRAMES, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                if not np.all(np.isfinite(block)):
                    raise ValueError(f"Audio data in {file_path} contains NaN or Inf values")
                for idx in unique_indices:
                    writers[idx].write(
                        np.ascontiguousarray(block[:, idx : idx + 1], dtype=np.float32)
                    )

            for idx in unique_indices:
                writers[idx].close()
                _finalize_deterministic_wav(tmp_by_idx[idx])
                os.replace(tmp_by_idx[idx], final_by_idx[idx])
        except Exception:
            for writer in writers.values():
                if not writer.closed:
                    writer.close()
            for tmp_path in tmp_by_idx.values():
                tmp_path.unlink(missing_ok=True)
            raise

    return [final_by_idx[idx] for idx in track_indices]
