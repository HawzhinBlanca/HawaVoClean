"""Audio file encoding and deterministic TPDF dithering for 24-bit PCM and 32-bit float WAV."""

import hashlib
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from hawavoclean.audio.types import AudioBuffer
from hawavoclean.errors import OutputValidationError
from hawavoclean.runtime import evict_memmap_pages

WAV_CLASSIC_LIMIT_BYTES = (1 << 32) - 1
WAV_HEADER_RESERVE_BYTES = 1 << 20


def _wav_container_format(
    channels: int,
    samples: int,
    subtype: Literal["PCM_24", "FLOAT"],
) -> Literal["WAV", "RF64"]:
    """Choose classic RIFF where it fits and RF64 before 32-bit overflow."""
    bytes_per_sample = 3 if subtype == "PCM_24" else 4
    estimated = channels * samples * bytes_per_sample + WAV_HEADER_RESERVE_BYTES
    return "RF64" if estimated > WAV_CLASSIC_LIMIT_BYTES else "WAV"


def _finalize_deterministic_wav(path: Path) -> None:
    """Zero libsndfile's wall-clock PEAK timestamp and durably close the WAV.

    libsndfile writes a PEAK chunk for float WAV/RF64 output. Its timestamp is
    the current wall clock, so two sample-identical renders otherwise receive
    different artifact hashes. A zero timestamp is the PEAK specification's
    valid unknown-time value and leaves every audio/peak field untouched.
    """
    try:
        with open(path, "r+b") as handle:
            header = handle.read(12)
            if len(header) != 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
                raise OutputValidationError(f"Encoded output is not a valid WAV/RF64 file: {path}")
            while True:
                chunk_header = handle.read(8)
                if not chunk_header:
                    break
                if len(chunk_header) != 8:
                    raise OutputValidationError(f"Encoded WAV has a truncated chunk header: {path}")
                chunk_id = chunk_header[:4]
                chunk_size = int.from_bytes(chunk_header[4:8], "little")
                payload_start = handle.tell()
                if chunk_id == b"PEAK":
                    if chunk_size < 8:
                        raise OutputValidationError(
                            f"Encoded WAV has a malformed PEAK chunk: {path}"
                        )
                    handle.seek(payload_start + 4)
                    handle.write(b"\0\0\0\0")
                    break
                if chunk_id == b"data":
                    break
                handle.seek(payload_start + chunk_size + (chunk_size & 1))
            handle.flush()
            os.fsync(handle.fileno())
    except OutputValidationError:
        raise
    except OSError as exc:
        raise OutputValidationError(f"Failed to finalize encoded WAV {path}: {exc}") from exc


def apply_tpdf_dither(
    channel_data: np.ndarray[Any, np.dtype[np.float32]],  # 1D float32 array in [-1.0, 1.0]
    seed_str: str,
    target_bits: int = 24,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply deterministic Triangular Probability Density Function (TPDF) dither."""
    # Derive integer seed from string hash
    seed_int = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)

    lsb = 1.0 / (2 ** (target_bits - 1))
    dither = rng.triangular(-lsb, 0.0, lsb, size=channel_data.shape).astype(np.float32)
    dithered = channel_data + dither
    scaled = np.round(dithered * (2 ** (target_bits - 1))) / (2 ** (target_bits - 1))
    return np.ascontiguousarray(np.clip(scaled, -1.0, 1.0 - lsb), dtype=np.float32)


def encode_audio(
    buffer: AudioBuffer,
    output_path: Path | str,
    output_bit_depth: Literal["pcm24", "float32"] = "pcm24",
    dither: bool = True,
    seed_context: str = "hawavoclean",
) -> Path:
    """Encode an AudioBuffer to WAV file on disk with strict validation."""
    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    data = buffer.data  # shape (channels, samples)
    if not np.all(np.isfinite(data)):
        raise OutputValidationError("Cannot encode audio: waveform contains NaN or Inf values.")

    channels, samples = data.shape
    if samples == 0 or channels == 0:
        raise OutputValidationError("Cannot encode empty audio buffer (zero samples or channels).")

    # Transpose to (samples, channels) for soundfile
    out_channels: list[np.ndarray] = []
    # Channels that carry the SAME signal (dual-mono) must get the same
    # dither so the output stays bit-identical L/R; distinct channels get
    # decorrelated dither as before.
    seed_for_channel: list[str] = []
    for ch_idx in range(channels):
        same_as = next(
            (j for j in range(ch_idx) if np.array_equal(buffer.data[j], buffer.data[ch_idx])),
            None,
        )
        seed_for_channel.append(
            seed_for_channel[same_as] if same_as is not None else f"{seed_context}:ch_{ch_idx}"
        )
    for ch_idx in range(channels):
        ch = data[ch_idx]
        if output_bit_depth == "pcm24" and dither:
            ch_seed = seed_for_channel[ch_idx]
            ch = apply_tpdf_dither(ch, ch_seed, target_bits=24)
        out_channels.append(ch)

    interleaved = np.column_stack(out_channels) if channels > 1 else out_channels[0].reshape(-1, 1)

    subtype: Literal["PCM_24", "FLOAT"] = "PCM_24" if output_bit_depth == "pcm24" else "FLOAT"
    container = _wav_container_format(channels, samples, subtype)

    try:
        sf.write(
            str(dest_path),
            interleaved,
            samplerate=buffer.sample_rate,
            subtype=subtype,
            format=container,
        )
        _finalize_deterministic_wav(dest_path)
    except OutputValidationError:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise OutputValidationError(f"Failed to write WAV output to {dest_path}: {e}") from e

    return dest_path


def _stream_seed_contexts(
    data: np.ndarray[Any, np.dtype[np.float32]],
    seed_context: str,
    chunk_samples: int,
) -> list[str]:
    """Find identical channels without allocating a file-length difference."""
    channels, samples = data.shape
    seeds: list[str] = []
    for channel in range(channels):
        same_as: int | None = None
        for prior in range(channel):
            equal = True
            for start in range(0, samples, chunk_samples):
                stop = min(samples, start + chunk_samples)
                if not np.array_equal(data[channel, start:stop], data[prior, start:stop]):
                    equal = False
                    break
            if equal:
                same_as = prior
                break
        seeds.append(seeds[same_as] if same_as is not None else f"{seed_context}:ch_{channel}")
    return seeds


def encode_audio_streaming(
    buffer: AudioBuffer,
    output_path: Path | str,
    output_bit_depth: Literal["pcm24", "float32"] = "pcm24",
    dither: bool = True,
    seed_context: str = "hawavoclean",
    *,
    chunk_samples: int = 1 << 20,
) -> Path:
    """Encode a disk-backed buffer with bounded interleave/dither storage.

    One RNG per channel preserves the canonical channel seed and NumPy's
    deterministic stream across chunk boundaries. Identical channels receive
    identical seeds just as :func:`encode_audio` does, so dual mono remains
    bit-identical. The encoded samples and WAV structure are byte-identical to
    the allocating encoder for the same buffer and seed.
    """
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")
    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    data = buffer.data
    channels, samples = data.shape
    if samples == 0 or channels == 0:
        raise OutputValidationError("Cannot encode empty audio buffer (zero samples or channels).")

    subtype: Literal["PCM_24", "FLOAT"] = "PCM_24" if output_bit_depth == "pcm24" else "FLOAT"
    container = _wav_container_format(channels, samples, subtype)
    seed_for_channel = _stream_seed_contexts(data, seed_context, chunk_samples)
    rngs = [
        np.random.default_rng(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16))
        for seed in seed_for_channel
    ]
    lsb = np.float32(1.0 / (2 ** (24 - 1)))
    scale = np.float32(2 ** (24 - 1))

    try:
        with sf.SoundFile(
            str(dest_path),
            mode="w",
            samplerate=buffer.sample_rate,
            channels=channels,
            subtype=subtype,
            format=container,
        ) as destination:
            for start in range(0, samples, chunk_samples):
                stop = min(samples, start + chunk_samples)
                n = stop - start
                interleaved = np.empty((n, channels), dtype=np.float32)
                for channel in range(channels):
                    source = data[channel, start:stop]
                    if not np.all(np.isfinite(source)):
                        raise OutputValidationError(
                            "Cannot encode audio: waveform contains NaN or Inf values."
                        )
                    if output_bit_depth == "pcm24" and dither:
                        noise = (
                            rngs[channel]
                            .triangular(-float(lsb), 0.0, float(lsb), size=n)
                            .astype(np.float32)
                        )
                        values = np.add(source, noise, dtype=np.float32)
                        values = np.round(values * scale) / scale
                        interleaved[:, channel] = np.clip(values, -1.0, 1.0 - lsb)
                    else:
                        interleaved[:, channel] = source
                destination.write(interleaved)
                evict_memmap_pages(data, start, stop)
        _finalize_deterministic_wav(dest_path)

    except OutputValidationError:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise OutputValidationError(f"Failed to write WAV output to {dest_path}: {exc}") from exc

    return dest_path
