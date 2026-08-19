"""Audio file encoding and deterministic TPDF dithering for 24-bit PCM and 32-bit float WAV."""

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from hawavoclean.audio.types import AudioBuffer
from hawavoclean.errors import OutputValidationError


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
    for ch_idx in range(channels):
        ch = data[ch_idx]
        if output_bit_depth == "pcm24" and dither:
            ch_seed = f"{seed_context}:ch_{ch_idx}"
            ch = apply_tpdf_dither(ch, ch_seed, target_bits=24)
        out_channels.append(ch)

    interleaved = np.column_stack(out_channels) if channels > 1 else out_channels[0].reshape(-1, 1)

    subtype = "PCM_24" if output_bit_depth == "pcm24" else "FLOAT"

    try:
        sf.write(
            str(dest_path),
            interleaved,
            samplerate=buffer.sample_rate,
            subtype=subtype,
            format="WAV",
        )
    except Exception as e:
        raise OutputValidationError(f"Failed to write WAV output to {dest_path}: {e}") from e

    return dest_path
