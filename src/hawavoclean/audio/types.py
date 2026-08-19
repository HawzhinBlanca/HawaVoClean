"""Audio data structures, metadata records, and buffer representations."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.hashing import hash_numpy


class ChannelMode(StrEnum):
    """Channel mode classification as specified in BLUEPRINT.md section 9.3."""

    MONO = "mono"
    DUAL_MONO_SAME = "dual_mono_same"
    SPLIT_SPEAKERS = "split_speakers"
    AMBIGUOUS_STEREO = "ambiguous_stereo"


@dataclass(frozen=True)
class AudioProbeResult:
    """Media probe metadata returned by FFprobe."""

    path: Path
    format_name: str
    codec_name: str
    sample_rate: int
    channels: int
    duration_s: float
    samples: int
    bit_depth: int | None
    sha256: str


@dataclass
class AudioBuffer:
    """Canonical memory buffer representing multichannel audio as (channels, samples) float32."""

    data: np.ndarray[Any, np.dtype[np.float32]]
    sample_rate: int
    channel_mode: ChannelMode = ChannelMode.MONO

    def __post_init__(self) -> None:
        if self.data.ndim == 1:
            self.data = np.expand_dims(self.data, axis=0)
        elif self.data.ndim != 2:
            raise ValueError(f"AudioBuffer data must be 1D or 2D, got shape {self.data.shape}")

        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

    @property
    def channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.samples / self.sample_rate)

    def slice(self, start_sample: int, end_sample: int) -> "AudioBuffer":
        """Extract a sample-accurate slice across all channels."""
        clamped_start = max(0, start_sample)
        clamped_end = min(self.samples, end_sample)
        return AudioBuffer(
            data=self.data[:, clamped_start:clamped_end].copy(),
            sample_rate=self.sample_rate,
            channel_mode=self.channel_mode,
        )

    def get_channel(self, idx: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Get 1D float32 slice for a specific channel index."""
        if not 0 <= idx < self.channels:
            raise IndexError(f"Channel index {idx} out of range (0..{self.channels - 1})")
        return np.ascontiguousarray(self.data[idx], dtype=np.float32)

    def to_mono(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Return 1D array by averaging all channels."""
        if self.channels == 1:
            return np.ascontiguousarray(self.data[0], dtype=np.float32)
        return np.ascontiguousarray(np.mean(self.data, axis=0), dtype=np.float32)

    def compute_sha256(self) -> str:
        """Compute SHA-256 of the raw audio data array."""
        return hash_numpy(self.data)

    def clone(self) -> "AudioBuffer":
        """Create an independent deep copy of the buffer."""
        return AudioBuffer(
            data=self.data.copy(),
            sample_rate=self.sample_rate,
            channel_mode=self.channel_mode,
        )
