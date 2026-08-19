"""Channel classification and safety verification according to BLUEPRINT.md section 9.3."""

from typing import Any

import numpy as np

from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.errors import AmbiguousStereoError


def classify_channels(
    buffer: AudioBuffer,
    declared_mode: str = "auto",
) -> ChannelMode:
    """Classify channel relationship or validate explicit user configuration."""
    if declared_mode != "auto":
        try:
            return ChannelMode(declared_mode)
        except ValueError as e:
            raise ValueError(f"Unknown channel_mode '{declared_mode}'") from e

    channels = buffer.channels
    if channels == 1:
        return ChannelMode.MONO

    if channels == 2:
        ch0 = buffer.get_channel(0)
        ch1 = buffer.get_channel(1)

        # Check identical / duplicated mono
        diff = np.abs(ch0 - ch1)
        max_diff = float(np.max(diff))
        if max_diff < 1e-5:
            return ChannelMode.DUAL_MONO_SAME

        # Pearson correlation and RMS level comparison
        norm0 = np.linalg.norm(ch0)
        norm1 = np.linalg.norm(ch1)

        if norm0 > 1e-6 and norm1 > 1e-6:
            correlation = float(np.dot(ch0, ch1) / (norm0 * norm1))
        else:
            correlation = 1.0 if (norm0 <= 1e-6 and norm1 <= 1e-6) else 0.0

        rms0 = float(np.sqrt(np.mean(ch0**2)))
        rms1 = float(np.sqrt(np.mean(ch1**2)))
        level_ratio = rms0 / (rms1 + 1e-9)

        if correlation > 0.999 and 0.98 <= level_ratio <= 1.02:
            return ChannelMode.DUAL_MONO_SAME

        # Check split speakers: low cross-correlation (<0.40) and active dialogue
        if correlation < 0.40:
            return ChannelMode.SPLIT_SPEAKERS

        # Ambiguous stereo: music, stereo reverb, panning, etc.
        raise AmbiguousStereoError(
            f"Input stereo channels exhibit correlation={correlation:.3f} and level_ratio={level_ratio:.2f}. "
            "Auto-classification returned 'ambiguous_stereo'. "
            "To prevent phase/spatial corruption, declare channel_mode in config explicitly: "
            "'mono', 'dual_mono_same', or 'split_speakers'."
        )

    raise AmbiguousStereoError(
        f"Multi-channel audio with {channels} channels is not supported without explicit split_speakers declaration."
    )


def handle_channel_layout(
    buffer: AudioBuffer,
    channel_mode: ChannelMode,
) -> tuple[list[np.ndarray[Any, np.dtype[np.float32]]], bool]:
    """Return list of channels to process and whether to duplicate result to 2nd channel."""
    if channel_mode == ChannelMode.MONO:
        return [buffer.get_channel(0)], False
    elif channel_mode == ChannelMode.DUAL_MONO_SAME:
        # Process first channel only, duplicate output
        return [buffer.get_channel(0)], True
    elif channel_mode == ChannelMode.SPLIT_SPEAKERS:
        return [buffer.get_channel(i) for i in range(buffer.channels)], False
    else:
        raise AmbiguousStereoError(f"Cannot process under unhandled channel mode: {channel_mode}")
