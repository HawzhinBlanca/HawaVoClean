"""Channel classification and safety verification according to BLUEPRINT.md section 9.3."""

from typing import Any

import numpy as np

from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.errors import AmbiguousStereoError, InvalidUserInputError

CHANNEL_ANALYSIS_CHUNK_SAMPLES = 1 << 20


def classify_channels(
    buffer: AudioBuffer,
    declared_mode: str = "auto",
) -> ChannelMode:
    """Classify channel relationship or validate explicit user configuration."""
    if declared_mode != "auto":
        try:
            mode = ChannelMode(declared_mode)
        except ValueError as e:
            raise ValueError(f"Unknown channel_mode '{declared_mode}'") from e
        # A declared mode that contradicts the file fails HERE, not after the
        # whole pipeline has run ('Assembled output channels 1 != 2').
        if mode == ChannelMode.MONO and buffer.channels != 1:
            raise InvalidUserInputError(
                f"channel_mode 'mono' declared but the file has {buffer.channels} channels; "
                "use 'dual_mono_same' (identical channels) or 'split_speakers'"
            )
        if mode in (ChannelMode.DUAL_MONO_SAME, ChannelMode.SPLIT_SPEAKERS) and buffer.channels < 2:
            raise InvalidUserInputError(
                f"channel_mode '{declared_mode}' declared but the file is mono"
            )
        if mode == ChannelMode.AMBIGUOUS_STEREO:
            raise InvalidUserInputError(
                "channel_mode 'ambiguous_stereo' is a classification result, not a declaration; "
                "declare 'dual_mono_same' or 'split_speakers'"
            )
        return mode

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
            "'dual_mono_same' (channels carry the same signal) or 'split_speakers' "
            "(one speaker per channel)."
        )

    raise AmbiguousStereoError(
        f"Multi-channel audio with {channels} channels is not supported without explicit split_speakers declaration."
    )


def classify_channels_bounded(
    buffer: AudioBuffer,
    declared_mode: str = "auto",
    *,
    chunk_samples: int = CHANNEL_ANALYSIS_CHUNK_SAMPLES,
) -> ChannelMode:
    """Classify channels with memory proportional to one analysis chunk.

    Explicit declarations and mono inputs use the ordinary validator. The
    stereo auto path computes the same maximum-difference, correlation and RMS
    criteria as :func:`classify_channels`, but reduces them in float64 chunks
    instead of allocating ``ch0 - ch1`` for the complete recording.
    """
    if declared_mode != "auto" or buffer.channels != 2:
        return classify_channels(buffer, declared_mode=declared_mode)
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples}")

    max_diff = 0.0
    dot = 0.0
    norm0_sq = 0.0
    norm1_sq = 0.0
    samples = buffer.samples
    for start in range(0, samples, chunk_samples):
        stop = min(samples, start + chunk_samples)
        ch0 = buffer.data[0, start:stop].astype(np.float64, copy=False)
        ch1 = buffer.data[1, start:stop].astype(np.float64, copy=False)
        max_diff = max(max_diff, float(np.max(np.abs(ch0 - ch1))))
        dot += float(np.dot(ch0, ch1))
        norm0_sq += float(np.dot(ch0, ch0))
        norm1_sq += float(np.dot(ch1, ch1))

    if max_diff < 1e-5:
        return ChannelMode.DUAL_MONO_SAME

    norm0 = float(np.sqrt(norm0_sq))
    norm1 = float(np.sqrt(norm1_sq))
    if norm0 > 1e-6 and norm1 > 1e-6:
        correlation = dot / (norm0 * norm1)
    else:
        correlation = 1.0 if (norm0 <= 1e-6 and norm1 <= 1e-6) else 0.0
    rms0 = float(np.sqrt(norm0_sq / max(samples, 1)))
    rms1 = float(np.sqrt(norm1_sq / max(samples, 1)))
    level_ratio = rms0 / (rms1 + 1e-9)

    if correlation > 0.999 and 0.98 <= level_ratio <= 1.02:
        return ChannelMode.DUAL_MONO_SAME
    if correlation < 0.40:
        return ChannelMode.SPLIT_SPEAKERS
    raise AmbiguousStereoError(
        f"Input stereo channels exhibit correlation={correlation:.3f} and "
        f"level_ratio={level_ratio:.2f}. Auto-classification returned "
        "'ambiguous_stereo'. To prevent phase/spatial corruption, declare "
        "channel_mode in config explicitly: 'dual_mono_same' (channels carry "
        "the same signal) or 'split_speakers' (one speaker per channel)."
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
