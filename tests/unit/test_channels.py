"""Unit tests for channel layout classification and ambiguous stereo rejection."""

import numpy as np
import pytest

from voiceclean.audio.channels import classify_channels, handle_channel_layout
from voiceclean.audio.types import AudioBuffer, ChannelMode
from voiceclean.errors import AmbiguousStereoError


@pytest.mark.unit
def test_channel_mode_mono() -> None:
    data = np.zeros((1, 1000), dtype=np.float32)
    buf = AudioBuffer(data=data, sample_rate=48000)
    mode = classify_channels(buf, declared_mode="auto")
    assert mode == ChannelMode.MONO
    channels, dup = handle_channel_layout(buf, mode)
    assert len(channels) == 1
    assert dup is False


@pytest.mark.unit
def test_channel_mode_dual_mono_identical() -> None:
    mono = np.random.default_rng(42).normal(0.0, 0.2, size=5000).astype(np.float32)
    stereo = np.stack([mono, mono], axis=0)
    buf = AudioBuffer(data=stereo, sample_rate=48000)
    mode = classify_channels(buf, declared_mode="auto")
    assert mode == ChannelMode.DUAL_MONO_SAME
    channels, dup = handle_channel_layout(buf, mode)
    assert len(channels) == 1
    assert dup is True


@pytest.mark.unit
def test_channel_mode_split_speakers() -> None:
    rng = np.random.default_rng(42)
    ch0 = np.zeros(10000, dtype=np.float32)
    ch1 = np.zeros(10000, dtype=np.float32)
    ch0[:4000] = rng.normal(0.0, 0.2, size=4000)
    ch1[5000:9000] = rng.normal(0.0, 0.2, size=4000)
    stereo = np.stack([ch0, ch1], axis=0)
    buf = AudioBuffer(data=stereo, sample_rate=48000)
    mode = classify_channels(buf, declared_mode="auto")
    assert mode == ChannelMode.SPLIT_SPEAKERS
    channels, dup = handle_channel_layout(buf, mode)
    assert len(channels) == 2
    assert dup is False


@pytest.mark.unit
def test_channel_mode_ambiguous_stereo_rejected() -> None:
    # Highly correlated but not identical with level difference
    rng = np.random.default_rng(42)
    mono = rng.normal(0.0, 0.2, size=5000).astype(np.float32)
    ch0 = mono * 0.8
    ch1 = mono * 0.4 + 0.1 * rng.normal(0.0, 0.1, size=5000).astype(np.float32)
    stereo = np.stack([ch0, ch1], axis=0)
    buf = AudioBuffer(data=stereo, sample_rate=48000)

    with pytest.raises(AmbiguousStereoError):
        classify_channels(buf, declared_mode="auto")
