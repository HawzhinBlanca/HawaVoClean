"""AudioBuffer surface and channel classification edge branches."""

import numpy as np
import pytest

from voiceclean.audio.channels import classify_channels, handle_channel_layout
from voiceclean.audio.types import AudioBuffer, ChannelMode
from voiceclean.errors import AmbiguousStereoError

SR = 48000


def _buf(data: np.ndarray, mode: ChannelMode = ChannelMode.MONO) -> AudioBuffer:
    return AudioBuffer(data=data, sample_rate=SR, channel_mode=mode)


def test_audiobuffer_surface() -> None:
    stereo = _buf(np.vstack([np.ones(SR), np.zeros(SR)]).astype(np.float32))
    assert stereo.channels == 2
    assert stereo.samples == SR
    assert stereo.duration_s == pytest.approx(1.0)

    sl = stereo.slice(-10, 100)
    assert sl.samples == 100

    mono_mix = stereo.to_mono()
    assert mono_mix[0] == pytest.approx(0.5)

    mono = _buf(np.ones(100, dtype=np.float32))
    assert mono.to_mono()[0] == 1.0

    with pytest.raises(IndexError):
        stereo.get_channel(5)

    clone = stereo.clone()
    clone.data[0, 0] = -1.0
    assert stereo.data[0, 0] == 1.0  # deep copy

    assert len(stereo.compute_sha256()) == 64

    with pytest.raises(ValueError):
        _buf(np.zeros((2, 2, 2), dtype=np.float32))

    f64 = _buf(np.zeros(10, dtype=np.float64))  # dtype coercion branch
    assert f64.data.dtype == np.float32


def test_declared_mode_overrides_classification() -> None:
    ambiguous = (
        np.vstack(
            [
                np.random.default_rng(0).standard_normal(SR),
                np.random.default_rng(1).standard_normal(SR),
            ]
        ).astype(np.float32)
        * 0.1
    )
    buf = _buf(ambiguous)
    assert classify_channels(buf, declared_mode="split_speakers") == ChannelMode.SPLIT_SPEAKERS
    with pytest.raises(ValueError):
        classify_channels(buf, declared_mode="quadraphonic")


def test_multichannel_beyond_stereo_rejected() -> None:
    five_one = _buf(np.zeros((6, SR), dtype=np.float32))
    with pytest.raises(AmbiguousStereoError):
        classify_channels(five_one, declared_mode="auto")


def test_silent_stereo_classifies_dual_mono() -> None:
    silent = _buf(np.zeros((2, SR), dtype=np.float32))
    assert classify_channels(silent, declared_mode="auto") == ChannelMode.DUAL_MONO_SAME


def test_handle_layout_variants() -> None:
    stereo = _buf(np.vstack([np.ones(SR), np.ones(SR)]).astype(np.float32))
    chans, dup = handle_channel_layout(stereo, ChannelMode.DUAL_MONO_SAME)
    assert len(chans) == 1 and dup is True

    chans, dup = handle_channel_layout(stereo, ChannelMode.SPLIT_SPEAKERS)
    assert len(chans) == 2 and dup is False

    mono = _buf(np.ones(SR, dtype=np.float32))
    chans, dup = handle_channel_layout(mono, ChannelMode.MONO)
    assert len(chans) == 1 and dup is False

    with pytest.raises(AmbiguousStereoError):
        handle_channel_layout(stereo, ChannelMode.AMBIGUOUS_STEREO)
