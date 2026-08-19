"""Unit tests for polyphase resampling."""

import numpy as np
import pytest

from voiceclean.audio.resample import resample_audio


@pytest.mark.unit
def test_resample_identity() -> None:
    wave = np.random.default_rng(42).normal(0.0, 0.2, size=1000).astype(np.float32)
    res = resample_audio(wave, orig_sr=48000, target_sr=48000)
    assert np.array_equal(wave, res)


@pytest.mark.unit
def test_resample_48k_to_16k_exact_length() -> None:
    wave_48k = np.zeros(48000, dtype=np.float32)
    res_16k = resample_audio(wave_48k, orig_sr=48000, target_sr=16000)
    assert len(res_16k) == 16000
    assert res_16k.dtype == np.float32
    assert np.all(np.isfinite(res_16k))


@pytest.mark.unit
def test_resample_with_target_samples_enforcement() -> None:
    wave_16k = np.zeros(16000, dtype=np.float32)
    res_48k = resample_audio(wave_16k, orig_sr=16000, target_sr=48000, target_samples=48000)
    assert len(res_48k) == 48000
