"""Unit tests for SpectralSignatureProbe and FixedProbe."""

import numpy as np
import pytest

from hawavoclean.guard.spectral_probe import SORANI_VOCAB, FixedProbe, SpectralSignatureProbe


@pytest.mark.unit
def test_sorani_vocab_non_empty() -> None:
    assert len(SORANI_VOCAB) > 30
    assert "<blank>" in SORANI_VOCAB
    assert "ی" in SORANI_VOCAB
    assert "ک" in SORANI_VOCAB


@pytest.mark.unit
def test_fake_sorani_asr_infer() -> None:
    asr = FixedProbe()
    sig = np.random.default_rng(42).normal(0.0, 0.2, size=48000).astype(np.float32)
    res = asr.infer(sig, 48000)
    assert len(res.normalized_signature) > 0
    assert len(res.tokens) > 0
    assert res.frame_distributions.shape[0] > 0
    assert res.frame_distributions.shape[1] == len(SORANI_VOCAB)


@pytest.mark.unit
def test_hawzhin_sorani_asr_fallback_mode() -> None:
    asr = SpectralSignatureProbe()
    sig = np.zeros(24000, dtype=np.float32)
    res = asr.infer(sig, 48000)
    assert res.frame_distributions.dtype == np.float32
    assert len(res.frame_timestamps) == len(res.frame_distributions)
