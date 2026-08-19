"""Unit tests for HawzhinSoraniASR and FakeSoraniASR."""

import numpy as np
import pytest

from voiceclean.guard.hawzhin_ctc import SORANI_VOCAB, FakeSoraniASR, HawzhinSoraniASR


@pytest.mark.unit
def test_sorani_vocab_non_empty() -> None:
    assert len(SORANI_VOCAB) > 30
    assert "<blank>" in SORANI_VOCAB
    assert "ی" in SORANI_VOCAB
    assert "ک" in SORANI_VOCAB


@pytest.mark.unit
def test_fake_sorani_asr_infer() -> None:
    asr = FakeSoraniASR()
    sig = np.random.default_rng(42).normal(0.0, 0.2, size=48000).astype(np.float32)
    res = asr.infer(sig, 48000)
    assert len(res.normalized_transcript) > 0
    assert len(res.tokens) > 0
    assert res.frame_posteriors.shape[0] > 0
    assert res.frame_posteriors.shape[1] == len(SORANI_VOCAB)


@pytest.mark.unit
def test_hawzhin_sorani_asr_fallback_mode() -> None:
    asr = HawzhinSoraniASR()
    sig = np.zeros(24000, dtype=np.float32)
    res = asr.infer(sig, 48000)
    assert res.frame_posteriors.dtype == np.float32
    assert len(res.frame_timestamps) == len(res.frame_posteriors)
