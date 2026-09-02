"""Unit tests for SoraniLinguisticGuard and phonetic stability checks."""

from __future__ import annotations

import numpy as np

from hawavoclean.restoration.linguistic_guard import (
    LinguisticGuardResult,
    SoraniLinguisticGuard,
)


def test_linguistic_guard_result_to_dict() -> None:
    res = LinguisticGuardResult(
        divergence=0.05,
        anchor_preserved=True,
        status="anchor_preserved",
        max_frame_divergence=0.10,
        passes_check=True,
    )
    d = res.to_dict()
    assert d["divergence"] == 0.05
    assert d["anchor_preserved"] is True
    assert d["status"] == "anchor_preserved"
    assert d["max_frame_divergence"] == 0.10
    assert d["passes_check"] is True


def test_linguistic_guard_audio_too_short() -> None:
    guard = SoraniLinguisticGuard(sample_rate=48000)
    short_nat = np.zeros(100, dtype=np.float32)
    short_rest = np.zeros(100, dtype=np.float32)
    res = guard.evaluate(short_nat, short_rest)
    assert res.status == "audio_too_short"
    assert res.passes_check is True
    assert res.divergence == 0.0


def test_linguistic_guard_stereo_and_speaker_adaptive_f0() -> None:
    guard = SoraniLinguisticGuard(sample_rate=48000, threshold=0.30)
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    # 2D stereo signal with 440 Hz fundamental
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    stereo_nat = np.stack([tone, tone], axis=0)
    stereo_rest = np.stack([tone, tone], axis=0)

    f0_stats = {"median_hz": 150.0, "p05_hz": 100.0, "p95_hz": 220.0}
    res = guard.evaluate(stereo_nat, stereo_rest, f0_statistics=f0_stats)
    assert res.passes_check is True
    assert res.status == "anchor_preserved"
    assert res.divergence < 0.05


def test_linguistic_guard_phonetic_divergence_detected() -> None:
    guard = SoraniLinguisticGuard(sample_rate=48000, threshold=0.15)
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    # Natural audio is 440 Hz tone, Restored is completely different 2500 Hz tone
    nat = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    rest = 0.5 * np.sin(2.0 * np.pi * 2500.0 * t).astype(np.float32)

    res = guard.evaluate(nat, rest)
    assert res.passes_check is False
    assert res.status == "phonetic_divergence_detected"
    assert res.divergence > 0.15


def test_linguistic_guard_silent_fallback() -> None:
    guard = SoraniLinguisticGuard(sample_rate=48000)
    # Near silence
    silence = np.full(48000, 1e-6, dtype=np.float32)
    res = guard.evaluate(silence, silence)
    assert res.passes_check is True
