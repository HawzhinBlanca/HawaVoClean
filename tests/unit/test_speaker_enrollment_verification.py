"""Unit tests for speaker enrollment governance and neural speaker verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.enrollment import enroll_speaker
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor

SR = 48000


def _generate_synthetic_speaker(f0: float, seed: int, duration_s: float = 1.0) -> np.ndarray:
    """Generate speech-like synthetic harmonic audio for testing speaker discrimination."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    f0_mod = f0 + 15.0 * np.sin(2.0 * np.pi * 3.0 * t)
    phase = 2.0 * np.pi * np.cumsum(f0_mod) / SR
    rng = np.random.default_rng(seed)
    speech = (
        0.5 * np.sin(phase)
        + 0.35 * np.sin(2 * phase)
        + 0.2 * np.sin(3 * phase)
        + 0.1 * rng.standard_normal(len(t))
    ).astype(np.float32)
    return speech


def test_enroll_speaker_refuses_without_consent(tmp_path: Path) -> None:
    """enroll_speaker must raise ValueError if consent_granted is False."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "sample.wav", np.zeros(SR * 2, dtype=np.float32), SR)

    with pytest.raises(ValueError, match="consent_granted=False"):
        enroll_speaker(
            speaker_id="test_speaker",
            display_name="Test Speaker",
            audio_dir=audio_dir,
            output_dir=tmp_path / "out",
            consent_granted=False,
        )


def test_enroll_speaker_refuses_insufficient_duration(tmp_path: Path) -> None:
    """enroll_speaker must raise ValueError if total audio duration is < 300 seconds."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    spk1 = _generate_synthetic_speaker(150.0, seed=1, duration_s=10.0)
    sf.write(audio_dir / "sample1.wav", spk1, SR)

    with pytest.raises(ValueError, match="Insufficient total audio duration"):
        enroll_speaker(
            speaker_id="test_speaker",
            display_name="Test Speaker",
            audio_dir=audio_dir,
            output_dir=tmp_path / "out",
            consent_granted=True,
            min_duration_s=300.0,
        )


def test_speaker_embed_sine_wave_returns_zero_vector() -> None:
    """Pure sine wave tones must yield all-zero embedding vector."""
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    sine = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)

    emb = extractor.extract(sine)
    assert emb.shape == (192,)
    assert float(np.linalg.norm(emb)) == 0.0


def test_speaker_embed_discriminates_distinct_speakers() -> None:
    """Different speaker formants/harmonics produce distinct embeddings (low similarity)."""
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)

    spk_a1 = _generate_synthetic_speaker(120.0, seed=10, duration_s=2.0)
    spk_a2 = _generate_synthetic_speaker(120.0, seed=10, duration_s=2.0)
    spk_b = _generate_synthetic_speaker(240.0, seed=99, duration_s=2.0)

    emb_a1 = extractor.extract(spk_a1)
    emb_a2 = extractor.extract(spk_a2)
    emb_b = extractor.extract(spk_b)

    # Same speaker cosine similarity should be high (> 0.85)
    sim_same = float(np.dot(emb_a1, emb_a2))
    assert sim_same > 0.85

    # Different speaker cosine similarity should be low (< 0.40)
    sim_diff = float(np.dot(emb_a1, emb_b))
    assert sim_diff < 0.40
