from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hawavoclean.paths import profiles_root, restoration_checkpoint_path
from hawavoclean.policy.strength import generate_strength_candidates
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from hawavoclean.server.policy import PathPolicyError, resolve_client_output_path


def test_generate_strength_candidates_incoherent() -> None:
    orig = np.zeros(1600, dtype=np.float32)
    enh = np.ones(1600, dtype=np.float32)
    candidates = generate_strength_candidates(orig, enh, [1.0, 0.8, 0.5], phase_coherent=False)
    assert len(candidates) == 1
    assert candidates[0][0] == 1.0
    assert np.allclose(candidates[0][1], enh)


def test_paths_environment_overrides(tmp_path: Path) -> None:
    fake_ckpt = tmp_path / "custom_checkpoint.pt"
    fake_ckpt.write_bytes(b"ckpt")
    with patch.dict(os.environ, {"HAWAVOCLEAN_RESTORATION_CHECKPOINT": str(fake_ckpt)}):
        assert restoration_checkpoint_path() == fake_ckpt.resolve()

    fake_profiles = tmp_path / "custom_profiles"
    fake_profiles.mkdir()
    with patch.dict(os.environ, {"HAWAVOCLEAN_PROFILES_DIR": str(fake_profiles)}):
        assert profiles_root() == fake_profiles.resolve()


def test_resolve_client_output_path_edge_cases() -> None:
    # 1. Empty string
    with pytest.raises(PathPolicyError) as exc_empty:
        resolve_client_output_path("   ")
    assert exc_empty.value.status == 400
    assert exc_empty.value.code == "bad_request"
    assert "path is required" in exc_empty.value.message

    # 2. Relative path
    with pytest.raises(PathPolicyError) as exc_rel:
        resolve_client_output_path("relative/path/to/output.wav")
    assert exc_rel.value.status == 400
    assert exc_rel.value.code == "bad_request"
    assert "path must be absolute" in exc_rel.value.message


def test_speaker_embed_dc_and_sine_rejection() -> None:
    extractor = SpeakerEmbeddingExtractor(sample_rate=48000)

    # 1. Constant DC audio (std < 1e-5) -> returns zero embedding
    dc_audio = np.full(48000, 0.5, dtype=np.float32)
    embed_dc = extractor.extract(dc_audio)
    assert np.all(embed_dc == 0.0)

    # 2. Pure sine wave (concentrated in top 3 bins) -> returns zero embedding
    t = np.linspace(0, 1.0, 48000, endpoint=False, dtype=np.float32)
    sine_audio = (0.8 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    embed_sine = extractor.extract(sine_audio)
    assert np.all(embed_sine == 0.0)
