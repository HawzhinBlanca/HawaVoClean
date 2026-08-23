"""Wiring tests for the HawaRestore-KD training pipeline.

A real (tiny) training run in synthetic fallback mode proves that the
machinery the release claims is actually wired in: speaker-disjoint
SplitManager splits, every composite loss term live and finite, and a
checkpoint that reloads into HawaRestoreKDNet carrying the honesty metadata.
A second run over real WAV files exercises the --data-dir path end to end.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from hawavoclean.restoration.hawarestore_kd import HawaRestoreKDNet
from research.restoration.train.train_hawarestore import ACTIVE_LOSS_TERMS, train_model


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One tiny synthetic-mode training run shared by the assertions below."""
    out_dir = tmp_path_factory.mktemp("hawarestore-synth")
    ckpt_path = train_model(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        output_dir=out_dir,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=7,
        device="cpu",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {"ckpt_path": ckpt_path, "ckpt": ckpt, "out_dir": out_dir}


def test_all_composite_loss_terms_active_and_finite(synthetic_run: dict[str, Any]) -> None:
    """Flow, STFT, envelope, and speaker terms must all be live with finite values."""
    ckpt = synthetic_run["ckpt"]
    assert ckpt["active_loss_terms"] == list(ACTIVE_LOSS_TERMS)
    assert set(ACTIVE_LOSS_TERMS) == {"flow", "stft", "envelope", "speaker", "total"}

    for split in ("train", "val"):
        term_losses = ckpt["final_losses"][split]
        assert set(term_losses) == set(ACTIVE_LOSS_TERMS)
        for term, value in term_losses.items():
            assert math.isfinite(value), f"{split}/{term} is not finite: {value}"
            assert value >= 0.0, f"{split}/{term} is negative: {value}"


def test_split_is_speaker_disjoint(synthetic_run: dict[str, Any]) -> None:
    """No speaker may appear in both train and validation, in metadata or manifests."""
    ckpt = synthetic_run["ckpt"]
    train_speakers = set(ckpt["train_speakers"])
    val_speakers = set(ckpt["val_speakers"])
    assert train_speakers and val_speakers
    assert not train_speakers & val_speakers

    manifests = Path(synthetic_run["out_dir"]) / "manifests"
    with open(manifests / "train.jsonl", encoding="utf-8") as f:
        train_entries = [json.loads(line) for line in f if line.strip()]
    with open(manifests / "development.jsonl", encoding="utf-8") as f:
        val_entries = [json.loads(line) for line in f if line.strip()]

    assert len(train_entries) == ckpt["n_train"]
    assert len(val_entries) == ckpt["n_val"]
    assert {e["speaker_id"] for e in train_entries} == train_speakers
    assert {e["speaker_id"] for e in val_entries} == val_speakers

    train_ids = {e["utterance_id"] for e in train_entries}
    val_ids = {e["utterance_id"] for e in val_entries}
    assert not train_ids & val_ids


def test_checkpoint_reloads_with_metadata(synthetic_run: dict[str, Any]) -> None:
    """The saved checkpoint carries the new metadata and reloads into the model."""
    ckpt = synthetic_run["ckpt"]
    assert ckpt["data_mode"] == "synthetic"
    assert ckpt["epochs"] == 1
    assert ckpt["split_seed"] == 7
    assert ckpt["n_train"] >= 1
    assert ckpt["n_val"] >= 1
    assert ckpt["n_train"] + ckpt["n_val"] == 4
    assert set(ckpt["loss_weights"]) == {"flow", "stft", "envelope", "speaker"}
    assert math.isfinite(ckpt["final_loss"])
    assert set(ckpt["manifest_hashes"]) >= {"train", "development"}

    net = HawaRestoreKDNet(**ckpt["config"])
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    n_freqs = ckpt["config"]["n_fft"] // 2 + 1
    x = torch.randn(1, 2, n_freqs, 12)
    with torch.no_grad():
        v = net(
            x,
            torch.tensor([0.5]),
            torch.tensor([4000.0]),
            torch.tensor([0]),
            torch.randn(1, 192),
        )
    assert v.shape == x.shape
    assert torch.isfinite(v).all()


def test_refuses_to_overwrite_existing_checkpoint(synthetic_run: dict[str, Any]) -> None:
    """A second run into the same output directory must fail loudly, not clobber."""
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_model(
            epochs=1,
            output_dir=synthetic_run["out_dir"],
            synthetic=True,
            num_synthetic_items=4,
            device="cpu",
        )


def test_real_data_mode_trains_from_wav_files(tmp_path: Path) -> None:
    """--data-dir mode: real WAVs (mixed sample rates) resampled, degraded, trained."""
    rng = np.random.default_rng(11)
    data_dir = tmp_path / "corpus"
    for i, (speaker, sr) in enumerate(
        [("spk_a", 16000), ("spk_b", 48000), ("spk_c", 22050), ("spk_d", 48000)]
    ):
        f0 = 110.0 + 40.0 * i
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        sig = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        sig = (sig + 0.01 * rng.standard_normal(sr)).astype(np.float32)
        speaker_dir = data_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        sf.write(speaker_dir / f"{speaker}_utt.wav", sig, sr)

    ckpt_path = train_model(
        epochs=1,
        batch_size=2,
        output_dir=tmp_path / "ckpt-real",
        data_dir=data_dir,
        num_synthetic_items=0,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=3,
        device="cpu",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["data_mode"] == "real"
    assert ckpt["n_train"] >= 1
    assert ckpt["n_val"] >= 1
    assert not set(ckpt["train_speakers"]) & set(ckpt["val_speakers"])
    for split in ("train", "val"):
        for term, value in ckpt["final_losses"][split].items():
            assert math.isfinite(value), f"{split}/{term} is not finite: {value}"


def test_requires_exactly_one_data_mode(tmp_path: Path) -> None:
    """Neither or both of --data-dir/--synthetic is an error, not a silent default."""
    with pytest.raises(ValueError, match="exactly one data mode"):
        train_model(output_dir=tmp_path / "x")
    with pytest.raises(ValueError, match="exactly one data mode"):
        train_model(output_dir=tmp_path / "y", synthetic=True, data_dir=tmp_path)
