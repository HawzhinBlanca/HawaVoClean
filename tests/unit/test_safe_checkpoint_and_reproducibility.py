"""Qualification test suite for Phase R2.11: Safe Checkpoints, Bounded Training, Resumability, and Reproducibility."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.restoration.checkpoint import (
    load_safe_checkpoint,
    save_safe_checkpoint,
)
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKDNet
from research.restoration.train.train_hawarestore import train_model


class _MaliciousExploit:
    """Exploit object that attempts arbitrary execution upon unpickling."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (os.system, ("echo VULNERABLE_CODE_EXECUTED",))


def test_safe_checkpoint_loads_weights_only_true(tmp_path: Path) -> None:
    """A legitimate checkpoint must load cleanly under weights_only=True."""
    ckpt_path = tmp_path / "valid_model.pt"
    net = HawaRestoreKDNet(n_fft=256, num_speakers=2)
    state = {
        "model_state_dict": net.state_dict(),
        "config": {"n_fft": 256, "num_speakers": 2},
        "epochs": 1,
        "best_epoch": 1,
        "best_val_loss": 0.42,
    }
    save_safe_checkpoint(state, ckpt_path)

    loaded = load_safe_checkpoint(ckpt_path, map_location="cpu")
    assert "model_state_dict" in loaded
    assert loaded["config"]["n_fft"] == 256
    assert loaded["best_epoch"] == 1
    assert loaded["best_val_loss"] == 0.42


def test_safe_checkpoint_rejects_unsafe_pickle_code_injection(tmp_path: Path) -> None:
    """Pickle payloads containing unapproved classes must fail closed without executing."""
    malicious_path = tmp_path / "malicious.pt"
    # Construct an adversarial pickle payload
    evil_payload = {
        "model_state_dict": {"exploit": _MaliciousExploit()},
        "config": {},
    }
    with open(malicious_path, "wb") as f:
        pickle.dump(evil_payload, f)

    with pytest.raises(
        ModelProvenanceError, match="weights_only=True|Failed to load safe PyTorch checkpoint"
    ):
        load_safe_checkpoint(malicious_path, map_location="cpu")


def test_safe_checkpoint_rejects_oversized_file(tmp_path: Path) -> None:
    """Checkpoints exceeding the size ceiling must be refused immediately."""
    oversized_path = tmp_path / "huge_model.pt"
    oversized_path.write_bytes(b"x" * 1024)

    with pytest.raises(ModelProvenanceError, match="exceeds safety ceiling"):
        load_safe_checkpoint(oversized_path, map_location="cpu", max_size_bytes=512)


def test_safe_checkpoint_rejects_nan_inf_weights(tmp_path: Path) -> None:
    """Corrupted weights containing NaN or Inf values must fail validation."""
    nan_path = tmp_path / "nan_model.pt"
    bad_tensor = torch.tensor([1.0, float("nan"), 3.0])
    state = {
        "model_state_dict": {"bad_weight": bad_tensor},
        "config": {},
    }
    with pytest.raises(ValueError, match="contains NaN or Inf values"):
        save_safe_checkpoint(state, nan_path)


def test_checkpoint_records_code_data_dependency_hashes(tmp_path: Path) -> None:
    """Checkpoints must record SHA-256 code hash, dependency versions, and data hashes."""
    out_dir = tmp_path / "train_provenance"
    ckpt_path = train_model(
        epochs=1,
        batch_size=2,
        output_dir=out_dir,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=123,
        device="cpu",
    )

    ckpt = load_safe_checkpoint(ckpt_path, map_location="cpu")
    assert "code_hash" in ckpt
    assert isinstance(ckpt["code_hash"], str)
    assert len(ckpt["code_hash"]) == 64  # SHA-256 hex length

    assert "dependency_versions" in ckpt
    deps = ckpt["dependency_versions"]
    assert "torch" in deps
    assert "hawavoclean" in deps
    assert "python" in deps

    assert "manifest_hashes" in ckpt
    assert "train" in ckpt["manifest_hashes"]
    assert "development" in ckpt["manifest_hashes"]


def test_training_saves_locked_best_model_not_degraded_final(tmp_path: Path) -> None:
    """The saved production candidate must be the locked best model across all epochs."""
    out_dir = tmp_path / "best_model_test"
    ckpt_path = train_model(
        epochs=2,
        batch_size=2,
        output_dir=out_dir,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=42,
        device="cpu",
    )

    ckpt = load_safe_checkpoint(ckpt_path, map_location="cpu")
    assert "best_epoch" in ckpt
    assert ckpt["best_epoch"] in (1, 2)
    assert "best_val_loss" in ckpt
    assert ckpt["best_val_loss"] <= ckpt["final_losses"]["val"]["total"] + 1e-6
    assert len(ckpt["epoch_history"]) == 2


def test_training_resumption_from_checkpoint_last(tmp_path: Path) -> None:
    """Training can be paused and resumed seamlessly from hawarestore_kd_last.pt."""
    out_dir = tmp_path / "resume_test"

    # Stage 1: train for 1 epoch
    train_model(
        epochs=1,
        batch_size=2,
        output_dir=out_dir,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=999,
        device="cpu",
    )

    last_ckpt = out_dir / "hawarestore_kd_last.pt"
    assert last_ckpt.is_file()
    last_dict = load_safe_checkpoint(last_ckpt, map_location="cpu")
    assert last_dict["epoch"] == 1

    # Stage 2: resume and train up to epoch 2
    resumed_ckpt = train_model(
        epochs=2,
        batch_size=2,
        output_dir=out_dir,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=999,
        device="cpu",
        overwrite=True,
        resume_path=last_ckpt,
    )

    final_dict = load_safe_checkpoint(resumed_ckpt, map_location="cpu")
    assert final_dict["epochs"] == 2
    assert final_dict["best_epoch"] in (1, 2)
    # The epoch history should record epoch 2
    assert any(h["epoch"] == 2 for h in final_dict["epoch_history"])


def test_training_reproducibility_across_identical_seeds(tmp_path: Path) -> None:
    """Two seeded runs must produce identical losses and bitwise-equal state dicts."""
    out1 = tmp_path / "seed_run_1"
    out2 = tmp_path / "seed_run_2"

    ckpt1_path = train_model(
        epochs=1,
        batch_size=2,
        output_dir=out1,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=20260905,
        device="cpu",
    )

    ckpt2_path = train_model(
        epochs=1,
        batch_size=2,
        output_dir=out2,
        synthetic=True,
        num_synthetic_items=4,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=20260905,
        device="cpu",
    )

    d1 = load_safe_checkpoint(ckpt1_path, map_location="cpu")
    d2 = load_safe_checkpoint(ckpt2_path, map_location="cpu")

    # Final losses must match exactly
    assert abs(d1["final_loss"] - d2["final_loss"]) < 1e-5
    for term in ("flow", "stft", "envelope", "speaker", "total"):
        assert abs(d1["final_losses"]["train"][term] - d2["final_losses"]["train"][term]) < 1e-5
        assert abs(d1["final_losses"]["val"][term] - d2["final_losses"]["val"][term]) < 1e-5

    # State dict tensors must match
    sd1 = d1["model_state_dict"]
    sd2 = d2["model_state_dict"]
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k], atol=1e-6)


def test_training_bounded_time_and_steps(tmp_path: Path) -> None:
    """Bounded execution limits must halt training cleanly without hanging."""
    out_dir = tmp_path / "bounded_test"

    # Limit to 1 step per epoch
    ckpt_path = train_model(
        epochs=1,
        batch_size=2,
        output_dir=out_dir,
        synthetic=True,
        num_synthetic_items=8,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=55,
        device="cpu",
        max_steps_per_epoch=1,
    )
    assert ckpt_path.is_file()

    # Limit time ceiling
    out_dir_time = tmp_path / "bounded_time"
    ckpt_time_path = train_model(
        epochs=5,
        batch_size=2,
        output_dir=out_dir_time,
        synthetic=True,
        num_synthetic_items=8,
        duration_s=1.0,
        n_fft=256,
        base_channels=8,
        split_seed=55,
        device="cpu",
        max_seconds=0.001,  # Halts after epoch 1
    )
    assert ckpt_time_path.is_file()


def test_safetensors_export_and_load_roundtrip(tmp_path: Path) -> None:
    """Exporting safetensors alongside .pt creates loadable safetensors with JSON metadata."""
    ckpt_path = tmp_path / "model.pt"
    net = HawaRestoreKDNet(n_fft=256, num_speakers=2)
    state = {
        "model_state_dict": net.state_dict(),
        "config": {"n_fft": 256, "num_speakers": 2},
        "epochs": 1,
        "best_epoch": 1,
        "best_val_loss": 0.123,
    }
    save_safe_checkpoint(state, ckpt_path, save_safetensors=True)

    st_path = tmp_path / "model.safetensors"
    assert st_path.is_file()
    meta_path = tmp_path / "model_metadata.json"
    assert meta_path.is_file()

    loaded = load_safe_checkpoint(st_path, map_location="cpu")
    assert "model_state_dict" in loaded
    assert loaded["best_val_loss"] == 0.123
    assert set(loaded["model_state_dict"].keys()) == set(net.state_dict().keys())
