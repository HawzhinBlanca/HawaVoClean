"""Unit tests for branch coverage in hawavoclean.restoration.checkpoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.restoration.checkpoint import (
    SafeCheckpointProvenance,
    compute_code_provenance,
    compute_dependency_provenance,
    load_safe_checkpoint,
    save_safe_checkpoint,
)


def test_provenance_to_dict() -> None:
    prov = SafeCheckpointProvenance(
        weights_sha256="abc",
        code_hash="def",
        dependency_versions={"torch": "2.2.0"},
        split_seed=42,
        data_mode="stereo",
        epochs=10,
        best_epoch=5,
        best_val_loss=0.123,
        final_loss=0.150,
        train_speakers=["spk1"],
        val_speakers=["spk2"],
        manifest_hashes={"train": "h1"},
        active_loss_terms=["l1"],
        loss_weights={"l1": 1.0},
        final_losses={"val": {"l1": 0.1}},
        config={"lr": 1e-4},
    )
    d = prov.to_dict()
    assert isinstance(d, dict)
    assert d["split_seed"] == 42
    assert d["train_speakers"] == ["spk1"]


def test_compute_code_provenance_with_env_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_SOURCE_REVISION", "git-rev-test-12345")
    hash_with_env = compute_code_provenance()
    monkeypatch.delenv("HAWAVOCLEAN_SOURCE_REVISION", raising=False)
    hash_without_env = compute_code_provenance()
    assert hash_with_env != hash_without_env
    assert len(hash_with_env) == 64


def test_compute_dependency_provenance() -> None:
    deps = compute_dependency_provenance()
    assert "python" in deps
    assert "hawavoclean" in deps
    assert "torch" in deps


def test_save_and_load_safetensors_and_metadata(tmp_path: Path) -> None:
    pt_path = tmp_path / "model.pt"
    st_path = tmp_path / "model.safetensors"
    meta_path = tmp_path / "model_metadata.json"

    state = {
        "model_state_dict": {"w": torch.tensor([1.0, 2.0, 3.0])},
        "config": {"hidden": 64},
        "extra_info": "provenance_test",
    }
    digest = save_safe_checkpoint(state, pt_path, save_safetensors=True)
    assert len(digest) == 64
    assert pt_path.is_file()
    assert st_path.is_file()
    assert meta_path.is_file()

    # Load from .safetensors path
    loaded = load_safe_checkpoint(st_path)
    assert "model_state_dict" in loaded
    assert torch.equal(loaded["model_state_dict"]["w"], torch.tensor([1.0, 2.0, 3.0]))
    assert loaded["extra_info"] == "provenance_test"

    # Corrupt metadata sidecar
    meta_path.write_text("NOT_JSON", encoding="utf-8")
    with pytest.raises(ModelProvenanceError, match="Corrupted metadata sidecar"):
        load_safe_checkpoint(st_path)

    # Safetensors load error
    corrupted_st = tmp_path / "corrupted.safetensors"
    corrupted_st.write_bytes(b"corrupted binary")
    with pytest.raises(ModelProvenanceError, match="Failed to load safetensors"):
        load_safe_checkpoint(corrupted_st)


def test_load_safe_checkpoint_error_branches(tmp_path: Path) -> None:
    # 1. Nonexistent file
    with pytest.raises(ModelProvenanceError, match="not found"):
        load_safe_checkpoint(tmp_path / "missing.pt")

    # 2. Directory path
    with pytest.raises(ModelProvenanceError, match="not found"):
        load_safe_checkpoint(tmp_path)

    # 3. File size exceeds max_size_bytes
    tiny_cap = tmp_path / "large.pt"
    save_safe_checkpoint({"model_state_dict": {"w": torch.tensor([1.0])}}, tiny_cap)
    with pytest.raises(ModelProvenanceError, match="exceeds safety ceiling"):
        load_safe_checkpoint(tiny_cap, max_size_bytes=10)

    # 4. Corrupted pytorch file
    corrupt_pt = tmp_path / "bad.pt"
    corrupt_pt.write_bytes(b"not a valid pt file")
    with pytest.raises(ModelProvenanceError, match="Failed to load safe PyTorch checkpoint"):
        load_safe_checkpoint(corrupt_pt)

    # 5. Payload is not a dict
    list_pt = tmp_path / "list.pt"
    torch.save(["not", "a", "dict"], list_pt)
    with pytest.raises(ModelProvenanceError, match="payload is not a dictionary"):
        load_safe_checkpoint(list_pt)

    # 6. Missing model_state_dict
    no_sd_pt = tmp_path / "no_sd.pt"
    torch.save({"config": {}}, no_sd_pt)
    with pytest.raises(ModelProvenanceError, match="missing 'model_state_dict'"):
        load_safe_checkpoint(no_sd_pt)

    # 7. model_state_dict not a dict
    bad_sd_pt = tmp_path / "bad_sd.pt"
    torch.save({"model_state_dict": "not_a_dict"}, bad_sd_pt)
    with pytest.raises(ModelProvenanceError, match="'model_state_dict' is not a dictionary"):
        load_safe_checkpoint(bad_sd_pt)

    # 8. Parameter not a torch.Tensor
    not_tensor_pt = tmp_path / "not_tensor.pt"
    torch.save({"model_state_dict": {"w": "string_not_tensor"}}, not_tensor_pt)
    with pytest.raises(ModelProvenanceError, match="is not a torch.Tensor"):
        load_safe_checkpoint(not_tensor_pt)


def test_save_safe_checkpoint_validation_branches(tmp_path: Path) -> None:
    # 1. Missing model_state_dict
    with pytest.raises(ValueError, match="missing required 'model_state_dict'"):
        save_safe_checkpoint({}, tmp_path / "out.pt")

    # 2. model_state_dict not a dict
    with pytest.raises(ValueError, match="'model_state_dict' must be a dictionary"):
        save_safe_checkpoint({"model_state_dict": 12345}, tmp_path / "out.pt")

    # 3. State dict entry not a torch.Tensor
    with pytest.raises(ValueError, match="is not a torch.Tensor"):
        save_safe_checkpoint({"model_state_dict": {"k": [1.0, 2.0]}}, tmp_path / "out.pt")

    # 4. State dict entry contains Inf
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        save_safe_checkpoint(
            {"model_state_dict": {"k": torch.tensor([float("inf")])}}, tmp_path / "out.pt"
        )

    # 5. safetensors export failure does not break primary save
    valid_state = {"model_state_dict": {"w": torch.tensor([1.0, 2.0])}}
    with patch("safetensors.torch.save_file", side_effect=RuntimeError("safetensors crash")):
        digest = save_safe_checkpoint(
            valid_state, tmp_path / "safe_fallback.pt", save_safetensors=True
        )
        assert len(digest) == 64
        assert (tmp_path / "safe_fallback.pt").is_file()
