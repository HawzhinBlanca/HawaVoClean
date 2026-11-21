"""Studio core: registry, provenance consistency, and (if installed) inference."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from hawavoclean.enhancement.factory import CORE_REGISTRY, resolve_core
from hawavoclean.enhancement.studio import studio_params_hash, studio_weight_digests
from hawavoclean.hashing import hash_json_canonical

MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "hawavoclean" / "resources" / "models"


def test_registry_resolves_known_cores_and_rejects_unknown() -> None:
    assert resolve_core("wiener-dd-48k-v1").lock_filename == "production-core.lock.toml"
    assert resolve_core("studio-dfn3-48k-v1").lock_filename == "studio-core.lock.toml"
    with pytest.raises(KeyError):
        resolve_core("imaginary-core")


def test_every_registered_core_has_a_consistent_lockfile() -> None:
    """For each core: lockfile exists, tables reconstruct params_hash, and
    params_hash matches the implementation — without loading any model."""
    for core_id, reg in CORE_REGISTRY.items():
        lock_path = MODELS_DIR / reg.lock_filename
        assert lock_path.exists(), f"{core_id}: lockfile missing"
        with open(lock_path, "rb") as f:
            lock = tomllib.load(f)
        assert lock["core_id"] == core_id

        payload: dict[str, object] = dict(lock.get("params", {}))
        weights = {str(k): str(v) for k, v in dict(lock.get("weight_sha256", {})).items()}
        if weights:
            payload["weights_sha256"] = weights
        assert hash_json_canonical(payload) == lock["params_hash"], (
            f"{core_id}: lock tables do not reconstruct params_hash"
        )
        assert reg.implementation_params_hash() == lock["params_hash"], (
            f"{core_id}: implementation drifted from lockfile"
        )


def test_studio_weight_digests_resolve_on_disk() -> None:
    digests = studio_weight_digests()
    assert digests and all(v != "MISSING" for v in digests.values()), digests
    assert studio_params_hash() != hash_json_canonical({})


def test_studio_inference_preserves_length_and_finiteness() -> None:
    # Gate on torch: the df import itself needs the torchaudio compat shim,
    # which StudioVoiceCore installs before importing df.
    pytest.importorskip("torch")
    from hawavoclean.enhancement.studio import StudioVoiceCore

    core = StudioVoiceCore()
    sr = 48000
    rng = np.random.default_rng(0)
    t = np.arange(sr * 2) / sr
    x = (
        0.3 * np.sin(2 * np.pi * 200 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        + 0.03 * rng.standard_normal(sr * 2)
    ).astype(np.float32)

    res = core.enhance(x, sr)
    assert len(res.waveform) == len(x)
    assert np.all(np.isfinite(res.waveform))
    # It must attenuate the noise bed between modulation peaks.
    assert float(np.sqrt(np.mean(res.waveform**2))) < float(np.sqrt(np.mean(x**2)))


def test_studio_core_validation_and_branch_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    from hawavoclean.enhancement.studio import (
        StudioVoiceCore,
        studio_weight_digests,
    )

    # 1. Phase coherent rejection
    with pytest.raises(ValueError, match="not phase-coherent"):
        StudioVoiceCore(phase_coherent=True)

    # 2. Sample rate mismatch rejection
    with pytest.raises(ValueError, match="runs at 48000 Hz internally"):
        StudioVoiceCore(sample_rate=16000)

    core = StudioVoiceCore()
    assert core.metadata.core_id == "studio-dfn3-48k-v1"
    assert core.metadata.phase_coherent is False

    # 3. Zero-length input
    empty = np.zeros(0, dtype=np.float32)
    empty_res = core.enhance(empty, 48000)
    assert len(empty_res.waveform) == 0
    assert empty_res.input_samples == 0

    # 4. studio_weight_digests with missing file
    import hawavoclean.enhancement.studio as studio_mod

    fake_path = Path("/nonexistent/models/deepfilternet3")
    monkeypatch.setattr(studio_mod, "_MODEL_DIR", fake_path)
    digests = studio_weight_digests()
    assert all(v == "MISSING" for v in digests.values())


def test_studio_core_advanced_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    import hawavoclean.enhancement.studio as studio_mod
    from hawavoclean.enhancement.studio import (
        STUDIO_PARAMS,
        StudioVoiceCore,
        load_deepfilternet3,
    )

    # 1. Device property
    core = StudioVoiceCore()
    assert core.device == "cpu"

    # 2. Non-cpu device warning
    mock_model = MagicMock()
    mock_state = MagicMock()
    monkeypatch.setattr(studio_mod, "load_deepfilternet3", lambda _dev: (mock_model, mock_state))
    monkeypatch.setattr(
        studio_mod,
        "resolve_device",
        lambda *_args, **_kwargs: MagicMock(resolved="cuda"),
    )
    gpu_core = StudioVoiceCore(device="cuda")
    gpu_core._ensure_model()
    assert gpu_core._model is mock_model

    # 3. Warmup
    mock_core = StudioVoiceCore()
    mock_core._model = mock_model
    mock_core._df_state = mock_state
    monkeypatch.setattr(
        studio_mod,
        "run_deepfilternet3",
        lambda _m, _s, a, _lim: np.zeros_like(a),
    )
    mock_core.warmup()

    # 4. Device mismatch in load_deepfilternet3
    import sys

    from hawavoclean.enhancement.dependency_probe import install_torchaudio_compat

    install_torchaudio_compat()
    import df.enhance  # noqa: F401
    import df.utils  # noqa: F401

    monkeypatch.setattr(
        sys.modules["df.enhance"],
        "init_df",
        lambda **_kwargs: (MagicMock(), MagicMock(), None),
    )
    monkeypatch.setattr(sys.modules["df.utils"], "get_device", lambda: "cuda")

    with pytest.raises(ValueError, match="DeepFilterNet resolved device"):
        load_deepfilternet3("cpu")

    # 5. WPE dereverb and tail_suppress branches
    wpe_core = StudioVoiceCore()
    wpe_core._model = mock_model
    wpe_core._df_state = mock_state
    monkeypatch.setitem(STUDIO_PARAMS, "wpe_dereverb", True)
    monkeypatch.setitem(STUDIO_PARAMS, "tail_suppress", True)

    sr = 48000
    audio = np.sin(np.linspace(0, 100 * np.pi, sr // 10, dtype=np.float32))
    # Return hot audio to trigger peak > 0.99 scaling
    monkeypatch.setattr(
        studio_mod,
        "run_deepfilternet3",
        lambda _m, _s, a, _lim: np.full_like(a, 2.0),
    )
    res = wpe_core.enhance(audio, sr)
    assert np.max(np.abs(res.waveform)) <= 0.991
