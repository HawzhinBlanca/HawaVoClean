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
