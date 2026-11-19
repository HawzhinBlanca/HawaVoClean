"""Unit tests for StudioProCore (MossFormer2_SE_48K), preferred_devices, and contracts."""

from __future__ import annotations

import builtins
import sys
import tomllib
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from hawavoclean import runtime
from hawavoclean.config import load_config
from hawavoclean.enhancement import dependency_probe, studio_pro
from hawavoclean.enhancement.factory import CORE_REGISTRY, resolve_core
from hawavoclean.enhancement.studio_pro import (
    STUDIO_PRO_CHECKPOINT_FILE,
    STUDIO_PRO_CORE_ID,
    STUDIO_PRO_MODEL_DIR_ENV,
    STUDIO_PRO_PARAMS,
    StudioProCore,
    _chunked_overlap_add,
    blend_wet_dry,
    load_mossformer2_se_48k,
    run_mossformer2_chunk,
    studio_pro_checkpoint_path,
    studio_pro_params_hash,
    verify_studio_pro_weights,
)
from hawavoclean.errors import ModelProvenanceError, PreflightError, WorkerOOMError
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.natural_contract import load_natural_route_contract
from hawavoclean.paths import models_dir, profile_config_path

MODELS_DIR = models_dir()


def test_registry_and_preferred_devices_preserve_legacy_bit_identity() -> None:
    assert resolve_core("wiener-dd-48k-v1").preferred_devices == ("cpu",)
    assert resolve_core("studio-dfn3-48k-v1").preferred_devices == ("cpu",)
    assert resolve_core("studio-dfn3-lowband-48k-v1").preferred_devices == ("cpu",)

    reg = resolve_core(STUDIO_PRO_CORE_ID)
    assert reg.enhancer_class is StudioProCore
    assert reg.lock_filename == "studio-pro-core.lock.toml"
    assert reg.device_aware is True
    assert reg.preferred_devices == ("mps", "cuda", "cpu")
    assert reg.requires_modules == ("torch", "torchaudio")
    assert (
        reg.dependency_probe
        == "hawavoclean.enhancement.dependency_probe:probe_studio_pro_runtime_contract"
    )


def test_studio_pro_lockfile_and_profile_config_are_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = MODELS_DIR / "studio-pro-core.lock.toml"
    assert lock_path.is_file()
    with open(lock_path, "rb") as f:
        lock = tomllib.load(f)

    assert lock["core_id"] == STUDIO_PRO_CORE_ID
    assert lock["model_license"] == "Apache-2.0"
    assert lock["phase_coherent"] is False
    assert lock["expected_sample_rates"] == [48000]
    assert lock["params_hash"] == studio_pro_params_hash()
    assert hash_json_canonical(dict(lock["params"])) == lock["params_hash"]

    baseline = studio_pro_params_hash()
    for key in ("model", "internal_sample_rate", "chunk_duration_s", "checkpoint_sha256"):
        mutated = dict(STUDIO_PRO_PARAMS)
        mutated[key] = "mutated" if isinstance(mutated[key], str) else 99999
        monkeypatch.setattr(studio_pro, "STUDIO_PRO_PARAMS", mutated)
        assert studio_pro_params_hash() != baseline

    monkeypatch.setattr(studio_pro, "STUDIO_PRO_PARAMS", STUDIO_PRO_PARAMS)
    cfg = load_config(profile_config_path("studio_pro"), is_production=True, activate=False)
    assert cfg.enhancement.core_id == STUDIO_PRO_CORE_ID
    assert cfg.enhancement.model_sample_rate == 48000
    assert cfg.enhancement.phase_coherent is False
    assert cfg.guard.mode == "integrity"
    assert cfg.runtime.device == "auto"


def test_resolve_device_honours_core_preferred_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda d: d in ("mps", "cuda", "cpu"))
    # Legacy cores MUST stay on CPU under auto to protect bit-identity.
    assert runtime.resolve_device("auto", core_id="wiener-dd-48k-v1").resolved == "cpu"
    assert runtime.resolve_device("auto", core_id="studio-dfn3-48k-v1").resolved == "cpu"
    assert runtime.resolve_device("auto", core_id="studio-dfn3-lowband-48k-v1").resolved == "cpu"
    assert runtime.resolve_device("auto", core_id="unregistered-core").resolved == "cpu"

    # Studio Pro automatically prefers mps -> cuda -> cpu.
    res_mps = runtime.resolve_device("auto", core_id=STUDIO_PRO_CORE_ID)
    assert res_mps.resolved == "mps"
    assert "mps" in res_mps.reason

    monkeypatch.setattr(runtime, "device_available", lambda d: d in ("cuda", "cpu"))
    assert runtime.resolve_device("auto", core_id=STUDIO_PRO_CORE_ID).resolved == "cuda"

    monkeypatch.setattr(runtime, "device_available", lambda d: d == "cpu")
    assert runtime.resolve_device("auto", core_id=STUDIO_PRO_CORE_ID).resolved == "cpu"

    # Explicit preference overrides core preferred_devices.
    monkeypatch.setattr(runtime, "device_available", lambda _d: True)
    assert (
        runtime.resolve_device("auto", core_id=STUDIO_PRO_CORE_ID, preference=("cpu",)).resolved
        == "cpu"
    )


def test_checkpoint_path_resolution_and_sha256_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Explicit HAWAVOCLEAN_STUDIO_PRO_MODEL_DIR override
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    monkeypatch.setenv(STUDIO_PRO_MODEL_DIR_ENV, str(explicit_dir))
    assert studio_pro_checkpoint_path() == explicit_dir.resolve() / STUDIO_PRO_CHECKPOINT_FILE

    ok_missing, msg_missing = verify_studio_pro_weights()
    assert not ok_missing
    assert "missing" in msg_missing
    assert str(tmp_path) not in msg_missing  # path-free contract

    ckpt = explicit_dir / STUDIO_PRO_CHECKPOINT_FILE
    ckpt.write_bytes(b"corrupted-weight-bytes")
    ok_bad, msg_bad = verify_studio_pro_weights()
    assert not ok_bad
    assert "digest mismatch" in msg_bad
    assert str(tmp_path) not in msg_bad

    real_digest = hash_file(ckpt)
    patched = dict(STUDIO_PRO_PARAMS)
    patched["checkpoint_sha256"] = real_digest
    monkeypatch.setattr(studio_pro, "STUDIO_PRO_PARAMS", patched)
    ok_good, msg_good = verify_studio_pro_weights()
    assert ok_good
    assert "verified" in msg_good

    # 2. Isolated HAWAVOCLEAN_MODEL_DIR fallback
    monkeypatch.delenv(STUDIO_PRO_MODEL_DIR_ENV, raising=False)
    iso_models = tmp_path / "isolated_models"
    iso_models.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(iso_models))
    assert (
        studio_pro_checkpoint_path()
        == iso_models.resolve() / "mossformer2_se_48k" / STUDIO_PRO_CHECKPOINT_FILE
    )

    # 3. Default search fallback when env vars are unset
    monkeypatch.delenv("HAWAVOCLEAN_MODEL_DIR", raising=False)
    fake_models = tmp_path / "pkg_models"
    fake_app = tmp_path / "app_data"
    fake_repo = tmp_path / "repo"
    monkeypatch.setattr(studio_pro, "models_dir", lambda: fake_models)
    monkeypatch.setattr(studio_pro, "app_data_root", lambda: fake_app)
    monkeypatch.setattr(studio_pro, "_REPO_ROOT", fake_repo)
    # None exist -> returns packaged candidate path
    assert (
        studio_pro_checkpoint_path()
        == fake_models / "mossformer2_se_48k" / STUDIO_PRO_CHECKPOINT_FILE
    )
    # Found in repo checkpoints cache
    repo_ckpt = fake_repo / "checkpoints" / "MossFormer2_SE_48K" / STUDIO_PRO_CHECKPOINT_FILE
    repo_ckpt.parent.mkdir(parents=True)
    repo_ckpt.write_bytes(b"ok")
    assert studio_pro_checkpoint_path() == repo_ckpt

    # .local-ai/models/mossformer2_se_48k takes precedence over repo checkpoints/
    local_ai_ckpt = (
        fake_repo / ".local-ai" / "models" / "mossformer2_se_48k" / STUDIO_PRO_CHECKPOINT_FILE
    )
    local_ai_ckpt.parent.mkdir(parents=True)
    local_ai_ckpt.write_bytes(b"ok-local-ai")
    assert studio_pro_checkpoint_path() == local_ai_ckpt


def test_blend_wet_dry_delay_alignment_and_bounds() -> None:
    sr = 48000
    rng = np.random.default_rng(42)
    n = sr // 2
    t = np.arange(n, dtype=np.float32) / sr
    # Broadband speech-like signal (formant harmonics + speech-band noise) so GCC-PHAT
    # has broadband phase energy across bins.
    orig = (
        0.25 * np.sin(2.0 * np.pi * 240.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 780.0 * t)
        + 0.10 * rng.standard_normal(n)
    ).astype(np.float32)
    # Delay enhanced by 12 samples (0.25 ms) — naive sum would comb-filter at 2 kHz.
    delayed = np.pad(orig, (12, 0), mode="constant")[:n].astype(np.float32)

    assert len(blend_wet_dry(np.zeros(0, dtype=np.float32), orig, sr, 0.5)) == 0
    assert np.array_equal(blend_wet_dry(orig, delayed, sr, 0.0), orig)
    assert np.array_equal(blend_wet_dry(orig, delayed, sr, 1.0), delayed)

    blended = blend_wet_dry(orig, delayed, sr, 0.5)
    assert blended.dtype == np.float32
    assert len(blended) == n
    # After GCC-PHAT alignment, interior matches original nearly identically.
    corr = float(np.corrcoef(orig[200:-200], blended[200:-200])[0, 1])
    assert corr > 0.999

    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="strength"):
            blend_wet_dry(orig, delayed, sr, bad)


def test_chunked_overlap_add_is_exact_partition_of_unity() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(5000).astype(np.float32)
    calls: list[int] = []

    def identity(
        chunk: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        calls.append(len(chunk))
        return chunk.copy()

    # Single-chunk path
    y_single = _chunked_overlap_add(x, identity, chunk_samples=6000, overlap_samples=500)
    assert len(calls) == 1
    assert np.allclose(y_single, x, atol=1e-6)

    # Multi-chunk raised-cosine OLA path
    calls.clear()
    y_multi = _chunked_overlap_add(x, identity, chunk_samples=1500, overlap_samples=300)
    assert len(calls) >= 4
    assert np.allclose(y_multi, x, atol=1e-6)


def test_load_mossformer2_se_48k_verifies_weights_and_state_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_path = tmp_path / STUDIO_PRO_CHECKPOINT_FILE
    with pytest.raises(ModelProvenanceError, match="missing"):
        load_mossformer2_se_48k("cpu", checkpoint_path=ckpt_path)

    ckpt_path.write_bytes(b"fake-ckpt-bytes")
    digest = hash_file(ckpt_path)
    patched = dict(STUDIO_PRO_PARAMS)
    patched["checkpoint_sha256"] = digest
    monkeypatch.setattr(studio_pro, "STUDIO_PRO_PARAMS", patched)

    class _FakeTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class _FakeNet:
        def __init__(self, keys: tuple[str, ...]) -> None:
            self._state = {k: _FakeTensor((2, 2)) for k in keys}
            self.loaded: dict[str, Any] | None = None
            self.eval_called = False
            self.device: Any = None

        def state_dict(self) -> dict[str, Any]:
            return dict(self._state)

        def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
            assert strict is True
            self.loaded = state

        def eval(self) -> _FakeNet:
            self.eval_called = True
            return self

        def to(self, dev: Any) -> _FakeNet:
            self.device = dev
            return self

    active_net: _FakeNet = _FakeNet(("direct", "prefixed", "module.stripped"))
    fake_arch_mod = types.ModuleType("hawavoclean.enhancement.mossformer2_arch")
    fake_arch_mod.MossFormer2SE48K = lambda: active_net  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hawavoclean.enhancement.mossformer2_arch", fake_arch_mod)

    fake_torch = types.ModuleType("torch")
    loaded_payload: dict[str, Any] = {}
    weights_only_seen: list[bool] = []

    def _fake_load(_p: str, map_location: str = "cpu", weights_only: bool = False) -> Any:
        assert map_location == "cpu"
        weights_only_seen.append(weights_only)
        return loaded_payload["value"]

    fake_torch.load = _fake_load  # type: ignore[attr-defined]
    fake_torch.device = lambda d: f"dev:{d}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Case 1: direct + module.-prefixed + stripped module. keys under {"model": ...}
    active_net = _FakeNet(("direct", "prefixed", "module.stripped"))
    loaded_payload["value"] = {
        "model": {
            "direct": _FakeTensor((2, 2)),
            "module.prefixed": _FakeTensor((2, 2)),
            "stripped": _FakeTensor((2, 2)),
        }
    }
    net = load_mossformer2_se_48k("cpu", checkpoint_path=ckpt_path)
    assert net is active_net
    assert weights_only_seen == [True]
    assert net.eval_called is True
    assert net.device == "dev:cpu"
    assert net.loaded is not None and set(net.loaded) == {"direct", "prefixed", "module.stripped"}

    # Case 2: missing or mismatched shape raises ModelProvenanceError
    active_net = _FakeNet(("missing_key",))
    loaded_payload["value"] = {"other": _FakeTensor((2, 2))}
    with pytest.raises(ModelProvenanceError, match="missing or mismatched"):
        load_mossformer2_se_48k("cpu", checkpoint_path=ckpt_path)


def test_run_mossformer2_chunk_with_identity_mask_preserves_length_and_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    # Provide lightweight torchaudio shim if torchaudio is absent in base CI.
    fake_ta = types.ModuleType("torchaudio")
    fake_comp = types.ModuleType("torchaudio.compliance")
    fake_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    fake_func = types.ModuleType("torchaudio.functional")

    def _fake_fbank(audio_in: Any, **_kwargs: Any) -> Any:
        n_samples = int(audio_in.shape[-1])
        win_len = int(STUDIO_PRO_PARAMS["win_len"])
        win_inc = int(STUDIO_PRO_PARAMS["win_inc"])
        n_frames = max(1, (n_samples - win_len) // win_inc + 1)
        return torch.zeros((n_frames, 60), dtype=torch.float32)

    fake_kaldi.fbank = _fake_fbank  # type: ignore[attr-defined]
    fake_func.compute_deltas = lambda x: torch.zeros_like(x)  # type: ignore[attr-defined]
    fake_comp.kaldi = fake_kaldi  # type: ignore[attr-defined]
    fake_ta.compliance = fake_comp  # type: ignore[attr-defined]
    fake_ta.functional = fake_func  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)
    monkeypatch.setitem(sys.modules, "torchaudio.compliance", fake_comp)
    monkeypatch.setitem(sys.modules, "torchaudio.compliance.kaldi", fake_kaldi)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", fake_func)

    class _IdentityMaskModel:
        def __call__(self, feats: Any) -> list[Any]:
            # feats: (1, T, 180) -> mask: (1, T, 961)
            n_frames = int(feats.shape[1])
            return [torch.ones((1, n_frames, 961), dtype=torch.float32)]

    # Empty chunk
    empty_out = run_mossformer2_chunk(_IdentityMaskModel(), "cpu", np.zeros(0, dtype=np.float32))
    assert len(empty_out) == 0

    # Short chunk (< win_len = 1920) pads safely and returns exact length
    short = np.linspace(-0.2, 0.2, 400, dtype=np.float32)
    short_out = run_mossformer2_chunk(_IdentityMaskModel(), "cpu", short)
    assert len(short_out) == 400
    assert np.all(np.isfinite(short_out))

    # Non-multiple chunk length (> win_len, (len - win_len) % win_inc != 0)
    odd = np.linspace(-0.2, 0.2, 2000, dtype=np.float32)
    odd_out = run_mossformer2_chunk(_IdentityMaskModel(), "cpu", odd)
    assert len(odd_out) == 2000
    assert np.all(np.isfinite(odd_out))

    # Standard 0.25s chunk at 48 kHz (exact multiple of win_inc)
    t = np.arange(12000, dtype=np.float32) / 48000.0
    tone = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    out = run_mossformer2_chunk(_IdentityMaskModel(), "cpu", tone)
    assert len(out) == len(tone)
    assert np.all(np.isfinite(out))
    # Identity mask preserves interior waveform cleanly.
    assert np.allclose(out[2000:-2000], tone[2000:-2000], atol=1e-3)


def test_mossformer2_pure_pytorch_arch_forward_and_state_dict() -> None:
    torch = pytest.importorskip("torch")
    from hawavoclean.enhancement.mossformer2_arch import MossFormer2SE48K

    # Exercise both padded (T=10 -> padding>0) and exact-group (T=256 -> padding==0)
    # attention paths using a fast 1-block model.
    small = MossFormer2SE48K(num_blocks=1).eval()
    with torch.no_grad():
        for seq_len in (10, 256):
            feats = torch.randn(1, seq_len, 180)
            out_list = small(feats)
            assert len(out_list) == 1
            mask = out_list[0]
            assert tuple(mask.shape) == (1, seq_len, 961)
            assert torch.all(torch.isfinite(mask))
            assert float(mask.min().item()) >= 0.0

    # Full 24-block architecture matches the 929 tensors of MossFormer2_SE_48K.
    full = MossFormer2SE48K()
    state = full.state_dict()
    assert len(state) == 929
    assert "mossformer.pos_enc.inv_freq" in state
    assert "mossformer.mdl.intra_mdl.mossformerM.layers.23.rotary_pos_emb.freqs" in state


def test_studio_pro_core_lifecycle_validation_and_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="not phase-coherent"):
        StudioProCore(phase_coherent=True)
    with pytest.raises(ValueError, match="runs at 48000 Hz"):
        StudioProCore(sample_rate=16000)

    monkeypatch.setattr(runtime, "device_available", lambda d: d == "cpu")
    core = StudioProCore(device="auto")
    assert core.device == "cpu"
    assert core.metadata.core_id == STUDIO_PRO_CORE_ID
    assert core.metadata.phase_coherent is False
    assert core.metadata.params_hash == studio_pro_params_hash()

    # Empty waveform fast path (no model load)
    empty_res = core.enhance(np.zeros(0, dtype=np.float32), 48000)
    assert empty_res.input_samples == 0
    assert empty_res.output_samples == 0

    # Warmup + enhance with mocked model & peak > 0.99 limiter + non-48 kHz resample
    fake_model = object()
    load_count = 0

    def _fake_load(_dev: str) -> object:
        nonlocal load_count
        load_count += 1
        return fake_model

    monkeypatch.setattr(studio_pro, "load_mossformer2_se_48k", _fake_load)
    monkeypatch.setattr(
        studio_pro,
        "run_mossformer2_chunk",
        lambda _m, _d, chunk: np.full_like(chunk, 1.5, dtype=np.float32),
    )
    core.warmup()
    assert load_count == 1

    # Second enhance reuses cached model, resamples 16k -> 48k -> 16k, clamps peak <= 0.99
    sig_16k = np.sin(np.linspace(0, 10 * np.pi, 1600, dtype=np.float32)).astype(np.float32)
    res = core.enhance(sig_16k, 16000)
    assert load_count == 1
    assert res.input_samples == 1600
    assert res.output_samples == 1600
    assert res.sample_rate == 16000
    assert float(np.max(np.abs(res.waveform))) <= 0.991

    # Memory budget enforcement raises WorkerOOMError before running
    monkeypatch.setattr(
        studio_pro,
        "check_memory_budget",
        MagicMock(side_effect=WorkerOOMError("budget exceeded")),
    )
    with pytest.raises(WorkerOOMError):
        core.enhance(sig_16k, 16000)


def test_probe_studio_pro_runtime_contract_fail_closed_and_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib
    from collections.abc import Iterator

    real_import = builtins.__import__

    @contextlib.contextmanager
    def _block_module(blocked: str) -> Iterator[None]:
        def _imp(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == blocked or name.startswith(f"{blocked}."):
                raise ModuleNotFoundError(f"blocked {blocked}")
            return real_import(name, *args, **kwargs)

        with monkeypatch.context() as mp:
            for key in list(sys.modules):
                if key == blocked or key.startswith(f"{blocked}."):
                    mp.setitem(sys.modules, key, None)
            mp.setattr(builtins, "__import__", _imp)
            yield

    # 1. Missing torch
    with (
        _block_module("torch"),
        pytest.raises(dependency_probe.RuntimeDependencyContractError, match="torch import failed"),
    ):
        dependency_probe.probe_studio_pro_runtime_contract()

    # 2. Missing torchaudio (supply stub torch so test works even without torch installed)
    fake_torch = types.ModuleType("torch")
    for attr in ("device", "from_numpy", "no_grad", "stft", "istft", "load"):
        setattr(fake_torch, attr, lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with (
        _block_module("torchaudio"),
        pytest.raises(
            dependency_probe.RuntimeDependencyContractError, match="torchaudio import failed"
        ),
    ):
        dependency_probe.probe_studio_pro_runtime_contract()

    # 3. Missing mossformer2_arch
    fake_ta = types.ModuleType("torchaudio")
    fake_comp = types.ModuleType("torchaudio.compliance")
    fake_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    fake_kaldi.fbank = lambda *_a, **_k: None  # type: ignore[attr-defined]
    fake_func = types.ModuleType("torchaudio.functional")
    fake_func.compute_deltas = lambda *_a, **_k: None  # type: ignore[attr-defined]
    fake_comp.kaldi = fake_kaldi  # type: ignore[attr-defined]
    fake_ta.compliance = fake_comp  # type: ignore[attr-defined]
    fake_ta.functional = fake_func  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)
    monkeypatch.setitem(sys.modules, "torchaudio.compliance", fake_comp)
    monkeypatch.setitem(sys.modules, "torchaudio.compliance.kaldi", fake_kaldi)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", fake_func)

    with (
        _block_module("hawavoclean.enhancement.mossformer2_arch"),
        pytest.raises(
            dependency_probe.RuntimeDependencyContractError,
            match="mossformer2_arch import failed",
        ),
    ):
        dependency_probe.probe_studio_pro_runtime_contract()

    # 4. Non-callable symbol on mossformer2_arch
    fake_arch = types.ModuleType("hawavoclean.enhancement.mossformer2_arch")
    fake_arch.MossFormer2SE48K = object()  # type: ignore[attr-defined]
    fake_arch.extract_mossformer2_features = lambda *_a, **_k: None  # type: ignore[attr-defined]
    fake_arch.apply_mossformer2_mask = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hawavoclean.enhancement.mossformer2_arch", fake_arch)
    with pytest.raises(
        dependency_probe.RuntimeDependencyContractError, match="missing or is not callable"
    ):
        dependency_probe.probe_studio_pro_runtime_contract()

    # 5. Callable symbols but missing weights fails closed
    fake_arch.MossFormer2SE48K = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setattr(
        studio_pro,
        "verify_studio_pro_weights",
        lambda: (
            False,
            "Studio Pro weights file missing (MossFormer2_SE_48K/last_best_checkpoint.pt)",
        ),
    )
    with pytest.raises(
        dependency_probe.RuntimeDependencyContractError, match="Studio Pro weights file missing"
    ):
        dependency_probe.probe_studio_pro_runtime_contract()

    # 6. Callable symbols + verified weights passes cleanly
    monkeypatch.setattr(
        studio_pro,
        "verify_studio_pro_weights",
        lambda: (True, "Studio Pro weights verified"),
    )
    dependency_probe.probe_studio_pro_runtime_contract()


def test_load_natural_route_contract_studio_pro_fails_closed_when_weights_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawavoclean.natural_contract import _probe_optional_runtime_contract

    # Point weights env to an empty directory so even hosts with torch/torchaudio fail closed cleanly.
    monkeypatch.setenv(STUDIO_PRO_MODEL_DIR_ENV, str(tmp_path))
    _probe_optional_runtime_contract.cache_clear()
    try:
        with pytest.raises(PreflightError):
            load_natural_route_contract("studio_pro", activate_runtime_config=False)
    finally:
        _probe_optional_runtime_contract.cache_clear()
    assert STUDIO_PRO_CORE_ID in CORE_REGISTRY


def test_dereverb_core_registration_lockfile_and_profile_config() -> None:
    from hawavoclean.enhancement.dereverb_neural import (
        DEREVERB_CHECKPOINT_SHA256,
        STUDIO_DEREVERB_CORE_ID,
        NeuralDereverbCore,
        studio_dereverb_params_hash,
    )

    reg = resolve_core(STUDIO_DEREVERB_CORE_ID)
    assert reg.enhancer_class is NeuralDereverbCore
    assert reg.lock_filename == "studio-dereverb-core.lock.toml"
    assert reg.device_aware is True
    assert reg.preferred_devices == ("mps", "cuda", "cpu")
    assert reg.requires_modules == ("torch", "torchaudio")
    assert (
        reg.dependency_probe
        == "hawavoclean.enhancement.dependency_probe:probe_dereverb_runtime_contract"
    )

    lock_path = MODELS_DIR / "studio-dereverb-core.lock.toml"
    assert lock_path.is_file()
    with open(lock_path, "rb") as f:
        lock = tomllib.load(f)

    assert lock["core_id"] == STUDIO_DEREVERB_CORE_ID
    assert lock["model_license"] == "MIT"
    assert lock["phase_coherent"] is False
    assert lock["expected_sample_rates"] == [48000]
    assert lock["params"]["enable_bwe"] is False
    assert lock["params"]["checkpoint_sha256"] == DEREVERB_CHECKPOINT_SHA256
    assert lock["params_hash"] == studio_dereverb_params_hash()
    assert hash_json_canonical(dict(lock["params"])) == lock["params_hash"]

    cfg = load_config(profile_config_path("dereverb"), is_production=True, activate=False)
    assert cfg.enhancement.core_id == STUDIO_DEREVERB_CORE_ID
    assert cfg.enhancement.model_sample_rate == 48000
    assert cfg.enhancement.phase_coherent is False
    assert cfg.guard.mode == "integrity"
    assert cfg.guard.calibration_file == "guard-calibration-dereverb.json"
    assert cfg.guard.spectral_hole_thresh == pytest.approx(0.30)
    assert cfg.guard.min_hf_preservation_ratio == pytest.approx(0.30)
    assert cfg.runtime.device == "auto"

    import json

    calib = json.loads((MODELS_DIR / "guard-calibration-dereverb.json").read_text(encoding="utf-8"))
    assert calib["calibration_id"] == hash_json_canonical(calib["thresholds"])
    assert calib["thresholds"]["spectral_hole_thresh"] == pytest.approx(0.30)
    assert calib["thresholds"]["min_hf_preservation_ratio"] == pytest.approx(0.30)


def test_probe_dereverb_runtime_contract_fail_closed_and_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib
    from collections.abc import Iterator

    real_import = builtins.__import__

    @contextlib.contextmanager
    def _block_module(blocked: str) -> Iterator[None]:
        def _imp(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == blocked or name.startswith(f"{blocked}."):
                raise ModuleNotFoundError(f"blocked {blocked}")
            return real_import(name, *args, **kwargs)

        with monkeypatch.context() as mp:
            for key in list(sys.modules):
                if key == blocked or key.startswith(f"{blocked}."):
                    mp.setitem(sys.modules, key, None)
            mp.setattr(builtins, "__import__", _imp)
            yield

    # 1. Missing torch
    with (
        _block_module("torch"),
        pytest.raises(dependency_probe.RuntimeDependencyContractError, match="torch import failed"),
    ):
        dependency_probe.probe_dereverb_runtime_contract()

    # 2. Missing torchaudio
    fake_torch = types.ModuleType("torch")
    for attr in ("device", "from_numpy", "inference_mode", "stft", "istft", "load"):
        setattr(fake_torch, attr, lambda *_a, **_k: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with (
        _block_module("torchaudio"),
        pytest.raises(
            dependency_probe.RuntimeDependencyContractError, match="torchaudio import failed"
        ),
    ):
        dependency_probe.probe_dereverb_runtime_contract()

    # 3. Missing dereverb_neural
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    with (
        _block_module("hawavoclean.enhancement.dereverb_neural"),
        pytest.raises(
            dependency_probe.RuntimeDependencyContractError, match="dereverb_neural import failed"
        ),
    ):
        dependency_probe.probe_dereverb_runtime_contract()

    # 4. Non-callable symbol on dereverb_neural
    fake_mod = types.ModuleType("hawavoclean.enhancement.dereverb_neural")
    fake_mod.NeuralDereverbCore = object()  # type: ignore[attr-defined]
    fake_mod.load_dereverb_model = lambda *_a, **_k: None  # type: ignore[attr-defined]
    fake_mod.verify_dereverb_weights = lambda: (True, "ok")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hawavoclean.enhancement.dereverb_neural", fake_mod)
    with pytest.raises(
        dependency_probe.RuntimeDependencyContractError, match="missing or is not callable"
    ):
        dependency_probe.probe_dereverb_runtime_contract()

    # 5. Missing weights fails closed
    fake_mod.NeuralDereverbCore = lambda *_a, **_k: None  # type: ignore[attr-defined]
    fake_mod.verify_dereverb_weights = lambda: (  # type: ignore[attr-defined]
        False,
        "Neural dereverb weights file missing (GAP-URGENet-Predictor/Predictor.pt)",
    )
    with pytest.raises(
        dependency_probe.RuntimeDependencyContractError,
        match="Neural dereverb weights file missing",
    ):
        dependency_probe.probe_dereverb_runtime_contract()

    # 6. Verified weights passes cleanly
    fake_mod.verify_dereverb_weights = lambda: (True, "Neural dereverb weights verified")  # type: ignore[attr-defined]
    dependency_probe.probe_dereverb_runtime_contract()


def test_load_natural_route_contract_dereverb_fails_closed_when_weights_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawavoclean.enhancement.dereverb_neural import (
        DEREVERB_MODEL_DIR_ENV,
        STUDIO_DEREVERB_CORE_ID,
    )
    from hawavoclean.natural_contract import _probe_optional_runtime_contract

    monkeypatch.setenv(DEREVERB_MODEL_DIR_ENV, str(tmp_path))
    _probe_optional_runtime_contract.cache_clear()
    try:
        with pytest.raises(PreflightError):
            load_natural_route_contract("dereverb", activate_runtime_config=False)
    finally:
        _probe_optional_runtime_contract.cache_clear()
    assert STUDIO_DEREVERB_CORE_ID in CORE_REGISTRY
