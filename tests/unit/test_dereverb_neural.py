"""Unit tests for lean pure-PyTorch neural mono dereverberation (TF-GridNetV3 + guarded BWE)."""

from __future__ import annotations

import builtins
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from hawavoclean.enhancement import dereverb_neural as dn
from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file


def _tiny_cfg() -> dict[str, Any]:
    return {
        "fft_len": 0.032,
        "hop_len": 0.016,
        "win_len": 0.032,
        "n_srcs": 1,
        "n_imics": 1,
        "n_layers": 1,
        "lstm_hidden_units": 8,
        "attn_n_head": 2,
        "attn_qk_output_channel": 2,
        "emb_dim": 8,
        "emb_ks": 4,
        "emb_hs": 1,
        "activation": "prelu",
        "eps": 1e-5,
    }


def _make_tiny_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = _tiny_cfg()
    model = dn._build_tfgridnet_v3(cfg)
    ckpt_path = tmp_path / "Predictor.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    digest = hash_file(ckpt_path)
    monkeypatch.setitem(dn.DEREVERB_PARAMS, "checkpoint_sha256", digest)
    dn._DEREVERB_MODEL_CACHE.clear()
    return ckpt_path


def _sine(
    sr: int, dur_s: float, freq: float = 440.0, amp: float = 0.25
) -> np.ndarray[Any, np.dtype[np.float32]]:
    t = np.arange(int(round(sr * dur_s)), dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_params_hash_and_license_metadata() -> None:
    h1 = dn.dereverb_params_hash()
    h2 = dn.dereverb_params_hash()
    assert h1 == h2
    assert len(h1) == 64
    assert dn.DEREVERB_MODEL_LICENSE == "MIT"
    assert dn.DEREVERB_ARCH_LICENSE == "Apache-2.0"


def test_checkpoint_path_resolution_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Explicit env var pointing to a directory
    custom_dir = tmp_path / "custom_dir"
    custom_dir.mkdir()
    monkeypatch.setenv(dn.DEREVERB_MODEL_DIR_ENV, str(custom_dir))
    assert dn.dereverb_checkpoint_path() == custom_dir.resolve() / dn.DEREVERB_CHECKPOINT_FILE

    # 2. Explicit env var pointing directly to an existing file
    direct_file = custom_dir / "custom_predictor.pt"
    direct_file.write_bytes(b"weights")
    monkeypatch.setenv(dn.DEREVERB_MODEL_DIR_ENV, str(direct_file))
    assert dn.dereverb_checkpoint_path() == direct_file.resolve()

    # 3. Exclusive HAWAVOCLEAN_MODEL_DIR isolation
    monkeypatch.delenv(dn.DEREVERB_MODEL_DIR_ENV, raising=False)
    iso_models = tmp_path / "isolated_models"
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(iso_models))
    assert (
        dn.dereverb_checkpoint_path()
        == iso_models.resolve() / "gap_urgenet_predictor" / dn.DEREVERB_CHECKPOINT_FILE
    )

    # 4. Default search order finds staged candidate or falls back to models_dir()
    monkeypatch.delenv("HAWAVOCLEAN_MODEL_DIR", raising=False)
    fake_models = tmp_path / "pkg_models"
    fake_app = tmp_path / "app_data"
    fake_local = tmp_path / "local_ai_models"
    monkeypatch.setattr(dn, "models_dir", lambda: fake_models)
    monkeypatch.setattr(dn, "app_data_root", lambda: fake_app)
    monkeypatch.setattr(dn, "_local_ai_models_dir", lambda: fake_local)
    assert (
        dn.dereverb_checkpoint_path()
        == fake_models / "gap_urgenet_predictor" / dn.DEREVERB_CHECKPOINT_FILE
    )

    staged = fake_local / "gap_urgenet_predictor" / dn.DEREVERB_CHECKPOINT_FILE
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"ok")
    assert dn.dereverb_checkpoint_path() == staged


def test_verify_weights_and_probe_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "secret_dir" / "Predictor.pt"
    ok, msg = dn.verify_dereverb_weights(missing)
    assert not ok
    assert "missing" in msg
    assert "secret_dir" not in msg

    missing.parent.mkdir(parents=True)
    missing.write_bytes(b"corrupted")
    ok2, msg2 = dn.verify_dereverb_weights(missing)
    assert not ok2
    assert "mismatch" in msg2
    assert "secret_dir" not in msg2

    ckpt = _make_tiny_checkpoint(tmp_path, monkeypatch)
    monkeypatch.setenv(dn.DEREVERB_MODEL_DIR_ENV, str(tmp_path))
    ok3, msg3 = dn.verify_dereverb_weights()
    assert ok3
    assert "verified" in msg3

    ok4, msg4 = dn.probe_dereverb_runtime(ckpt)
    assert ok4
    assert "verified" in msg4

    # Simulate missing torch without touching sys.modules permanently
    real_import = builtins.__import__

    def _fail_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_torch)
    ok5, msg5 = dn.probe_dereverb_runtime(ckpt)
    assert not ok5
    assert "torch unavailable" in msg5


def test_build_geometry_validation_and_load_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_cfg = _tiny_cfg()
    bad_cfg["emb_ks"] = 1
    bad_cfg["emb_hs"] = 1
    with pytest.raises(ValueError, match="Unsupported TFGridNetV3 geometry"):
        dn._build_tfgridnet_v3(bad_cfg)

    bad_cfg2 = _tiny_cfg()
    bad_cfg2["attn_n_head"] = 3  # 8 % 3 != 0
    with pytest.raises(ValueError, match="Unsupported TFGridNetV3 geometry"):
        dn._build_tfgridnet_v3(bad_cfg2)

    # Missing file -> ModelProvenanceError
    with pytest.raises(ModelProvenanceError, match="missing"):
        dn.load_dereverb_model("cpu", checkpoint_path=tmp_path / "nope.pt")

    # Bad payload structure -> ModelProvenanceError
    bad_payload = tmp_path / "bad_payload.pt"
    torch.save({"unexpected": 1}, bad_payload)
    monkeypatch.setitem(dn.DEREVERB_PARAMS, "checkpoint_sha256", hash_file(bad_payload))
    dn._DEREVERB_MODEL_CACHE.clear()
    with pytest.raises(ModelProvenanceError, match="missing expected model/cfg"):
        dn.load_dereverb_model("cpu", checkpoint_path=bad_payload)

    # State-dict mismatch -> ModelProvenanceError
    bad_sd = tmp_path / "bad_sd.pt"
    torch.save({"model": {"wrong.key": torch.zeros(1)}, "cfg": _tiny_cfg()}, bad_sd)
    monkeypatch.setitem(dn.DEREVERB_PARAMS, "checkpoint_sha256", hash_file(bad_sd))
    dn._DEREVERB_MODEL_CACHE.clear()
    with pytest.raises(ModelProvenanceError, match="state_dict mismatch"):
        dn.load_dereverb_model("cpu", checkpoint_path=bad_sd)


def test_load_cache_and_tiny_end_to_end_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = _make_tiny_checkpoint(tmp_path, monkeypatch)
    m1 = dn.load_dereverb_model("cpu", checkpoint_path=ckpt)
    m2 = dn.load_dereverb_model("cpu", checkpoint_path=ckpt)
    assert m1 is m2

    # Direct chunk edge cases: empty, silent, ultra-short (< 512 samples), normal
    assert len(dn._run_dereverb_chunk(m1, "cpu", np.zeros(0, dtype=np.float32))) == 0
    silent = np.zeros(1000, dtype=np.float32)
    np.testing.assert_array_equal(dn._run_dereverb_chunk(m1, "cpu", silent), silent)
    short = _sine(16000, 0.01)  # 160 samples < 512 n_fft
    short_out = dn._run_dereverb_chunk(m1, "cpu", short)
    assert len(short_out) == len(short)
    assert bool(np.all(np.isfinite(short_out)))

    # Full NeuralDereverberator warmup + multi-chunk inference + default checkpoint path
    monkeypatch.setenv(dn.DEREVERB_MODEL_DIR_ENV, str(tmp_path))
    derev = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=False,
        chunk_duration_s=0.15,
        overlap_duration_s=0.05,
    )
    assert derev.device == "cpu"
    derev.warmup()
    sig = _sine(16000, 0.35)
    res = derev.dereverb(sig, 16000)
    assert res.sample_rate == 16000
    assert res.input_samples == len(sig)
    assert res.output_samples == len(sig)
    assert not res.bwe_applied
    assert res.protected_band_err_db is None
    assert bool(np.all(np.isfinite(res.waveform)))


def test_overlap_add_and_match_length_and_strength_blend() -> None:
    sig = _sine(16000, 0.5)
    # Partition-of-unity overlap-add must be exact on identity transform
    out_multi = dn._chunked_overlap_add(
        sig, lambda c: c.copy(), chunk_samples=2000, overlap_samples=400
    )
    np.testing.assert_allclose(out_multi, sig, atol=1e-6)

    out_single = dn._chunked_overlap_add(
        sig[:1000], lambda c: c.copy(), chunk_samples=2000, overlap_samples=400
    )
    np.testing.assert_array_equal(out_single, sig[:1000])

    # _match_length pad & trim
    assert len(dn._match_length(sig[:10], 20)) == 20
    assert len(dn._match_length(sig[:30], 20)) == 20

    # _blend_strength edge cases
    with pytest.raises(ValueError, match="strength"):
        dn._blend_strength(sig, sig, 16000, -0.1)
    with pytest.raises(ValueError, match="strength"):
        dn._blend_strength(sig, sig, 16000, math.nan)
    np.testing.assert_array_equal(dn._blend_strength(sig, sig * 0.5, 16000, 0.0), sig)
    np.testing.assert_array_equal(dn._blend_strength(sig, sig * 0.5, 16000, 1.0), sig * 0.5)
    blended = dn._blend_strength(sig, sig * 0.5, 16000, 0.5)
    np.testing.assert_allclose(blended, sig * 0.75, atol=1e-4)


def test_constructor_validation_and_fast_paths() -> None:
    with pytest.raises(ValueError, match="chunking"):
        dn.NeuralDereverberator(device="cpu", chunk_duration_s=0.0)
    with pytest.raises(ValueError, match="chunking"):
        dn.NeuralDereverberator(device="cpu", chunk_duration_s=2.0, overlap_duration_s=2.0)
    with pytest.raises(ValueError, match="bwe_cutoff_hz"):
        dn.NeuralDereverberator(device="cpu", bwe_cutoff_hz=8500.0)

    derev = dn.NeuralDereverberator(device="cpu", enable_bwe=False, chunk_runner=lambda c: c)
    sig = _sine(48000, 0.1)

    with pytest.raises(ValueError, match="sample_rate"):
        derev.dereverb(sig, 0)
    with pytest.raises(ValueError, match="strength"):
        derev.dereverb(sig, 48000, strength=1.5)

    res_empty = derev.dereverb(np.zeros(0, dtype=np.float32), 48000)
    assert res_empty.output_samples == 0

    res_silent = derev.dereverb(np.zeros(4800, dtype=np.float32), 48000)
    assert res_silent.model_runtime_ms == 0.0

    res_dry = derev.dereverb(sig, 48000, strength=0.0)
    np.testing.assert_array_equal(res_dry.waveform, sig)


def test_nonfinite_revert_and_peak_limiting() -> None:
    sig = _sine(48000, 0.1)

    # Non-finite model output fails closed to original input
    derev_nan = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=False,
        chunk_runner=lambda c: np.full_like(c, np.nan),
    )
    res_nan = derev_nan.dereverb(sig, 48000)
    np.testing.assert_array_equal(res_nan.waveform, sig)
    assert not res_nan.bwe_applied

    # Hot output peak (> 0.99) is cleanly scaled to 0.99
    derev_hot = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=False,
        chunk_runner=lambda c: c * 10.0,
    )
    res_hot = derev_hot.dereverb(sig, 48000)
    assert float(np.max(np.abs(res_hot.waveform))) == pytest.approx(0.99, abs=1e-4)


def test_optional_bwe_paths_and_fail_closed_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    sig = _sine(48000, 0.15)

    # 1. BWE runtime unavailable -> skips cleanly
    monkeypatch.setattr(dn, "probe_neural_bwe_runtime", lambda: (False, "missing"))
    derev_missing = dn.NeuralDereverberator(device="cpu", enable_bwe=True, chunk_runner=lambda c: c)
    res1 = derev_missing.dereverb(sig, 48000)
    assert not res1.bwe_applied
    assert res1.protected_band_err_db is None

    # 2. BWE restorer succeeds with finite active candidate
    class _GoodRestorer:
        def render(self, audio_48k: np.ndarray[Any, Any], **_kwargs: Any) -> Any:
            return SimpleNamespace(
                success=True,
                candidates=[
                    SimpleNamespace(strength=0.0, audio=audio_48k, protected_band_error=0.0),
                    SimpleNamespace(strength=1.0, audio=audio_48k * 0.9, protected_band_error=0.01),
                ],
            )

    derev_good = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=True,
        bwe_restorer=_GoodRestorer(),  # type: ignore[arg-type]
        chunk_runner=lambda c: c,
    )
    res2 = derev_good.dereverb(sig, 48000, strength=0.5)
    assert res2.bwe_applied
    assert res2.protected_band_err_db == pytest.approx(0.01)

    # 3. BWE restorer returns non-finite active candidate -> fails closed
    class _NanRestorer:
        def render(self, audio_48k: np.ndarray[Any, Any], **_kwargs: Any) -> Any:
            return SimpleNamespace(
                success=True,
                candidates=[
                    SimpleNamespace(
                        strength=1.0,
                        audio=np.full_like(audio_48k, np.nan),
                        protected_band_error=0.0,
                    )
                ],
            )

    derev_nan_bwe = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=True,
        bwe_restorer=_NanRestorer(),  # type: ignore[arg-type]
        chunk_runner=lambda c: c,
    )
    res3 = derev_nan_bwe.dereverb(sig, 48000)
    assert not res3.bwe_applied

    # 4. BWE restorer returns success=False -> fails closed
    class _FailRestorer:
        def render(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(success=False, candidates=[])

    derev_fail_bwe = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=True,
        bwe_restorer=_FailRestorer(),  # type: ignore[arg-type]
        chunk_runner=lambda c: c,
    )
    res4 = derev_fail_bwe.dereverb(sig, 48000)
    assert not res4.bwe_applied

    # 5. BWE restorer raises -> fails closed
    class _CrashRestorer:
        def render(self, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

    derev_crash_bwe = dn.NeuralDereverberator(
        device="cpu",
        enable_bwe=True,
        bwe_restorer=_CrashRestorer(),  # type: ignore[arg-type]
        chunk_runner=lambda c: c,
    )
    res5 = derev_crash_bwe.dereverb(sig, 48000)
    assert not res5.bwe_applied

    # 6. Lazy instantiation path when probe succeeds
    monkeypatch.setattr(dn, "probe_neural_bwe_runtime", lambda: (True, "ok"))
    monkeypatch.setattr(dn, "NeuralBWERestorer", lambda **_kwargs: _GoodRestorer())
    derev_lazy = dn.NeuralDereverberator(device="cpu", enable_bwe=True, chunk_runner=lambda c: c)
    res6 = derev_lazy.dereverb(sig, 48000)
    assert res6.bwe_applied
    # Also exercise _local_ai_models_dir helper directly
    assert dn._local_ai_models_dir().name == "models"


def test_neural_dereverb_core_protocol_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="phase_coherent"):
        dn.NeuralDereverbCore(phase_coherent=True, device="cpu")
    with pytest.raises(ValueError, match="48000"):
        dn.NeuralDereverbCore(sample_rate=16000, device="cpu")

    # device=None follows active_device() (HAWAVOCLEAN_DEVICE) and defaults enable_bwe=False
    monkeypatch.setenv("HAWAVOCLEAN_DEVICE", "cpu")
    assert dn.STUDIO_DEREVERB_PARAMS["enable_bwe"] is False
    bwe_called = False

    def _fail_if_bwe_probed() -> tuple[bool, str]:
        nonlocal bwe_called
        bwe_called = True
        return True, "unexpected"

    monkeypatch.setattr(dn, "probe_neural_bwe_runtime", _fail_if_bwe_probed)
    core_default = dn.NeuralDereverbCore(
        chunk_runner=lambda c: c,
    )
    assert core_default.device == "cpu"
    assert core_default.metadata.core_id == dn.STUDIO_DEREVERB_CORE_ID
    assert core_default.metadata.sample_rate == 48000
    assert core_default.metadata.phase_coherent is False
    assert core_default.metadata.params_hash == dn.studio_dereverb_params_hash()
    _ = core_default.enhance(_sine(48000, 0.1), 48000)
    assert bwe_called is False

    # Explicit device + end-to-end tiny checkpoint warmup() + enhance()
    ckpt = _make_tiny_checkpoint(tmp_path, monkeypatch)
    budget_calls: list[str] = []
    monkeypatch.setattr(
        dn, "check_memory_budget", lambda ctx="enhancement": budget_calls.append(ctx)
    )
    core = dn.NeuralDereverbCore(
        device="cpu",
        enable_bwe=False,
        checkpoint_path=ckpt,
    )
    assert core.device == "cpu"
    core.warmup()
    sig = _sine(48000, 0.1)
    res = core.enhance(sig, 48000)
    assert res.sample_rate == 48000
    assert res.input_samples == len(sig)
    assert res.output_samples == len(sig)
    assert bool(np.all(np.isfinite(res.waveform)))
    assert "dereverb" in budget_calls
