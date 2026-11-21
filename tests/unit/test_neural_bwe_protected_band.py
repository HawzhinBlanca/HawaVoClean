"""Unit tests for NeuralBWERestorer (LavaSR v2) strict protected-band invariance."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import signal

from hawavoclean import runtime
from hawavoclean.errors import ModelProvenanceError, WorkerOOMError
from hawavoclean.hashing import hash_file
from hawavoclean.restoration import neural_bwe
from hawavoclean.restoration.base import Restorer
from hawavoclean.restoration.neural_bwe import (
    LAVASR_MODEL_DIR_ENV,
    LAVASR_MODEL_NAME,
    LAVASR_V2_CONFIG_FILE,
    LAVASR_V2_WEIGHTS_FILE,
    NeuralBWERestorer,
    _load_lavasr_v2,
    _run_lavasr_v2_chunk,
    lavasr_model_dir,
    probe_neural_bwe_runtime,
    verify_lavasr_weights,
)
from hawavoclean.restoration.protected_band import verify_protected_band_invariance


def _make_bandlimited_speech_like(
    sr: int = 48000,
    duration_s: float = 0.6,
    cutoff_hz: float = 3600.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float64) / sr
    # Multi-harmonic low/mid-band speech-like complex tone (200 Hz .. 2800 Hz).
    sig = (
        0.35 * np.sin(2.0 * np.pi * 200.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 600.0 * t + 0.3)
        + 0.18 * np.sin(2.0 * np.pi * 1200.0 * t + 0.7)
        + 0.12 * np.sin(2.0 * np.pi * 2400.0 * t + 1.1)
    )
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.0 * t) ** 2
    sos = signal.butter(8, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return np.asarray(signal.sosfiltfilt(sos, sig * env), dtype=np.float32)


def _synthetic_highband_generator(
    chunk: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Synthesize plausible upper-band harmonics (5-12 kHz) plus a distorted low band.

    Intentionally corrupts the low band (x0.2 + polarity flip) to PROVE that
    NeuralBWERestorer's protected-band merge discards the generator's low band
    and mathematically preserves 0..cutoff_hz from the observed signal!
    """
    t = np.arange(len(chunk), dtype=np.float64) / sr
    hf = (
        0.06 * np.sin(2.0 * np.pi * 5500.0 * t)
        + 0.04 * np.sin(2.0 * np.pi * 8500.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 11500.0 * t)
    )
    corrupted_low = -0.2 * chunk.astype(np.float64)
    return np.asarray(corrupted_low + hf, dtype=np.float32)


def test_constructor_validation_and_protocol_conformance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "device_available", lambda d: d in ("mps", "cpu"))
    restorer = NeuralBWERestorer(device="auto")
    assert isinstance(restorer, Restorer)
    assert restorer.device == "mps"

    with pytest.raises(ValueError, match="48000 Hz"):
        NeuralBWERestorer(sample_rate=16000)
    with pytest.raises(ValueError, match="max_third_octave_deviation_db"):
        NeuralBWERestorer(max_third_octave_deviation_db=0.0)
    with pytest.raises(ValueError, match="max_third_octave_deviation_db"):
        NeuralBWERestorer(max_third_octave_deviation_db=7.5)


def test_protected_band_invariance_holds_even_with_corrupted_generator_lowband() -> None:
    sr = 48000
    obs = _make_bandlimited_speech_like(sr=sr, duration_s=0.6, cutoff_hz=3600.0)
    restorer = NeuralBWERestorer(
        device="cpu",
        generator=_synthetic_highband_generator,
    )

    res = restorer.render(obs, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.5, 0.0])
    assert res.success is True
    assert res.fallback_status == "none"
    assert res.model_name == LAVASR_MODEL_NAME
    assert res.provider == "cpu"
    assert [c.strength for c in res.candidates] == [1.0, 0.5, 0.0]

    # Full-strength candidate must add real high-band energy (> 4.5 kHz) while keeping
    # protected-band third-octave deviation near 0 dB (strictly <= 6.0 dB).
    full_cand = res.candidates[0]
    assert full_cand.protected_band_error <= 0.5
    ver = verify_protected_band_invariance(
        obs,
        full_cand.audio,
        sample_rate=sr,
        cutoff_hz=3600.0,
        tolerance_rms=0.05,
        tolerance_stft=0.10,
        max_third_octave_deviation_db=6.0,
    )
    assert ver.passes_invariance is True
    assert ver.worst_band_energy_deviation_db <= 0.5

    sos_hp = signal.butter(6, 4500.0, btype="highpass", fs=sr, output="sos")
    obs_hf = float(np.sqrt(np.mean(signal.sosfiltfilt(sos_hp, obs) ** 2)))
    rest_hf = float(np.sqrt(np.mean(signal.sosfiltfilt(sos_hp, full_cand.audio) ** 2)))
    assert rest_hf > obs_hf * 5.0

    # restore() delegates directly to render().candidates
    cands = restorer.restore(obs, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert [c.strength for c in cands] == [1.0, 0.0]


def test_stereo_chunked_and_non_48k_resampling_preserve_shape_and_invariance() -> None:
    sr = 48000
    mono = _make_bandlimited_speech_like(sr=sr, duration_s=0.7, cutoff_hz=3600.0)
    stereo = np.stack([mono, mono * 0.8], axis=0).astype(np.float32)

    # Small chunk_duration_s forces multi-chunk Hann OLA stitching.
    restorer = NeuralBWERestorer(
        device="cpu",
        chunk_duration_s=0.25,
        overlap_duration_s=0.05,
        generator=_synthetic_highband_generator,
    )
    res_stereo = restorer.render(stereo, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert res_stereo.success is True
    assert res_stereo.candidates[0].audio.shape == stereo.shape
    assert res_stereo.candidates[0].protected_band_error <= 0.5

    # Non-48 kHz mono and stereo inputs resample cleanly to 48k and back to exact length.
    mono_24k = signal.resample_poly(mono, 1, 2).astype(np.float32)
    res_m24 = restorer.render(mono_24k, 24000, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert res_m24.success is True
    assert res_m24.candidates[0].audio.shape == mono_24k.shape

    stereo_24k = np.stack([mono_24k, mono_24k], axis=0).astype(np.float32)
    res_s24 = restorer.render(stereo_24k, 24000, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert res_s24.success is True
    assert res_s24.candidates[0].audio.shape == stereo_24k.shape


def test_fail_closed_revert_on_invariance_violation_nan_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sr = 48000
    obs = _make_bandlimited_speech_like(sr=sr, duration_s=0.5, cutoff_hz=3600.0)

    # 1. Non-finite generator output -> fails closed to 0.0 original audio
    nan_restorer = NeuralBWERestorer(
        device="cpu",
        generator=lambda chunk, _sr: np.full_like(chunk, np.nan, dtype=np.float32),
    )
    res_nan = nan_restorer.render(obs, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.5, 0.0])
    assert res_nan.success is False
    assert res_nan.fallback_status == "invariance_violation"
    assert [c.strength for c in res_nan.candidates] == [0.0]
    assert np.array_equal(res_nan.candidates[0].audio, obs)

    # 2. Sub-band mutation > 6 dB inside protected band -> fails closed revert
    tampered_restorer = NeuralBWERestorer(
        device="cpu",
        generator=_synthetic_highband_generator,
    )
    sos_notch = signal.butter(6, [900.0, 1500.0], btype="bandstop", fs=sr, output="sos")

    def _notched_merge(
        observed: np.ndarray[Any, np.dtype[np.float32]],
        _gen: np.ndarray[Any, np.dtype[np.float32]],
        _cut: float,
        strength: float = 1.0,  # noqa: ARG001
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.asarray(signal.sosfiltfilt(sos_notch, observed), dtype=np.float32)

    monkeypatch.setattr(tampered_restorer, "_merge_channel_protected", _notched_merge)
    res_viol = tampered_restorer.render(obs, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert res_viol.success is False
    assert res_viol.fallback_status == "invariance_violation"
    assert res_viol.error_message is not None and "invariance" in res_viol.error_message
    assert [c.strength for c in res_viol.candidates] == [0.0]
    assert np.array_equal(res_viol.candidates[0].audio, obs)

    # 3. Runtime exception in generator -> fail-closed runtime_error with 0.0 fallback
    err_restorer = NeuralBWERestorer(
        device="cpu",
        generator=MagicMock(side_effect=RuntimeError("gpu boom")),
    )
    res_err = err_restorer.render(obs, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert res_err.success is False
    assert res_err.fallback_status == "runtime_error"
    assert res_err.error_message is not None and "gpu boom" in res_err.error_message
    assert [c.strength for c in res_err.candidates] == [0.0]

    # 4. OOM budget violation -> fail-closed runtime_error
    oom_restorer = NeuralBWERestorer(device="cpu", generator=_synthetic_highband_generator)
    monkeypatch.setattr(
        neural_bwe, "check_memory_budget", MagicMock(side_effect=WorkerOOMError("oom"))
    )
    res_oom = oom_restorer.render(obs, sr, effective_cutoff_hz=3600.0, strengths=[1.0, 0.0])
    assert res_oom.success is False
    assert res_oom.fallback_status == "runtime_error"
    assert [c.strength for c in res_oom.candidates] == [0.0]


def test_edge_inputs_empty_short_and_zero_strength() -> None:
    restorer = NeuralBWERestorer(device="cpu", generator=_synthetic_highband_generator)

    # Empty input
    res_empty = restorer.render(np.zeros(0, dtype=np.float32), 48000, effective_cutoff_hz=3600.0)
    assert res_empty.success is False
    assert res_empty.fallback_status == "empty_input"

    # Too short (< n_fft = 2048)
    short = np.zeros(512, dtype=np.float32)
    res_short = restorer.render(short, 48000, effective_cutoff_hz=3600.0)
    assert res_short.success is False
    assert res_short.fallback_status == "input_too_short"
    assert [c.strength for c in res_short.candidates] == [0.0]

    # No active strengths (only 0.0)
    valid = _make_bandlimited_speech_like(duration_s=0.2)
    res_zero = restorer.render(valid, 48000, effective_cutoff_hz=3600.0, strengths=[0.0])
    assert res_zero.success is False
    assert res_zero.fallback_status == "no_active_strengths"
    assert [c.strength for c in res_zero.candidates] == [0.0]


def test_lavasr_model_dir_weight_verification_and_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Explicit env dir (direct and nested enhancer_v2)
    direct = tmp_path / "lava_direct"
    direct.mkdir()
    monkeypatch.setenv(LAVASR_MODEL_DIR_ENV, str(direct))
    assert lavasr_model_dir() == direct.resolve()

    ok_missing, msg_missing = verify_lavasr_weights()
    assert not ok_missing
    assert "missing" in msg_missing
    assert str(tmp_path) not in msg_missing

    nested = direct / "enhancer_v2"
    nested.mkdir()
    cfg_file = nested / LAVASR_V2_CONFIG_FILE
    wt_file = nested / LAVASR_V2_WEIGHTS_FILE
    cfg_file.write_text("yaml: ok\n", encoding="utf-8")
    wt_file.write_bytes(b"weights-payload")
    assert lavasr_model_dir() == nested.resolve()

    # Mismatched SHA-256 fails closed
    ok_bad, msg_bad = verify_lavasr_weights()
    assert not ok_bad
    assert "digest mismatch" in msg_bad
    assert str(tmp_path) not in msg_bad

    # Matching SHA-256 passes
    monkeypatch.setattr(neural_bwe, "LAVASR_V2_CONFIG_SHA256", hash_file(cfg_file))
    monkeypatch.setattr(neural_bwe, "LAVASR_V2_WEIGHTS_SHA256", hash_file(wt_file))
    ok_good, msg_good = verify_lavasr_weights()
    assert ok_good
    assert "verified" in msg_good

    # 2. HAWAVOCLEAN_MODEL_DIR isolation
    monkeypatch.delenv(LAVASR_MODEL_DIR_ENV, raising=False)
    iso = tmp_path / "iso_models"
    iso.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(iso))
    assert lavasr_model_dir() == iso.resolve() / "lavasr" / "enhancer_v2"

    # 3. Default scan including offline HF snapshot cache
    monkeypatch.delenv("HAWAVOCLEAN_MODEL_DIR", raising=False)
    fake_home = tmp_path / "home"
    hf_v2 = (
        fake_home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--YatharthS--LavaSR"
        / "snapshots"
        / "abc123"
        / "enhancer_v2"
    )
    hf_v2.mkdir(parents=True)
    monkeypatch.setattr(neural_bwe, "models_dir", lambda: tmp_path / "empty_pkg")
    monkeypatch.setattr(neural_bwe, "app_data_root", lambda: tmp_path / "empty_app")
    monkeypatch.setattr(neural_bwe, "_local_ai_models_dir", lambda: tmp_path / "empty_local")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: fake_home))
    # Snapshot exists without weights file -> returns default packaged candidate
    assert lavasr_model_dir() == tmp_path / "empty_pkg" / "lavasr" / "enhancer_v2"
    (hf_v2 / LAVASR_V2_WEIGHTS_FILE).write_bytes(b"x")
    assert lavasr_model_dir() == hf_v2


def test_probe_load_and_run_native_lavasr_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib
    from collections.abc import Iterator

    torch = pytest.importorskip("torch")
    neural_bwe._LAVASR_MODEL_CACHE.clear()
    real_import = builtins.__import__

    @contextlib.contextmanager
    def _block(blocked: str) -> Iterator[None]:
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

    # 1. Missing torch fails closed cleanly
    with _block("torch"):
        ok, msg = probe_neural_bwe_runtime(tmp_path)
    assert not ok and "torch unavailable" in msg

    # 2. Build a tiny genuine native Vocos BWE checkpoint (padding="same")
    tiny_hparams: dict[str, int | str] = {
        "n_fft": 256,
        "hop_length": 64,
        "n_mels": 16,
        "padding": "same",
        "input_channels": 16,
        "dim": 32,
        "intermediate_dim": 64,
        "num_layers": 2,
    }
    torch.manual_seed(0)
    tiny_model, tiny_lr = neural_bwe._build_native_lavasr_v2(tiny_hparams, "cpu")
    v2 = tmp_path / "v2"
    v2.mkdir()
    cfg = v2 / LAVASR_V2_CONFIG_FILE
    wt = v2 / LAVASR_V2_WEIGHTS_FILE
    cfg.write_text(
        "\n".join(
            [
                "feature_extractor:",
                "  init_args:",
                "    n_fft: 256",
                "    hop_length: 64",
                "    n_mels: 16",
                '    padding: "same"',
                "backbone:",
                "  init_args:",
                "    input_channels: 16",
                "    dim: 32",
                "    intermediate_dim: 64",
                "    num_layers: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(tiny_model.state_dict(), str(wt))

    # Before pinning hash, load must refuse with ModelProvenanceError
    with pytest.raises(ModelProvenanceError, match="digest mismatch"):
        _load_lavasr_v2("cpu", model_dir=v2)

    monkeypatch.setattr(neural_bwe, "LAVASR_V2_CONFIG_SHA256", hash_file(cfg))
    monkeypatch.setattr(neural_bwe, "LAVASR_V2_WEIGHTS_SHA256", hash_file(wt))

    ok, msg = probe_neural_bwe_runtime(v2)
    assert ok and "verified" in msg

    bwe_model, lr_refiner = _load_lavasr_v2("cpu", model_dir=v2)
    # Second load hits model cache
    bwe_model_cached, _ = _load_lavasr_v2("cpu", model_dir=v2)
    assert bwe_model_cached is bwe_model

    # Run _run_lavasr_v2_chunk on empty and normal audio
    assert (
        len(_run_lavasr_v2_chunk(bwe_model, lr_refiner, "cpu", np.zeros(0, dtype=np.float32))) == 0
    )
    obs = _make_bandlimited_speech_like(duration_s=0.15)
    out = _run_lavasr_v2_chunk(bwe_model, lr_refiner, "cpu", obs)
    assert len(out) == len(obs)
    assert np.all(np.isfinite(out))

    # Short tensor (len <= pad=96, len + 2*pad >= n_fft=256) hits replicate-pad guard;
    # FastLRMerge mask cache hit + eviction > 16 entries
    short_t = torch.zeros((1, 80), dtype=torch.float32)
    assert tiny_model.feature_extractor(short_t).shape[1] == 16
    for i in range(18):
        _ = tiny_lr._get_mask(128 + i, 2)
    assert len(tiny_lr.mask_cache) <= 16
    _ = tiny_lr._get_mask(128 + 17, 2)

    # Also exercise padding="center" branch and empty config fallback
    center_hparams = dict(tiny_hparams)
    center_hparams["padding"] = "center"
    center_model, center_lr = neural_bwe._build_native_lavasr_v2(center_hparams, "cpu")
    center_out = _run_lavasr_v2_chunk(center_model.eval(), center_lr, "cpu", obs)
    assert len(center_out) == len(obs) and np.all(np.isfinite(center_out))
    empty_cfg = tmp_path / "empty_cfg.yaml"
    empty_cfg.write_text("# defaults only\n", encoding="utf-8")
    assert neural_bwe._parse_lavasr_v2_hparams(empty_cfg)["n_fft"] == 2048

    # Exercise NeuralBWERestorer default backend path (generator=None) end-to-end
    restorer = NeuralBWERestorer(device="cpu", model_dir=v2)
    res = restorer.render(
        _make_bandlimited_speech_like(duration_s=0.2),
        48000,
        effective_cutoff_hz=3600.0,
        strengths=[1.0, 0.0],
    )
    assert res.success is True
    assert res.candidates[0].protected_band_error <= 0.5
    neural_bwe._LAVASR_MODEL_CACHE.clear()


def test_smart_safe_preview_restore_source_wires_neural_bwe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawavoclean.restoration.base import RestorationCandidate, RestorationRenderResult
    from hawavoclean.smart_safe import preview as preview_mod
    from hawavoclean.smart_safe.preview import SmartSafePreviewEngine

    sr = 48000
    obs = _make_bandlimited_speech_like(sr=sr, duration_s=0.3, cutoff_hz=3600.0)
    engine = SmartSafePreviewEngine(sample_rate=sr, cutoff_hz=3600.0, allow_research_restore=True)

    # 1. When probe_neural_bwe_runtime()[0] is True, restore_source uses NeuralBWERestorer
    fake_out = _synthetic_highband_generator(obs, sr)
    calls: list[str] = []

    class _OkNeuralBWE:
        def __init__(self, device: str = "auto") -> None:
            self.device = device

        def render(
            self,
            audio_48k: np.ndarray[Any, Any],
            sample_rate: int,
            effective_cutoff_hz: float,
            strengths: list[float] | None = None,
            **_kwargs: Any,
        ) -> RestorationRenderResult:
            _ = (audio_48k, sample_rate, strengths)
            calls.append("neural_bwe")
            return RestorationRenderResult(
                success=True,
                fallback_status="none",
                model_name=LAVASR_MODEL_NAME,
                provider="cpu",
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(
                        strength=1.0,
                        audio=fake_out,
                        cutoff_hz=effective_cutoff_hz,
                        protected_band_error=0.0,
                    )
                ],
            )

    monkeypatch.setattr(preview_mod, "probe_neural_bwe_runtime", lambda: (True, "verified"))
    monkeypatch.setattr(preview_mod, "NeuralBWERestorer", _OkNeuralBWE)
    rendered, err = engine.render_route_audio("restore_source", obs, sr)
    assert err is None
    assert calls == ["neural_bwe"]
    assert np.array_equal(rendered, fake_out)

    # 2. Fail-closed when NeuralBWERestorer reports invariance failure
    class _FailNeuralBWE(_OkNeuralBWE):
        def render(
            self,
            audio_48k: np.ndarray[Any, Any],
            sample_rate: int,
            effective_cutoff_hz: float,
            strengths: list[float] | None = None,
            **_kwargs: Any,
        ) -> RestorationRenderResult:
            _ = (audio_48k, sample_rate, strengths)
            return RestorationRenderResult(
                success=False,
                fallback_status="invariance_violation",
                model_name=LAVASR_MODEL_NAME,
                provider="cpu",
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(
                        strength=0.0, audio=obs.copy(), cutoff_hz=effective_cutoff_hz
                    )
                ],
            )

    monkeypatch.setattr(preview_mod, "NeuralBWERestorer", _FailNeuralBWE)
    reverted, fail_err = engine.render_route_audio("restore_source", obs, sr)
    assert fail_err is not None and "invariance_violation" in fail_err
    assert np.array_equal(reverted, obs)

    # 3. When probe_neural_bwe_runtime()[0] is False, gracefully falls back to HawaRestoreKD
    monkeypatch.setattr(preview_mod, "probe_neural_bwe_runtime", lambda: (False, "missing"))
    fb_wave, fb_err = engine.render_route_audio("restore_source", obs, sr)
    assert fb_err is None
    assert fb_wave.shape == obs.shape
