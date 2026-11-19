"""Unit tests for Phase 3 Perceptual MOS Oracle, Active-Speech Phonetic Guard, and Truthful Smart Safe."""

from __future__ import annotations

import hashlib
import importlib
import math
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from hawavoclean.enhancement.factory import CORE_REGISTRY, CoreRegistration
from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.guard.perceptual_mos import (
    PerceptualQualityScore,
    PerceptualQualityScorer,
    perceptual_scorer_status,
)
from hawavoclean.guard.phonetic_guard import (
    _WAVLM_MODEL_CACHE,
    PhoneticStabilityGuard,
    PhoneticStabilityResult,
    phonetic_guard_status,
)
from hawavoclean.smart_safe.decision import (
    AcousticEvidence,
    CandidateEvidence,
    SmartSafePolicy,
    SmartSafeRanker,
    decide_smart_safe,
    eligible_routes,
)
from hawavoclean.smart_safe.preview import (
    STUDIO_DEREVERB_CORE_ID,
    STUDIO_PRO_CORE_ID,
    SmartSafePreviewEngine,
    _crossfade_concat,
    _speaker_similarity_passes,
    extract_preview_slice,
    passes_perceptual_artifact_check,
    verify_candidate_evidence_integrity,
    verify_post_master_invariants,
)

pytestmark = pytest.mark.unit


def _make_speech(duration_s: float = 2.0, sr: int = 48000, seed: int = 11) -> np.ndarray:
    """Deterministic speech-like modulated multi-formant voiced signal."""
    n = int(sr * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float32)
    # Syllabic bursts with clean inter-syllable pauses
    syllable_env = np.clip(np.sin(2.0 * np.pi * 3.0 * t) - 0.15, 0.0, 1.0) ** 1.2
    f0 = 155.0 + 18.0 * np.sin(2.0 * np.pi * 1.1 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    voiced = (
        0.32 * np.sin(phase)
        + 0.24 * np.sin(2.0 * phase)
        + 0.18 * np.sin(3.0 * phase)
        + 0.12 * np.sin(5.0 * phase)
        + 0.09 * np.sin(9.0 * phase)
        + 0.06 * np.sin(26.0 * phase)  # ~4 kHz upper formant / fricative presence
    )
    rng = np.random.default_rng(seed)
    fric_env = np.clip(np.sin(2.0 * np.pi * 3.0 * t + 1.2) - 0.65, 0.0, 1.0)
    fric = signal.sosfilt(
        signal.butter(4, [3500.0, 7500.0], btype="bandpass", fs=sr, output="sos"),
        rng.standard_normal(n),
    )
    wave = (syllable_env * voiced + 0.06 * fric_env * fric).astype(np.float32)
    rms = float(np.sqrt(np.mean(wave**2) + 1e-12))
    return np.ascontiguousarray(wave * (0.08 / rms), dtype=np.float32)


def _make_rir(sr: int, rt60: float = 0.65, drr_db: float = 3.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * min(1.2, rt60 * 1.4))
    t = np.arange(n, dtype=np.float64) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    tail = signal.sosfilt(signal.butter(2, 5500.0, btype="lowpass", fs=sr, output="sos"), tail)
    tail[: int(sr * 0.010)] = 0.0
    reverb = tail / (float(np.sqrt(np.sum(tail**2))) + 1e-12) * math.sqrt(10.0 ** (-drr_db / 10.0))
    rir = reverb.copy()
    rir[0] = 1.0
    out: np.ndarray = (rir / float(np.sqrt(np.sum(rir**2)) + 1e-12)).astype(np.float32)
    return out


def _evidence(**overrides: object) -> AcousticEvidence:
    base: dict[str, object] = {
        "speech_dominance": 0.94,
        "music_risk": 0.03,
        "crosstalk_risk": 0.02,
        "rumble_confidence": 0.82,
        "band_limited_confidence": 0.90,
        "recorded_high_frequency_speech_confidence": 0.02,
        "speaker_match_confidence": 0.95,
        "speaker_match_verified": True,
        "reconstruction_consent": True,
    }
    base.update(overrides)
    return AcousticEvidence(**base)  # type: ignore[arg-type]


def _ranker(*, qualified: bool = True) -> SmartSafeRanker:
    return SmartSafeRanker(
        version="ranker-perceptual-test-v1",
        artifact_sha256=hashlib.sha256(b"ranker-perceptual-test-v1").hexdigest(),
        signed=qualified,
        qualified=qualified,
    )


class _FakeTensor:
    """Minimal torch.Tensor stand-in for deterministic neural backend unit tests."""

    def __init__(self, data: np.ndarray) -> None:
        self._arr = np.asarray(data, dtype=np.float32)
        self.shape = self._arr.shape

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def to(self, _device: object) -> _FakeTensor:
        return self

    def unsqueeze(self, axis: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self._arr, axis=axis))

    def squeeze(self, axis: int) -> _FakeTensor:
        return _FakeTensor(np.squeeze(self._arr, axis=axis))

    def numpy(self) -> np.ndarray:
        return self._arr

    def tolist(self) -> list[float]:
        return [float(x) for x in self._arr.tolist()]

    def __getitem__(self, item: slice | int) -> _FakeTensor:
        return _FakeTensor(self._arr[item])


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def _make_fake_torch(*, mps: bool = False, cuda: bool = False) -> types.SimpleNamespace:
    def _cos_sim(a: _FakeTensor, b: _FakeTensor, dim: int = -1) -> _FakeTensor:
        arr_a = a.numpy()
        arr_b = b.numpy()
        num = np.sum(arr_a * arr_b, axis=dim)
        den = np.linalg.norm(arr_a, axis=dim) * np.linalg.norm(arr_b, axis=dim) + 1e-8
        return _FakeTensor(num / den)

    return types.SimpleNamespace(
        no_grad=lambda: _NoGrad(),
        from_numpy=lambda arr: _FakeTensor(arr),
        device=lambda name: str(name),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        nn=types.SimpleNamespace(functional=types.SimpleNamespace(cosine_similarity=_cos_sim)),
    )


# ---------------------------------------------------------------------------
# 1. PerceptualQualityScore & PerceptualQualityScorer tests
# ---------------------------------------------------------------------------


def test_perceptual_quality_score_validation() -> None:
    valid = PerceptualQualityScore(
        mos=4.2,
        noisiness=4.1,
        discontinuity=4.4,
        coloration=4.0,
        loudness=4.3,
        confidence=0.88,
        backend="nisqa_v2",
        calibrated=True,
    )
    assert valid.mos == 4.2
    assert valid.calibrated is True

    with pytest.raises(ValueError, match="between 1.0 and 5.0"):
        PerceptualQualityScore(0.5, 4.0, 4.0, 4.0, 4.0, 0.8, "nisqa_v2", True)
    with pytest.raises(ValueError, match="between 1.0 and 5.0"):
        PerceptualQualityScore(4.0, float("nan"), 4.0, 4.0, 4.0, 0.8, "nisqa_v2", True)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        PerceptualQualityScore(4.0, 4.0, 4.0, 4.0, 4.0, 1.5, "nisqa_v2", True)
    with pytest.raises(ValueError, match="backend must be non-empty"):
        PerceptualQualityScore(4.0, 4.0, 4.0, 4.0, 4.0, 0.8, "", False)


def test_perceptual_scorer_status_reports_honest_maturity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. Neither neural backend available -> acoustic_proxy, calibrated=False
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: False)
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_dnsmos_available", lambda: False)
    status_proxy = perceptual_scorer_status()
    assert status_proxy["available"] is True
    assert status_proxy["neural_available"] is False
    assert status_proxy["backend"] == "acoustic_proxy"
    assert status_proxy["calibrated"] is False
    assert status_proxy["qualified_ranker"] is False
    assert status_proxy["maturity"] == "experimental"

    # 2. DNSMOS only
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_dnsmos_available", lambda: True)
    status_dns = perceptual_scorer_status()
    assert status_dns["backend"] == "dnsmos"
    assert status_dns["neural_available"] is True
    assert status_dns["calibrated"] is True
    assert status_dns["qualified_ranker"] is False
    assert status_dns["maturity"] == "experimental"

    # 3. NISQA primary
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: True)
    status_nisqa = perceptual_scorer_status()
    assert status_nisqa["backend"] == "nisqa_v2"
    assert status_nisqa["nisqa_available"] is True
    assert status_nisqa["calibrated"] is True
    assert status_nisqa["qualified_ranker"] is False
    assert status_nisqa["maturity"] == "experimental"

    # 4. Env overrides resolve weight paths cleanly without touching default home dir
    monkeypatch.undo()
    from hawavoclean.guard import nisqa_arch

    fake_weights = tmp_path / "nisqa.tar"
    fake_weights.write_bytes(b"stub")
    dns_dir = tmp_path / "dnsmos"
    (dns_dir / "DNSMOS").mkdir(parents=True)
    (dns_dir / "DNSMOS" / "model_v8.onnx").write_bytes(b"stub")
    (dns_dir / "DNSMOS" / "sig_bak_ovr.onnx").write_bytes(b"stub")
    monkeypatch.setenv("HAWAVOCLEAN_NISQA_WEIGHTS", str(fake_weights))
    monkeypatch.setenv("HAWAVOCLEAN_DNSMOS_DIR", str(dns_dir))

    # Native SHA-256 check fails closed on stub bytes when torchmetrics is absent
    monkeypatch.setattr(
        "hawavoclean.guard.perceptual_mos._modules_present",
        lambda *names: names == ("torch",),
    )
    assert perceptual_scorer_status()["nisqa_available"] is False

    # Pinning matching SHA-256 activates native NISQA v2 without torchmetrics
    monkeypatch.setattr(
        nisqa_arch,
        "NISQA_V2_WEIGHTS_SHA256",
        hashlib.sha256(b"stub").hexdigest(),
    )
    status_native = perceptual_scorer_status()
    assert status_native["nisqa_available"] is True
    assert status_native["backend"] == "nisqa_v2"

    # When all optional modules are present, DNSMOS is also detected
    monkeypatch.setattr(
        "hawavoclean.guard.perceptual_mos._modules_present",
        lambda *_names: True,
    )
    status_env = perceptual_scorer_status()
    assert status_env["nisqa_available"] is True
    assert status_env["dnsmos_available"] is True


def test_perceptual_scorer_acoustic_proxy_separates_degradations() -> None:
    sr = 48000
    clean = _make_speech(duration_s=2.0, sr=sr)
    rng = np.random.default_rng(19)
    noisy = (clean + rng.standard_normal(len(clean)).astype(np.float32) * 0.14).astype(np.float32)
    clipped = np.clip(clean * 6.0, -0.20, 0.20).astype(np.float32)
    muffled = signal.sosfiltfilt(
        signal.butter(8, 2200.0, btype="lowpass", fs=sr, output="sos"), clean
    ).astype(np.float32)

    scorer = PerceptualQualityScorer(backend="acoustic_proxy")
    assert scorer.active_backend == "acoustic_proxy"

    s_clean = scorer.score(clean, sr)
    s_noisy = scorer.score(noisy, sr)
    s_clipped = scorer.score(clipped, sr)
    s_muffled = scorer.score(muffled, sr)

    assert s_clean.backend == "acoustic_proxy"
    assert s_clean.calibrated is False
    assert s_clean.mos > s_noisy.mos
    assert s_clean.noisiness > s_noisy.noisiness
    assert s_clean.discontinuity > s_clipped.discontinuity
    assert s_clean.coloration > s_muffled.coloration
    assert 0.56 <= s_clean.confidence <= 0.76


def test_perceptual_scorer_edge_inputs_and_validations(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown perceptual scorer backend"):
        PerceptualQualityScorer(backend="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_rms"):
        PerceptualQualityScorer(target_rms=0.0)

    scorer = PerceptualQualityScorer(backend="acoustic_proxy")
    with pytest.raises(ValueError, match="sample_rate"):
        scorer.score(np.zeros(100, dtype=np.float32), 0)

    # Empty audio
    empty_s = scorer.score(np.zeros(0, dtype=np.float32), 48000)
    assert empty_s.mos == 3.0 and empty_s.calibrated is False

    # Non-finite audio fails closed with minimum MOS
    nan_arr = np.array([0.1, float("nan"), 0.2], dtype=np.float32)
    nan_s = scorer.score(nan_arr, 48000)
    assert nan_s.mos == 1.0 and nan_s.confidence == 0.95

    # Silence
    sil_s = scorer.score(np.zeros(4800, dtype=np.float32), 48000)
    assert sil_s.mos == 3.0 and sil_s.confidence == 0.50

    # Stereo (2, N), (N, 2), very short (< 20 ms), and very low-sample-rate edge
    stereo_2n = np.stack([_make_speech(0.5, 48000)] * 2, axis=0)
    assert 1.0 <= scorer.score(stereo_2n, 48000).mos <= 5.0
    assert 1.0 <= scorer.score(stereo_2n.T, 48000).mos <= 5.0
    tiny = np.array([0.1, -0.1, 0.08], dtype=np.float32)
    assert 1.0 <= scorer.score(tiny, 8000).mos <= 5.0
    # Extreme peak scaling branch in _normalize_rms
    spiky = np.zeros(4800, dtype=np.float32)
    spiky[100] = 1.0
    assert 1.0 <= scorer.score(spiky, 48000).mos <= 5.0

    # Unavailable requested neural backends with allow_fallback True vs False
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: False)
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_dnsmos_available", lambda: False)
    assert (
        PerceptualQualityScorer(backend="nisqa_v2", allow_fallback=True).active_backend
        == "acoustic_proxy"
    )
    assert (
        PerceptualQualityScorer(backend="dnsmos", allow_fallback=True).active_backend
        == "acoustic_proxy"
    )
    with pytest.raises(RuntimeError, match="NISQA v2"):
        _ = PerceptualQualityScorer(backend="nisqa_v2", allow_fallback=False).active_backend
    with pytest.raises(RuntimeError, match="DNSMOS"):
        _ = PerceptualQualityScorer(backend="dnsmos", allow_fallback=False).active_backend


def test_perceptual_scorer_neural_nisqa_and_dnsmos_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sr = 48000
    speech = _make_speech(duration_s=3.0, sr=sr)
    fake_torch = _make_fake_torch()
    nisqa_output = [4.42, 4.15, 4.61, 4.30, 4.28]
    dnsmos_output = [3.95, 4.10, 4.20, 3.85]

    nisqa_mod = types.SimpleNamespace(
        non_intrusive_speech_quality_assessment=lambda _t, _sr: _FakeTensor(
            np.array(nisqa_output, dtype=np.float32)
        )
    )
    dnsmos_mod = types.SimpleNamespace(
        deep_noise_suppression_mean_opinion_score=lambda _t, _sr, **_kw: _FakeTensor(
            np.array(dnsmos_output, dtype=np.float32)
        )
    )

    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None) -> object:
        if name == "torch":
            return fake_torch
        if name == "torchmetrics.functional.audio.nisqa":
            return nisqa_mod
        if name == "torchmetrics.functional.audio.dnsmos":
            return dnsmos_mod
        return real_import(name, package)

    monkeypatch.setattr(
        "hawavoclean.guard.perceptual_mos._is_native_nisqa_available", lambda: False
    )
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: True)
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_dnsmos_available", lambda: True)
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos.importlib.import_module", _fake_import)

    # 1. Auto selects NISQA v2 and returns exact 5 axes + calibrated=True
    auto_scorer = PerceptualQualityScorer(backend="auto")
    assert auto_scorer.active_backend == "nisqa_v2"
    score_n = auto_scorer.score(speech, sr)
    assert score_n.backend == "nisqa_v2"
    assert score_n.calibrated is True
    assert score_n.mos == 4.42
    assert score_n.noisiness == 4.15
    assert score_n.discontinuity == 4.61
    assert score_n.coloration == 4.30
    assert score_n.loudness == 4.28
    assert score_n.confidence >= 0.88

    # 2. Explicit DNSMOS (both 48 kHz resampled and native 16 kHz)
    dns_scorer = PerceptualQualityScorer(backend="dnsmos")
    assert dns_scorer.active_backend == "dnsmos"
    score_d = dns_scorer.score(speech, sr)
    assert score_d.backend == "dnsmos"
    assert score_d.calibrated is True
    assert score_d.mos == 3.85
    assert score_d.noisiness == 4.20
    assert score_d.discontinuity == 4.10
    assert score_d.loudness == 3.95
    score_d16 = dns_scorer.score(speech[:16000], 16000)
    assert score_d16.backend == "dnsmos"

    # 3. Auto selects DNSMOS when NISQA unavailable
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: False)
    assert PerceptualQualityScorer(backend="auto").active_backend == "dnsmos"

    # 4. NISQA malformed output falls back to DNSMOS, and if DNSMOS also fails -> acoustic_proxy
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: True)
    nisqa_mod.non_intrusive_speech_quality_assessment = lambda _t, _sr: _FakeTensor(
        np.array([1.0], dtype=np.float32)
    )
    fallback_to_dns = PerceptualQualityScorer(backend="nisqa_v2", allow_fallback=True).score(
        speech, sr
    )
    assert fallback_to_dns.backend == "dnsmos"

    with pytest.raises(RuntimeError, match="unexpected NISQA output"):
        PerceptualQualityScorer(backend="nisqa_v2", allow_fallback=False).score(speech, sr)

    dnsmos_mod.deep_noise_suppression_mean_opinion_score = lambda _t, _sr, **_kw: _FakeTensor(
        np.array([float("nan")] * 4, dtype=np.float32)
    )
    fallback_to_proxy = PerceptualQualityScorer(backend="nisqa_v2", allow_fallback=True).score(
        speech, sr
    )
    assert fallback_to_proxy.backend == "acoustic_proxy"
    assert fallback_to_proxy.calibrated is False

    fallback_dns_proxy = PerceptualQualityScorer(backend="dnsmos", allow_fallback=True).score(
        speech, sr
    )
    assert fallback_dns_proxy.backend == "acoustic_proxy"
    with pytest.raises(RuntimeError, match="unexpected DNSMOS output"):
        PerceptualQualityScorer(backend="dnsmos", allow_fallback=False).score(speech, sr)


def test_native_nisqa_arch_weights_melspec_and_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from hawavoclean.errors import ModelProvenanceError
    from hawavoclean.guard import nisqa_arch, perceptual_mos

    nisqa_arch._load_nisqa_cached.cache_clear()

    # 1. Weight path resolution order: explicit env, HAWAVOCLEAN_MODEL_DIR, and default scan
    explicit = tmp_path / "custom_nisqa.tar"
    monkeypatch.setenv(nisqa_arch.NISQA_WEIGHTS_ENV, str(explicit))
    assert nisqa_arch.nisqa_weights_path() == explicit.resolve()
    ok_missing, msg_missing = nisqa_arch.verify_nisqa_weights()
    assert not ok_missing and "missing" in msg_missing and str(tmp_path) not in msg_missing

    monkeypatch.delenv(nisqa_arch.NISQA_WEIGHTS_ENV, raising=False)
    iso = tmp_path / "iso_models"
    iso.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(iso))
    assert nisqa_arch.nisqa_weights_path() == iso.resolve() / "nisqa" / "nisqa.tar"

    monkeypatch.delenv("HAWAVOCLEAN_MODEL_DIR", raising=False)
    fake_home = tmp_path / "home"
    cache_weights = fake_home / ".cache" / "torchmetrics" / "NISQA" / "nisqa.tar"
    monkeypatch.setattr(nisqa_arch, "models_dir", lambda: tmp_path / "empty_pkg")
    monkeypatch.setattr(nisqa_arch, "app_data_root", lambda: tmp_path / "empty_app")
    monkeypatch.setattr(nisqa_arch, "_local_ai_models_dir", lambda: tmp_path / "empty_local")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: fake_home))
    assert nisqa_arch.nisqa_weights_path() == tmp_path / "empty_pkg" / "nisqa" / "nisqa.tar"
    cache_weights.parent.mkdir(parents=True)
    cache_weights.write_bytes(b"corrupt")
    assert nisqa_arch.nisqa_weights_path() == cache_weights

    # Digest mismatch fails closed with path-free reason
    ok_bad, msg_bad = nisqa_arch.verify_nisqa_weights(cache_weights)
    assert not ok_bad and "digest mismatch" in msg_bad and str(tmp_path) not in msg_bad
    with pytest.raises(ModelProvenanceError, match="digest mismatch"):
        nisqa_arch.load_nisqa_v2_model(cache_weights)

    # Corrupt checkpoint missing required fields raises ModelProvenanceError
    bad_ckpt = tmp_path / "bad_ckpt.tar"
    torch.save({"unexpected": 1}, str(bad_ckpt))
    monkeypatch.setattr(
        nisqa_arch,
        "NISQA_V2_WEIGHTS_SHA256",
        hashlib.sha256(bad_ckpt.read_bytes()).hexdigest(),
    )
    with pytest.raises(ModelProvenanceError, match="missing required fields"):
        nisqa_arch.load_nisqa_v2_model(bad_ckpt)

    # 2. Build a tiny genuine NISQADimModel checkpoint and run full pure-PyTorch inference
    mini_args: dict[str, object] = {
        "ms_fmax": 20000,
        "ms_n_fft": 1024,
        "ms_hop_length": 0.01,
        "ms_win_length": 0.02,
        "ms_n_mels": 16,
        "ms_seg_length": 10,
        "ms_seg_hop_length": 4,
        "ms_max_segments": 64,
        "cnn_c_out_1": 4,
        "cnn_c_out_2": 4,
        "cnn_c_out_3": 8,
        "cnn_kernel_size": (3, 3),
        "cnn_pool_1": [8, 5],
        "cnn_pool_2": [4, 3],
        "cnn_pool_3": [2, 2],
        "td_sa_d_model": 8,
        "td_sa_nhead": 1,
        "td_sa_num_layers": 1,
        "td_sa_h": 8,
        "td_sa_dropout": 0.0,
        "pool_att_h": 8,
    }
    torch.manual_seed(0)
    mini_model = nisqa_arch.NISQADimModel(mini_args)
    with torch.no_grad():
        for pool in mini_model.pool_layers:
            for name, param in pool.named_parameters():
                if name.endswith("linear3.weight"):
                    param.zero_()
                elif name.endswith("linear3.bias"):
                    param.fill_(3.75)
    mini_ckpt = tmp_path / "mini_nisqa.tar"
    torch.save({"args": mini_args, "model_state_dict": mini_model.state_dict()}, str(mini_ckpt))
    monkeypatch.setenv(nisqa_arch.NISQA_WEIGHTS_ENV, str(mini_ckpt))
    monkeypatch.setattr(
        nisqa_arch,
        "NISQA_V2_WEIGHTS_SHA256",
        hashlib.sha256(mini_ckpt.read_bytes()).hexdigest(),
    )
    nisqa_arch._load_nisqa_cached.cache_clear()

    speech = _make_speech(duration_s=0.6, sr=48000)
    vals = nisqa_arch.score_nisqa_v2(speech, 48000)
    assert len(vals) == 5
    assert all(abs(v - 3.75) < 1e-4 for v in vals)

    # Also test 1D kernel padding branch (cnn_kernel_size=(1, 3))
    k1_args = dict(mini_args)
    k1_args["cnn_kernel_size"] = (1, 3)
    k1_model = nisqa_arch.NISQADimModel(k1_args).eval()
    assert k1_model.cnn.model.conv1.padding == (1, 0)

    # 3. PerceptualQualityScorer uses native nisqa_arch first (without torchmetrics!)
    monkeypatch.setattr(
        "hawavoclean.guard.perceptual_mos._modules_present",
        lambda *names: names == ("torch",),
    )
    scorer = PerceptualQualityScorer(backend="auto")
    assert scorer.active_backend == "nisqa_v2"
    score = scorer.score(speech, 48000)
    assert score.backend == "nisqa_v2" and score.calibrated is True and score.mos == 3.75

    # 4. Edge/error branches in nisqa_arch & perceptual_mos without torch
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        nisqa_arch.score_nisqa_v2(speech, 0)
    with pytest.raises(RuntimeError, match="too short"):
        nisqa_arch.compute_nisqa_melspec(np.array([0.1], dtype=np.float32), 48000, mini_args)
    with pytest.raises(ValueError, match="Invalid NISQA STFT parameters"):
        nisqa_arch.compute_nisqa_melspec(speech, 10, mini_args)
    with pytest.raises(RuntimeError, match="too short"):
        nisqa_arch.segment_nisqa_specs(torch.zeros((1, 16, 4)), mini_args)
    with pytest.raises(RuntimeError, match="Maximum number"):
        nisqa_arch.segment_nisqa_specs(torch.zeros((1, 16, 500)), mini_args)

    # Non-finite model prediction fails closed
    nan_model = nisqa_arch.NISQADimModel(mini_args)
    with torch.no_grad():
        for name, param in nan_model.pool_layers[0].named_parameters():
            if name.endswith("linear3.bias"):
                param.fill_(float("nan"))
    monkeypatch.setattr(
        nisqa_arch, "load_nisqa_v2_model", lambda **_k: (nan_model.eval(), mini_args)
    )
    with pytest.raises(RuntimeError, match="unexpected NISQA output"):
        nisqa_arch.score_nisqa_v2(speech, 48000)

    # Ordering preservation below 1.08 in _clip_mos
    assert 1.0 <= perceptual_mos._clip_mos(0.79) < perceptual_mos._clip_mos(0.96) < 1.08
    assert perceptual_mos._clip_mos(4.32432) == 4.324

    # Fallback path when torch is absent
    monkeypatch.delenv(nisqa_arch.NISQA_WEIGHTS_ENV, raising=False)
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._modules_present", lambda *_n: False)
    assert perceptual_mos._is_native_nisqa_available() is False
    assert perceptual_mos._nisqa_weights_path().name == "nisqa.tar"
    nisqa_arch._load_nisqa_cached.cache_clear()


# ---------------------------------------------------------------------------
# 2. PhoneticStabilityResult & PhoneticStabilityGuard tests
# ---------------------------------------------------------------------------


def test_phonetic_stability_result_validation() -> None:
    ok = PhoneticStabilityResult(
        cosine_similarity=0.95,
        min_frame_cosine=0.80,
        active_speech_frames=42,
        backend="mfcc_active_speech",
        passes_check=True,
        status="phonetic_preserved",
    )
    assert ok.passes_check is True

    with pytest.raises(ValueError, match="within"):
        PhoneticStabilityResult(1.5, 0.5, 10, "mfcc_active_speech", True, "ok")
    with pytest.raises(ValueError, match="non-negative"):
        PhoneticStabilityResult(0.9, 0.5, -1, "mfcc_active_speech", True, "ok")
    with pytest.raises(ValueError, match="backend must be non-empty"):
        PhoneticStabilityResult(0.9, 0.5, 5, "", True, "ok")
    with pytest.raises(ValueError, match="status must be non-empty"):
        PhoneticStabilityResult(0.9, 0.5, 5, "mfcc_active_speech", True, "")


def test_phonetic_guard_mfcc_dereverb_safe_and_catches_collapse() -> None:
    sr = 48000
    clean = _make_speech(duration_s=2.0, sr=sr)
    wiener = WienerSpectralEnhancer().enhance(clean, sr).waveform

    guard = PhoneticStabilityGuard(sample_rate=sr, backend="mfcc_active_speech")
    assert guard.active_backend == "mfcc_active_speech"

    # 1. Clean vs Wiener passes with high cosine
    res_wiener = guard.evaluate(clean, wiener)
    assert res_wiener.passes_check is True
    assert res_wiener.backend == "mfcc_active_speech"
    assert res_wiener.status == "phonetic_preserved"
    assert res_wiener.cosine_similarity >= 0.95
    assert res_wiener.active_speech_frames > 20

    # 2. Reverberant -> dereverbed (removing reverb decay tails) passes without false revert!
    # Use a realistic room reverb (DRR +5 dB, RT60 0.65s) with subtle natural syllable transition
    rir_room = _make_rir(sr, rt60=0.65, drr_db=5.0)
    reverberant_room = signal.fftconvolve(clean, rir_room, mode="full")[: len(clean)].astype(
        np.float32
    )
    res_dereverb = guard.evaluate(reverberant_room, clean)
    assert res_dereverb.passes_check is True
    assert res_dereverb.status == "phonetic_preserved"

    # 2b. Heavy continuous background noise cleaned down to quiet pauses passes without false revert
    rng_noise = np.random.default_rng(13)
    noise_bed = signal.sosfilt(
        signal.butter(2, 900.0, btype="lowpass", fs=sr, output="sos"),
        rng_noise.standard_normal(len(clean)),
    ).astype(np.float32)
    noise_bed *= (
        float(np.sqrt(np.mean(clean**2))) / (float(np.sqrt(np.mean(noise_bed**2))) + 1e-12) * 0.45
    )
    noisy_speech = (clean + noise_bed).astype(np.float32)
    res_denoise = guard.evaluate(noisy_speech, clean)
    assert res_denoise.passes_check is True
    assert res_denoise.status == "phonetic_preserved"

    # 3. Positive and negative latency shifts are delay-aligned cleanly
    shift = int(0.015 * sr)
    shifted_pos = np.concatenate([np.zeros(shift, dtype=np.float32), clean[:-shift]])
    shifted_neg = np.concatenate([clean[shift:], np.zeros(shift, dtype=np.float32)])
    assert guard.evaluate(clean, shifted_pos).passes_check is True
    assert guard.evaluate(clean, shifted_neg).passes_check is True

    # 4. Impulse corruption fails closed
    impulse = np.zeros_like(clean)
    impulse[::4] = 0.8
    res_imp = guard.evaluate(clean, impulse)
    assert res_imp.passes_check is False
    assert res_imp.status == "phonetic_drift_detected"

    # 5. Severe speech dropout fails closed
    dropped = clean.copy()
    dropped[sr // 4 : (3 * sr) // 2] = 0.0
    res_drop = guard.evaluate(clean, dropped)
    assert res_drop.passes_check is False

    # 6. Frequency-shifted / formant-collapsed audio fails closed
    t = np.arange(len(clean), dtype=np.float64) / sr
    freq_shifted = np.real(signal.hilbert(clean) * np.exp(1j * 2.0 * np.pi * 650.0 * t)).astype(
        np.float32
    )
    res_shift = guard.evaluate(clean, freq_shifted)
    assert res_shift.passes_check is False

    # 7. Localized consonant/formant collapse triggers phonetic_collapse_detected when mean ok
    strict_p05_guard = PhoneticStabilityGuard(
        sample_rate=sr,
        backend="mfcc_active_speech",
        min_cosine=0.40,
        min_p05_cosine=0.90,
    )
    partial_drop = clean.copy()
    partial_drop[sr // 2 : sr // 2 + sr // 4] = 0.0
    res_collapse = strict_p05_guard.evaluate(clean, partial_drop)
    assert res_collapse.passes_check is False
    assert res_collapse.status == "phonetic_collapse_detected"


def test_phonetic_guard_edge_cases_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        PhoneticStabilityGuard(sample_rate=0)
    with pytest.raises(ValueError, match="unknown phonetic guard backend"):
        PhoneticStabilityGuard(backend="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown device"):
        PhoneticStabilityGuard(device="tpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layer_index"):
        PhoneticStabilityGuard(layer_index=0)

    guard = PhoneticStabilityGuard(sample_rate=48000, backend="mfcc_active_speech")
    with pytest.raises(ValueError, match="sample_rate"):
        guard.evaluate(
            np.zeros(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32), sample_rate=0
        )

    # Non-finite fails closed
    nan_res = guard.evaluate(
        np.array([0.1, float("nan")], dtype=np.float32), np.zeros(2, dtype=np.float32)
    )
    assert nan_res.passes_check is False and nan_res.status == "non_finite_audio"

    # Too short passes cleanly
    short_res = guard.evaluate(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32))
    assert short_res.passes_check is True and short_res.status == "audio_too_short"

    # Pure silence -> no_active_speech
    sil_res = guard.evaluate(
        np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.float32), sample_rate=16000
    )
    assert sil_res.passes_check is True and sil_res.status == "no_active_speech"

    # Stereo (2, N) and (N, 2) inputs
    speech16 = _make_speech(0.5, 16000)
    st_res = guard.evaluate(
        np.stack([speech16, speech16], axis=0),
        np.stack([speech16, speech16], axis=1),
        sample_rate=16000,
    )
    assert st_res.passes_check is True

    # Weight discovery branches: local dir, HF hub cache, HF_HOME, invalid ref
    local_dir = tmp_path / "wavlm_local"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}")
    monkeypatch.setattr(
        "hawavoclean.guard.phonetic_guard.importlib.util.find_spec", lambda _n: object()
    )
    st_local = phonetic_guard_status(model_path=local_dir)
    assert st_local["wavlm_available"] is True and st_local["backend"] == "wavlm_layer6"

    # HF hub cache via HF_HOME
    hf_home = tmp_path / "hf_home"
    snap = hf_home / "hub" / "models--microsoft--wavlm-base-plus" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HAWAVOCLEAN_WAVLM_MODEL_PATH", raising=False)
    st_hub = phonetic_guard_status()
    assert st_hub["wavlm_available"] is True

    # Missing model dir and non-repo name
    monkeypatch.setenv("HAWAVOCLEAN_WAVLM_MODEL_PATH", "not-a-repo-id")
    assert phonetic_guard_status()["wavlm_available"] is False
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "empty_cache"))
    monkeypatch.delenv("HAWAVOCLEAN_WAVLM_MODEL_PATH", raising=False)
    assert phonetic_guard_status()["wavlm_available"] is False

    with pytest.raises(RuntimeError, match="WavLM backend requested"):
        _ = PhoneticStabilityGuard(backend="wavlm_layer6", allow_fallback=False).active_backend


def test_phonetic_guard_wavlm_backend_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    sr = 16000
    speech = _make_speech(duration_s=1.0, sr=sr)
    _WAVLM_MODEL_CACHE.clear()

    class _FakeWavLM:
        def __init__(self) -> None:
            self.calls = 0
            self.should_boom = False

        def eval(self) -> _FakeWavLM:
            return self

        def to(self, _dev: object) -> _FakeWavLM:
            return self

        def __call__(self, tensor: _FakeTensor, output_hidden_states: bool = True) -> object:
            assert output_hidden_states is True
            if self.should_boom:
                raise RuntimeError("wavlm runtime boom")
            self.calls += 1
            arr = tensor.numpy()[0]
            # Deterministic 50 Hz frame embeddings derived from waveform energy spectrum
            n_frames = max(1, (len(arr) - 400) // 320 + 1)
            feats = np.zeros((1, n_frames, 8), dtype=np.float32)
            for i in range(n_frames):
                seg = arr[i * 320 : i * 320 + 400]
                spec = np.abs(np.fft.rfft(seg, n=64))[:8]
                feats[0, i] = spec + 1e-4
            return types.SimpleNamespace(hidden_states=[_FakeTensor(feats)] * 7)

    fake_model = _FakeWavLM()
    fake_torch = _make_fake_torch(mps=True)

    def _fake_from_pretrained(_ref: str, *, local_files_only: bool = False) -> _FakeWavLM:
        assert local_files_only is True
        return fake_model

    fake_tf = types.SimpleNamespace(
        WavLMModel=types.SimpleNamespace(from_pretrained=_fake_from_pretrained)
    )

    real_import = importlib.import_module

    def _fake_import(name: str, package: str | None = None) -> object:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_tf
        return real_import(name, package)

    monkeypatch.setattr("hawavoclean.guard.phonetic_guard._is_wavlm_available", lambda _ref: True)
    monkeypatch.setattr("hawavoclean.guard.phonetic_guard.importlib.import_module", _fake_import)

    guard = PhoneticStabilityGuard(sample_rate=sr, backend="auto", device="auto")
    assert guard.active_backend == "wavlm_layer6"
    res = guard.evaluate(speech, speech, sample_rate=sr)
    assert res.backend == "wavlm_layer6"
    assert res.passes_check is True
    assert res.cosine_similarity == 1.0

    # Second call hits model cache (and test explicit CPU + CUDA resolution branches)
    res2 = guard.evaluate(speech, speech, sample_rate=sr)
    assert res2.backend == "wavlm_layer6"
    assert guard._resolve_torch_device(_make_fake_torch(mps=False, cuda=True)) == "cuda"
    assert guard._resolve_torch_device(_make_fake_torch(mps=False, cuda=False)) == "cpu"
    assert PhoneticStabilityGuard(device="cpu")._resolve_torch_device(fake_torch) == "cpu"

    # Runtime error inside WavLM falls back cleanly when allow_fallback=True, re-raises when False
    fake_model.should_boom = True
    res_fb = guard.evaluate(speech, speech, sample_rate=sr)
    assert res_fb.backend == "mfcc_active_speech"
    assert res_fb.passes_check is True

    strict_guard = PhoneticStabilityGuard(
        sample_rate=sr, backend="wavlm_layer6", allow_fallback=False
    )
    with pytest.raises(RuntimeError, match="wavlm runtime boom"):
        strict_guard.evaluate(speech, speech, sample_rate=sr)
    _WAVLM_MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# 3. Multi-window extract_preview_slice, artifact guard, clean abstention, studio_pro
# ---------------------------------------------------------------------------


def test_extract_preview_slice_skips_intro_silence_and_picks_speech_windows() -> None:
    sr = 48000
    # 12s dead intro silence + 5s clean speech + 5s mid speech + 6s noisy speech = 28s
    intro_silence = np.zeros(sr * 12, dtype=np.float32)
    speech_a = _make_speech(duration_s=5.0, sr=sr, seed=1) * 1.4
    speech_b = _make_speech(duration_s=5.0, sr=sr, seed=2)
    rng = np.random.default_rng(99)
    speech_noisy = (
        _make_speech(duration_s=6.0, sr=sr, seed=3)
        + rng.standard_normal(sr * 6).astype(np.float32) * 0.04
    ).astype(np.float32)
    long_clip = np.concatenate([intro_silence, speech_a, speech_b, speech_noisy])

    preview = extract_preview_slice(long_clip, sr, max_duration_s=10.0, target_sample_rate=sr)
    assert len(preview) <= 10 * sr
    assert len(preview) >= int(9.5 * sr)
    # Blind mono[:10s] would be pure zero RMS; speech-active preview has strong speech energy!
    assert float(np.sqrt(np.mean(preview**2))) > 0.05

    # Short preview budget (< 1.5s) still skips intro silence
    short_budget = extract_preview_slice(long_clip, sr, max_duration_s=1.0, target_sample_rate=sr)
    assert len(short_budget) == sr
    assert float(np.sqrt(np.mean(short_budget**2))) > 0.05

    # Single active speech island surrounded by silence centers around speech
    island = np.concatenate(
        [
            np.zeros(sr * 12, dtype=np.float32),
            _make_speech(2.0, sr),
            np.zeros(sr * 12, dtype=np.float32),
        ]
    )
    island_preview = extract_preview_slice(island, sr, max_duration_s=10.0, target_sample_rate=sr)
    assert len(island_preview) == 10 * sr
    assert float(np.sqrt(np.mean(island_preview**2))) > 0.01

    # Silent clip falls back cleanly to mono[:max_samples]
    silent = np.zeros(sr * 15, dtype=np.float32)
    sil_preview = extract_preview_slice(silent, sr, max_duration_s=10.0, target_sample_rate=sr)
    assert len(sil_preview) == 10 * sr
    assert float(np.max(np.abs(sil_preview))) == 0.0

    # Short clip (< max_duration_s) returns exact length
    short_clip = _make_speech(1.0, sr)
    assert len(extract_preview_slice(short_clip, sr, max_duration_s=10.0)) == len(short_clip)

    # Tight duration (10.05s -> 10.0s) hits fade=0 non-overlapping branch cleanly
    tight = _make_speech(10.05, sr)
    tight_prev = extract_preview_slice(tight, sr, max_duration_s=10.0, target_sample_rate=sr)
    assert 0 < len(tight_prev) <= 10 * sr

    # _crossfade_concat edge cases
    assert len(_crossfade_concat([], 100)) == 0
    tiny_join = _crossfade_concat([np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)], 10)
    assert len(tiny_join) == 4


def test_perceptual_artifact_guard_discontinuity_and_coloration() -> None:
    ref_cal = PerceptualQualityScore(4.2, 4.0, 4.3, 4.1, 4.2, 0.90, "nisqa_v2", True)
    good_cand = PerceptualQualityScore(4.4, 4.5, 4.4, 4.2, 4.3, 0.90, "nisqa_v2", True)
    bad_dis = PerceptualQualityScore(3.5, 4.2, 2.35, 4.0, 4.1, 0.90, "nisqa_v2", True)
    bad_col = PerceptualQualityScore(3.4, 4.2, 4.1, 2.10, 4.1, 0.90, "nisqa_v2", True)
    uncal = PerceptualQualityScore(2.0, 2.0, 1.5, 1.5, 2.0, 0.70, "acoustic_proxy", False)

    assert passes_perceptual_artifact_check(ref_cal, good_cand) is True
    assert passes_perceptual_artifact_check(ref_cal, bad_dis) is False
    assert passes_perceptual_artifact_check(ref_cal, bad_col) is False
    assert passes_perceptual_artifact_check(uncal, uncal) is True

    # Reference already had low coloration (2.20), candidate improved to 2.45 -> MUST PASS
    reverb_ref = PerceptualQualityScore(1.6, 2.5, 2.88, 2.20, 3.0, 0.90, "nisqa_v2", True)
    reverb_improved = PerceptualQualityScore(2.6, 3.5, 2.99, 2.45, 3.3, 0.90, "nisqa_v2", True)
    assert passes_perceptual_artifact_check(reverb_ref, reverb_improved) is True

    # Wired inside SmartSafePreviewEngine: neural discontinuity collapse fails artifact_guard_passed!
    class _CollapsingScorer(PerceptualQualityScorer):
        def __init__(self) -> None:
            super().__init__(backend="acoustic_proxy")
            self.calls = 0

        def score(self, _audio: np.ndarray, _sr: int) -> PerceptualQualityScore:
            self.calls += 1
            return ref_cal if self.calls == 1 else bad_dis

    sr = 48000
    speech = _make_speech(0.5, sr)
    engine = SmartSafePreviewEngine(perceptual_scorer=_CollapsingScorer())
    prev = engine.generate_preview("production", speech, sr, acoustic_evidence=_evidence())
    assert prev.evidence.artifact_guard_passed is False
    assert prev.evidence.predicted_quality_mos == 1.50
    assert verify_candidate_evidence_integrity(prev) is True


def test_clean_speech_true_gain_abstention_selects_preserve() -> None:
    evidence = _evidence()
    ranker = _ranker(qualified=True)

    # 1. Input already clean (preserve MOS 4.40), best candidate only gains +0.14 penalized -> abstain!
    cands_marginal = [
        CandidateEvidence("preserve", 4.40, 0.92, True, True, True, True, True),
        CandidateEvidence("production", 4.47, 0.92, True, True, True, True, True),
        CandidateEvidence(
            "studio", 4.60, 0.92, True, True, True, True, True
        ),  # 4.60 - 0.06 = 4.54 (+0.14 < 0.15)
    ]
    decision = decide_smart_safe(
        evidence,
        cands_marginal,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=ranker,
        require_qualified_ranker=True,
    )
    assert decision.abstained is True
    assert decision.selected_route == "preserve"
    assert "minimum gain threshold" in decision.reason

    # 2. Input already clean (4.25), candidate genuinely gains +0.35 net penalized -> wins!
    cands_clear_win = [
        CandidateEvidence("preserve", 4.25, 0.92, True, True, True, True, True),
        CandidateEvidence(
            "production", 4.65, 0.92, True, True, True, True, True
        ),  # 4.65 - 0.03 = 4.62 (+0.37 >= 0.15)
    ]
    decision_win = decide_smart_safe(
        evidence,
        cands_clear_win,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=ranker,
        require_qualified_ranker=True,
    )
    assert decision_win.abstained is False
    assert decision_win.selected_route == "production"

    # 3. Policy validation for new clean abstention thresholds
    with pytest.raises(ValueError, match="clean_preserve_mos_min"):
        SmartSafePolicy(clean_preserve_mos_min=0.5)
    with pytest.raises(ValueError, match="clean_min_gain_mos"):
        SmartSafePolicy(clean_min_gain_mos=-0.1)


def test_studio_pro_route_registration_rendering_and_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sr = 48000
    speech = _make_speech(0.5, sr)
    evidence = _evidence()

    # 1. Eligibility includes studio_pro and dereverb when include_optional=True
    elig = eligible_routes(
        evidence, restore_policy="disabled", speaker_profile_id=None, include_optional=True
    )
    assert "studio_pro" in elig and elig["studio_pro"] == (True, ())
    assert "dereverb" in elig and elig["dereverb"] == (True, ())

    elig_music = eligible_routes(
        _evidence(music_risk=0.80, crosstalk_risk=0.50),
        restore_policy="disabled",
        speaker_profile_id=None,
        include_optional=True,
    )
    assert elig_music["dereverb"][0] is False
    assert elig_music["dereverb"] == elig_music["studio_pro"]

    # 2. When studio_pro / dereverb are NOT registered, render_route_audio fails closed cleanly
    monkeypatch.delitem(CORE_REGISTRY, STUDIO_PRO_CORE_ID, raising=False)
    monkeypatch.delitem(CORE_REGISTRY, STUDIO_DEREVERB_CORE_ID, raising=False)
    engine = SmartSafePreviewEngine(include_studio_pro=True)
    _, err = engine.render_route_audio("studio_pro", speech, sr)
    assert err is not None and "not registered" in err
    _, dereverb_err = engine.render_route_audio("dereverb", speech, sr)
    assert dereverb_err is not None and "not registered" in dereverb_err

    # 3. When studio_pro IS registered in CORE_REGISTRY, preview + decide include it cleanly
    class _DummyStudioPro:
        def __init__(
            self,
            core_id: str = STUDIO_PRO_CORE_ID,
            sample_rate: int = 48000,
            phase_coherent: bool = True,
        ) -> None:
            self._meta = EnhancerMetadata(
                core_id=core_id,
                version="1.0",
                algorithm="dummy-studio-pro",
                sample_rate=sample_rate,
                phase_coherent=phase_coherent,
            )

        @property
        def metadata(self) -> EnhancerMetadata:
            return self._meta

        def warmup(self) -> None:
            return None

        def enhance(self, waveform: np.ndarray, sample_rate: int) -> EnhancementResult:
            out = WienerSpectralEnhancer().enhance(waveform, sample_rate).waveform
            return EnhancementResult(
                waveform=out,
                sample_rate=sample_rate,
                model_runtime_ms=1.0,
                input_samples=len(waveform),
                output_samples=len(out),
            )

    monkeypatch.setitem(
        CORE_REGISTRY,
        STUDIO_PRO_CORE_ID,
        CoreRegistration(
            enhancer_class=_DummyStudioPro,
            lock_filename="studio-pro-core.lock.toml",
            implementation_params_hash=lambda: "0" * 64,
        ),
    )

    decision, previews = engine.decide(
        speech,
        sr,
        acoustic_evidence=evidence,
        restore_policy="disabled",
        ranker=_ranker(qualified=False),
        require_qualified_ranker=False,
        include_studio_pro=True,
    )
    assert "studio_pro" in previews
    assert "dereverb" not in previews
    assert len(previews) == 8
    assert verify_candidate_evidence_integrity(previews["studio_pro"]) is True
    assert any(c.route == "studio_pro" for c in decision.candidates)

    # 3b. When dereverb IS also registered in CORE_REGISTRY, preview + decide include both
    monkeypatch.setitem(
        CORE_REGISTRY,
        STUDIO_DEREVERB_CORE_ID,
        CoreRegistration(
            enhancer_class=_DummyStudioPro,
            lock_filename="studio-dereverb-core.lock.toml",
            implementation_params_hash=lambda: "1" * 64,
        ),
    )
    dec_both, previews_both = engine.decide(
        speech,
        sr,
        acoustic_evidence=evidence,
        restore_policy="disabled",
        ranker=_ranker(qualified=False),
        require_qualified_ranker=False,
        include_studio_pro=True,
    )
    assert "studio_pro" in previews_both and "dereverb" in previews_both
    assert len(previews_both) == 9
    assert verify_candidate_evidence_integrity(previews_both["dereverb"]) is True
    assert any(c.route == "dereverb" for c in dec_both.candidates)

    # Explicit include_dereverb=True with include_studio_pro=False evaluates dereverb alone
    prev_only_dereverb = engine.generate_all_previews(
        speech,
        sr,
        acoustic_evidence=evidence,
        restore_policy="disabled",
        include_studio_pro=False,
        include_dereverb=True,
    )
    assert "dereverb" in prev_only_dereverb and "studio_pro" not in prev_only_dereverb
    assert len(prev_only_dereverb) == 8

    # 4. Decision with studio_pro and dereverb winning directly
    dec_direct = decide_smart_safe(
        evidence,
        [
            CandidateEvidence("preserve", 2.0, 0.92, True, True, True, True, True),
            CandidateEvidence("production", 2.8, 0.92, True, True, True, True, True),
            CandidateEvidence("studio_pro", 4.4, 0.92, True, True, True, True, True),
        ],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(qualified=True),
        require_qualified_ranker=True,
    )
    assert dec_direct.selected_route == "studio_pro"
    assert dec_direct.abstained is False

    dec_dereverb = decide_smart_safe(
        evidence,
        [
            CandidateEvidence("preserve", 1.8, 0.92, True, True, True, True, True),
            CandidateEvidence("production", 2.1, 0.92, True, True, True, True, True),
            CandidateEvidence("studio_pro", 2.5, 0.92, True, True, True, True, True),
            CandidateEvidence("dereverb", 4.3, 0.92, True, True, True, True, True),
        ],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(qualified=True),
        require_qualified_ranker=True,
    )
    assert dec_dereverb.selected_route == "dereverb"
    assert dec_dereverb.abstained is False

    # 5. Same-tier near-tie (studio_pro vs dereverb, both cost 2, within 0.05 MOS) picks the
    #    higher-MOS winner instead of abstaining back to a degraded preserve (1.049 MOS).
    same_tier_cands = [
        CandidateEvidence("preserve", 1.049, 0.92, True, True, True, True, True),
        CandidateEvidence("production", 1.897, 0.92, True, True, True, True, True),
        CandidateEvidence("studio", 3.310, 0.92, True, True, True, True, True),
        CandidateEvidence("studio_pro", 4.488, 0.92, True, True, True, True, True),
        CandidateEvidence("dereverb", 4.443, 0.92, True, True, True, True, True),
    ]
    dec_same_tier = decide_smart_safe(
        evidence,
        same_tier_cands,
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(qualified=True),
        require_qualified_ranker=True,
    )
    assert dec_same_tier.selected_route == "studio_pro"
    assert dec_same_tier.abstained is False
    # Permutation invariance holds for same-tier near-ties
    dec_same_tier_rev = decide_smart_safe(
        evidence,
        list(reversed(same_tier_cands)),
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(qualified=True),
        require_qualified_ranker=True,
    )
    assert dec_same_tier_rev.selected_route == "studio_pro"
    assert dec_same_tier_rev.decision_sha256 == dec_same_tier.decision_sha256

    # 6. Cross-tier tie (studio vs production within 0.05 MOS) abstains to the lower-cost
    #    member of the tied band (production), NOT a far-worse preserve (1.5 MOS).
    dec_cross_tier = decide_smart_safe(
        evidence,
        [
            CandidateEvidence("preserve", 1.50, 0.92, True, True, True, True, True),
            CandidateEvidence("production", 4.10, 0.92, True, True, True, True, True),  # 4.07
            CandidateEvidence(
                "studio", 4.15, 0.92, True, True, True, True, True
            ),  # 4.09 (diff 0.02)
        ],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=_ranker(qualified=True),
        require_qualified_ranker=True,
    )
    assert dec_cross_tier.selected_route == "production"
    assert dec_cross_tier.abstained is True
    assert "tied" in dec_cross_tier.reason


def test_post_master_invariants_rejects_phonetic_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sr = 48000
    speech = _make_speech(1.0, sr)
    bad_phonetic = PhoneticStabilityResult(
        cosine_similarity=0.25,
        min_frame_cosine=0.0,
        active_speech_frames=30,
        backend="mfcc_active_speech",
        passes_check=False,
        status="phonetic_drift_detected",
    )

    class _FailingPhoneticGuard(PhoneticStabilityGuard):
        def evaluate(
            self,
            _ref: np.ndarray,
            _cand: np.ndarray,
            *,
            sample_rate: int | None = None,  # noqa: ARG002
        ) -> PhoneticStabilityResult:
            return bad_phonetic

    ok, reason = verify_post_master_invariants(
        speech,
        speech,
        "production",
        sr,
        phonetic_guard=_FailingPhoneticGuard(sample_rate=sr),
    )
    assert ok is False
    assert "post-master phonetic guard failed" in reason

    # Also verify non-finite check when SoraniLinguisticGuard passes
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.PhoneticStabilityGuard.evaluate",
        lambda *_a, **_k: bad_phonetic,
    )
    ok2, reason2 = verify_post_master_invariants(speech, speech, "production", sr)
    assert ok2 is False
    assert "phonetic_drift_detected" in reason2


def test_active_speech_speaker_similarity_fallback_handles_denoised_pauses() -> None:
    from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor

    sr = 48000
    clean = _make_speech(2.0, sr)
    rng = np.random.default_rng(21)
    noisy = (
        clean
        + signal.sosfilt(
            signal.butter(2, 900.0, btype="lowpass", fs=sr, output="sos"),
            rng.standard_normal(len(clean)),
        ).astype(np.float32)
        * 0.04
    ).astype(np.float32)
    extractor = SpeakerEmbeddingExtractor(sr)

    # Same speaker with denoised pauses passes via active-speech fallback
    ok_same, sim_same = _speaker_similarity_passes(extractor, noisy, clean, sr, threshold=0.80)
    assert ok_same is True
    assert sim_same >= 0.80

    # Genuinely altered voice/impulse fails closed on both full clip and active frames
    impulse = np.zeros_like(clean)
    impulse[::4] = 0.7
    ok_diff, _ = _speaker_similarity_passes(extractor, clean, impulse, sr, threshold=0.80)
    assert ok_diff is False

    # Short non-matching clip fails closed when active pair cannot be extracted
    short_a = _make_speech(0.1, sr)
    short_b = np.zeros_like(short_a)
    short_b[::4] = 0.7
    ok_short, _ = _speaker_similarity_passes(extractor, short_a, short_b, sr, threshold=0.80)
    assert ok_short is False


def test_enhancement_hf_artifact_helper_and_device_instantiation() -> None:
    from hawavoclean.restoration.highband_events import HighBandEventResult
    from hawavoclean.smart_safe.preview import (
        _enhancement_hf_artifact_ok,
        _instantiate_enhancer,
    )

    sr = 48000
    clean = _make_speech(0.5, sr)
    rng = np.random.default_rng(77)
    hf_noise = signal.sosfilt(
        signal.butter(4, 9000.0, btype="highpass", fs=sr, output="sos"),
        rng.standard_normal(len(clean)),
    ).astype(np.float32)
    noisy_hf = (clean + 0.15 * hf_noise).astype(np.float32)

    ok_hf = HighBandEventResult(0.0, 0, 0.0, 1.0, True)
    fail_burst = HighBandEventResult(0.8, 12, 0.7, 5.0, False)
    fail_imp_only = HighBandEventResult(0.0, 0, 0.0, 22.0, False)

    # 1. Passing HF detector passes immediately
    assert _enhancement_hf_artifact_ok(clean, clean, sr, 8000.0, ok_hf) is True

    # 2. Legitimate subtractive HF denoising passes even when delta_hf flags bursts
    assert _enhancement_hf_artifact_ok(noisy_hf, clean, sr, 8000.0, fail_burst) is True
    assert (
        _enhancement_hf_artifact_ok(
            np.stack([noisy_hf, noisy_hf], axis=0),
            np.stack([clean, clean], axis=0),
            sr,
            8000.0,
            fail_burst,
        )
        is True
    )

    # 3. Dereverb transient sharpening with zero bursts/leakage + calibrated NISQA improvement passes
    ref_s = PerceptualQualityScore(2.1, 2.0, 3.7, 3.1, 3.0, 0.90, "nisqa_v2", True)
    cand_good = PerceptualQualityScore(4.0, 4.1, 4.4, 4.0, 4.0, 0.90, "nisqa_v2", True)
    cand_bad_dis = PerceptualQualityScore(2.5, 3.0, 2.9, 3.5, 3.5, 0.90, "nisqa_v2", True)
    assert (
        _enhancement_hf_artifact_ok(clean, clean, sr, 8000.0, fail_imp_only, ref_s, cand_good)
        is True
    )
    assert (
        _enhancement_hf_artifact_ok(clean, clean, sr, 8000.0, fail_imp_only, ref_s, cand_bad_dis)
        is False
    )

    # 4. Real added HF bursts fail closed even with calibrated scorer
    assert (
        _enhancement_hf_artifact_ok(clean, noisy_hf, sr, 8000.0, fail_burst, ref_s, cand_good)
        is False
    )
    # Short clips (< 32 samples) skip HF RMS shortcut and fail closed without calibrated pass
    tiny = np.zeros(16, dtype=np.float32)
    assert _enhancement_hf_artifact_ok(tiny, tiny, sr, 8000.0, fail_burst) is False

    # 5. _instantiate_enhancer forwards device="auto" when accepted, else zero-arg
    class _WithDevice:
        def __init__(self, device: str | None = None) -> None:
            self.device = device

    class _NoDevice:
        def __init__(self) -> None:
            self.ok = True

    inst_dev = _instantiate_enhancer(_WithDevice)
    assert isinstance(inst_dev, _WithDevice) and inst_dev.device == "auto"
    inst_plain = _instantiate_enhancer(_NoDevice)
    assert isinstance(inst_plain, _NoDevice) and inst_plain.ok is True


def test_perceptual_scorer_extreme_peak_no_float32_rms_overflow() -> None:
    import warnings

    from hawavoclean.guard.perceptual_mos import _normalize_rms

    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    base = (
        0.12 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.06 * np.sin(2.0 * np.pi * 1200.0 * t)
        + 0.03 * np.sin(2.0 * np.pi * 4500.0 * t)
    ).astype(np.float32)
    extreme = (base * 1e35).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        norm_base = _normalize_rms(base, 0.08)
        norm_ext = _normalize_rms(extreme, 0.08)
        scorer = PerceptualQualityScorer(backend="acoustic_proxy")
        score_base = scorer.score(base, sr)
        score_ext = scorer.score(extreme, sr)

    assert bool(np.all(np.isfinite(norm_ext)))
    assert np.allclose(norm_base, norm_ext, atol=1e-5)
    assert abs(score_base.noisiness - score_ext.noisiness) < 0.05
    assert abs(score_base.coloration - score_ext.coloration) < 0.05
