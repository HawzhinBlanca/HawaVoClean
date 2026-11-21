"""Unit tests for typed RestorationRenderResult and solver provenance integrity (R2.2, R2.13)."""

import numpy as np
import pytest
import scipy.signal as signal
import torch

from hawavoclean.restoration.bandwidth import BandwidthEstimate, BandwidthEvidence
from hawavoclean.restoration.base import RestorationRenderResult, Restorer
from hawavoclean.restoration.config import RestorationConfig
from hawavoclean.restoration.guard import RestorationGuard
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.policy import RestorationPolicyManager


def _synthetic_voice(f0_hz: float = 160.0, sr: int = 48000, duration_s: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * f0_hz * t) + 0.3 * np.sin(2 * np.pi * 3 * f0_hz * t)).astype(
        np.float32
    )
    sos = signal.butter(6, 4000 / (sr / 2), btype="lowpass", output="sos")
    filtered = signal.sosfiltfilt(sos, sig)
    return np.asarray(filtered, dtype=np.float32)


def _bandwidth_estimate(cutoff_hz: float = 4000.0) -> BandwidthEstimate:
    return BandwidthEstimate(
        effective_cutoff_hz=cutoff_hz,
        confidence=0.95,
        shape="codec_lowpass",
        restore_recommended=True,
        evidence=BandwidthEvidence(
            spectral_rolloff=0.0,
            above_cutoff_snr_db=0.0,
            stationarity=1.0,
            high_band_energy_ratio_db=-60.0,
        ),
    )


def test_restorer_protocol_compliance() -> None:
    """HawaRestoreKD conforms to the Restorer runtime checkable protocol."""
    restorer = HawaRestoreKD(sample_rate=48000)
    assert isinstance(restorer, Restorer)


def test_hawarestore_render_success() -> None:
    """Normal voice audio renders successfully with full candidate ladder and typed telemetry."""
    sr = 48000
    audio = _synthetic_voice(160.0, sr=sr, duration_s=0.5)
    restorer = HawaRestoreKD(sample_rate=sr)

    res = restorer.render(
        audio_48k=audio,
        sample_rate=sr,
        effective_cutoff_hz=4000.0,
        seed=42,
    )

    assert isinstance(res, RestorationRenderResult)
    assert res.success is True
    assert res.fallback_status == "none"
    assert res.model_name == "hawarestore-kd"
    assert res.provider == restorer.device
    assert res.solver == "midpoint"
    assert res.error_message is None

    # Full strength ladder present
    strengths = [c.strength for c in res.candidates]
    assert strengths == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_hawarestore_render_short_audio_fails_closed_and_never_claims_restored() -> None:
    """Audio shorter than win_length (2048 samples) must fail closed to 0.0 Natural.

    It must NEVER emit Natural as active-strength (s > 0.0) candidates, and policy
    must NEVER report 'restored' (R2.2).
    """
    sr = 48000
    # 500 samples is ~10.4 ms, less than 2048 win_length
    short_audio = np.ones(500, dtype=np.float32) * 0.1
    restorer = HawaRestoreKD(sample_rate=sr)

    res = restorer.render(
        audio_48k=short_audio,
        sample_rate=sr,
        effective_cutoff_hz=4000.0,
        seed=42,
    )

    assert res.success is False
    assert res.fallback_status == "input_too_short"
    assert "less than analysis window" in str(res.error_message)

    # Active strengths were NOT fabricated! Only strength 0.0 exists
    assert len(res.candidates) == 1
    assert res.candidates[0].strength == 0.0
    np.testing.assert_array_equal(res.candidates[0].audio, short_audio)

    # When processed through RestorationPolicyManager:
    cfg = RestorationConfig(enabled=True, mode="explicit")
    guard = RestorationGuard(sample_rate=sr)
    policy = RestorationPolicyManager(config=cfg, restorer=restorer, guard=guard)

    out_audio, decision = policy.process_segment(
        natural_audio=short_audio,
        sample_rate=sr,
        bandwidth_est=_bandwidth_estimate(4000.0),
    )

    assert decision.action == "bypassed"
    assert decision.applied_strength == 0.0
    assert decision.render_result is not None
    assert decision.render_result.fallback_status == "input_too_short"
    assert decision.guard_result is not None
    assert "input_too_short" in decision.guard_result.reason
    np.testing.assert_array_equal(out_audio, short_audio)


def test_hawarestore_render_solver_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injected ODE solver error must emit exact Natural, strength 0, and never say restored."""
    sr = 48000
    audio = _synthetic_voice(160.0, sr=sr, duration_s=0.5)
    restorer = HawaRestoreKD(sample_rate=sr)

    def mock_solve(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("ODE integration divergence in test")

    monkeypatch.setattr(restorer, "_solve_flow_ode", mock_solve)

    res = restorer.render(
        audio_48k=audio,
        sample_rate=sr,
        effective_cutoff_hz=4000.0,
        seed=42,
    )

    assert res.success is False
    assert res.fallback_status == "solver_failure"
    assert "ODE integration divergence in test" in str(res.error_message)

    # Only strength 0.0 candidate emitted
    assert len(res.candidates) == 1
    assert res.candidates[0].strength == 0.0

    # Policy integration verification
    cfg = RestorationConfig(enabled=True, mode="explicit")
    guard = RestorationGuard(sample_rate=sr)
    policy = RestorationPolicyManager(config=cfg, restorer=restorer, guard=guard)

    out_audio, decision = policy.process_segment(
        natural_audio=audio,
        sample_rate=sr,
        bandwidth_est=_bandwidth_estimate(4000.0),
    )

    assert decision.action == "bypassed"
    assert decision.applied_strength == 0.0
    assert decision.render_result is not None
    assert decision.render_result.fallback_status == "solver_failure"
    assert decision.guard_result is not None
    assert "solver_failure" in decision.guard_result.reason
    np.testing.assert_array_equal(out_audio, audio)


def test_hawarestore_render_empty_audio_fails_closed() -> None:
    """Empty audio input returns empty_input fallback status."""
    sr = 48000
    empty = np.zeros(0, dtype=np.float32)
    restorer = HawaRestoreKD(sample_rate=sr)

    res = restorer.render(
        audio_48k=empty,
        sample_rate=sr,
        effective_cutoff_hz=4000.0,
    )

    assert res.success is False
    assert res.fallback_status == "empty_input"
    assert res.error_message == "Input audio is empty"
