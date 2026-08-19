"""Unit tests for unit selection policy, strength ladder, and fail-closed fallback."""

import numpy as np
import pytest

from hawavoclean.config import GuardConfig, PolicyConfig
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.guard.verdict import GuardVerdict
from hawavoclean.policy.continuity import enforce_source_continuity
from hawavoclean.policy.decision import UnitPolicyDecision, evaluate_unit_policy
from hawavoclean.segmentation.types import SpeechUnit


@pytest.mark.unit
def test_policy_non_speech_passthrough() -> None:
    wave = np.zeros(1000, dtype=np.float32)
    asr = FixedProbe()
    dec, _ = evaluate_unit_policy(
        orig_core_waveform=wave,
        enh_core_waveform=None,
        sample_rate=48000,
        is_speech=False,
        probe=asr,
        guard_config=GuardConfig(),
        policy_config=PolicyConfig(),
    )
    assert dec.is_enhanced is False
    assert dec.guard_verdict == GuardVerdict.NO_SPEECH


@pytest.mark.unit
def test_policy_enhancer_error_fails_closed() -> None:
    wave = 0.5 * np.ones(1000, dtype=np.float32)
    asr = FixedProbe()
    dec, _ = evaluate_unit_policy(
        orig_core_waveform=wave,
        enh_core_waveform=None,  # Enhancer failed
        sample_rate=48000,
        is_speech=True,
        probe=asr,
        guard_config=GuardConfig(),
        policy_config=PolicyConfig(),
    )
    assert dec.is_enhanced is False
    assert dec.guard_verdict == GuardVerdict.ERROR
    assert np.array_equal(dec.selected_waveform, wave)


@pytest.mark.unit
def test_enforce_continuity_reverts_cut_speech() -> None:
    orig = np.zeros(1000, dtype=np.float32)
    # Unit 0 is reverted (original), Unit 1 is enhanced with forced boundary
    u0 = SpeechUnit(0, 0, 0, 1000, 0, 1000, is_speech=True, forced_boundary=False)
    u1 = SpeechUnit(1, 0, 1000, 2000, 1000, 2000, is_speech=True, forced_boundary=True)

    d0 = UnitPolicyDecision(orig.copy(), False, 0.0, GuardVerdict.REVERT)
    d1 = UnitPolicyDecision(orig.copy() + 0.1, True, 1.0, GuardVerdict.PASS)

    adjusted = enforce_source_continuity([u0, u1], [d0, d1], [orig, orig])
    # Unit 1 must be reverted by continuity rule
    assert adjusted[1].is_enhanced is False
    assert "continuity rule" in adjusted[1].decision_reason
