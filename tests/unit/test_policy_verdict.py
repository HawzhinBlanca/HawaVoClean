"""Unit tests for unit selection policy, strength ladder, and fail-closed fallback."""

import numpy as np
import pytest

from hawavoclean.config import GuardConfig, PolicyConfig
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.guard.verdict import GuardVerdict
from hawavoclean.policy.continuity import CONTINUITY_TAPER_MS, resolve_source_continuity
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
def test_enforce_continuity_fades_cut_speech() -> None:
    sr = 48000
    taper = int(round(sr * CONTINUITY_TAPER_MS / 1000.0))
    n = sr  # 1 s: long enough to afford the fade
    orig = np.zeros(n, dtype=np.float32)
    # Unit 0 is reverted (original) and its END is a forced mid-speech cut;
    # Unit 1, right across that cut, is enhanced -> seam on unit 1's LEFT edge.
    u0 = SpeechUnit(0, 0, 0, n, 0, n, is_speech=True, forced_boundary=True)
    u1 = SpeechUnit(1, 0, n, 2 * n, n, 2 * n, is_speech=True, forced_boundary=False)

    d0 = UnitPolicyDecision(orig.copy(), False, 0.0, GuardVerdict.REVERT)
    d1 = UnitPolicyDecision(orig.copy() + 0.1, True, 1.0, GuardVerdict.PASS)

    res = resolve_source_continuity([u0, u1], [d0, d1], [orig, orig], sr)
    # Unit 1 keeps its enhancement and pays for the seam at the edge that seams.
    assert res.decisions[1].is_enhanced is True
    assert res.fade_in_samples == [0, taper]
    assert res.fade_out_samples == [0, 0]
    assert res.reverted_ids == set()


@pytest.mark.unit
def test_enforce_continuity_reverts_a_unit_too_short_to_fade() -> None:
    """The fail-closed path survives: below 4x the fade length the old remedy
    is still the only one available."""
    sr = 48000
    short = 1000  # ~21 ms, far under the 120 ms a 30 ms fade needs
    orig = np.zeros(short, dtype=np.float32)
    u0 = SpeechUnit(0, 0, 0, short, 0, short, is_speech=True, forced_boundary=True)
    u1 = SpeechUnit(1, 0, short, 2 * short, short, 2 * short, is_speech=True)

    d0 = UnitPolicyDecision(orig.copy() + 0.1, True, 1.0, GuardVerdict.PASS)
    d1 = UnitPolicyDecision(orig.copy(), False, 0.0, GuardVerdict.REVERT)

    res = resolve_source_continuity([u0, u1], [d0, d1], [orig, orig], sr)
    assert res.decisions[0].is_enhanced is False
    assert "continuity rule" in res.decisions[0].decision_reason
