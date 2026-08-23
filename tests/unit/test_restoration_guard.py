"""Unit tests for Guard R (restoration safety guard)."""

import numpy as np
import scipy.signal as signal

from hawavoclean.restoration.base import RestorationCandidate
from hawavoclean.restoration.guard import RestorationGuard


def test_guard_r_candidate_selection_pass() -> None:
    """Test candidate selection when candidate conforms to all safety layers."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    clean = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)).astype(
        np.float32
    )
    sos = signal.butter(6, 4000 / 24000, btype="lowpass", output="sos")
    nat_audio = signal.sosfiltfilt(sos, clean).astype(np.float32)

    guard = RestorationGuard(sample_rate=sr)
    candidates = [
        RestorationCandidate(strength=1.0, audio=nat_audio, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.5, audio=nat_audio, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.0, audio=nat_audio, cutoff_hz=4000.0),
    ]

    sel_audio, res = guard.select_best_candidate(nat_audio, candidates, cutoff_hz=4000.0)
    assert res.verdict == "PASS"
    assert res.accepted_strength == 1.0


def test_guard_r_fallback_on_corrupted_highband() -> None:
    """Test that a candidate with severe highband corruption/clicks is rejected down the ladder."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    nat_audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # Corrupted candidate with massive click at the end
    bad_cand_audio = nat_audio.copy()
    bad_cand_audio[-10:] += 5.0  # Click spike

    guard = RestorationGuard(sample_rate=sr)
    candidates = [
        RestorationCandidate(strength=1.0, audio=bad_cand_audio, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.0, audio=nat_audio, cutoff_hz=4000.0),
    ]

    sel_audio, res = guard.select_best_candidate(nat_audio, candidates, cutoff_hz=4000.0)
    assert res.accepted_strength == 0.0
    np.testing.assert_array_equal(sel_audio, nat_audio)


def test_guard_r_speaker_similarity_rejection() -> None:
    """Test that candidate with mismatched speaker voice identity is rejected."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    # Speaker A (120 Hz)
    spk_a_audio = (0.5 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    # Speaker B (400 Hz)
    spk_b_audio = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    guard = RestorationGuard(sample_rate=sr)
    emb_a = guard.speaker_extractor.extract(spk_a_audio)

    # Evaluate spk_b candidate against spk_a canonical embedding
    passes, reason, metrics = guard.evaluate_candidate(
        natural_audio=spk_b_audio,
        candidate_audio=spk_b_audio,
        cutoff_hz=4000.0,
        canonical_embedding=emb_a,
    )
    # Speaker similarity should be below threshold
    assert not passes
    assert "Speaker similarity" in reason


def test_guard_r_linguistic_posterior_divergence() -> None:
    """Test that candidate with severe acoustic-phonetic distortion in speech band is rejected."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    nat_audio = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t)).astype(
        np.float32
    )

    # Distorted candidate with heavy modulated noise in speech band (500-2000 Hz)
    distorted = nat_audio.copy()
    noise = np.random.normal(0, 0.8, len(t)).astype(np.float32)
    sos = signal.butter(4, [500.0, 2000.0], btype="bandpass", fs=sr, output="sos")
    distorted += signal.sosfiltfilt(sos, noise).astype(np.float32)

    guard = RestorationGuard(sample_rate=sr)
    passes, reason, metrics = guard.evaluate_candidate(
        natural_audio=nat_audio,
        candidate_audio=distorted,
        cutoff_hz=4000.0,
    )
    assert not passes


def test_guard_r_revert_is_reported_as_fail_not_pass() -> None:
    """A total revert must not be published as a passing verdict.

    The strengths ladder always carries a 0.0 candidate that *is* the Natural
    audio. If the guard scored it like any other candidate it would pass every
    layer trivially, and a run where every real candidate was rejected would be
    audited as "passed all Guard R layers".
    """
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    nat_audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # Louder in the protected band: rejected by the protected-band layer, so the
    # verdict carries that layer's measurements rather than a bare structural error.
    broken = (nat_audio * 1.5).astype(np.float32)

    guard = RestorationGuard(sample_rate=sr)
    candidates = [
        RestorationCandidate(strength=1.0, audio=broken, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.5, audio=broken, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.0, audio=nat_audio, cutoff_hz=4000.0),
    ]

    sel_audio, res = guard.select_best_candidate(nat_audio, candidates, cutoff_hz=4000.0)

    assert res.verdict == "FAIL"
    assert res.accepted_strength == 0.0
    assert "reverted to Natural-safe audio" in res.reason
    # The evidence must describe a rejected candidate, not an empty dict.
    assert res.protected_band or res.highband_events
    np.testing.assert_array_equal(sel_audio, nat_audio)
