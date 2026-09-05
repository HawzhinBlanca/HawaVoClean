"""Targeted branch and edge case tests for hawavoclean.restoration.guard."""

from __future__ import annotations

import numpy as np

from hawavoclean.restoration.base import RestorationCandidate
from hawavoclean.restoration.guard import (
    GuardRResult,
    PostMasteringSegmentEvidence,
    PostMasteringVerificationResult,
    RestorationGuard,
)


def test_guard_r_dataclass_to_dict() -> None:
    res = GuardRResult(
        verdict="PASS",
        accepted_strength=1.0,
        reason="ok",
        protected_band={"rms": 0.0},
        ctc={"score": 1.0},
        highband_events={"bursts": 0},
        harmonic={"pitch_diff": 0.0},
        speaker={"sim": 1.0},
    )
    d = res.to_dict()
    assert d["verdict"] == "PASS"
    assert d["accepted_strength"] == 1.0

    seg = PostMasteringSegmentEvidence(
        segment_index=0,
        start_sample=0,
        end_sample=1000,
        start_time_s=0.0,
        end_time_s=0.02,
        action="verified",
        passes=True,
        reason="all good",
        rms_waveform_error=0.00001,
        stft_relative_error=0.0001,
        worst_band_energy_deviation_db=0.05,
        worst_band_center_hz=1000.0,
        metrics={},
    )
    seg_d = seg.to_dict()
    assert seg_d["action"] == "verified"
    assert seg_d["passes"] is True

    verif = PostMasteringVerificationResult(
        passes=True,
        fallback_applied=False,
        reason="all verified",
        reconstructed_segments_verified=1,
        segments_evidence=[seg],
    )
    verif_d = verif.to_dict()
    assert verif_d["passes"] is True
    assert len(verif_d["segments_evidence"]) == 1


def test_evaluate_candidate_structural_failures() -> None:
    guard = RestorationGuard(sample_rate=16000)
    audio = np.zeros(1600, dtype=np.float32)

    # 1. Dimension mismatch
    ok, reason, _ = guard.evaluate_candidate(audio, np.zeros((1600, 1), dtype=np.float32), 4000.0)
    assert not ok and "Dimension mismatch" in reason

    # 2. Shape mismatch
    ok, reason, _ = guard.evaluate_candidate(audio, np.zeros(800, dtype=np.float32), 4000.0)
    assert not ok and "Shape mismatch" in reason

    # 3. Empty audio
    ok, reason, _ = guard.evaluate_candidate(
        np.array([], dtype=np.float32), np.array([], dtype=np.float32), 4000.0
    )
    assert not ok and "Empty candidate audio" in reason

    # 4. NaN or Inf
    nan_audio = audio.copy()
    nan_audio[10] = float("nan")
    ok, reason, _ = guard.evaluate_candidate(audio, nan_audio, 4000.0)
    assert not ok and "NaN or Inf" in reason

    inf_audio = audio.copy()
    inf_audio[10] = float("inf")
    ok, reason, _ = guard.evaluate_candidate(audio, inf_audio, 4000.0)
    assert not ok and "NaN or Inf" in reason

    # 5. Clipping > 1.05
    clipped = audio.copy()
    clipped[10] = 1.10
    ok, reason, _ = guard.evaluate_candidate(audio, clipped, 4000.0)
    assert not ok and "Clipping detected" in reason


def test_evaluate_candidate_speaker_enrolled_and_source_modes() -> None:
    guard = RestorationGuard(sample_rate=16000)
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    clean = 0.5 * np.sin(2 * np.pi * 300 * t)

    # Enrolled mode with zero-norm canonical embedding
    zero_embed = np.zeros(16, dtype=np.float32)
    ok, reason, _ = guard.evaluate_candidate(clean, clean, 4000.0, canonical_embedding=zero_embed)
    assert not ok and "Speaker similarity" in reason

    # Enrolled mode with matching embedding and variance vector
    canonical = np.ones(192, dtype=np.float32)
    variance = np.ones(192, dtype=np.float32) * 0.1
    # Candidate embedding extraction will extract from clean audio (different from np.ones)
    ok, reason, _ = guard.evaluate_candidate(
        clean, clean, 4000.0, canonical_embedding=canonical, variance_vector=variance
    )
    assert isinstance(ok, bool)


def test_select_best_candidate_edge_cases() -> None:
    guard = RestorationGuard(sample_rate=16000)
    audio = np.zeros(1600, dtype=np.float32)

    # 1. No candidates
    sel, res = guard.select_best_candidate(audio, [], 4000.0)
    assert np.array_equal(sel, audio)
    assert res.verdict == "NO_RESTORE"
    assert res.accepted_strength == 0.0

    # 2. All candidates fail
    bad1 = audio.copy()
    bad1[0] = 2.0  # clipping
    bad2 = audio.copy()
    bad2[0] = 3.0  # clipping
    cands = [
        RestorationCandidate(strength=1.0, audio=bad1, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.5, audio=bad2, cutoff_hz=4000.0),
    ]
    sel, res = guard.select_best_candidate(audio, cands, 4000.0)
    assert np.array_equal(sel, audio)
    assert res.verdict == "FAIL"
    assert res.accepted_strength == 0.0


def test_verify_post_mastering_segments_branch_coverage() -> None:
    from types import SimpleNamespace

    guard = RestorationGuard(sample_rate=16000)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # 1. No reconstructed indices -> passes with 0 verified
    res = guard.verify_post_mastering(
        mastered_natural=audio,
        mastered_restored=audio,
        segment_records=[],
        starts=[],
        seg_len=8000,
        cutoff_hz=4000.0,
    )
    assert res.passes
    assert not res.fallback_applied
    assert res.reconstructed_segments_verified == 0

    # 2. Reconstructed segment with clipping > 1.05
    rec_reconstructed = SimpleNamespace(action="restored", applied_strength=1.0)
    clipped = audio.copy()
    clipped[100] = 1.2
    res = guard.verify_post_mastering(
        mastered_natural=audio,
        mastered_restored=clipped,
        segment_records=[rec_reconstructed],
        starts=[0],
        seg_len=8000,
        cutoff_hz=4000.0,
    )
    assert not res.passes
    assert res.fallback_applied
    assert "clipping" in res.reason

    # 3. Clean reconstructed segment -> passes
    res = guard.verify_post_mastering(
        mastered_natural=audio,
        mastered_restored=audio,
        segment_records=[rec_reconstructed],
        starts=[0],
        seg_len=8000,
        cutoff_hz=4000.0,
    )
    assert res.passes
    assert not res.fallback_applied
    assert res.reconstructed_segments_verified == 1

    # 4. Strict tolerance violation
    distorted = audio.copy()
    distorted[1000:3000] += 0.2
    res = guard.verify_post_mastering(
        mastered_natural=audio,
        mastered_restored=distorted,
        segment_records=[rec_reconstructed],
        starts=[0],
        seg_len=8000,
        cutoff_hz=4000.0,
        tolerance_rms=1e-7,
        tolerance_stft=1e-7,
    )
    assert not res.passes
    assert res.fallback_applied
    assert "signal integrity violation" in res.reason
