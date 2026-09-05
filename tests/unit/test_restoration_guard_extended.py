"""Extended branch coverage tests for hawavoclean.restoration.guard."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from hawavoclean.restoration.guard import (
    RestorationGuard,
)
from hawavoclean.restoration.highband_events import HighBandEventResult
from hawavoclean.restoration.protected_band import ProtectedBandVerification


@dataclass
class _Candidate:
    strength: float
    audio: np.ndarray


@dataclass
class _SegmentRecord:
    action: str
    applied_strength: float = 0.5


def test_select_best_candidate_strength_levels_and_empty() -> None:
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 300 * t)

    # 1. Empty candidate list
    _, res_empty = guard.select_best_candidate(
        natural_audio=audio,
        candidates=[],
        cutoff_hz=4000.0,
    )
    assert res_empty.verdict == "NO_RESTORE"
    assert res_empty.accepted_strength == 0.0

    # 2. Only zero-strength candidates
    cand_zero = _Candidate(strength=0.0, audio=audio)
    _, res_zero = guard.select_best_candidate(
        natural_audio=audio,
        candidates=[cand_zero],
        cutoff_hz=4000.0,
    )
    assert res_zero.verdict == "NO_RESTORE"

    # 3. PASS verdict (strength >= 0.75) vs WARN verdict (strength < 0.75)
    cand_warn = _Candidate(strength=0.6, audio=audio)
    with patch.object(guard, "evaluate_candidate", return_value=(True, "ok", {})):
        _, res_warn = guard.select_best_candidate(
            natural_audio=audio,
            candidates=[cand_warn],
            cutoff_hz=4000.0,
        )
        assert res_warn.verdict == "WARN"
        assert res_warn.accepted_strength == 0.6

    cand_pass = _Candidate(strength=0.85, audio=audio)
    with patch.object(guard, "evaluate_candidate", return_value=(True, "ok", {})):
        _, res_pass = guard.select_best_candidate(
            natural_audio=audio,
            candidates=[cand_pass],
            cutoff_hz=4000.0,
        )
        assert res_pass.verdict == "PASS"
        assert res_pass.accepted_strength == 0.85


def test_evaluate_candidate_zero_norms_and_variance() -> None:
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 300 * t)

    # 1. Canonical embedding with zero norm
    zero_embed = np.zeros(192, dtype=np.float32)
    var_vec = np.ones(192, dtype=np.float32)

    with patch.object(
        guard.speaker_extractor, "extract", return_value=np.zeros(192, dtype=np.float32)
    ):
        passes, reason, metrics = guard.evaluate_candidate(
            natural_audio=audio,
            candidate_audio=audio,
            cutoff_hz=4000.0,
            canonical_embedding=zero_embed,
            variance_vector=var_vec,
        )
        assert not passes
        assert "Speaker similarity" in reason


def test_verify_post_mastering_failure_branches() -> None:
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    seg_len = 1000
    n_samples = 4000
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    nat_audio = 0.2 * np.sin(2 * np.pi * 400 * t)
    rest_audio = 0.2 * np.sin(2 * np.pi * 400 * t)

    starts = [0, 1000, 2000]

    # 1. Segment index exceeds len(starts)
    records_bad_index = [
        _SegmentRecord(action="restored"),
        _SegmentRecord(action="restored"),
        _SegmentRecord(action="restored"),
        _SegmentRecord(action="restored"),  # index 3 >= len(starts)
    ]
    res_bad = guard.verify_post_mastering(
        mastered_natural=nat_audio,
        mastered_restored=rest_audio,
        segment_records=records_bad_index,
        starts=starts,
        seg_len=seg_len,
        cutoff_hz=4000.0,
    )
    assert not res_bad.passes
    assert res_bad.fallback_applied
    assert any("start index missing" in r for r in [res_bad.reason])

    # 2. Protected band invariance violations
    rec = [_SegmentRecord(action="restored")]
    starts_single = [0]

    # 2a. RMS violation
    mock_prot_rms = ProtectedBandVerification(
        passes_invariance=False,
        max_waveform_abs_error=0.1,
        rms_waveform_error=0.05,
        complex_stft_relative_error=0.0001,
        max_phase_deviation_rad=0.01,
        worst_band_energy_deviation_db=0.1,
        worst_band_center_hz=1000.0,
    )
    with patch(
        "hawavoclean.restoration.guard.verify_protected_band_invariance", return_value=mock_prot_rms
    ):
        res_rms = guard.verify_post_mastering(
            mastered_natural=nat_audio,
            mastered_restored=rest_audio,
            segment_records=rec,
            starts=starts_single,
            seg_len=seg_len,
            cutoff_hz=4000.0,
            tolerance_rms=1e-4,
        )
        assert not res_rms.passes
        assert "RMS" in res_rms.segments_evidence[0].reason

    # 2b. STFT violation
    mock_prot_stft = ProtectedBandVerification(
        passes_invariance=False,
        max_waveform_abs_error=0.001,
        rms_waveform_error=1e-5,
        complex_stft_relative_error=0.05,
        max_phase_deviation_rad=0.01,
        worst_band_energy_deviation_db=0.1,
        worst_band_center_hz=1000.0,
    )
    with patch(
        "hawavoclean.restoration.guard.verify_protected_band_invariance",
        return_value=mock_prot_stft,
    ):
        res_stft = guard.verify_post_mastering(
            mastered_natural=nat_audio,
            mastered_restored=rest_audio,
            segment_records=rec,
            starts=starts_single,
            seg_len=seg_len,
            cutoff_hz=4000.0,
            tolerance_stft=1e-3,
        )
        assert not res_stft.passes
        assert "STFT" in res_stft.segments_evidence[0].reason

    # 2c. 1/3-octave violation
    mock_prot_oct = ProtectedBandVerification(
        passes_invariance=False,
        max_waveform_abs_error=0.001,
        rms_waveform_error=1e-5,
        complex_stft_relative_error=1e-4,
        max_phase_deviation_rad=0.01,
        worst_band_energy_deviation_db=1.5,
        worst_band_center_hz=1000.0,
    )
    with patch(
        "hawavoclean.restoration.guard.verify_protected_band_invariance", return_value=mock_prot_oct
    ):
        res_oct = guard.verify_post_mastering(
            mastered_natural=nat_audio,
            mastered_restored=rest_audio,
            segment_records=rec,
            starts=starts_single,
            seg_len=seg_len,
            cutoff_hz=4000.0,
            tolerance_third_octave_db=0.25,
        )
        assert not res_oct.passes
        assert "1/3-octave" in res_oct.segments_evidence[0].reason

    # 3. High-band event inconsistency
    mock_prot_pass = ProtectedBandVerification(
        passes_invariance=True,
        max_waveform_abs_error=0.001,
        rms_waveform_error=1e-5,
        complex_stft_relative_error=1e-4,
        max_phase_deviation_rad=0.01,
        worst_band_energy_deviation_db=0.1,
        worst_band_center_hz=1000.0,
    )
    mock_hf_fail = HighBandEventResult(
        speech_window_leakage=0.5,
        spurious_burst_count=2,
        hf_envelope_divergence=0.8,
        impulse_discontinuity_ratio=15.0,
        passes_event_check=False,
    )
    with (
        patch(
            "hawavoclean.restoration.guard.verify_protected_band_invariance",
            return_value=mock_prot_pass,
        ),
        patch.object(guard.hf_event_detector, "evaluate", return_value=mock_hf_fail),
    ):
        res_hf = guard.verify_post_mastering(
            mastered_natural=nat_audio,
            mastered_restored=rest_audio,
            segment_records=rec,
            starts=starts_single,
            seg_len=seg_len,
            cutoff_hz=4000.0,
        )
        assert not res_hf.passes
        assert "high-band event inconsistency" in res_hf.segments_evidence[0].reason

    # 4. Speaker similarity failure post-mastering
    mock_hf_pass = HighBandEventResult(
        speech_window_leakage=0.05,
        spurious_burst_count=0,
        hf_envelope_divergence=0.1,
        impulse_discontinuity_ratio=1.0,
        passes_event_check=True,
    )
    cand_emb_dissimilar = np.ones(192, dtype=np.float32)
    source_emb = -np.ones(192, dtype=np.float32)
    with (
        patch(
            "hawavoclean.restoration.guard.verify_protected_band_invariance",
            return_value=mock_prot_pass,
        ),
        patch.object(guard.hf_event_detector, "evaluate", return_value=mock_hf_pass),
        patch.object(
            guard.speaker_extractor, "extract", side_effect=[cand_emb_dissimilar, source_emb]
        ),
    ):
        res_spk = guard.verify_post_mastering(
            mastered_natural=nat_audio,
            mastered_restored=rest_audio,
            segment_records=rec,
            starts=starts_single,
            seg_len=seg_len,
            cutoff_hz=4000.0,
        )
        assert not res_spk.passes
        assert "speaker similarity" in res_spk.segments_evidence[0].reason
