"""Branch contracts for the restoration DSP stack.

Covers the structural rejection ladder of Guard R, the degenerate-input and
edge branches of the F0 / high-band / bandwidth / protected-band analyzers,
and the fallback paths of HawaRestore-KD (unknown speaker, malformed
embedding, ODE solver failure).
"""

import numpy as np
import pytest
import scipy.signal as signal
import torch

from hawavoclean.restoration.bandwidth import BandwidthDetector
from hawavoclean.restoration.base import RestorationCandidate
from hawavoclean.restoration.f0 import F0Extractor
from hawavoclean.restoration.guard import RestorationGuard
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.highband_events import HighBandEventDetector
from hawavoclean.restoration.linguistic_guard import SoraniLinguisticGuard
from hawavoclean.restoration.protected_band import (
    compute_transition_mask,
    verify_protected_band_invariance,
)
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor

SR = 48000


def _tone(freq_hz: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


@pytest.fixture(scope="module")
def restorer() -> HawaRestoreKD:
    return HawaRestoreKD(sample_rate=SR)


# ---------------------------------------------------------------------------
# Guard R: structural rejection ladder
# ---------------------------------------------------------------------------


def test_guard_rejects_dimension_mismatch() -> None:
    """A stereo candidate for a mono natural must be rejected before any DSP runs."""
    nat = _tone(300.0, 0.02)
    cand = np.stack([nat, nat], axis=0)
    guard = RestorationGuard(sample_rate=SR)
    passes, reason, metrics = guard.evaluate_candidate(nat, cand, cutoff_hz=4000.0)
    assert not passes
    assert "Dimension mismatch" in reason
    assert metrics == {}


def test_guard_rejects_shape_mismatch() -> None:
    """A candidate that dropped samples must be rejected, not silently truncated."""
    nat = _tone(300.0, 0.02)
    guard = RestorationGuard(sample_rate=SR)
    passes, reason, metrics = guard.evaluate_candidate(nat, nat[:-1], cutoff_hz=4000.0)
    assert not passes
    assert "Shape mismatch" in reason
    assert metrics == {}


def test_guard_rejects_empty_candidate() -> None:
    """Zero-length audio can never be an acceptable restoration."""
    empty = np.zeros(0, dtype=np.float32)
    guard = RestorationGuard(sample_rate=SR)
    passes, reason, _ = guard.evaluate_candidate(empty, empty.copy(), cutoff_hz=4000.0)
    assert not passes
    assert "Empty candidate" in reason


def test_guard_rejects_nan_and_inf_candidate() -> None:
    """Non-finite samples must fail closed before reaching the spectral layers."""
    nat = _tone(300.0, 0.02)
    guard = RestorationGuard(sample_rate=SR)

    with_nan = nat.copy()
    with_nan[10] = np.nan
    passes, reason, _ = guard.evaluate_candidate(nat, with_nan, cutoff_hz=4000.0)
    assert not passes
    assert "NaN or Inf" in reason

    with_inf = nat.copy()
    with_inf[20] = np.inf
    passes, reason, _ = guard.evaluate_candidate(nat, with_inf, cutoff_hz=4000.0)
    assert not passes
    assert "NaN or Inf" in reason


def test_guard_rejects_clipping_above_headroom() -> None:
    """Peaks above the 1.05 headroom ceiling are a structural rejection."""
    nat = _tone(300.0, 0.02)
    guard = RestorationGuard(sample_rate=SR)
    passes, reason, _ = guard.evaluate_candidate(
        nat, (nat * 3.0).astype(np.float32), cutoff_hz=4000.0
    )
    assert not passes
    assert "Clipping detected" in reason


def test_guard_zero_norm_canonical_embedding_scores_zero_similarity() -> None:
    """A degenerate all-zero canonical embedding must reject, never divide by zero."""
    nat = _tone(300.0, 0.3)
    guard = RestorationGuard(sample_rate=SR)
    passes, reason, metrics = guard.evaluate_candidate(
        nat,
        nat.copy(),
        cutoff_hz=4000.0,
        canonical_embedding=np.zeros(192, dtype=np.float32),
    )
    assert not passes
    assert "Speaker similarity" in reason
    assert metrics["speaker_similarity"] == 0.0


def test_guard_accepts_identical_candidate_with_speech_mask() -> None:
    """An externally supplied VAD mask must flow through all layers without breaking a pass."""
    nat = (_tone(300.0, 0.3) + _tone(1200.0, 0.3, amp=0.2)).astype(np.float32)
    guard = RestorationGuard(sample_rate=SR)
    speech_mask = np.ones(len(nat) // guard.hf_event_detector.frame_length, dtype=np.float32)
    passes, reason, metrics = guard.evaluate_candidate(
        nat, nat.copy(), cutoff_hz=4000.0, speech_mask=speech_mask
    )
    assert passes
    assert reason == "Passed all Guard R layers"
    assert set(metrics) == {"protected_band", "highband_events", "harmonic", "speaker", "ctc"}


def test_guard_select_best_candidate_no_candidates_is_no_restore() -> None:
    """No candidates and a zero-strength-only ladder must both preserve the Natural audio."""
    nat = _tone(300.0, 0.1)
    guard = RestorationGuard(sample_rate=SR)

    sel, res = guard.select_best_candidate(nat, [], cutoff_hz=4000.0)
    assert sel is nat
    assert res.verdict == "NO_RESTORE"
    assert res.accepted_strength == 0.0
    assert "No restoration candidates" in res.reason

    zero_only = [RestorationCandidate(strength=0.0, audio=nat.copy(), cutoff_hz=4000.0)]
    sel, res = guard.select_best_candidate(nat, zero_only, cutoff_hz=4000.0)
    assert sel is nat
    assert res.verdict == "NO_RESTORE"
    assert "No active restoration candidates" in res.reason


# ---------------------------------------------------------------------------
# F0 extractor
# ---------------------------------------------------------------------------


def test_f0_too_short_input_returns_single_unvoiced_frame() -> None:
    """Audio shorter than one analysis frame yields a defined, all-unvoiced trajectory."""
    traj = F0Extractor(sample_rate=SR).extract(_tone(200.0, 0.02))
    assert traj.f0_hz.shape == (1,)
    assert traj.vuv_mask.shape == (1,)
    assert traj.statistics.median_hz == 0.0
    assert traj.statistics.voiced_fraction == 0.0


def test_f0_silence_is_fully_unvoiced_with_zero_statistics() -> None:
    """Silent frames must be classified unvoiced and produce zeroed statistics."""
    traj = F0Extractor(sample_rate=SR).extract(np.zeros(int(SR * 0.25), dtype=np.float32))
    assert np.all(traj.vuv_mask == 0.0)
    assert np.all(traj.f0_hz == 0.0)
    assert traj.statistics.median_hz == 0.0
    assert traj.statistics.p05_hz == 0.0
    assert traj.statistics.p95_hz == 0.0
    assert traj.statistics.voiced_fraction == 0.0


def test_f0_tracks_pitch_with_tight_statistics() -> None:
    """A steady 150 Hz tone must be tracked as voiced with a tight percentile spread."""
    traj = F0Extractor(sample_rate=SR).extract(_tone(150.0, 0.3))
    assert traj.statistics.voiced_fraction > 0.9
    assert abs(traj.statistics.median_hz - 150.0) < 2.0
    assert traj.statistics.p05_hz <= traj.statistics.median_hz <= traj.statistics.p95_hz
    assert traj.statistics.p95_hz - traj.statistics.p05_hz < 2.0


def test_f0_peak_at_search_boundary_is_still_tracked() -> None:
    """A tone at the upper pitch limit lands on the minimum lag, skipping parabolic refinement.

    600 Hz at 48 kHz is exactly the 80-sample minimum lag, so the interpolation
    guard must fall back to the raw lag instead of reading out of the search range.
    """
    traj = F0Extractor(sample_rate=SR, f0_max_hz=600.0).extract(_tone(600.0, 0.3))
    assert traj.statistics.voiced_fraction > 0.9
    assert traj.statistics.median_hz == pytest.approx(600.0, abs=0.5)
    voiced_f0 = traj.f0_hz[traj.vuv_mask > 0.5]
    assert np.all(voiced_f0 <= 600.0 + 1e-3)


def test_f0_stereo_input_matches_mono_mixdown() -> None:
    """A 2D input must be averaged to mono, giving the exact mono trajectory."""
    mono = _tone(150.0, 0.3)
    stereo = np.stack([mono, mono], axis=0)
    extractor = F0Extractor(sample_rate=SR)
    np.testing.assert_array_equal(extractor.extract(stereo).f0_hz, extractor.extract(mono).f0_hz)


# ---------------------------------------------------------------------------
# High-band event detector
# ---------------------------------------------------------------------------


def test_highband_too_short_input_passes_with_zero_metrics() -> None:
    """Audio below four frames cannot be judged and must pass with neutral metrics."""
    short = np.zeros(1000, dtype=np.float32)
    res = HighBandEventDetector(sample_rate=SR).evaluate(short, short.copy())
    assert res.passes_event_check
    assert res.speech_window_leakage == 0.0
    assert res.spurious_burst_count == 0
    assert res.hf_envelope_divergence == 0.0
    assert res.impulse_discontinuity_ratio == 0.0


def test_highband_spurious_bursts_outside_speech_are_counted_and_rejected() -> None:
    """HF energy injected into masked-out silence must be counted as spurious bursts."""
    n = int(SR * 0.4)
    t = np.arange(n, dtype=np.float32) / SR
    nat = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    rest = nat.copy()
    half = n // 2
    env = np.hanning(half).astype(np.float32)
    rest[:half] += (0.05 * env * np.sin(2 * np.pi * 10000 * t[:half])).astype(np.float32)
    b0, b1 = 12000, 13440  # frames 25-27: inside the masked-out region
    burst_env = np.hanning(b1 - b0).astype(np.float32)
    rest[b0:b1] += (0.4 * burst_env * np.sin(2 * np.pi * 10000 * t[b0:b1])).astype(np.float32)

    detector = HighBandEventDetector(sample_rate=SR)
    mask = np.zeros(n // detector.frame_length, dtype=np.float32)
    mask[:20] = 1.0

    res = detector.evaluate(nat, rest, speech_mask=mask, cutoff_hz=8000.0)
    assert res.spurious_burst_count >= 3
    assert not res.passes_event_check


def test_highband_all_speech_mask_accepts_correlated_hf() -> None:
    """With every frame masked as speech there is no silence to leak into: must pass."""
    n = int(SR * 0.4)
    t = np.arange(n, dtype=np.float32) / SR
    env = (0.55 + 0.45 * np.sin(2 * np.pi * 3.0 * t)).astype(np.float32)
    nat = (env * 0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    rest = nat + (env * 0.04 * np.sin(2 * np.pi * 10000 * t)).astype(np.float32)

    detector = HighBandEventDetector(sample_rate=SR)
    mask = np.ones(n // detector.frame_length, dtype=np.float32)

    res = detector.evaluate(nat, rest, speech_mask=mask, cutoff_hz=8000.0)
    assert res.passes_event_check
    assert res.speech_window_leakage == 0.0
    assert res.spurious_burst_count == 0
    assert res.hf_envelope_divergence < 0.35


def test_highband_no_speech_mask_identical_audio_passes() -> None:
    """An all-silence mask with an unchanged candidate produces no false rejection."""
    nat = _tone(1000.0, 0.4, amp=0.3)
    detector = HighBandEventDetector(sample_rate=SR)
    mask = np.zeros(len(nat) // detector.frame_length, dtype=np.float32)
    res = detector.evaluate(nat, nat.copy(), speech_mask=mask, cutoff_hz=8000.0)
    assert res.passes_event_check
    assert res.speech_window_leakage == 0.0
    assert res.spurious_burst_count == 0


def test_highband_inaudible_residue_is_suppressed_by_absolute_floor() -> None:
    """Microscopic HF residue below -54 dBFS must not blow up the leakage ratio."""
    n = int(SR * 0.4)
    t = np.arange(n, dtype=np.float32) / SR
    nat = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    rest = nat.copy()
    rest[n // 2 :] += (1e-4 * np.sin(2 * np.pi * 10000 * t[n // 2 :])).astype(np.float32)

    detector = HighBandEventDetector(sample_rate=SR)
    mask = np.zeros(n // detector.frame_length, dtype=np.float32)
    mask[:20] = 1.0

    res = detector.evaluate(nat, rest, speech_mask=mask, cutoff_hz=8000.0)
    assert res.speech_window_leakage == 0.0
    assert res.passes_event_check


def test_highband_click_is_rejected_wherever_it_lands() -> None:
    """A click is a local outlier, and it counts anywhere -- not only at an edge.

    The check read exactly two samples of the segment, |rest[0] - nat[0]| and
    |rest[-1] - nat[-1]|, so a click one sample further in was invisible while
    an ordinary endpoint offset failed the whole candidate.
    """
    nat = _tone(1000.0, 0.4, amp=0.3)
    detector = HighBandEventDetector(sample_rate=SR)

    for where in (0, len(nat) // 2, len(nat) - 1):
        rest = nat.copy()
        rest[where] += 0.5
        res = detector.evaluate(nat, rest, cutoff_hz=8000.0)
        assert not res.passes_event_check, f"a 0.5 click at sample {where} was accepted"
        assert res.impulse_discontinuity_ratio > detector.impulse_threshold


def test_highband_accepts_a_restoration_that_only_added_a_high_band() -> None:
    """Added high-frequency content is not a click, and must not read as one.

    At the old threshold the shipped model failed at every non-zero strength
    (0.14-0.21 against 0.08) while the untouched candidate scored exactly
    0.0 -- the layer admitted only the candidate that changes nothing.
    """
    nat = _tone(1000.0, 0.4, amp=0.3)
    rng = np.random.default_rng(7)
    # A dense, quiet high band: every sample moves, no sample jumps.
    hf = signal.sosfiltfilt(
        signal.butter(8, 6000 / (SR / 2), btype="highpass", output="sos"),
        rng.standard_normal(len(nat)) * 0.05,
    )
    rest = (nat + hf).astype(np.float32)

    res = HighBandEventDetector(sample_rate=SR).evaluate(nat, rest, cutoff_hz=8000.0)
    # A literal, not ``detector.impulse_threshold``: comparing the metric to
    # the same bound the code uses would pass no matter what either becomes.
    assert res.impulse_discontinuity_ratio < 8.0, (
        "broadly added high-band content was mistaken for an impulse"
    )


# ---------------------------------------------------------------------------
# Bandwidth detector
# ---------------------------------------------------------------------------


def test_bandwidth_silence_and_too_short_inputs_refuse_restoration() -> None:
    """Silence and sub-window inputs must never recommend restoration."""
    detector = BandwidthDetector(sample_rate=SR)

    silent = detector.detect(np.zeros(int(SR * 0.1), dtype=np.float32))
    assert silent.shape == "silence"
    assert silent.confidence == 1.0
    assert silent.restore_recommended is False
    assert silent.effective_cutoff_hz == detector.max_cutoff_hz

    short = detector.detect(_tone(1000.0, 0.02))  # 960 samples < n_fft
    assert short.shape == "fullband"
    assert short.confidence == 0.5
    assert short.restore_recommended is False


def test_bandwidth_manual_override_is_clipped_to_valid_range() -> None:
    """Manual cutoffs outside [min, max] must be clamped, never trusted verbatim."""
    detector = BandwidthDetector(sample_rate=SR)
    sig = _tone(1000.0, 0.1)

    low = detector.detect(sig, override_cutoff_hz=100.0)
    assert low.effective_cutoff_hz == detector.min_cutoff_hz
    assert low.shape == "manual_override"
    assert low.confidence == 1.0
    assert low.restore_recommended is True

    high = detector.detect(sig, override_cutoff_hz=30000.0)
    assert high.effective_cutoff_hz == detector.max_cutoff_hz
    assert high.shape == "manual_override"


def test_bandwidth_speech_mask_restricts_evidence_to_masked_frames() -> None:
    """A speech mask excluding the full-band half must change what is detected.

    The signal has to be speech-like in both halves: the detector separates a
    filter cliff from spectral tilt, and a spectrum of two pure tones is
    numerical floor everywhere between them, so masking would change which
    fixture artefact is measured rather than which bandwidth is seen.
    """
    rng = np.random.default_rng(5)
    t = np.arange(SR, dtype=np.float32) / SR
    dense = np.zeros_like(t)
    harmonic = 1
    while 130.0 * harmonic < SR / 2:
        dense += (1.0 / (harmonic**1.6)) * np.sin(
            2 * np.pi * 130.0 * harmonic * t + rng.uniform(0, 2 * np.pi)
        )
        harmonic += 1
    dense = (dense / np.max(np.abs(dense))).astype(np.float32)
    # First half band-limited at 4 kHz, second half left full-band.
    limited = signal.sosfiltfilt(
        signal.butter(16, 4000 / (SR / 2), btype="lowpass", output="sos"), dense
    ).astype(np.float32)
    sig = dense.copy()
    sig[: SR // 2] = limited[: SR // 2]

    detector = BandwidthDetector(sample_rate=SR)
    _, _, Zxx = signal.stft(
        sig,
        fs=SR,
        window="hann",
        nperseg=detector.n_fft,
        noverlap=detector.n_fft - detector.hop_length,
        boundary=None,
        padded=False,
    )
    n_frames = Zxx.shape[1]
    limit = (SR // 2 - detector.n_fft) // detector.hop_length + 1
    mask = np.zeros(n_frames, dtype=np.float32)
    mask[:limit] = 1.0

    unmasked = detector.detect(sig)
    masked = detector.detect(sig, speech_mask=mask)

    # Whole file: full-band content is present, so nothing is restored.
    assert unmasked.restore_recommended is False
    # Masked to the band-limited half: the 4 kHz edge becomes visible.
    assert masked.restore_recommended is True
    assert 3600.0 <= masked.effective_cutoff_hz <= 6000.0

    evidence = masked.to_dict()["evidence"]
    assert set(evidence) == {
        "spectral_rolloff",
        "above_cutoff_snr_db",
        "stationarity",
        "high_band_energy_ratio_db",
    }
    assert evidence["stationarity"] >= 0.0


def test_a_pure_tone_is_not_offered_for_restoration() -> None:
    """A signal with no speech structure must not be restored.

    This used to snap to the 2 kHz floor and report a steep brickwall with
    restore_recommended=True, which would have licensed the model to
    synthesise a high band onto a bare sine. Nothing above the tone is a
    filtered-away band; it is simply a signal that has no content there, and
    the detector now declines rather than inventing an edge.
    """
    est = BandwidthDetector(sample_rate=SR).detect(_tone(300.0, 0.3))
    assert est.restore_recommended is False
    assert est.shape == "fullband"


def test_transition_mask_empty_for_nonpositive_bin_count() -> None:
    """Zero frequency bins must yield an empty float32 mask, not an exception."""
    mask = compute_transition_mask(0, SR, 8000.0)
    assert mask.size == 0
    assert mask.dtype == np.float32


def test_transition_mask_zero_width_is_a_binary_step() -> None:
    """With no transition band the mask must be a hard step exactly at the cutoff."""
    n_freqs = 1025
    mask = compute_transition_mask(n_freqs, SR, 8000.0, transition_hz=0.0)
    freqs = np.fft.rfftfreq((n_freqs - 1) * 2, d=1.0 / SR)
    np.testing.assert_array_equal(mask, (freqs >= 8000.0).astype(np.float32))


def test_verify_invariance_empty_audio_passes_trivially() -> None:
    """Degenerate empty inputs must pass with zeroed metrics instead of crashing."""
    empty = np.zeros(0, dtype=np.float32)
    chk = verify_protected_band_invariance(empty, empty.copy(), sample_rate=SR, cutoff_hz=4000.0)
    assert chk.passes_invariance
    assert chk.rms_waveform_error == 0.0
    assert chk.max_waveform_abs_error == 0.0


def test_verify_invariance_short_audio_uses_waveform_only_comparison() -> None:
    """Below one STFT window the check must still catch waveform tampering."""
    orig = _tone(300.0, 0.02)  # 960 samples < n_fft

    same = verify_protected_band_invariance(orig, orig.copy(), sample_rate=SR, cutoff_hz=4000.0)
    assert same.passes_invariance
    assert same.rms_waveform_error == 0.0
    assert same.complex_stft_relative_error == 0.0

    shifted = (orig + 0.01).astype(np.float32)
    diff = verify_protected_band_invariance(orig, shifted, sample_rate=SR, cutoff_hz=4000.0)
    assert not diff.passes_invariance
    assert diff.max_waveform_abs_error == pytest.approx(0.01, abs=1e-6)
    # Fingerprint of the waveform-only path: no STFT metric was computed.
    assert diff.complex_stft_relative_error == 0.0


def test_verify_invariance_floors_protection_at_500hz_for_zero_cutoff() -> None:
    """Even a cutoff of 0 Hz must keep protecting content below the 500 Hz floor."""
    orig = _tone(300.0, 0.3, amp=0.3)

    ok = verify_protected_band_invariance(orig, orig.copy(), sample_rate=SR, cutoff_hz=0.0)
    assert ok.passes_invariance

    tampered = (orig * 1.5).astype(np.float32)
    chk = verify_protected_band_invariance(orig, tampered, sample_rate=SR, cutoff_hz=0.0)
    assert not chk.passes_invariance
    assert chk.rms_waveform_error > 0.01


# ---------------------------------------------------------------------------
# Speaker embedding
# ---------------------------------------------------------------------------


def test_speaker_embed_short_and_silent_inputs_yield_zero_vector() -> None:
    """Inputs with no usable voice content map to the all-zero embedding."""
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)

    short = extractor.extract(_tone(200.0, 0.01))  # 480 samples < 4 hops
    assert short.shape == (192,)
    assert float(np.linalg.norm(short)) == 0.0

    silent = extractor.extract(np.zeros(SR // 4, dtype=np.float32))
    assert float(np.linalg.norm(silent)) == 0.0


def test_speaker_embed_is_unit_norm_and_mixdown_matches_mono() -> None:
    """Real audio yields a unit-norm 192-dim vector; dual-mono equals the mono result."""
    mono = (_tone(220.0, 0.3, amp=0.4) + _tone(1800.0, 0.3, amp=0.2)).astype(np.float32)
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)

    emb = extractor.extract(mono)
    assert emb.shape == (192,)
    assert emb.dtype == np.float32
    assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-5)

    stereo = np.stack([mono, mono], axis=0)
    np.testing.assert_array_equal(extractor.extract(stereo), emb)


# ---------------------------------------------------------------------------
# Linguistic guard
# ---------------------------------------------------------------------------


def test_linguistic_guard_too_short_audio_passes_with_status() -> None:
    """Audio below four hops cannot be judged and must pass with an explicit status."""
    short = np.zeros(500, dtype=np.float32)
    res = SoraniLinguisticGuard(sample_rate=SR).evaluate(short, short.copy())
    assert res.passes_check
    assert res.status == "audio_too_short"
    assert res.divergence == 0.0


def test_linguistic_guard_identical_audio_has_zero_divergence() -> None:
    """An unchanged candidate must measure exactly zero phonetic divergence."""
    nat = (_tone(500.0, 0.3) + _tone(1500.0, 0.3, amp=0.3)).astype(np.float32)
    res = SoraniLinguisticGuard(sample_rate=SR).evaluate(nat, nat.copy())
    assert res.passes_check
    assert res.divergence == 0.0
    assert res.max_frame_divergence == 0.0
    assert res.status == "anchor_preserved"
    d = res.to_dict()
    assert d["passes_check"] is True
    assert d["divergence"] == 0.0


def test_linguistic_guard_mismatched_lengths_compares_common_prefix() -> None:
    """A shorter candidate is evaluated over the shared prefix, not rejected outright."""
    nat = (_tone(500.0, 0.3) + _tone(1500.0, 0.3, amp=0.3)).astype(np.float32)
    res = SoraniLinguisticGuard(sample_rate=SR).evaluate(nat, nat[: len(nat) // 2])
    assert res.passes_check
    assert res.divergence < 0.05


# ---------------------------------------------------------------------------
# HawaRestore-KD fallback branches
# ---------------------------------------------------------------------------


def test_hawarestore_empty_audio_returns_default_ladder_with_clipped_cutoff(
    restorer: HawaRestoreKD,
) -> None:
    """Empty input returns the full default strengths ladder and clamps the cutoff."""
    cands = restorer.restore(
        np.zeros(0, dtype=np.float32), sample_rate=SR, effective_cutoff_hz=100.0
    )
    assert [c.strength for c in cands] == [1.0, 0.75, 0.5, 0.25, 0.0]
    for cand in cands:
        assert cand.audio.size == 0
        assert cand.cutoff_hz == 500.0  # clipped up from 100 Hz


def test_hawarestore_unknown_speaker_and_bad_embedding_are_ignored(
    restorer: HawaRestoreKD,
) -> None:
    """An unknown speaker_id and a wrong-length embedding must not change the output."""
    sig = (_tone(440.0, 0.15, amp=0.4) + _tone(2500.0, 0.15, amp=0.2)).astype(np.float32)

    baseline = restorer.restore(
        sig, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0, 0.0], seed=11
    )
    conditioned = restorer.restore(
        sig,
        sample_rate=SR,
        effective_cutoff_hz=4000.0,
        speaker_id="character_99",  # not in the roster
        speaker_embedding=np.ones(7, dtype=np.float32),  # not 192-dim
        strengths=[1.0, 0.0],
        seed=11,
    )

    for base_cand, cond_cand in zip(baseline, conditioned, strict=True):
        assert base_cand.strength == cond_cand.strength
        np.testing.assert_array_equal(base_cand.audio, cond_cand.audio)
        assert np.all(np.isfinite(cond_cand.audio))


def test_hawarestore_ode_failure_falls_back_to_dsp_extrapolation(
    restorer: HawaRestoreKD, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashing ODE solver must degrade to tiled DSP extrapolation, not fail the job."""

    def _boom(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("simulated ODE solver failure")

    monkeypatch.setattr(HawaRestoreKD, "_solve_flow_ode", _boom)

    sig = (_tone(440.0, 0.25, amp=0.4) + _tone(3800.0, 0.25, amp=0.25)).astype(np.float32)
    cands = restorer.restore(
        sig, sample_rate=SR, effective_cutoff_hz=4000.0, strengths=[1.0, 0.0], seed=3
    )

    restored = next(c.audio for c in cands if c.strength == 1.0)
    passthrough = next(c.audio for c in cands if c.strength == 0.0)

    np.testing.assert_array_equal(passthrough, sig)
    assert restored.shape == sig.shape
    assert np.all(np.isfinite(restored))

    sos = signal.butter(6, 6000.0, btype="highpass", fs=SR, output="sos")
    hf_in = float(np.sqrt(np.mean(signal.sosfiltfilt(sos, sig) ** 2)))
    hf_out = float(np.sqrt(np.mean(signal.sosfiltfilt(sos, restored) ** 2)))
    assert hf_out > 1e-4
    assert hf_out > 50.0 * hf_in
