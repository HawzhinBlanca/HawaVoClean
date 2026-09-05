"""Unit tests for Smart Safe internal preview generation and hard guards (I3.2, I3.3)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.restoration.profiles import SpeakerF0Stats, SpeakerProfile
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from hawavoclean.smart_safe import (
    AcousticEvidence,
    CandidateEvidence,
    CandidateOutcome,
    CandidatePreview,
    SmartSafePreviewEngine,
    SmartSafeRanker,
    abstain_to_least_intervention,
    compute_evidence_sha256,
    decide_smart_safe,
    extract_preview_slice,
    verify_candidate_evidence_integrity,
    verify_post_master_invariants,
)

pytestmark = pytest.mark.unit


def _ranker(*, qualified: bool = False) -> SmartSafeRanker:
    return SmartSafeRanker(
        version="ranker-preview-test-v1",
        artifact_sha256=hashlib.sha256(b"preview-ranker").hexdigest(),
        signed=bool(qualified),
        qualified=qualified,
    )


def _evidence(**changes: object) -> AcousticEvidence:
    values: dict[str, object] = {
        "speech_dominance": 0.95,
        "music_risk": 0.02,
        "crosstalk_risk": 0.02,
        "rumble_confidence": 0.88,
        "band_limited_confidence": 0.92,
        "recorded_high_frequency_speech_confidence": 0.01,
        "speaker_match_confidence": 0.96,
        "speaker_match_verified": True,
        "reconstruction_consent": True,
    }
    values.update(changes)
    return AcousticEvidence(**values)  # type: ignore[arg-type]


def _make_speech_signal(duration_s: float = 1.0, sr: int = 48000) -> np.ndarray:
    """Deterministic synthetic voiced speech signal with speech envelope."""
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0.0, 1.0).astype(np.float32)
    harmonics = (
        0.30 * np.sin(2 * np.pi * 160.0 * t)
        + 0.20 * np.sin(2 * np.pi * 320.0 * t)
        + 0.15 * np.sin(2 * np.pi * 480.0 * t)
        + 0.10 * np.sin(2 * np.pi * 640.0 * t)
        + 0.05 * np.sin(2 * np.pi * 1280.0 * t)
    )
    return np.ascontiguousarray(env * harmonics, dtype=np.float32)


def _make_speaker_profile(audio: np.ndarray, sr: int = 48000) -> SpeakerProfile:
    extractor = SpeakerEmbeddingExtractor(sample_rate=sr)
    emb = extractor.extract(audio)
    return SpeakerProfile(
        schema_version="1.0",
        speaker_id="test_spk_01",
        display_name="Test Speaker",
        consent_record="consent.json",
        canonical_audio_manifest="manifest.jsonl",
        canonical_audio_sha256=["a" * 64],
        profile_embedding_path="profile.npy",
        profile_embedding_sha256="b" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=160.0, p05_hz=100.0, p95_hz=220.0),
        training_split_id="test",
        adapter=None,
        created_by_commit="0123456",
        notes="Preview test fixture profile",
        embedding_vector=emb,
        status="active",
        embedding_dim=len(emb),
    )


def test_all_seven_routes_generate_bounded_previews() -> None:
    """I3.2: Internal engine produces bounded previews for all 7 routes."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    profile = _make_speaker_profile(audio, sr=sr)

    engine = SmartSafePreviewEngine(
        max_preview_seconds=2.0,
        sample_rate=sr,
        cutoff_hz=8000.0,
        allow_research_restore=True,
    )

    previews = engine.generate_all_previews(
        audio,
        sr,
        acoustic_evidence=evidence,
        restore_policy="auto",
        speaker_profile=profile,
        speaker_profile_id="test_spk_01",
    )

    assert len(previews) == 7
    expected_routes = {
        "preserve",
        "production",
        "studio",
        "lowband",
        "lowband_then_production",
        "restore_source",
        "restore_enrolled",
    }
    assert set(previews.keys()) == expected_routes

    for route, preview in previews.items():
        assert preview.route == route
        assert preview.sample_rate == sr
        assert 0 < len(preview.audio) <= int(2.0 * sr)
        assert len(preview.audio_sha256) == 64
        assert preview.audio_sha256 == hashlib.sha256(preview.audio.tobytes()).hexdigest()

        ev = preview.evidence
        assert ev.route == route
        assert 1.0 <= ev.predicted_quality_mos <= 5.0
        assert 0.0 <= ev.prediction_confidence <= 1.0
        assert ev.evidence_sha256 is not None
        assert len(ev.evidence_sha256) == 64

        # Evidence is internally validated
        assert verify_candidate_evidence_integrity(preview) is True


def test_caller_cannot_inject_synthetic_guard_booleans() -> None:
    """I3.2: Callers cannot alter internal evidence or supply arbitrary guard booleans."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    preview = engine.generate_preview(
        "production",
        audio,
        sr,
        acoustic_evidence=evidence,
    )
    assert verify_candidate_evidence_integrity(preview) is True

    # 1. Caller tampers with content_guard_passed boolean
    tampered_evidence = replace(preview.evidence, content_guard_passed=False)
    tampered_preview = CandidatePreview(
        route=preview.route,
        audio=preview.audio,
        sample_rate=preview.sample_rate,
        duration_s=preview.duration_s,
        audio_sha256=preview.audio_sha256,
        evidence=tampered_evidence,
    )
    assert verify_candidate_evidence_integrity(tampered_preview) is False

    # 2. Caller injects an arbitrary inflated MOS score
    tampered_evidence_mos = replace(preview.evidence, predicted_quality_mos=4.99)
    tampered_preview_mos = CandidatePreview(
        route=preview.route,
        audio=preview.audio,
        sample_rate=preview.sample_rate,
        duration_s=preview.duration_s,
        audio_sha256=preview.audio_sha256,
        evidence=tampered_evidence_mos,
    )
    assert verify_candidate_evidence_integrity(tampered_preview_mos) is False

    # 3. Caller swaps the audio waveform with unverified data
    tampered_audio = preview.audio.copy()
    tampered_audio[0] += 0.1
    tampered_preview_audio = CandidatePreview(
        route=preview.route,
        audio=tampered_audio,
        sample_rate=preview.sample_rate,
        duration_s=preview.duration_s,
        audio_sha256=preview.audio_sha256,
        evidence=preview.evidence,
    )
    assert verify_candidate_evidence_integrity(tampered_preview_audio) is False


def test_content_guard_rejection_blocks_candidate_before_ranking() -> None:
    """I3.3: Content guard rejection excludes candidate before ranking."""
    sr = 48000
    ref_audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    ranker = _ranker(qualified=False)

    # Candidate with unnatural linguistic corruption
    corrupted_audio = np.zeros_like(ref_audio)
    corrupted_audio[::4] = 0.8  # Harsh impulse train causing high divergence

    engine = SmartSafePreviewEngine(allow_research_restore=True)
    valid_preview = engine.generate_preview("preserve", ref_audio, sr, acoustic_evidence=evidence)

    # Compute genuine sha256 for corrupted audio
    corr_audio_sha = hashlib.sha256(corrupted_audio.tobytes()).hexdigest()
    corr_evidence_sha = compute_evidence_sha256(
        route="production",
        audio_sha256=corr_audio_sha,
        predicted_quality_mos=1.5,
        prediction_confidence=0.95,
        content_guard_passed=False,  # Failed!
        speaker_guard_passed=True,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
        reconstruction_disclosed=False,
    )
    corrupted_evidence = CandidateEvidence(
        route="production",
        predicted_quality_mos=1.5,
        prediction_confidence=0.95,
        content_guard_passed=False,
        speaker_guard_passed=True,
        protected_band_guard_passed=True,
        artifact_guard_passed=True,
        post_master_guard_passed=True,
        reconstruction_disclosed=False,
        evidence_sha256=corr_evidence_sha,
    )

    decision = decide_smart_safe(
        evidence,
        [valid_preview.evidence, corrupted_evidence],
        restore_policy="disabled",
        speaker_profile_id=None,
        ranker=ranker,
        require_qualified_ranker=False,
    )

    # Corrupted candidate must be safe=False and cannot win
    outcomes = {c.route: c for c in decision.candidates}
    assert outcomes["production"].safe is False
    assert "content guard failed" in outcomes["production"].reasons
    assert decision.selected_route == "preserve"


def test_speaker_guard_rejection_blocks_wrong_profile() -> None:
    """I3.3: Mismatched speaker profile fails speaker guard and is excluded."""
    sr = 48000
    audio_speaker_a = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence(reconstruction_consent=True, band_limited_confidence=0.95)

    # Profile with completely orthogonal/different speaker embedding
    different_emb = np.zeros(192, dtype=np.float32)
    different_emb[0] = 1.0  # Unit vector on first axis
    mismatched_profile = SpeakerProfile(
        schema_version="1.0",
        speaker_id="mismatched_spk",
        display_name="Mismatched",
        consent_record="consent.json",
        canonical_audio_manifest="manifest.jsonl",
        canonical_audio_sha256=["c" * 64],
        profile_embedding_path="profile.npy",
        profile_embedding_sha256="d" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=250.0, p05_hz=180.0, p95_hz=350.0),
        training_split_id="test",
        adapter=None,
        created_by_commit="0123456",
        notes="Mismatched speaker fixture",
        embedding_vector=different_emb,
        status="active",
        embedding_dim=192,
    )

    engine = SmartSafePreviewEngine(allow_research_restore=True)
    preview = engine.generate_preview(
        "restore_enrolled",
        audio_speaker_a,
        sr,
        acoustic_evidence=evidence,
        speaker_profile=mismatched_profile,
        speaker_profile_id="mismatched_spk",
    )

    # Speaker guard must fail because audio_speaker_a embedding does not match different_emb
    assert preview.evidence.speaker_guard_passed is False

    ranker = _ranker(qualified=False)
    preserve_prev = engine.generate_preview(
        "preserve", audio_speaker_a, sr, acoustic_evidence=evidence
    )
    decision = decide_smart_safe(
        evidence,
        [preserve_prev.evidence, preview.evidence],
        restore_policy="enrolled_only",
        speaker_profile_id="mismatched_spk",
        ranker=ranker,
        require_qualified_ranker=False,
    )

    outcomes = {c.route: c for c in decision.candidates}
    assert outcomes["restore_enrolled"].safe is False
    assert "speaker guard failed" in outcomes["restore_enrolled"].reasons
    assert decision.selected_route == "preserve"

    # Zero-norm profile embedding also fails speaker guard in preview generation
    zero_prof = replace(mismatched_profile, embedding_vector=np.zeros(192, dtype=np.float32))
    prev_zero = engine.generate_preview(
        "restore_enrolled",
        audio_speaker_a,
        sr,
        acoustic_evidence=evidence,
        speaker_profile=zero_prof,
        speaker_profile_id="zero_spk",
    )
    assert prev_zero.evidence.speaker_guard_passed is False


def test_research_restoration_quarantine_in_preview_engine() -> None:
    """R2.1 / I3.2: Research restoration routes fail closed when quarantine is active."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence(reconstruction_consent=True)

    # Default engine: allow_research_restore=False
    engine = SmartSafePreviewEngine(allow_research_restore=False)

    prev_source = engine.generate_preview(
        "restore_source",
        audio,
        sr,
        acoustic_evidence=evidence,
    )
    assert prev_source.evidence.content_guard_passed is False
    assert prev_source.evidence.speaker_guard_passed is False
    assert prev_source.evidence.protected_band_guard_passed is False
    assert prev_source.evidence.artifact_guard_passed is False
    assert prev_source.evidence.post_master_guard_passed is False

    prev_enrolled = engine.generate_preview(
        "restore_enrolled",
        audio,
        sr,
        acoustic_evidence=evidence,
    )
    assert prev_enrolled.evidence.post_master_guard_passed is False


def test_post_master_failure_abstains_to_least_intervention() -> None:
    """I3.3: Post-master failure falls back down the intervention ladder to least-modified result."""
    sr = 48000
    ref_audio = _make_speech_signal(duration_s=0.5, sr=sr)

    # 1. Master with severe clipping (peak > 1.05)
    clipped_master = ref_audio.copy()
    clipped_master[0] = 1.50
    passed, reason = verify_post_master_invariants(
        clipped_master, ref_audio, route="studio", sample_rate=sr
    )
    assert passed is False
    assert "clipping detected" in reason

    # 2. Master with NaN
    nan_master = ref_audio.copy()
    nan_master[10] = np.nan
    passed_nan, reason_nan = verify_post_master_invariants(
        nan_master, ref_audio, route="production", sample_rate=sr
    )
    assert passed_nan is False
    assert "NaN or Inf" in reason_nan

    # 3. Abstention fallback down the intervention ladder
    # Scenario: studio was selected, but failed post-master check.
    # Surviving candidates: preserve (cost 0), production (cost 1), studio (cost 2)
    candidates: list[CandidateOutcome] = [
        CandidateOutcome("preserve", eligible=True, safe=True, reasons=()),
        CandidateOutcome("production", eligible=True, safe=True, reasons=()),
        CandidateOutcome("studio", eligible=True, safe=True, reasons=()),
    ]
    # Dropping failed 'studio' route must fall back to lowest cost surviving safe candidate: 'preserve' vs 'production' -> 'preserve'
    safest = abstain_to_least_intervention(candidates, failed_route="studio")
    assert safest == "preserve"

    # If only production and studio survived and failed route is None:
    candidates_no_pres: list[CandidateOutcome] = [
        CandidateOutcome("production", eligible=True, safe=True, reasons=()),
        CandidateOutcome("studio", eligible=True, safe=True, reasons=()),
    ]
    safest_prod = abstain_to_least_intervention(candidates_no_pres, failed_route="studio")
    assert safest_prod == "production"

    # If all non-preserve candidates fail:
    empty_candidates: list[CandidateOutcome] = [
        CandidateOutcome("studio", eligible=True, safe=True, reasons=()),
    ]
    fallback_pres = abstain_to_least_intervention(empty_candidates, failed_route="studio")
    assert fallback_pres == "preserve"


def test_reproducibility_and_audio_sha256() -> None:
    """I3.2: Two runs on identical input produce bit-identical audio and digests."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()

    engine1 = SmartSafePreviewEngine(allow_research_restore=True)
    engine2 = SmartSafePreviewEngine(allow_research_restore=True)

    prev1 = engine1.generate_preview("production", audio, sr, acoustic_evidence=evidence)
    prev2 = engine2.generate_preview("production", audio, sr, acoustic_evidence=evidence)

    assert np.array_equal(prev1.audio, prev2.audio)
    assert prev1.audio_sha256 == prev2.audio_sha256
    assert prev1.evidence.evidence_sha256 == prev2.evidence.evidence_sha256
    assert prev1.evidence.predicted_quality_mos == prev2.evidence.predicted_quality_mos


def test_decide_convenience_method() -> None:
    """End-to-end SmartSafePreviewEngine.decide produces consistent outcomes."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    ranker = _ranker(qualified=False)
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    decision, previews = engine.decide(
        audio,
        sr,
        acoustic_evidence=evidence,
        restore_policy="disabled",
        ranker=ranker,
        require_qualified_ranker=False,
    )

    assert decision.selected_route in previews
    assert decision.decision_sha256 is not None
    assert len(decision.candidates) == 7


def test_only_eligible_flag_in_preview_engine() -> None:
    """When only_eligible=True, disqualified routes are skipped."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    # Restore routes ineligible (reconstruction_consent=False)
    evidence = _evidence(reconstruction_consent=False)
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    previews = engine.generate_all_previews(
        audio,
        sr,
        acoustic_evidence=evidence,
        restore_policy="disabled",
        only_eligible=True,
    )

    # restore_source and restore_enrolled must not be in previews
    assert "restore_source" not in previews
    assert "restore_enrolled" not in previews
    assert "preserve" in previews
    assert "production" in previews


def test_extract_preview_slice_stereo_and_long_audio() -> None:
    """extract_preview_slice handles stereo input and bounds duration to max_duration_s."""
    sr = 48000
    long_stereo = np.ones((2, sr * 3), dtype=np.float32)
    sliced = extract_preview_slice(long_stereo, sr, max_duration_s=1.5, target_sample_rate=sr)

    assert sliced.ndim == 1
    assert len(sliced) == int(1.5 * sr)
    assert np.allclose(sliced, 1.0)


def test_extract_preview_slice_transposed_stereo_and_resampling() -> None:
    """extract_preview_slice handles (N, 2) stereo input and sample rate conversion."""
    sr_in = 24000
    sr_out = 48000
    transposed_stereo = np.ones((sr_in * 2, 2), dtype=np.float32)
    sliced = extract_preview_slice(
        transposed_stereo, sr_in, max_duration_s=1.0, target_sample_rate=sr_out
    )

    assert sliced.ndim == 1
    assert len(sliced) == sr_out


def test_unknown_route_raises_value_error() -> None:
    """generate_preview raises ValueError when asked for an unknown route."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.2, sr=sr)
    evidence = _evidence()
    engine = SmartSafePreviewEngine()

    with pytest.raises(ValueError, match="unknown route"):
        engine.generate_preview("nonexistent_route", audio, sr, acoustic_evidence=evidence)  # type: ignore[arg-type]


def test_estimate_candidate_mos_near_silent_and_evidence_weighting() -> None:
    """_estimate_candidate_mos covers silence, studio risks, and rumble/restore boosts."""
    sr = 48000
    silent_audio = np.zeros(sr // 2, dtype=np.float32)
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    # 1. Near-silent audio produces default score 3.0
    ev_silent = engine.generate_preview(
        "production", silent_audio, sr, acoustic_evidence=_evidence()
    )
    assert ev_silent.evidence.predicted_quality_mos == 3.0

    # 2. Studio with high crosstalk and music risk penalizes MOS
    from hawavoclean.smart_safe.preview import _estimate_candidate_mos

    ev_risky = _evidence(speech_dominance=0.72, music_risk=0.19, crosstalk_risk=0.19)
    ev_clean = _evidence(speech_dominance=0.98, music_risk=0.01, crosstalk_risk=0.01)
    mos_risky, _ = _estimate_candidate_mos(
        "studio", audio, audio, guards_passed=True, evidence=ev_risky
    )
    mos_clean, _ = _estimate_candidate_mos(
        "studio", audio, audio, guards_passed=True, evidence=ev_clean
    )
    assert mos_risky < mos_clean

    # 3. Lowband and lowband_then_production with rumble confidence boost
    ev_rumble = _evidence(rumble_confidence=0.98)
    mos_lb, _ = _estimate_candidate_mos(
        "lowband", audio, audio, guards_passed=True, evidence=ev_rumble
    )
    mos_lb_prod, _ = _estimate_candidate_mos(
        "lowband_then_production", audio, audio, guards_passed=True, evidence=ev_rumble
    )
    assert mos_lb >= 3.75
    assert mos_lb_prod >= 4.0

    # 4. Restore enrolled with verified speaker match
    ev_enrolled = _evidence(
        band_limited_confidence=0.98, speaker_match_verified=True, speaker_match_confidence=0.99
    )
    mos_enrolled, _ = _estimate_candidate_mos(
        "restore_enrolled", audio, audio, guards_passed=True, evidence=ev_enrolled
    )
    assert mos_enrolled >= 4.2


def test_core_exception_handling_in_preview_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preview generation fails closed when underlying cores raise exceptions."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    # 1. StudioVoiceCore fails
    def _mock_studio_fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated studio failure")

    monkeypatch.setattr("hawavoclean.smart_safe.preview.StudioVoiceCore.enhance", _mock_studio_fail)
    prev_studio = engine.generate_preview("studio", audio, sr, acoustic_evidence=evidence)
    assert prev_studio.evidence.content_guard_passed is False
    assert prev_studio.evidence.post_master_guard_passed is False

    # 2. StudioLowBandCore fails on lowband
    def _mock_lowband_fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated lowband failure")

    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.StudioLowBandCore.enhance", _mock_lowband_fail
    )
    prev_lb = engine.generate_preview("lowband", audio, sr, acoustic_evidence=evidence)
    assert prev_lb.evidence.content_guard_passed is False
    assert prev_lb.evidence.post_master_guard_passed is False

    prev_lb_prod = engine.generate_preview(
        "lowband_then_production", audio, sr, acoustic_evidence=evidence
    )
    assert prev_lb_prod.evidence.content_guard_passed is False

    # 3. HawaRestoreKD fails to render candidates
    def _mock_render_fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated restore failure")

    monkeypatch.setattr("hawavoclean.smart_safe.preview.HawaRestoreKD.render", _mock_render_fail)
    prev_src = engine.generate_preview("restore_source", audio, sr, acoustic_evidence=evidence)
    assert prev_src.evidence.content_guard_passed is False
    assert prev_src.evidence.post_master_guard_passed is False

    prev_enr = engine.generate_preview("restore_enrolled", audio, sr, acoustic_evidence=evidence)
    assert prev_enr.evidence.content_guard_passed is False


def test_hawarestore_empty_candidates_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preview generation handles HawaRestoreKD returning empty candidates."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    from hawavoclean.restoration.base import RestorationRenderResult

    empty_res = RestorationRenderResult(
        success=False,
        fallback_status="failed",
        model_name="test",
        provider="cpu",
        solver="midpoint",
        candidates=[],
    )
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.HawaRestoreKD.render", lambda *_args, **_kwargs: empty_res
    )

    prev_src = engine.generate_preview("restore_source", audio, sr, acoustic_evidence=evidence)
    assert prev_src.evidence.content_guard_passed is False

    prev_enr = engine.generate_preview("restore_enrolled", audio, sr, acoustic_evidence=evidence)
    assert prev_enr.evidence.content_guard_passed is False


def test_post_master_invariants_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_post_master_invariants covers all guard branches and failure conditions."""
    sr = 48000
    ref_audio = _make_speech_signal(duration_s=0.5, sr=sr)
    profile = _make_speaker_profile(ref_audio, sr=sr)

    # 1. Empty master
    ok, reason = verify_post_master_invariants(
        np.array([], dtype=np.float32), ref_audio, "production", sr
    )
    assert ok is False
    assert "empty" in reason

    # 2. Preserve passes unconditionally
    ok_pres, _ = verify_post_master_invariants(ref_audio, ref_audio, "preserve", sr)
    assert ok_pres is True

    # 3. Content guard failure
    from hawavoclean.restoration.linguistic_guard import LinguisticGuardResult

    bad_ling = LinguisticGuardResult(
        divergence=0.85,
        anchor_preserved=False,
        status="failed",
        max_frame_divergence=0.95,
        passes_check=False,
    )
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.SoraniLinguisticGuard.evaluate", lambda *_a, **_k: bad_ling
    )
    ok_content, reason_content = verify_post_master_invariants(
        ref_audio, ref_audio, "production", sr
    )
    assert ok_content is False
    assert "content guard failed" in reason_content
    monkeypatch.undo()

    # 4. Speaker identity failure on restore_enrolled with zero-norm profile embedding
    zero_prof = replace(profile, embedding_vector=np.zeros(192, dtype=np.float32))
    ok_zero_spk, reason_zero_spk = verify_post_master_invariants(
        ref_audio, ref_audio, "restore_enrolled", sr, speaker_profile=zero_prof
    )
    assert ok_zero_spk is False
    assert "speaker embedding extraction failed" in reason_zero_spk

    # 5. Speaker identity failure on restore_enrolled with mismatched profile
    diff_emb = np.zeros(192, dtype=np.float32)
    diff_emb[10] = 1.0
    mismatch_prof = replace(profile, embedding_vector=diff_emb)
    ok_mismatch, reason_mismatch = verify_post_master_invariants(
        ref_audio, ref_audio, "restore_enrolled", sr, speaker_profile=mismatch_prof
    )
    assert ok_mismatch is False
    assert "speaker identity failed" in reason_mismatch

    # 6. Speaker identity failure on standard route when master voice identity diverges
    def _mock_low_sim(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.array([1.0] + [0.0] * 191, dtype=np.float32)

    def _mock_orth_sim(*_args: object, **_kwargs: object) -> np.ndarray:
        return np.array([0.0, 1.0] + [0.0] * 190, dtype=np.float32)

    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.SpeakerEmbeddingExtractor.extract", _mock_low_sim
    )
    # Master returns orthogonal vector
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.SpeakerEmbeddingExtractor.extract",
        lambda _self, arr: _mock_orth_sim() if arr is ref_audio else _mock_low_sim(),
    )
    ok_spk_div, reason_spk_div = verify_post_master_invariants(
        ref_audio, ref_audio, "production", sr
    )
    # Undo monkeypatch
    monkeypatch.undo()

    # 7. Protected-band violation on restore route
    from hawavoclean.restoration.protected_band import ProtectedBandVerification

    bad_prot = ProtectedBandVerification(
        max_waveform_abs_error=0.5,
        rms_waveform_error=0.15,
        complex_stft_relative_error=0.3,
        max_phase_deviation_rad=1.2,
        worst_band_energy_deviation_db=9.5,
        worst_band_center_hz=500.0,
        passes_invariance=False,
    )
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.verify_protected_band_invariance",
        lambda *_a, **_k: bad_prot,
    )
    ok_prot, reason_prot = verify_post_master_invariants(
        ref_audio, ref_audio, "restore_source", sr, cutoff_hz=8000.0
    )
    assert ok_prot is False
    assert "protected-band violation" in reason_prot
    monkeypatch.undo()

    # 8. Artifact guard failure on post-master
    from hawavoclean.restoration.highband_events import HighBandEventResult

    bad_hf = HighBandEventResult(
        speech_window_leakage=0.5,
        spurious_burst_count=2,
        hf_envelope_divergence=0.6,
        impulse_discontinuity_ratio=15.0,
        passes_event_check=False,
    )
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.HighBandEventDetector.evaluate", lambda *_a, **_k: bad_hf
    )
    ok_art, reason_art = verify_post_master_invariants(
        ref_audio, ref_audio, "production", sr, cutoff_hz=8000.0
    )
    assert ok_art is False
    assert "artifact guard failed" in reason_art
    monkeypatch.undo()

    # 9. Clean production master passes all invariants
    enhancer = WienerSpectralEnhancer()
    clean_master = enhancer.enhance(ref_audio, sr).waveform
    ok_clean, reason_clean = verify_post_master_invariants(
        clean_master, ref_audio, "production", sr
    )
    assert ok_clean is True
    assert "all post-master invariants verified" in reason_clean


def test_abstain_to_least_intervention_preview_dict_and_evidence_list() -> None:
    """abstain_to_least_intervention accepts dict of previews and list of CandidateEvidence."""
    sr = 48000
    audio = _make_speech_signal(duration_s=0.5, sr=sr)
    evidence = _evidence()
    engine = SmartSafePreviewEngine(allow_research_restore=True)

    previews = engine.generate_all_previews(audio, sr, acoustic_evidence=evidence)

    # Passing dict[Route, CandidatePreview]
    safest_from_dict = abstain_to_least_intervention(previews)
    assert safest_from_dict == "preserve"

    # Passing list of CandidateEvidence
    evidence_list = [p.evidence for p in previews.values()]
    safest_from_ev = abstain_to_least_intervention(evidence_list)
    assert safest_from_ev == "preserve"

    # With preserve failing, picks lowest surviving cost (cost 1: lowband / production)
    ev_no_pres = [e for e in evidence_list if e.route != "preserve"]
    safest_no_pres = abstain_to_least_intervention(ev_no_pres)
    assert safest_no_pres in ("lowband", "production")

    # When lowband also fails, falls back to production
    ev_only_prod = [e for e in ev_no_pres if e.route != "lowband"]
    assert abstain_to_least_intervention(ev_only_prod) == "production"
