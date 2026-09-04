"""Unit tests for restoration identity gating, source mode, and enrolled mode (R2.7, R2.8).

Verifies:
1. Enrolled Mode preflight gating: input audio departing from selected profile
   falls back to Natural audio before model execution (restorer is never called).
2. Enrolled Mode matching: input audio matching selected profile proceeds to restorer.
3. Missing or degenerate profile embedding falls back before model execution.
4. Source Mode: unseen speaker audio (speaker_profile=None) proceeds to restorer,
   and Guard R validates source-candidate speaker consistency.
5. Multi-session variance vector is loaded and evaluated in Guard R.
"""

import numpy as np

from hawavoclean.restoration.bandwidth import BandwidthEstimate, BandwidthEvidence
from hawavoclean.restoration.base import (
    RestorationCandidate,
    RestorationRenderResult,
)
from hawavoclean.restoration.config import RestorationConfig
from hawavoclean.restoration.guard import RestorationGuard
from hawavoclean.restoration.policy import RestorationPolicyManager
from hawavoclean.restoration.profiles import (
    SpeakerF0Stats,
    SpeakerProfile,
)
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor

SR = 48000


def _generate_synthetic_speaker(
    f0_hz: float, seed: int = 42, duration_s: float = 1.0
) -> np.ndarray:
    """Generate harmonic voice-like synthetic audio with a given F0."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    sig = np.zeros_like(t)
    rng = np.random.default_rng(seed)
    for harmonic in range(1, 12):
        freq = f0_hz * harmonic
        if freq < SR / 2:
            phase = rng.uniform(0, 2 * np.pi)
            weight = 1.0 / (harmonic**0.8)
            sig += (weight * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    max_amp = float(np.max(np.abs(sig)))
    scaled = sig / (max_amp + 1e-6) * 0.7
    return np.asarray(scaled, dtype=np.float32)


class SpyRestorer:
    """Restorer stub recording whether and how it was invoked."""

    def __init__(self, candidates: list[RestorationCandidate] | None = None) -> None:
        self.calls = 0
        self.last_speaker_id: str | None = None
        self.last_speaker_embedding: np.ndarray | None = None
        self.candidates = candidates or []

    def restore(
        self,
        audio_48k: np.ndarray,
        sample_rate: int,  # noqa: ARG002
        effective_cutoff_hz: float,
        speaker_id: str | None = None,
        speaker_embedding: np.ndarray | None = None,
        f0_trajectory: np.ndarray | None = None,  # noqa: ARG002
        vuv_mask: np.ndarray | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,  # noqa: ARG002
        seed: int = 42,  # noqa: ARG002
    ) -> list[RestorationCandidate]:
        self.calls += 1
        self.last_speaker_id = speaker_id
        self.last_speaker_embedding = speaker_embedding
        if self.candidates:
            return self.candidates
        return [
            RestorationCandidate(strength=1.0, audio=audio_48k, cutoff_hz=effective_cutoff_hz),
            RestorationCandidate(strength=0.0, audio=audio_48k, cutoff_hz=effective_cutoff_hz),
        ]

    def render(
        self,
        audio_48k: np.ndarray,
        sample_rate: int,
        effective_cutoff_hz: float,
        speaker_id: str | None = None,
        speaker_embedding: np.ndarray | None = None,
        f0_trajectory: np.ndarray | None = None,
        vuv_mask: np.ndarray | None = None,
        strengths: list[float] | None = None,
        seed: int = 42,
    ) -> RestorationRenderResult:
        cands = self.restore(
            audio_48k=audio_48k,
            sample_rate=sample_rate,
            effective_cutoff_hz=effective_cutoff_hz,
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
            f0_trajectory=f0_trajectory,
            vuv_mask=vuv_mask,
            strengths=strengths,
            seed=seed,
        )
        has_active = any(c.strength > 0.0 for c in cands)
        return RestorationRenderResult(
            success=has_active,
            fallback_status="none" if has_active else "no_active_candidates",
            model_name="spy-restorer",
            provider="cpu",
            solver="mock",
            candidates=cands,
        )


def _bandwidth_estimate(cutoff_hz: float = 4000.0) -> BandwidthEstimate:
    return BandwidthEstimate(
        effective_cutoff_hz=cutoff_hz,
        confidence=0.95,
        shape="codec_lowpass",
        restore_recommended=True,
        evidence=BandwidthEvidence(
            spectral_rolloff=cutoff_hz,
            above_cutoff_snr_db=5.0,
            stationarity=0.95,
            high_band_energy_ratio_db=-40.0,
        ),
    )


def test_wrong_enrolled_profile_falls_back_before_model_execution() -> None:
    """Input audio departing from enrolled profile must fall back before model execution (R2.7, R2.8)."""
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)

    # Audio of Speaker A (120 Hz)
    spk_a_audio = _generate_synthetic_speaker(120.0, seed=1, duration_s=1.0)
    emb_a = extractor.extract(spk_a_audio)

    # Profile of Speaker B (350 Hz)
    spk_b_audio = _generate_synthetic_speaker(350.0, seed=2, duration_s=1.0)
    emb_b = extractor.extract(spk_b_audio)

    # Verify that Speaker A and Speaker B embeddings are divergent
    sim = float(np.dot(emb_a, emb_b))
    assert sim < 0.75

    prof_b = SpeakerProfile(
        schema_version="1.0",
        speaker_id="speaker_b",
        display_name="Speaker B",
        consent_record="dummy.json",
        canonical_audio_manifest="dummy.jsonl",
        canonical_audio_sha256=["a" * 64],
        profile_embedding_path="dummy.npy",
        profile_embedding_sha256="b" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=350.0, p05_hz=300.0, p95_hz=400.0),
        training_split_id="split_01",
        adapter=None,
        created_by_commit="test",
        notes="Synthetic test profile",
        embedding_vector=emb_b,
    )

    restorer = SpyRestorer()
    cfg = RestorationConfig(enabled=True, mode="explicit")
    guard = RestorationGuard(config=cfg.guard, sample_rate=SR)
    policy = RestorationPolicyManager(config=cfg, restorer=restorer, guard=guard)

    out_audio, decision = policy.process_segment(
        natural_audio=spk_a_audio,
        sample_rate=SR,
        bandwidth_est=_bandwidth_estimate(4000.0),
        speaker_profile=prof_b,
    )

    # Model was NEVER called: zero wasted compute and zero risk of misattribution
    assert restorer.calls == 0
    assert decision.action == "bypassed"
    assert decision.applied_strength == 0.0
    assert decision.guard_result is not None
    assert "does not match enrolled profile" in decision.guard_result.reason
    assert "falling back before model execution" in decision.guard_result.reason
    assert decision.guard_result.speaker.get("mode") == "enrolled_preflight_rejected"
    np.testing.assert_array_equal(out_audio, spk_a_audio)


def test_matching_enrolled_profile_proceeds_to_model_execution() -> None:
    """Input audio matching enrolled profile proceeds to model execution."""
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)
    spk_audio = _generate_synthetic_speaker(160.0, seed=42, duration_s=1.0)
    emb = extractor.extract(spk_audio)

    prof = SpeakerProfile(
        schema_version="1.0",
        speaker_id="matching_spk",
        display_name="Matching Speaker",
        consent_record="dummy.json",
        canonical_audio_manifest="dummy.jsonl",
        canonical_audio_sha256=["a" * 64],
        profile_embedding_path="dummy.npy",
        profile_embedding_sha256="b" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=160.0, p05_hz=140.0, p95_hz=180.0),
        training_split_id="split_01",
        adapter=None,
        created_by_commit="test",
        notes="Synthetic test profile",
        embedding_vector=emb,
    )

    restorer = SpyRestorer()
    cfg = RestorationConfig(enabled=True, mode="explicit")
    guard = RestorationGuard(config=cfg.guard, sample_rate=SR)
    policy = RestorationPolicyManager(config=cfg, restorer=restorer, guard=guard)

    out_audio, decision = policy.process_segment(
        natural_audio=spk_audio,
        sample_rate=SR,
        bandwidth_est=_bandwidth_estimate(4000.0),
        speaker_profile=prof,
    )

    # Restorer was invoked with the matching profile
    assert restorer.calls == 1
    assert restorer.last_speaker_id == "matching_spk"
    np.testing.assert_array_equal(restorer.last_speaker_embedding, emb)
    assert decision.action in {"restored", "reduced"}


def test_missing_or_degenerate_profile_embedding_falls_back_before_model_execution() -> None:
    """A profile missing its embedding vector falls back before model execution."""
    spk_audio = _generate_synthetic_speaker(150.0, duration_s=1.0)
    empty_prof = SpeakerProfile(
        schema_version="1.0",
        speaker_id="empty_spk",
        display_name="Empty Speaker",
        consent_record="dummy.json",
        canonical_audio_manifest="dummy.jsonl",
        canonical_audio_sha256=["a" * 64],
        profile_embedding_path="dummy.npy",
        profile_embedding_sha256="b" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=150.0, p05_hz=120.0, p95_hz=180.0),
        training_split_id="split_01",
        adapter=None,
        created_by_commit="test",
        notes="",
        embedding_vector=None,
    )

    restorer = SpyRestorer()
    cfg = RestorationConfig(enabled=True, mode="explicit")
    policy = RestorationPolicyManager(
        config=cfg, restorer=restorer, guard=RestorationGuard(sample_rate=SR)
    )

    out_audio, decision = policy.process_segment(
        natural_audio=spk_audio,
        sample_rate=SR,
        bandwidth_est=_bandwidth_estimate(4000.0),
        speaker_profile=empty_prof,
    )

    assert restorer.calls == 0
    assert decision.action == "bypassed"
    assert decision.guard_result is not None
    assert "Missing or degenerate embedding" in decision.guard_result.reason
    np.testing.assert_array_equal(out_audio, spk_audio)


def test_source_mode_unseen_speaker_proceeds_and_guards_source_identity() -> None:
    """Source mode (speaker_profile=None) restores unseen speakers and guards source identity."""
    spk_audio = _generate_synthetic_speaker(175.0, seed=12, duration_s=1.0)

    restorer = SpyRestorer()
    cfg = RestorationConfig(enabled=True, mode="explicit")
    guard = RestorationGuard(config=cfg.guard, sample_rate=SR)
    policy = RestorationPolicyManager(config=cfg, restorer=restorer, guard=guard)

    out_audio, decision = policy.process_segment(
        natural_audio=spk_audio,
        sample_rate=SR,
        bandwidth_est=_bandwidth_estimate(4000.0),
        speaker_profile=None,  # Source Mode
    )

    # Restorer called in source mode (speaker_id=None)
    assert restorer.calls == 1
    assert restorer.last_speaker_id is None
    assert restorer.last_speaker_embedding is None
    assert decision.guard_result is not None
    assert decision.guard_result.speaker.get("mode") == "source"
    assert decision.guard_result.speaker.get("speaker_similarity", 0.0) >= 0.99


def test_enrolled_mode_multi_session_variance_recorded_in_guard() -> None:
    """Enrolled profile with variance_vector evaluates variance departure in Guard R."""
    extractor = SpeakerEmbeddingExtractor(sample_rate=SR)
    spk_audio = _generate_synthetic_speaker(160.0, seed=5, duration_s=1.0)
    emb = extractor.extract(spk_audio)
    var = np.full_like(emb, 0.01)

    prof = SpeakerProfile(
        schema_version="1.0",
        speaker_id="var_spk",
        display_name="Variance Speaker",
        consent_record="dummy.json",
        canonical_audio_manifest="dummy.jsonl",
        canonical_audio_sha256=["a" * 64],
        profile_embedding_path="embedding/profile.npy",
        profile_embedding_sha256="a" * 64,
        profile_variance_path="embedding/variance.npy",
        profile_variance_sha256="v" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=160.0, p05_hz=140.0, p95_hz=180.0),
        training_split_id="split_01",
        adapter=None,
        created_by_commit="test",
        notes="",
        embedding_vector=emb,
        variance_vector=var,
    )

    guard = RestorationGuard(sample_rate=SR)
    cand = RestorationCandidate(strength=1.0, audio=spk_audio, cutoff_hz=4000.0)
    selected_audio, res = guard.select_best_candidate(
        natural_audio=spk_audio,
        candidates=[cand],
        cutoff_hz=4000.0,
        speaker_embedding=prof.embedding_vector,
        canonical_embedding=prof.embedding_vector,
        variance_vector=prof.variance_vector,
    )

    assert res.verdict == "PASS"
    assert res.speaker.get("mode") == "enrolled"
    assert "variance_departure" in res.speaker
    assert res.speaker["variance_departure"] < 1.0
