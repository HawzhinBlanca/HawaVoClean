"""Engine-internal preview generation and hard guard verification for Smart Safe (I3.2, I3.3).

This module enforces True-10 invariants for Smart Safe:
* All 7 candidate route previews are generated inside the engine on bounded segments.
* Callers cannot supply arbitrary MOS scores or guard booleans into candidate evidence.
* Hard content, identity, protected-band, and artifact guards are evaluated internally before ranking.
* Full-length masters are post-verified against all safety invariants.
* Any post-master verification failure triggers deterministic abstention down the intervention ladder.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.enhancement.studio import StudioVoiceCore
from hawavoclean.enhancement.studio_lowband import StudioLowBandCore
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.highband_events import HighBandEventDetector
from hawavoclean.restoration.linguistic_guard import SoraniLinguisticGuard
from hawavoclean.restoration.profiles import SpeakerProfile
from hawavoclean.restoration.protected_band import verify_protected_band_invariance
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from hawavoclean.smart_safe.decision import (
    DEFAULT_POLICY,
    INTERVENTION_COST,
    RESTORE_ROUTES,
    ROUTES,
    AcousticEvidence,
    CandidateEvidence,
    CandidateOutcome,
    RestorePolicy,
    Route,
    SmartSafeDecision,
    SmartSafePolicy,
    SmartSafeRanker,
    decide_smart_safe,
    eligible_routes,
)


@dataclass(frozen=True, slots=True)
class CandidatePreview:
    """Bounded audio preview and internally evaluated evidence for a candidate route."""

    route: Route
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    audio_sha256: str
    evidence: CandidateEvidence


def compute_evidence_sha256(
    *,
    route: Route,
    audio_sha256: str,
    predicted_quality_mos: float,
    prediction_confidence: float,
    content_guard_passed: bool,
    speaker_guard_passed: bool,
    protected_band_guard_passed: bool,
    artifact_guard_passed: bool,
    post_master_guard_passed: bool,
    reconstruction_disclosed: bool,
) -> str:
    """Compute deterministic canonical SHA-256 digest over candidate evidence."""
    payload = {
        "route": route,
        "audio_sha256": audio_sha256,
        "predicted_quality_mos": round(float(predicted_quality_mos), 4),
        "prediction_confidence": round(float(prediction_confidence), 4),
        "content_guard_passed": bool(content_guard_passed),
        "speaker_guard_passed": bool(speaker_guard_passed),
        "protected_band_guard_passed": bool(protected_band_guard_passed),
        "artifact_guard_passed": bool(artifact_guard_passed),
        "post_master_guard_passed": bool(post_master_guard_passed),
        "reconstruction_disclosed": bool(reconstruction_disclosed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_candidate_evidence_integrity(preview: CandidatePreview) -> bool:
    """Verify that a CandidatePreview's evidence was derived internally from its audio."""
    recomputed_audio_sha256 = hashlib.sha256(preview.audio.tobytes()).hexdigest()
    if preview.audio_sha256 != recomputed_audio_sha256:
        return False
    expected_evidence_sha256 = compute_evidence_sha256(
        route=preview.route,
        audio_sha256=recomputed_audio_sha256,
        predicted_quality_mos=preview.evidence.predicted_quality_mos,
        prediction_confidence=preview.evidence.prediction_confidence,
        content_guard_passed=preview.evidence.content_guard_passed,
        speaker_guard_passed=preview.evidence.speaker_guard_passed,
        protected_band_guard_passed=preview.evidence.protected_band_guard_passed,
        artifact_guard_passed=preview.evidence.artifact_guard_passed,
        post_master_guard_passed=preview.evidence.post_master_guard_passed,
        reconstruction_disclosed=preview.evidence.reconstruction_disclosed,
    )
    return preview.evidence.evidence_sha256 == expected_evidence_sha256


def extract_preview_slice(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_duration_s: float = 10.0,
    target_sample_rate: int = 48000,
) -> np.ndarray:
    """Extract a bounded preview segment resampled to 48 kHz mono float32."""
    mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float32)
    if sample_rate != target_sample_rate:
        mono = resample_audio(mono, sample_rate, target_sample_rate)
    max_samples = int(max_duration_s * target_sample_rate)
    if len(mono) > max_samples:
        mono = mono[:max_samples]
    return np.ascontiguousarray(mono, dtype=np.float32)


def _estimate_candidate_mos(
    route: Route,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    guards_passed: bool,
    evidence: AcousticEvidence,
) -> tuple[float, float]:
    """Derive internal objective MOS (1.0 - 5.0) and confidence (0.0 - 1.0)."""
    if not guards_passed:
        return 1.50, 0.95

    ref_rms = float(np.sqrt(np.mean(reference**2))) if reference.size > 0 else 0.0
    cand_rms = float(np.sqrt(np.mean(candidate**2))) if candidate.size > 0 else 0.0
    if ref_rms < 1e-4:
        return 3.0, 0.90

    base_scores: dict[Route, float] = {
        "preserve": 3.40,
        "production": 3.90,
        "studio": 4.15,
        "lowband": 3.75,
        "lowband_then_production": 4.05,
        "restore_source": 4.10,
        "restore_enrolled": 4.30,
    }
    base = base_scores.get(route, 3.50)

    if route == "studio":
        speech_boost = (evidence.speech_dominance - 0.70) * 0.50
        risk_penalty = (evidence.music_risk + evidence.crosstalk_risk) * 0.80
        base += speech_boost - risk_penalty
    elif route in ("lowband", "lowband_then_production"):
        rumble_boost = (evidence.rumble_confidence - 0.70) * 0.40
        base += rumble_boost
    elif route in ("restore_source", "restore_enrolled"):
        band_boost = (evidence.band_limited_confidence - 0.85) * 0.50
        base += band_boost
        if route == "restore_enrolled" and evidence.speaker_match_verified:
            base += (evidence.speaker_match_confidence - 0.85) * 0.40

    noise_reduction = max(0.0, min(1.0, (ref_rms - cand_rms) / (ref_rms + 1e-6)))
    if route != "preserve":
        base += noise_reduction * 0.20

    mos = float(np.clip(base, 1.0, 5.0))
    confidence = 0.92
    return round(mos, 3), round(confidence, 3)


class SmartSafePreviewEngine:
    """Internal engine for generating reproducible candidate previews and evaluating guards."""

    def __init__(
        self,
        *,
        max_preview_seconds: float = 10.0,
        sample_rate: int = 48000,
        cutoff_hz: float = 8000.0,
        allow_research_restore: bool = False,
    ) -> None:
        self.max_preview_seconds = max_preview_seconds
        self.sample_rate = sample_rate
        self.cutoff_hz = cutoff_hz
        self.allow_research_restore = allow_research_restore
        self._linguistic_guard = SoraniLinguisticGuard(sample_rate=sample_rate)
        self._hf_detector = HighBandEventDetector(sample_rate=sample_rate)
        self._speaker_extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)

    def generate_preview(
        self,
        route: Route,
        audio: np.ndarray,
        sample_rate: int,
        *,
        acoustic_evidence: AcousticEvidence,
        speaker_profile: SpeakerProfile | None = None,
        speaker_profile_id: str | None = None,
    ) -> CandidatePreview:
        """Generate a single preview route and evaluate its hard guards internally."""
        if route not in ROUTES:
            raise ValueError(f"unknown route: {route!r}")

        ref_preview = extract_preview_slice(
            audio,
            sample_rate,
            max_duration_s=self.max_preview_seconds,
            target_sample_rate=self.sample_rate,
        )

        cand_audio: np.ndarray
        route_error_reason: str | None = None

        if route == "preserve":
            cand_audio = ref_preview.copy()
        elif route == "production":
            enhancer = WienerSpectralEnhancer()
            cand_audio = enhancer.enhance(ref_preview, self.sample_rate).waveform
        elif route == "studio":
            try:
                core = StudioVoiceCore()
                cand_audio = core.enhance(ref_preview, self.sample_rate).waveform
            except Exception as e:
                cand_audio = ref_preview.copy()
                route_error_reason = f"studio core error: {e}"
        elif route == "lowband":
            try:
                core_lb = StudioLowBandCore()
                cand_audio = core_lb.enhance(ref_preview, self.sample_rate).waveform
            except Exception as e:
                cand_audio = ref_preview.copy()
                route_error_reason = f"lowband core error: {e}"
        elif route == "lowband_then_production":
            try:
                core_lb = StudioLowBandCore()
                lb_wave = core_lb.enhance(ref_preview, self.sample_rate).waveform
                enhancer = WienerSpectralEnhancer()
                cand_audio = enhancer.enhance(lb_wave, self.sample_rate).waveform
            except Exception as e:
                cand_audio = ref_preview.copy()
                route_error_reason = f"lowband+production error: {e}"
        elif route == "restore_source":
            if not self.allow_research_restore:
                cand_audio = ref_preview.copy()
                route_error_reason = "research restoration is quarantined"
            else:
                try:
                    restorer = HawaRestoreKD(sample_rate=self.sample_rate)
                    res = restorer.render(
                        ref_preview,
                        self.sample_rate,
                        effective_cutoff_hz=self.cutoff_hz,
                        strengths=[1.0],
                    )
                    if res.success and res.candidates:
                        rest_wave = res.candidates[0].audio
                        enhancer = WienerSpectralEnhancer()
                        cand_audio = enhancer.enhance(rest_wave, self.sample_rate).waveform
                    else:
                        cand_audio = ref_preview.copy()
                        route_error_reason = "hawarestore-kd failed to render candidates"
                except Exception as e:
                    cand_audio = ref_preview.copy()
                    route_error_reason = f"restore_source error: {e}"
        elif route == "restore_enrolled":
            if not self.allow_research_restore:
                cand_audio = ref_preview.copy()
                route_error_reason = "research restoration is quarantined"
            else:
                try:
                    speaker_emb = speaker_profile.embedding_vector if speaker_profile else None
                    restorer = HawaRestoreKD(sample_rate=self.sample_rate)
                    res = restorer.render(
                        ref_preview,
                        self.sample_rate,
                        effective_cutoff_hz=self.cutoff_hz,
                        speaker_id=speaker_profile_id,
                        speaker_embedding=speaker_emb,
                        strengths=[1.0],
                    )
                    if res.success and res.candidates:
                        rest_wave = res.candidates[0].audio
                        enhancer = WienerSpectralEnhancer()
                        cand_audio = enhancer.enhance(rest_wave, self.sample_rate).waveform
                    else:
                        cand_audio = ref_preview.copy()
                        route_error_reason = "hawarestore-kd failed to render enrolled candidates"
                except Exception as e:
                    cand_audio = ref_preview.copy()
                    route_error_reason = f"restore_enrolled error: {e}"
        else:
            cand_audio = ref_preview.copy()

        cand_audio = np.ascontiguousarray(cand_audio, dtype=np.float32)
        audio_sha256 = hashlib.sha256(cand_audio.tobytes()).hexdigest()

        # Evaluate hard guards internally
        peak = float(np.max(np.abs(cand_audio))) if cand_audio.size > 0 else 0.0
        post_master_guard_passed = bool(
            peak <= 1.05 and np.all(np.isfinite(cand_audio)) and route_error_reason is None
        )

        ling_res = self._linguistic_guard.evaluate(ref_preview, cand_audio)
        content_guard_passed = (
            ling_res.passes_check
            and not np.isnan(ling_res.divergence)
            and route_error_reason is None
        )

        speaker_guard_passed = True
        if route_error_reason is not None:
            speaker_guard_passed = False
        elif (
            route == "restore_enrolled"
            and speaker_profile is not None
            and speaker_profile.embedding_vector is not None
        ):
            cand_emb = self._speaker_extractor.extract(cand_audio)
            norm_prof = float(np.linalg.norm(speaker_profile.embedding_vector))
            norm_cand = float(np.linalg.norm(cand_emb))
            if norm_prof > 1e-9 and norm_cand > 1e-9:
                sim = float(
                    np.dot(speaker_profile.embedding_vector, cand_emb) / (norm_prof * norm_cand)
                )
                speaker_guard_passed = sim >= 0.75
            else:
                speaker_guard_passed = False
        elif route != "preserve":
            ref_emb = self._speaker_extractor.extract(ref_preview)
            cand_emb = self._speaker_extractor.extract(cand_audio)
            norm_ref = float(np.linalg.norm(ref_emb))
            norm_cand = float(np.linalg.norm(cand_emb))
            if norm_ref > 1e-9 and norm_cand > 1e-9:
                sim = float(np.dot(ref_emb, cand_emb) / (norm_ref * norm_cand))
                speaker_guard_passed = sim >= 0.80
            else:
                speaker_guard_passed = True

        if route_error_reason is not None:
            protected_band_guard_passed = False
        elif route in RESTORE_ROUTES:
            prot = verify_protected_band_invariance(
                ref_preview, cand_audio, self.sample_rate, self.cutoff_hz
            )
            protected_band_guard_passed = prot.passes_invariance
        else:
            protected_band_guard_passed = True

        if route_error_reason is not None:
            artifact_guard_passed = False
        elif route == "preserve":
            artifact_guard_passed = True
        else:
            hf_res = self._hf_detector.evaluate(ref_preview, cand_audio, cutoff_hz=self.cutoff_hz)
            artifact_guard_passed = hf_res.passes_event_check

        reconstruction_disclosed = (
            acoustic_evidence.reconstruction_consent if route in RESTORE_ROUTES else False
        )

        all_guards_ok = (
            content_guard_passed
            and speaker_guard_passed
            and protected_band_guard_passed
            and artifact_guard_passed
            and post_master_guard_passed
            and (route not in RESTORE_ROUTES or reconstruction_disclosed)
        )

        mos, conf = _estimate_candidate_mos(
            route,
            ref_preview,
            cand_audio,
            guards_passed=all_guards_ok,
            evidence=acoustic_evidence,
        )

        evidence_sha256 = compute_evidence_sha256(
            route=route,
            audio_sha256=audio_sha256,
            predicted_quality_mos=mos,
            prediction_confidence=conf,
            content_guard_passed=content_guard_passed,
            speaker_guard_passed=speaker_guard_passed,
            protected_band_guard_passed=protected_band_guard_passed,
            artifact_guard_passed=artifact_guard_passed,
            post_master_guard_passed=post_master_guard_passed,
            reconstruction_disclosed=reconstruction_disclosed,
        )

        evidence = CandidateEvidence(
            route=route,
            predicted_quality_mos=mos,
            prediction_confidence=conf,
            content_guard_passed=content_guard_passed,
            speaker_guard_passed=speaker_guard_passed,
            protected_band_guard_passed=protected_band_guard_passed,
            artifact_guard_passed=artifact_guard_passed,
            post_master_guard_passed=post_master_guard_passed,
            reconstruction_disclosed=reconstruction_disclosed,
            evidence_sha256=evidence_sha256,
        )

        return CandidatePreview(
            route=route,
            audio=cand_audio,
            sample_rate=self.sample_rate,
            duration_s=len(cand_audio) / self.sample_rate,
            audio_sha256=audio_sha256,
            evidence=evidence,
        )

    def generate_all_previews(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        acoustic_evidence: AcousticEvidence,
        restore_policy: RestorePolicy = "disabled",
        speaker_profile: SpeakerProfile | None = None,
        speaker_profile_id: str | None = None,
        policy: SmartSafePolicy = DEFAULT_POLICY,
        only_eligible: bool = False,
    ) -> dict[Route, CandidatePreview]:
        """Generate internally derived previews for routes."""
        eligibility = eligible_routes(
            acoustic_evidence,
            restore_policy=restore_policy,
            speaker_profile_id=speaker_profile_id,
            policy=policy,
        )
        previews: dict[Route, CandidatePreview] = {}
        for route in ROUTES:
            is_eligible, _ = eligibility[route]
            if only_eligible and not is_eligible:
                continue
            preview = self.generate_preview(
                route,
                audio,
                sample_rate,
                acoustic_evidence=acoustic_evidence,
                speaker_profile=speaker_profile,
                speaker_profile_id=speaker_profile_id,
            )
            previews[route] = preview
        return previews

    def decide(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        acoustic_evidence: AcousticEvidence,
        restore_policy: RestorePolicy = "disabled",
        speaker_profile: SpeakerProfile | None = None,
        speaker_profile_id: str | None = None,
        ranker: SmartSafeRanker,
        policy: SmartSafePolicy = DEFAULT_POLICY,
        require_qualified_ranker: bool = False,
    ) -> tuple[SmartSafeDecision, dict[Route, CandidatePreview]]:
        """Generate all previews internally and decide the optimal safe route."""
        previews = self.generate_all_previews(
            audio,
            sample_rate,
            acoustic_evidence=acoustic_evidence,
            restore_policy=restore_policy,
            speaker_profile=speaker_profile,
            speaker_profile_id=speaker_profile_id,
            policy=policy,
        )
        candidates = [p.evidence for p in previews.values()]
        decision = decide_smart_safe(
            acoustic_evidence,
            candidates,
            restore_policy=restore_policy,
            speaker_profile_id=speaker_profile_id,
            ranker=ranker,
            policy=policy,
            require_qualified_ranker=require_qualified_ranker,
        )
        return decision, previews


def verify_post_master_invariants(
    master_audio: np.ndarray,
    reference_audio: np.ndarray,
    route: Route,
    sample_rate: int = 48000,
    *,
    cutoff_hz: float = 8000.0,
    speaker_profile: SpeakerProfile | None = None,
    max_peak: float = 1.05,
) -> tuple[bool, str]:
    """Verify that full rendered/mastered audio satisfies all hard guards.

    Returns (passed, failure_reason).
    """
    if master_audio.size == 0:
        return False, "master audio is empty"
    if not np.all(np.isfinite(master_audio)):
        return False, "NaN or Inf detected in master audio"

    peak = float(np.max(np.abs(master_audio)))
    if peak > max_peak:
        return False, f"clipping detected: peak {peak:.3f} > {max_peak}"

    if route == "preserve":
        return True, "all post-master invariants verified"

    # Protected-band guard
    if route in RESTORE_ROUTES:
        prot = verify_protected_band_invariance(
            reference_audio, master_audio, sample_rate, cutoff_hz
        )
        if not prot.passes_invariance:
            return (
                False,
                f"post-master protected-band violation: RMS error {prot.rms_waveform_error:.5f}",
            )

    # Artifact guard
    hf_detector = HighBandEventDetector(sample_rate=sample_rate)
    hf_res = hf_detector.evaluate(reference_audio, master_audio, cutoff_hz=cutoff_hz)
    if not hf_res.passes_event_check:
        return False, "post-master artifact guard failed: spurious bursts or envelope divergence"

    # Speaker guard
    speaker_extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)
    if (
        route == "restore_enrolled"
        and speaker_profile is not None
        and speaker_profile.embedding_vector is not None
    ):
        master_emb = speaker_extractor.extract(master_audio)
        norm_prof = float(np.linalg.norm(speaker_profile.embedding_vector))
        norm_master = float(np.linalg.norm(master_emb))
        if norm_prof > 1e-9 and norm_master > 1e-9:
            sim = float(
                np.dot(speaker_profile.embedding_vector, master_emb) / (norm_prof * norm_master)
            )
            if sim < 0.75:
                return (
                    False,
                    f"post-master speaker identity failed: similarity {sim:.3f} < 0.75",
                )
        else:
            return False, "post-master speaker embedding extraction failed"
    else:
        ref_emb = speaker_extractor.extract(reference_audio)
        master_emb = speaker_extractor.extract(master_audio)
        norm_ref = float(np.linalg.norm(ref_emb))
        norm_master = float(np.linalg.norm(master_emb))
        if norm_ref > 1e-9 and norm_master > 1e-9:
            sim = float(np.dot(ref_emb, master_emb) / (norm_ref * norm_master))
            if sim < 0.80:
                return (
                    False,
                    f"post-master speaker identity divergence: similarity {sim:.3f} < 0.80",
                )

    # Content guard
    ling_guard = SoraniLinguisticGuard(sample_rate=sample_rate)
    ling_res = ling_guard.evaluate(reference_audio, master_audio)
    if not ling_res.passes_check or np.isnan(ling_res.divergence):
        return (
            False,
            f"post-master content guard failed: linguistic divergence {ling_res.divergence:.4f}",
        )

    return True, "all post-master invariants verified"


def abstain_to_least_intervention(
    candidates: Sequence[CandidateOutcome | CandidateEvidence] | dict[Route, CandidatePreview],
    failed_route: Route | None = None,
) -> Route:
    """Fall back down the intervention cost ladder to the least-modified safe route."""
    safe_routes: set[Route] = set()

    if isinstance(candidates, dict):
        for route, prev in candidates.items():
            ev = prev.evidence
            if (
                ev.content_guard_passed
                and ev.speaker_guard_passed
                and ev.protected_band_guard_passed
                and ev.artifact_guard_passed
                and ev.post_master_guard_passed
            ):
                safe_routes.add(route)
    else:
        for item in candidates:
            if isinstance(item, CandidateOutcome):
                if item.safe:
                    safe_routes.add(item.route)
            elif (
                isinstance(item, CandidateEvidence)
                and item.content_guard_passed
                and item.speaker_guard_passed
                and item.protected_band_guard_passed
                and item.artifact_guard_passed
                and item.post_master_guard_passed
            ):
                safe_routes.add(item.route)

    if failed_route is not None:
        safe_routes.discard(failed_route)

    # Always ensure preserve is an option if nothing else survives
    if not safe_routes:
        return "preserve"

    return min(safe_routes, key=lambda r: (INTERVENTION_COST[r], r))
