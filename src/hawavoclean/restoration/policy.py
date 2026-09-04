"""Restoration policy management, candidate ladder evaluation, and fail-closed fallback."""

from dataclasses import dataclass

import numpy as np

from hawavoclean.logging import get_logger
from hawavoclean.restoration.bandwidth import BandwidthEstimate
from hawavoclean.restoration.base import RestorationCandidate, Restorer
from hawavoclean.restoration.config import RestorationConfig
from hawavoclean.restoration.guard import GuardRResult, RestorationGuard
from hawavoclean.restoration.profiles import SpeakerProfile

logger = get_logger("restoration.policy")


@dataclass(frozen=True)
class SegmentRestorationDecision:
    """Outcome of restoration decision on an audio segment."""

    action: str  # "restored", "reduced", "reverted", "bypassed", "error"
    applied_strength: float
    cutoff_hz: float
    guard_result: GuardRResult | None
    error_message: str | None = None


class RestorationPolicyManager:
    """Orchestrates bandwidth detection, candidate generation, and Guard R selection."""

    def __init__(
        self,
        config: RestorationConfig,
        restorer: Restorer,
        guard: RestorationGuard,
    ) -> None:
        self.config = config
        self.restorer = restorer
        self.guard = guard

    def process_segment(
        self,
        natural_audio: np.ndarray,
        sample_rate: int,
        bandwidth_est: BandwidthEstimate,
        speaker_profile: SpeakerProfile | None = None,
        speech_mask: np.ndarray | None = None,
        segment_seed: int = 42,
    ) -> tuple[np.ndarray, SegmentRestorationDecision]:
        """Apply restoration policy to a Natural-safe candidate segment."""
        # 0. Input validation — fail closed on degenerate input
        if natural_audio.size == 0 or not np.all(np.isfinite(natural_audio)):
            err_reason = (
                "Empty audio"
                if natural_audio.size == 0
                else "Non-finite values in Natural candidate"
            )
            logger.warning("Input validation failed: %s; failing closed to Natural", err_reason)
            err_res = GuardRResult(
                verdict="ERROR",
                accepted_strength=0.0,
                reason=f"Input validation: {err_reason}",
                protected_band={},
                ctc={},
                highband_events={},
                harmonic={},
                speaker={},
            )
            return natural_audio, SegmentRestorationDecision(
                action="error",
                applied_strength=0.0,
                cutoff_hz=bandwidth_est.effective_cutoff_hz,
                guard_result=err_res,
                error_message=err_reason,
            )

        # 1. Healthy audio or low confidence check -> Bypass
        if (
            not bandwidth_est.restore_recommended
            or bandwidth_est.confidence < self.config.cutoff_confidence_min
        ):
            bypass_reason = (
                f"Bandwidth healthy ({bandwidth_est.effective_cutoff_hz:.0f} Hz)"
                if not bandwidth_est.restore_recommended
                else f"Low confidence ({bandwidth_est.confidence:.2f} < {self.config.cutoff_confidence_min:.2f})"
            )
            logger.info("Bypassing restoration: %s", bypass_reason)
            no_restore_res = GuardRResult(
                verdict="NO_RESTORE",
                accepted_strength=0.0,
                reason=bypass_reason,
                protected_band={},
                ctc={},
                highband_events={},
                harmonic={},
                speaker={},
            )
            return natural_audio, SegmentRestorationDecision(
                action="bypassed",
                applied_strength=0.0,
                cutoff_hz=bandwidth_est.effective_cutoff_hz,
                guard_result=no_restore_res,
            )

        # 2. Extract speaker embeddings and F0 stats if profile present
        speaker_id = speaker_profile.speaker_id if speaker_profile else None
        speaker_emb = speaker_profile.embedding_vector if speaker_profile else None
        speaker_var = speaker_profile.variance_vector if speaker_profile else None
        f0_stats: dict[str, float] | None = None
        if speaker_profile is not None and speaker_profile.f0_statistics is not None:
            f0_stats = {
                "median_hz": speaker_profile.f0_statistics.median_hz,
                "p05_hz": speaker_profile.f0_statistics.p05_hz,
                "p95_hz": speaker_profile.f0_statistics.p95_hz,
            }

        # 2b. Enrolled Mode Pre-flight Identity Gating (R2.7, R2.8):
        # Verify selected enrollment against current input before rendering.
        # Wrong or missing enrollment falls back before model execution.
        if speaker_profile is not None:
            if speaker_emb is None or speaker_emb.size == 0:
                reason = (
                    f"Missing or degenerate embedding in enrolled profile '{speaker_id}'; "
                    "falling back before model execution"
                )
                logger.warning(reason)
                fb_res = GuardRResult(
                    verdict="NO_RESTORE",
                    accepted_strength=0.0,
                    reason=reason,
                    protected_band={},
                    ctc={},
                    highband_events={},
                    harmonic={},
                    speaker={"status": "missing_enrollment_embedding"},
                )
                return natural_audio, SegmentRestorationDecision(
                    action="bypassed",
                    applied_strength=0.0,
                    cutoff_hz=bandwidth_est.effective_cutoff_hz,
                    guard_result=fb_res,
                )

            if self.guard.speaker_extractor is not None and speaker_emb.shape == (
                self.guard.speaker_extractor.embed_dim,
            ):
                input_emb = self.guard.speaker_extractor.extract(natural_audio)
                norm_input = float(np.linalg.norm(input_emb))
                norm_prof = float(np.linalg.norm(speaker_emb))
                if norm_input > 1e-6 and norm_prof > 1e-6:
                    sim = float(np.dot(input_emb, speaker_emb) / (norm_input * norm_prof))
                    if sim < self.config.guard.speaker_threshold:
                        reason = (
                            f"Input audio does not match enrolled profile '{speaker_id}' "
                            f"(similarity {sim:.3f} < {self.config.guard.speaker_threshold}); "
                            "falling back before model execution"
                        )
                        logger.warning(reason)
                        fb_res = GuardRResult(
                            verdict="NO_RESTORE",
                            accepted_strength=0.0,
                            reason=reason,
                            protected_band={},
                            ctc={},
                            highband_events={},
                            harmonic={},
                            speaker={
                                "speaker_similarity": sim,
                                "threshold": self.config.guard.speaker_threshold,
                                "mode": "enrolled_preflight_rejected",
                            },
                        )
                        return natural_audio, SegmentRestorationDecision(
                            action="bypassed",
                            applied_strength=0.0,
                            cutoff_hz=bandwidth_est.effective_cutoff_hz,
                            guard_result=fb_res,
                        )

        # 3. Generate candidate ladder
        try:
            candidates: list[RestorationCandidate] = self.restorer.restore(
                audio_48k=natural_audio,
                sample_rate=sample_rate,
                effective_cutoff_hz=bandwidth_est.effective_cutoff_hz,
                speaker_id=speaker_id,
                speaker_embedding=speaker_emb,
                strengths=self.config.strengths,
                seed=segment_seed,
            )
        except Exception as e:
            logger.warning(
                "Restoration model exception on segment; failing closed to Natural: %s", e
            )
            err_res = GuardRResult(
                verdict="ERROR",
                accepted_strength=0.0,
                reason=f"Restorer runtime error: {e}",
                protected_band={},
                ctc={},
                highband_events={},
                harmonic={},
                speaker={},
            )
            return natural_audio, SegmentRestorationDecision(
                action="error",
                applied_strength=0.0,
                cutoff_hz=bandwidth_est.effective_cutoff_hz,
                guard_result=err_res,
                error_message=str(e),
            )

        # 4. Guard R Candidate Selection
        try:
            selected_audio, guard_res = self.guard.select_best_candidate(
                natural_audio=natural_audio,
                candidates=candidates,
                cutoff_hz=bandwidth_est.effective_cutoff_hz,
                speaker_embedding=speaker_emb,
                canonical_embedding=speaker_emb,
                variance_vector=speaker_var,
                speech_mask=speech_mask,
                f0_statistics=f0_stats,
            )

        except Exception as e:
            logger.warning("Guard R exception on segment; failing closed to Natural: %s", e)
            err_res = GuardRResult(
                verdict="ERROR",
                accepted_strength=0.0,
                reason=f"Guard R evaluation error: {e}",
                protected_band={},
                ctc={},
                highband_events={},
                harmonic={},
                speaker={},
            )
            return natural_audio, SegmentRestorationDecision(
                action="error",
                applied_strength=0.0,
                cutoff_hz=bandwidth_est.effective_cutoff_hz,
                guard_result=err_res,
                error_message=str(e),
            )

        if guard_res.verdict == "NO_RESTORE":
            # The restorer offered nothing to judge; that is a bypass, not a
            # rejection, and the counts must not claim the guard turned work away.
            action = "bypassed"
        elif guard_res.accepted_strength >= 0.75:
            action = "restored"
        elif guard_res.accepted_strength > 0.0:
            action = "reduced"
        else:
            action = "reverted"

        return selected_audio, SegmentRestorationDecision(
            action=action,
            applied_strength=guard_res.accepted_strength,
            cutoff_hz=bandwidth_est.effective_cutoff_hz,
            guard_result=guard_res,
        )
