"""Restoration Guard R: Multi-layer fidelity, phonetic, and speaker validation."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from hawavoclean.restoration.config import RestorationGuardConfig
from hawavoclean.restoration.f0 import F0Extractor
from hawavoclean.restoration.highband_events import HighBandEventDetector, HighBandEventResult
from hawavoclean.restoration.linguistic_guard import LinguisticGuardResult, SoraniLinguisticGuard
from hawavoclean.restoration.protected_band import (
    ProtectedBandVerification,
    verify_protected_band_invariance,
)
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor


@dataclass(frozen=True)
class GuardRResult:
    """Audit verdict and scores from Restoration Guard R."""

    verdict: Literal["PASS", "WARN", "FAIL", "ERROR", "NO_RESTORE"]
    accepted_strength: float
    reason: str
    protected_band: dict[str, Any]
    ctc: dict[str, Any]
    highband_events: dict[str, Any]
    harmonic: dict[str, Any]
    speaker: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable audit dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PostMasteringSegmentEvidence:
    """Evidence retained per reconstructed segment after final mastering."""

    segment_index: int
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    action: str  # "verified", "fallback_reverted", "bypassed"
    passes: bool
    reason: str
    rms_waveform_error: float
    stft_relative_error: float
    worst_band_energy_deviation_db: float
    worst_band_center_hz: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert segment evidence to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PostMasteringVerificationResult:
    """Summary of Guard R re-verification after provider quantization and final mastering."""

    passes: bool
    fallback_applied: bool
    reason: str
    reconstructed_segments_verified: int
    segments_evidence: list[PostMasteringSegmentEvidence]

    def to_dict(self) -> dict[str, Any]:
        """Convert verification result to serializable dictionary."""
        return {
            "passes": self.passes,
            "fallback_applied": self.fallback_applied,
            "reason": self.reason,
            "reconstructed_segments_verified": self.reconstructed_segments_verified,
            "segments_evidence": [seg.to_dict() for seg in self.segments_evidence],
        }


class RestorationGuard:
    """Restoration Guard R orchestrating multi-layer fidelity verification."""

    def __init__(
        self,
        config: RestorationGuardConfig | None = None,
        sample_rate: int = 48000,
    ) -> None:
        self.config = config or RestorationGuardConfig()
        self.sample_rate = sample_rate
        self.hf_event_detector = HighBandEventDetector(sample_rate=sample_rate)
        self.f0_extractor = F0Extractor(sample_rate=sample_rate)
        self.speaker_extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)
        self.linguistic_guard = SoraniLinguisticGuard(
            sample_rate=sample_rate, threshold=self.config.ctc_threshold
        )

    def evaluate_candidate(
        self,
        natural_audio: np.ndarray,
        candidate_audio: np.ndarray,
        cutoff_hz: float,
        speaker_embedding: np.ndarray | None = None,  # noqa: ARG002
        canonical_embedding: np.ndarray | None = None,
        variance_vector: np.ndarray | None = None,
        speech_mask: np.ndarray | None = None,
        f0_statistics: dict[str, float] | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Evaluate a single restoration candidate against all Guard R layers.

        Every layer is unconditional: a candidate that reaches this method is an
        *active* restoration proposal and has to earn its acceptance. The Natural
        fallback is never routed through here — it is what the guard falls back
        *to*, so scoring it against itself would only manufacture a passing verdict.

        Returns (passes, reason, detailed_metrics).
        """
        # 1. Structural Integrity Check
        if natural_audio.ndim != candidate_audio.ndim:
            return (
                False,
                f"Dimension mismatch: {natural_audio.ndim}D vs {candidate_audio.ndim}D",
                {},
            )

        if natural_audio.shape != candidate_audio.shape:
            return False, f"Shape mismatch: {natural_audio.shape} vs {candidate_audio.shape}", {}

        if candidate_audio.size == 0:
            return False, "Empty candidate audio", {}

        if np.any(np.isnan(candidate_audio)) or np.any(np.isinf(candidate_audio)):
            return False, "NaN or Inf detected in restored candidate", {}

        peak_amp = float(np.max(np.abs(candidate_audio)))
        if peak_amp > 1.05:
            return (
                False,
                f"Clipping detected (peak {peak_amp:.3f} > 1.05) in restored candidate",
                {},
            )

        # 2. Protected-Band Invariance Check
        prot_verif: ProtectedBandVerification = verify_protected_band_invariance(
            natural_audio,
            candidate_audio,
            sample_rate=self.sample_rate,
            cutoff_hz=cutoff_hz,
            tolerance_rms=self.config.protected_band_threshold,
            tolerance_stft=self.config.protected_band_threshold * 2.0,
        )
        if not prot_verif.passes_invariance:
            return (
                False,
                f"Protected-band violation: RMS error {prot_verif.rms_waveform_error:.5f} > {self.config.protected_band_threshold}",
                {"protected_band": asdict(prot_verif)},
            )

        # 3. High-Frequency Event Consistency Check
        hf_res: HighBandEventResult = self.hf_event_detector.evaluate(
            natural_audio,
            candidate_audio,
            speech_mask=speech_mask,
            cutoff_hz=cutoff_hz,
        )
        if not hf_res.passes_event_check:
            # Name the metric that actually failed. Quoting leakage and burst
            # count unconditionally produced audit reasons reading
            # "leakage=0.000, bursts=0" — both passing — while the real cause
            # was the envelope or the boundary check, which sends a reader
            # looking in the wrong place.
            hf_cfg = self.hf_event_detector
            failed = [
                name
                for name, ok in (
                    (
                        f"non-speech leakage {hf_res.speech_window_leakage:.3f} > "
                        f"{hf_cfg.leakage_threshold}",
                        hf_res.speech_window_leakage <= hf_cfg.leakage_threshold,
                    ),
                    (
                        f"{hf_res.spurious_burst_count} spurious burst(s) > "
                        f"{hf_cfg.max_spurious_bursts}",
                        hf_res.spurious_burst_count <= hf_cfg.max_spurious_bursts,
                    ),
                    (
                        f"envelope divergence {hf_res.hf_envelope_divergence:.3f} > "
                        f"{hf_cfg.envelope_threshold}",
                        hf_res.hf_envelope_divergence <= hf_cfg.envelope_threshold,
                    ),
                    (
                        f"impulse discontinuity {hf_res.impulse_discontinuity_ratio:.2f}x "
                        f"local step > {hf_cfg.impulse_threshold}x",
                        hf_res.impulse_discontinuity_ratio <= hf_cfg.impulse_threshold,
                    ),
                )
                if not ok
            ]
            return (
                False,
                "High-band event inconsistency: " + "; ".join(failed or ["unspecified"]),
                {"highband_events": asdict(hf_res)},
            )

        # 4. F0 and Harmonic Consistency Check
        f0_nat = self.f0_extractor.extract(natural_audio)
        f0_cand = self.f0_extractor.extract(candidate_audio)

        pitch_diff = 0.0
        n_min = min(len(f0_nat.f0_hz), len(f0_cand.f0_hz))
        if n_min > 0:
            voiced = (f0_nat.vuv_mask[:n_min] > 0.5) & (f0_cand.vuv_mask[:n_min] > 0.5)
            if np.any(voiced):
                pitch_diff = float(
                    np.mean(
                        np.abs(f0_nat.f0_hz[:n_min][voiced] - f0_cand.f0_hz[:n_min][voiced])
                        / (f0_nat.f0_hz[:n_min][voiced] + 1e-6)
                    )
                )

        harmonic_info = {
            "pitch_divergence": pitch_diff,
            "nat_median_f0": f0_nat.statistics.median_hz,
            "cand_median_f0": f0_cand.statistics.median_hz,
        }
        if pitch_diff > self.config.harmonic_threshold:
            return (
                False,
                f"Harmonic pitch divergence {pitch_diff:.3f} > {self.config.harmonic_threshold}",
                harmonic_info,
            )

        # 5. Speaker Identity Check (extract embedding from candidate audio vs canonical/source)
        speaker_sim = 1.0
        var_departure: float | None = None
        if canonical_embedding is not None and canonical_embedding.size > 0:
            # Enrolled Mode: verify against canonical profile embedding
            cand_embedding = self.speaker_extractor.extract(candidate_audio)
            norm_a = float(np.linalg.norm(canonical_embedding))
            norm_b = float(np.linalg.norm(cand_embedding))
            if norm_a > 1e-9 and norm_b > 1e-9:
                speaker_sim = float(np.dot(canonical_embedding, cand_embedding) / (norm_a * norm_b))
            else:
                speaker_sim = 0.0

            if variance_vector is not None and variance_vector.size == canonical_embedding.size:
                sq_diff = (cand_embedding - canonical_embedding) ** 2
                var_departure = float(np.mean(sq_diff / (variance_vector + 1e-4)))

            mode = "enrolled"
            should_check = True
        else:
            # Source Mode: verify candidate against natural input audio speech embedding
            source_embedding = self.speaker_extractor.extract(natural_audio)
            norm_source = float(np.linalg.norm(source_embedding))
            if norm_source > 1e-9:
                cand_embedding = self.speaker_extractor.extract(candidate_audio)
                norm_cand = float(np.linalg.norm(cand_embedding))
                if norm_cand > 1e-9:
                    speaker_sim = float(
                        np.dot(source_embedding, cand_embedding) / (norm_source * norm_cand)
                    )
                else:
                    speaker_sim = 0.0
                should_check = True
            else:
                speaker_sim = 1.0
                should_check = False
            mode = "source"

        speaker_info: dict[str, Any] = {
            "speaker_similarity": speaker_sim,
            "threshold": self.config.speaker_threshold,
            "mode": mode,
        }
        if var_departure is not None:
            speaker_info["variance_departure"] = var_departure

        if should_check and speaker_sim < self.config.speaker_threshold:
            return (
                False,
                f"Speaker similarity {speaker_sim:.3f} < {self.config.speaker_threshold} ({mode} mode)",
                speaker_info,
            )

        # 6. Sorani Linguistic / Acoustic Posterior Stability Check
        ling_res: LinguisticGuardResult = self.linguistic_guard.evaluate(
            natural_audio,
            candidate_audio,
            speech_mask=speech_mask,
            f0_statistics=f0_statistics,
        )
        ctc_info = ling_res.to_dict()
        if not ling_res.passes_check:
            return (
                False,
                f"Linguistic posterior divergence {ling_res.divergence:.4f} > {self.config.ctc_threshold}",
                ctc_info,
            )

        metrics = {
            "protected_band": asdict(prot_verif),
            "highband_events": asdict(hf_res),
            "harmonic": harmonic_info,
            "speaker": speaker_info,
            "ctc": ctc_info,
        }
        return True, "Passed all Guard R layers", metrics

    def select_best_candidate(
        self,
        natural_audio: np.ndarray,
        candidates: list[Any],
        cutoff_hz: float,
        speaker_embedding: np.ndarray | None = None,
        canonical_embedding: np.ndarray | None = None,
        variance_vector: np.ndarray | None = None,
        speech_mask: np.ndarray | None = None,
        f0_statistics: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, GuardRResult]:
        if not candidates:
            result = GuardRResult(
                verdict="NO_RESTORE",
                accepted_strength=0.0,
                reason="No restoration candidates provided; preserved Natural audio",
                protected_band={},
                ctc={},
                highband_events={},
                harmonic={},
                speaker={},
            )
            return natural_audio, result

        # Only candidates that actually propose new content are evaluated. A
        # zero-strength candidate is the Natural audio by construction, so it
        # would pass every layer trivially and mask a total revert as a PASS.
        active_cands = sorted(
            (c for c in candidates if c.strength > 0.0), key=lambda c: c.strength, reverse=True
        )
        if not active_cands:
            result = GuardRResult(
                verdict="NO_RESTORE",
                accepted_strength=0.0,
                reason="No active restoration candidates offered; preserved Natural audio",
                protected_band={},
                ctc={},
                highband_events={},
                harmonic={},
                speaker={},
            )
            return natural_audio, result

        last_metrics: dict[str, Any] = {}
        last_reason = "no candidate evaluated"
        for cand in active_cands:
            passes, reason, metrics = self.evaluate_candidate(
                natural_audio=natural_audio,
                candidate_audio=cand.audio,
                cutoff_hz=cutoff_hz,
                speaker_embedding=speaker_embedding,
                canonical_embedding=canonical_embedding,
                variance_vector=variance_vector,
                speech_mask=speech_mask,
                f0_statistics=f0_statistics,
            )
            last_metrics = metrics
            last_reason = f"strength {cand.strength:.2f}: {reason}"

            if passes:
                verdict: Literal["PASS", "WARN", "FAIL", "ERROR", "NO_RESTORE"] = (
                    "PASS" if cand.strength >= 0.75 else "WARN"
                )
                result = GuardRResult(
                    verdict=verdict,
                    accepted_strength=cand.strength,
                    reason=f"Accepted strength {cand.strength:.2f}: {reason}",
                    protected_band=metrics.get("protected_band", {}),
                    ctc=metrics.get("ctc", {}),
                    highband_events=metrics.get("highband_events", {}),
                    harmonic=metrics.get("harmonic", {}),
                    speaker=metrics.get("speaker", {}),
                )
                return cand.audio, result

        # Every active candidate was rejected: revert to Natural and keep the
        # weakest candidate's metrics, which is the evidence that explains why.
        result = GuardRResult(
            verdict="FAIL",
            accepted_strength=0.0,
            reason=(
                f"All {len(active_cands)} active restoration candidates failed Guard R; "
                f"reverted to Natural-safe audio. Last rejection — {last_reason}"
            ),
            protected_band=last_metrics.get("protected_band", {}),
            ctc=last_metrics.get("ctc", {}),
            highband_events=last_metrics.get("highband_events", {}),
            harmonic=last_metrics.get("harmonic", {}),
            speaker=last_metrics.get("speaker", {}),
        )
        return natural_audio, result

    def verify_post_mastering(
        self,
        *,
        mastered_natural: np.ndarray,
        mastered_restored: np.ndarray,
        segment_records: list[Any],
        starts: list[int],
        seg_len: int,
        cutoff_hz: float,
        transition_hz: float = 500.0,
        tolerance_rms: float = 1e-4,
        tolerance_stft: float = 1e-3,
        tolerance_third_octave_db: float = 0.25,
        canonical_embedding: np.ndarray | None = None,
        variance_vector: np.ndarray | None = None,
        speech_mask: np.ndarray | None = None,
        f0_statistics: dict[str, float] | None = None,
    ) -> PostMasteringVerificationResult:
        """Rerun Guard R checks after provider quantization and final mastering.

        Compares the mastered restored audio against an equally mastered Natural
        reference segment by segment. Reconstructed segments (restored or reduced)
        must strictly satisfy signal integrity (RMS <= tolerance_rms, relative
        STFT <= tolerance_stft, and 1/3-octave deviation <= tolerance_third_octave_db
        outside the transition band) as well as Guard R structural and high-band
        consistency.

        Missing or failed evidence forces an explicit Natural fallback.
        """
        n_samples = mastered_natural.shape[-1]
        reconstructed_indices = [
            i
            for i, rec in enumerate(segment_records)
            if getattr(rec, "action", "") in ("restored", "reduced")
            or getattr(rec, "applied_strength", 0.0) > 0.0
        ]

        if not reconstructed_indices:
            return PostMasteringVerificationResult(
                passes=True,
                fallback_applied=False,
                reason="No active reconstructed segments to verify; preserved Natural audio",
                reconstructed_segments_verified=0,
                segments_evidence=[],
            )

        evidence_list: list[PostMasteringSegmentEvidence] = []
        all_passed = True
        failure_reasons: list[str] = []

        for index in reconstructed_indices:
            if index >= len(starts):
                all_passed = False
                failure_reasons.append(f"Segment {index} start index missing from bounds")
                continue

            start = starts[index]
            stop = min(start + seg_len, n_samples)
            start_time = round(start / self.sample_rate, 3)
            end_time = round(stop / self.sample_rate, 3)

            nat_seg = mastered_natural[..., start:stop]
            rest_seg = mastered_restored[..., start:stop]

            # 1. Structural integrity
            if rest_seg.size == 0 or np.any(np.isnan(rest_seg)) or np.any(np.isinf(rest_seg)):
                all_passed = False
                reason = f"Segment {index}: non-finite or empty post-mastered audio"
                failure_reasons.append(reason)
                evidence_list.append(
                    PostMasteringSegmentEvidence(
                        segment_index=index,
                        start_sample=start,
                        end_sample=stop,
                        start_time_s=start_time,
                        end_time_s=end_time,
                        action="fallback_reverted",
                        passes=False,
                        reason=reason,
                        rms_waveform_error=float("inf"),
                        stft_relative_error=float("inf"),
                        worst_band_energy_deviation_db=float("inf"),
                        worst_band_center_hz=0.0,
                        metrics={},
                    )
                )
                continue

            peak_amp = float(np.max(np.abs(rest_seg)))
            if peak_amp > 1.05:
                all_passed = False
                reason = f"Segment {index}: clipping post-mastering (peak {peak_amp:.3f} > 1.05)"
                failure_reasons.append(reason)
                evidence_list.append(
                    PostMasteringSegmentEvidence(
                        segment_index=index,
                        start_sample=start,
                        end_sample=stop,
                        start_time_s=start_time,
                        end_time_s=end_time,
                        action="fallback_reverted",
                        passes=False,
                        reason=reason,
                        rms_waveform_error=float("inf"),
                        stft_relative_error=float("inf"),
                        worst_band_energy_deviation_db=float("inf"),
                        worst_band_center_hz=0.0,
                        metrics={"peak_amp": peak_amp},
                    )
                )
                continue

            # 2. Protected-band signal integrity
            prot_verif = verify_protected_band_invariance(
                original_audio=nat_seg,
                restored_audio=rest_seg,
                sample_rate=self.sample_rate,
                cutoff_hz=cutoff_hz,
                transition_hz=transition_hz,
                tolerance_rms=tolerance_rms,
                tolerance_stft=tolerance_stft,
                max_third_octave_deviation_db=tolerance_third_octave_db,
            )

            seg_metrics: dict[str, Any] = {"protected_band": asdict(prot_verif)}

            if not prot_verif.passes_invariance:
                all_passed = False
                violations: list[str] = []
                if prot_verif.rms_waveform_error > tolerance_rms:
                    violations.append(f"RMS {prot_verif.rms_waveform_error:.6f} > {tolerance_rms}")
                if prot_verif.complex_stft_relative_error > tolerance_stft:
                    violations.append(
                        f"STFT {prot_verif.complex_stft_relative_error:.6f} > {tolerance_stft}"
                    )
                if prot_verif.worst_band_energy_deviation_db > tolerance_third_octave_db:
                    violations.append(
                        f"1/3-octave {prot_verif.worst_band_energy_deviation_db:.3f} dB > "
                        f"{tolerance_third_octave_db} dB at {prot_verif.worst_band_center_hz:.0f} Hz"
                    )
                reason = f"Segment {index} signal integrity violation: " + "; ".join(violations)
                failure_reasons.append(reason)
                evidence_list.append(
                    PostMasteringSegmentEvidence(
                        segment_index=index,
                        start_sample=start,
                        end_sample=stop,
                        start_time_s=start_time,
                        end_time_s=end_time,
                        action="fallback_reverted",
                        passes=False,
                        reason=reason,
                        rms_waveform_error=prot_verif.rms_waveform_error,
                        stft_relative_error=prot_verif.complex_stft_relative_error,
                        worst_band_energy_deviation_db=prot_verif.worst_band_energy_deviation_db,
                        worst_band_center_hz=prot_verif.worst_band_center_hz,
                        metrics=seg_metrics,
                    )
                )
                continue

            # 3. High-band event consistency
            hf_res = self.hf_event_detector.evaluate(
                nat_seg,
                rest_seg,
                speech_mask=speech_mask,
                cutoff_hz=cutoff_hz,
            )
            seg_metrics["highband_events"] = asdict(hf_res)
            if not hf_res.passes_event_check:
                all_passed = False
                reason = f"Segment {index}: high-band event inconsistency post-mastering"
                failure_reasons.append(reason)
                evidence_list.append(
                    PostMasteringSegmentEvidence(
                        segment_index=index,
                        start_sample=start,
                        end_sample=stop,
                        start_time_s=start_time,
                        end_time_s=end_time,
                        action="fallback_reverted",
                        passes=False,
                        reason=reason,
                        rms_waveform_error=prot_verif.rms_waveform_error,
                        stft_relative_error=prot_verif.complex_stft_relative_error,
                        worst_band_energy_deviation_db=prot_verif.worst_band_energy_deviation_db,
                        worst_band_center_hz=prot_verif.worst_band_center_hz,
                        metrics=seg_metrics,
                    )
                )
                continue

            # 4. Harmonic pitch consistency
            f0_nat = self.f0_extractor.extract(nat_seg)
            f0_cand = self.f0_extractor.extract(rest_seg)
            pitch_diff = 0.0
            n_min = min(len(f0_nat.f0_hz), len(f0_cand.f0_hz))
            if n_min > 0:
                voiced = (f0_nat.vuv_mask[:n_min] > 0.5) & (f0_cand.vuv_mask[:n_min] > 0.5)
                if np.any(voiced):
                    pitch_diff = float(
                        np.mean(
                            np.abs(f0_nat.f0_hz[:n_min][voiced] - f0_cand.f0_hz[:n_min][voiced])
                            / (f0_nat.f0_hz[:n_min][voiced] + 1e-6)
                        )
                    )
            harmonic_info: dict[str, Any] = {
                "pitch_divergence": pitch_diff,
                "nat_median_f0": f0_nat.statistics.median_hz,
                "cand_median_f0": f0_cand.statistics.median_hz,
            }
            if f0_statistics is not None:
                harmonic_info["target_f0_stats"] = f0_statistics
            seg_metrics["harmonic"] = harmonic_info
            if pitch_diff > self.config.harmonic_threshold:
                all_passed = False
                reason = (
                    f"Segment {index}: harmonic pitch divergence {pitch_diff:.3f} > "
                    f"{self.config.harmonic_threshold}"
                )
                failure_reasons.append(reason)
                evidence_list.append(
                    PostMasteringSegmentEvidence(
                        segment_index=index,
                        start_sample=start,
                        end_sample=stop,
                        start_time_s=start_time,
                        end_time_s=end_time,
                        action="fallback_reverted",
                        passes=False,
                        reason=reason,
                        rms_waveform_error=prot_verif.rms_waveform_error,
                        stft_relative_error=prot_verif.complex_stft_relative_error,
                        worst_band_energy_deviation_db=prot_verif.worst_band_energy_deviation_db,
                        worst_band_center_hz=prot_verif.worst_band_center_hz,
                        metrics=seg_metrics,
                    )
                )
                continue

            # 5. Speaker identity check
            cand_embedding = self.speaker_extractor.extract(rest_seg)
            speaker_sim = 1.0
            var_departure: float | None = None
            should_check_speaker = True
            if canonical_embedding is not None and canonical_embedding.size > 0:
                norm_a = float(np.linalg.norm(canonical_embedding))
                norm_b = float(np.linalg.norm(cand_embedding))
                if norm_a > 1e-9 and norm_b > 1e-9:
                    speaker_sim = float(
                        np.dot(canonical_embedding, cand_embedding) / (norm_a * norm_b)
                    )
                else:
                    speaker_sim = 0.0

                if variance_vector is not None and variance_vector.size == canonical_embedding.size:
                    sq_diff = (cand_embedding - canonical_embedding) ** 2
                    var_departure = float(np.mean(sq_diff / (variance_vector + 1e-4)))
            else:
                source_embedding = self.speaker_extractor.extract(nat_seg)
                norm_source = float(np.linalg.norm(source_embedding))
                norm_cand = float(np.linalg.norm(cand_embedding))
                if norm_source > 1e-9 and norm_cand > 1e-9:
                    speaker_sim = float(
                        np.dot(source_embedding, cand_embedding) / (norm_source * norm_cand)
                    )
                else:
                    speaker_sim = 1.0
                    should_check_speaker = False

            speaker_info: dict[str, Any] = {
                "speaker_similarity": speaker_sim,
                "threshold": self.config.speaker_threshold,
            }
            if var_departure is not None:
                speaker_info["variance_departure"] = var_departure
            seg_metrics["speaker"] = speaker_info

            if should_check_speaker and speaker_sim < self.config.speaker_threshold:
                all_passed = False
                reason = (
                    f"Segment {index}: speaker similarity {speaker_sim:.3f} < "
                    f"{self.config.speaker_threshold}"
                )
                failure_reasons.append(reason)
                evidence_list.append(
                    PostMasteringSegmentEvidence(
                        segment_index=index,
                        start_sample=start,
                        end_sample=stop,
                        start_time_s=start_time,
                        end_time_s=end_time,
                        action="fallback_reverted",
                        passes=False,
                        reason=reason,
                        rms_waveform_error=prot_verif.rms_waveform_error,
                        stft_relative_error=prot_verif.complex_stft_relative_error,
                        worst_band_energy_deviation_db=prot_verif.worst_band_energy_deviation_db,
                        worst_band_center_hz=prot_verif.worst_band_center_hz,
                        metrics=seg_metrics,
                    )
                )
                continue

            # Segment verified
            evidence_list.append(
                PostMasteringSegmentEvidence(
                    segment_index=index,
                    start_sample=start,
                    end_sample=stop,
                    start_time_s=start_time,
                    end_time_s=end_time,
                    action="verified",
                    passes=True,
                    reason="Passed post-mastering signal integrity and Guard R verification",
                    rms_waveform_error=prot_verif.rms_waveform_error,
                    stft_relative_error=prot_verif.complex_stft_relative_error,
                    worst_band_energy_deviation_db=prot_verif.worst_band_energy_deviation_db,
                    worst_band_center_hz=prot_verif.worst_band_center_hz,
                    metrics=seg_metrics,
                )
            )

        if not all_passed or not evidence_list:
            return PostMasteringVerificationResult(
                passes=False,
                fallback_applied=True,
                reason="Post-mastering verification failed; forced explicit Natural fallback: "
                + "; ".join(failure_reasons or ["Missing verification evidence"]),
                reconstructed_segments_verified=len(reconstructed_indices),
                segments_evidence=evidence_list,
            )

        return PostMasteringVerificationResult(
            passes=True,
            fallback_applied=False,
            reason=f"All {len(evidence_list)} reconstructed segments verified successfully post-mastering",
            reconstructed_segments_verified=len(reconstructed_indices),
            segments_evidence=evidence_list,
        )
