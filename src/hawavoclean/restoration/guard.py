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


def compute_highband_lsd(
    ref_audio: np.ndarray,
    cand_audio: np.ndarray,
    sample_rate: int = 48000,
    cutoff_hz: float = 4000.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """Compute Log-Spectral Distance (LSD) in dB over the generated high band."""
    from scipy import signal

    ref_mono = np.mean(ref_audio, axis=0) if ref_audio.ndim == 2 else ref_audio
    cand_mono = np.mean(cand_audio, axis=0) if cand_audio.ndim == 2 else cand_audio

    min_len = min(len(ref_mono), len(cand_mono))
    if min_len < n_fft:
        return 0.0

    ref_mono = ref_mono[:min_len]
    cand_mono = cand_mono[:min_len]

    _, _, Z_ref = signal.stft(
        ref_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    _, _, Z_cand = signal.stft(
        cand_mono,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band_mask = (freqs >= cutoff_hz) & (freqs <= sample_rate / 2.0)

    if not np.any(band_mask):
        return 0.0

    P_ref = np.abs(Z_ref[band_mask, :]) ** 2 + 1e-12
    P_cand = np.abs(Z_cand[band_mask, :]) ** 2 + 1e-12

    log_ref = 10.0 * np.log10(P_ref)
    log_cand = 10.0 * np.log10(P_cand)

    frame_lsd = np.sqrt(np.mean((log_ref - log_cand) ** 2, axis=0))
    return float(np.mean(frame_lsd))


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

        # 5. Speaker Identity Check (extract embedding from candidate audio vs canonical)
        speaker_sim = 1.0
        if canonical_embedding is not None and canonical_embedding.size > 0:
            cand_embedding = self.speaker_extractor.extract(candidate_audio)
            norm_a = float(np.linalg.norm(canonical_embedding))
            norm_b = float(np.linalg.norm(cand_embedding))
            if norm_a > 1e-9 and norm_b > 1e-9:
                speaker_sim = float(np.dot(canonical_embedding, cand_embedding) / (norm_a * norm_b))
            else:
                speaker_sim = 0.0

        speaker_info = {
            "speaker_similarity": speaker_sim,
            "threshold": self.config.speaker_threshold,
        }
        if speaker_sim < self.config.speaker_threshold and canonical_embedding is not None:
            return (
                False,
                f"Speaker similarity {speaker_sim:.3f} < {self.config.speaker_threshold}",
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

        # 7. High-Band Objective Spectral Distance (LSD) Check
        lsd_highband = compute_highband_lsd(
            natural_audio, candidate_audio, sample_rate=self.sample_rate, cutoff_hz=cutoff_hz
        )
        lsd_info = {"highband_lsd_db": round(lsd_highband, 2), "threshold_db": 18.0}
        if lsd_highband > 18.0:
            return (
                False,
                f"High-band LSD divergence {lsd_highband:.2f} dB > 18.00 dB limit",
                {"highband_lsd": lsd_info},
            )

        metrics = {
            "protected_band": asdict(prot_verif),
            "highband_events": asdict(hf_res),
            "harmonic": harmonic_info,
            "speaker": speaker_info,
            "ctc": ctc_info,
            "highband_lsd": lsd_info,
        }
        return True, "Passed all Guard R layers", metrics

    def select_best_candidate(
        self,
        natural_audio: np.ndarray,
        candidates: list[Any],
        cutoff_hz: float,
        speaker_embedding: np.ndarray | None = None,
        canonical_embedding: np.ndarray | None = None,
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
