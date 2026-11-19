"""Engine-internal preview generation and hard guard verification for Smart Safe (I3.2, I3.3).

This module enforces True-10 invariants for Smart Safe:
* Candidate route previews are generated inside the engine on representative speech-active windows.
* Callers cannot supply arbitrary MOS scores or guard booleans into candidate evidence.
* Hard content (linguistic + active-speech phonetic), identity, protected-band, and artifact
  (HF event + neural discontinuity/coloration) guards are evaluated internally before ranking.
* Perceptual quality is scored from the actual candidate waveform (NISQA v2 / DNSMOS / proxy),
  never from hardcoded per-route lookup tables.
* Full-length masters are post-verified against all safety invariants.
* Any post-master verification failure triggers deterministic abstention down the intervention ladder.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal

from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.enhancement.studio import StudioVoiceCore
from hawavoclean.enhancement.studio_lowband import StudioLowBandCore
from hawavoclean.guard.perceptual_mos import PerceptualQualityScore, PerceptualQualityScorer
from hawavoclean.guard.phonetic_guard import PhoneticStabilityGuard
from hawavoclean.hashing import hash_bytes, hash_json_canonical
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.highband_events import HighBandEventDetector, HighBandEventResult
from hawavoclean.restoration.linguistic_guard import SoraniLinguisticGuard
from hawavoclean.restoration.neural_bwe import NeuralBWERestorer, probe_neural_bwe_runtime
from hawavoclean.restoration.profiles import SpeakerProfile
from hawavoclean.restoration.protected_band import verify_protected_band_invariance
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from hawavoclean.smart_safe.decision import (
    ALL_ROUTES,
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

STUDIO_PRO_CORE_ID = "studio-mossformer2-48k-v1"
STUDIO_DEREVERB_CORE_ID = "studio-dereverb-48k-v1"

# Neural perceptual artifact guard thresholds (applied when scorer is calibrated)
_NEURAL_DISCONTINUITY_FLOOR = 2.8
_NEURAL_COLORATION_FLOOR = 2.5
_NEURAL_AXIS_DROP_MIN = 0.20

_DEFAULT_PERCEPTUAL_SCORER = PerceptualQualityScorer()
_DEFAULT_HAWARESTORE_RENDER = HawaRestoreKD.render


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
    return hash_json_canonical(payload)


def verify_candidate_evidence_integrity(preview: CandidatePreview) -> bool:
    """Verify that a CandidatePreview's evidence was derived internally from its audio."""
    recomputed_audio_sha256 = hash_bytes(preview.audio.tobytes())
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


def _crossfade_concat(segments: Sequence[np.ndarray], fade_samples: int) -> np.ndarray:
    """Concatenate 1D float32 segments with raised-cosine linear-sum crossfades."""
    if not segments:
        return np.zeros(0, dtype=np.float32)
    out = np.asarray(segments[0], dtype=np.float32).copy()
    for seg in segments[1:]:
        nxt = np.asarray(seg, dtype=np.float32)
        fade = min(fade_samples, len(out) // 2, len(nxt) // 2)
        if fade <= 1:
            out = np.concatenate([out, nxt])
            continue
        t = np.linspace(0.0, np.pi, fade, endpoint=True, dtype=np.float32)
        fade_in = 0.5 * (1.0 - np.cos(t))
        fade_out = 1.0 - fade_in
        blended = out[-fade:] * fade_out + nxt[:fade] * fade_in
        out = np.concatenate([out[:-fade], blended, nxt[fade:]])
    return np.ascontiguousarray(out, dtype=np.float32)


def _select_representative_windows(
    mono: np.ndarray,
    sample_rate: int,
    max_samples: int,
) -> tuple[list[tuple[int, int]], int]:
    """Select up to 3 representative speech-active windows within ``max_samples``.

    Returns ``(windows, fade_samples)`` sorted chronologically. Falls back to
    ``[(0, max_samples)]`` for short, silent, or stationary-envelope clips.
    """
    n_samples = len(mono)
    if n_samples <= max_samples or max_samples <= 0:
        return [(0, min(n_samples, max(0, max_samples)))], 0

    overall_rms = float(np.sqrt(np.mean(mono**2)))
    if overall_rms < 1e-4:
        return [(0, max_samples)], 0

    frame_len = max(16, int(sample_rate * 0.020))
    frame_hop = frame_len
    n_frames = (n_samples - frame_len) // frame_hop + 1
    if n_frames < 4:
        return [(0, max_samples)], 0

    frames = np.lib.stride_tricks.sliding_window_view(mono, frame_len)[::frame_hop][:n_frames]
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12).astype(np.float32)
    mean_rms = float(np.mean(frame_rms))
    if mean_rms < 1e-4 or float(np.std(frame_rms)) < 1e-4 * mean_rms:
        return [(0, max_samples)], 0

    # For very short preview budgets (< 1.5s), select one peak-speech window.
    if max_samples < int(1.5 * sample_rate):
        win_frames = max(1, max_samples // frame_hop)
        step_frames = max(1, win_frames // 4)
        best_start = 0
        best_score = -1.0
        for f_start in range(0, max(1, n_frames - win_frames + 1), step_frames):
            f_end = min(n_frames, f_start + win_frames)
            score = float(np.percentile(frame_rms[f_start:f_end], 85))
            if score > best_score:
                best_score = score
                best_start = min(n_samples - max_samples, f_start * frame_hop)
        return [(best_start, best_start + max_samples)], 0

    # Standard multi-window selection: up to 3 non-overlapping speech-active windows
    fade = min(int(0.020 * sample_rate), max_samples // 20)
    if n_samples >= max_samples + 2 * fade:
        win_len = (max_samples + 2 * fade) // 3
    else:
        fade = 0
        win_len = max_samples // 3

    win_frames = max(1, win_len // frame_hop)
    step_frames = max(1, win_frames // 4)
    candidates: list[tuple[int, float, float]] = []
    for f_start in range(0, max(1, n_frames - win_frames + 1), step_frames):
        f_end = min(n_frames, f_start + win_frames)
        s_start = min(n_samples - win_len, f_start * frame_hop)
        seg_rms = frame_rms[f_start:f_end]
        speech_e = float(np.percentile(seg_rms, 85))
        floor_e = float(np.percentile(seg_rms, 15))
        contrast = floor_e / (speech_e + 1e-6)
        candidates.append((s_start, speech_e, contrast))

    max_speech = max(c[1] for c in candidates)
    if max_speech < 1e-4:
        return [(0, max_samples)], 0

    active_threshold = max(1e-4, 0.25 * max_speech)
    active = [c for c in candidates if c[1] >= active_threshold]
    if not active:
        return [(0, max_samples)], 0

    chosen: list[int] = []

    def _non_overlapping(pool: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
        return [c for c in pool if all(abs(c[0] - prev) >= win_len for prev in chosen)]

    # 1. Highest active speech energy window
    peak_win = max(active, key=lambda c: c[1])
    chosen.append(peak_win[0])

    # 2. Mid-recording active window
    rem_mid = _non_overlapping(active)
    if rem_mid:
        midpoint = (n_samples - win_len) / 2.0
        mid_win = min(rem_mid, key=lambda c: abs(c[0] - midpoint))
        chosen.append(mid_win[0])

    # 3. Highest noise/reverb contrast active window
    rem_contrast = _non_overlapping(active)
    if rem_contrast:
        contrast_win = max(rem_contrast, key=lambda c: c[2])
        chosen.append(contrast_win[0])

    chosen.sort()
    if len(chosen) == 1:
        center = chosen[0] + win_len // 2
        start = max(0, min(n_samples - max_samples, center - max_samples // 2))
        return [(start, start + max_samples)], 0

    windows = [(s, min(n_samples, s + win_len)) for s in chosen]
    return windows, fade


def extract_preview_slice(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_duration_s: float = 10.0,
    target_sample_rate: int = 48000,
) -> np.ndarray:
    """Extract a representative bounded preview resampled to 48 kHz mono float32.

    For recordings longer than ``max_duration_s``, selects up to 3 representative
    speech-active windows (highest active speech energy, mid-recording active
    window, and highest noise/reverb contrast window) spliced with raised-cosine
    crossfades. Falls back cleanly to ``mono[:max_samples]`` on short, silent,
    or stationary clips.
    """
    mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1) if audio.ndim == 2 else audio
    mono = np.asarray(mono, dtype=np.float32)
    if sample_rate != target_sample_rate:
        mono = resample_audio(mono, sample_rate, target_sample_rate)
    max_samples = int(max_duration_s * target_sample_rate)
    if len(mono) <= max_samples or max_samples <= 0:
        return np.ascontiguousarray(mono[: max(0, max_samples)], dtype=np.float32)

    windows, fade = _select_representative_windows(mono, target_sample_rate, max_samples)
    if len(windows) == 1:
        start, end = windows[0]
        return np.ascontiguousarray(mono[start:end], dtype=np.float32)

    segments = [mono[start:end] for start, end in windows]
    spliced = _crossfade_concat(segments, fade)
    if len(spliced) > max_samples:
        spliced = spliced[:max_samples]
    return np.ascontiguousarray(spliced, dtype=np.float32)


def _enhancement_hf_artifact_ok(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    hf_res: HighBandEventResult,
    ref_score: PerceptualQualityScore | None = None,
    cand_score: PerceptualQualityScore | None = None,
) -> bool:
    """Allow legitimate subtractive HF noise removal and dereverb transient sharpening."""
    if hf_res.passes_event_check:
        return True
    ref_1d = np.mean(reference, axis=0) if reference.ndim == 2 else reference
    cand_1d = np.mean(candidate, axis=0) if candidate.ndim == 2 else candidate
    min_len = min(len(ref_1d), len(cand_1d))
    if min_len >= 32:
        hp_freq = max(1000.0, min(sample_rate / 2.0 - 500.0, cutoff_hz))
        sos_hp = signal.butter(4, hp_freq, btype="highpass", fs=sample_rate, output="sos")
        ref_hf_rms = float(
            np.sqrt(np.mean(signal.sosfiltfilt(sos_hp, ref_1d[:min_len]) ** 2) + 1e-12)
        )
        cand_hf_rms = float(
            np.sqrt(np.mean(signal.sosfiltfilt(sos_hp, cand_1d[:min_len]) ** 2) + 1e-12)
        )
        if ref_hf_rms > 1e-4 and cand_hf_rms <= ref_hf_rms * 0.85:
            return True
    return bool(
        ref_score is not None
        and cand_score is not None
        and ref_score.calibrated
        and cand_score.calibrated
        and hf_res.speech_window_leakage <= 0.25
        and hf_res.spurious_burst_count == 0
        and hf_res.hf_envelope_divergence <= 0.35
        and cand_score.discontinuity >= ref_score.discontinuity
    )


def _instantiate_enhancer(enhancer_class: type[object]) -> object:
    """Construct an enhancer core using ``device="auto"`` when supported."""
    params = inspect.signature(enhancer_class.__init__).parameters
    if "device" in params:
        return enhancer_class(device="auto")  # type: ignore[call-arg]
    return enhancer_class()


def passes_perceptual_artifact_check(
    reference_score: PerceptualQualityScore,
    candidate_score: PerceptualQualityScore,
) -> bool:
    """Fail closed when a calibrated neural scorer detects discontinuity or coloration collapse."""
    if not (reference_score.calibrated and candidate_score.calibrated):
        return True
    discontinuity_drop = reference_score.discontinuity - candidate_score.discontinuity
    if (
        candidate_score.discontinuity < _NEURAL_DISCONTINUITY_FLOOR
        and discontinuity_drop > _NEURAL_AXIS_DROP_MIN
    ):
        return False
    coloration_drop = reference_score.coloration - candidate_score.coloration
    return not (
        candidate_score.coloration < _NEURAL_COLORATION_FLOOR
        and coloration_drop > _NEURAL_AXIS_DROP_MIN
    )


def _estimate_candidate_mos(
    route: Route,  # noqa: ARG001
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    guards_passed: bool,
    evidence: AcousticEvidence,  # noqa: ARG001
    sample_rate: int = 48000,
    scorer: PerceptualQualityScorer | None = None,
    candidate_score: PerceptualQualityScore | None = None,
) -> tuple[float, float]:
    """Derive internal perceptual MOS (1.0 - 5.0) and confidence (0.0 - 1.0)."""
    if not guards_passed:
        return 1.50, 0.95

    ref_rms = float(np.sqrt(np.mean(reference**2))) if reference.size > 0 else 0.0
    if ref_rms < 1e-4:
        return 3.0, 0.90

    active_scorer = _DEFAULT_PERCEPTUAL_SCORER if scorer is None else scorer
    score = (
        candidate_score
        if candidate_score is not None
        else active_scorer.score(candidate, sample_rate)
    )
    return round(float(score.mos), 3), round(float(score.confidence), 3)


def _extract_active_speech_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    peak_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return concatenated SNR-active 20 ms reference/candidate speech frames, or None."""
    ref_1d = np.mean(reference, axis=0) if reference.ndim == 2 else reference
    cand_1d = np.mean(candidate, axis=0) if candidate.ndim == 2 else candidate
    frame_len = max(16, int(sample_rate * 0.020))
    n_frames = min(len(ref_1d), len(cand_1d)) // frame_len
    if n_frames < 8:
        return None
    r_fr = ref_1d[: n_frames * frame_len].reshape(n_frames, frame_len)
    c_fr = cand_1d[: n_frames * frame_len].reshape(n_frames, frame_len)
    r_rms = np.sqrt(np.mean(r_fr**2, axis=1) + 1e-12)
    if peak_percentile is not None:
        thresh = max(1e-4, float(np.percentile(r_rms, peak_percentile)))
        mask = r_rms >= thresh
    else:
        r_floor = float(np.percentile(r_rms, 15))
        r_p85 = float(np.percentile(r_rms, 85))
        mask = r_rms > max(1e-4, r_floor + 0.35 * max(0.0, r_p85 - r_floor), r_p85 * 0.22)
    if int(np.sum(mask)) < 4:
        return None
    return (
        np.ascontiguousarray(r_fr[mask].reshape(-1), dtype=np.float32),
        np.ascontiguousarray(c_fr[mask].reshape(-1), dtype=np.float32),
    )


def _speaker_similarity_passes(
    extractor: SpeakerEmbeddingExtractor,
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.80,
) -> tuple[bool, float]:
    """Compare speaker identity, falling back to active/direct speech when pauses or tails were cleaned."""
    ref_emb = extractor.extract(reference)
    cand_emb = extractor.extract(candidate)
    norm_ref = float(np.linalg.norm(ref_emb))
    norm_cand = float(np.linalg.norm(cand_emb))
    if norm_ref <= 1e-9 or norm_cand <= 1e-9:
        return True, 1.0
    sim = float(np.dot(ref_emb, cand_emb) / (norm_ref * norm_cand))
    if sim >= threshold or reference is candidate:
        return sim >= threshold, sim
    best_sim = sim
    for pct in (None, 75.0):
        active_pair = _extract_active_speech_pair(
            reference, candidate, sample_rate, peak_percentile=pct
        )
        if active_pair is None:
            continue
        act_ref_emb = extractor.extract(active_pair[0])
        act_cand_emb = extractor.extract(active_pair[1])
        norm_act_ref = float(np.linalg.norm(act_ref_emb))
        norm_act_cand = float(np.linalg.norm(act_cand_emb))
        if norm_act_ref > 1e-9 and norm_act_cand > 1e-9:
            act_sim = float(np.dot(act_ref_emb, act_cand_emb) / (norm_act_ref * norm_act_cand))
            best_sim = max(best_sim, act_sim)
            if act_sim >= threshold:
                return True, act_sim
    return False, best_sim


def _is_studio_pro_registered() -> bool:
    from hawavoclean.enhancement.factory import CORE_REGISTRY

    return STUDIO_PRO_CORE_ID in CORE_REGISTRY


def _is_dereverb_registered() -> bool:
    from hawavoclean.enhancement.factory import CORE_REGISTRY

    return STUDIO_DEREVERB_CORE_ID in CORE_REGISTRY


class SmartSafePreviewEngine:
    """Internal engine for generating reproducible candidate previews and evaluating guards."""

    def __init__(
        self,
        *,
        max_preview_seconds: float = 10.0,
        sample_rate: int = 48000,
        cutoff_hz: float = 8000.0,
        allow_research_restore: bool = False,
        include_studio_pro: bool = False,
        include_dereverb: bool | None = None,
        perceptual_scorer: PerceptualQualityScorer | None = None,
        phonetic_guard: PhoneticStabilityGuard | None = None,
    ) -> None:
        self.max_preview_seconds = max_preview_seconds
        self.sample_rate = sample_rate
        self.cutoff_hz = cutoff_hz
        self.allow_research_restore = allow_research_restore
        self.include_studio_pro = include_studio_pro
        self.include_dereverb = include_dereverb
        self._linguistic_guard = SoraniLinguisticGuard(sample_rate=sample_rate)
        self._phonetic_guard = (
            phonetic_guard
            if phonetic_guard is not None
            else PhoneticStabilityGuard(sample_rate=sample_rate)
        )
        self._perceptual_scorer = (
            perceptual_scorer if perceptual_scorer is not None else PerceptualQualityScorer()
        )
        self._hf_detector = HighBandEventDetector(sample_rate=sample_rate)
        self._speaker_extractor = SpeakerEmbeddingExtractor(sample_rate=sample_rate)

    @property
    def perceptual_backend(self) -> str:
        """Name of the active perceptual quality scoring backend (e.g. 'nisqa_v2')."""
        return self._perceptual_scorer.active_backend

    def render_route_audio(
        self,
        route: Route,
        audio: np.ndarray,
        sample_rate: int,
        *,
        speaker_profile: SpeakerProfile | None = None,
        speaker_profile_id: str | None = None,
        allow_research_restore: bool | None = None,
    ) -> tuple[np.ndarray, str | None]:
        """Render audio for a specific route, returning (waveform, error_reason)."""
        if route not in ALL_ROUTES:
            raise ValueError(f"unknown route: {route!r}")

        allow_restore = (
            self.allow_research_restore
            if allow_research_restore is None
            else allow_research_restore
        )

        if audio.ndim == 2:
            rendered_channels = []
            channel_err: str | None = None
            for ch in range(audio.shape[0]):
                ch_wave, ch_err = self.render_route_audio(
                    route,
                    audio[ch],
                    sample_rate,
                    speaker_profile=speaker_profile,
                    speaker_profile_id=speaker_profile_id,
                    allow_research_restore=allow_research_restore,
                )
                if ch_err is not None:
                    channel_err = ch_err
                rendered_channels.append(ch_wave)
            return np.stack(rendered_channels, axis=0), channel_err

        def _render_studio_pro() -> tuple[np.ndarray, str | None]:
            from hawavoclean.enhancement.factory import CORE_REGISTRY

            reg = CORE_REGISTRY.get(STUDIO_PRO_CORE_ID)
            if reg is None:
                return audio.copy(), f"studio_pro core ({STUDIO_PRO_CORE_ID}) is not registered"
            enhancer = _instantiate_enhancer(reg.enhancer_class)
            return enhancer.enhance(audio, sample_rate).waveform, None  # type: ignore[attr-defined]

        def _render_dereverb() -> tuple[np.ndarray, str | None]:
            from hawavoclean.enhancement.factory import CORE_REGISTRY

            reg = CORE_REGISTRY.get(STUDIO_DEREVERB_CORE_ID)
            if reg is None:
                return audio.copy(), f"dereverb core ({STUDIO_DEREVERB_CORE_ID}) is not registered"
            enhancer = _instantiate_enhancer(reg.enhancer_class)
            return enhancer.enhance(audio, sample_rate).waveform, None  # type: ignore[attr-defined]

        def _render_restore(*, enrolled: bool) -> tuple[np.ndarray, str | None]:
            if not allow_restore:
                return audio.copy(), "research restoration is quarantined"
            if (
                not enrolled
                and HawaRestoreKD.render is _DEFAULT_HAWARESTORE_RENDER
                and probe_neural_bwe_runtime()[0]
            ):
                bwe_res = NeuralBWERestorer(device="auto").render(
                    audio,
                    sample_rate,
                    effective_cutoff_hz=self.cutoff_hz,
                    strengths=[1.0],
                )
                active_cand = next((c for c in bwe_res.candidates if c.strength > 0.0), None)
                if bwe_res.success and active_cand is not None:
                    return active_cand.audio, None
                return (
                    audio.copy(),
                    f"neural BWE failed to render candidates ({bwe_res.fallback_status})",
                )
            speaker_emb = (
                speaker_profile.embedding_vector if (enrolled and speaker_profile) else None
            )
            res = HawaRestoreKD(sample_rate=sample_rate).render(
                audio,
                sample_rate,
                effective_cutoff_hz=self.cutoff_hz,
                speaker_id=speaker_profile_id if enrolled else None,
                speaker_embedding=speaker_emb,
                strengths=[1.0],
            )
            if not (res.success and res.candidates):
                label = "enrolled candidates" if enrolled else "candidates"
                return audio.copy(), f"hawarestore-kd failed to render {label}"
            return WienerSpectralEnhancer().enhance(
                res.candidates[0].audio, sample_rate
            ).waveform, None

        runners: dict[Route, tuple[str, Callable[[], tuple[np.ndarray, str | None]]]] = {
            "preserve": ("preserve", lambda: (audio.copy(), None)),
            "production": (
                "production",
                lambda: (WienerSpectralEnhancer().enhance(audio, sample_rate).waveform, None),
            ),
            "studio": (
                "studio core",
                lambda: (StudioVoiceCore(device="cpu").enhance(audio, sample_rate).waveform, None),
            ),
            "studio_pro": ("studio_pro core", _render_studio_pro),
            "dereverb": ("dereverb core", _render_dereverb),
            "lowband": (
                "lowband core",
                lambda: (
                    StudioLowBandCore(device="cpu").enhance(audio, sample_rate).waveform,
                    None,
                ),
            ),
            "lowband_then_production": (
                "lowband+production",
                lambda: (
                    WienerSpectralEnhancer()
                    .enhance(
                        StudioLowBandCore(device="cpu").enhance(audio, sample_rate).waveform,
                        sample_rate,
                    )
                    .waveform,
                    None,
                ),
            ),
            "restore_source": ("restore_source", lambda: _render_restore(enrolled=False)),
            "restore_enrolled": ("restore_enrolled", lambda: _render_restore(enrolled=True)),
        }

        route_error_reason: str | None = None
        label, fn = runners[route]
        try:
            cand_audio, route_error_reason = fn()
        except Exception as e:
            cand_audio = audio.copy()
            route_error_reason = f"{label} error: {e}"

        cand_audio = np.ascontiguousarray(cand_audio, dtype=np.float32)
        return cand_audio, route_error_reason

    def generate_preview(
        self,
        route: Route,
        audio: np.ndarray,
        sample_rate: int,
        *,
        acoustic_evidence: AcousticEvidence,
        speaker_profile: SpeakerProfile | None = None,
        speaker_profile_id: str | None = None,
        _ref_preview: np.ndarray | None = None,
        _ref_score: PerceptualQualityScore | None = None,
    ) -> CandidatePreview:
        """Generate a single preview route and evaluate its hard guards internally."""
        if route not in ALL_ROUTES:
            raise ValueError(f"unknown route: {route!r}")

        ref_preview = (
            _ref_preview
            if _ref_preview is not None
            else extract_preview_slice(
                audio,
                sample_rate,
                max_duration_s=self.max_preview_seconds,
                target_sample_rate=self.sample_rate,
            )
        )
        ref_score = (
            _ref_score
            if _ref_score is not None
            else self._perceptual_scorer.score(ref_preview, self.sample_rate)
        )

        cand_audio, route_error_reason = self.render_route_audio(
            route,
            ref_preview,
            self.sample_rate,
            speaker_profile=speaker_profile,
            speaker_profile_id=speaker_profile_id,
        )

        cand_audio = np.ascontiguousarray(cand_audio, dtype=np.float32)
        audio_sha256 = hash_bytes(cand_audio.tobytes())

        # Evaluate hard guards internally
        peak = float(np.max(np.abs(cand_audio))) if cand_audio.size > 0 else 0.0
        post_master_guard_passed = bool(
            peak <= 1.05 and np.all(np.isfinite(cand_audio)) and route_error_reason is None
        )

        if route_error_reason is not None:
            content_guard_passed = False
        else:
            ling_res = self._linguistic_guard.evaluate(ref_preview, cand_audio)
            phonetic_res = self._phonetic_guard.evaluate(
                ref_preview, cand_audio, sample_rate=self.sample_rate
            )
            content_guard_passed = bool(
                ling_res.passes_check
                and not np.isnan(ling_res.divergence)
                and phonetic_res.passes_check
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
            speaker_guard_passed, _ = _speaker_similarity_passes(
                self._speaker_extractor,
                ref_preview,
                cand_audio,
                self.sample_rate,
                threshold=0.80,
            )

        if route_error_reason is not None:
            protected_band_guard_passed = False
        elif route in RESTORE_ROUTES:
            prot = verify_protected_band_invariance(
                ref_preview,
                cand_audio,
                self.sample_rate,
                self.cutoff_hz,
                tolerance_rms=0.05,
                tolerance_stft=0.10,
                max_third_octave_deviation_db=6.0,
            )
            protected_band_guard_passed = prot.passes_invariance
        else:
            protected_band_guard_passed = True

        cand_score: PerceptualQualityScore | None = None
        if route_error_reason is None:
            cand_score = (
                ref_score
                if route == "preserve"
                else self._perceptual_scorer.score(cand_audio, self.sample_rate)
            )

        if route_error_reason is not None:
            artifact_guard_passed = False
        elif route == "preserve":
            artifact_guard_passed = True
        else:
            hf_res = self._hf_detector.evaluate(ref_preview, cand_audio, cutoff_hz=self.cutoff_hz)
            perceptual_ok = (
                passes_perceptual_artifact_check(ref_score, cand_score)
                if cand_score is not None
                else False
            )
            hf_ok = (
                hf_res.passes_event_check
                if route in RESTORE_ROUTES
                else _enhancement_hf_artifact_ok(
                    ref_preview,
                    cand_audio,
                    self.sample_rate,
                    self.cutoff_hz,
                    hf_res,
                    ref_score,
                    cand_score,
                )
            )
            artifact_guard_passed = bool(hf_ok and perceptual_ok)

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
            sample_rate=self.sample_rate,
            scorer=self._perceptual_scorer,
            candidate_score=cand_score,
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
        include_studio_pro: bool | None = None,
        include_dereverb: bool | None = None,
    ) -> dict[Route, CandidatePreview]:
        """Generate internally derived previews for routes."""
        want_studio_pro = (
            self.include_studio_pro if include_studio_pro is None else include_studio_pro
        )
        effective_dereverb = self.include_dereverb if include_dereverb is None else include_dereverb
        want_dereverb = want_studio_pro if effective_dereverb is None else effective_dereverb
        use_studio_pro = bool(want_studio_pro) and _is_studio_pro_registered()
        use_dereverb = bool(want_dereverb) and _is_dereverb_registered()
        eval_routes: list[Route] = list(ROUTES)
        if use_studio_pro:
            eval_routes.append("studio_pro")
        if use_dereverb:
            eval_routes.append("dereverb")
        eligibility = eligible_routes(
            acoustic_evidence,
            restore_policy=restore_policy,
            speaker_profile_id=speaker_profile_id,
            policy=policy,
            include_optional=use_studio_pro or use_dereverb,
        )
        ref_preview = extract_preview_slice(
            audio,
            sample_rate,
            max_duration_s=self.max_preview_seconds,
            target_sample_rate=self.sample_rate,
        )
        ref_score = self._perceptual_scorer.score(ref_preview, self.sample_rate)

        previews: dict[Route, CandidatePreview] = {}
        for route in eval_routes:
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
                _ref_preview=ref_preview,
                _ref_score=ref_score,
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
        ranker: SmartSafeRanker | None = None,
        policy: SmartSafePolicy = DEFAULT_POLICY,
        require_qualified_ranker: bool = False,
        include_studio_pro: bool | None = None,
        include_dereverb: bool | None = None,
    ) -> tuple[SmartSafeDecision, dict[Route, CandidatePreview]]:
        """Generate all previews internally and decide the optimal safe route."""
        effective_ranker = ranker
        if effective_ranker is None:
            effective_ranker = SmartSafeRanker(
                version="smart-safe-baseline-v1",
                artifact_sha256=hash_bytes(b"smart-safe-baseline-v1"),
                signed=False,
                qualified=False,
            )
        previews = self.generate_all_previews(
            audio,
            sample_rate,
            acoustic_evidence=acoustic_evidence,
            restore_policy=restore_policy,
            speaker_profile=speaker_profile,
            speaker_profile_id=speaker_profile_id,
            policy=policy,
            include_studio_pro=include_studio_pro,
            include_dereverb=include_dereverb,
        )
        candidates = [p.evidence for p in previews.values()]
        decision = decide_smart_safe(
            acoustic_evidence,
            candidates,
            restore_policy=restore_policy,
            speaker_profile_id=speaker_profile_id,
            ranker=effective_ranker,
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
    phonetic_guard: PhoneticStabilityGuard | None = None,
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
            reference_audio,
            master_audio,
            sample_rate,
            cutoff_hz,
            tolerance_rms=0.05,
            tolerance_stft=0.10,
            max_third_octave_deviation_db=6.0,
        )
        if not prot.passes_invariance:
            return (
                False,
                f"post-master protected-band violation: RMS error {prot.rms_waveform_error:.5f}",
            )

    # Artifact guard
    hf_detector = HighBandEventDetector(sample_rate=sample_rate)
    hf_res = hf_detector.evaluate(reference_audio, master_audio, cutoff_hz=cutoff_hz)
    hf_ok = (
        hf_res.passes_event_check
        if route in RESTORE_ROUTES
        else _enhancement_hf_artifact_ok(
            reference_audio, master_audio, sample_rate, cutoff_hz, hf_res
        )
    )
    if not hf_ok:
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
        spk_ok, sim = _speaker_similarity_passes(
            speaker_extractor, reference_audio, master_audio, sample_rate, threshold=0.80
        )
        if not spk_ok:
            return (
                False,
                f"post-master speaker identity divergence: similarity {sim:.3f} < 0.80",
            )

    # Content guard (spectral linguistic + active-speech phonetic stability)
    ling_guard = SoraniLinguisticGuard(sample_rate=sample_rate)
    ling_res = ling_guard.evaluate(reference_audio, master_audio)
    if not ling_res.passes_check or np.isnan(ling_res.divergence):
        return (
            False,
            f"post-master content guard failed: linguistic divergence {ling_res.divergence:.4f}",
        )

    active_phonetic_guard = (
        phonetic_guard
        if phonetic_guard is not None
        else PhoneticStabilityGuard(sample_rate=sample_rate)
    )
    phonetic_res = active_phonetic_guard.evaluate(
        reference_audio, master_audio, sample_rate=sample_rate
    )
    if not phonetic_res.passes_check:
        return (
            False,
            f"post-master phonetic guard failed: {phonetic_res.status} "
            f"(cosine {phonetic_res.cosine_similarity:.3f})",
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
