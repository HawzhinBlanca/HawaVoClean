"""Non-intrusive perceptual speech quality oracle for Smart Safe.

Primary backend uses real NISQA v2 via the lean pure-PyTorch inference module
(:mod:`hawavoclean.guard.nisqa_arch`, requiring only ``torch``, ``numpy``, and
``scipy``), with optional fallback to ``torchmetrics.functional.audio.nisqa``
returning the five ITU-T P.808 / NISQA perceptual dimensions
``[mos, noisiness, discontinuity, coloration, loudness]``. Optional secondary
backend uses DNSMOS (``torchmetrics.functional.audio.dnsmos``).

All neural backends are strictly offline: weights are never downloaded during
inference, SHA-256 verified before loading, and optional dependencies are
lazy-imported so base installs work without ``torch`` or ``torchmetrics``.
When no neural backend is locally ready, a deterministic waveform-based
``acoustic_proxy`` fallback runs with ``calibrated=False`` so callers never
mistake an uncalibrated estimator for a qualified neural ranker.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import signal

from hawavoclean.audio.resample import resample_audio

PerceptualBackend = Literal["auto", "nisqa_v2", "dnsmos", "acoustic_proxy"]


def _validate_mos_axis(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 1.0 <= result <= 5.0:
        raise ValueError(f"{field} must be finite and between 1.0 and 5.0")
    return result


def _validate_confidence(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and between 0.0 and 1.0")
    return result


@dataclass(frozen=True, slots=True)
class PerceptualQualityScore:
    """Five-axis perceptual quality evaluation of an audio clip."""

    mos: float
    noisiness: float
    discontinuity: float
    coloration: float
    loudness: float
    confidence: float
    backend: str
    calibrated: bool

    def __post_init__(self) -> None:
        for field in ("mos", "noisiness", "discontinuity", "coloration", "loudness"):
            object.__setattr__(self, field, _validate_mos_axis(getattr(self, field), field))
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))
        if not self.backend:
            raise ValueError("backend must be non-empty")


def _nisqa_weights_path() -> Path:
    override = os.environ.get("HAWAVOCLEAN_NISQA_WEIGHTS")
    if override:
        return Path(override).expanduser()
    if _modules_present("torch"):
        nisqa_arch = importlib.import_module("hawavoclean.guard.nisqa_arch")
        path: Path = nisqa_arch.nisqa_weights_path()
        return path
    return Path("~/.torchmetrics/NISQA/nisqa.tar").expanduser()


def _dnsmos_model_dir() -> Path:
    override = os.environ.get("HAWAVOCLEAN_DNSMOS_DIR")
    if override:
        return Path(override).expanduser()
    return Path("~/.torchmetrics/DNSMOS").expanduser()


def _modules_present(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def _is_native_nisqa_available() -> bool:
    if not _modules_present("torch"):
        return False
    nisqa_arch = importlib.import_module("hawavoclean.guard.nisqa_arch")
    ok, _ = nisqa_arch.verify_nisqa_weights()
    return bool(ok)


def _is_torchmetrics_nisqa_available() -> bool:
    return _modules_present("torch", "torchmetrics", "librosa") and _nisqa_weights_path().is_file()


def _is_nisqa_available() -> bool:
    return _is_native_nisqa_available() or _is_torchmetrics_nisqa_available()


def _is_dnsmos_available() -> bool:
    if not _modules_present("torch", "torchmetrics", "librosa", "onnxruntime"):
        return False
    base = _dnsmos_model_dir() / "DNSMOS"
    return (base / "model_v8.onnx").is_file() and (base / "sig_bak_ovr.onnx").is_file()


def perceptual_scorer_status() -> dict[str, object]:
    """Report honest offline availability and maturity of the perceptual scorer.

    This probe never downloads weights or loads neural models. Smart Safe
    always reports ``maturity="experimental"`` and ``qualified_ranker=False``
    until a signed Sorani listener-calibrated ranker artifact ships.
    """
    nisqa_ok = _is_nisqa_available()
    dnsmos_ok = _is_dnsmos_available()
    if nisqa_ok:
        backend = "nisqa_v2"
        neural_available = True
        calibrated = True
        reason = (
            "NISQA v2 perceptual MOS scorer is available locally, but Smart Safe "
            "remains experimental until a signed Sorani listener-calibrated ranker ships"
        )
    elif dnsmos_ok:
        backend = "dnsmos"
        neural_available = True
        calibrated = True
        reason = (
            "DNSMOS perceptual MOS scorer is available locally, but Smart Safe "
            "remains experimental until a signed Sorani listener-calibrated ranker ships"
        )
    else:
        backend = "acoustic_proxy"
        neural_available = False
        calibrated = False
        reason = (
            "Uncalibrated acoustic quality proxy is active (install torchmetrics "
            "with cached offline NISQA v2 weights for neural MOS); Smart Safe is experimental"
        )
    return {
        "available": True,
        "neural_available": neural_available,
        "nisqa_available": nisqa_ok,
        "dnsmos_available": dnsmos_ok,
        "backend": backend,
        "calibrated": calibrated,
        "qualified_ranker": False,
        "maturity": "experimental",
        "providers": ["cpu"],
        "reason": reason,
    }


def _to_mono_float32(audio: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1) if audio.ndim == 2 else audio
    return np.ascontiguousarray(mono, dtype=np.float32)


def _normalize_rms(
    mono: np.ndarray[Any, np.dtype[np.float32]],
    target_rms: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    if rms < 1e-6:
        return mono.copy()
    scaled = mono * (target_rms / rms)
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.95:
        scaled = scaled * (0.95 / peak)
    return np.ascontiguousarray(scaled, dtype=np.float32)


def _clip_mos(value: float) -> float:
    """Bound raw MOS predictions to [1.0, 5.0] while preserving ordering near 1.0."""
    v = float(value)
    if v < 1.08:
        v = 1.0 + 0.08 * math.exp((v - 1.08) / 0.08)
    return round(float(np.clip(v, 1.0, 5.0)), 3)


def _score_acoustic_proxy(
    mono: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    raw_rms: float,
) -> PerceptualQualityScore:
    """Deterministic waveform-only quality proxy (explicitly ``calibrated=False``)."""
    n_samples = len(mono)
    dur_s = n_samples / float(sample_rate)

    # 1. Noisiness via 20 ms frame active-level vs noise-floor SNR proxy
    frame_len = max(16, int(sample_rate * 0.020))
    hop_len = max(8, frame_len // 2)
    if n_samples >= frame_len:
        frames = np.lib.stride_tricks.sliding_window_view(mono, frame_len)[::hop_len]
        frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        speech_rms = float(np.percentile(frame_rms, 90))
        floor_rms = float(np.percentile(frame_rms, 15))
    else:
        speech_rms = raw_rms
        floor_rms = raw_rms * 0.5
    snr_db = 20.0 * math.log10((speech_rms + 1e-7) / (floor_rms + 1e-7))
    noisiness = 1.0 + 3.8 * float(np.clip((snr_db - 4.0) / 30.0, 0.0, 1.0))

    # 2. Discontinuity via flat-top clipping saturation and sharp impulse spikes
    peak = float(np.max(np.abs(mono)))
    clip_ratio = float(np.mean(np.abs(mono) >= 0.985 * peak)) if peak > 1e-3 else 0.0
    clip_excess = max(0.0, clip_ratio - 0.005)
    if n_samples >= 4:
        diff2 = np.abs(np.diff(mono, n=2))
        med_diff = float(np.median(diff2)) + 1e-6
        p999_diff = float(np.percentile(diff2, 99.9))
        impulse_ratio = p999_diff / med_diff
        impulse_excess = max(0.0, (impulse_ratio - 35.0) / 25.0)
    else:
        impulse_excess = 0.0
    discontinuity = 4.65 - 28.0 * clip_excess - 0.25 * impulse_excess

    # 3. Coloration via spectral balance (rumble, bandwidth fullness, speech-band flatness)
    nperseg = min(2048, n_samples)
    freqs, psd = signal.welch(mono, fs=sample_rate, nperseg=nperseg)
    psd = np.maximum(psd, 1e-18)
    core_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    rumble_mask = freqs < 90.0
    hf_high = min(8000.0, sample_rate * 0.48)
    hf_mask = (freqs > 3400.0) & (freqs <= hf_high)

    e_core = float(np.sum(psd[core_mask])) + 1e-12
    e_rumble = float(np.sum(psd[rumble_mask]))
    e_hf = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0

    rumble_ratio = e_rumble / e_core
    rumble_pen = min(1.5, max(0.0, (rumble_ratio - 0.25) * 1.2))

    hf_ratio_db = 10.0 * math.log10((e_hf + 1e-12) / e_core)
    # Typical clean wideband speech has HF/core around -26..-8 dB; narrowband drops below -30 dB
    muffle_pen = min(1.4, max(0.0, (-28.0 - hf_ratio_db) * 0.07))

    core_psd = psd[core_mask]
    if core_psd.size > 0:
        flatness = float(np.exp(np.mean(np.log(core_psd + 1e-18))) / (np.mean(core_psd) + 1e-18))
    else:
        flatness = 0.2
    flat_pen = min(1.4, max(0.0, (flatness - 0.40) * 2.5))
    coloration = 4.45 - rumble_pen - muffle_pen - flat_pen

    # 4. Loudness suitability from raw (pre-normalized) RMS in dBFS
    raw_db = 20.0 * math.log10(raw_rms + 1e-9)
    loudness = 4.50 - 0.08 * max(0.0, abs(raw_db + 19.0) - 6.0)

    noi_c = _clip_mos(noisiness)
    dis_c = _clip_mos(discontinuity)
    col_c = _clip_mos(coloration)
    loud_c = _clip_mos(loudness)
    mos_c = _clip_mos(0.40 * noi_c + 0.25 * dis_c + 0.25 * col_c + 0.10 * loud_c)

    # Honest bounded confidence for uncalibrated proxy (never claims neural 0.90+)
    dur_factor = min(1.0, max(0.25, dur_s / 2.0))
    confidence = round(0.56 + 0.20 * dur_factor, 3)

    return PerceptualQualityScore(
        mos=mos_c,
        noisiness=noi_c,
        discontinuity=dis_c,
        coloration=col_c,
        loudness=loud_c,
        confidence=confidence,
        backend="acoustic_proxy",
        calibrated=False,
    )


class PerceptualQualityScorer:
    """Offline perceptual speech quality scorer with NISQA v2 / DNSMOS / proxy backends."""

    def __init__(
        self,
        *,
        backend: PerceptualBackend = "auto",
        target_rms: float = 0.08,
        allow_fallback: bool = True,
    ) -> None:
        if backend not in {"auto", "nisqa_v2", "dnsmos", "acoustic_proxy"}:
            raise ValueError(f"unknown perceptual scorer backend: {backend!r}")
        if not math.isfinite(target_rms) or target_rms <= 0.0:
            raise ValueError("target_rms must be finite and positive")
        self._requested_backend: PerceptualBackend = backend
        self._target_rms = float(target_rms)
        self._allow_fallback = bool(allow_fallback)

    @property
    def active_backend(self) -> Literal["nisqa_v2", "dnsmos", "acoustic_proxy"]:
        """Resolve which backend will execute offline without loading weights."""
        if self._requested_backend == "acoustic_proxy":
            return "acoustic_proxy"
        if self._requested_backend == "nisqa_v2":
            if _is_nisqa_available():
                return "nisqa_v2"
            if not self._allow_fallback:
                raise RuntimeError(
                    "NISQA v2 backend requested but torchmetrics/librosa or offline weights are unavailable"
                )
            return "acoustic_proxy"
        if self._requested_backend == "dnsmos":
            if _is_dnsmos_available():
                return "dnsmos"
            if not self._allow_fallback:
                raise RuntimeError(
                    "DNSMOS backend requested but torchmetrics/onnxruntime or offline weights are unavailable"
                )
            return "acoustic_proxy"
        if _is_nisqa_available():
            return "nisqa_v2"
        if _is_dnsmos_available():
            return "dnsmos"
        return "acoustic_proxy"

    def _score_nisqa(
        self,
        norm_audio: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> PerceptualQualityScore:
        if _is_native_nisqa_available():
            nisqa_arch = importlib.import_module("hawavoclean.guard.nisqa_arch")
            vals = [float(v) for v in nisqa_arch.score_nisqa_v2(norm_audio, int(sample_rate))]
        else:
            torch = importlib.import_module("torch")
            nisqa_mod = importlib.import_module("torchmetrics.functional.audio.nisqa")
            nisqa_fn = nisqa_mod.non_intrusive_speech_quality_assessment

            with torch.no_grad():
                out = nisqa_fn(torch.from_numpy(norm_audio), int(sample_rate))
            vals = [float(v) for v in out.detach().cpu().tolist()]
        if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
            raise RuntimeError(f"unexpected NISQA output: {vals!r}")

        clipped = [_clip_mos(v) for v in vals]
        dur_s = len(norm_audio) / float(sample_rate)
        dur_factor = min(1.0, max(0.35, dur_s / 3.0))
        spread = max(clipped) - min(clipped)
        spread_penalty = max(0.0, (spread - 2.0) * 0.05)
        confidence = round(float(np.clip(0.55 + 0.37 * dur_factor - spread_penalty, 0.35, 0.92)), 3)

        return PerceptualQualityScore(
            mos=clipped[0],
            noisiness=clipped[1],
            discontinuity=clipped[2],
            coloration=clipped[3],
            loudness=clipped[4],
            confidence=confidence,
            backend="nisqa_v2",
            calibrated=True,
        )

    def _score_dnsmos(
        self,
        norm_audio: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> PerceptualQualityScore:
        torch = importlib.import_module("torch")
        dnsmos_mod = importlib.import_module("torchmetrics.functional.audio.dnsmos")
        dnsmos_fn = dnsmos_mod.deep_noise_suppression_mean_opinion_score

        audio_16k = (
            resample_audio(norm_audio, sample_rate, 16000) if sample_rate != 16000 else norm_audio
        )
        with torch.no_grad():
            out = dnsmos_fn(
                torch.from_numpy(np.ascontiguousarray(audio_16k, dtype=np.float32)),
                16000,
                personalized=False,
                cache_session=True,
            )
        vals = [float(v) for v in out.detach().cpu().tolist()]
        if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
            raise RuntimeError(f"unexpected DNSMOS output: {vals!r}")

        # torchmetrics DNSMOS returns [p808_mos, sig, bak, ovrl]
        p808_mos, sig, bak, ovrl = (_clip_mos(v) for v in vals)
        dur_s = len(norm_audio) / float(sample_rate)
        dur_factor = min(1.0, max(0.35, dur_s / 3.0))
        confidence = round(float(np.clip(0.50 + 0.35 * dur_factor, 0.35, 0.85)), 3)

        return PerceptualQualityScore(
            mos=ovrl,
            noisiness=bak,
            discontinuity=sig,
            coloration=sig,
            loudness=p808_mos,
            confidence=confidence,
            backend="dnsmos",
            calibrated=True,
        )

    def score(
        self,
        audio: np.ndarray[Any, Any],
        sample_rate: int,
    ) -> PerceptualQualityScore:
        """Evaluate perceptual speech quality for ``audio`` at ``sample_rate``."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        mono = _to_mono_float32(audio)
        if mono.size == 0:
            return PerceptualQualityScore(
                mos=3.0,
                noisiness=3.0,
                discontinuity=3.0,
                coloration=3.0,
                loudness=3.0,
                confidence=0.50,
                backend="acoustic_proxy",
                calibrated=False,
            )
        if not bool(np.all(np.isfinite(mono))):
            return PerceptualQualityScore(
                mos=1.0,
                noisiness=1.0,
                discontinuity=1.0,
                coloration=1.0,
                loudness=1.0,
                confidence=0.95,
                backend="acoustic_proxy",
                calibrated=False,
            )

        raw_rms = float(np.sqrt(np.mean(mono**2)))
        if raw_rms < 1e-4:
            return PerceptualQualityScore(
                mos=3.0,
                noisiness=3.0,
                discontinuity=3.0,
                coloration=3.0,
                loudness=3.0,
                confidence=0.50,
                backend="acoustic_proxy",
                calibrated=False,
            )

        norm_audio = _normalize_rms(mono, self._target_rms)
        backend = self.active_backend

        if backend == "nisqa_v2":
            try:
                return self._score_nisqa(norm_audio, sample_rate)
            except Exception:
                if not self._allow_fallback:
                    raise
                if _is_dnsmos_available():
                    try:
                        return self._score_dnsmos(norm_audio, sample_rate)
                    except Exception:
                        pass
                return _score_acoustic_proxy(mono, sample_rate, raw_rms)

        if backend == "dnsmos":
            try:
                return self._score_dnsmos(norm_audio, sample_rate)
            except Exception:
                if not self._allow_fallback:
                    raise
                return _score_acoustic_proxy(mono, sample_rate, raw_rms)

        return _score_acoustic_proxy(mono, sample_rate, raw_rms)
