"""Sorani-safe active-speech phonetic stability guard.

Primary backend uses offline ``microsoft/wavlm-base-plus`` middle-layer
(layer 6) frame cosine similarity computed ONLY on active speech onset/nucleus
frames, ignoring silence and exponentially decaying reverb tails so legitimate
dereverberation does not trigger false-revert cascades.

When ``transformers`` or local WavLM weights are unavailable (for example in
minimal CI environments), a deterministic Mel-cepstral + temporal-modulation
fallback (``backend="mfcc_active_speech"``) runs on the exact same active
speech onset/nucleus mask, strictly catching consonant/formant collapse and
dropout without tripping on reverb tail removal.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import signal
from scipy.fft import dct

from hawavoclean.audio.resample import resample_audio

PhoneticBackend = Literal["auto", "wavlm_layer6", "mfcc_active_speech"]
PhoneticDevice = Literal["auto", "cpu", "mps", "cuda"]

DEFAULT_WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
_INTERNAL_SR = 16000
_FRAME_WIN = 400  # 25 ms at 16 kHz
_FRAME_HOP = 320  # 20 ms at 16 kHz (exact WavLM 50 Hz frame rate)
_MAX_ALIGN_SHIFT = 800  # 50 ms bounded delay alignment

_WAVLM_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _validate_cosine(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not -1.0001 <= result <= 1.0001:
        raise ValueError(f"{field} must be finite and within [-1.0, 1.0]")
    return float(np.clip(result, -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class PhoneticStabilityResult:
    """Result of active-speech phonetic stability verification."""

    cosine_similarity: float
    min_frame_cosine: float
    active_speech_frames: int
    backend: str
    passes_check: bool
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cosine_similarity",
            _validate_cosine(self.cosine_similarity, "cosine_similarity"),
        )
        object.__setattr__(
            self,
            "min_frame_cosine",
            _validate_cosine(self.min_frame_cosine, "min_frame_cosine"),
        )
        if self.active_speech_frames < 0:
            raise ValueError("active_speech_frames must be non-negative")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.status:
            raise ValueError("status must be non-empty")


def _resolve_model_ref(model_path: str | Path | None) -> str:
    if model_path is not None:
        return str(Path(model_path).expanduser()) if isinstance(model_path, Path) else model_path
    env_override = os.environ.get("HAWAVOCLEAN_WAVLM_MODEL_PATH")
    if env_override:
        return env_override
    return DEFAULT_WAVLM_MODEL_ID


def _hf_hub_cache_dir() -> Path:
    for env_var in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        raw = os.environ.get(env_var)
        if raw:
            return Path(raw).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def _has_local_wavlm_weights(model_ref: str) -> bool:
    candidate_dir = Path(model_ref).expanduser()
    if candidate_dir.is_dir():
        return (candidate_dir / "config.json").is_file()
    if "/" not in model_ref:
        return False
    cache_folder = _hf_hub_cache_dir() / f"models--{model_ref.replace('/', '--')}" / "snapshots"
    if not cache_folder.is_dir():
        return False
    return any((snap / "config.json").is_file() for snap in cache_folder.iterdir() if snap.is_dir())


def _is_wavlm_available(model_ref: str) -> bool:
    if (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("transformers") is None
    ):
        return False
    return _has_local_wavlm_weights(model_ref)


def phonetic_guard_status(model_path: str | Path | None = None) -> dict[str, object]:
    """Report offline availability of WavLM vs MFCC active-speech phonetic guard."""
    model_ref = _resolve_model_ref(model_path)
    wavlm_ok = _is_wavlm_available(model_ref)
    return {
        "backend": "wavlm_layer6" if wavlm_ok else "mfcc_active_speech",
        "wavlm_available": wavlm_ok,
        "model_ref": model_ref,
    }


def _to_mono_float32(audio: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    mono = np.mean(audio, axis=0 if audio.shape[0] <= 2 else 1) if audio.ndim == 2 else audio
    return np.ascontiguousarray(mono, dtype=np.float32)


def _delay_align_16k(
    ref: np.ndarray[Any, np.dtype[np.float32]],
    cand: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    n = min(len(ref), len(cand))
    r = ref[:n]
    c = cand[:n]
    if n < _FRAME_WIN:
        return r, c
    r_ac = (r - float(np.mean(r))).astype(np.float64)
    c_ac = (c - float(np.mean(c))).astype(np.float64)
    if float(np.sqrt(np.mean(r_ac**2))) < 1e-5 or float(np.sqrt(np.mean(c_ac**2))) < 1e-5:
        return r, c
    corr = signal.correlate(c_ac, r_ac, mode="full", method="fft")
    mid = n - 1
    max_shift = min(_MAX_ALIGN_SHIFT, n // 4)
    lo = max(0, mid - max_shift)
    hi = min(len(corr), mid + max_shift + 1)
    shift = int(np.argmax(corr[lo:hi]) + lo - mid)
    if shift > 0:
        return r[: n - shift], c[shift:n]
    if shift < 0:
        s = -shift
        return r[s:n], c[: n - s]
    return r, c


def _frame_ac_rms(
    x: np.ndarray[Any, np.dtype[np.float32]],
    n_frames: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    frames = np.lib.stride_tricks.sliding_window_view(x, _FRAME_WIN)[::_FRAME_HOP][:n_frames]
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=1) + 1e-12).astype(np.float32)


def _active_onset_nucleus_mask(
    ref_16k: np.ndarray[Any, np.dtype[np.float32]],
    cand_16k: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, np.dtype[np.float32]],
]:
    """Identify reference active speech onset/nucleus frames while excluding reverb decay tails.

    The selection mask depends strictly on the reference signal so a broken
    enhancer that erases speech cannot evade verification by outputting silence.
    """
    n = min(len(ref_16k), len(cand_16k))
    n_frames = max(1, (n - _FRAME_WIN) // _FRAME_HOP + 1)
    r_rms = _frame_ac_rms(ref_16k[:n], n_frames)
    c_rms = _frame_ac_rms(cand_16k[:n], n_frames)

    r_floor = float(np.percentile(r_rms, 15))
    p85 = float(np.percentile(r_rms, 85))
    # SNR-aware floor: exclude continuous background noise pauses (where r_rms sits
    # at r_floor) so strong denoisers that clean pauses are never falsely penalized.
    floor = max(1e-4, r_floor + 0.35 * max(0.0, p85 - r_floor), p85 * 0.22)
    active = r_rms > floor
    if not bool(np.any(active)):
        return active, c_rms

    keep = active.copy()
    for idx in range(1, n_frames):
        if not active[idx]:
            continue
        prev_peak = float(np.max(r_rms[max(0, idx - 18) : idx]))
        is_strong_onset = r_rms[idx] > 1.15 * r_rms[idx - 1] and r_rms[idx] >= 0.45 * prev_peak
        if (not is_strong_onset) and r_rms[idx] < 0.65 * prev_peak:
            keep[idx] = False
    return keep, c_rms


@lru_cache(maxsize=2)
def _mel_filterbank(
    sample_rate: int = _INTERNAL_SR,
    n_fft: int = 512,
    n_mels: int = 40,
    fmin: float = 80.0,
    fmax: float = 7600.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        for k in range(left, center):
            fb[i, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fb[i, k] = (right - k) / max(1, right - center)
    return fb


def _extract_mfcc_modulation(
    x_16k: np.ndarray[Any, np.dtype[np.float32]],
    n_frames: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    frames = np.lib.stride_tricks.sliding_window_view(x_16k, _FRAME_WIN)[::_FRAME_HOP][:n_frames]
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    window = np.hanning(_FRAME_WIN).astype(np.float32)
    power = np.abs(np.fft.rfft(centered * window, n=512, axis=-1)) ** 2
    fb = _mel_filterbank()
    log_mel = np.log(power @ fb.T + 1e-8)
    cep = np.asarray(dct(log_mel, type=2, axis=-1, norm="ortho"), dtype=np.float32)[:, 1:17]
    delta = np.zeros_like(cep)
    if n_frames >= 3:
        delta[1:-1] = 0.5 * (cep[2:] - cep[:-2])
    return np.concatenate([cep, 1.5 * delta], axis=-1).astype(np.float32)


def _normalize_active_rms(
    x: np.ndarray[Any, np.dtype[np.float32]],
    target_rms: float = 0.08,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    if rms < 1e-6:
        return x.copy()
    return np.ascontiguousarray(x * (target_rms / rms), dtype=np.float32)


class PhoneticStabilityGuard:
    """Active-speech phonetic stability guard (WavLM layer 6 with MFCC fallback)."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        backend: PhoneticBackend = "auto",
        model_path: str | Path | None = None,
        device: PhoneticDevice = "auto",
        layer_index: int = 6,
        min_cosine: float | None = None,
        min_p05_cosine: float | None = None,
        allow_fallback: bool = True,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if backend not in {"auto", "wavlm_layer6", "mfcc_active_speech"}:
            raise ValueError(f"unknown phonetic guard backend: {backend!r}")
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"unknown device: {device!r}")
        if layer_index < 1:
            raise ValueError("layer_index must be >= 1")
        self.sample_rate = sample_rate
        self._requested_backend: PhoneticBackend = backend
        self._model_ref = _resolve_model_ref(model_path)
        self._device: PhoneticDevice = device
        self._layer_index = layer_index
        self._min_cosine_override = min_cosine
        self._min_p05_override = min_p05_cosine
        self._allow_fallback = bool(allow_fallback)

    @property
    def active_backend(self) -> Literal["wavlm_layer6", "mfcc_active_speech"]:
        """Resolve active phonetic backend without loading neural weights."""
        if self._requested_backend == "mfcc_active_speech":
            return "mfcc_active_speech"
        if _is_wavlm_available(self._model_ref):
            return "wavlm_layer6"
        if self._requested_backend == "wavlm_layer6" and not self._allow_fallback:
            raise RuntimeError(
                f"WavLM backend requested but transformers/local weights are unavailable for {self._model_ref!r}"
            )
        return "mfcc_active_speech"

    def _resolve_torch_device(self, torch_mod: Any) -> str:
        if self._device != "auto":
            return self._device
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return "mps"
        if torch_mod.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_wavlm(self) -> tuple[Any, Any, str]:
        torch_mod = importlib.import_module("torch")
        device_str = self._resolve_torch_device(torch_mod)
        cache_key = (self._model_ref, device_str)
        model = _WAVLM_MODEL_CACHE.get(cache_key)
        if model is None:
            tf_mod = importlib.import_module("transformers")
            model = tf_mod.WavLMModel.from_pretrained(self._model_ref, local_files_only=True)
            model.eval().to(torch_mod.device(device_str))
            _WAVLM_MODEL_CACHE[cache_key] = model
        return model, torch_mod, device_str

    def _frame_cosines_wavlm(
        self,
        ref_16k: np.ndarray[Any, np.dtype[np.float32]],
        cand_16k: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        model, torch_mod, device_str = self._load_wavlm()
        ref_norm = _normalize_active_rms(ref_16k)
        cand_norm = _normalize_active_rms(cand_16k)
        dev = torch_mod.device(device_str)
        with torch_mod.no_grad():
            tr = torch_mod.from_numpy(ref_norm).unsqueeze(0).to(dev)
            tc = torch_mod.from_numpy(cand_norm).unsqueeze(0).to(dev)
            hr_states = model(tr, output_hidden_states=True).hidden_states
            hc_states = model(tc, output_hidden_states=True).hidden_states
            layer = min(self._layer_index, len(hr_states) - 1)
            hr = hr_states[layer].squeeze(0)
            hc = hc_states[layer].squeeze(0)
            t_frames = min(int(hr.shape[0]), int(hc.shape[0]))
            cos = torch_mod.nn.functional.cosine_similarity(hr[:t_frames], hc[:t_frames], dim=-1)
            return np.asarray(cos.detach().cpu().numpy(), dtype=np.float32)

    def _frame_cosines_mfcc(
        self,
        ref_16k: np.ndarray[Any, np.dtype[np.float32]],
        cand_16k: np.ndarray[Any, np.dtype[np.float32]],
        n_frames: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        fa = _extract_mfcc_modulation(ref_16k, n_frames)
        fb = _extract_mfcc_modulation(cand_16k, n_frames)
        denom = np.linalg.norm(fa, axis=-1) * np.linalg.norm(fb, axis=-1) + 1e-8
        return np.asarray(np.sum(fa * fb, axis=-1) / denom, dtype=np.float32)

    def _thresholds_for(self, backend: str) -> tuple[float, float]:
        if backend == "wavlm_layer6":
            default_mean, default_p05 = 0.52, 0.18
        else:
            default_mean, default_p05 = 0.72, 0.25
        return (
            default_mean if self._min_cosine_override is None else float(self._min_cosine_override),
            default_p05 if self._min_p05_override is None else float(self._min_p05_override),
        )

    def evaluate(
        self,
        reference_audio: np.ndarray[Any, Any],
        candidate_audio: np.ndarray[Any, Any],
        *,
        sample_rate: int | None = None,
    ) -> PhoneticStabilityResult:
        """Compare active-speech phonetic representations of reference and candidate."""
        sr = self.sample_rate if sample_rate is None else int(sample_rate)
        if sr <= 0:
            raise ValueError("sample_rate must be positive")

        ref_mono = _to_mono_float32(reference_audio)
        cand_mono = _to_mono_float32(candidate_audio)
        backend = self.active_backend

        if not (bool(np.all(np.isfinite(ref_mono))) and bool(np.all(np.isfinite(cand_mono)))):
            return PhoneticStabilityResult(
                cosine_similarity=0.0,
                min_frame_cosine=0.0,
                active_speech_frames=0,
                backend=backend,
                passes_check=False,
                status="non_finite_audio",
            )

        min_len = min(len(ref_mono), len(cand_mono))
        min_samples_sr = int(math.ceil((_FRAME_WIN / float(_INTERNAL_SR)) * sr))
        if min_len < max(16, min_samples_sr):
            return PhoneticStabilityResult(
                cosine_similarity=1.0,
                min_frame_cosine=1.0,
                active_speech_frames=0,
                backend=backend,
                passes_check=True,
                status="audio_too_short",
            )

        ref_16k = (
            resample_audio(ref_mono[:min_len], sr, _INTERNAL_SR)
            if sr != _INTERNAL_SR
            else ref_mono[:min_len]
        )
        cand_16k = (
            resample_audio(cand_mono[:min_len], sr, _INTERNAL_SR)
            if sr != _INTERNAL_SR
            else cand_mono[:min_len]
        )
        ref_16k, cand_16k = _delay_align_16k(
            np.ascontiguousarray(ref_16k, dtype=np.float32),
            np.ascontiguousarray(cand_16k, dtype=np.float32),
        )
        if len(ref_16k) < _FRAME_WIN or len(cand_16k) < _FRAME_WIN:
            return PhoneticStabilityResult(
                cosine_similarity=1.0,
                min_frame_cosine=1.0,
                active_speech_frames=0,
                backend=backend,
                passes_check=True,
                status="audio_too_short",
            )

        active_mask, cand_rms = _active_onset_nucleus_mask(ref_16k, cand_16k)
        n_active = int(np.sum(active_mask))
        if n_active == 0:
            return PhoneticStabilityResult(
                cosine_similarity=1.0,
                min_frame_cosine=1.0,
                active_speech_frames=0,
                backend=backend,
                passes_check=True,
                status="no_active_speech",
            )

        used_backend: str = backend
        if backend == "wavlm_layer6":
            try:
                frame_cos = self._frame_cosines_wavlm(ref_16k, cand_16k)
            except Exception:
                if not self._allow_fallback:
                    raise
                used_backend = "mfcc_active_speech"
                frame_cos = self._frame_cosines_mfcc(ref_16k, cand_16k, len(active_mask))
        else:
            frame_cos = self._frame_cosines_mfcc(ref_16k, cand_16k, len(active_mask))

        t_common = min(len(frame_cos), len(active_mask))
        frame_cos = np.clip(frame_cos[:t_common], -1.0, 1.0).copy()
        mask = active_mask[:t_common]
        # Candidate silence or zero-AC energy on active speech frames is an explicit dropout.
        frame_cos[cand_rms[:t_common] < 1e-4] = 0.0

        active_vals = frame_cos[mask]
        if active_vals.size == 0:
            return PhoneticStabilityResult(
                cosine_similarity=1.0,
                min_frame_cosine=1.0,
                active_speech_frames=0,
                backend=used_backend,
                passes_check=True,
                status="no_active_speech",
            )

        mean_cos = float(np.mean(active_vals))
        p05_cos = float(np.percentile(active_vals, 5))
        min_cos = float(np.min(active_vals))
        threshold_mean, threshold_p05 = self._thresholds_for(used_backend)

        passes = (mean_cos >= threshold_mean) and (p05_cos >= threshold_p05)
        if passes:
            status = "phonetic_preserved"
        elif mean_cos < threshold_mean:
            status = "phonetic_drift_detected"
        else:
            status = "phonetic_collapse_detected"

        return PhoneticStabilityResult(
            cosine_similarity=round(mean_cos, 4),
            min_frame_cosine=round(min_cos, 4),
            active_speech_frames=int(active_vals.size),
            backend=used_backend,
            passes_check=passes,
            status=status,
        )
