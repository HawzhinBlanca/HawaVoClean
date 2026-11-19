"""Guarded neural bandwidth extension (LavaSR v2, Apache-2.0) inside protected_band.

Runs ``LavaSR v2`` (Apache-2.0, Vocos-based 16 kHz -> 48 kHz bandwidth extender)
natively in pure PyTorch (zero ``LavaSR``, ``vocos``, ``torchaudio``, or ``pyyaml``
runtime dependencies) strictly inside :mod:`hawavoclean.restoration.protected_band`:

1. Observed spectrum below ``effective_cutoff_hz`` is preserved mathematically
   via :func:`compute_transition_mask` and :func:`merge_protected_spectrum`.
2. Every emitted active candidate (``strength > 0``) is verified via
   :func:`verify_protected_band_invariance` (``<= 6.0 dB`` max per-1/3-octave
   deviation below the protected boundary, plus RMS and STFT tolerances).
3. Any candidate that violates protected-band invariance or produces non-finite
   samples fails closed and reverts to original audio (``strength == 0.0``).

Attribution & Licensing:
- LavaSR v2 inference head & Linkwitz-Riley merge: Copyright Yatharth Sharma
  (Apache License 2.0, https://github.com/ysharma3501/LavaSR).
- Vocos backbone / ISTFT modules: Copyright Hubert Siuzdak / charactr-platform
  (MIT License, https://github.com/charactr-platform/vocos).

``torch`` is lazy-imported so base/CI installs import and test the guarded
pipeline cleanly without optional neural dependencies or weights.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from hawavoclean.audio.resample import resample_audio
from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file
from hawavoclean.logging import get_logger
from hawavoclean.paths import app_data_root, models_dir
from hawavoclean.restoration.base import (
    RestorationCandidate,
    RestorationRenderResult,
    Restorer,
)
from hawavoclean.restoration.protected_band import (
    compute_transition_mask,
    merge_protected_spectrum,
    verify_protected_band_invariance,
)
from hawavoclean.runtime import check_memory_budget, resolve_device

logger = get_logger("restoration.neural_bwe")

LAVASR_MODEL_NAME = "lavasr-v2-48k"
LAVASR_MODEL_LICENSE = "Apache-2.0"
LAVASR_MODEL_DIR_ENV = "HAWAVOCLEAN_LAVASR_MODEL_DIR"
LAVASR_V2_CONFIG_FILE = "config.yaml"
LAVASR_V2_WEIGHTS_FILE = "pytorch_model.bin"
LAVASR_V2_CONFIG_SHA256 = "0d970f5cdc1913730c417b49e476bb09bb8b874583d113f71a7f10c8bb3a4b7d"
LAVASR_V2_WEIGHTS_SHA256 = "d100db961b2c125d77a52a12215c689e44cd9926a72f117513395b7e25e6de12"
_LAVASR_INPUT_SR = 16000
_LAVASR_OUTPUT_SR = 48000
_REPO_ROOT = Path(__file__).resolve().parents[3]

_LAVASR_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}

BWEGeneratorFn = Callable[
    [np.ndarray[Any, np.dtype[np.float32]], int],
    np.ndarray[Any, np.dtype[np.float32]],
]


def _local_ai_models_dir() -> Path:
    return _REPO_ROOT / ".local-ai" / "models"


def _normalize_v2_dir(base: Path) -> Path:
    nested = base / "enhancer_v2"
    if (nested / LAVASR_V2_WEIGHTS_FILE).is_file():
        return nested
    return base


def lavasr_model_dir() -> Path:
    """Resolve the local LavaSR v2 ``enhancer_v2`` directory without network I/O.

    Resolution order:
    1. ``HAWAVOCLEAN_LAVASR_MODEL_DIR`` (exclusive override when set).
    2. ``HAWAVOCLEAN_MODEL_DIR / lavasr / enhancer_v2`` (exclusive when
       ``HAWAVOCLEAN_MODEL_DIR`` is set, preserving test isolation).
    3. Packaged ``models_dir() / lavasr / enhancer_v2``, user
       ``app_data_root() / models / lavasr / enhancer_v2``, repo-staged
       ``.local-ai/models/lavasr/enhancer_v2``, and local offline HuggingFace
       cache snapshots (never triggers network downloads).
    """
    explicit = os.environ.get(LAVASR_MODEL_DIR_ENV)
    if explicit:
        return _normalize_v2_dir(Path(explicit).resolve())

    if os.environ.get("HAWAVOCLEAN_MODEL_DIR"):
        return models_dir() / "lavasr" / "enhancer_v2"

    candidates: list[Path] = [
        models_dir() / "lavasr" / "enhancer_v2",
        app_data_root() / "models" / "lavasr" / "enhancer_v2",
        _local_ai_models_dir() / "lavasr" / "enhancer_v2",
    ]
    hf_snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--YatharthS--LavaSR"
    snapshots_root = hf_snapshots / "snapshots"
    if snapshots_root.is_dir():
        for snap in sorted(snapshots_root.iterdir(), reverse=True):
            candidates.append(snap / "enhancer_v2")

    for candidate in candidates:
        if (candidate / LAVASR_V2_WEIGHTS_FILE).is_file():
            return candidate
    return candidates[0]


def verify_lavasr_weights(model_dir: Path | None = None) -> tuple[bool, str]:
    """Verify LavaSR v2 config + checkpoint existence and SHA-256 digests.

    Returns ``(ok, path_free_message)``.
    """
    root = lavasr_model_dir() if model_dir is None else _normalize_v2_dir(model_dir)
    expected_files = (
        (LAVASR_V2_CONFIG_FILE, LAVASR_V2_CONFIG_SHA256),
        (LAVASR_V2_WEIGHTS_FILE, LAVASR_V2_WEIGHTS_SHA256),
    )
    for filename, expected_sha in expected_files:
        path = root / filename
        label = f"LavaSR/enhancer_v2/{filename}"
        if not path.is_file():
            return False, f"LavaSR v2 weights file missing ({label})"
        if hash_file(path) != expected_sha:
            return False, f"LavaSR v2 weights digest mismatch ({label})"
    return True, "LavaSR v2 weights verified (Apache-2.0)"


def probe_neural_bwe_runtime(model_dir: Path | None = None) -> tuple[bool, str]:
    """Model-cold readiness probe for native pure-PyTorch LavaSR v2 BWE."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch unavailable ({type(exc).__name__})"
    return verify_lavasr_weights(model_dir)


def _match_length(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    target_len: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if len(audio) >= target_len:
        return np.ascontiguousarray(audio[:target_len], dtype=np.float32)
    return np.ascontiguousarray(np.pad(audio, (0, target_len - len(audio))), dtype=np.float32)


def _parse_lavasr_v2_hparams(config_path: Path) -> dict[str, int | str]:
    """Extract Vocos BWE hyperparameters from LavaSR v2 ``config.yaml`` without PyYAML."""
    defaults: dict[str, int | str] = {
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 80,
        "padding": "same",
        "input_channels": 80,
        "dim": 512,
        "intermediate_dim": 1536,
        "num_layers": 8,
    }
    text = config_path.read_text(encoding="utf-8")
    int_keys = (
        "n_fft",
        "hop_length",
        "n_mels",
        "input_channels",
        "dim",
        "intermediate_dim",
        "num_layers",
    )
    for key in int_keys:
        match = re.search(rf"^\s*{key}\s*:\s*(\d+)\b", text, flags=re.MULTILINE)
        if match:
            defaults[key] = int(match.group(1))
    pad_match = re.search(
        r'^\s*padding\s*:\s*["\']?(same|center)["\']?\b', text, flags=re.MULTILINE
    )
    if pad_match:
        defaults["padding"] = pad_match.group(1)
    return defaults


def _build_native_lavasr_v2(
    hparams: dict[str, int | str],
    device: str,
) -> tuple[Any, Any]:
    """Construct pure-PyTorch Vocos BWE + Linkwitz-Riley refiner matching LavaSR v2."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    n_fft = int(hparams["n_fft"])
    hop_length = int(hparams["hop_length"])
    n_mels = int(hparams["n_mels"])
    padding = str(hparams["padding"])
    input_channels = int(hparams["input_channels"])
    dim = int(hparams["dim"])
    intermediate_dim = int(hparams["intermediate_dim"])
    num_layers = int(hparams["num_layers"])

    class _SpectrogramBuffers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", torch.hann_window(n_fft))

    class _MelScaleBuffers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("fb", torch.zeros(n_fft // 2 + 1, n_mels))

    class _MelSpecContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spectrogram = _SpectrogramBuffers()
            self.mel_scale = _MelScaleBuffers()

    class _MelSpectrogramFeatures(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mel_spec = _MelSpecContainer()

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            if padding == "same":
                pad = (n_fft - hop_length) // 2
                pad_mode = "reflect" if audio.shape[-1] > pad else "replicate"
                audio = F.pad(audio, (pad, pad), mode=pad_mode)
            window: torch.Tensor = self.mel_spec.spectrogram.window  # type: ignore[assignment]
            fb: torch.Tensor = self.mel_spec.mel_scale.fb  # type: ignore[assignment]
            stft = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=padding == "center",
                onesided=True,
                return_complex=True,
            )
            mel = torch.matmul(stft.abs().transpose(-1, -2), fb).transpose(-1, -2)
            return torch.log(torch.clamp(mel, min=1e-5))

    class _ConvNeXtBlock(nn.Module):
        def __init__(self, layer_scale_init_value: float) -> None:
            super().__init__()
            self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, intermediate_dim)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(intermediate_dim, dim)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.dwconv(x).transpose(1, 2)
            x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
            x = (self.gamma * x).transpose(1, 2)
            return residual + x

    class _VocosBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            scale = 1.0 / max(1, num_layers)
            self.convnext = nn.ModuleList([_ConvNeXtBlock(scale) for _ in range(num_layers)])
            self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
            for block in self.convnext:
                x = block(x)
            out: torch.Tensor = self.final_layer_norm(x.transpose(1, 2))
            return out

    class _ISTFT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("window", torch.hann_window(n_fft))

        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            window: torch.Tensor = self.window  # type: ignore[assignment]
            if padding == "center":
                return torch.istft(spec, n_fft, hop_length, n_fft, window, center=True)
            pad = (n_fft - hop_length) // 2
            n_frames = int(spec.shape[2])
            ifft = torch.fft.irfft(spec, n_fft, dim=1, norm="backward") * window[None, :, None]
            output_size = (n_frames - 1) * hop_length + n_fft
            y = F.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, n_fft),
                stride=(1, hop_length),
            )[:, 0, 0, pad:-pad]
            win_sq = window.square().expand(1, n_frames, -1).transpose(1, 2)
            win_env = F.fold(
                win_sq,
                output_size=(1, output_size),
                kernel_size=(1, n_fft),
                stride=(1, hop_length),
            ).squeeze()[pad:-pad]
            return y / win_env

    class _ISTFTHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out = nn.Linear(dim, n_fft + 2)
            self.istft = _ISTFT()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            proj = self.out(x).transpose(1, 2)
            mag, phase = proj.chunk(2, dim=1)
            mag = torch.exp(mag).clip(max=1e3)
            spec = mag * (torch.cos(phase) + 1j * torch.sin(phase))
            out: torch.Tensor = self.istft(spec)
            return out

    class _NativeVocosBWE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = _MelSpectrogramFeatures()
            self.backbone = _VocosBackbone()
            self.head = _ISTFTHead()

    class _FastLRMerge:
        """Linkwitz-Riley spectral crossover matching LavaSR ``FastLRMerge``."""

        def __init__(
            self,
            sample_rate: int = _LAVASR_OUTPUT_SR,
            cutoff: int = 4000,
            transition_bins: int = 256,
            target_device: str = "cpu",
        ) -> None:
            self.sample_rate = sample_rate
            self.cutoff = cutoff
            self.transition_bins = transition_bins
            self.device = torch.device(target_device)
            self.mask_cache: dict[tuple[int, int], torch.Tensor] = {}
            x = torch.linspace(-1.0, 1.0, steps=transition_bins, device=self.device)
            t = (x + 1.0) / 2.0
            self.fade_template = (3.0 * t**2 - 2.0 * t**3).to(torch.complex64)

        def _get_mask(self, n_bins: int, ndim: int) -> torch.Tensor:
            key = (n_bins, ndim)
            cached = self.mask_cache.get(key)
            if cached is not None:
                return cached
            if len(self.mask_cache) >= 16:
                self.mask_cache.clear()
            cutoff_bin = int((self.cutoff / (self.sample_rate / 2.0)) * n_bins)
            mask = torch.ones(n_bins, device=self.device, dtype=torch.complex64)
            half = self.transition_bins // 2
            start = max(0, cutoff_bin - half)
            end = min(n_bins, cutoff_bin + half)
            mask[:start] = 0
            mask[start:end] = self.fade_template[: end - start]
            mask[end:] = 1
            for _ in range(ndim - 1):
                mask = mask.unsqueeze(0)
            self.mask_cache[key] = mask
            return mask

        def __call__(self, pred: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            spec1 = torch.fft.rfft(pred, dim=-1)
            spec2 = torch.fft.rfft(base, dim=-1)
            mask = self._get_mask(int(spec1.size(-1)), int(spec1.ndim))
            merged = spec2 + (spec1 - spec2) * mask
            out: torch.Tensor = torch.fft.irfft(merged, n=int(pred.size(-1)), dim=-1)
            return out

    return _NativeVocosBWE(), _FastLRMerge(target_device=device)


def _load_lavasr_v2(device: str, model_dir: Path | None = None) -> tuple[Any, Any]:
    """Verify SHA-256 lock and load native pure-PyTorch LavaSR v2 strictly offline."""
    root = lavasr_model_dir() if model_dir is None else _normalize_v2_dir(model_dir)
    ok, reason = verify_lavasr_weights(root)
    if not ok:
        raise ModelProvenanceError(reason)

    cache_key = (str(root.resolve()), device, LAVASR_V2_WEIGHTS_SHA256)
    cached = _LAVASR_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch

    hparams = _parse_lavasr_v2_hparams(root / LAVASR_V2_CONFIG_FILE)
    state_dict = torch.load(
        str(root / LAVASR_V2_WEIGHTS_FILE),
        map_location="cpu",
        weights_only=True,
    )
    bwe_model, lr_refiner = _build_native_lavasr_v2(hparams, device)
    bwe_model.load_state_dict(state_dict, strict=True)
    bwe_model = bwe_model.eval().to(torch.device(device))
    _LAVASR_MODEL_CACHE[cache_key] = (bwe_model, lr_refiner)
    return bwe_model, lr_refiner


def _run_lavasr_v2_chunk(
    bwe_model: Any,
    lr_refiner: Any,
    device: str,
    chunk_48k: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Run LavaSR v2 on one 48 kHz mono chunk (16 kHz -> 48 kHz BWE)."""
    import torch

    orig_len = len(chunk_48k)
    if orig_len == 0:
        return np.zeros(0, dtype=np.float32)

    # Match LavaEnhance2(..., denoise=False): resample to 16 kHz bandlimited
    # representation, upsample back to 48 kHz for Vocos BWE feature extractor.
    down_16k = resample_audio(chunk_48k, _LAVASR_OUTPUT_SR, _LAVASR_INPUT_SR)
    up_48k = resample_audio(down_16k, _LAVASR_INPUT_SR, _LAVASR_OUTPUT_SR, target_samples=orig_len)

    with torch.no_grad():
        wav_t = torch.from_numpy(np.ascontiguousarray(up_48k, dtype=np.float32)).unsqueeze(0)
        wav_t = wav_t.to(torch.device(device))
        features_input = bwe_model.feature_extractor(wav_t)
        features = bwe_model.backbone(features_input)
        pred_audio = bwe_model.head(features)
        common = min(int(pred_audio.shape[1]), int(wav_t.shape[1]))
        refined = lr_refiner(pred_audio[:, :common].float(), wav_t[:, :common].float())
        out = refined.reshape(-1).detach().cpu().float().numpy()

    return _match_length(np.asarray(out, dtype=np.float32), orig_len)


def _chunked_generate(
    channel_48k: np.ndarray[Any, np.dtype[np.float32]],
    chunk_fn: Callable[
        [np.ndarray[Any, np.dtype[np.float32]]], np.ndarray[Any, np.dtype[np.float32]]
    ],
    chunk_samples: int,
    overlap_samples: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Bounded-memory raised-cosine overlap-add wrapper for long inputs."""
    n = len(channel_48k)
    if n <= chunk_samples:
        check_memory_budget("neural_bwe")
        return _match_length(chunk_fn(channel_48k), n)

    hop = max(1, chunk_samples - overlap_samples)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    covered = 0

    while pos < n:
        check_memory_budget("neural_bwe")
        end = min(pos + chunk_samples, n)
        gen_chunk = _match_length(chunk_fn(channel_48k[pos:end]), end - pos)
        xfade = max(0, min(covered - pos, end - pos)) if pos > 0 else 0
        if xfade > 0:
            phase = np.linspace(0.0, np.pi, xfade, endpoint=False, dtype=np.float32)
            fade_in = 0.5 * (1.0 - np.cos(phase))
            out[pos : pos + xfade] = (
                out[pos : pos + xfade] * (1.0 - fade_in) + gen_chunk[:xfade] * fade_in
            )
            out[pos + xfade : end] = gen_chunk[xfade:]
        else:
            out[pos:end] = gen_chunk
        covered = end
        if end >= n:
            break
        pos += hop

    return out


class NeuralBWERestorer(Restorer):
    """Guarded LavaSR v2 (Apache-2.0) bandwidth restorer with protected-band invariance."""

    def __init__(
        self,
        device: str = "auto",
        *,
        sample_rate: int = _LAVASR_OUTPUT_SR,
        n_fft: int = 2048,
        hop_length: int = 512,
        transition_hz: float = 500.0,
        max_third_octave_deviation_db: float = 6.0,
        tolerance_rms: float = 0.05,
        tolerance_stft: float = 0.10,
        chunk_duration_s: float = 12.0,
        overlap_duration_s: float = 1.0,
        model_dir: Path | None = None,
        generator: BWEGeneratorFn | None = None,
    ) -> None:
        if sample_rate != _LAVASR_OUTPUT_SR:
            raise ValueError(
                f"NeuralBWERestorer operates at {_LAVASR_OUTPUT_SR} Hz, got {sample_rate}"
            )
        if max_third_octave_deviation_db <= 0.0 or max_third_octave_deviation_db > 6.0:
            raise ValueError(
                "max_third_octave_deviation_db must be in (0.0, 6.0], "
                f"got {max_third_octave_deviation_db}"
            )
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transition_hz = transition_hz
        self.max_third_octave_deviation_db = max_third_octave_deviation_db
        self.tolerance_rms = tolerance_rms
        self.tolerance_stft = tolerance_stft
        self.chunk_duration_s = chunk_duration_s
        self.overlap_duration_s = overlap_duration_s
        self._model_dir = model_dir
        self._generator = generator
        self.device = resolve_device(device, preference=("mps", "cuda", "cpu")).resolved
        self._bwe_model: Any = None
        self._lr_refiner: Any = None

    def _ensure_backend(self) -> None:
        if self._generator is not None or self._bwe_model is not None:
            return
        self._bwe_model, self._lr_refiner = _load_lavasr_v2(self.device, self._model_dir)
        logger.info(f"Loaded {LAVASR_MODEL_NAME} on device={self.device!r}")

    def _generate_channel(
        self,
        ch_audio: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        chunk_samples = max(self.n_fft, int(round(self.chunk_duration_s * self.sample_rate)))
        overlap_samples = min(
            chunk_samples // 2,
            max(0, int(round(self.overlap_duration_s * self.sample_rate))),
        )
        if self._generator is not None:
            gen_fn = self._generator
            return _chunked_generate(
                ch_audio,
                lambda chunk: np.asarray(gen_fn(chunk, self.sample_rate), dtype=np.float32),
                chunk_samples=chunk_samples,
                overlap_samples=overlap_samples,
            )
        return _chunked_generate(
            ch_audio,
            lambda chunk: _run_lavasr_v2_chunk(
                self._bwe_model, self._lr_refiner, self.device, chunk
            ),
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
        )

    def _merge_channel_protected(
        self,
        observed: np.ndarray[Any, np.dtype[np.float32]],
        generated: np.ndarray[Any, np.dtype[np.float32]],
        effective_cutoff_hz: float,
        strength: float,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        n = len(observed)
        _, _, z_obs = signal.stft(
            observed,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
            padded=True,
        )
        _, _, z_gen = signal.stft(
            generated,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
            padded=True,
        )
        mask = compute_transition_mask(
            z_obs.shape[0],
            self.sample_rate,
            cutoff_hz=effective_cutoff_hz,
            transition_hz=self.transition_hz,
        )
        merged_z = merge_protected_spectrum(
            np.asarray(z_obs, dtype=np.complex64),
            np.asarray(z_gen, dtype=np.complex64),
            mask,
            strength=strength,
        )
        _, merged = signal.istft(
            merged_z,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=True,
        )
        return _match_length(np.asarray(merged, dtype=np.float32), n)

    def _verify_candidate(
        self,
        original: np.ndarray[Any, np.dtype[np.float32]],
        candidate: np.ndarray[Any, np.dtype[np.float32]],
        effective_cutoff_hz: float,
    ) -> tuple[bool, float]:
        if not np.all(np.isfinite(candidate)):
            return False, float("inf")
        verification = verify_protected_band_invariance(
            original,
            candidate,
            sample_rate=self.sample_rate,
            cutoff_hz=effective_cutoff_hz,
            transition_hz=self.transition_hz,
            tolerance_rms=self.tolerance_rms,
            tolerance_stft=self.tolerance_stft,
            max_third_octave_deviation_db=self.max_third_octave_deviation_db,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return verification.passes_invariance, float(verification.worst_band_energy_deviation_db)

    def render(
        self,
        audio_48k: np.ndarray[Any, Any],
        sample_rate: int,
        effective_cutoff_hz: float,
        speaker_id: str | None = None,  # noqa: ARG002
        speaker_embedding: np.ndarray[Any, Any] | None = None,  # noqa: ARG002
        f0_trajectory: np.ndarray[Any, Any] | None = None,  # noqa: ARG002
        vuv_mask: np.ndarray[Any, Any] | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,
        seed: int = 42,  # noqa: ARG002
    ) -> RestorationRenderResult:
        """Generate protected-band-verified high-band restoration candidates."""
        if strengths is None:
            strengths = [1.0, 0.75, 0.5, 0.25, 0.0]

        nyquist = self.sample_rate / 2.0
        cutoff_hz = float(np.clip(effective_cutoff_hz, 500.0, nyquist - 100.0))
        audio_in = np.asarray(audio_48k, dtype=np.float32)

        if audio_in.size == 0:
            cands = [
                RestorationCandidate(strength=s, audio=audio_in.copy(), cutoff_hz=cutoff_hz)
                for s in strengths
            ]
            return RestorationRenderResult(
                success=False,
                fallback_status="empty_input",
                model_name=LAVASR_MODEL_NAME,
                provider=self.device,
                solver="lavasr_v2_protected_band",
                candidates=cands,
                error_message="Input audio is empty",
            )

        is_multichannel = audio_in.ndim == 2
        channels_in = audio_in if is_multichannel else audio_in[np.newaxis, :]
        n_channels = int(channels_in.shape[0])
        orig_len = int(channels_in.shape[1])

        if sample_rate != self.sample_rate:
            channels_48k = np.stack(
                [
                    resample_audio(channels_in[c], sample_rate, self.sample_rate)
                    for c in range(n_channels)
                ],
                axis=0,
            )
        else:
            channels_48k = channels_in

        n_samples = int(channels_48k.shape[1])
        active_strengths = [s for s in strengths if s > 0.0]

        if n_samples < self.n_fft:
            return RestorationRenderResult(
                success=False,
                fallback_status="input_too_short",
                model_name=LAVASR_MODEL_NAME,
                provider=self.device,
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(strength=0.0, audio=audio_in.copy(), cutoff_hz=cutoff_hz)
                ],
                error_message=(
                    f"Audio length ({n_samples} samples) is shorter than analysis window "
                    f"({self.n_fft} samples)"
                ),
            )

        if not active_strengths:
            return RestorationRenderResult(
                success=False,
                fallback_status="no_active_strengths",
                model_name=LAVASR_MODEL_NAME,
                provider=self.device,
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(strength=0.0, audio=audio_in.copy(), cutoff_hz=cutoff_hz)
                ],
                error_message=None,
            )

        try:
            self._ensure_backend()
            generated_channels = [
                self._generate_channel(channels_48k[c]) for c in range(n_channels)
            ]
        except Exception as exc:
            return RestorationRenderResult(
                success=False,
                fallback_status="runtime_error",
                model_name=LAVASR_MODEL_NAME,
                provider=self.device,
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(strength=0.0, audio=audio_in.copy(), cutoff_hz=cutoff_hz)
                ],
                error_message=f"{type(exc).__name__}: {exc}",
            )

        candidates: list[RestorationCandidate] = []
        invariance_failures: list[str] = []

        for s in active_strengths:
            merged_channels: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            for c in range(n_channels):
                merged_channels.append(
                    self._merge_channel_protected(
                        channels_48k[c], generated_channels[c], cutoff_hz, strength=s
                    )
                )
            merged_48k = (
                np.stack(merged_channels, axis=0) if is_multichannel else merged_channels[0]
            )
            orig_48k = channels_48k if is_multichannel else channels_48k[0]

            passed, worst_db = self._verify_candidate(orig_48k, merged_48k, cutoff_hz)
            if not passed:
                invariance_failures.append(
                    f"strength={s:g} failed protected-band invariance (worst_band={worst_db:.2f} dB)"
                )
                continue

            if sample_rate != self.sample_rate:
                if is_multichannel:
                    out_audio = np.stack(
                        [
                            resample_audio(
                                merged_channels[c],
                                self.sample_rate,
                                sample_rate,
                                target_samples=orig_len,
                            )
                            for c in range(n_channels)
                        ],
                        axis=0,
                    )
                else:
                    out_audio = resample_audio(
                        merged_channels[0],
                        self.sample_rate,
                        sample_rate,
                        target_samples=orig_len,
                    )
            else:
                out_audio = merged_48k

            candidates.append(
                RestorationCandidate(
                    strength=s,
                    audio=np.ascontiguousarray(out_audio, dtype=np.float32),
                    cutoff_hz=cutoff_hz,
                    protected_band_error=worst_db,
                )
            )

        # Always append the 0.0 original fallback candidate (fail-closed contract).
        candidates.append(
            RestorationCandidate(
                strength=0.0,
                audio=audio_in.copy(),
                cutoff_hz=cutoff_hz,
                protected_band_error=0.0,
            )
        )

        has_active = any(c.strength > 0.0 for c in candidates)
        if has_active:
            return RestorationRenderResult(
                success=True,
                fallback_status="none",
                model_name=LAVASR_MODEL_NAME,
                provider=self.device,
                solver="lavasr_v2_protected_band",
                candidates=candidates,
                error_message=None,
            )

        return RestorationRenderResult(
            success=False,
            fallback_status="invariance_violation",
            model_name=LAVASR_MODEL_NAME,
            provider=self.device,
            solver="lavasr_v2_protected_band",
            candidates=candidates,
            error_message="; ".join(invariance_failures) or "All active candidates failed guard",
        )

    def restore(
        self,
        audio_48k: np.ndarray[Any, Any],
        sample_rate: int,
        effective_cutoff_hz: float,
        speaker_id: str | None = None,
        speaker_embedding: np.ndarray[Any, Any] | None = None,
        f0_trajectory: np.ndarray[Any, Any] | None = None,
        vuv_mask: np.ndarray[Any, Any] | None = None,
        strengths: list[float] | None = None,
        seed: int = 42,
    ) -> list[RestorationCandidate]:
        """Generate protected-band-verified restoration candidates across strengths."""
        return self.render(
            audio_48k=audio_48k,
            sample_rate=sample_rate,
            effective_cutoff_hz=effective_cutoff_hz,
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
            f0_trajectory=f0_trajectory,
            vuv_mask=vuv_mask,
            strengths=strengths,
            seed=seed,
        ).candidates


__all__ = [
    "BWEGeneratorFn",
    "LAVASR_MODEL_DIR_ENV",
    "LAVASR_MODEL_LICENSE",
    "LAVASR_MODEL_NAME",
    "LAVASR_V2_CONFIG_FILE",
    "LAVASR_V2_CONFIG_SHA256",
    "LAVASR_V2_WEIGHTS_FILE",
    "LAVASR_V2_WEIGHTS_SHA256",
    "NeuralBWERestorer",
    "lavasr_model_dir",
    "probe_neural_bwe_runtime",
    "verify_lavasr_weights",
]
