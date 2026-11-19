"""Studio Pro flagship 48 kHz speech enhancement core (MossFormer2_SE_48K).

Uses strictly Apache-2.0 ``MossFormer2_SE_48K`` (Alibaba ClearerVoice-Studio)
directly in memory at native 48 kHz with bounded-memory raised-cosine
overlap-add chunking, ``check_memory_budget("studio_pro")`` self-policing,
and SHA-256 weight verification (``torch.load(..., weights_only=True)``) prior
to loading.

Phase 1a/1c shootout measurements on real 48 kHz Sorani Kurdish speech
(``data/kurdish_speakers/``) established that standalone ``MossFormer2_SE_48K``
on Apple Silicon MPS runs at ~0.023 RTF (13.3x faster than CPU), leaves clean
speech transparent (SI-SDR 43.0 dB, PESQ 4.63, WavLM cos 1.0000), and
outperforms WPE+MF2 (which degraded clean SI-SDR to 22.8 dB) and generative
Resemble-Enhance CFM (which hallucinated over Kurdish speech). No dereverb
pre-stage is bundled.

All optional neural imports (``torch``, ``torchaudio``, and the in-tree
pure-PyTorch ``hawavoclean.enhancement.mossformer2_arch``) are lazy so base/CI
installs import this module and verify provenance cleanly with zero
``clearvoice`` dependency.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.alignment.delay import estimate_gcc_phat_delay
from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer, EnhancerMetadata
from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.logging import get_logger
from hawavoclean.paths import app_data_root, models_dir
from hawavoclean.runtime import active_device, check_memory_budget, resolve_device

logger = get_logger("studio-pro-core")

STUDIO_PRO_CORE_ID = "studio-mossformer2-48k-v1"
STUDIO_PRO_MODEL_DIR_ENV = "HAWAVOCLEAN_STUDIO_PRO_MODEL_DIR"
STUDIO_PRO_CHECKPOINT_FILE = "last_best_checkpoint.pt"
STUDIO_PRO_CHECKPOINT_SHA256 = "03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6"

_MAX_WAV_VALUE = 32768.0
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Frozen parameters. Changing any entry is a new core and requires relocking
# studio-pro-core.lock.toml (verified by audit-models and pipeline preflight).
STUDIO_PRO_PARAMS: dict[str, float | int | bool | str] = {
    "model": "MossFormer2_SE_48K",
    "model_license": "Apache-2.0",
    "internal_sample_rate": 48000,
    "win_type": "hamming",
    "win_len": 1920,
    "win_inc": 384,
    "fft_len": 1920,
    "num_mels": 60,
    "chunk_duration_s": 12.0,
    "overlap_duration_s": 1.5,
    "checkpoint_file": STUDIO_PRO_CHECKPOINT_FILE,
    "checkpoint_sha256": STUDIO_PRO_CHECKPOINT_SHA256,
}


def studio_pro_params_hash() -> str:
    """Canonical hash of Studio Pro's frozen parameters and pinned checkpoint digest.

    Importable without torch or torchaudio so provenance can be audited in any
    base install.
    """
    return hash_json_canonical(dict(STUDIO_PRO_PARAMS))


def studio_pro_checkpoint_path() -> Path:
    """Resolve the local MossFormer2_SE_48K checkpoint path without network I/O.

    Resolution order:
    1. ``HAWAVOCLEAN_STUDIO_PRO_MODEL_DIR`` (exclusive override when set).
    2. ``HAWAVOCLEAN_MODEL_DIR / mossformer2_se_48k`` (exclusive when
       ``HAWAVOCLEAN_MODEL_DIR`` is set, preserving test isolation).
    3. Packaged ``models_dir() / mossformer2_se_48k``, per-user
       ``app_data_root() / models / mossformer2_se_48k``, local workspace
       ``.local-ai/models/mossformer2_se_48k``, and local checkout
       ``checkpoints/MossFormer2_SE_48K``.
    """
    filename = str(STUDIO_PRO_PARAMS["checkpoint_file"])

    explicit = os.environ.get(STUDIO_PRO_MODEL_DIR_ENV)
    if explicit:
        return Path(explicit).resolve() / filename

    if os.environ.get("HAWAVOCLEAN_MODEL_DIR"):
        return models_dir() / "mossformer2_se_48k" / filename

    candidates = (
        models_dir() / "mossformer2_se_48k" / filename,
        app_data_root() / "models" / "mossformer2_se_48k" / filename,
        _REPO_ROOT / ".local-ai" / "models" / "mossformer2_se_48k" / filename,
        _REPO_ROOT / "checkpoints" / "MossFormer2_SE_48K" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def verify_studio_pro_weights(checkpoint_path: Path | None = None) -> tuple[bool, str]:
    """Verify MossFormer2_SE_48K checkpoint existence and pinned SHA-256 digest.

    Returns ``(ok, path_free_message)`` so callers in capability probes never
    leak local filesystem paths.
    """
    path = studio_pro_checkpoint_path() if checkpoint_path is None else checkpoint_path
    label = f"MossFormer2_SE_48K/{STUDIO_PRO_PARAMS['checkpoint_file']}"
    if not path.is_file():
        return False, f"Studio Pro weights file missing ({label})"
    expected = str(STUDIO_PRO_PARAMS["checkpoint_sha256"])
    actual = hash_file(path)
    if actual != expected:
        return False, f"Studio Pro weights digest mismatch ({label})"
    return True, f"Studio Pro weights verified ({label})"


def blend_wet_dry(
    original: np.ndarray[Any, np.dtype[np.float32]],
    enhanced: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    strength: float,
    max_delay_ms: float = 50.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Delay-aligned parallel wet/dry blend using GCC-PHAT alignment.

    Aligns ``enhanced`` to ``original`` before mixing so non-phase-coherent
    neural cores do not introduce latency-driven comb filtering at partial
    strengths.
    """
    if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError(f"strength must be finite and within [0.0, 1.0], got {strength!r}")
    orig32 = np.ascontiguousarray(original, dtype=np.float32)
    enh32 = np.ascontiguousarray(enhanced, dtype=np.float32)
    n = min(len(orig32), len(enh32))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    orig32 = orig32[:n]
    enh32 = enh32[:n]

    if strength <= 0.0:
        return orig32.copy()
    if strength >= 1.0:
        return enh32.copy()

    aligned = estimate_gcc_phat_delay(
        orig32, enh32, sample_rate=sample_rate, max_delay_ms=max_delay_ms
    ).aligned_candidate[:n]
    blended = (1.0 - float(strength)) * orig32 + float(strength) * aligned
    return np.ascontiguousarray(blended, dtype=np.float32)


def load_mossformer2_se_48k(
    device: str,
    checkpoint_path: Path | None = None,
) -> Any:
    """Verify SHA-256 lock and load pure-PyTorch MossFormer2_SE_48K strictly offline."""
    path = studio_pro_checkpoint_path() if checkpoint_path is None else checkpoint_path
    ok, reason = verify_studio_pro_weights(path)
    if not ok:
        raise ModelProvenanceError(reason)

    import torch

    arch_mod = importlib.import_module("hawavoclean.enhancement.mossformer2_arch")
    net = arch_mod.MossFormer2SE48K()
    raw_ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    pretrained = (
        raw_ckpt["model"] if isinstance(raw_ckpt, dict) and "model" in raw_ckpt else raw_ckpt
    )

    target_state = net.state_dict()
    loaded_state: dict[str, Any] = {}
    for key, tensor in target_state.items():
        if key in pretrained and pretrained[key].shape == tensor.shape:
            loaded_state[key] = pretrained[key]
        elif key.removeprefix("module.") in pretrained and (
            pretrained[key.removeprefix("module.")].shape == tensor.shape
        ):
            loaded_state[key] = pretrained[key.removeprefix("module.")]
        elif f"module.{key}" in pretrained and pretrained[f"module.{key}"].shape == tensor.shape:
            loaded_state[key] = pretrained[f"module.{key}"]
        else:
            raise ModelProvenanceError(
                f"MossFormer2_SE_48K checkpoint missing or mismatched parameter: {key}"
            )
    net.load_state_dict(loaded_state, strict=True)
    net.eval()
    net.to(torch.device(device))
    return net


def run_mossformer2_chunk(
    model: Any,
    device: str,
    chunk_48k: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Run MossFormer2_SE_48K inference on a single 48 kHz mono chunk."""
    import torch

    arch_mod = importlib.import_module("hawavoclean.enhancement.mossformer2_arch")

    orig_len = len(chunk_48k)
    if orig_len == 0:
        return np.zeros(0, dtype=np.float32)

    sr = int(STUDIO_PRO_PARAMS["internal_sample_rate"])
    win_len = int(STUDIO_PRO_PARAMS["win_len"])
    win_inc = int(STUDIO_PRO_PARAMS["win_inc"])
    fft_len = int(STUDIO_PRO_PARAMS["fft_len"])
    num_mels = int(STUDIO_PRO_PARAMS["num_mels"])
    win_type = str(STUDIO_PRO_PARAMS["win_type"])

    work = np.ascontiguousarray(chunk_48k, dtype=np.float32)
    if len(work) < win_len:
        pad_len = win_len - len(work)
    else:
        rem = (len(work) - win_len) % win_inc
        pad_len = (win_inc - rem) if rem else 0
    if pad_len > 0:
        work = np.pad(work, (0, pad_len), mode="constant")

    audio_scaled = torch.from_numpy(work * _MAX_WAV_VALUE)

    with torch.no_grad():
        features = arch_mod.extract_mossformer2_features(
            audio_scaled,
            sample_rate=sr,
            win_len=win_len,
            win_inc=win_inc,
            num_mels=num_mels,
            win_type=win_type,
        )
        out_list = model(features.to(torch.device(device)))
        out = arch_mod.apply_mossformer2_mask(
            audio_scaled,
            out_list[-1],
            fft_len=fft_len,
            win_len=win_len,
            win_inc=win_inc,
        )

    out32 = np.asarray(out[:orig_len], dtype=np.float32)
    if len(out32) < orig_len:
        out32 = np.pad(out32, (0, orig_len - len(out32)), mode="constant")
    return np.ascontiguousarray(out32, dtype=np.float32)


def _chunked_overlap_add(
    audio_48k: np.ndarray[Any, np.dtype[np.float32]],
    chunk_fn: Any,
    chunk_samples: int,
    overlap_samples: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Process arbitrary-length 48 kHz audio in bounded-memory raised-cosine chunks."""
    n = len(audio_48k)
    if n <= chunk_samples:
        check_memory_budget("studio_pro")
        return np.ascontiguousarray(chunk_fn(audio_48k), dtype=np.float32)

    hop = max(1, chunk_samples - overlap_samples)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    covered = 0

    while pos < n:
        check_memory_budget("studio_pro")
        end = min(pos + chunk_samples, n)
        enhanced_chunk = np.ascontiguousarray(chunk_fn(audio_48k[pos:end]), dtype=np.float32)
        xfade = max(0, min(covered - pos, end - pos)) if pos > 0 else 0
        if xfade > 0:
            # Partition-of-unity raised-cosine crossfade across the overlapped seam.
            phase = np.linspace(0.0, np.pi, xfade, endpoint=False, dtype=np.float32)
            fade_in = 0.5 * (1.0 - np.cos(phase))
            out[pos : pos + xfade] = (
                out[pos : pos + xfade] * (1.0 - fade_in) + enhanced_chunk[:xfade] * fade_in
            )
            out[pos + xfade : end] = enhanced_chunk[xfade:]
        else:
            out[pos:end] = enhanced_chunk
        covered = end
        if end >= n:
            break
        pos += hop

    return out


class StudioProCore(Enhancer):
    """Flagship native 48 kHz MossFormer2_SE_48K speech enhancement core."""

    def __init__(
        self,
        core_id: str = STUDIO_PRO_CORE_ID,
        sample_rate: int = 48000,
        phase_coherent: bool = False,
        device: str | None = None,
    ) -> None:
        if phase_coherent:
            raise ValueError(
                "StudioProCore is not phase-coherent (spectral masking modifies phase); "
                "configure phase_coherent = false"
            )
        internal_sr = int(STUDIO_PRO_PARAMS["internal_sample_rate"])
        if sample_rate != internal_sr:
            raise ValueError(
                f"StudioProCore runs at {internal_sr} Hz internally; "
                f"model_sample_rate must be {internal_sr}, got {sample_rate}"
            )
        self._core_id = core_id
        self._sample_rate = internal_sr
        self._device = (
            active_device() if device is None else resolve_device(device, core_id=core_id).resolved
        )
        self._metadata = EnhancerMetadata(
            core_id=core_id,
            version="1.0.0",
            algorithm=(
                "MossFormer2_SE_48K native 48 kHz neural speech enhancement "
                "(Apache-2.0, hash-locked weights)"
            ),
            sample_rate=self._sample_rate,
            phase_coherent=False,
            params_hash=studio_pro_params_hash(),
        )
        self._model: Any = None

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    @property
    def device(self) -> str:
        """The compute device this core's inference runs on."""
        return self._device

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._model = load_mossformer2_se_48k(self._device)
        logger.info(f"Loaded StudioProCore ({self._core_id}) on device={self._device!r}")

    def warmup(self) -> None:
        self._ensure_model()
        dummy = np.zeros(self._sample_rate // 2, dtype=np.float32)
        self.enhance(dummy, self._sample_rate)

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        check_memory_budget("studio_pro")
        t_start = time.perf_counter()
        orig_len = len(waveform)
        if orig_len == 0:
            return EnhancementResult(
                waveform=waveform.copy(),
                sample_rate=sample_rate,
                model_runtime_ms=0.0,
                input_samples=0,
                output_samples=0,
            )

        self._ensure_model()
        audio_48k = resample_audio(waveform, sample_rate, self._sample_rate)
        chunk_samples = int(round(float(STUDIO_PRO_PARAMS["chunk_duration_s"]) * self._sample_rate))
        overlap_samples = int(
            round(float(STUDIO_PRO_PARAMS["overlap_duration_s"]) * self._sample_rate)
        )
        enhanced_48k = _chunked_overlap_add(
            audio_48k,
            lambda chunk: run_mossformer2_chunk(self._model, self._device, chunk),
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
        )

        enhanced_out = resample_audio(
            enhanced_48k, self._sample_rate, sample_rate, target_samples=orig_len
        )
        peak = float(np.max(np.abs(enhanced_out))) if len(enhanced_out) > 0 else 0.0
        if peak > 0.99:
            enhanced_out = np.ascontiguousarray(enhanced_out * (0.99 / peak), dtype=np.float32)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return EnhancementResult(
            waveform=enhanced_out,
            sample_rate=sample_rate,
            model_runtime_ms=t_elapsed_ms,
            input_samples=orig_len,
            output_samples=len(enhanced_out),
        )


__all__ = [
    "STUDIO_PRO_CHECKPOINT_FILE",
    "STUDIO_PRO_CHECKPOINT_SHA256",
    "STUDIO_PRO_CORE_ID",
    "STUDIO_PRO_MODEL_DIR_ENV",
    "STUDIO_PRO_PARAMS",
    "StudioProCore",
    "blend_wet_dry",
    "load_mossformer2_se_48k",
    "run_mossformer2_chunk",
    "studio_pro_checkpoint_path",
    "studio_pro_params_hash",
    "verify_studio_pro_weights",
]
