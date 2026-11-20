# Copyright (c) 2023 ESPnet Contributors (TFGridNetV3, Apache License 2.0)
# Copyright (c) 2026 Rong Xiaobin (GAP-URGENet Predictor, MIT License)
# Copyright (c) 2026 HawaVoClean Contributors (inference-only pure-PyTorch module)
#
# Licensed under the Apache License, Version 2.0 (architecture) and MIT License
# (pretrained GAP-URGENet Predictor weights).

"""Lean pure-PyTorch neural mono speech dereverberation (TF-GridNetV3 + guarded BWE).

Uses strictly permissive weights:
- **Dereverberation core**: ``Xiaobin-Rong/gap-urgenet/Predictor.pt`` (88.5 MB,
  **MIT License**) running the **Apache-2.0** ESPnet ``TFGridNetV3`` pure
  discriminative STFT architecture at 16 kHz with bounded-memory raised-cosine
  overlap-add chunking, ``check_memory_budget("dereverb_neural")``, and
  strict ``torch.load(..., weights_only=True)`` + SHA-256 lock verification.
- **Optional full-band 48 kHz extension**: composes with HawaVoClean's in-tree
  **Apache-2.0** ``LavaSR v2`` (:class:`~hawavoclean.restoration.neural_bwe.NeuralBWERestorer`)
  strictly inside :mod:`hawavoclean.restoration.protected_band` (``cutoff_hz = 7200``),
  so 0..7.2 kHz stays mathematically locked to the discriminative dereverberator
  and fails closed to direct resampled TF-GridNet output if BWE weights are
  unavailable or fail protected-band invariance.

Item 4 shootout results on real 48 kHz Sorani Kurdish speech
(``research/benchmarks/dereverb_2026_report.md``) verified that this combination
clears every production gate with zero Kurdish hallucination:
- ``reverb_heavy`` (DRR +1 dB, RT60 1.1 s): NISQA 3.63, ESTOI 0.666, WavLM 0.8273
- ``reverb_plus_noise`` (RT60 0.8 s, +7 dB): NISQA 3.78, ESTOI 0.639, WavLM 0.7639
- ``clean`` control: WavLM 0.9972, ESTOI 1.000
- Steady-state Apple M5 Max MPS RTF: ~0.12 (max ~0.26 <= 0.35 gate).

``torch`` is lazy-imported so base/CI installs import and probe this module
cleanly without optional neural dependencies or weights.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal

from hawavoclean.alignment.delay import estimate_gcc_phat_delay
from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer, EnhancerMetadata
from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.logging import get_logger
from hawavoclean.paths import app_data_root, models_dir
from hawavoclean.restoration.neural_bwe import NeuralBWERestorer, probe_neural_bwe_runtime
from hawavoclean.runtime import active_device, check_memory_budget, resolve_device

logger = get_logger("enhancement.dereverb_neural")

STUDIO_DEREVERB_CORE_ID = "studio-dereverb-48k-v1"
DEREVERB_MODEL_ID = "gap-urgenet-predictor-tfgridnet-v1"
DEREVERB_MODEL_LICENSE = "MIT"
DEREVERB_ARCH_LICENSE = "Apache-2.0"
DEREVERB_MODEL_DIR_ENV = "HAWAVOCLEAN_DEREVERB_MODEL_DIR"
DEREVERB_CHECKPOINT_FILE = "Predictor.pt"
DEREVERB_CHECKPOINT_SHA256 = "1e58c270c12ab953e4df32a34e4b580f780d016bee9bf5580684d6c982f77866"

_INTERNAL_SAMPLE_RATE = 16000
_BWE_SAMPLE_RATE = 48000
_SILENCE_RMS_THRESHOLD = 1e-8
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEREVERB_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}

STUDIO_DEREVERB_PARAMS: dict[str, float | int | bool | str] = {
    "core_id": STUDIO_DEREVERB_CORE_ID,
    "model_id": DEREVERB_MODEL_ID,
    "model_license": DEREVERB_MODEL_LICENSE,
    "arch_license": DEREVERB_ARCH_LICENSE,
    "internal_sample_rate": _INTERNAL_SAMPLE_RATE,
    "output_sample_rate": _BWE_SAMPLE_RATE,
    "fft_len_s": 0.032,
    "hop_len_s": 0.016,
    "win_len_s": 0.032,
    "n_layers": 10,
    "lstm_hidden_units": 200,
    "attn_n_head": 8,
    "attn_qk_output_channel": 4,
    "emb_dim": 96,
    "emb_ks": 4,
    "emb_hs": 1,
    "chunk_duration_s": 12.0,
    "overlap_duration_s": 1.5,
    "enable_bwe": False,
    "bwe_cutoff_hz": 7200.0,
    "checkpoint_file": DEREVERB_CHECKPOINT_FILE,
    "checkpoint_sha256": DEREVERB_CHECKPOINT_SHA256,
}
DEREVERB_PARAMS = STUDIO_DEREVERB_PARAMS

DereverbChunkFn = Callable[
    [np.ndarray[Any, np.dtype[np.float32]]],
    np.ndarray[Any, np.dtype[np.float32]],
]


@dataclass(frozen=True)
class DereverbResult:
    """Output waveform and telemetry from :class:`NeuralDereverberator`."""

    waveform: np.ndarray[Any, np.dtype[np.float32]]
    sample_rate: int
    device: str
    bwe_applied: bool
    protected_band_err_db: float | None
    model_runtime_ms: float
    input_samples: int
    output_samples: int


def studio_dereverb_params_hash() -> str:
    """Canonical hash of frozen ``studio-dereverb-48k-v1`` parameters and SHA-256 digest."""
    return hash_json_canonical(dict(STUDIO_DEREVERB_PARAMS))


def dereverb_params_hash() -> str:
    """Alias for :func:`studio_dereverb_params_hash`."""
    return studio_dereverb_params_hash()


def _local_ai_models_dir() -> Path:
    return _REPO_ROOT / ".local-ai" / "models"


def dereverb_checkpoint_path() -> Path:
    """Resolve the local ``Predictor.pt`` checkpoint path without network I/O.

    Resolution order:
    1. ``HAWAVOCLEAN_DEREVERB_MODEL_DIR`` (exclusive override; accepts a file or
       directory containing ``Predictor.pt``).
    2. ``HAWAVOCLEAN_MODEL_DIR / gap_urgenet_predictor / Predictor.pt``
       (exclusive when ``HAWAVOCLEAN_MODEL_DIR`` is set, preserving test isolation).
    3. Packaged ``models_dir()``, user ``app_data_root() / models``, and repo-staged
       ``.local-ai/models/gap_urgenet_predictor/Predictor.pt``.
    """
    filename = str(DEREVERB_PARAMS["checkpoint_file"])

    explicit = os.environ.get(DEREVERB_MODEL_DIR_ENV)
    if explicit:
        resolved = Path(explicit).resolve()
        return resolved if resolved.is_file() else resolved / filename

    if os.environ.get("HAWAVOCLEAN_MODEL_DIR"):
        return models_dir() / "gap_urgenet_predictor" / filename

    candidates = (
        models_dir() / "gap_urgenet_predictor" / filename,
        app_data_root() / "models" / "gap_urgenet_predictor" / filename,
        _local_ai_models_dir() / "gap_urgenet_predictor" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def verify_dereverb_weights(checkpoint_path: Path | None = None) -> tuple[bool, str]:
    """Verify ``Predictor.pt`` existence and pinned SHA-256 digest.

    Returns ``(ok, path_free_message)`` so callers never leak local paths.
    """
    path = dereverb_checkpoint_path() if checkpoint_path is None else checkpoint_path
    label = f"GAP-URGENet-Predictor/{DEREVERB_PARAMS['checkpoint_file']}"
    if not path.is_file():
        return False, f"Neural dereverb weights file missing ({label})"
    expected = str(DEREVERB_PARAMS["checkpoint_sha256"])
    if hash_file(path) != expected:
        return False, f"Neural dereverb weights digest mismatch ({label})"
    return True, f"Neural dereverb weights verified ({label}, MIT/Apache-2.0)"


def probe_dereverb_runtime(checkpoint_path: Path | None = None) -> tuple[bool, str]:
    """Model-cold readiness probe for native pure-PyTorch neural dereverberation."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch unavailable ({type(exc).__name__})"
    return verify_dereverb_weights(checkpoint_path)


def _match_length(
    audio: np.ndarray[Any, np.dtype[np.float32]],
    target_len: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if len(audio) >= target_len:
        return np.ascontiguousarray(audio[:target_len], dtype=np.float32)
    return np.ascontiguousarray(np.pad(audio, (0, target_len - len(audio))), dtype=np.float32)


def _build_tfgridnet_v3(cfg: dict[str, Any]) -> Any:
    """Construct pure-PyTorch ESPnet TFGridNetV3 matching ``Predictor.pt``."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    fft_len = float(cfg.get("fft_len", 0.032))
    hop_len = float(cfg.get("hop_len", 0.016))
    win_len = float(cfg.get("win_len", 0.032))
    n_layers = int(cfg.get("n_layers", 10))
    lstm_hidden = int(cfg.get("lstm_hidden_units", 200))
    n_head = int(cfg.get("attn_n_head", 8))
    qk_ch = int(cfg.get("attn_qk_output_channel", 4))
    emb_dim = int(cfg.get("emb_dim", 96))
    emb_ks = int(cfg.get("emb_ks", 4))
    emb_hs = int(cfg.get("emb_hs", 1))
    eps = float(cfg.get("eps", 1e-5))

    if emb_ks <= emb_hs or emb_dim % n_head != 0:
        raise ValueError(
            f"Unsupported TFGridNetV3 geometry: emb_dim={emb_dim}, n_head={n_head}, "
            f"emb_ks={emb_ks}, emb_hs={emb_hs}"
        )

    class _LayerNorm4D(nn.Module):
        def __init__(self, input_dim: int, norm_eps: float) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(1, input_dim, 1, 1, dtype=torch.float32))
            self.beta = nn.Parameter(torch.zeros(1, input_dim, 1, 1, dtype=torch.float32))
            self.eps = norm_eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mu = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, unbiased=False, keepdim=True) + self.eps)
            return ((x - mu) / std) * self.gamma + self.beta

    class _AllHeadPReLULayerNorm4DC(nn.Module):
        def __init__(self, heads: int, channels: int, norm_eps: float) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(1, heads, channels, 1, 1, dtype=torch.float32))
            self.beta = nn.Parameter(torch.zeros(1, heads, channels, 1, 1, dtype=torch.float32))
            self.act = nn.PReLU(num_parameters=heads, init=0.25)
            self.eps = norm_eps
            self.heads = heads
            self.channels = channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, _, frames, freqs = x.shape
            y: torch.Tensor = self.act(x.view(bsz, self.heads, self.channels, frames, freqs))
            mu = y.mean(dim=2, keepdim=True)
            std = torch.sqrt(y.var(dim=2, unbiased=False, keepdim=True) + self.eps)
            out: torch.Tensor = ((y - mu) / std) * self.gamma + self.beta
            return out

    class _GridNetV3Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_ch = emb_dim * emb_ks
            self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
            self.intra_rnn = nn.LSTM(in_ch, lstm_hidden, 1, batch_first=True, bidirectional=True)
            self.intra_linear = nn.ConvTranspose1d(lstm_hidden * 2, emb_dim, emb_ks, stride=emb_hs)

            self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
            self.inter_rnn = nn.LSTM(in_ch, lstm_hidden, 1, batch_first=True, bidirectional=True)
            self.inter_linear = nn.ConvTranspose1d(lstm_hidden * 2, emb_dim, emb_ks, stride=emb_hs)

            self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * qk_ch, 1)
            self.attn_norm_Q = _AllHeadPReLULayerNorm4DC(n_head, qk_ch, eps)
            self.attn_conv_K = nn.Conv2d(emb_dim, n_head * qk_ch, 1)
            self.attn_norm_K = _AllHeadPReLULayerNorm4DC(n_head, qk_ch, eps)
            self.attn_conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
            self.attn_norm_V = _AllHeadPReLULayerNorm4DC(n_head, emb_dim // n_head, eps)
            self.attn_concat_proj = nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                nn.PReLU(),
                _LayerNorm4D(emb_dim, eps),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz, ch, old_t, old_q = x.shape
            olp = emb_ks - emb_hs
            t_pad = math.ceil((old_t + 2 * olp - emb_ks) / emb_hs) * emb_hs + emb_ks
            q_pad = math.ceil((old_q + 2 * olp - emb_ks) / emb_hs) * emb_hs + emb_ks

            padded = F.pad(
                x.permute(0, 2, 3, 1),
                (0, 0, olp, q_pad - old_q - olp, olp, t_pad - old_t - olp),
            )

            # Intra-frame frequency BiLSTM
            intra = self.intra_norm(padded).reshape(bsz * t_pad, q_pad, ch).transpose(1, 2)
            intra = F.unfold(intra[..., None], (emb_ks, 1), stride=(emb_hs, 1)).transpose(1, 2)
            intra, _ = self.intra_rnn(intra)
            intra = self.intra_linear(intra.transpose(1, 2)).reshape(bsz, t_pad, ch, q_pad)
            intra = intra.transpose(-2, -1) + padded

            # Inter-frame temporal BiLSTM
            inter_in = intra.transpose(1, 2)
            inter = self.inter_norm(inter_in).reshape(bsz * q_pad, t_pad, ch).transpose(1, 2)
            inter = F.unfold(inter[..., None], (emb_ks, 1), stride=(emb_hs, 1)).transpose(1, 2)
            inter, _ = self.inter_rnn(inter)
            inter = self.inter_linear(inter.transpose(1, 2)).reshape(bsz, q_pad, ch, t_pad)
            inter = (inter.transpose(-2, -1) + inter_in).permute(0, 3, 2, 1)
            batch = inter[..., olp : olp + old_t, olp : olp + old_q]

            # Full-sequence multi-head cross-frame self-attention
            q = self.attn_norm_Q(self.attn_conv_Q(batch))
            k = self.attn_norm_K(self.attn_conv_K(batch))
            v = self.attn_norm_V(self.attn_conv_V(batch))
            q = q.reshape(-1, *q.shape[2:]).transpose(1, 2).flatten(start_dim=2)
            k = (
                k.reshape(-1, *k.shape[2:])
                .transpose(2, 3)
                .contiguous()
                .view(bsz * n_head, -1, old_t)
            )
            v = v.reshape(-1, *v.shape[2:]).transpose(1, 2)
            v_shape = v.shape
            v = v.flatten(start_dim=2)

            attn = F.softmax(torch.matmul(q, k) / (q.shape[-1] ** 0.5), dim=2)
            out_v = torch.matmul(attn, v).reshape(v_shape).transpose(1, 2)
            proj: torch.Tensor = self.attn_concat_proj(
                out_v.contiguous().view(bsz, n_head * out_v.shape[1], old_t, old_q)
            )
            res: torch.Tensor = proj + batch
            return res

    class _TFGridNetV3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2, emb_dim, (3, 3), padding=(1, 1)),
                nn.GroupNorm(1, emb_dim, eps=eps),
            )
            self.blocks = nn.ModuleList([_GridNetV3Block() for _ in range(n_layers)])
            self.deconv = nn.ConvTranspose2d(emb_dim, 2, (3, 3), padding=(1, 1))

        def forward(self, wav: torch.Tensor, sr: int = _INTERNAL_SAMPLE_RATE) -> torch.Tensor:
            n_samples = int(wav.shape[-1])
            peak = wav.abs().amax(dim=1, keepdim=True)
            pre_scale = torch.where(peak > 1e6, peak, torch.ones_like(peak))
            wav_scaled = wav / pre_scale
            std = torch.std(wav_scaled, dim=1, keepdim=True) + 1e-12
            mean = torch.mean(wav_scaled, dim=1, keepdim=True)
            normed = (wav_scaled - mean) / std

            nfft = int(round(sr * fft_len))
            nhop = int(round(sr * hop_len))
            nwin = int(round(sr * win_len))
            window = torch.hann_window(nwin, device=wav.device, dtype=wav.dtype)
            spec = torch.stft(
                normed,
                n_fft=nfft,
                hop_length=nhop,
                win_length=nwin,
                window=window,
                return_complex=True,
            )
            feat = self.conv(torch.view_as_real(spec).permute(0, 3, 2, 1))
            for block in self.blocks:
                feat = block(feat)
            out_ri = self.deconv(feat).permute(0, 3, 2, 1)
            out_c = torch.complex(out_ri[..., 0], out_ri[..., 1])
            out = torch.istft(
                out_c,
                n_fft=nfft,
                hop_length=nhop,
                win_length=nfft,
                window=window,
                length=n_samples,
            )
            return out * (std * pre_scale)

    return _TFGridNetV3()


def load_dereverb_model(
    device: str,
    checkpoint_path: Path | None = None,
) -> Any:
    """Verify SHA-256 lock and load ``TFGridNetV3`` strictly offline (``weights_only=True``)."""
    path = dereverb_checkpoint_path() if checkpoint_path is None else checkpoint_path
    ok, reason = verify_dereverb_weights(path)
    if not ok:
        raise ModelProvenanceError(reason)

    cache_key = (str(path.resolve()), device, str(DEREVERB_PARAMS["checkpoint_sha256"]))
    cached = _DEREVERB_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch

    raw_ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(raw_ckpt, dict) or "model" not in raw_ckpt or "cfg" not in raw_ckpt:
        raise ModelProvenanceError("Neural dereverb checkpoint missing expected model/cfg payload")

    model = _build_tfgridnet_v3(dict(raw_ckpt["cfg"]))
    try:
        model.load_state_dict(raw_ckpt["model"], strict=True)
    except Exception as exc:
        raise ModelProvenanceError(
            f"Neural dereverb checkpoint state_dict mismatch ({type(exc).__name__})"
        ) from exc

    model.eval()
    model.to(torch.device(device))
    _DEREVERB_MODEL_CACHE[cache_key] = model
    return model


def _run_dereverb_chunk(
    model: Any,
    device: str,
    chunk_16k: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Run TFGridNetV3 inference on a single 16 kHz mono chunk."""
    import torch

    orig_len = len(chunk_16k)
    if orig_len == 0:
        return np.zeros(0, dtype=np.float32)

    work = np.ascontiguousarray(chunk_16k, dtype=np.float32)
    if float(np.sqrt(np.mean(np.square(work, dtype=np.float64)))) < _SILENCE_RMS_THRESHOLD:
        return work.copy()

    # Minimum STFT window length at 16 kHz is 512 samples
    min_len = int(round(_INTERNAL_SAMPLE_RATE * float(DEREVERB_PARAMS["fft_len_s"])))
    if len(work) < min_len:
        work = np.pad(work, (0, min_len - len(work)), mode="constant")

    tensor = torch.from_numpy(work).unsqueeze(0).to(torch.device(device))
    with torch.inference_mode():
        out_t = model(tensor, sr=_INTERNAL_SAMPLE_RATE)
        out = out_t.squeeze(0).detach().cpu().float().numpy()

    return _match_length(np.asarray(out, dtype=np.float32), orig_len)


def _chunked_overlap_add(
    audio_16k: np.ndarray[Any, np.dtype[np.float32]],
    chunk_fn: DereverbChunkFn,
    chunk_samples: int,
    overlap_samples: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Process arbitrary-length 16 kHz audio in bounded-memory raised-cosine chunks."""
    n = len(audio_16k)
    if n <= chunk_samples:
        check_memory_budget("dereverb_neural")
        return np.ascontiguousarray(chunk_fn(audio_16k), dtype=np.float32)

    hop = max(1, chunk_samples - overlap_samples)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    covered = 0

    while pos < n:
        check_memory_budget("dereverb_neural")
        end = min(pos + chunk_samples, n)
        enhanced_chunk = _match_length(
            np.ascontiguousarray(chunk_fn(audio_16k[pos:end]), dtype=np.float32),
            end - pos,
        )
        xfade = max(0, min(covered - pos, end - pos)) if pos > 0 else 0
        if xfade > 0:
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


def _preserve_high_band_air(
    original: np.ndarray[Any, np.dtype[np.float32]],
    dereverbed: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    *,
    crossover_hz: float = 7200.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Preserve natural >8 kHz consonant air on >16 kHz inputs via LR4 crossover + HF envelope.

    Uses a complementary zero-phase 4th-order Linkwitz-Riley split (2nd-order
    Butterworth passed through ``sosfiltfilt``, whose squared magnitudes satisfy
    ``|H_lp(w)|^2 + |H_hp(w)|^2 == 1`` everywhere with zero phase shift) and
    tracks the 3-7 kHz dereverbed/original envelope ratio so dry consonant air
    is preserved while late high-frequency reverb tails stay suppressed.
    """
    n = len(original)
    if sample_rate <= _INTERNAL_SAMPLE_RATE or n < 128 or crossover_hz >= 0.45 * sample_rate:
        return np.ascontiguousarray(dereverbed, dtype=np.float32)

    orig_raw = original.astype(np.float64, copy=False)
    derev_matched = _match_length(dereverbed, n)
    derev_raw = derev_matched.astype(np.float64, copy=False)
    if not (bool(np.all(np.isfinite(orig_raw))) and bool(np.all(np.isfinite(derev_raw)))):
        return np.ascontiguousarray(dereverbed, dtype=np.float32)

    scale = max(
        float(np.max(np.abs(orig_raw))) if n > 0 else 0.0,
        float(np.max(np.abs(derev_raw))) if n > 0 else 0.0,
    )
    if scale <= 0.0 or not np.isfinite(scale):
        return np.ascontiguousarray(dereverbed, dtype=np.float32)

    orig_norm32 = np.ascontiguousarray(orig_raw / scale, dtype=np.float32)
    derev_norm32 = np.ascontiguousarray(derev_raw / scale, dtype=np.float32)
    derev_aligned_norm = estimate_gcc_phat_delay(
        orig_norm32,
        derev_norm32,
        sample_rate=sample_rate,
        max_delay_ms=50.0,
    ).aligned_candidate[:n]
    orig64 = orig_norm32.astype(np.float64, copy=False)
    derev64 = derev_aligned_norm.astype(np.float64, copy=False)

    sos_lp = scipy.signal.butter(2, crossover_hz, btype="lowpass", fs=sample_rate, output="sos")
    sos_hp = scipy.signal.butter(2, crossover_hz, btype="highpass", fs=sample_rate, output="sos")

    derev_low = np.asarray(scipy.signal.sosfiltfilt(sos_lp, derev64), dtype=np.float64)
    orig_low = np.asarray(scipy.signal.sosfiltfilt(sos_lp, orig64), dtype=np.float64)
    orig_high = np.asarray(scipy.signal.sosfiltfilt(sos_hp, orig64), dtype=np.float64)

    # Track gain-matched envelope ratio in the upper-mid reference band (2.5-6.5 kHz)
    # strictly inside the low-passed (<crossover_hz) band so >8 kHz consonant air
    # cannot leak into orig_ref and falsely attenuate itself.
    ref_lo = min(2500.0, crossover_hz * 0.4)
    ref_hi = min(6500.0, crossover_hz * 0.9)
    sos_ref = scipy.signal.butter(
        4, [ref_lo, ref_hi], btype="bandpass", fs=sample_rate, output="sos"
    )
    orig_ref = np.asarray(scipy.signal.sosfiltfilt(sos_ref, orig_low), dtype=np.float64)
    derev_ref = np.asarray(scipy.signal.sosfiltfilt(sos_ref, derev_low), dtype=np.float64)

    env_win = max(16, int(round(sample_rate * 0.020)))
    smooth_win = max(8, int(round(sample_rate * 0.010)))
    orig_env = np.sqrt(
        np.maximum(
            scipy.ndimage.uniform_filter1d(orig_ref * orig_ref, size=env_win, mode="nearest"), 0.0
        )
        + 1e-12
    )
    derev_env = np.sqrt(
        np.maximum(
            scipy.ndimage.uniform_filter1d(derev_ref * derev_ref, size=env_win, mode="nearest"),
            0.0,
        )
        + 1e-12
    )
    # Also include low-band envelope fallback when input has strong >8 kHz air without 2.5-6.5 kHz content
    orig_low_env = np.sqrt(
        np.maximum(
            scipy.ndimage.uniform_filter1d(orig_low * orig_low, size=env_win, mode="nearest"), 0.0
        )
        + 1e-12
    )
    derev_low_env = np.sqrt(
        np.maximum(
            scipy.ndimage.uniform_filter1d(derev_low * derev_low, size=env_win, mode="nearest"),
            0.0,
        )
        + 1e-12
    )
    ref_active = orig_env > np.maximum(1e-4, 0.02 * orig_low_env)
    ratio_ref = np.clip(derev_env / (orig_env + 1e-9), 0.0, 1.0)
    ratio_low = np.clip(derev_low_env / (orig_low_env + 1e-9), 0.0, 1.0)
    hf_gain = np.where(ref_active, ratio_ref, ratio_low)
    hf_gain = scipy.ndimage.uniform_filter1d(hf_gain, size=smooth_win, mode="nearest")
    hf_gain = np.clip(hf_gain, 0.0, 1.0)

    combined = (derev_low + orig_high * hf_gain) * scale
    if not bool(np.all(np.isfinite(combined))):
        return np.ascontiguousarray(dereverbed, dtype=np.float32)
    f32_max = float(np.finfo(np.float32).max) * 0.5
    combined = np.clip(combined, -f32_max, f32_max)
    return np.ascontiguousarray(combined, dtype=np.float32)


def _blend_strength(
    original: np.ndarray[Any, np.dtype[np.float32]],
    dereverbed: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    strength: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Delay-aligned wet/dry blend using GCC-PHAT alignment."""
    if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError(f"strength must be finite and within [0.0, 1.0], got {strength!r}")
    if strength <= 0.0:
        return original.copy()
    if strength >= 1.0:
        return dereverbed.copy()
    aligned = estimate_gcc_phat_delay(
        original, dereverbed, sample_rate=sample_rate, max_delay_ms=50.0
    ).aligned_candidate[: len(original)]
    return np.ascontiguousarray(
        (1.0 - float(strength)) * original + float(strength) * aligned,
        dtype=np.float32,
    )


class NeuralDereverberator:
    """Pure-PyTorch mono speech dereverberator (MIT TF-GridNetV3 + guarded Apache-2.0 BWE)."""

    def __init__(
        self,
        device: str = "auto",
        *,
        enable_bwe: bool = True,
        bwe_cutoff_hz: float = 7200.0,
        chunk_duration_s: float = 12.0,
        overlap_duration_s: float = 1.5,
        checkpoint_path: Path | None = None,
        bwe_restorer: NeuralBWERestorer | None = None,
        chunk_runner: DereverbChunkFn | None = None,
    ) -> None:
        if (
            chunk_duration_s <= 0.0
            or overlap_duration_s < 0.0
            or overlap_duration_s >= chunk_duration_s
        ):
            raise ValueError(
                f"Invalid chunking geometry: chunk_duration_s={chunk_duration_s}, "
                f"overlap_duration_s={overlap_duration_s}"
            )
        if bwe_cutoff_hz <= 1000.0 or bwe_cutoff_hz >= 8000.0:
            raise ValueError(f"bwe_cutoff_hz must lie in (1000, 8000), got {bwe_cutoff_hz}")

        self._device = resolve_device(device, preference=("mps", "cuda", "cpu")).resolved
        self._enable_bwe = enable_bwe
        self._bwe_cutoff_hz = float(bwe_cutoff_hz)
        self._chunk_duration_s = float(chunk_duration_s)
        self._overlap_duration_s = float(overlap_duration_s)
        self._checkpoint_path = checkpoint_path
        self._bwe_restorer = bwe_restorer
        self._chunk_runner = chunk_runner
        self._model: Any = None

    @property
    def device(self) -> str:
        """Resolved compute device (``mps``, ``cuda``, or ``cpu``)."""
        return self._device

    def _ensure_model(self) -> None:
        if self._chunk_runner is not None or self._model is not None:
            return
        self._model = load_dereverb_model(self._device, checkpoint_path=self._checkpoint_path)
        logger.info(f"Loaded NeuralDereverberator ({DEREVERB_MODEL_ID}) on device={self._device!r}")

    def _apply_optional_bwe(
        self,
        derev_48k: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], bool, float | None]:
        if not self._enable_bwe:
            return derev_48k, False, None

        restorer = self._bwe_restorer
        if restorer is None:
            bwe_ok, _ = probe_neural_bwe_runtime()
            if not bwe_ok:
                return derev_48k, False, None
            restorer = NeuralBWERestorer(device=self._device)
            self._bwe_restorer = restorer

        try:
            result = restorer.render(
                audio_48k=derev_48k,
                sample_rate=_BWE_SAMPLE_RATE,
                effective_cutoff_hz=self._bwe_cutoff_hz,
                strengths=[1.0],
            )
        except Exception as exc:
            logger.warning(
                f"Optional guarded BWE skipped ({type(exc).__name__}); using 16k->48k output"
            )
            return derev_48k, False, None

        if result.success:
            for candidate in result.candidates:
                if candidate.strength > 0.0:
                    audio = _match_length(
                        np.asarray(candidate.audio, dtype=np.float32), len(derev_48k)
                    )
                    if bool(np.all(np.isfinite(audio))):
                        return audio, True, float(candidate.protected_band_error)
        return derev_48k, False, None

    def warmup(self) -> None:
        """Load weights and run a short non-zero test tone so device/MPS kernels compile."""
        self._ensure_model()
        n_warm = _BWE_SAMPLE_RATE // 2
        t = np.arange(n_warm, dtype=np.float32) / float(_BWE_SAMPLE_RATE)
        dummy = np.ascontiguousarray(1e-3 * np.sin(2.0 * np.pi * 440.0 * t), dtype=np.float32)
        self.dereverb(dummy, _BWE_SAMPLE_RATE)

    def dereverb(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
        *,
        strength: float = 1.0,
    ) -> DereverbResult:
        """Dereverberate mono speech audio with bounded memory and fail-closed BWE."""
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValueError(f"strength must be finite and within [0.0, 1.0], got {strength!r}")

        check_memory_budget("dereverb_neural")
        t_start = time.perf_counter()
        orig32 = np.ascontiguousarray(waveform, dtype=np.float32)
        orig_len = len(orig32)
        if (
            orig_len == 0
            or float(np.sqrt(np.mean(np.square(orig32, dtype=np.float64)))) < _SILENCE_RMS_THRESHOLD
        ):
            return DereverbResult(
                waveform=orig32.copy(),
                sample_rate=sample_rate,
                device=self._device,
                bwe_applied=False,
                protected_band_err_db=None,
                model_runtime_ms=0.0,
                input_samples=orig_len,
                output_samples=orig_len,
            )

        if strength <= 0.0:
            return DereverbResult(
                waveform=orig32.copy(),
                sample_rate=sample_rate,
                device=self._device,
                bwe_applied=False,
                protected_band_err_db=None,
                model_runtime_ms=0.0,
                input_samples=orig_len,
                output_samples=orig_len,
            )

        self._ensure_model()
        audio_16k = resample_audio(orig32, sample_rate, _INTERNAL_SAMPLE_RATE)
        chunk_samples = int(round(self._chunk_duration_s * _INTERNAL_SAMPLE_RATE))
        overlap_samples = int(round(self._overlap_duration_s * _INTERNAL_SAMPLE_RATE))
        runner: DereverbChunkFn = (
            self._chunk_runner
            if self._chunk_runner is not None
            else (lambda chunk: _run_dereverb_chunk(self._model, self._device, chunk))
        )
        derev_16k = _chunked_overlap_add(
            audio_16k,
            runner,
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
        )

        # Fail closed if the model produced any non-finite samples
        if not bool(np.all(np.isfinite(derev_16k))):
            logger.warning("NeuralDereverberator produced non-finite samples; reverting to input")
            return DereverbResult(
                waveform=orig32.copy(),
                sample_rate=sample_rate,
                device=self._device,
                bwe_applied=False,
                protected_band_err_db=None,
                model_runtime_ms=(time.perf_counter() - t_start) * 1000.0,
                input_samples=orig_len,
                output_samples=orig_len,
            )

        bwe_applied = False
        pb_err_db: float | None = None
        if sample_rate >= 24000:
            derev_48k = resample_audio(derev_16k, _INTERNAL_SAMPLE_RATE, _BWE_SAMPLE_RATE)
            derev_48k, bwe_applied, pb_err_db = self._apply_optional_bwe(derev_48k)
            derev_out = resample_audio(
                derev_48k, _BWE_SAMPLE_RATE, sample_rate, target_samples=orig_len
            )
            if not bwe_applied:
                derev_out = _preserve_high_band_air(
                    orig32,
                    derev_out,
                    sample_rate,
                    crossover_hz=self._bwe_cutoff_hz,
                )
        else:
            derev_out = resample_audio(
                derev_16k, _INTERNAL_SAMPLE_RATE, sample_rate, target_samples=orig_len
            )

        blended = _blend_strength(orig32, _match_length(derev_out, orig_len), sample_rate, strength)
        peak = float(np.max(np.abs(blended))) if len(blended) > 0 else 0.0
        if peak > 0.99:
            blended = np.ascontiguousarray(
                (blended.astype(np.float64) * (0.99 / peak)).astype(np.float32),
                dtype=np.float32,
            )

        return DereverbResult(
            waveform=blended,
            sample_rate=sample_rate,
            device=self._device,
            bwe_applied=bwe_applied,
            protected_band_err_db=pb_err_db,
            model_runtime_ms=(time.perf_counter() - t_start) * 1000.0,
            input_samples=orig_len,
            output_samples=len(blended),
        )


class NeuralDereverbCore(Enhancer):
    """Registered 48 kHz mono dereverberation core (``studio-dereverb-48k-v1``)."""

    def __init__(
        self,
        core_id: str = STUDIO_DEREVERB_CORE_ID,
        sample_rate: int = 48000,
        phase_coherent: bool = False,
        device: str | None = None,
        *,
        enable_bwe: bool = False,
        checkpoint_path: Path | None = None,
        bwe_restorer: NeuralBWERestorer | None = None,
        chunk_runner: DereverbChunkFn | None = None,
    ) -> None:
        if phase_coherent:
            raise ValueError(
                "NeuralDereverbCore is not phase-coherent (STFT dereverberation modifies phase); "
                "configure phase_coherent = false"
            )
        output_sr = int(STUDIO_DEREVERB_PARAMS["output_sample_rate"])
        if sample_rate != output_sr:
            raise ValueError(
                f"NeuralDereverbCore runs at {output_sr} Hz output; "
                f"model_sample_rate must be {output_sr}, got {sample_rate}"
            )
        self._core_id = core_id
        self._sample_rate = output_sr
        self._device = (
            active_device()
            if device is None
            else resolve_device(
                device,
                core_id=core_id,
                preference=("mps", "cuda", "cpu"),
            ).resolved
        )
        self._metadata = EnhancerMetadata(
            core_id=core_id,
            version="1.0.0",
            algorithm=(
                "GAP-URGENet Predictor TF-GridNetV3 pure discriminative mono "
                "dereverberation (MIT/Apache-2.0, hash-locked weights)"
            ),
            sample_rate=self._sample_rate,
            phase_coherent=False,
            params_hash=studio_dereverb_params_hash(),
        )
        self._engine = NeuralDereverberator(
            device=self._device,
            enable_bwe=enable_bwe,
            bwe_cutoff_hz=float(STUDIO_DEREVERB_PARAMS["bwe_cutoff_hz"]),
            chunk_duration_s=float(STUDIO_DEREVERB_PARAMS["chunk_duration_s"]),
            overlap_duration_s=float(STUDIO_DEREVERB_PARAMS["overlap_duration_s"]),
            checkpoint_path=checkpoint_path,
            bwe_restorer=bwe_restorer,
            chunk_runner=chunk_runner,
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    @property
    def device(self) -> str:
        """The compute device this core's inference runs on."""
        return self._device

    def warmup(self) -> None:
        self._engine.warmup()

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        check_memory_budget("dereverb")
        res = self._engine.dereverb(waveform, sample_rate, strength=1.0)
        return EnhancementResult(
            waveform=res.waveform,
            sample_rate=res.sample_rate,
            model_runtime_ms=res.model_runtime_ms,
            input_samples=res.input_samples,
            output_samples=res.output_samples,
        )


__all__ = [
    "DEREVERB_ARCH_LICENSE",
    "DEREVERB_CHECKPOINT_FILE",
    "DEREVERB_CHECKPOINT_SHA256",
    "DEREVERB_MODEL_DIR_ENV",
    "DEREVERB_MODEL_ID",
    "DEREVERB_MODEL_LICENSE",
    "DEREVERB_PARAMS",
    "STUDIO_DEREVERB_CORE_ID",
    "STUDIO_DEREVERB_PARAMS",
    "DereverbChunkFn",
    "DereverbResult",
    "NeuralDereverbCore",
    "NeuralDereverberator",
    "dereverb_checkpoint_path",
    "dereverb_params_hash",
    "load_dereverb_model",
    "probe_dereverb_runtime",
    "studio_dereverb_params_hash",
    "verify_dereverb_weights",
]
