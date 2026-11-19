"""Lean pure-PyTorch inference architecture for NISQA v2 speech quality assessment.

Attribution & Licensing:
- NISQA model architecture: Copyright (c) 2021 Gabriel Mittag, Quality and
  Usability Lab (MIT License, https://github.com/gabrielmittag/NISQA).
- Functional inference adaptation: Copyright The Lightning team
  (Apache License 2.0, torchmetrics.functional.audio.nisqa).

This module runs strictly offline using only ``torch``, ``numpy``, and ``scipy``
(zero ``librosa``, ``torchmetrics``, or ``requests`` dependencies). All weights
are verified against a pinned SHA-256 digest and loaded via
``torch.load(..., map_location="cpu", weights_only=True)``.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from torch import Tensor
from torch.nn.functional import adaptive_max_pool2d, relu, softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file
from hawavoclean.paths import app_data_root, models_dir

NISQA_WEIGHTS_ENV = "HAWAVOCLEAN_NISQA_WEIGHTS"
NISQA_WEIGHTS_FILENAME = "nisqa.tar"
NISQA_V2_WEIGHTS_SHA256 = "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _local_ai_models_dir() -> Path:
    return _REPO_ROOT / ".local-ai" / "models"


def nisqa_weights_path() -> Path:
    """Resolve the offline NISQA v2 weights file without network I/O.

    Resolution order:
    1. ``HAWAVOCLEAN_NISQA_WEIGHTS`` (exclusive override when set).
    2. ``HAWAVOCLEAN_MODEL_DIR / nisqa / nisqa.tar`` (exclusive when
       ``HAWAVOCLEAN_MODEL_DIR`` is set, preserving test isolation).
    3. Packaged ``models_dir() / nisqa / nisqa.tar``, per-user
       ``app_data_root() / models / nisqa / nisqa.tar``, repo-staged
       ``.local-ai/models/nisqa/nisqa.tar``, and local ``~/.torchmetrics`` /
       ``~/.cache/torchmetrics`` caches.
    """
    override = os.environ.get(NISQA_WEIGHTS_ENV)
    if override:
        return Path(override).expanduser().resolve()

    if os.environ.get("HAWAVOCLEAN_MODEL_DIR"):
        return models_dir() / "nisqa" / NISQA_WEIGHTS_FILENAME

    home = Path.home()
    candidates: list[Path] = [
        models_dir() / "nisqa" / NISQA_WEIGHTS_FILENAME,
        app_data_root() / "models" / "nisqa" / NISQA_WEIGHTS_FILENAME,
        _local_ai_models_dir() / "nisqa" / NISQA_WEIGHTS_FILENAME,
        home / ".torchmetrics" / "NISQA" / NISQA_WEIGHTS_FILENAME,
        home / ".cache" / "torchmetrics" / "NISQA" / NISQA_WEIGHTS_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def verify_nisqa_weights(weights_path: Path | None = None) -> tuple[bool, str]:
    """Verify NISQA v2 checkpoint presence and pinned SHA-256 digest.

    Returns ``(ok, path_free_message)``.
    """
    path = nisqa_weights_path() if weights_path is None else weights_path
    if not path.is_file():
        return False, f"NISQA v2 weights file missing ({NISQA_WEIGHTS_FILENAME})"
    if hash_file(path) != NISQA_V2_WEIGHTS_SHA256:
        return False, f"NISQA v2 weights digest mismatch ({NISQA_WEIGHTS_FILENAME})"
    return True, "NISQA v2 weights verified (MIT)"


def _hz_to_mel_slaney(freqs: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    arr = np.asarray(freqs, dtype=np.float64)
    mels = arr / f_sp
    log_mask = arr >= min_log_hz
    if np.any(log_mask):
        safe = np.where(log_mask, arr, min_log_hz)
        mels = np.where(log_mask, min_log_mel + np.log(safe / min_log_hz) / logstep, mels)
    return np.asarray(mels, dtype=np.float64)


def _mel_to_hz_slaney(mels: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    arr = np.asarray(mels, dtype=np.float64)
    freqs = f_sp * arr
    log_mask = arr >= min_log_mel
    if np.any(log_mask):
        freqs = np.where(log_mask, min_log_hz * np.exp(logstep * (arr - min_log_mel)), freqs)
    return np.asarray(freqs, dtype=np.float64)


@lru_cache(maxsize=16)
def _slaney_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Build exact Slaney-normalized triangular mel filterbank matching librosa."""
    fftfreqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    min_mel = float(_hz_to_mel_slaney(np.array([fmin]))[0])
    max_mel = float(_hz_to_mel_slaney(np.array([fmax]))[0])
    mel_f = _mel_to_hz_slaney(np.linspace(min_mel, max_mel, n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return np.ascontiguousarray(weights, dtype=np.float32)


def compute_nisqa_melspec(
    y: np.ndarray[Any, Any],
    sr: int,
    args: dict[str, Any],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute NISQA v2 log-mel spectrogram matching librosa preprocessing.

    Args:
        y: Waveform with shape ``(time,)`` or ``(batch_size, time)``.
        sr: Sampling rate in Hz.
        args: Checkpoint hyperparameter dictionary.

    Returns:
        Array of shape ``(batch_size, n_mels, n_frames)`` in dB.
    """
    y_2d = np.atleast_2d(np.asarray(y, dtype=np.float32))
    if y_2d.shape[-1] < 2:
        raise RuntimeError("Input signal is too short.")

    n_fft = int(args["ms_n_fft"])
    hop_length = int(sr * float(args["ms_hop_length"]))
    win_length = int(sr * float(args["ms_win_length"]))
    n_mels = int(args["ms_n_mels"])
    fmax = float(args["ms_fmax"])

    if hop_length <= 0 or win_length <= 0 or win_length > n_fft:
        raise ValueError(
            f"Invalid NISQA STFT parameters for sr={sr}: "
            f"n_fft={n_fft}, win_length={win_length}, hop_length={hop_length}"
        )

    hann = signal.get_window("hann", win_length, fftbins=True).astype(np.float32)
    fft_window = np.zeros(n_fft, dtype=np.float32)
    left = (n_fft - win_length) // 2
    fft_window[left : left + win_length] = hann

    pad = n_fft // 2
    y_padded = np.pad(y_2d, ((0, 0), (pad, pad)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(y_padded, n_fft, axis=-1)[:, ::hop_length, :]
    spec_mag = np.abs(np.fft.rfft(frames * fft_window, axis=-1)).astype(np.float32)

    mel_fb = _slaney_mel_filterbank(sr, n_fft, n_mels, 0.0, fmax)
    melspec = np.einsum("mf,btf->bmt", mel_fb, spec_mag, optimize=True)

    db_items: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for item in melspec:
        log_spec = 20.0 * np.log10(np.maximum(1e-4, item))
        log_spec = np.maximum(log_spec, float(np.max(log_spec)) - 80.0)
        db_items.append(np.ascontiguousarray(log_spec, dtype=np.float32))
    return np.stack(db_items, axis=0)


def segment_nisqa_specs(x: Tensor, args: dict[str, Any]) -> tuple[Tensor, Tensor]:
    """Slice mel spectrogram ``(B, n_mels, T)`` into fixed overlapping windows."""
    seg_length = int(args["ms_seg_length"])
    seg_hop = int(args["ms_seg_hop_length"])
    max_length = int(args["ms_max_segments"])

    n_wins = int(x.shape[2]) - (seg_length - 1)
    if n_wins < 1:
        raise RuntimeError("Input signal is too short.")

    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    segmented = x.transpose(2, 1)[:, idx3, :].transpose(3, 2)
    segmented = segmented[:, ::seg_hop]
    n_wins = math.ceil(n_wins / seg_hop)
    if max_length < n_wins:
        raise RuntimeError("Maximum number of mel spectrogram windows exceeded. Use shorter audio.")

    x_padded = torch.zeros(
        (segmented.shape[0], max_length, segmented.shape[2], segmented.shape[3]),
        dtype=segmented.dtype,
    )
    x_padded[:, :n_wins] = segmented
    return x_padded, torch.tensor(n_wins, dtype=torch.long)


class _AdaptCNN(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.pool_1: tuple[int, int] = (int(args["cnn_pool_1"][0]), int(args["cnn_pool_1"][1]))
        self.pool_2: tuple[int, int] = (int(args["cnn_pool_2"][0]), int(args["cnn_pool_2"][1]))
        self.pool_3: tuple[int, int] = (int(args["cnn_pool_3"][0]), int(args["cnn_pool_3"][1]))
        k0, k1 = int(args["cnn_kernel_size"][0]), int(args["cnn_kernel_size"][1])
        cnn_pad = (1, 0) if k0 == 1 else (1, 1)
        c1 = int(args["cnn_c_out_1"])
        c2 = int(args["cnn_c_out_2"])
        c3 = int(args["cnn_c_out_3"])

        self.conv1 = nn.Conv2d(1, c1, (k0, k1), padding=cnn_pad)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, (k0, k1), padding=cnn_pad)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, (k0, k1), padding=cnn_pad)
        self.bn3 = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c3, (k0, k1), padding=cnn_pad)
        self.bn4 = nn.BatchNorm2d(c3)
        self.conv5 = nn.Conv2d(c3, c3, (k0, k1), padding=cnn_pad)
        self.bn5 = nn.BatchNorm2d(c3)
        self.conv6 = nn.Conv2d(c3, c3, (k0, self.pool_3[1]), padding=(1, 0))
        self.bn6 = nn.BatchNorm2d(c3)

    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.bn1(self.conv1(x)))
        x = adaptive_max_pool2d(x, output_size=self.pool_1)
        x = relu(self.bn2(self.conv2(x)))
        x = adaptive_max_pool2d(x, output_size=self.pool_2)
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        x = adaptive_max_pool2d(x, output_size=self.pool_3)
        x = relu(self.bn5(self.conv5(x)))
        x = relu(self.bn6(self.conv6(x)))
        return x.view(-1, self.conv6.out_channels * self.pool_3[0])


class _Framewise(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.model = _AdaptCNN(args)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        x_packed = pack_padded_sequence(x, n_wins, batch_first=True, enforce_sorted=False)
        out = self.model(x_packed.data.unsqueeze(1))
        x_packed = x_packed._replace(data=out)
        padded, _ = pad_packed_sequence(
            x_packed, batch_first=True, padding_value=0.0, total_length=int(n_wins.max())
        )
        return padded


class _SelfAttentionLayer(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(args["td_sa_d_model"])
        nhead = int(args["td_sa_nhead"])
        hidden = int(args["td_sa_h"])
        dropout = float(args["td_sa_dropout"])
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, n_wins: Tensor) -> tuple[Tensor, Tensor]:
        mask = (
            torch.arange(src.shape[0], device=src.device)[None, :] < n_wins.to(src.device)[:, None]
        )
        src2: Tensor = self.self_attn(src, src, src, key_padding_mask=~mask)[0]
        src = self.norm1(src + src2)
        src2 = self.linear2(relu(self.linear1(src)))
        src = self.norm2(src + src2)
        return src, n_wins


class _SelfAttention(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(args["td_sa_d_model"])
        in_features = int(args["cnn_c_out_3"]) * int(args["cnn_pool_3"][0])
        num_layers = int(args["td_sa_num_layers"])
        self.norm1 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(in_features, d_model)
        self.layers = nn.ModuleList([_SelfAttentionLayer(args) for _ in range(num_layers)])

    def forward(self, src: Tensor, n_wins: Tensor) -> tuple[Tensor, Tensor]:
        output: Tensor = self.norm1(self.linear(src).transpose(1, 0))
        for mod in self.layers:
            output, n_wins = mod(output, n_wins)
        return output.transpose(1, 0), n_wins


class _TimeDependency(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.model = _SelfAttention(args)

    def forward(self, x: Tensor, n_wins: Tensor) -> tuple[Tensor, Tensor]:
        result: tuple[Tensor, Tensor] = self.model(x, n_wins)
        return result


class _PoolAttFF(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(args["td_sa_d_model"])
        att_h = int(args["pool_att_h"])
        self.linear1 = nn.Linear(d_model, att_h)
        self.linear2 = nn.Linear(att_h, 1)
        self.linear3 = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        att: Tensor = self.linear2(relu(self.linear1(x))).transpose(2, 1)
        mask = (
            torch.arange(att.shape[2], device=att.device)[None, :] < n_wins.to(att.device)[:, None]
        )
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = softmax(att, dim=2)
        pooled = torch.bmm(att, x).squeeze(1)
        out: Tensor = self.linear3(pooled)
        return out


class _Pooling(nn.Module):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.model = _PoolAttFF(args)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        out: Tensor = self.model(x, n_wins)
        return out


class NISQADimModel(nn.Module):
    """Five-axis NISQA v2 model outputting ``[mos, noi, dis, col, loud]``."""

    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.cnn = _Framewise(args)
        self.time_dependency = _TimeDependency(args)
        self.pool_layers = nn.ModuleList([_Pooling(args) for _ in range(5)])

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        return torch.cat([mod(x, n_wins) for mod in self.pool_layers], dim=1)


@lru_cache(maxsize=4)
def _load_nisqa_cached(
    resolved_path: str, expected_sha256: str
) -> tuple[NISQADimModel, dict[str, Any]]:
    _ = expected_sha256
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or "args" not in checkpoint
        or "model_state_dict" not in checkpoint
    ):
        raise ModelProvenanceError("NISQA v2 checkpoint is missing required fields")
    args = dict(checkpoint["args"])
    model = NISQADimModel(args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, args


def load_nisqa_v2_model(
    weights_path: Path | None = None,
) -> tuple[NISQADimModel, dict[str, Any]]:
    """Verify SHA-256 lock and load the cached pure-PyTorch NISQA v2 model."""
    path = nisqa_weights_path() if weights_path is None else weights_path
    ok, reason = verify_nisqa_weights(path)
    if not ok:
        raise ModelProvenanceError(reason)
    return _load_nisqa_cached(str(path.resolve()), NISQA_V2_WEIGHTS_SHA256)


def score_nisqa_v2(
    audio: np.ndarray[Any, Any],
    sample_rate: int,
    *,
    weights_path: Path | None = None,
) -> tuple[float, float, float, float, float]:
    """Run offline pure-PyTorch NISQA v2 inference on a mono waveform.

    Returns ``(mos, noisiness, discontinuity, coloration, loudness)``.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    model, args = load_nisqa_v2_model(weights_path=weights_path)
    mono = np.ascontiguousarray(audio, dtype=np.float32).reshape(1, -1)
    melspec = compute_nisqa_melspec(mono, int(sample_rate), args)
    segmented, n_wins = segment_nisqa_specs(torch.from_numpy(melspec), args)
    with torch.no_grad():
        out = model(segmented, n_wins.expand(segmented.shape[0]))
    vals = [float(v) for v in out[0].detach().cpu().tolist()]
    if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
        raise RuntimeError(f"unexpected NISQA output: {vals!r}")
    return (vals[0], vals[1], vals[2], vals[3], vals[4])


__all__ = [
    "NISQA_V2_WEIGHTS_SHA256",
    "NISQA_WEIGHTS_ENV",
    "NISQA_WEIGHTS_FILENAME",
    "NISQADimModel",
    "compute_nisqa_melspec",
    "load_nisqa_v2_model",
    "nisqa_weights_path",
    "score_nisqa_v2",
    "segment_nisqa_specs",
    "verify_nisqa_weights",
]
