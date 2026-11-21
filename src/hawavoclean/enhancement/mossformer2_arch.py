# Copyright (c) 2024 Alibaba Inc (Alibaba Speech Lab, ClearerVoice-Studio)
# Copyright (c) 2026 HawaVoClean Contributors (inference-only pure-PyTorch extraction)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lean pure-PyTorch inference architecture for ``MossFormer2_SE_48K``.

Derived strictly from Alibaba Speech Lab's Apache-2.0 ClearerVoice-Studio
``MossFormer2_SE_48K`` model (with dual-attention FLASH blocks, Gated FSMN
recurrence, and rotary positional embeddings implemented directly in PyTorch).
Zero training code and zero ``clearvoice`` / ``einops`` /
``rotary_embedding_torch`` / ``librosa`` / ``opencv`` dependencies.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MAX_WAV_VALUE: float = 32768.0


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate interleaved even/odd channel pairs for RoPE."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = pairs.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Partial rotary positional embedding (GPT-J / FLASH-pytorch layout)."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)

    def rotate_queries_or_keys(self, t: Tensor) -> Tensor:
        """Apply partial rotary embedding along the sequence axis (``dim=-2``)."""
        seq_len = t.shape[-2]
        seq = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
        freqs = torch.einsum("i, j -> i j", seq, self.freqs).repeat_interleave(2, dim=-1)
        rot_dim = freqs.shape[-1]
        t_rot = t[..., :rot_dim]
        t_pass = t[..., rot_dim:]
        t_transformed = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
        return torch.cat((t_transformed, t_pass), dim=-1).to(t.dtype)


class ScaleNorm(nn.Module):
    """Scaled L2 normalization used in MossFormer2 FLASH projections."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale: float = dim**-0.5
        self.eps: float = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True) * self.scale
        res: Tensor = x / norm.clamp(min=self.eps) * self.g
        return res


class ScaledSinuEmbedding(nn.Module):
    """Learnable-scale sinusoidal positional embedding."""

    inv_freq: Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        sinu = torch.einsum("i, j -> i j", t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)
        return emb * self.scale


class OffsetScale(nn.Module):
    """Multi-head learned affine offset and scale."""

    def __init__(self, dim: int, heads: int = 1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        out = torch.einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class Transpose(nn.Module):
    """Transpose layer preserving Sequential index layout for depthwise conv."""

    def __init__(self, shape: tuple[int, int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    """1-D depthwise convolution matching ClearerVoice checkpoint naming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv(x)
        return out


class ConvModule(nn.Module):
    """Residual 1-D depthwise convolution block."""

    def __init__(self, in_channels: int, kernel_size: int = 17) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.sequential(x)
        return x + out.transpose(1, 2)


class FFConvM(nn.Module):
    """Feed-forward convolutional projection (Norm -> Linear -> SiLU -> ConvModule)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        norm_klass: Callable[[int], nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.mdl(x)
        return out


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization over ``[B, C, T]`` tensors."""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class UniDeepFsmn(nn.Module):
    """Single-deep Feedforward Sequential Memory Network (FSMN)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lorder: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.lorder = lorder
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim,
            (lorder * 2 - 1, 1),
            (1, 1),
            groups=output_dim,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        f1 = F.relu(self.linear(x))
        p1 = self.project(f1)
        x_per = p1.unsqueeze(1).permute(0, 3, 2, 1)
        y = F.pad(x_per, (0, 0, self.lorder - 1, self.lorder - 1))
        out = x_per + self.conv1(y)
        res: Tensor = x + out.permute(0, 3, 2, 1).squeeze(1)
        return res


class GatedFSMN(nn.Module):
    """Gated FSMN combining dual FFConvM projections with UniDeepFsmn."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lorder: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.to_u = FFConvM(dim_in=in_channels, dim_out=hidden_size, norm_klass=nn.LayerNorm)
        self.to_v = FFConvM(dim_in=in_channels, dim_out=hidden_size, norm_klass=nn.LayerNorm)
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder=lorder, hidden_size=hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x_u: Tensor = self.fsmn(self.to_u(x))
        x_v: Tensor = self.to_v(x)
        return x_v * x_u + x


class GatedFSMNBlock(nn.Module):
    """Bottleneck residual block wrapping a GatedFSMN layer."""

    def __init__(self, dim: int, inner_channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = GatedFSMN(
            inner_channels, inner_channels, lorder=20, hidden_size=inner_channels
        )
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2: Tensor = self.conv2(norm2)
        return conv2.transpose(2, 1) + x


class FLASHShareAFFConvM(nn.Module):
    """Non-causal shared dual-attention (local quadratic + global linear) block."""

    def __init__(
        self,
        *,
        dim: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
        rotary_pos_emb: RotaryEmbedding,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.rotary_pos_emb = rotary_pos_emb
        self.to_hidden = FFConvM(dim_in=dim, dim_out=hidden_dim, norm_klass=ScaleNorm)
        self.to_qk = FFConvM(dim_in=dim, dim_out=query_key_dim, norm_klass=ScaleNorm)
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = FFConvM(dim_in=dim * 2, dim_out=dim, norm_klass=ScaleNorm)

    def forward(self, x: Tensor) -> Tensor:
        x_shift, x_pass = x.chunk(2, dim=-1)
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
        normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self._attention(x, quad_q, lin_q, quad_k, lin_k, v, u)
        out = (att_u * v) * torch.sigmoid(att_v * u)
        proj: Tensor = self.to_out(out)
        return x + proj

    def _attention(
        self,
        x: Tensor,
        quad_q: Tensor,
        lin_q: Tensor,
        quad_k: Tensor,
        lin_k: Tensor,
        v: Tensor,
        u: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, n = x.shape[0], x.shape[-2]
        g = self.group_size

        quad_q = self.rotary_pos_emb.rotate_queries_or_keys(quad_q)
        lin_q = self.rotary_pos_emb.rotate_queries_or_keys(lin_q)
        quad_k = self.rotary_pos_emb.rotate_queries_or_keys(quad_k)
        lin_k = self.rotary_pos_emb.rotate_queries_or_keys(lin_k)

        rem = n % g
        padding = (g - rem) if rem else 0
        mask: Tensor | None = None
        if padding > 0:
            pad_spec = (0, 0, 0, padding)
            quad_q = F.pad(quad_q, pad_spec, value=0.0)
            quad_k = F.pad(quad_k, pad_spec, value=0.0)
            lin_q = F.pad(lin_q, pad_spec, value=0.0)
            lin_k = F.pad(lin_k, pad_spec, value=0.0)
            v = F.pad(v, pad_spec, value=0.0)
            u = F.pad(u, pad_spec, value=0.0)
            mask = torch.ones((b, n), device=x.device, dtype=torch.bool)
            mask = F.pad(mask, (0, padding), value=False)

        n_groups = (n + padding) // g
        quad_q = quad_q.reshape(b, n_groups, g, quad_q.shape[-1])
        quad_k = quad_k.reshape(b, n_groups, g, quad_k.shape[-1])
        lin_q = lin_q.reshape(b, n_groups, g, lin_q.shape[-1])
        lin_k = lin_k.reshape(b, n_groups, g, lin_k.shape[-1])
        v = v.reshape(b, n_groups, g, v.shape[-1])
        u = u.reshape(b, n_groups, g, u.shape[-1])

        sim = torch.einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g
        attn = F.relu(sim) ** 2
        if mask is not None:
            attn = attn.masked_fill(~mask.reshape(b, n_groups, 1, g), 0.0)

        quad_out_v = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        quad_out_u = torch.einsum("... i j, ... j d -> ... i d", attn, u)

        lin_kv = torch.einsum("b g n d, b g n e -> b d e", lin_k, v) / n
        lin_out_v = torch.einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)
        lin_ku = torch.einsum("b g n d, b g n e -> b d e", lin_k, u) / n
        lin_out_u = torch.einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

        out_v = (quad_out_v + lin_out_v).reshape(b, n_groups * g, v.shape[-1])[:, :n]
        out_u = (quad_out_u + lin_out_u).reshape(b, n_groups * g, u.shape[-1])[:, :n]
        return out_v, out_u


class MossformerBlockGFSMN(nn.Module):
    """Stack of interleaved FLASH attention and Gated FSMN blocks."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
    ) -> None:
        super().__init__()
        self.fsmn = nn.ModuleList([GatedFSMNBlock(dim) for _ in range(depth)])
        self.layers = nn.ModuleList(
            [
                FLASHShareAFFConvM(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    rotary_pos_emb=RotaryEmbedding(dim=min(32, query_key_dim)),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for flash, fsmn in zip(self.layers, self.fsmn, strict=True):
            x = flash(x)
            x = fsmn(x)
        return x


class MossFormerM(nn.Module):
    """MossFormer2 sequence encoder followed by final LayerNorm."""

    def __init__(self, num_blocks: int, d_model: int) -> None:
        super().__init__()
        self.mossformerM = MossformerBlockGFSMN(dim=d_model, depth=num_blocks)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src: Tensor) -> Tensor:
        out: Tensor = self.norm(self.mossformerM(src))
        return out


class ComputationBlock(nn.Module):
    """Residual computation block operating over ``[B, C, T]`` representations."""

    def __init__(self, num_blocks: int, out_channels: int) -> None:
        super().__init__()
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)

    def forward(self, x: Tensor) -> Tensor:
        intra = x.permute(0, 2, 1).contiguous()
        intra = self.intra_mdl(intra)
        intra = intra.permute(0, 2, 1).contiguous()
        intra = self.intra_norm(intra)
        res: Tensor = intra + x
        return res


class MossFormerMaskNet(nn.Module):
    """Mask prediction network mapping 180-dim fbank features to 961 STFT bins."""

    def __init__(
        self,
        in_channels: int = 180,
        out_channels: int = 512,
        out_channels_final: int = 961,
        num_blocks: int = 24,
        num_spks: int = 2,
    ) -> None:
        super().__init__()
        self.num_spks = num_spks
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pos_enc = ScaledSinuEmbedding(out_channels)
        self.mdl = ComputationBlock(num_blocks=num_blocks, out_channels=out_channels)
        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(out_channels, out_channels_final, kernel_size=1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv1d_encoder(x)
        emb = self.pos_enc(x.transpose(1, -1)).transpose(0, -1)
        x = x + emb
        x = self.mdl(x)
        x = self.prelu(x)
        x = self.conv1d_out(x)
        batch_size, _, seq_len = x.shape
        x = x.view(batch_size * self.num_spks, -1, seq_len)
        x = self.output(x) * self.output_gate(x)
        x = self.conv1_decoder(x)
        _, out_bins, out_frames = x.shape
        x = x.view(batch_size, self.num_spks, out_bins, out_frames)
        x = self.activation(x)
        return x[:, 0].transpose(1, 2)


class MossFormer2SE48K(nn.Module):
    """Top-level pure-PyTorch ``MossFormer2_SE_48K`` inference model.

    Matches the exact parameter tree and state-dict keys of
    ``alibabasglab/MossFormer2_SE_48K/last_best_checkpoint.pt`` (929 tensors).
    """

    def __init__(
        self,
        *,
        in_channels: int = 180,
        out_channels: int = 512,
        out_channels_final: int = 961,
        num_blocks: int = 24,
        num_spks: int = 2,
    ) -> None:
        super().__init__()
        self.mossformer = MossFormerMaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_final=out_channels_final,
            num_blocks=num_blocks,
            num_spks=num_spks,
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """Predict real-valued STFT magnitude mask from ``[B, T, 180]`` features."""
        mask: Tensor = self.mossformer(inputs.transpose(1, 2))
        return [mask]


def extract_mossformer2_features(
    audio_scaled: Tensor,
    *,
    sample_rate: int = 48000,
    win_len: int = 1920,
    win_inc: int = 384,
    num_mels: int = 60,
    win_type: str = "hamming",
    dither: float = 1.0,
) -> Tensor:
    """Extract 180-channel Kaldi fbank + delta + delta-delta features ``[1, T, 180]``."""
    ta_kaldi = importlib.import_module("torchaudio.compliance.kaldi")
    ta_functional = importlib.import_module("torchaudio.functional")

    frame_length_ms = win_len / sample_rate * 1000.0
    frame_shift_ms = win_inc / sample_rate * 1000.0
    fbanks: Tensor = ta_kaldi.fbank(
        audio_scaled.unsqueeze(0),
        dither=dither,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        num_mel_bins=num_mels,
        sample_frequency=sample_rate,
        window_type=win_type,
    )
    fbank_tr = fbanks.transpose(0, 1)
    fbank_delta: Tensor = ta_functional.compute_deltas(fbank_tr)
    fbank_delta_delta: Tensor = ta_functional.compute_deltas(fbank_delta)
    features = torch.cat(
        [fbanks, fbank_delta.transpose(0, 1), fbank_delta_delta.transpose(0, 1)],
        dim=1,
    )
    return features.unsqueeze(0)


def apply_mossformer2_mask(
    audio_scaled: Tensor,
    pred_mask: Tensor,
    *,
    fft_len: int = 1920,
    win_len: int = 1920,
    win_inc: int = 384,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply predicted STFT mask ``[1, T, F]`` and reconstruct float32 waveform in ``[-1, 1]``."""
    window = torch.hamming_window(win_len, periodic=False)
    spectrum = torch.stft(
        audio_scaled,
        n_fft=fft_len,
        hop_length=win_inc,
        win_length=win_len,
        window=window,
        center=False,
        return_complex=True,
    )
    mask_2d = pred_mask.detach().cpu()[0].transpose(0, 1)
    n_frames = min(int(spectrum.shape[1]), int(mask_2d.shape[1]))
    masked_complex = spectrum[:, :n_frames] * mask_2d[:, :n_frames]
    reconstructed = torch.istft(
        masked_complex,
        n_fft=fft_len,
        hop_length=win_inc,
        win_length=win_len,
        window=window,
        center=False,
        length=int(audio_scaled.shape[-1]),
        return_complex=False,
    )
    out: np.ndarray[Any, np.dtype[np.float32]] = np.asarray(
        reconstructed.detach().cpu().float().numpy() / MAX_WAV_VALUE,
        dtype=np.float32,
    )
    return np.ascontiguousarray(out, dtype=np.float32)


__all__ = [
    "MAX_WAV_VALUE",
    "MossFormer2SE48K",
    "apply_mossformer2_mask",
    "extract_mossformer2_features",
]
