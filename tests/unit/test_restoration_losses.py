"""Unit tests for HawaRestore-KD differentiable losses and F0 conditioning."""

import pytest
import torch

from hawavoclean.restoration.hawarestore_kd import HawaRestoreKDNet
from research.restoration.train.losses import (
    CrossBandEnvelopeLoss,
    F0HarmonicConsistencyLoss,
    HawaRestoreLoss,
    MultiResComplexSTFTLoss,
    ProtectedBandInvarianceLoss,
)


def test_multires_stft_loss_per_example_cutoffs() -> None:
    B = 2
    sr = 48000
    pred = torch.randn(B, sr // 2)
    target = torch.randn(B, sr // 2)
    cutoffs = torch.tensor([4000.0, 8000.0])

    loss_fn = MultiResComplexSTFTLoss(fft_sizes=[512, 1024], hop_sizes=[128, 256])
    loss = loss_fn(pred, target, cutoff_hz=cutoffs, sr=sr)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_protected_band_invariance_loss() -> None:
    B = 2
    sr = 48000
    pred = torch.randn(B, sr // 2)
    target = torch.randn(B, sr // 2)
    cutoffs = torch.tensor([4000.0, 8000.0])

    loss_fn = ProtectedBandInvarianceLoss(n_fft=1024, hop=256)
    loss = loss_fn(pred, target, cutoff_hz=cutoffs, sr=sr)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    # Identical pred and target must have 0 protected band loss
    zero_loss = loss_fn(pred, pred, cutoff_hz=cutoffs, sr=sr)
    assert zero_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_f0_harmonic_consistency_loss_nonzero() -> None:
    B = 2
    sr = 48000
    t = torch.linspace(0, 0.5, sr // 2).view(1, -1).repeat(B, 1)
    f0 = torch.tensor([150.0, 220.0])
    # Target harmonic series
    target = torch.sin(2 * torch.pi * f0.view(B, 1) * t) + 0.5 * torch.sin(
        2 * torch.pi * 2 * f0.view(B, 1) * t
    )
    # Pred has perturbed harmonics
    pred = torch.sin(2 * torch.pi * (f0.view(B, 1) * 1.05) * t)

    cutoffs = torch.tensor([200.0, 200.0])
    loss_fn = F0HarmonicConsistencyLoss(n_fft=1024, hop=256)
    loss = loss_fn(pred, f0_hz=f0, cutoff_hz=cutoffs, target_audio=target, sr=sr)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    # For zero / unvoiced f0, loss is zero
    unvoiced_f0 = torch.tensor([0.0, 0.0])
    loss_unvoiced = loss_fn(pred, f0_hz=unvoiced_f0, cutoff_hz=cutoffs, target_audio=target, sr=sr)
    assert loss_unvoiced.item() == 0.0


def test_cross_band_envelope_loss() -> None:
    B = 2
    sr = 48000
    pred = torch.randn(B, sr // 2)
    target = torch.randn(B, sr // 2)

    loss_fn = CrossBandEnvelopeLoss()
    loss = loss_fn(pred, target)
    assert loss.dim() == 0


def test_hawarestore_loss_all_metric_keys_present() -> None:
    B = 2
    F_bins = 257
    T_frames = 64
    sr = 48000
    pred_v = torch.randn(B, 2, F_bins, T_frames)
    target_v = torch.randn(B, 2, F_bins, T_frames)
    pred_audio = torch.randn(B, sr // 4)
    target_audio = torch.randn(B, sr // 4)
    cutoffs = torch.tensor([4000.0, 6000.0])
    f0 = torch.tensor([140.0, 210.0])
    pred_spk = torch.randn(B, 192)
    target_spk = torch.randn(B, 192)

    loss_fn = HawaRestoreLoss(
        lambda_flow=1.0,
        lambda_stft=1.0,
        lambda_envelope=0.5,
        lambda_speaker=0.2,
        lambda_protected=1.0,
        lambda_harmonic=0.1,
    )
    total_loss, metrics = loss_fn(
        pred_v=pred_v,
        target_v=target_v,
        pred_audio=pred_audio,
        target_audio=target_audio,
        cutoff_hz=cutoffs,
        pred_speaker_emb=pred_spk,
        target_speaker_emb=target_spk,
        f0_hz=f0,
    )

    assert total_loss.dim() == 0
    assert total_loss.item() > 0.0

    required_keys = {
        "loss_flow",
        "loss_stft",
        "loss_env",
        "loss_harmonic",
        "loss_speaker",
        "loss_total",
    }
    assert required_keys.issubset(metrics.keys())
    for k in required_keys:
        assert isinstance(metrics[k], float)
        assert metrics[k] >= 0.0


def test_hawarestore_kd_net_f0_conditioning() -> None:
    net = HawaRestoreKDNet(n_fft=512, num_speakers=10, use_f0_cond=True)
    B = 2
    F_bins = 257
    T_frames = 32
    x_t = torch.randn(B, 2, F_bins, T_frames)
    t = torch.tensor([0.5, 0.7])
    cutoff = torch.tensor([4000.0, 6000.0])
    spk_idx = torch.tensor([1, 4])
    spk_proto = torch.randn(B, 192)
    f0 = torch.tensor([130.0, 200.0])
    vuv = torch.tensor([0.9, 0.8])

    out = net(
        x_t=x_t,
        t=t,
        cutoff_hz=cutoff,
        speaker_idx=spk_idx,
        speaker_proto=spk_proto,
        f0_hz=f0,
        vuv=vuv,
    )
    assert out.shape == x_t.shape
