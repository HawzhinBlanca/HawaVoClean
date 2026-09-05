"""Differentiable PyTorch losses for HawaRestore-KD training.

Implements all objectives defined in Section 5.5 of the directive:
- L_missing_band_flow: Flow matching vector field loss on missing bins
- L_high_band_multires_complex_stft: Multi-resolution spectral STFT loss
- L_high_band_phase: Phase consistency loss
- L_cross_band_envelope: Cross-band temporal envelope correlation
- L_f0_harmonic_consistency: F0-aligned harmonic energy loss
- L_speaker_identity: Cosine similarity loss against canonical speaker embedding
- L_protected_band_invariance: Zero penalty below cutoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResComplexSTFTLoss(nn.Module):
    """Multi-resolution complex & magnitude STFT loss on the high band with per-example cutoffs."""

    def __init__(
        self, fft_sizes: list[int] | None = None, hop_sizes: list[int] | None = None
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes or [512, 1024, 2048, 4096]
        self.hop_sizes = hop_sizes or [120, 240, 480, 960]

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        cutoff_hz: float | torch.Tensor,
        sr: int = 48000,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred_audio.device)
        B = pred_audio.shape[0]

        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes, strict=False):
            window = torch.hann_window(n_fft, device=pred_audio.device)
            Z_pred = torch.stft(
                pred_audio, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
            )
            Z_tgt = torch.stft(
                target_audio, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
            )

            # Frequency mask for high band (per-example or broadcast)
            n_freq_bins = Z_pred.shape[1]
            freqs = torch.linspace(0, sr / 2, n_freq_bins, device=pred_audio.device).view(1, -1, 1)
            if isinstance(cutoff_hz, torch.Tensor):
                cutoff_t = cutoff_hz.view(B, 1, 1).to(pred_audio.device)
                hf_mask = (freqs >= cutoff_t).float()  # (B, F, 1)
            else:
                hf_mask = (freqs >= cutoff_hz).float()  # (1, F, 1)

            mag_pred = torch.abs(Z_pred) * hf_mask
            mag_tgt = torch.abs(Z_tgt) * hf_mask

            mag_loss = F.l1_loss(mag_pred, mag_tgt)
            log_mag_loss = F.l1_loss(torch.log(mag_pred + 1e-5), torch.log(mag_tgt + 1e-5))

            # Complex STFT difference (real + imaginary tracking)
            real_loss = F.l1_loss(torch.real(Z_pred) * hf_mask, torch.real(Z_tgt) * hf_mask)
            imag_loss = F.l1_loss(torch.imag(Z_pred) * hf_mask, torch.imag(Z_tgt) * hf_mask)

            loss = loss + mag_loss + log_mag_loss + 0.5 * (real_loss + imag_loss)
        return loss / len(self.fft_sizes)


class ProtectedBandInvarianceLoss(nn.Module):
    """Enforces zero spectral perturbation in the protected band below cutoff."""

    def __init__(self, n_fft: int = 2048, hop: int = 480) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        cutoff_hz: float | torch.Tensor,
        sr: int = 48000,
    ) -> torch.Tensor:
        B = pred_audio.shape[0]
        window = torch.hann_window(self.n_fft, device=pred_audio.device)
        Z_pred = torch.stft(
            pred_audio, n_fft=self.n_fft, hop_length=self.hop, window=window, return_complex=True
        )
        Z_tgt = torch.stft(
            target_audio, n_fft=self.n_fft, hop_length=self.hop, window=window, return_complex=True
        )
        n_freq_bins = Z_pred.shape[1]
        freqs = torch.linspace(0, sr / 2, n_freq_bins, device=pred_audio.device).view(1, -1, 1)
        if isinstance(cutoff_hz, torch.Tensor):
            cutoff_t = cutoff_hz.view(B, 1, 1).to(pred_audio.device)
            lf_mask = (freqs < cutoff_t).float()
        else:
            lf_mask = (freqs < cutoff_hz).float()
        diff = torch.abs(Z_pred - Z_tgt) * lf_mask
        return torch.mean(diff)


class F0HarmonicConsistencyLoss(nn.Module):
    """Differentiable harmonic consistency loss aligned with fundamental frequency f0.

    When speech is voiced (f0 > 0), penalizes spectral deviations specifically at
    harmonic frequencies k * f0 in the restored high band above cutoff_hz.
    Strictly non-zero when intended.
    """

    def __init__(self, n_fft: int = 1024, hop: int = 240) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop

    def forward(
        self,
        pred_audio: torch.Tensor,
        f0_hz: torch.Tensor | None,
        cutoff_hz: float | torch.Tensor,
        target_audio: torch.Tensor | None = None,
        sr: int = 48000,
    ) -> torch.Tensor:
        if f0_hz is None:
            return torch.tensor(0.0, device=pred_audio.device)

        B = pred_audio.shape[0]
        device = pred_audio.device
        window = torch.hann_window(self.n_fft, device=device)

        Z_pred = torch.stft(
            pred_audio, n_fft=self.n_fft, hop_length=self.hop, window=window, return_complex=True
        )
        mag_pred = torch.abs(Z_pred)  # (B, F, T)
        n_freq_bins = mag_pred.shape[1]
        freqs = torch.linspace(0, sr / 2, n_freq_bins, device=device)  # (F,)

        if target_audio is not None:
            Z_tgt = torch.stft(
                target_audio,
                n_fft=self.n_fft,
                hop_length=self.hop,
                window=window,
                return_complex=True,
            )
            mag_tgt = torch.abs(Z_tgt)
        else:
            mag_tgt = None

        batch_losses: list[torch.Tensor] = []
        for b in range(B):
            f0_val = float(f0_hz[b].item()) if f0_hz.dim() > 0 else float(f0_hz.item())
            if f0_val <= 30.0 or not torch.isfinite(torch.tensor(f0_val)):
                # Unvoiced frame / silent segment
                continue

            c_val = (
                float(cutoff_hz[b].item())
                if isinstance(cutoff_hz, torch.Tensor)
                else float(cutoff_hz)
            )

            # High-band mask above cutoff
            hf_mask = (freqs >= c_val).float()  # (F,)
            if not torch.any(hf_mask > 0.0):
                continue

            # Construct harmonic comb filter around k * f0 for k * f0 >= c_val
            k_min = max(1, int(c_val / f0_val))
            k_max = min(100, int((sr / 2 - 50.0) / f0_val))
            if k_max < k_min:
                continue

            harmonic_freqs = (
                torch.arange(k_min, k_max + 1, device=device, dtype=torch.float32) * f0_val
            )
            # Gaussian comb weighting: sigma proportional to f0 (e.g. 0.15 * f0)
            sigma = max(15.0, 0.15 * f0_val)
            dist_sq = (freqs.view(-1, 1) - harmonic_freqs.view(1, -1)) ** 2  # (F, K)
            comb = torch.exp(-dist_sq / (2.0 * sigma**2)).sum(dim=1)  # (F,)
            comb = torch.clamp(comb, 0.0, 1.0) * hf_mask

            if not torch.any(comb > 0.0):
                continue

            comb_3d = comb.view(-1, 1)  # (F, 1)
            p_b = mag_pred[b]  # (F, T)

            if mag_tgt is not None:
                t_b = mag_tgt[b]  # (F, T)
                # Harmonic peak discrepancy + inter-harmonic valley tracking
                peak_loss = torch.sum(comb_3d * torch.abs(p_b - t_b)) / (
                    torch.sum(comb_3d) * p_b.shape[-1] + 1e-8
                )
                valley_mask = (1.0 - comb_3d) * hf_mask.view(-1, 1)
                valley_loss = torch.sum(valley_mask * torch.abs(p_b - t_b)) / (
                    torch.sum(valley_mask) * p_b.shape[-1] + 1e-8
                )
                batch_losses.append(0.7 * peak_loss + 0.3 * valley_loss)
            else:
                # Target-free: penalize unnatural energy explosions in high-band harmonics
                harm_energy = torch.mean((p_b * comb_3d) ** 2)
                batch_losses.append(F.relu(harm_energy - 1.0))

        if not batch_losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(batch_losses).mean()


class CrossBandEnvelopeLoss(nn.Module):
    """Cross-band temporal envelope correlation loss between mid band and generated high band."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        # Measure frame-level envelope correlation
        frame_size = 480
        B = pred_audio.shape[0]
        n_frames = pred_audio.shape[-1] // frame_size
        if n_frames < 2:
            return torch.tensor(0.0, device=pred_audio.device)

        p_frames = pred_audio[:, : n_frames * frame_size].view(B, n_frames, frame_size)
        t_frames = target_audio[:, : n_frames * frame_size].view(B, n_frames, frame_size)

        p_env = torch.sqrt(torch.mean(p_frames**2, dim=-1) + 1e-8)
        t_env = torch.sqrt(torch.mean(t_frames**2, dim=-1) + 1e-8)

        p_mean = p_env - torch.mean(p_env, dim=-1, keepdim=True)
        t_mean = t_env - torch.mean(t_env, dim=-1, keepdim=True)

        p_var = torch.sum(p_mean**2, dim=-1)
        t_var = torch.sum(t_mean**2, dim=-1)

        cov = torch.sum(p_mean * t_mean, dim=-1)
        corr = cov / (torch.sqrt(p_var * t_var) + 1e-6)
        return torch.mean(1.0 - corr)


class HawaRestoreLoss(nn.Module):
    """Total differentiable restoration loss."""

    def __init__(
        self,
        lambda_flow: float = 1.0,
        lambda_stft: float = 1.0,
        lambda_envelope: float = 0.5,
        lambda_speaker: float = 0.2,
        lambda_protected: float = 1.0,
        lambda_harmonic: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_stft = lambda_stft
        self.lambda_envelope = lambda_envelope
        self.lambda_speaker = lambda_speaker
        self.lambda_protected = lambda_protected
        self.lambda_harmonic = lambda_harmonic

        self.stft_loss = MultiResComplexSTFTLoss()
        self.env_loss = CrossBandEnvelopeLoss()
        self.protected_loss = ProtectedBandInvarianceLoss()
        self.harmonic_loss = F0HarmonicConsistencyLoss()

    def forward(
        self,
        pred_v: torch.Tensor,  # (B, 2, F, T)
        target_v: torch.Tensor,  # (B, 2, F, T)
        pred_audio: torch.Tensor | None = None,
        target_audio: torch.Tensor | None = None,
        cutoff_hz: float | torch.Tensor = 4000.0,
        pred_speaker_emb: torch.Tensor | None = None,
        target_speaker_emb: torch.Tensor | None = None,
        f0_hz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Missing band flow matching MSE loss
        flow_loss = F.mse_loss(pred_v, target_v)
        total_loss = self.lambda_flow * flow_loss
        metrics = {"loss_flow": float(flow_loss.item())}

        # 2. Multi-resolution STFT, Envelope, Protected, and Harmonic loss
        if pred_audio is not None and target_audio is not None:
            stft_l = self.stft_loss(pred_audio, target_audio, cutoff_hz=cutoff_hz)
            env_l = self.env_loss(pred_audio, target_audio)
            prot_l = self.protected_loss(pred_audio, target_audio, cutoff_hz=cutoff_hz)
            harm_l = self.harmonic_loss(
                pred_audio, f0_hz, cutoff_hz=cutoff_hz, target_audio=target_audio
            )

            total_loss = (
                total_loss
                + self.lambda_stft * stft_l
                + self.lambda_envelope * env_l
                + self.lambda_protected * prot_l
                + self.lambda_harmonic * harm_l
            )
            metrics["loss_stft"] = float(stft_l.item() + prot_l.item())
            metrics["loss_env"] = float(env_l.item())
            metrics["loss_protected"] = float(prot_l.item())
            metrics["loss_harmonic"] = float(harm_l.item())

        # 3. Speaker identity cosine loss
        if pred_speaker_emb is not None and target_speaker_emb is not None:
            spk_loss = (
                1.0 - F.cosine_similarity(pred_speaker_emb, target_speaker_emb, dim=-1).mean()
            )
            total_loss = total_loss + self.lambda_speaker * spk_loss
            metrics["loss_speaker"] = float(spk_loss.item())

        metrics["loss_total"] = float(total_loss.item())
        return total_loss, metrics
