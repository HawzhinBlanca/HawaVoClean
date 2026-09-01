"""HawaRestore-KD: Protected-Band Kurdish Bandwidth-Restoration Model.

Architecture:
- Continuous variable cutoff conditioning (2 kHz - 22 kHz).
- Frequency-aware complex-STFT representation.
- Learned Speaker-ID embedding & Cam++ prototype conditioning via FiLM.
- F0 and Voiced/Unvoiced trajectory modulation.
- Vocoder-free flow matching ODE solver predicting only missing high band.
- Strict protected-band invariance by mathematical construction.
"""

import math
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file
from hawavoclean.paths import restoration_checkpoint_path
from hawavoclean.restoration.base import RestorationCandidate, Restorer
from hawavoclean.restoration.protected_band import (
    compute_transition_mask,
    merge_protected_spectrum,
)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal frequency/time embedding."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed 1D scalar tensor of shape (B,) or (B, 1) into (B, dim)."""
        x = x.view(-1, 1)
        freqs = torch.exp(
            torch.arange(self.half_dim, device=x.device, dtype=torch.float32)
            * -(math.log(10000.0) / max(1, self.half_dim - 1))
        ).unsqueeze(0)
        args = x * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FrequencyAwareBlock(nn.Module):
    """Frequency-axis convolutional block with residual connection and LayerNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1))
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1))
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return cast(torch.Tensor, self.act(h + res))


class SpeakerFiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioned on speaker embedding & cutoff."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) -> (B, 2*channels, 1, 1)
        gamma_beta = self.proj(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1.0 + gamma) + beta


class HawaRestoreKDNet(nn.Module):
    """Trainable Neural Vector Field Network for Kurdish missing-band flow matching.

    Predicts the velocity field v_t(x_t) in the complex STFT domain for missing high frequencies.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Real + Imag
        out_channels: int = 2,
        base_channels: int = 64,
        num_speakers: int = 10,
        speaker_embed_dim: int = 64,
        prototype_dim: int = 192,
        cond_dim: int = 256,
        n_fft: int = 1024,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.num_freq_bins = n_fft // 2 + 1

        # Embeddings
        self.time_embed = SinusoidalEmbedding(64)
        self.cutoff_embed = SinusoidalEmbedding(64)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embed_dim)
        self.proto_proj = nn.Linear(prototype_dim, 64)

        # Total condition projector
        # time(64) + cutoff(64) + spk_id(64) + proto(64) = 256
        self.cond_mlp = nn.Sequential(
            nn.Linear(64 + 64 + speaker_embed_dim + 64, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # UNet / ConvNeXt backbone for complex spectrogram processing
        self.in_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.block1 = FrequencyAwareBlock(base_channels)
        self.film1 = SpeakerFiLM(cond_dim, base_channels)

        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=3, stride=(2, 1), padding=1
        )
        self.block2 = FrequencyAwareBlock(base_channels * 2)
        self.film2 = SpeakerFiLM(cond_dim, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)
        )
        self.block3 = FrequencyAwareBlock(base_channels)
        self.film3 = SpeakerFiLM(cond_dim, base_channels)

        self.out_proj = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,  # (B, 2, F, T)
        t: torch.Tensor,  # (B,)
        cutoff_hz: torch.Tensor,  # (B,)
        speaker_idx: torch.Tensor | None = None,  # (B,)
        speaker_prototype: torch.Tensor | None = None,  # (B, 192)
        x_obs: torch.Tensor | None = None,  # (B, 2, F, T)
        _f0_hz: torch.Tensor | None = None,  # (B,)
    ) -> torch.Tensor:
        """Predict velocity field v_t(x_t)."""
        B = x_t.shape[0]
        device = x_t.device

        # 1. Compute condition vector
        t_emb = self.time_embed(t)
        c_emb = self.cutoff_embed(cutoff_hz / 1000.0)  # Normalized to kHz

        if speaker_idx is not None:
            spk_emb = self.speaker_embedding(speaker_idx)
        else:
            spk_emb = torch.zeros(B, self.speaker_embedding.embedding_dim, device=device)

        if speaker_prototype is not None:
            proto_emb = self.proto_proj(speaker_prototype)
        else:
            proto_emb = torch.zeros(B, 64, device=device)

        cond = self.cond_mlp(torch.cat([t_emb, c_emb, spk_emb, proto_emb], dim=-1))

        # 2. Forward through vector field network with low-band observed conditioning
        h = self.in_proj(x_t)
        if x_obs is not None:
            # Low-band observed guidance injection
            h = h + 0.5 * self.in_proj(x_obs)
        h = self.film1(self.block1(h), cond)

        h_down = self.down1(h)
        h_down = self.film2(self.block2(h_down), cond)

        h_up = self.up1(h_down)
        if h_up.shape != h.shape:
            h_up = F.interpolate(
                h_up, size=(h.shape[2], h.shape[3]), mode="bilinear", align_corners=False
            )

        h_out = self.film3(self.block3(h_up + h), cond)
        v_t = self.out_proj(h_out)
        return cast(torch.Tensor, v_t)


class HawaRestoreKD(Restorer):
    """Production HawaRestore-KD Subsystem.

    Combines the trainable PyTorch vector-field model with continuous cutoff detection,
    deterministic ODE flow simulation, and strict protected-band copying.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
        transition_hz: float = 500.0,
        device: str | None = None,
        checkpoint_path: Path | str | None = None,
        chunk_seconds: float = 1.0,
        chunk_overlap_seconds: float = 0.25,
        solver: str = "midpoint",
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.transition_hz = transition_hz
        self.chunk_seconds = chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self.solver = solver

        # CPU by default, deliberately. torch.manual_seed does not give the same
        # stream on CUDA/MPS as on CPU, and the ODE solver starts from randn, so
        # an auto-selected accelerator would make the restored master differ
        # between a dev machine and CI while the report claims a fixed seed
        # policy. An explicit device is honoured for callers who accept that.
        self.device = device or "cpu"

        ckpt_path = (
            Path(checkpoint_path) if checkpoint_path is not None else restoration_checkpoint_path()
        )
        self.checkpoint_path = ckpt_path
        if not ckpt_path.is_file():
            raise ModelProvenanceError(
                f"HawaRestore-KD checkpoint not found: {ckpt_path}. Restore mode refuses to run "
                "on untrained weights; set HAWAVOCLEAN_RESTORATION_CHECKPOINT or install the model."
            )
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                raise ModelProvenanceError(
                    f"HawaRestore-KD checkpoint {ckpt_path} has no 'model_state_dict' entry."
                )

            # Dynamic config from checkpoint (real-data trained) or legacy defaults.
            net_config = ckpt.get("config", {})
            num_speakers = int(net_config.get("num_speakers", 10))
            ckpt_n_fft = int(net_config.get("n_fft", n_fft))

            # Speaker table: from checkpoint metadata or legacy character_XX list.
            train_speakers = ckpt.get("train_speakers", [])
            val_speakers = ckpt.get("val_speakers", [])
            if train_speakers or val_speakers:
                self.speaker_ids = sorted(set(train_speakers + val_speakers))
            else:
                self.speaker_ids = [f"character_{i:02d}" for i in range(1, num_speakers + 1)]

            self.net = (
                HawaRestoreKDNet(
                    n_fft=ckpt_n_fft,
                    num_speakers=num_speakers,
                )
                .to(self.device)
                .eval()
            )
            self.net.load_state_dict(ckpt["model_state_dict"])
        except ModelProvenanceError:
            raise
        except Exception as e:
            # Swallowing this would leave the net randomly initialised while the
            # report still attests the checkpoint's hash.
            raise ModelProvenanceError(
                f"Failed to load HawaRestore-KD checkpoint {ckpt_path}: {e}"
            ) from e

        #: SHA-256 of the checkpoint that was actually loaded into ``self.net``.
        #: Reports must quote this rather than re-hashing a path they assume exists.
        self.weights_sha256 = hash_file(ckpt_path)

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute complex STFT."""
        _, _, Zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            boundary="zeros",
            padded=True,
        )
        return np.asarray(Zxx, dtype=np.complex64)

    def _istft(self, Zxx: np.ndarray) -> np.ndarray:
        """Compute inverse complex STFT."""
        _, x = signal.istft(
            Zxx,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            boundary="zeros",
        )
        return np.asarray(x, dtype=np.float32)

    def _solve_flow_ode(
        self,
        Z_obs_real: torch.Tensor,
        cutoff_hz: float,
        speaker_idx: torch.Tensor | None,
        speaker_proto: torch.Tensor | None,
        generator: torch.Generator,
        steps: int = 4,
    ) -> torch.Tensor:
        """Midpoint or Heun flow-matching ODE solver with low-band guidance."""
        dt = 1.0 / steps
        # An explicit generator, not torch.manual_seed: seeding the global RNG
        # from inside inference would reach every other torch consumer in the
        # process, and randn_like cannot take a generator.
        x_t = torch.randn(
            Z_obs_real.shape,
            generator=generator,
            device=Z_obs_real.device,
            dtype=Z_obs_real.dtype,
        )
        cutoff_t = torch.tensor([cutoff_hz], device=self.device, dtype=torch.float32)

        for step in range(steps):
            t_val = step * dt
            t_curr = torch.tensor([t_val], device=self.device, dtype=torch.float32)

            with torch.no_grad():
                if self.solver == "heun":
                    # Heun 2nd-order predictor-corrector
                    t_next = torch.tensor(
                        [min(1.0, t_val + dt)], device=self.device, dtype=torch.float32
                    )
                    v_0 = self.net(
                        x_t, t_curr, cutoff_t, speaker_idx, speaker_proto, x_obs=Z_obs_real
                    )
                    x_pred = x_t + v_0 * dt
                    v_1 = self.net(
                        x_pred, t_next, cutoff_t, speaker_idx, speaker_proto, x_obs=Z_obs_real
                    )
                    x_t = x_t + 0.5 * (v_0 + v_1) * dt
                else:
                    # Midpoint 2nd-order solver
                    t_mid = torch.tensor(
                        [t_val + dt / 2.0], device=self.device, dtype=torch.float32
                    )
                    v_curr = self.net(
                        x_t, t_curr, cutoff_t, speaker_idx, speaker_proto, x_obs=Z_obs_real
                    )
                    x_mid = x_t + v_curr * (dt / 2.0)
                    v_mid = self.net(
                        x_mid, t_mid, cutoff_t, speaker_idx, speaker_proto, x_obs=Z_obs_real
                    )
                    x_t = x_t + v_mid * dt

        return x_t

    def _block_len(self) -> int:
        """Samples per processing block."""
        return max(self.win_length * 4, int(self.chunk_seconds * self.sample_rate))

    def _block_positions(self, n_samples: int) -> list[int]:
        """Start offsets of the overlapping blocks covering ``n_samples``.

        The whole-file spectrogram cannot go through the network: at 100 frames
        per second and 1025 bins, a single 64-channel activation costs ~26 MB per
        second of audio, and the solver evaluates the net eight times. Measured
        end to end that was ~1 GB of RAM per second of input, so a few minutes of
        speech exhausted the machine. Blocks keep the cost flat.
        """
        block_len = self._block_len()
        overlap = min(
            block_len // 2, max(self.win_length, int(self.chunk_overlap_seconds * self.sample_rate))
        )
        hop = max(1, block_len - overlap)

        positions: list[int] = []
        pos = 0
        while True:
            positions.append(pos)
            if pos + block_len >= n_samples:
                break
            pos += hop

        # A short tail block would be below the STFT window and pass through
        # unrestored, leaving an audible band edge at the end of the file.
        if len(positions) > 1 and n_samples - positions[-1] < self.win_length:
            pulled = max(0, n_samples - block_len)
            if pulled > positions[-2]:
                positions[-1] = pulled
            else:
                # Pulling back that far would start before the previous block,
                # which already reaches the end — drop the stub instead.
                positions.pop()
        return positions

    def _restore_block(
        self,
        block: np.ndarray,
        effective_cutoff_hz: float,
        strengths: list[float],
        spk_idx_t: torch.Tensor | None,
        proto_t: torch.Tensor | None,
        generator: torch.Generator,
    ) -> dict[float, np.ndarray]:
        """Restore one block, returning one waveform per active strength."""
        n_block = len(block)
        Z_obs = self._stft(block)
        n_freqs, _ = Z_obs.shape

        mask = compute_transition_mask(
            n_freqs, self.sample_rate, effective_cutoff_hz, transition_hz=self.transition_hz
        )

        Z_real = torch.from_numpy(np.real(Z_obs)).float().unsqueeze(0).unsqueeze(0)
        Z_imag = torch.from_numpy(np.imag(Z_obs)).float().unsqueeze(0).unsqueeze(0)
        Z_tensor = torch.cat([Z_real, Z_imag], dim=1).to(self.device)

        Z_flow = self._solve_flow_ode(
            Z_tensor,
            cutoff_hz=effective_cutoff_hz,
            speaker_idx=spk_idx_t,
            speaker_proto=proto_t,
            generator=generator,
            steps=4,
        )
        flow_np = Z_flow.squeeze(0).cpu().numpy()
        Z_gen = flow_np[0] + 1j * flow_np[1]

        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        cutoff_bin = max(1, int(np.argmin(np.abs(freqs - effective_cutoff_hz))))

        # Acoustic envelope matching and energy gating
        obs_band_energy = np.sqrt(np.mean(np.abs(Z_obs[:cutoff_bin, :]) ** 2, axis=0) + 1e-12)
        obs_energy_median = float(np.median(obs_band_energy))
        obs_energy_threshold = obs_energy_median * 0.15

        hf_freqs = freqs[cutoff_bin:]
        decay = np.exp(-0.00015 * (hf_freqs - effective_cutoff_hz))[:, np.newaxis]
        obs_mag_ref = np.mean(np.abs(Z_obs[max(0, cutoff_bin - 20) : cutoff_bin, :])) + 1e-6
        gen_mag_ref = np.mean(np.abs(Z_gen[cutoff_bin:, :])) + 1e-6
        scale_factor = float((obs_mag_ref / gen_mag_ref) * 0.4)

        Z_gen_scaled = np.zeros_like(Z_obs)
        Z_gen_scaled[cutoff_bin:, :] = Z_gen[cutoff_bin:, :] * scale_factor * decay
        silent_frames = obs_band_energy < obs_energy_threshold
        Z_gen_scaled[:, silent_frames] = 0.0

        out: dict[float, np.ndarray] = {}
        for s in strengths:
            Z_merged = merge_protected_spectrum(Z_obs, Z_gen_scaled, mask, strength=s)
            x_rec = self._istft(Z_merged)
            if len(x_rec) < n_block:
                x_rec = np.pad(x_rec, (0, n_block - len(x_rec)))
            else:
                x_rec = x_rec[:n_block]
            # Deliberately not clipped to ±1: hard-clipping here would guarantee
            # a peak below Guard R's 1.05 rejection threshold, hiding an
            # overshoot from the guard and distorting the protected band with it.
            out[s] = x_rec.astype(np.float32)
        return out

    def restore(
        self,
        audio_48k: np.ndarray,
        sample_rate: int,  # noqa: ARG002
        effective_cutoff_hz: float,
        speaker_id: str | None = None,
        speaker_embedding: np.ndarray | None = None,
        f0_trajectory: np.ndarray | None = None,  # noqa: ARG002
        vuv_mask: np.ndarray | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,
        seed: int = 42,
    ) -> list[RestorationCandidate]:
        """Generate restored candidate waveforms across candidate high-band strengths."""
        if strengths is None:
            strengths = [1.0, 0.75, 0.5, 0.25, 0.0]

        nyquist = self.sample_rate / 2.0
        effective_cutoff_hz = float(np.clip(effective_cutoff_hz, 500.0, nyquist - 100.0))

        if audio_48k.size == 0:
            return [
                RestorationCandidate(
                    strength=s, audio=audio_48k.copy(), cutoff_hz=effective_cutoff_hz
                )
                for s in strengths
            ]

        is_multichannel = audio_48k.ndim == 2
        channels = audio_48k if is_multichannel else audio_48k[np.newaxis, :]
        n_channels = channels.shape[0]
        n_samples = channels.shape[1]

        candidates: list[RestorationCandidate] = []
        restored_channels_by_strength: dict[float, list[np.ndarray]] = {s: [] for s in strengths}

        spk_idx_t: torch.Tensor | None = None
        if speaker_id is not None and speaker_id in self.speaker_ids:
            idx = self.speaker_ids.index(speaker_id)
            spk_idx_t = torch.tensor([idx], device=self.device, dtype=torch.long)

        # Auto-load enrolled profile embedding when available.
        if speaker_embedding is None and speaker_id is not None:
            try:
                from hawavoclean.paths import profiles_root
                from hawavoclean.restoration.profiles import load_speaker_profile

                prof = load_speaker_profile(speaker_id, profiles_root=profiles_root())
                if prof.embedding_vector is not None:
                    speaker_embedding = prof.embedding_vector
            except Exception:
                pass  # Profile not found or not enrolled — proceed without prototype.

        proto_t: torch.Tensor | None = None
        if speaker_embedding is not None and len(speaker_embedding) == 192:
            proto_t = torch.from_numpy(speaker_embedding).float().unsqueeze(0).to(self.device)

        active_strengths = [s for s in strengths if s > 0.0]

        for ch_idx in range(n_channels):
            ch_audio = channels[ch_idx]
            # Too short to analyse, or nothing to generate: pass the channel
            # through rather than running the network for a discarded result.
            if len(ch_audio) < self.win_length or not active_strengths:
                for s in strengths:
                    restored_channels_by_strength[s].append(ch_audio)
                continue

            blocks = self._block_positions(n_samples)
            chan_out = {s: np.zeros(n_samples, dtype=np.float32) for s in active_strengths}
            covered = 0

            for block_idx, p0 in enumerate(blocks):
                p1 = min(p0 + self._block_len(), n_samples)
                # Seed per block, not per call: the result then depends only on
                # (job seed, channel, block) and not on how the file was split.
                gen = torch.Generator(device=self.device)
                gen.manual_seed((seed + 10007 * ch_idx + 65537 * block_idx) % (2**63 - 1))

                block_out = self._restore_block(
                    ch_audio[p0:p1],
                    effective_cutoff_hz=effective_cutoff_hz,
                    strengths=active_strengths,
                    spk_idx_t=spk_idx_t,
                    proto_t=proto_t,
                    generator=gen,
                )

                # Blocks overlap; cross-fade the seam so no click is introduced
                # at a boundary the listener never chose.
                xfade = max(0, min(covered - p0, p1 - p0)) if block_idx > 0 else 0
                for s, y in block_out.items():
                    dst = chan_out[s]
                    if xfade > 0:
                        ramp = np.linspace(0.0, 1.0, xfade, endpoint=False, dtype=np.float32)
                        dst[p0 : p0 + xfade] *= 1.0 - ramp
                        dst[p0 : p0 + xfade] += y[:xfade] * ramp
                        dst[p0 + xfade : p1] = y[xfade:]
                    else:
                        dst[p0:p1] = y
                covered = p1

            for s in strengths:
                if s == 0.0:
                    restored_channels_by_strength[s].append(ch_audio)
                else:
                    restored_channels_by_strength[s].append(chan_out[s])

        for s in strengths:
            arr_list = restored_channels_by_strength[s]
            merged_audio = np.stack(arr_list, axis=0) if is_multichannel else arr_list[0]
            final_audio = np.asarray(merged_audio, dtype=np.float32)
            if np.any(~np.isfinite(final_audio)):
                final_audio = np.zeros_like(final_audio)
            candidates.append(
                RestorationCandidate(
                    strength=s,
                    audio=final_audio,
                    cutoff_hz=effective_cutoff_hz,
                )
            )

        return candidates
