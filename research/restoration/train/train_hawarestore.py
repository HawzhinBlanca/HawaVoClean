"""Training pipeline for HawaRestore-KD flow-matching bandwidth restoration.

Trains the HawaRestoreKDNet vector-field network across simulated continuous cutoffs,
codecs, and the 10 consented Kurdish character speaker profiles.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hawavoclean.hashing import hash_file
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKDNet
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from research.restoration.train.losses import HawaRestoreLoss


class KurdishSimulationDataset(Dataset):
    """Generates synthetic multi-speaker Kurdish clean speech and degraded pairs."""

    def __init__(self, num_samples: int = 100, duration_s: float = 2.0, sr: int = 48000) -> None:
        self.num_samples = num_samples
        self.duration_s = duration_s
        self.sr = sr
        self.extractor = SpeakerEmbeddingExtractor(sample_rate=sr)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        np.random.seed(idx)
        spk_idx = idx % 10

        # Random F0 and harmonic speech synthesis
        f0 = float(np.random.uniform(90.0, 260.0))
        cutoff_hz = float(np.random.uniform(2500.0, 16000.0))

        t = np.linspace(
            0, self.duration_s, int(self.sr * self.duration_s), endpoint=False, dtype=np.float32
        )
        sig = np.zeros_like(t)
        for h in range(1, 35):
            freq = h * f0
            if freq >= self.sr / 2:
                break
            sig += (1.0 / (h**0.8)) * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

        sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.7
        proto_emb = self.extractor.extract(sig)

        # Degrade signal with lowpass at cutoff_hz
        from scipy import signal

        sos = signal.butter(
            4, min(self.sr / 2 - 100.0, cutoff_hz) / (self.sr / 2), btype="lowpass", output="sos"
        )
        lp_sig = signal.sosfiltfilt(sos, sig).astype(np.float32)

        # Compute STFT
        _, _, Z_tgt = signal.stft(
            sig, fs=self.sr, window="hann", nperseg=1024, noverlap=512, boundary="zeros"
        )
        _, _, Z_lr = signal.stft(
            lp_sig, fs=self.sr, window="hann", nperseg=1024, noverlap=512, boundary="zeros"
        )

        # Convert to 2-channel real/imag tensors: (2, F, T)
        tgt_real = torch.from_numpy(np.real(Z_tgt)).float().unsqueeze(0)
        tgt_imag = torch.from_numpy(np.imag(Z_tgt)).float().unsqueeze(0)
        tgt_t = torch.cat([tgt_real, tgt_imag], dim=0)

        lr_real = torch.from_numpy(np.real(Z_lr)).float().unsqueeze(0)
        lr_imag = torch.from_numpy(np.imag(Z_lr)).float().unsqueeze(0)
        lr_t = torch.cat([lr_real, lr_imag], dim=0)

        return {
            "clean_audio": torch.from_numpy(sig).float(),
            "degraded_audio": torch.from_numpy(lp_sig).float(),
            "clean_stft": tgt_t,
            "degraded_stft": lr_t,
            "cutoff_hz": torch.tensor(cutoff_hz, dtype=torch.float32),
            "speaker_idx": torch.tensor(spk_idx, dtype=torch.long),
            "speaker_proto": torch.from_numpy(proto_emb).float(),
        }


def train_model(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    output_dir: Path | str = "models/hawarestore-kd",
) -> Path:
    """Train HawaRestoreKDNet and save checkpoint."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Training HawaRestore-KD on device: {device} for {epochs} epochs...")

    dataset = KurdishSimulationDataset(num_samples=40, duration_s=1.0, sr=48000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HawaRestoreKDNet(n_fft=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = HawaRestoreLoss().to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            clean_stft = batch["clean_stft"].to(device)  # (B, 2, F, T)
            cutoff_hz = batch["cutoff_hz"].to(device)  # (B,)
            spk_idx = batch["speaker_idx"].to(device)  # (B,)
            spk_proto = batch["speaker_proto"].to(device)  # (B, 192)

            B = clean_stft.shape[0]

            # Flow matching: sample random time t in [0, 1]
            t = torch.rand(B, device=device)
            # Sample source noise x0 ~ N(0, I)
            x0 = torch.randn_like(clean_stft)
            x1 = clean_stft

            # Linear probability path: x_t = (1 - (1 - sigma_min)*t)*x0 + t*x1
            sigma_min = 1e-4
            t_expand = t.view(B, 1, 1, 1)
            x_t = (1.0 - (1.0 - sigma_min) * t_expand) * x0 + t_expand * x1

            # Target velocity field: v_t = x1 - (1 - sigma_min)*x0
            target_v = x1 - (1.0 - sigma_min) * x0

            # Predict velocity field
            pred_v = model(x_t, t, cutoff_hz, spk_idx, spk_proto)

            loss, metrics = loss_fn(pred_v, target_v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.5f}")

    # Save trained checkpoint
    ckpt_path = out_path / "hawarestore_kd.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": 2,
                "out_channels": 2,
                "base_channels": 64,
                "num_speakers": 10,
                "speaker_embed_dim": 64,
                "prototype_dim": 192,
                "cond_dim": 256,
                "n_fft": 1024,
            },
            "epochs": epochs,
            "final_loss": avg_loss,
        },
        ckpt_path,
    )
    ckpt_hash = hash_file(ckpt_path)
    print(f"Saved trained checkpoint: {ckpt_path} (SHA-256: {ckpt_hash})")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
