"""Training pipeline for HawaRestore-KD flow-matching bandwidth restoration.

Two data modes, selected explicitly on the command line:

- ``--data-dir DIR``: train on real clean WAV/FLAC files (any sample rate;
  resampled to 48 kHz), with degradations applied on the fly by
  ``research.restoration.simulation.degradation.DegradationSimulator``.
- ``--synthetic``: engineering fallback that generates the sine-harmonic
  simulation audio. It validates the training machinery only; the resulting
  weights have never heard Kurdish speech and must not be shipped as a
  restoration model.

Splits are speaker-disjoint and registered through ``SplitManager``
(``research/restoration/train/dataset.py``), which raises on any utterance
appearing in more than one split. The full composite loss from
``research/restoration/train/losses.py`` is active on every step:
flow-matching MSE, multi-resolution high-band STFT, cross-band envelope,
and speaker-identity cosine terms.

Run from the repository root as a module::

    .venv/bin/python -m research.restoration.train.train_hawarestore --synthetic
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy import signal
from torch.utils.data import DataLoader, Dataset

from hawavoclean.hashing import hash_bytes, hash_file
from hawavoclean.restoration.checkpoint import (
    compute_code_provenance,
    compute_dependency_provenance,
    load_safe_checkpoint,
    save_safe_checkpoint,
)
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKDNet
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor
from research.restoration.simulation.degradation import DegradationSimulator
from research.restoration.train.dataset import SplitManager, UtteranceEntry
from research.restoration.train.losses import HawaRestoreLoss

SAMPLE_RATE = 48000
SIGMA_MIN = 1e-4

#: Metric keys HawaRestoreLoss must report when every term is wired in.
REQUIRED_METRIC_KEYS = ("loss_flow", "loss_stft", "loss_env", "loss_speaker", "loss_total")

#: Checkpoint-facing names for the active loss terms, in REQUIRED_METRIC_KEYS order.
ACTIVE_LOSS_TERMS = ("flow", "stft", "envelope", "speaker", "total")


@dataclass(frozen=True)
class ItemSpec:
    """One training utterance: either a synthetic recipe or a real-file chunk."""

    utterance_id: str
    speaker_id: str
    source: str  # "synthetic" or absolute path to an audio file
    start_frame: int  # chunk start in source-file frames (0 for synthetic)
    duration_s: float
    seed: int  # per-item RNG seed for generation / degradation draws
    sha256: str
    session_id: str


def build_synthetic_items(
    num_items: int, duration_s: float, num_speakers: int = 10
) -> list[ItemSpec]:
    """Recipes for the sine-harmonic simulation set (round-robin over synthetic speakers)."""
    items: list[ItemSpec] = []
    for i in range(num_items):
        speaker_id = f"synthetic_{i % num_speakers:02d}"
        descriptor = f"synthetic:{i}:{speaker_id}:{duration_s}"
        items.append(
            ItemSpec(
                utterance_id=f"synthetic_utt_{i:04d}",
                speaker_id=speaker_id,
                source="synthetic",
                start_frame=0,
                duration_s=duration_s,
                seed=i,
                sha256=hash_bytes(descriptor.encode("utf-8")),
                session_id="synthetic",
            )
        )
    return items


def build_real_items(data_dir: Path, duration_s: float) -> list[ItemSpec]:
    """Chunk every WAV/FLAC under ``data_dir`` into fixed-duration utterances.

    Speaker attribution: the immediate parent directory name when files are
    organised as ``data_dir/<speaker>/<file>``, otherwise the filename stem up
    to the first underscore.
    """
    audio_paths = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
    )
    if not audio_paths:
        raise FileNotFoundError(f"No .wav/.flac files found under {data_dir}")

    items: list[ItemSpec] = []
    for file_idx, path in enumerate(audio_paths):
        info = sf.info(str(path))
        if path.parent.resolve() == data_dir.resolve():
            speaker_id = path.stem.split("_")[0]
        else:
            speaker_id = path.parent.name
        chunk_frames = int(duration_s * info.samplerate)
        if chunk_frames <= 0:
            raise ValueError(f"duration_s={duration_s} yields empty chunks for {path}")
        file_hash = hash_file(path)
        for chunk_idx in range(int(info.frames) // chunk_frames):
            start = chunk_idx * chunk_frames
            items.append(
                ItemSpec(
                    utterance_id=f"utt{file_idx:05d}_{path.stem}_c{chunk_idx:04d}",
                    speaker_id=speaker_id,
                    source=str(path.resolve()),
                    start_frame=start,
                    duration_s=duration_s,
                    seed=file_idx * 100_003 + chunk_idx,
                    sha256=hash_bytes(f"{file_hash}:{start}:{start + chunk_frames}".encode()),
                    session_id=path.stem,
                )
            )
    return items


def split_items_speaker_disjoint(
    items: list[ItemSpec], split_seed: int, val_fraction: float = 0.2
) -> tuple[list[ItemSpec], list[ItemSpec], list[str], list[str]]:
    """Partition items so no speaker appears in both train and validation."""
    speakers = sorted({item.speaker_id for item in items})
    if len(speakers) < 2:
        raise ValueError(f"Speaker-disjoint splitting requires at least 2 speakers; got {speakers}")
    rng = np.random.default_rng(split_seed)
    order = [speakers[int(i)] for i in rng.permutation(len(speakers))]
    n_val = min(max(1, round(val_fraction * len(speakers))), len(speakers) - 1)
    val_speakers = set(order[:n_val])
    train_speakers = set(order[n_val:])
    train_items = [item for item in items if item.speaker_id in train_speakers]
    val_items = [item for item in items if item.speaker_id in val_speakers]
    return train_items, val_items, sorted(train_speakers), sorted(val_speakers)


def register_splits(
    manifests_dir: Path, train_items: list[ItemSpec], val_items: list[ItemSpec]
) -> dict[str, str]:
    """Register both splits with SplitManager (leakage hard-fail) and save manifests."""
    manager = SplitManager(manifests_dir)
    for split, split_items in (("train", train_items), ("development", val_items)):
        for item in split_items:
            audio_path = (
                item.source if item.source != "synthetic" else f"synthetic://{item.utterance_id}"
            )
            manager.add_utterance(
                split,
                UtteranceEntry(
                    utterance_id=item.utterance_id,
                    speaker_id=item.speaker_id,
                    audio_path=audio_path,
                    duration_s=item.duration_s,
                    sha256=item.sha256,
                    session_id=item.session_id,
                ),
            )
    return manager.save_manifests()


class RestorationTrainingDataset(Dataset[dict[str, torch.Tensor]]):
    """Clean/degraded STFT pairs from synthetic recipes or real WAV chunks.

    Synthetic items reproduce the original sine-harmonic simulation (zero-phase
    Butterworth lowpass). Real items are loaded from disk, resampled to 48 kHz,
    peak-normalised, and degraded on the fly with DegradationSimulator using a
    per-item deterministic seed.
    """

    def __init__(
        self,
        items: list[ItemSpec],
        speaker_table: list[str],
        sr: int = SAMPLE_RATE,
        n_fft: int = 1024,
        profiles_dir: Path | None = None,
    ) -> None:
        self.items = items
        self.speaker_index = {speaker: i for i, speaker in enumerate(speaker_table)}
        self.sr = sr
        self.n_fft = n_fft
        self.hop = n_fft // 2
        self.extractor = SpeakerEmbeddingExtractor(sample_rate=sr)
        self.simulator = DegradationSimulator(sample_rate=sr)

        # Load canonical embeddings from enrolled profiles when available.
        # Falls back to per-chunk extraction for speakers without profiles.
        self.canonical_embeddings: dict[str, np.ndarray] = {}
        if profiles_dir is not None:
            for speaker_id in speaker_table:
                emb_path = profiles_dir / speaker_id / "embedding" / "profile.npy"
                if emb_path.exists():
                    emb = np.load(emb_path).astype(np.float32)
                    if len(emb) == 192 and np.linalg.norm(emb) > 1e-6:
                        self.canonical_embeddings[speaker_id] = emb

        # In-memory file cache: source_path -> (resampled_full_audio_48k, orig_sr)
        # Eliminates millions of redundant soundfile read/resample calls.
        self._file_cache: dict[str, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _get_full_audio(self, source_path: str) -> tuple[np.ndarray, int]:
        if source_path not in self._file_cache:
            audio, file_sr = sf.read(source_path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            if int(file_sr) != self.sr:
                g = int(np.gcd(self.sr, int(file_sr)))
                audio = signal.resample_poly(audio, self.sr // g, int(file_sr) // g)
            self._file_cache[source_path] = (audio.astype(np.float32), int(file_sr))
        return self._file_cache[source_path]

    def _synthesize(self, item: ItemSpec, rng: np.random.Generator) -> np.ndarray:
        f0 = float(rng.uniform(90.0, 260.0))
        n = int(self.sr * item.duration_s)
        t = np.linspace(0, item.duration_s, n, endpoint=False, dtype=np.float32)
        sig = np.zeros_like(t)
        for h in range(1, 35):
            freq = h * f0
            if freq >= self.sr / 2:
                break
            sig += (1.0 / (h**0.8)) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
        out: np.ndarray = (sig / (np.max(np.abs(sig)) + 1e-6) * 0.7).astype(np.float32)
        return out

    def _load_real(self, item: ItemSpec) -> np.ndarray:
        full_audio, orig_sr = self._get_full_audio(item.source)
        if orig_sr != self.sr:
            start_48k = int(round(item.start_frame * (self.sr / orig_sr)))
        else:
            start_48k = item.start_frame
        n = int(self.sr * item.duration_s)
        audio = full_audio[start_48k : start_48k + n]
        if len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        audio = audio[:n]
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = audio / peak * 0.7
        return audio.astype(np.float32)

    def _degrade(
        self, clean: np.ndarray, item: ItemSpec, rng: np.random.Generator, cutoff_hz: float
    ) -> np.ndarray:
        if item.source == "synthetic":
            # The original synthetic pipeline: zero-phase Butterworth lowpass only.
            sos = signal.butter(
                4,
                min(self.sr / 2 - 100.0, cutoff_hz) / (self.sr / 2),
                btype="lowpass",
                output="sos",
            )
            return np.asarray(signal.sosfiltfilt(sos, clean), dtype=np.float32)
        filter_type = str(rng.choice(["butterworth", "chebyshev", "codec_shape"]))
        snr_db = float(rng.uniform(30.0, 55.0))
        degraded, _ = self.simulator.degrade(
            clean,
            cutoff_hz=cutoff_hz,
            filter_type=filter_type,
            filter_order=8,
            add_noise_snr_db=snr_db,
            seed=item.seed,
        )
        return degraded

    def _stft_tensor(self, audio: np.ndarray) -> torch.Tensor:
        _, _, Z = signal.stft(
            audio,
            fs=self.sr,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop,
            boundary="zeros",
        )
        real = torch.from_numpy(np.ascontiguousarray(np.real(Z))).float()
        imag = torch.from_numpy(np.ascontiguousarray(np.imag(Z))).float()
        return torch.stack([real, imag], dim=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        rng = np.random.default_rng(item.seed)
        cutoff_hz = float(rng.uniform(2500.0, 16000.0))
        clean = self._synthesize(item, rng) if item.source == "synthetic" else self._load_real(item)
        degraded = self._degrade(clean, item, rng, cutoff_hz)

        # Prefer enrolled canonical embedding; fall back to per-chunk extraction.
        if item.speaker_id in self.canonical_embeddings:
            proto = self.canonical_embeddings[item.speaker_id]
        else:
            proto = self.extractor.extract(clean)

        return {
            "clean_audio": torch.from_numpy(np.ascontiguousarray(clean)).float(),
            "degraded_audio": torch.from_numpy(np.ascontiguousarray(degraded)).float(),
            "clean_stft": self._stft_tensor(clean),
            "degraded_stft": self._stft_tensor(degraded),
            "cutoff_hz": torch.tensor(cutoff_hz, dtype=torch.float32),
            "speaker_idx": torch.tensor(self.speaker_index[item.speaker_id], dtype=torch.long),
            "speaker_proto": torch.from_numpy(proto).float(),
        }


class DifferentiableSpeakerEmbed(nn.Module):
    """Differentiable 192-dimensional neural speaker embedding for speaker-identity loss.

    Torch re-implementation of ``SpeakerEmbeddingExtractor`` (which is numpy and non-differentiable),
    so gradients flow from the cosine identity term back through the predicted audio into 192 dimensions.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 1024,
        n_mels: int = 40,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.embed_dim = embed_dim
        n_freqs = n_fft // 2 + 1

        mel_low = 0.0
        mel_high = 2595.0 * math.log10(1.0 + 8000.0 / 700.0)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(1, n_mels + 1):
            lo, mid, hi = int(bin_points[m - 1]), int(bin_points[m]), int(bin_points[m + 1])
            for k in range(lo, mid):
                if mid > lo and k < n_freqs:
                    fbank[m - 1, k] = (k - lo) / (mid - lo)
            for k in range(mid, hi):
                if hi > mid and k < n_freqs:
                    fbank[m - 1, k] = (hi - k) / (hi - mid)

        # DCT basis for MFCCs (20 coefficients)
        n_mfcc = 20
        dct_basis = np.zeros((n_mfcc, n_mels), dtype=np.float32)
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct_basis[i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * n_mels))

        # Deterministic projection matrix (38 -> 192)
        rng = np.random.default_rng(42)
        raw_proj = rng.standard_normal((192, embed_dim), dtype=np.float32)
        q_proj, _ = np.linalg.qr(raw_proj)
        q_proj = q_proj[:38, :].astype(np.float32)

        self.register_buffer("fbank", torch.from_numpy(fbank))
        self.register_buffer("dct_basis", torch.from_numpy(dct_basis))
        self.register_buffer("proj_matrix", torch.from_numpy(q_proj))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, N) waveforms -> (B, 192) unit-normalised embeddings."""
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=cast(torch.Tensor, self.window),
            return_complex=True,
        )
        mag = torch.abs(spec) + 1e-10  # (B, F, T)
        mel = torch.log(torch.einsum("mf,bft->bmt", self.fbank, mag) + 1e-6)  # (B, 40, T)

        # MFCCs (20, T)
        mfcc = torch.einsum("km,bmt->bkt", self.dct_basis, mel)
        mfcc_shape = mfcc[:, 1:, :]  # (B, 19, T) discarding gain MFCC 0

        mfcc_mean = mfcc_shape.mean(dim=-1)  # (B, 19)
        mfcc_std = mfcc_shape.std(dim=-1)  # (B, 19)
        feat_38 = torch.cat([mfcc_mean, mfcc_std], dim=-1)  # (B, 38)
        feat_norm = feat_38 / (torch.norm(feat_38, dim=-1, keepdim=True) + 1e-9)

        # Neural feature projection & GELU activation
        projected = torch.matmul(feat_norm, cast(torch.Tensor, self.proj_matrix))  # (B, 192)
        gelu = torch.nn.functional.gelu(projected)
        return torch.nn.functional.normalize(gelu, dim=-1)


def _run_epoch(
    model: HawaRestoreKDNet,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_fn: HawaRestoreLoss,
    spk_embed: DifferentiableSpeakerEmbed,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    n_fft: int,
    max_steps: int | None = None,
) -> dict[str, float]:
    """One pass over ``loader``; trains when ``optimizer`` is given, else evaluates.

    Returns the per-term average of the metrics reported by HawaRestoreLoss and
    raises if any composite-loss term failed to activate (wiring regression).
    """
    training = optimizer is not None
    model.train(training)
    hop = n_fft // 2
    window = torch.hann_window(n_fft, device=device)
    sums: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        clean_stft = batch["clean_stft"].to(device)  # (B, 2, F, T)
        cutoff_hz = batch["cutoff_hz"].to(device)  # (B,)
        spk_idx = batch["speaker_idx"].to(device)  # (B,)
        spk_proto = batch["speaker_proto"].to(device)  # (B, 192)
        B = clean_stft.shape[0]

        with torch.set_grad_enabled(training):
            # Use real degraded observation STFT from DegradationSimulator
            x_obs = batch["degraded_stft"].to(device)

            # Flow matching on the linear probability path with low-band guidance
            t = torch.rand(B, device=device)
            x0 = torch.randn_like(clean_stft)
            x1 = clean_stft
            t_expand = t.view(B, 1, 1, 1)
            x_t = (1.0 - (1.0 - SIGMA_MIN) * t_expand) * x0 + t_expand * x1
            target_v = x1 - (1.0 - SIGMA_MIN) * x0
            pred_v = model(x_t, t, cutoff_hz, spk_idx, spk_proto, x_obs=x_obs)

            # One-step clean estimate from the path identity
            #   x1 = (1 - sigma_min) * x_t + (1 - (1 - sigma_min) * t) * v,
            # inverted to audio so the STFT / envelope / speaker terms are live.
            x1_hat = (1.0 - SIGMA_MIN) * x_t + (1.0 - (1.0 - SIGMA_MIN) * t_expand) * pred_v
            pred_audio = torch.istft(
                torch.complex(x1_hat[:, 0], x1_hat[:, 1]),
                n_fft=n_fft,
                hop_length=hop,
                window=window,
            )
            target_audio = torch.istft(
                torch.complex(x1[:, 0], x1[:, 1]), n_fft=n_fft, hop_length=hop, window=window
            )

            # Batch-min cutoff: the high-band mask must cover every item's
            # missing band, so the most conservative (lowest) cutoff wins.
            loss, metrics = loss_fn(
                pred_v,
                target_v,
                pred_audio=pred_audio,
                target_audio=target_audio,
                cutoff_hz=float(cutoff_hz.min().item()),
                pred_speaker_emb=spk_embed(pred_audio),
                target_speaker_emb=spk_embed(target_audio),
            )

        missing = set(REQUIRED_METRIC_KEYS) - set(metrics)
        if missing:
            raise RuntimeError(
                f"Loss wiring regression: terms {sorted(missing)} were not activated. "
                "Every composite term must be live on every step."
            )

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        n_batches += 1

        if n_batches % 10 == 0 or n_batches == len(loader):
            mode_lbl = "Train" if training else "Val"
            total_b = len(loader)
            loss_val = metrics.get("loss_total", 0.0)
            flow_val = metrics.get("loss_flow", 0.0)
            stft_val = metrics.get("loss_stft", 0.0)
            print(
                f"[{mode_lbl}] Batch {n_batches}/{total_b} - "
                f"loss={loss_val:.4f} (flow={flow_val:.4f}, stft={stft_val:.4f})",
                flush=True,
            )

        if max_steps is not None and n_batches >= max_steps:
            break

    return {key: value / max(1, n_batches) for key, value in sums.items()}


def _term_losses(metrics: dict[str, float]) -> dict[str, float]:
    """Map HawaRestoreLoss metric keys to checkpoint-facing term names."""
    key_map = dict(zip(ACTIVE_LOSS_TERMS, REQUIRED_METRIC_KEYS, strict=True))
    return {term: metrics[key] for term, key in key_map.items()}


def _serialize_rng_state() -> dict[str, Any]:
    """Capture PyTorch and NumPy RNG states using weights_only-safe types."""
    np_s = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "numpy": {
            "algo": str(np_s[0]),
            "keys": torch.from_numpy(np_s[1].copy()),
            "pos": int(np_s[2]),
            "has_gauss": int(np_s[3]),
            "cached_gauss": float(np_s[4]),
        },
    }


def _restore_rng_state(rng_s: dict[str, Any]) -> None:
    """Restore PyTorch and NumPy RNG states safely."""
    if "torch" in rng_s and isinstance(rng_s["torch"], torch.Tensor):
        torch.set_rng_state(rng_s["torch"])
    if "numpy" in rng_s and isinstance(rng_s["numpy"], dict):
        np_dict = rng_s["numpy"]
        keys = np_dict.get("keys")
        if isinstance(keys, torch.Tensor):
            keys_arr = keys.numpy()
        else:
            keys_arr = np.array(keys, dtype=np.uint32)
        np.random.set_state(
            (
                str(np_dict["algo"]),
                keys_arr,
                int(np_dict["pos"]),
                int(np_dict["has_gauss"]),
                float(np_dict["cached_gauss"]),
            )
        )


def train_model(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    output_dir: Path | str = "models/hawarestore-kd-candidate",
    data_dir: Path | str | None = None,
    synthetic: bool = False,
    num_synthetic_items: int = 40,
    duration_s: float = 1.0,
    n_fft: int = 1024,
    base_channels: int = 64,
    split_seed: int = 20260824,
    val_fraction: float = 0.2,
    device: str | None = None,
    overwrite: bool = False,
    profiles_dir: Path | str | None = None,
    num_workers: int = 4,
    resume_path: Path | str | None = None,
    max_seconds: float | None = None,
    max_steps_per_epoch: int | None = None,
    save_safetensors: bool = False,
) -> Path:
    """Train HawaRestoreKDNet with speaker-disjoint splits and the full composite loss.

    Saves ``hawarestore_kd.pt`` plus split manifests under ``output_dir`` and
    returns the checkpoint path. Refuses to overwrite an existing checkpoint
    unless ``overwrite=True``; promoting a candidate into
    ``models/hawarestore-kd/`` is a deliberate, user-gated step.

    Features:
    - Locked best model saving: records the weights with lowest validation loss.
    - Resumable training: restores optimizer, epoch, and RNG states via ``resume_path``.
    - Safe checkpoint contract: loaded with ``weights_only=True``, saves provenance hashes.
    - Bounded execution: stops cleanly when ``max_seconds`` or ``max_steps_per_epoch`` is met.
    """
    if synthetic == (data_dir is not None):
        raise ValueError("Choose exactly one data mode: --data-dir <real WAVs> or --synthetic.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / "hawarestore_kd.pt"
    if ckpt_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint {ckpt_path}. Pass --overwrite "
            "to replace it deliberately. The committed production checkpoint under "
            "models/hawarestore-kd/ is only replaced via the user-gated real-data process."
        )

    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    if synthetic:
        print(
            "WARNING - SYNTHETIC FALLBACK MODE: training on generated sine-harmonic "
            "audio. This validates the training machinery only; the resulting weights "
            "have never heard Kurdish speech and must not be shipped as a restoration model."
        )
        items = build_synthetic_items(num_synthetic_items, duration_s=duration_s)
        data_mode = "synthetic"
    else:
        assert data_dir is not None
        items = build_real_items(Path(data_dir), duration_s=duration_s)
        data_mode = "real"

    train_items, val_items, train_speakers, val_speakers = split_items_speaker_disjoint(
        items, split_seed=split_seed, val_fraction=val_fraction
    )
    manifest_hashes = register_splits(out_path / "manifests", train_items, val_items)
    speaker_table = sorted(train_speakers + val_speakers)

    print(
        f"Training HawaRestore-KD on device: {device} for {epochs} epochs "
        f"({data_mode} mode; {len(train_items)} train / {len(val_items)} val items; "
        f"{len(train_speakers)} train / {len(val_speakers)} val speakers, disjoint)"
    )

    prof_path = Path(profiles_dir) if profiles_dir is not None else None
    train_ds = RestorationTrainingDataset(
        train_items, speaker_table, n_fft=n_fft, profiles_dir=prof_path
    )
    val_ds = RestorationTrainingDataset(
        val_items, speaker_table, n_fft=n_fft, profiles_dir=prof_path
    )
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    # Seed RNGs deterministically for reproducibility
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    config = {
        "in_channels": 2,
        "out_channels": 2,
        "base_channels": base_channels,
        "num_speakers": len(speaker_table),
        "speaker_embed_dim": 64,
        "prototype_dim": 192,
        "cond_dim": 256,
        "n_fft": n_fft,
    }
    model = HawaRestoreKDNet(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = HawaRestoreLoss().to(device)
    spk_embed = DifferentiableSpeakerEmbed().to(device)

    start_epoch = 1
    best_epoch = 1
    best_val_loss = float("inf")
    best_model_state_dict: dict[str, torch.Tensor] = {}
    best_train_metrics: dict[str, float] = {}
    best_val_metrics: dict[str, float] = {}

    if resume_path is not None:
        print(f"Resuming HawaRestore-KD training from checkpoint: {resume_path}")
        resume_dict = load_safe_checkpoint(resume_path, map_location=device)
        model.load_state_dict(resume_dict["model_state_dict"])
        if (
            "optimizer_state_dict" in resume_dict
            and resume_dict["optimizer_state_dict"] is not None
        ):
            optimizer.load_state_dict(resume_dict["optimizer_state_dict"])
        start_epoch = int(resume_dict.get("epoch", 0)) + 1
        best_epoch = int(resume_dict.get("best_epoch", 1))
        best_val_loss = float(resume_dict.get("best_val_loss", float("inf")))
        if "best_model_state_dict" in resume_dict and isinstance(
            resume_dict["best_model_state_dict"], dict
        ):
            best_model_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in resume_dict["best_model_state_dict"].items()
                if isinstance(v, torch.Tensor)
            }
        if "best_train_metrics" in resume_dict and isinstance(
            resume_dict["best_train_metrics"], dict
        ):
            best_train_metrics = dict(resume_dict["best_train_metrics"])
        if "best_val_metrics" in resume_dict and isinstance(resume_dict["best_val_metrics"], dict):
            best_val_metrics = dict(resume_dict["best_val_metrics"])
        if "rng_state" in resume_dict and isinstance(resume_dict["rng_state"], dict):
            _restore_rng_state(resume_dict["rng_state"])

    train_metrics: dict[str, float] = {}
    val_metrics: dict[str, float] = {}
    epoch_history: list[dict[str, Any]] = []
    time_start = time.monotonic()

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            loss_fn,
            spk_embed,
            optimizer,
            device,
            n_fft,
            max_steps=max_steps_per_epoch,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            loss_fn,
            spk_embed,
            None,
            device,
            n_fft,
            max_steps=max_steps_per_epoch,
        )
        train_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(train_metrics.items()))
        val_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
        print(f"Epoch {epoch}/{epochs} - train: {train_str} | val: {val_str}", flush=True)

        epoch_history.append(
            {
                "epoch": epoch,
                "train": _term_losses(train_metrics),
                "val": _term_losses(val_metrics),
            }
        )

        current_val_loss = val_metrics["loss_total"]
        if current_val_loss < best_val_loss or not best_model_state_dict:
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_model_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_train_metrics = dict(train_metrics)
            best_val_metrics = dict(val_metrics)
            print(
                f"  --> Locked new best validation loss: {best_val_loss:.4f} at epoch {epoch}",
                flush=True,
            )

        # Save resumable checkpoint each epoch
        last_ckpt_path = out_path / "hawarestore_kd_last.pt"
        last_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_model_state_dict": best_model_state_dict,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "config": config,
            "data_mode": data_mode,
            "n_train": len(train_items),
            "n_val": len(val_items),
            "split_seed": split_seed,
            "train_speakers": train_speakers,
            "val_speakers": val_speakers,
            "manifest_hashes": manifest_hashes,
            "active_loss_terms": list(ACTIVE_LOSS_TERMS),
            "loss_weights": {
                "flow": loss_fn.lambda_flow,
                "stft": loss_fn.lambda_stft,
                "envelope": loss_fn.lambda_envelope,
                "speaker": loss_fn.lambda_speaker,
            },
            "rng_state": _serialize_rng_state(),
        }
        save_safe_checkpoint(last_payload, last_ckpt_path, save_safetensors=False)

        # Bounded execution check
        if max_seconds is not None:
            elapsed = time.monotonic() - time_start
            if elapsed >= max_seconds:
                print(
                    f"Bounded training time ceiling reached ({elapsed:.1f}s >= {max_seconds}s). "
                    f"Halting at epoch {epoch}.",
                    flush=True,
                )
                break

    # Save final best-model checkpoint (locked best model, not incidental final)
    final_payload = {
        "model_state_dict": best_model_state_dict if best_model_state_dict else model.state_dict(),
        "config": config,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_loss": best_train_metrics.get("loss_total", train_metrics.get("loss_total", 0.0)),
        "data_mode": data_mode,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "split_seed": split_seed,
        "train_speakers": train_speakers,
        "val_speakers": val_speakers,
        "manifest_hashes": manifest_hashes,
        "active_loss_terms": list(ACTIVE_LOSS_TERMS),
        "loss_weights": {
            "flow": loss_fn.lambda_flow,
            "stft": loss_fn.lambda_stft,
            "envelope": loss_fn.lambda_envelope,
            "speaker": loss_fn.lambda_speaker,
        },
        "final_losses": {
            "train": _term_losses(best_train_metrics if best_train_metrics else train_metrics),
            "val": _term_losses(best_val_metrics if best_val_metrics else val_metrics),
        },
        "epoch_history": epoch_history,
        "code_hash": compute_code_provenance(),
        "dependency_versions": compute_dependency_provenance(),
    }
    save_safe_checkpoint(final_payload, ckpt_path, save_safetensors=save_safetensors)
    ckpt_hash = hash_file(ckpt_path)
    print(
        f"Saved trained best checkpoint: {ckpt_path} (SHA-256: {ckpt_hash}, best_epoch: {best_epoch})"
    )
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory of real clean WAV/FLAC files (any sample rate; resampled to 48 kHz).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="EXPLICIT FALLBACK: train on generated sine-harmonic simulation audio "
        "(machinery validation only, not Kurdish speech).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/hawarestore-kd-candidate"))
    parser.add_argument("--num-synthetic-items", type=int, default=40)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=20260824)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing checkpoint at the output path.",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Path to enrolled profiles directory (e.g., profiles/). "
        "Uses canonical embeddings for voice conditioning when available.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker subprocesses for parallel audio loading.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing checkpoint to resume training from.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Maximum training runtime ceiling in seconds.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Maximum number of mini-batches to process per epoch.",
    )
    parser.add_argument(
        "--save-safetensors",
        action="store_true",
        help="Export a companion HuggingFace safetensors file and metadata JSON.",
    )
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        num_synthetic_items=args.num_synthetic_items,
        duration_s=args.duration_s,
        n_fft=args.n_fft,
        base_channels=args.base_channels,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        device=args.device,
        overwrite=args.overwrite,
        profiles_dir=args.profiles_dir,
        num_workers=args.num_workers,
        resume_path=args.resume,
        max_seconds=args.max_seconds,
        max_steps_per_epoch=args.max_steps_per_epoch,
        save_safetensors=args.save_safetensors,
    )


if __name__ == "__main__":
    main()
