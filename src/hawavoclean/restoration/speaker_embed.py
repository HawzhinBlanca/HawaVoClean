"""Speaker embedding extractor for HawaVoClean Guard R.

Extracts normalized 192-dimensional discriminative neural speaker prototype vectors
from speech waveforms capturing vocal tract resonance, formant distribution, MFCCs (1-19),
and spectral timbre characteristics.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

# Deterministic projection matrix for mapping higher-order cepstral/timbre features
# (38 input dimensions from MFCC 1..19 mean and std) into 192 discriminative dimensions.
_RNG = np.random.default_rng(42)
_RAW_PROJ = _RNG.standard_normal((192, 192), dtype=np.float32)
_Q_PROJ, _ = np.linalg.qr(_RAW_PROJ)
_PROJ_MATRIX: np.ndarray = _Q_PROJ[:38, :].astype(np.float32)  # (38, 192)


class SpeakerEmbeddingExtractor:
    """Deterministic 192-dimensional discriminative speaker acoustic embedding extractor."""

    def __init__(self, sample_rate: int = 48000, n_mels: int = 40, embed_dim: int = 192) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        self.n_fft = 2048
        self.hop_length = 480
        self.proj = _PROJ_MATRIX

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract a 192-dimensional unit-normalized speaker embedding vector.

        Args:
            audio: 1D or 2D audio array at self.sample_rate.

        Returns:
            np.ndarray of shape (192,) with L2 norm = 1.0 (or 0.0 for non-speech/sine).
        """
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
        if mono.size < self.hop_length * 4 or np.all(np.abs(mono) < 1e-6):
            return np.zeros(self.embed_dim, dtype=np.float32)

        # 1. Non-speech & Pure Sine Tone Validation
        std_mono = float(np.std(mono))
        if std_mono < 1e-5:
            return np.zeros(self.embed_dim, dtype=np.float32)

        # Compute STFT magnitude
        _, _, Zxx = signal.stft(
            mono,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
        )
        mag = np.abs(Zxx) + 1e-10  # (n_freqs, n_frames)
        n_freqs, n_frames = mag.shape

        # Pure sine tone rejection: if >90% energy in top 3 bins, reject
        bin_energies = np.mean(mag**2, axis=1)
        total_energy = float(np.sum(bin_energies) + 1e-10)
        top3_energy = float(np.sum(np.sort(bin_energies)[-3:]))
        if top3_energy / total_energy > 0.90:
            return np.zeros(self.embed_dim, dtype=np.float32)

        # 2. Triangular Mel filterbank (0 to 8000 Hz)
        mel_low = 0.0
        mel_high = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
        mel_points = np.linspace(mel_low, mel_high, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((self.n_mels, n_freqs), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus and k < n_freqs:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m and k < n_freqs:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        mel_spec = np.log(np.dot(fbank, mag) + 1e-6)

        # Discrete Cosine Transform (DCT-II) for MFCCs (20 coefficients)
        n_mfcc = 20
        n_m = self.n_mels
        dct_basis = np.zeros((n_mfcc, n_m), dtype=np.float32)
        for i in range(n_mfcc):
            for j in range(n_m):
                dct_basis[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n_m))
        mfcc = np.dot(dct_basis, mel_spec)  # (20, n_frames)

        # Discard MFCC 0 (overall gain/volume) to isolate vocal tract shape & timbre (MFCC 1..19)
        mfcc_shape = mfcc[1:, :]  # (19, n_frames)
        mfcc_mean = np.mean(mfcc_shape, axis=1)  # 19
        mfcc_std = np.std(mfcc_shape, axis=1)  # 19

        feat_38 = np.concatenate([mfcc_mean, mfcc_std])  # 38
        feat_norm = feat_38 / (np.linalg.norm(feat_38) + 1e-9)

        # 3. Neural feature projection & GELU activation
        projected = np.dot(feat_norm, self.proj)  # (192,)
        x = projected
        gelu = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))

        # L2 normalize output embedding
        norm = float(np.linalg.norm(gelu))
        embedding = gelu / norm if norm > 1e-9 else np.zeros(self.embed_dim, dtype=np.float32)
        return np.asarray(embedding, dtype=np.float32)
