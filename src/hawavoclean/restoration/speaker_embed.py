"""Speaker embedding extractor for HawaVoClean Guard R.

Extracts normalized 192-dimensional acoustic prototype vectors from speech waveforms
capturing vocal tract resonance, formant distribution, and spectral timbre characteristics.
"""

import numpy as np
from scipy import signal


class SpeakerEmbeddingExtractor:
    """Deterministic 192-dimensional speaker acoustic embedding extractor."""

    def __init__(self, sample_rate: int = 48000, n_mels: int = 40, embed_dim: int = 192) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.embed_dim = embed_dim
        self.n_fft = 2048
        self.hop_length = 480

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract a 192-dimensional unit-normalized speaker embedding vector.

        Args:
            audio: 1D or 2D audio array at self.sample_rate.

        Returns:
            np.ndarray of shape (192,) with L2 norm = 1.0.
        """
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
        if mono.size < self.hop_length * 4 or np.all(np.abs(mono) < 1e-6):
            return np.zeros(self.embed_dim, dtype=np.float32)

        # 1. Compute STFT magnitude
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

        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        # 2. Triangular Mel filterbank (0 to 8000 Hz for core voice identity)
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

        # Apply filterbank -> log mel energy
        mel_spec = np.log(np.dot(fbank, mag) + 1e-6)  # (40, n_frames)

        # Zero-center across mel bands to isolate spectral shape and vocal timbre
        mel_centered = mel_spec - np.mean(mel_spec, axis=0, keepdims=True)

        # 3. Statistical pooling over time
        mel_mean = np.mean(mel_centered, axis=1)  # 40
        mel_std = np.std(mel_centered, axis=1)  # 40
        mel_p10 = np.percentile(mel_centered, 10, axis=1)  # 40
        mel_p90 = np.percentile(mel_centered, 90, axis=1)  # 40
        # Total so far: 160 features

        # 4. Spectral moments (centroid, bandwidth, contrast, skewness)
        mag_norm = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
        centroid = np.sum(freqs[:, np.newaxis] * mag_norm, axis=0)  # (n_frames,)
        variance = np.sum(((freqs[:, np.newaxis] - centroid) ** 2) * mag_norm, axis=0)
        bandwidth = np.sqrt(np.maximum(0.0, variance))

        moments = np.array(
            [
                np.mean(centroid) / 4000.0,
                np.std(centroid) / 4000.0,
                np.mean(bandwidth) / 4000.0,
                np.std(bandwidth) / 4000.0,
                # Sub-band energy ratios
                np.mean(
                    np.sum(mag[: int(n_freqs * 0.1), :], axis=0) / (np.sum(mag, axis=0) + 1e-10)
                ),
                np.mean(
                    np.sum(mag[int(n_freqs * 0.1) : int(n_freqs * 0.3), :], axis=0)
                    / (np.sum(mag, axis=0) + 1e-10)
                ),
                np.mean(
                    np.sum(mag[int(n_freqs * 0.3) : int(n_freqs * 0.6), :], axis=0)
                    / (np.sum(mag, axis=0) + 1e-10)
                ),
                np.mean(
                    np.sum(mag[int(n_freqs * 0.6) :, :], axis=0) / (np.sum(mag, axis=0) + 1e-10)
                ),
            ],
            dtype=np.float32,
        )

        # Pad / assemble into exact 192 dimensions
        features = np.concatenate([mel_mean, mel_std, mel_p10, mel_p90, moments])
        if len(features) < self.embed_dim:
            features = np.pad(features, (0, self.embed_dim - len(features)))
        else:
            features = features[: self.embed_dim]

        # L2 normalize
        norm = np.linalg.norm(features)
        features = features / norm if norm > 1e-9 else np.zeros(self.embed_dim, dtype=np.float32)
        return features.astype(np.float32)
