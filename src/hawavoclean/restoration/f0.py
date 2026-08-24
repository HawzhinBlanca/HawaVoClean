"""Deterministic F0 trajectory and Voiced/Unvoiced (V/UV) extraction."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class F0Statistics:
    """Pitch statistics for a speaker or utterance."""

    median_hz: float
    p05_hz: float
    p95_hz: float
    voiced_fraction: float


@dataclass(frozen=True)
class F0Trajectory:
    """Extracted pitch trajectory and voicing mask."""

    f0_hz: np.ndarray  # Shape: (n_frames,), float32
    vuv_mask: np.ndarray  # Shape: (n_frames,), float32 (1.0 = voiced, 0.0 = unvoiced)
    frame_times_s: np.ndarray  # Shape: (n_frames,), float32
    sample_rate: int
    hop_length: int
    statistics: F0Statistics


class F0Extractor:
    """Deterministic F0 and V/UV extractor based on normalized cross-correlation & harmonic salience."""

    def __init__(
        self,
        sample_rate: int = 48000,
        hop_length: int = 480,  # 10 ms at 48 kHz
        frame_length: int = 2048,
        f0_min_hz: float = 60.0,
        f0_max_hz: float = 600.0,
        voicing_threshold: float = 0.45,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.voicing_threshold = voicing_threshold

    def extract(self, audio: np.ndarray) -> F0Trajectory:
        """Extract deterministic F0 trajectory and voicing mask from 48 kHz audio."""
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio

        n_samples = len(mono)
        if n_samples < self.frame_length:
            return F0Trajectory(
                f0_hz=np.zeros(1, dtype=np.float32),
                vuv_mask=np.zeros(1, dtype=np.float32),
                frame_times_s=np.zeros(1, dtype=np.float32),
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                statistics=F0Statistics(median_hz=0.0, p05_hz=0.0, p95_hz=0.0, voiced_fraction=0.0),
            )

        n_frames = max(1, (n_samples - self.frame_length) // self.hop_length + 1)
        f0_hz = np.zeros(n_frames, dtype=np.float32)
        vuv_mask = np.zeros(n_frames, dtype=np.float32)
        frame_times = (
            np.arange(n_frames) * self.hop_length + self.frame_length / 2.0
        ) / self.sample_rate

        min_lag = int(self.sample_rate / self.f0_max_hz)
        max_lag = int(self.sample_rate / self.f0_min_hz)

        # Center-clipped autocorrelation / normalized cross-correlation
        window = np.hanning(self.frame_length).astype(np.float32)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = mono[start:end] * window

            energy = np.sum(frame**2)
            if energy < 1e-7:
                f0_hz[i] = 0.0
                vuv_mask[i] = 0.0
                continue

            # Normalized autocorrelation via FFT
            n_fft = 2 ** int(np.ceil(np.log2(2 * self.frame_length)))
            fft_frame = np.fft.rfft(frame, n=n_fft)
            corr = np.fft.irfft(np.abs(fft_frame) ** 2, n=n_fft)[: self.frame_length]
            corr = corr / (corr[0] + 1e-12)

            # Search in valid pitch lag range [min_lag, max_lag]
            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                continue

            peak_idx = int(np.argmax(search_region)) + min_lag
            peak_val = corr[peak_idx]

            # Parabolic interpolation for fine frequency resolution
            if min_lag < peak_idx < max_lag - 1:
                alpha = corr[peak_idx - 1]
                beta = corr[peak_idx]
                gamma = corr[peak_idx + 1]
                denom = 2.0 * (alpha - 2.0 * beta + gamma)
                if abs(denom) > 1e-12:
                    delta = (alpha - gamma) / denom
                    refined_lag = peak_idx + delta
                else:
                    refined_lag = float(peak_idx)
            else:
                refined_lag = float(peak_idx)

            est_f0 = self.sample_rate / max(1.0, refined_lag)

            if peak_val >= self.voicing_threshold and self.f0_min_hz <= est_f0 <= self.f0_max_hz:
                f0_hz[i] = float(est_f0)
                vuv_mask[i] = 1.0
            else:
                f0_hz[i] = 0.0
                vuv_mask[i] = 0.0

        # Calculate statistics across voiced frames
        voiced_f0 = f0_hz[vuv_mask > 0.5]
        if len(voiced_f0) > 0:
            median_hz = float(np.median(voiced_f0))
            p05_hz = float(np.percentile(voiced_f0, 5))
            p95_hz = float(np.percentile(voiced_f0, 95))
            voiced_frac = float(len(voiced_f0) / n_frames)
        else:
            median_hz = 0.0
            p05_hz = 0.0
            p95_hz = 0.0
            voiced_frac = 0.0

        return F0Trajectory(
            f0_hz=f0_hz,
            vuv_mask=vuv_mask,
            frame_times_s=frame_times.astype(np.float32),
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            statistics=F0Statistics(
                median_hz=median_hz,
                p05_hz=p05_hz,
                p95_hz=p95_hz,
                voiced_fraction=voiced_frac,
            ),
        )
