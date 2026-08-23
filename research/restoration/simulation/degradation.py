"""Acoustic degradation simulation for bandwidth restoration training and benchmarking."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class DegradationParams:
    """Parameters applied during degradation simulation."""

    cutoff_hz: float
    filter_order: int
    filter_type: str  # "butterworth", "chebyshev", "brickwall", "codec_shape"
    snr_db: float
    packet_loss_rate: float
    clipping_threshold: float


class DegradationSimulator:
    """Simulates realistic bandwidth limitation and channel degradations."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate

    def apply_lowpass(
        self,
        audio: np.ndarray,
        cutoff_hz: float,
        filter_order: int = 8,
        filter_type: str = "butterworth",
    ) -> np.ndarray:
        """Apply low-pass filter with specified cutoff and slope."""
        nyq = self.sample_rate / 2.0
        norm_cutoff = min(0.95, max(0.05, cutoff_hz / nyq))

        if filter_type == "butterworth":
            sos = signal.butter(filter_order, norm_cutoff, btype="lowpass", output="sos")
        elif filter_type == "chebyshev":
            sos = signal.cheby1(filter_order, 1.0, norm_cutoff, btype="lowpass", output="sos")
        elif filter_type == "codec_shape":
            # Codec-like steep transition with high-frequency attenuation shelf
            sos = signal.ellip(6, 1.0, 40.0, norm_cutoff, btype="lowpass", output="sos")
        else:
            sos = signal.butter(filter_order, norm_cutoff, btype="lowpass", output="sos")

        if audio.ndim == 2:
            filt = np.stack(
                [signal.sosfiltfilt(sos, audio[ch]) for ch in range(audio.shape[0])], axis=0
            )
            return np.asarray(filt, dtype=np.float32)
        return np.asarray(signal.sosfiltfilt(sos, audio), dtype=np.float32)

    def apply_resampling_chain(
        self,
        audio: np.ndarray,
        intermediate_rate: int,
    ) -> np.ndarray:
        """Simulate downsampling to intermediate rate (e.g. 8k, 16k, 24k) and upsampling back to 48k."""
        # Rational resampling
        gcd = np.gcd(self.sample_rate, intermediate_rate)
        down = self.sample_rate // gcd
        up = intermediate_rate // gcd

        downsampled = signal.resample_poly(audio, up, down, axis=-1)
        upsampled = signal.resample_poly(downsampled, down, up, axis=-1)

        # Match exact input length
        if upsampled.shape[-1] < audio.shape[-1]:
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, audio.shape[-1] - upsampled.shape[-1])]
            upsampled = np.pad(upsampled, pad_width)
        elif upsampled.shape[-1] > audio.shape[-1]:
            upsampled = upsampled[..., : audio.shape[-1]]

        return np.asarray(upsampled, dtype=np.float32)

    def degrade(
        self,
        clean_audio: np.ndarray,
        cutoff_hz: float,
        filter_type: str = "butterworth",
        filter_order: int = 8,
        add_noise_snr_db: float | None = None,
        packet_loss_rate: float = 0.0,
        clip_level: float = 1.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, DegradationParams]:
        """Apply comprehensive degradation to clean reference audio."""
        rng = np.random.default_rng(seed)

        # 1. Bandwidth limitation
        degraded = self.apply_lowpass(
            clean_audio,
            cutoff_hz=cutoff_hz,
            filter_order=filter_order,
            filter_type=filter_type,
        )

        # 2. Add light additive background noise if specified
        if add_noise_snr_db is not None and add_noise_snr_db < 60.0:
            sig_power = np.mean(degraded**2) + 1e-12
            noise_power = sig_power / (10.0 ** (add_noise_snr_db / 10.0))
            noise = rng.normal(0, np.sqrt(noise_power), degraded.shape).astype(np.float32)
            degraded = degraded + noise

        # 3. Packet loss / burst dropouts
        if packet_loss_rate > 0.0:
            block_size = int(self.sample_rate * 0.02)  # 20 ms frames
            n_blocks = degraded.shape[-1] // block_size
            for b in range(n_blocks):
                if rng.uniform(0, 1) < packet_loss_rate:
                    degraded[..., b * block_size : (b + 1) * block_size] *= 0.05

        # 4. Clipping
        if clip_level < 1.0:
            degraded = np.clip(degraded, -clip_level, clip_level)

        params = DegradationParams(
            cutoff_hz=cutoff_hz,
            filter_order=filter_order,
            filter_type=filter_type,
            snr_db=add_noise_snr_db if add_noise_snr_db is not None else 60.0,
            packet_loss_rate=packet_loss_rate,
            clipping_threshold=clip_level,
        )
        return degraded.astype(np.float32), params
