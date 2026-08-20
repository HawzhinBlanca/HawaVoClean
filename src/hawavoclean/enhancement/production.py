"""Production enhancement core: decision-directed spectral Wiener filtering.

This is classical DSP, not a neural network. It attenuates steady-state
noise by per-bin Wiener gains driven by decision-directed a priori SNR
tracking (Ephraim-Malah / McAulay-Malpass), preserving the original phase
exactly. It cannot synthesize, substitute, or hallucinate content — it can
only attenuate spectral magnitude, bounded below by a gain floor.

Device note: this core is numpy and scipy on the CPU. ``runtime.device`` does
not apply to it, and the registry records that (``device_aware=False``) so a
production run's report says ``compute_device = "cpu"`` — the device that ran
— even when the configuration asked for a GPU.
"""

import time
from typing import Any

import numpy as np

from hawavoclean.audio.resample import resample_audio
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer, EnhancerMetadata
from hawavoclean.hashing import hash_json_canonical
from hawavoclean.runtime import check_memory_budget

WIENER_PARAMS: dict[str, float | int] = {
    "n_fft": 2048,
    "hop": 512,
    "noise_percentile": 15,
    "dd_alpha": 0.95,
    "gain_floor": 0.05,
    "internal_sample_rate": 48000,
}


def wiener_params_hash() -> str:
    """Canonical hash of the algorithm parameters — the core's real provenance."""
    return hash_json_canonical(WIENER_PARAMS)


class WienerSpectralEnhancer(Enhancer):
    """Frozen decision-directed Wiener spectral-subtraction core."""

    def __init__(
        self,
        core_id: str = "wiener-dd-48k-v1",
        sample_rate: int = 48000,
        phase_coherent: bool = True,
    ) -> None:
        self._core_id = core_id
        self._sample_rate = sample_rate
        self._phase_coherent = phase_coherent
        self._metadata = EnhancerMetadata(
            core_id=core_id,
            version="1.0.0",
            algorithm="decision-directed spectral Wiener filter (Ephraim-Malah SNR tracking)",
            sample_rate=sample_rate,
            phase_coherent=phase_coherent,
            params_hash=wiener_params_hash(),
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    def warmup(self) -> None:
        """Warmup buffers and pre-allocate memory."""
        dummy = np.zeros(self._sample_rate, dtype=np.float32)
        self.enhance(dummy, self._sample_rate)

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        """Run Wiener spectral filtering on an audio waveform."""
        # Before taking the unit, not after: a worker that has already blown
        # runtime.worker_memory_limit_mb stops accepting work rather than
        # discarding a result it has already paid for.
        check_memory_budget("production")
        t_start = time.perf_counter()
        orig_len = len(waveform)

        if orig_len == 0:
            return EnhancementResult(
                waveform=waveform.copy(),
                sample_rate=sample_rate,
                model_runtime_ms=0.0,
                input_samples=0,
                output_samples=0,
            )

        # 1. Resample to the core's internal rate (48kHz)
        audio_48k = resample_audio(waveform, sample_rate, self._sample_rate)

        # 2. Decision-directed spectral Wiener filtering with exact phase preservation
        n_fft = int(WIENER_PARAMS["n_fft"])
        hop = int(WIENER_PARAMS["hop"])
        win = np.hanning(n_fft)

        num_frames = max(1, int(np.ceil((len(audio_48k) - n_fft) / hop)) + 1)
        padded_len = (num_frames - 1) * hop + n_fft
        pad_amount = max(0, padded_len - len(audio_48k))
        x_pad = np.pad(audio_48k, (0, pad_amount)) if pad_amount > 0 else audio_48k

        stft = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.complex64)
        for i in range(num_frames):
            chunk = x_pad[i * hop : i * hop + n_fft] * win
            stft[i] = np.fft.rfft(chunk, n=n_fft)

        mag = np.abs(stft)
        phase = np.angle(stft)
        power = mag**2

        # Noise power estimate: low percentile across time frames per bin
        noise_power = np.percentile(power, int(WIENER_PARAMS["noise_percentile"]), axis=0) + 1e-10

        # Decision-directed a priori SNR tracking (Ephraim-Malah / McAulay-Malpass)
        alpha = float(WIENER_PARAMS["dd_alpha"])
        gain_floor = float(WIENER_PARAMS["gain_floor"])
        gain = np.zeros_like(power)
        prev_enh_power = power[0].copy()

        for i in range(num_frames):
            post_snr = power[i] / noise_power
            if i == 0:
                prior_snr = np.maximum(post_snr - 1.0, 0.01)
            else:
                prior_snr = alpha * (prev_enh_power / noise_power) + (1.0 - alpha) * np.maximum(
                    post_snr - 1.0, 0.0
                )

            # Wiener gain with musical-noise suppression floor
            g = prior_snr / (prior_snr + 1.0)
            g = np.clip(g, gain_floor, 1.0)
            gain[i] = g
            prev_enh_power = (mag[i] * g) ** 2

        # Reconstruct complex STFT preserving the exact original phase
        enhanced_stft = (mag * gain) * np.exp(1j * phase)

        # Overlap-add ISTFT
        enhanced_48k = np.zeros(len(x_pad), dtype=np.float32)
        win_sum = np.zeros(len(x_pad), dtype=np.float32)

        for i in range(num_frames):
            time_chunk = np.fft.irfft(enhanced_stft[i], n=n_fft) * win
            enhanced_48k[i * hop : i * hop + n_fft] += time_chunk
            win_sum[i * hop : i * hop + n_fft] += win**2

        # Normalize window overlap
        valid_idx = win_sum > 1e-4
        enhanced_48k[valid_idx] /= win_sum[valid_idx]
        enhanced_48k = enhanced_48k[: len(audio_48k)]

        # 3. Resample back to the original rate and match exact length
        enhanced_out = resample_audio(
            enhanced_48k, self._sample_rate, sample_rate, target_samples=orig_len
        )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return EnhancementResult(
            waveform=enhanced_out,
            sample_rate=sample_rate,
            model_runtime_ms=t_elapsed_ms,
            input_samples=orig_len,
            output_samples=len(enhanced_out),
            warnings=[],
        )


class NoOpEnhancer(Enhancer):
    """No-op passthrough enhancer for baseline comparisons and unit testing."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self._metadata = EnhancerMetadata(
            core_id="noop-enhancer",
            version="1.0.0",
            algorithm="identity passthrough",
            sample_rate=sample_rate,
            phase_coherent=True,
            params_hash=hash_json_canonical({"algorithm": "noop"}),
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._metadata

    def warmup(self) -> None:
        pass

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        return EnhancementResult(
            waveform=waveform.copy(),
            sample_rate=sample_rate,
            model_runtime_ms=0.5,
            input_samples=len(waveform),
            output_samples=len(waveform),
        )
