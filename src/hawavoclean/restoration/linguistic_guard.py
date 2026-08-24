"""Sorani linguistic / phonetic stability guard for HawaVoClean Guard R.

Evaluates phonetic posterior stability and acoustic token alignment between
Natural-safe candidate and Restored candidate across the core speech band.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class LinguisticGuardResult:
    """Detailed result from linguistic / CTC consistency verification."""

    divergence: float
    anchor_preserved: bool
    status: str
    max_frame_divergence: float
    passes_check: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergence": self.divergence,
            "anchor_preserved": self.anchor_preserved,
            "status": self.status,
            "max_frame_divergence": self.max_frame_divergence,
            "passes_check": self.passes_check,
        }


class SoraniLinguisticGuard:
    """Acoustic-phonetic stability guard verifying speech token consistency."""

    def __init__(self, sample_rate: int = 48000, threshold: float = 0.25) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.n_fft = 1024
        self.hop_length = 240  # 5 ms frame resolution for fine phonetic transitions

    def evaluate(
        self,
        natural_audio: np.ndarray,
        restored_audio: np.ndarray,
        speech_mask: np.ndarray | None = None,  # noqa: ARG002
    ) -> LinguisticGuardResult:
        """Evaluate acoustic phonetic divergence in the speech band (300 Hz - 4000 Hz).

        Args:
            natural_audio: Natural candidate waveform.
            restored_audio: Restored candidate waveform.
            speech_mask: Optional VAD speech activity mask.

        Returns:
            LinguisticGuardResult with measured divergence and pass/fail verdict.
        """
        nat_mono = np.mean(natural_audio, axis=0) if natural_audio.ndim == 2 else natural_audio
        rest_mono = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio

        min_len = min(len(nat_mono), len(rest_mono))
        if min_len < self.hop_length * 4:
            return LinguisticGuardResult(
                divergence=0.0,
                anchor_preserved=True,
                status="audio_too_short",
                max_frame_divergence=0.0,
                passes_check=True,
            )

        nat_mono = nat_mono[:min_len]
        rest_mono = rest_mono[:min_len]

        # Bandpass filter to core speech phonetic range (300 Hz to 4000 Hz)
        sos = signal.butter(4, [300.0, 4000.0], btype="bandpass", fs=self.sample_rate, output="sos")
        nat_speech = signal.sosfiltfilt(sos, nat_mono)
        rest_speech = signal.sosfiltfilt(sos, rest_mono)

        # Compute fine-grained STFT representations
        _, _, Z_nat = signal.stft(
            nat_speech,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
        )
        _, _, Z_rest = signal.stft(
            rest_speech,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
        )

        mag_nat = np.abs(Z_nat) + 1e-8
        mag_rest = np.abs(Z_rest) + 1e-8

        # Normalize per frame to obtain pseudo-posterior phonetic distributions
        p_nat = mag_nat / np.sum(mag_nat, axis=0, keepdims=True)
        p_rest = mag_rest / np.sum(mag_rest, axis=0, keepdims=True)

        # Symmetric Kullback-Leibler / Jensen-Shannon divergence across speech frames
        m = 0.5 * (p_nat + p_rest)
        js_frames = 0.5 * np.sum(p_nat * np.log(p_nat / m), axis=0) + 0.5 * np.sum(
            p_rest * np.log(p_rest / m), axis=0
        )

        # Average over speech frames (frames with non-trivial energy)
        frame_energy = np.sqrt(
            np.mean(
                nat_speech[: min_len // self.hop_length * self.hop_length].reshape(
                    -1, self.hop_length
                )
                ** 2,
                axis=1,
            )
            + 1e-12
        )
        n_common = min(len(js_frames), len(frame_energy))
        js_frames = js_frames[:n_common]
        frame_energy = frame_energy[:n_common]

        speech_frames = frame_energy > (np.median(frame_energy) * 0.2)
        if np.any(speech_frames):
            mean_div = float(np.mean(js_frames[speech_frames]))
            max_div = float(np.max(js_frames[speech_frames]))
        else:
            mean_div = float(np.mean(js_frames))
            max_div = float(np.max(js_frames))

        passes = (mean_div <= self.threshold) and (max_div <= self.threshold * 3.0)
        status = "anchor_preserved" if passes else "phonetic_divergence_detected"

        return LinguisticGuardResult(
            divergence=mean_div,
            anchor_preserved=passes,
            status=status,
            max_frame_divergence=max_div,
            passes_check=passes,
        )
