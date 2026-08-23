"""Continuous spectral bandwidth and cutoff frequency detection."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BandwidthEvidence:
    """Spectral measurements supporting the cutoff decision."""

    spectral_rolloff: float
    above_cutoff_snr_db: float
    stationarity: float
    high_band_energy_ratio_db: float


@dataclass(frozen=True)
class BandwidthEstimate:
    """Estimated bandwidth profile of an audio signal."""

    effective_cutoff_hz: float
    confidence: float
    shape: str  # "codec_lowpass", "steep_brickwall", "gentle_rolloff", "fullband"
    restore_recommended: bool
    evidence: BandwidthEvidence

    def to_dict(self) -> dict[str, Any]:
        """Serialize estimate to canonical dictionary."""
        d = asdict(self)
        return d


class BandwidthDetector:
    """Deterministic spectral bandwidth and cutoff frequency detector.

    Analyzes complex spectral envelopes across active speech frames to detect
    low-pass filtering, telephony band-limiting (e.g. 3.4 kHz, 7.5 kHz),
    codec roll-offs (e.g. 12 kHz, 16 kHz), or full-band content.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        min_cutoff_hz: float = 2000.0,
        max_cutoff_hz: float = 22000.0,
        confidence_threshold: float = 0.80,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_cutoff_hz = min_cutoff_hz
        self.max_cutoff_hz = max_cutoff_hz
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        audio: np.ndarray,
        speech_mask: np.ndarray | None = None,
        override_cutoff_hz: float | None = None,
    ) -> BandwidthEstimate:
        """Estimate the effective high-frequency cutoff of the given 48 kHz audio."""
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio

        rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0
        if len(mono) < self.n_fft or rms < 1e-6:
            # Signal too short or pure silence
            return BandwidthEstimate(
                effective_cutoff_hz=self.max_cutoff_hz,
                confidence=1.0 if rms < 1e-6 else 0.5,
                shape="silence" if rms < 1e-6 else "fullband",
                restore_recommended=False,
                evidence=BandwidthEvidence(
                    spectral_rolloff=0.0,
                    above_cutoff_snr_db=0.0,
                    stationarity=1.0,
                    high_band_energy_ratio_db=0.0,
                ),
            )

        if override_cutoff_hz is not None:
            cutoff = float(np.clip(override_cutoff_hz, self.min_cutoff_hz, self.max_cutoff_hz))
            return BandwidthEstimate(
                effective_cutoff_hz=cutoff,
                confidence=1.0,
                shape="manual_override",
                restore_recommended=cutoff < (self.sample_rate / 2.0 - 1500.0),
                evidence=BandwidthEvidence(
                    spectral_rolloff=0.0,
                    above_cutoff_snr_db=0.0,
                    stationarity=1.0,
                    high_band_energy_ratio_db=0.0,
                ),
            )

        # Compute STFT
        _, _, Zxx = signal.stft(
            mono,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
            padded=False,
        )

        mag_sq = np.abs(Zxx) ** 2  # (n_freqs, n_frames)
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        # If speech mask is provided, filter frames
        if speech_mask is not None and len(speech_mask) == mag_sq.shape[1]:
            active_frames = mag_sq[:, speech_mask > 0.5]
            if active_frames.shape[1] > 5:
                mag_sq = active_frames

        # Average PSD across active frames
        mean_psd = np.mean(mag_sq, axis=1) + 1e-12
        psd_db = 10.0 * np.log10(mean_psd)

        # Baseline peak in core voice band (300 Hz to 3500 Hz)
        voice_band = (freqs >= 300.0) & (freqs <= 3500.0)
        peak_db = float(np.max(psd_db[voice_band])) if np.any(voice_band) else float(np.max(psd_db))
        ref_db = (
            float(10.0 * np.log10(np.mean(mean_psd[voice_band]) + 1e-12))
            if np.any(voice_band)
            else peak_db
        )

        thresh_db = peak_db - 35.0
        active_bins = np.where((freqs >= self.min_cutoff_hz) & (psd_db >= thresh_db))[0]

        detected_cutoff = float(self.max_cutoff_hz)
        detected_shape = "fullband"
        detected_conf = 0.95
        rolloff_rate = 0.0

        if len(active_bins) > 0:
            highest_active_freq = float(freqs[active_bins[-1]])
            if highest_active_freq < (self.sample_rate / 2.0 - 2500.0):
                detected_cutoff = float(
                    np.clip(highest_active_freq + 250.0, self.min_cutoff_hz, self.max_cutoff_hz)
                )
                detected_conf = 0.99
                # Calculate local slope around detected cutoff
                f_low = max(500.0, detected_cutoff - 500.0)
                f_high = min(self.sample_rate / 2.0, detected_cutoff + 500.0)
                idx_low = int(np.argmin(np.abs(freqs - f_low)))
                idx_high = int(np.argmin(np.abs(freqs - f_high)))
                log_ratio = np.log2(max(1.01, f_high / max(1.0, f_low)))
                slope = float((psd_db[idx_low] - psd_db[idx_high]) / max(0.01, log_ratio))
                rolloff_rate = slope
                if slope > 36.0:
                    detected_shape = "steep_brickwall"
                elif slope > 18.0:
                    detected_shape = "codec_lowpass"
                else:
                    detected_shape = "gentle_rolloff"
        else:
            detected_cutoff = float(self.min_cutoff_hz)
            detected_shape = "steep_brickwall"

        # Evidence statistics
        fullband_hf_mask = freqs >= 18000.0
        if np.any(fullband_hf_mask):
            fullband_hf_db = float(np.median(psd_db[fullband_hf_mask]))
        else:
            fullband_hf_db = float(np.min(psd_db)) if len(psd_db) > 0 else -80.0
        snr_above = float(np.clip(ref_db - fullband_hf_db, 0.0, 100.0))
        ratio_db = float(ref_db - fullband_hf_db)

        # Stationarity / variance of HF frames
        hf_band_frames = mag_sq[freqs >= detected_cutoff, :]
        if hf_band_frames.shape[0] > 0 and hf_band_frames.shape[1] > 1:
            frame_energies = np.sum(hf_band_frames, axis=0) + 1e-12
            stationarity = float(np.std(frame_energies) / (np.mean(frame_energies) + 1e-6))
        else:
            stationarity = 0.0

        restore_recommended = bool(detected_cutoff <= 16000.0)
        if restore_recommended:
            detected_conf = float(np.clip(0.70 + (ref_db - fullband_hf_db) / 100.0, 0.60, 0.99))
        else:
            detected_cutoff = float(self.max_cutoff_hz)
            detected_shape = "fullband"
            detected_conf = 0.95

        return BandwidthEstimate(
            effective_cutoff_hz=detected_cutoff,
            confidence=detected_conf,
            shape=detected_shape,
            restore_recommended=restore_recommended,
            evidence=BandwidthEvidence(
                spectral_rolloff=rolloff_rate,
                above_cutoff_snr_db=snr_above,
                stationarity=float(stationarity),
                high_band_energy_ratio_db=ratio_db,
            ),
        )
