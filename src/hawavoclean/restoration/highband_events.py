"""High-frequency consonant and acoustic event consistency detection."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class HighBandEventResult:
    """Evaluation metrics for high-frequency acoustic event consistency."""

    speech_window_leakage: float  # High-band energy ratio in non-speech vs speech
    spurious_burst_count: int  # Spurious transient bursts outside speech
    hf_envelope_divergence: float  # Divergence between 3-8 kHz envelope and >8 kHz envelope
    boundary_discontinuity_score: float  # Boundary jump / click metric
    passes_event_check: bool


class HighBandEventDetector:
    """Validates that generated high-frequency energy adheres to speech timing and phonetics."""

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_ms: float = 10.0,
        leakage_threshold: float = 0.25,
        envelope_threshold: float = 0.35,
        boundary_threshold: float = 0.08,
        max_spurious_bursts: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_ms / 1000.0))
        self.leakage_threshold = leakage_threshold
        self.envelope_threshold = envelope_threshold
        self.boundary_threshold = boundary_threshold
        self.max_spurious_bursts = max_spurious_bursts

    def evaluate(
        self,
        natural_audio: np.ndarray,
        restored_audio: np.ndarray,
        speech_mask: np.ndarray | None = None,
        cutoff_hz: float = 8000.0,
    ) -> HighBandEventResult:
        """Evaluate high-frequency consonant and event consistency between Natural and Restored candidates."""
        nat_mono = np.mean(natural_audio, axis=0) if natural_audio.ndim == 2 else natural_audio
        rest_mono = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio

        min_len = min(len(nat_mono), len(rest_mono))
        nat_mono = nat_mono[:min_len]
        rest_mono = rest_mono[:min_len]

        if min_len < self.frame_length * 4:
            return HighBandEventResult(
                speech_window_leakage=0.0,
                spurious_burst_count=0,
                hf_envelope_divergence=0.0,
                boundary_discontinuity_score=0.0,
                passes_event_check=True,
            )

        # High-pass filter above cutoff to isolate generated high band
        hp_freq = max(1000.0, min(self.sample_rate / 2.0 - 500.0, cutoff_hz))
        sos_hp = signal.butter(4, hp_freq, btype="highpass", fs=self.sample_rate, output="sos")
        nat_hf = signal.sosfiltfilt(sos_hp, nat_mono)
        rest_hf = signal.sosfiltfilt(sos_hp, rest_mono)

        # Measure the DELTA highband (energy added by restoration, not absolute)
        # This prevents false rejections when the Natural pipeline itself already
        # has residual energy above the cutoff from enhancement processing.
        delta_hf = rest_hf - nat_hf

        # Reference band: use the speech-dominant range (300 Hz to cutoff or 8 kHz, whichever is larger)
        # This gives a robust energy reference even at very low cutoffs (e.g. 3.3 kHz)
        f_ref_low = 300.0
        f_ref_high = max(
            f_ref_low + 500.0, min(self.sample_rate / 2.0 - 200.0, max(cutoff_hz, 8000.0))
        )
        sos_ref = signal.butter(
            4, [f_ref_low, f_ref_high], btype="bandpass", fs=self.sample_rate, output="sos"
        )
        nat_ref = signal.sosfiltfilt(sos_ref, nat_mono)

        # Compute frame energies
        n_frames = min_len // self.frame_length
        delta_hf_frames = delta_hf[: n_frames * self.frame_length].reshape(
            (n_frames, self.frame_length)
        )
        nat_ref_frames = nat_ref[: n_frames * self.frame_length].reshape(
            (n_frames, self.frame_length)
        )

        hf_frame_rms = np.sqrt(np.mean(delta_hf_frames**2, axis=1) + 1e-12)
        mid_frame_rms = np.sqrt(np.mean(nat_ref_frames**2, axis=1) + 1e-12)

        # Determine speech frames if speech mask is not given
        if speech_mask is None or len(speech_mask) != n_frames:
            # Use the observed (Natural) reference band energy for speech detection
            # At low cutoffs, use a higher threshold to be more conservative
            percentile = 50 if cutoff_hz < 6000.0 else 35
            speech_thresh = np.percentile(mid_frame_rms, percentile)
            is_speech = mid_frame_rms > max(1e-4, speech_thresh)
        else:
            is_speech = speech_mask > 0.5

        # 1. Non-speech leakage: check if high band is active during non-speech pauses
        has_speech = bool(np.any(is_speech))
        has_non_speech = bool(np.any(~is_speech))

        speech_hf_energy = float(np.mean(hf_frame_rms[is_speech])) if has_speech else 1e-4
        non_speech_hf_energy = float(np.mean(hf_frame_rms[~is_speech])) if has_non_speech else 0.0

        # Absolute energy floor: if the generated delta is effectively inaudible
        # (below -54 dBFS ≈ 0.002 RMS), suppress the leakage ratio.
        # This prevents false rejections on microscopic numerical residue.
        abs_floor = 0.002
        if speech_hf_energy < abs_floor and non_speech_hf_energy < abs_floor:
            leakage = 0.0
        else:
            leakage = float(non_speech_hf_energy / (speech_hf_energy + 1e-8))

        # 2. Spurious bursts outside speech
        if has_speech and has_non_speech and speech_hf_energy >= abs_floor:
            burst_threshold = speech_hf_energy * 0.8
            spurious_bursts = int(np.sum((~is_speech) & (hf_frame_rms > burst_threshold)))
        else:
            spurious_bursts = 0

        # 3. Envelope correlation / divergence in active speech
        # Only compute if the delta HF energy is above the audibility floor,
        # otherwise correlation of near-zero signals is dominated by noise.
        if has_speech and np.sum(is_speech) > 2 and speech_hf_energy >= abs_floor:
            mid_speech_rms = mid_frame_rms[is_speech]
            hf_speech_rms = hf_frame_rms[is_speech]
            if np.std(mid_speech_rms) > 1e-7 and np.std(hf_speech_rms) > 1e-7:
                corr = np.corrcoef(mid_speech_rms, hf_speech_rms)[0, 1]
                envelope_div = float(np.clip(1.0 - corr, 0.0, 2.0)) if not np.isnan(corr) else 0.0
            else:
                envelope_div = 0.0
        else:
            envelope_div = 0.0

        # 4. Boundary discontinuity check (energy jump at start/end of audio)
        start_jump = float(np.abs(rest_mono[0] - nat_mono[0]))
        end_jump = float(np.abs(rest_mono[-1] - nat_mono[-1]))
        boundary_disc = max(start_jump, end_jump)

        passes = (
            (leakage <= self.leakage_threshold)
            and (spurious_bursts <= self.max_spurious_bursts)
            and (envelope_div <= self.envelope_threshold)
            and (boundary_disc <= self.boundary_threshold)
        )

        return HighBandEventResult(
            speech_window_leakage=leakage,
            spurious_burst_count=spurious_bursts,
            hf_envelope_divergence=envelope_div,
            boundary_discontinuity_score=boundary_disc,
            passes_event_check=passes,
        )
