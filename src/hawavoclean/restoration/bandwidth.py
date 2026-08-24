"""Continuous spectral bandwidth and cutoff frequency detection."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage, signal

#: Half-octave span used to measure the local slope at a candidate band edge.
_HALF_OCTAVE = 1.4142135623730951
#: Bins of log-spectrum smoothing, to suppress harmonic ripple before slope
#: estimation without blurring a genuine cliff.
_SMOOTH_BINS = 9
#: Minimum fall, in dB per octave, for a candidate to count as a filter cliff
#: rather than the natural tilt of a voice.
_CLIFF_DB_PER_OCTAVE = 35.0
#: How far below the voice band the region above the cliff must sit.
_FLOOR_MARGIN_DB = 45.0
#: Maximum spread of that region: a floor is flat, a roll-off is not.
_PLATEAU_FLAT_DB = 9.0
#: How far the loudest bin above a candidate edge may stand above its own
#: local median. A floor is featureless; a surviving tone is not.
_ABOVE_PEAKINESS_DB = 20.0
#: Content is "present" until it falls this far below the in-band peak.
_EDGE_DROP_DB = 75.0
#: The walk from cliff foot to band edge never exceeds half an octave.
_EDGE_WALK_LIMIT = 1.5


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

        # Evidence statistics, independent of how the cutoff is decided.
        fullband_hf_mask = freqs >= 18000.0
        if np.any(fullband_hf_mask):
            fullband_hf_db = float(np.median(psd_db[fullband_hf_mask]))
        else:
            fullband_hf_db = float(np.min(psd_db)) if len(psd_db) > 0 else -80.0
        snr_above = float(np.clip(ref_db - fullband_hf_db, 0.0, 100.0))
        ratio_db = float(ref_db - fullband_hf_db)

        # --- Band-limit detection --------------------------------------------
        #
        # A band limit is a CLIFF into a FLAT FLOOR, and that is what this
        # detects. The previous rule -- "the highest bin still within 35 dB of
        # the in-band peak" -- measured spectral TILT instead. Natural voiced
        # speech falls much further than 35 dB from its low-frequency peak to
        # the top of its range, so genuinely full-band speech read as
        # band-limited: measured across spectral tilts 1.0-3.4, four of five
        # unfiltered signals came back restore_recommended=True at 0.90-0.99
        # confidence, the reported cutoff tracking the tilt rather than any
        # real band edge. Restoration would then synthesise over recorded
        # content -- exactly the false restoration this subsystem promises not
        # to perform.
        #
        # Three conditions must hold together, because ordinary speech
        # satisfies any one of them on its own:
        #   1. a steep local slope in dB/octave -- the cliff;
        #   2. everything above it far below the voice band -- the floor;
        #   3. that region flat -- a floor, not a continuing roll-off.
        smoothed = ndimage.uniform_filter1d(psd_db, size=_SMOOTH_BINS)
        nyquist = self.sample_rate / 2.0
        qualifying: list[tuple[float, float, float, float]] = []
        for freq in freqs:
            freq_f = float(freq)
            if freq_f < self.min_cutoff_hz or freq_f <= 0.0:
                continue
            f_lo, f_hi = freq_f / _HALF_OCTAVE, freq_f * _HALF_OCTAVE
            if f_hi > nyquist:
                break
            i_lo = int(np.argmin(np.abs(freqs - f_lo)))
            i_hi = int(np.argmin(np.abs(freqs - f_hi)))
            octaves = float(np.log2(freqs[i_hi] / max(float(freqs[i_lo]), 1e-9)))
            if octaves <= 0.0:
                continue
            slope = float((smoothed[i_lo] - smoothed[i_hi]) / octaves)
            if slope < _CLIFF_DB_PER_OCTAVE:
                continue
            above = freqs >= min(freq_f * 1.15, nyquist * 0.99)
            if not np.any(above):
                continue
            drop = ref_db - float(np.median(smoothed[above]))
            flat = float(np.std(smoothed[above]))
            # Above a real band edge the spectrum is FEATURELESS floor. A
            # median test is not enough: a sparse signal with a genuine tone at
            # 18 kHz is mostly floor above 11 kHz, so the median reads as
            # band-limited and licenses the model to synthesise over that tone.
            # Measuring the peak against its own local median needs no absolute
            # reference and separates the two cleanly -- a surviving tone stood
            # 59 dB above its surroundings where a true edge showed under 5.
            peakiness = float(np.max(smoothed[above]) - np.median(smoothed[above]))
            if (
                drop >= _FLOOR_MARGIN_DB
                and flat <= _PLATEAU_FLAT_DB
                and peakiness <= _ABOVE_PEAKINESS_DB
            ):
                qualifying.append((freq_f, slope, drop, flat))

        if not qualifying:
            return BandwidthEstimate(
                effective_cutoff_hz=float(self.max_cutoff_hz),
                confidence=0.95,
                shape="fullband",
                restore_recommended=False,
                evidence=BandwidthEvidence(
                    spectral_rolloff=0.0,
                    above_cutoff_snr_db=snr_above,
                    stationarity=0.0,
                    high_band_energy_ratio_db=ratio_db,
                ),
            )

        # The band edge is the LOWEST qualifying frequency. Every bin above the
        # cliff also satisfies the test -- the floor stays flat and deep -- so
        # taking the highest, or the steepest, reports an edge far above the
        # real one.
        cliff_hz, rolloff_rate, drop_db, flatness = min(qualifying, key=lambda q: q[0])

        # That frequency is where the cliff BEGINS; real content continues into
        # the transition above it. Reporting the foot would end the protected
        # band below the genuine edge and let the model overwrite recorded
        # audio -- the one error direction that is never acceptable -- so walk
        # up to where content actually stops. Contiguously, and bounded to half
        # an octave: taking the last member of a sparse match set lets a single
        # isolated numerical spike drag the estimate up with it.
        edge_floor = peak_db - _EDGE_DROP_DB
        walk_limit = cliff_hz * _EDGE_WALK_LIMIT
        start_idx = int(np.searchsorted(freqs, cliff_hz))
        edge_idx = start_idx
        for j in range(start_idx, len(freqs)):
            if float(freqs[j]) > walk_limit or smoothed[j] < edge_floor:
                break
            edge_idx = j
        detected_cutoff = float(
            np.clip(float(freqs[edge_idx]), self.min_cutoff_hz, self.max_cutoff_hz)
        )
        detected_shape = "steep_brickwall" if rolloff_rate >= 60.0 else "codec_lowpass"
        stationarity = flatness

        # Above 16 kHz is not the telephony/codec case restoration exists for:
        # an edge that high is an anti-alias filter on a full-rate recording,
        # and synthesising over it would invent content nobody removed.
        #
        # Deliberately kept although no input is known to reach it. The cliff
        # test needs half an octave of floor above its candidate, so the search
        # cannot place an edge past nyquist/sqrt(2) -- about 17 kHz at 48 kHz --
        # and a sweep over four sample rates and three filter orders produced
        # no detection above 16 kHz at all. That is evidence, not a proof, and
        # the case it guards is the dangerous direction, so it stays. It
        # carries no mutation for the same reason: a mutation on a branch
        # nothing reaches can never be caught, and the gate would be lying.
        restore_recommended = bool(detected_cutoff <= 16000.0)
        if not restore_recommended:
            detected_cutoff = float(self.max_cutoff_hz)
            detected_shape = "fullband"
            detected_conf = 0.95
        else:
            # Confidence follows the evidence that made the call: a steeper
            # cliff into a deeper floor is a more certain band limit.
            detected_conf = float(
                np.clip(0.70 + (rolloff_rate / 400.0) + (drop_db / 400.0), 0.60, 0.99)
            )

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
