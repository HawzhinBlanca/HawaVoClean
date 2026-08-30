"""Bounded-memory acoustic evidence for the experimental Smart Safe workflow.

This module is intentionally *not* a production classifier.  It provides a
deterministic, decode-once-compatible analyzer while the governed Sorani data,
listener-trained models, calibration, and independent qualification required
by the product plan do not yet exist.  Ambiguous evidence is widened in the
dangerous direction: intervention-enabling evidence receives a lower bound,
while content-risk evidence receives an upper bound.

The analyzer consumes fixed, non-overlapping frames from an iterable of
``AudioBuffer`` chunks.  Only one partial frame, running spectral summaries,
and scalar moments are retained.  Its state therefore does not grow with the
recording duration, and its result does not depend on decoder chunk boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.smart_safe.decision import AcousticEvidence

_Float64Array = NDArray[np.float64]
_Float32Array = NDArray[np.float32]
ConservativeDirection = Literal["lower", "upper"]
BandwidthShape = Literal["unknown", "silence", "fullband", "steep_lowpass"]

_QUALIFICATION: Final = "experimental_unqualified"
_MIN_SAMPLE_RATE: Final = 8_000
_MAX_SAMPLE_RATE: Final = 192_000


def _clip_probability(value: float) -> float:
    """Return a finite value in [0, 1]."""

    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _sigmoid(value: float) -> float:
    """Overflow-safe logistic function."""

    if value >= 0.0:
        exp = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exp)
    exp = math.exp(max(value, -60.0))
    return exp / (1.0 + exp)


@dataclass(frozen=True, slots=True)
class ProbabilityEstimate:
    """A proxy estimate plus the fail-closed value exposed to decisions.

    ``direction`` identifies which side is dangerous.  Positive eligibility
    evidence (speech, rumble, band limiting, coherence) uses a lower bound;
    content/artifact risks use an upper bound.
    """

    value: float
    confidence: float
    conservative: float
    direction: ConservativeDirection
    rationale: str

    def __post_init__(self) -> None:
        for field_name in ("value", "confidence", "conservative"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and between 0 and 1")
            object.__setattr__(self, field_name, value)
        if self.direction not in {"lower", "upper"}:
            raise ValueError("direction must be lower or upper")
        if not self.rationale:
            raise ValueError("rationale must not be empty")


def _estimate(
    value: float,
    confidence: float,
    direction: ConservativeDirection,
    rationale: str,
    *,
    margin_scale: float = 0.75,
) -> ProbabilityEstimate:
    value = _clip_probability(value)
    confidence = _clip_probability(confidence)
    margin = (1.0 - confidence) * margin_scale
    conservative = max(0.0, value - margin) if direction == "lower" else min(1.0, value + margin)
    return ProbabilityEstimate(value, confidence, conservative, direction, rationale)


@dataclass(frozen=True, slots=True)
class StreamingAcousticReport:
    """Conservative file-level acoustic evidence.

    The probability-like fields are deterministic engineering proxies, not
    calibrated probabilities.  ``qualification`` remains explicitly
    unqualified until locked Sorani validation and listening gates exist.
    """

    qualification: Literal["experimental_unqualified"]
    valid: bool
    sample_rate: int | None
    channels: int | None
    samples: int
    analyzed_frames: int
    duration_s: float
    speech_dominance: ProbabilityEstimate
    music_risk: ProbabilityEstimate
    crosstalk_risk: ProbabilityEstimate
    band_limited_confidence: ProbabilityEstimate
    estimated_cutoff_hz: float | None
    cutoff_confidence: float
    bandwidth_shape: BandwidthShape
    noise_risk: ProbabilityEstimate
    hum_confidence: ProbabilityEstimate
    reverberation_risk: ProbabilityEstimate
    clipping_risk: ProbabilityEstimate
    clipping_fraction: float
    codec_damage_risk: ProbabilityEstimate
    channel_coherence: ProbabilityEstimate
    rumble_confidence: ProbabilityEstimate
    recorded_high_frequency_speech_confidence: ProbabilityEstimate
    uncertainty: float
    uncertainty_reasons: tuple[str, ...]
    state_bound_bytes: int

    def __post_init__(self) -> None:
        if self.qualification != _QUALIFICATION:
            raise ValueError("the streaming analyzer is not production-qualified")
        for value, name in (
            (self.cutoff_confidence, "cutoff_confidence"),
            (self.uncertainty, "uncertainty"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if not math.isfinite(self.clipping_fraction) or not 0.0 <= self.clipping_fraction <= 1.0:
            raise ValueError("clipping_fraction must be finite and between 0 and 1")
        if self.estimated_cutoff_hz is not None and (
            not math.isfinite(self.estimated_cutoff_hz) or self.estimated_cutoff_hz <= 0.0
        ):
            raise ValueError("estimated_cutoff_hz must be a positive finite frequency")
        if self.samples < 0 or self.analyzed_frames < 0 or self.duration_s < 0.0:
            raise ValueError("sample and duration counters must be non-negative")

    def decision_evidence(
        self,
        *,
        reconstruction_consent: bool,
        speaker_match_confidence: float = 0.0,
        speaker_match_verified: bool = False,
    ) -> AcousticEvidence:
        """Map the conservative bounds to the Smart Safe decision contract.

        Invalid reports always map to the safest possible evidence, irrespective
        of caller-supplied consent or enrollment state.
        """

        if not self.valid:
            return AcousticEvidence(
                speech_dominance=0.0,
                music_risk=1.0,
                crosstalk_risk=1.0,
                rumble_confidence=0.0,
                band_limited_confidence=0.0,
                recorded_high_frequency_speech_confidence=1.0,
                reconstruction_consent=False,
            )
        return AcousticEvidence(
            speech_dominance=self.speech_dominance.conservative,
            music_risk=self.music_risk.conservative,
            crosstalk_risk=self.crosstalk_risk.conservative,
            rumble_confidence=self.rumble_confidence.conservative,
            band_limited_confidence=self.band_limited_confidence.conservative,
            recorded_high_frequency_speech_confidence=(
                self.recorded_high_frequency_speech_confidence.conservative
            ),
            speaker_match_confidence=speaker_match_confidence,
            speaker_match_verified=speaker_match_verified,
            reconstruction_consent=reconstruction_consent,
        )


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    """Fixed-memory analyzer configuration."""

    frame_samples: int = 2048
    min_analysis_s: float = 8.0
    energy_bins: int = 96

    def __post_init__(self) -> None:
        if self.frame_samples < 256 or self.frame_samples > 8192:
            raise ValueError("frame_samples must be between 256 and 8192")
        if self.frame_samples & (self.frame_samples - 1):
            raise ValueError("frame_samples must be a power of two")
        if not math.isfinite(self.min_analysis_s) or self.min_analysis_s <= 0.0:
            raise ValueError("min_analysis_s must be positive and finite")
        if self.energy_bins < 32 or self.energy_bins > 512:
            raise ValueError("energy_bins must be between 32 and 512")


DEFAULT_ANALYZER_CONFIG: Final = AnalyzerConfig()


class StreamingAcousticAnalyzer:
    """Incremental, fixed-state acoustic analyzer.

    A caller may feed any chunking of the same mono/stereo stream.  Sample
    rate, channel count, and channel mode must remain stable.  Malformed audio
    marks the result invalid; it can never create positive route eligibility.
    """

    def __init__(self, config: AnalyzerConfig = DEFAULT_ANALYZER_CONFIG) -> None:
        self.config = config
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._channel_mode: ChannelMode | None = None
        self._pending: _Float32Array | None = None
        self._pending_count = 0
        self._window = np.hanning(config.frame_samples).astype(np.float64)
        self._freqs: _Float64Array | None = None
        self._psd_sum: _Float64Array | None = None
        self._speech_psd_sum: _Float64Array | None = None
        self._previous_psd: _Float64Array | None = None
        self._energy_hist = np.zeros(config.energy_bins, dtype=np.int64)
        self._invalid_reasons: set[str] = set()
        self._finished = False

        self._samples = 0
        self._frames = 0
        self._speech_sum = 0.0
        self._speech_weight = 0.0
        self._music_weighted_sum = 0.0
        self._noise_weighted_sum = 0.0
        self._codec_weighted_sum = 0.0
        self._spectral_flux_sum = 0.0
        self._spectral_flux_count = 0
        self._decay_sum = 0.0
        self._decay_opportunities = 0
        self._previous_rms_db: float | None = None
        self._clipped_samples = 0

        self._corr_count = 0
        self._sum_left = 0.0
        self._sum_right = 0.0
        self._sum_left_sq = 0.0
        self._sum_right_sq = 0.0
        self._sum_cross = 0.0
        self._hum_frequencies: _Float64Array | None = None
        self._hum_real: _Float64Array | None = None
        self._hum_imag: _Float64Array | None = None
        self._hum_sample_count = 0
        self._mono_square_sum = 0.0
        self._max_state_bytes = self.state_nbytes

    @property
    def valid(self) -> bool:
        return not self._invalid_reasons

    @property
    def state_nbytes(self) -> int:
        """Bytes owned by duration-independent NumPy state arrays."""

        arrays = (
            self._pending,
            self._window,
            self._freqs,
            self._psd_sum,
            self._speech_psd_sum,
            self._previous_psd,
            self._energy_hist,
            self._hum_frequencies,
            self._hum_real,
            self._hum_imag,
        )
        return sum(array.nbytes for array in arrays if array is not None)

    def accept(self, chunk: AudioBuffer) -> None:
        """Consume one decoded chunk without retaining it."""

        if self._finished:
            raise RuntimeError("cannot accept audio after finish")
        if not isinstance(chunk, AudioBuffer):
            self._invalid_reasons.add("stream yielded an object that is not an AudioBuffer")
            return
        if chunk.sample_rate < _MIN_SAMPLE_RATE or chunk.sample_rate > _MAX_SAMPLE_RATE:
            self._invalid_reasons.add("sample rate is outside the supported analysis range")
            return
        if chunk.channels not in {1, 2}:
            self._invalid_reasons.add("only mono and stereo analysis are supported")
            return

        if self._sample_rate is None:
            self._initialize_stream(chunk)
        elif (
            chunk.sample_rate != self._sample_rate
            or chunk.channels != self._channels
            or chunk.channel_mode != self._channel_mode
        ):
            self._invalid_reasons.add("sample rate, channels, or channel mode changed mid-stream")
            return

        if not self.valid or chunk.samples == 0:
            return
        assert self._pending is not None

        source_offset = 0
        while source_offset < chunk.samples:
            take = min(
                self.config.frame_samples - self._pending_count, chunk.samples - source_offset
            )
            source = chunk.data[:, source_offset : source_offset + take]
            if not bool(np.all(np.isfinite(source))):
                self._invalid_reasons.add("audio contains NaN or infinite samples")
                return
            self._pending[:, self._pending_count : self._pending_count + take] = source
            self._pending_count += take
            self._samples += take
            source_offset += take
            if self._pending_count == self.config.frame_samples:
                self._process_frame(self._pending, self.config.frame_samples)
                self._pending_count = 0
        self._max_state_bytes = max(self._max_state_bytes, self.state_nbytes)

    def _initialize_stream(self, chunk: AudioBuffer) -> None:
        self._sample_rate = chunk.sample_rate
        self._channels = chunk.channels
        self._channel_mode = chunk.channel_mode
        self._pending = np.zeros(
            (chunk.channels, self.config.frame_samples),
            dtype=np.float32,
        )
        self._freqs = np.fft.rfftfreq(
            self.config.frame_samples,
            d=1.0 / chunk.sample_rate,
        )
        bins = self.config.frame_samples // 2 + 1
        self._psd_sum = np.zeros(bins, dtype=np.float64)
        self._speech_psd_sum = np.zeros(bins, dtype=np.float64)
        self._hum_frequencies = np.asarray(
            [50.0 * harmonic for harmonic in range(1, 6)]
            + [60.0 * harmonic for harmonic in range(1, 6)],
            dtype=np.float64,
        )
        self._hum_real = np.zeros(self._hum_frequencies.size, dtype=np.float64)
        self._hum_imag = np.zeros(self._hum_frequencies.size, dtype=np.float64)
        self._max_state_bytes = max(self._max_state_bytes, self.state_nbytes)

    def _process_frame(self, frame: _Float32Array, valid_samples: int) -> None:
        assert self._sample_rate is not None
        assert self._freqs is not None
        assert self._psd_sum is not None
        assert self._speech_psd_sum is not None
        assert self._hum_frequencies is not None
        assert self._hum_real is not None
        assert self._hum_imag is not None
        mono = np.mean(frame[:, :valid_samples], axis=0, dtype=np.float64)
        indices = np.arange(
            self._hum_sample_count,
            self._hum_sample_count + valid_samples,
            dtype=np.float64,
        )
        for index, frequency in enumerate(self._hum_frequencies):
            phase = 2.0 * np.pi * float(frequency) * indices / self._sample_rate
            self._hum_real[index] += float(np.dot(mono, np.cos(phase)))
            self._hum_imag[index] -= float(np.dot(mono, np.sin(phase)))
        self._hum_sample_count += valid_samples
        self._mono_square_sum += float(np.dot(mono, mono))
        mono -= float(np.mean(mono))
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-24))
        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        hist_index = int(
            np.clip(
                math.floor((rms_db + 120.0) * self.config.energy_bins / 126.0),
                0,
                self.config.energy_bins - 1,
            )
        )
        self._energy_hist[hist_index] += 1

        padded = np.zeros(self.config.frame_samples, dtype=np.float64)
        padded[:valid_samples] = mono
        spectrum = np.fft.rfft(padded * self._window)
        power = np.square(np.abs(spectrum)) + 1e-24
        self._psd_sum += power

        total_mask = (self._freqs >= 50.0) & (self._freqs <= min(10_000.0, self._sample_rate / 2.0))
        voice_mask = (self._freqs >= 250.0) & (self._freqs <= 4_000.0)
        total_power = float(np.sum(power[total_mask])) + 1e-24
        voice_ratio = float(np.sum(power[voice_mask])) / total_power
        active_power = power[total_mask]
        flatness = float(
            math.exp(float(np.mean(np.log(active_power))))
            / max(float(np.mean(active_power)), 1e-24)
        )
        centroid = float(np.sum(self._freqs[total_mask] * active_power) / total_power)
        zcr = (
            float(np.mean(np.signbit(mono[1:]) != np.signbit(mono[:-1])))
            if valid_samples > 1
            else 0.0
        )

        energy_score = _clip_probability((rms_db + 62.0) / 22.0)
        voice_score = _clip_probability((voice_ratio - 0.20) / 0.65)
        centroid_score = _clip_probability(1.0 - abs(centroid - 1_700.0) / 2_800.0)
        zcr_score = _clip_probability(1.0 - abs(zcr - 0.10) / 0.22)
        flatness_score = _clip_probability(1.0 - abs(flatness - 0.16) / 0.38)
        speech_score = energy_score * (
            0.45 * voice_score + 0.20 * centroid_score + 0.15 * zcr_score + 0.20 * flatness_score
        )
        speech_score = _clip_probability(speech_score)
        self._speech_sum += speech_score
        self._speech_weight += speech_score
        self._speech_psd_sum += power * speech_score

        normalized = power / float(np.sum(power))
        flux = 0.0
        if self._previous_psd is not None:
            flux = 0.5 * float(np.sum(np.abs(normalized - self._previous_psd)))
            self._spectral_flux_sum += flux
            self._spectral_flux_count += 1
        self._previous_psd = normalized

        stable_tonal = _clip_probability((0.22 - flatness) / 0.20) * _clip_probability(
            (0.32 - flux) / 0.28
        )
        wideband = _clip_probability((centroid - 700.0) / 3_500.0)
        self._music_weighted_sum += speech_score * (0.75 * stable_tonal + 0.25 * wideband)
        self._noise_weighted_sum += energy_score * _clip_probability((flatness - 0.12) / 0.55)

        spec_db = 10.0 * np.log10(power[voice_mask])
        if spec_db.size >= 9:
            local = np.convolve(spec_db, np.ones(9, dtype=np.float64) / 9.0, mode="same")
            holes = float(np.mean(spec_db[4:-4] < local[4:-4] - 18.0))
        else:
            holes = 0.0
        jitter = _clip_probability((flux - 0.18) / 0.55)
        self._codec_weighted_sum += speech_score * (0.65 * holes + 0.35 * jitter)

        if self._previous_rms_db is not None and self._previous_rms_db > -48.0:
            drop = self._previous_rms_db - rms_db
            if 0.0 < drop <= 15.0 and rms_db > -68.0:
                self._decay_sum += 1.0 - drop / 15.0
                self._decay_opportunities += 1
        self._previous_rms_db = rms_db

        valid = frame[:, :valid_samples].astype(np.float64, copy=False)
        self._clipped_samples += int(np.count_nonzero(np.abs(valid) >= 0.999))
        if self._channels == 2:
            left, right = valid[0], valid[1]
            self._corr_count += valid_samples
            self._sum_left += float(np.sum(left))
            self._sum_right += float(np.sum(right))
            self._sum_left_sq += float(np.dot(left, left))
            self._sum_right_sq += float(np.dot(right, right))
            self._sum_cross += float(np.dot(left, right))
        self._frames += 1

    def finish(self) -> StreamingAcousticReport:
        """Finalize the bounded summaries and return conservative evidence."""

        if self._finished:
            raise RuntimeError("finish may only be called once")
        self._finished = True
        if self.valid and self._pending_count:
            assert self._pending is not None
            self._pending[:, self._pending_count :] = 0.0
            self._process_frame(self._pending, self._pending_count)
            self._pending_count = 0

        if self._sample_rate is None:
            self._invalid_reasons.add("audio stream is empty")
        if not self.valid or self._frames == 0:
            return self._invalid_report()
        return self._valid_report()

    def _invalid_report(self) -> StreamingAcousticReport:
        duration = self._samples / self._sample_rate if self._sample_rate else 0.0
        reasons = tuple(sorted(self._invalid_reasons or {"no analyzable frames"}))
        return StreamingAcousticReport(
            qualification=_QUALIFICATION,
            valid=False,
            sample_rate=self._sample_rate,
            channels=self._channels,
            samples=self._samples,
            analyzed_frames=self._frames,
            duration_s=duration,
            speech_dominance=_estimate(0.0, 0.0, "lower", "invalid audio cannot establish speech"),
            music_risk=_estimate(1.0, 0.0, "upper", "invalid audio cannot exclude music"),
            crosstalk_risk=_estimate(1.0, 0.0, "upper", "invalid audio cannot exclude crosstalk"),
            band_limited_confidence=_estimate(
                0.0, 0.0, "lower", "invalid audio cannot establish a protected band edge"
            ),
            estimated_cutoff_hz=None,
            cutoff_confidence=0.0,
            bandwidth_shape="unknown",
            noise_risk=_estimate(1.0, 0.0, "upper", "invalid audio cannot quantify noise"),
            hum_confidence=_estimate(0.0, 0.0, "lower", "invalid audio cannot establish hum"),
            reverberation_risk=_estimate(
                1.0, 0.0, "upper", "invalid audio cannot quantify reverberation"
            ),
            clipping_risk=_estimate(1.0, 0.0, "upper", "invalid audio cannot exclude clipping"),
            clipping_fraction=0.0,
            codec_damage_risk=_estimate(
                1.0, 0.0, "upper", "invalid audio cannot quantify codec damage"
            ),
            channel_coherence=_estimate(
                0.0, 0.0, "lower", "invalid audio cannot establish channel coherence"
            ),
            rumble_confidence=_estimate(0.0, 0.0, "lower", "invalid audio cannot establish rumble"),
            recorded_high_frequency_speech_confidence=_estimate(
                1.0, 0.0, "upper", "invalid audio cannot exclude recorded high-frequency speech"
            ),
            uncertainty=1.0,
            uncertainty_reasons=reasons,
            state_bound_bytes=self._max_state_bytes,
        )

    def _valid_report(self) -> StreamingAcousticReport:
        assert self._sample_rate is not None
        assert self._channels is not None
        assert self._freqs is not None
        assert self._psd_sum is not None
        assert self._speech_psd_sum is not None
        duration = self._samples / self._sample_rate
        duration_conf = _clip_probability(duration / self.config.min_analysis_s)
        speech_raw = self._speech_sum / self._frames
        q10, q90 = self._energy_quantiles()
        dynamic_conf = _clip_probability((q90 - q10) / 24.0)
        speech_conf = min(0.72, duration_conf * (0.35 + 0.65 * dynamic_conf))

        speech_denom = max(self._speech_weight, 1e-12)
        music_raw = self._music_weighted_sum / speech_denom
        noise_flat = self._noise_weighted_sum / self._frames
        floor_close = _clip_probability(1.0 - (q90 - q10) / 30.0)
        noise_raw = 0.65 * noise_flat + 0.35 * floor_close
        codec_raw = self._codec_weighted_sum / speech_denom

        spectrum = (
            self._speech_psd_sum / speech_denom
            if self._speech_weight > 0.05
            else self._psd_sum / self._frames
        )
        cutoff_hz, cutoff_conf, shape, above_peakiness = self._estimate_bandwidth(spectrum)
        band_raw = cutoff_conf if shape == "steep_lowpass" else 0.0
        hf_raw = self._recorded_hf_confidence(spectrum, cutoff_hz, shape, above_peakiness)
        hum_raw = self._hum_confidence()
        rumble_raw = self._rumble_confidence(spectrum)
        coherence_raw, coherence_conf = self._channel_coherence(duration_conf)

        if self._channels == 1:
            crosstalk_raw, crosstalk_conf = 0.50, 0.0
        else:
            crosstalk_raw = _clip_probability((0.82 - coherence_raw) / 0.72)
            crosstalk_conf = min(0.45, duration_conf * coherence_conf)

        reverb_raw = self._decay_sum / max(self._decay_opportunities, 1)
        reverb_conf = min(0.40, duration_conf * self._decay_opportunities / 40.0)
        clipping_fraction = self._clipped_samples / max(self._samples * self._channels, 1)
        clipping_raw = _clip_probability(clipping_fraction / 0.001)
        clipping_conf = min(0.95, 0.55 + duration_conf * 0.40)

        reasons = [
            "deterministic spectral proxies are not a listener-trained or calibrated Sorani classifier"
        ]
        if duration < self.config.min_analysis_s:
            reasons.append("recording is shorter than the minimum stable analysis duration")
        if speech_raw < 0.30:
            reasons.append(
                "too little speech-like energy was observed for confident content decisions"
            )
        if self._channels == 1:
            reasons.append(
                "single-channel audio cannot establish whether overlapping speakers are present"
            )
        if self._sample_rate < 32_000:
            reasons.append(
                "sample rate provides insufficient high-frequency headroom for Restore evidence"
            )
        uncertainty = max(0.35, 1.0 - duration_conf * max(speech_raw, 0.15) * 0.75)

        return StreamingAcousticReport(
            qualification=_QUALIFICATION,
            valid=True,
            sample_rate=self._sample_rate,
            channels=self._channels,
            samples=self._samples,
            analyzed_frames=self._frames,
            duration_s=duration,
            speech_dominance=_estimate(
                speech_raw,
                speech_conf,
                "lower",
                "energy, voice-band ratio, centroid, ZCR and flatness",
            ),
            music_risk=_estimate(
                music_raw,
                min(0.50, duration_conf * speech_raw),
                "upper",
                "stable tonality and low spectral flux; no learned music classifier",
            ),
            crosstalk_risk=_estimate(
                crosstalk_raw,
                crosstalk_conf,
                "upper",
                "zero-lag stereo coherence only; speaker separation is unavailable",
            ),
            band_limited_confidence=_estimate(
                band_raw,
                min(0.92, duration_conf * cutoff_conf),
                "lower",
                "steep spectral cliff into a flat, featureless floor",
            ),
            estimated_cutoff_hz=cutoff_hz,
            cutoff_confidence=cutoff_conf,
            bandwidth_shape=shape,
            noise_risk=_estimate(
                noise_raw,
                min(0.55, duration_conf * dynamic_conf),
                "upper",
                "spectral flatness and frame-energy floor proximity",
            ),
            hum_confidence=_estimate(
                hum_raw,
                min(0.65, duration_conf),
                "lower",
                "phase-coherent 50/60 Hz fundamentals and harmonic series",
            ),
            reverberation_risk=_estimate(
                reverb_raw,
                reverb_conf,
                "upper",
                "slow frame-energy decay proxy; this is not an RT60 estimate",
            ),
            clipping_risk=_estimate(
                clipping_raw,
                clipping_conf,
                "upper",
                "fraction of decoded samples at or above 0.999 full scale",
            ),
            clipping_fraction=clipping_fraction,
            codec_damage_risk=_estimate(
                codec_raw,
                min(0.35, duration_conf * speech_raw),
                "upper",
                "spectral-hole and inter-frame flux proxy; codec identification is unavailable",
            ),
            channel_coherence=_estimate(
                coherence_raw,
                coherence_conf,
                "lower",
                "absolute zero-lag Pearson channel correlation",
            ),
            rumble_confidence=_estimate(
                rumble_raw,
                min(0.70, duration_conf),
                "lower",
                "sub-120 Hz energy relative to the core voice band",
            ),
            recorded_high_frequency_speech_confidence=_estimate(
                hf_raw,
                min(0.75, duration_conf * max(speech_raw, 0.25)),
                "upper",
                "speech-weighted energy and peaks above the estimated band edge",
            ),
            uncertainty=_clip_probability(uncertainty),
            uncertainty_reasons=tuple(reasons),
            state_bound_bytes=self._max_state_bytes,
        )

    def _energy_quantiles(self) -> tuple[float, float]:
        count = int(np.sum(self._energy_hist))
        if count == 0:
            return -120.0, -120.0
        cumulative = np.cumsum(self._energy_hist)
        lo = int(np.searchsorted(cumulative, max(1, math.ceil(count * 0.10))))
        hi = int(np.searchsorted(cumulative, max(1, math.ceil(count * 0.90))))
        scale = 126.0 / self.config.energy_bins
        return -120.0 + (lo + 0.5) * scale, -120.0 + (hi + 0.5) * scale

    def _estimate_bandwidth(
        self, spectrum: _Float64Array
    ) -> tuple[float, float, BandwidthShape, float]:
        assert self._sample_rate is not None
        assert self._freqs is not None
        nyquist = self._sample_rate / 2.0
        if float(np.sum(spectrum)) <= 1e-18:
            return nyquist, 0.95, "silence", 0.0
        if self._sample_rate < 32_000:
            return nyquist, 0.0, "unknown", 1.0

        db = 10.0 * np.log10(spectrum + 1e-24)
        smoothed = np.convolve(db, np.ones(7, dtype=np.float64) / 7.0, mode="same")
        candidates: list[tuple[float, float, float]] = []
        upper_search = min(16_000.0, nyquist * 0.72)
        for cutoff in self._freqs[(self._freqs >= 2_000.0) & (self._freqs <= upper_search)]:
            below = (self._freqs >= cutoff * 0.78) & (self._freqs <= cutoff * 0.95)
            above = (self._freqs >= cutoff * 1.08) & (
                self._freqs <= min(cutoff * 1.45, nyquist * 0.96)
            )
            if np.count_nonzero(below) < 4 or np.count_nonzero(above) < 6:
                continue
            below_db = float(np.median(smoothed[below]))
            above_values = smoothed[above]
            above_db = float(np.median(above_values))
            drop = below_db - above_db
            flatness_db = float(np.std(above_values))
            peakiness = float(np.max(above_values) - above_db)
            if drop >= 32.0 and flatness_db <= 8.0 and peakiness <= 15.0:
                candidates.append((float(cutoff), drop, peakiness))
        if not candidates:
            return nyquist, 0.90, "fullband", 1.0
        cutoff, drop, peakiness = min(candidates, key=lambda item: item[0])
        confidence = _clip_probability(0.72 + (drop - 32.0) / 120.0 - peakiness / 100.0)
        return cutoff, confidence, "steep_lowpass", peakiness

    def _recorded_hf_confidence(
        self,
        spectrum: _Float64Array,
        cutoff_hz: float,
        shape: BandwidthShape,
        peakiness: float,
    ) -> float:
        assert self._freqs is not None
        if shape != "steep_lowpass":
            return 1.0
        voice = (self._freqs >= 300.0) & (self._freqs <= 3_500.0)
        above = self._freqs >= min(cutoff_hz * 1.10, float(self._freqs[-1]))
        voice_power = float(np.mean(spectrum[voice])) + 1e-24
        above_power = float(np.max(spectrum[above])) + 1e-24 if np.any(above) else voice_power
        ratio_db = 10.0 * math.log10(above_power / voice_power)
        energy_score = _clip_probability((ratio_db + 65.0) / 35.0)
        peak_score = _clip_probability((peakiness - 5.0) / 12.0)
        return max(energy_score, peak_score)

    def _hum_confidence(self) -> float:
        assert self._hum_real is not None
        assert self._hum_imag is not None
        if self._hum_sample_count == 0 or self._mono_square_sum <= 1e-18:
            return 0.0
        amplitudes = 2.0 * np.hypot(self._hum_real, self._hum_imag) / float(self._hum_sample_count)
        rms = math.sqrt(self._mono_square_sum / self._hum_sample_count)
        group_50 = amplitudes[:5]
        group_60 = amplitudes[5:]
        scores: list[float] = []
        for group in (group_50, group_60):
            fundamental_ratio = float(group[0]) / max(rms, 1e-12)
            harmonic_ratio = float(np.sum(group[1:])) / max(4.0 * rms, 1e-12)
            ratio = fundamental_ratio + 0.35 * harmonic_ratio
            ratio_db = 20.0 * math.log10(max(ratio, 1e-12))
            scores.append(_clip_probability(_sigmoid((ratio_db + 27.0) / 4.5)))
        return max(scores)

    def _rumble_confidence(self, spectrum: _Float64Array) -> float:
        assert self._freqs is not None
        if float(np.sum(spectrum)) <= 1e-18:
            return 0.0
        low = (self._freqs >= 25.0) & (self._freqs <= 120.0)
        voice = (self._freqs >= 300.0) & (self._freqs <= 3_000.0)
        low_power = float(np.mean(spectrum[low])) + 1e-24 if np.any(low) else 1e-24
        voice_power = float(np.mean(spectrum[voice])) + 1e-24
        ratio_db = 10.0 * math.log10(low_power / voice_power)
        return _clip_probability(_sigmoid((ratio_db + 16.0) / 3.5))

    def _channel_coherence(self, duration_confidence: float) -> tuple[float, float]:
        if self._channels == 1:
            return 1.0, 1.0
        if self._corr_count < 2:
            return 0.0, 0.0
        count = float(self._corr_count)
        covariance = self._sum_cross - self._sum_left * self._sum_right / count
        left_var = self._sum_left_sq - self._sum_left * self._sum_left / count
        right_var = self._sum_right_sq - self._sum_right * self._sum_right / count
        denominator = math.sqrt(max(left_var * right_var, 0.0))
        if denominator <= 1e-18:
            return 0.0, 0.0
        return _clip_probability(abs(covariance / denominator)), min(0.90, duration_confidence)


def analyze_audio_stream(
    chunks: Iterable[AudioBuffer],
    *,
    config: AnalyzerConfig = DEFAULT_ANALYZER_CONFIG,
) -> StreamingAcousticReport:
    """Consume one iterable of decoded chunks and return bounded acoustic evidence."""

    analyzer = StreamingAcousticAnalyzer(config)
    for chunk in chunks:
        analyzer.accept(chunk)
    return analyzer.finish()


__all__ = [
    "AnalyzerConfig",
    "DEFAULT_ANALYZER_CONFIG",
    "ProbabilityEstimate",
    "StreamingAcousticAnalyzer",
    "StreamingAcousticReport",
    "analyze_audio_stream",
]
