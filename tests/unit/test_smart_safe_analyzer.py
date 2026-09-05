from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import signal

from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.smart_safe.analyzer import (
    AnalyzerConfig,
    StreamingAcousticAnalyzer,
    analyze_audio_stream,
)

pytestmark = pytest.mark.unit

SR = 48_000


def _speechlike(seconds: float = 10.0, *, seed: int = 7) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * SR), dtype=np.float64) / SR
    envelope = 0.30 + 0.70 * np.square(np.sin(2.0 * np.pi * 2.7 * t))
    voiced = (
        0.13 * np.sin(2.0 * np.pi * 175.0 * t)
        + 0.06 * np.sin(2.0 * np.pi * 525.0 * t)
        + 0.03 * np.sin(2.0 * np.pi * 1_225.0 * t)
    )
    frication = signal.sosfilt(
        signal.butter(4, [2_000.0, 11_000.0], btype="bandpass", fs=SR, output="sos"),
        rng.normal(size=t.size),
    )
    return np.asarray(envelope * voiced + 0.012 * frication, dtype=np.float32)


def _chunks(audio: NDArray[np.float32], sizes: tuple[int, ...]) -> Iterator[AudioBuffer]:
    offset = 0
    index = 0
    while offset < audio.shape[1]:
        size = sizes[index % len(sizes)]
        yield AudioBuffer(
            audio[:, offset : offset + size],
            SR,
            ChannelMode.AMBIGUOUS_STEREO if audio.shape[0] == 2 else ChannelMode.MONO,
        )
        offset += size
        index += 1


def test_chunk_boundaries_do_not_change_acoustic_report() -> None:
    mono = _speechlike(9.137)
    stereo = np.stack((mono, 0.91 * mono), axis=0)

    whole = analyze_audio_stream(list(_chunks(stereo, (stereo.shape[1],))))
    irregular = analyze_audio_stream(_chunks(stereo, (1, 17, 509, 2048, 3, 8191, 127)))

    assert irregular == whole
    assert whole.qualification == "experimental_unqualified"
    assert whole.state_bound_bytes < 100_000


def test_state_is_bounded_for_a_long_lazy_stream() -> None:
    analyzer = StreamingAcousticAnalyzer()
    rng = np.random.default_rng(19)
    chunk = AudioBuffer(
        np.asarray(rng.normal(0.0, 0.02, size=257), dtype=np.float32),
        SR,
    )

    for _ in range(32):
        analyzer.accept(chunk)
    early_bytes = analyzer.state_nbytes
    for _ in range(49_968):  # More than 4 minutes; state must remain duration-independent.
        analyzer.accept(chunk)
    late_bytes = analyzer.state_nbytes
    report = analyzer.finish()

    assert report.duration_s > 260.0
    assert late_bytes == early_bytes
    assert report.state_bound_bytes == late_bytes
    assert late_bytes < 100_000


def test_nonfinite_and_stream_shape_changes_fail_closed() -> None:
    bad = _speechlike(0.2)
    bad[41] = np.nan
    report = analyze_audio_stream([AudioBuffer(bad, SR)])
    evidence = report.decision_evidence(reconstruction_consent=True)

    assert report.valid is False
    assert report.uncertainty == 1.0
    assert "NaN" in " ".join(report.uncertainty_reasons)
    assert evidence.reconstruction_consent is False
    assert evidence.speech_dominance == 0.0
    assert evidence.music_risk == 1.0
    assert evidence.band_limited_confidence == 0.0
    assert evidence.recorded_high_frequency_speech_confidence == 1.0

    analyzer = StreamingAcousticAnalyzer()
    analyzer.accept(AudioBuffer(np.zeros(2048, dtype=np.float32), SR))
    analyzer.accept(AudioBuffer(np.zeros(2048, dtype=np.float32), 44_100))
    changed = analyzer.finish()
    assert changed.valid is False
    assert "changed mid-stream" in " ".join(changed.uncertainty_reasons)


def test_empty_and_unsupported_layouts_fail_closed() -> None:
    assert analyze_audio_stream([]).valid is False
    three_channels = AudioBuffer(np.zeros((3, 2048), dtype=np.float32), SR)
    report = analyze_audio_stream([three_channels])
    assert report.valid is False
    assert report.channels is None
    assert "mono and stereo" in " ".join(report.uncertainty_reasons)


def test_fullband_and_steep_lowpass_are_distinguished_conservatively() -> None:
    source = _speechlike(12.0)
    lowpassed = signal.sosfilt(
        signal.butter(20, 7_800.0, btype="lowpass", fs=SR, output="sos"),
        source,
    ).astype(np.float32)

    full = analyze_audio_stream([AudioBuffer(source, SR)])
    low = analyze_audio_stream([AudioBuffer(lowpassed, SR)])

    assert full.bandwidth_shape == "fullband"
    assert full.band_limited_confidence.conservative == 0.0
    assert full.recorded_high_frequency_speech_confidence.conservative == 1.0
    assert low.bandwidth_shape == "steep_lowpass"
    assert low.estimated_cutoff_hz is not None
    assert 7_000.0 <= low.estimated_cutoff_hz <= 12_000.0
    assert low.band_limited_confidence.value > full.band_limited_confidence.value
    # The proxy is intentionally too uncertain to grant Restore eligibility.
    assert low.decision_evidence(reconstruction_consent=True).band_limited_confidence < 0.85


def test_hum_rumble_clipping_and_channel_coherence_are_reported() -> None:
    base = _speechlike(10.0)
    t = np.arange(base.size, dtype=np.float64) / SR
    contaminated = np.asarray(base + 0.18 * np.sin(2.0 * np.pi * 50.0 * t), dtype=np.float32)
    clipped = np.clip(contaminated * 8.0, -1.0, 1.0).astype(np.float32)
    coherent = np.stack((clipped, clipped), axis=0)
    independent = np.stack((base, np.roll(base, 337)), axis=0)

    clean_report = analyze_audio_stream([AudioBuffer(base, SR)])
    contaminated_report = analyze_audio_stream(
        [AudioBuffer(coherent, SR, ChannelMode.AMBIGUOUS_STEREO)]
    )
    independent_report = analyze_audio_stream(
        [AudioBuffer(independent, SR, ChannelMode.AMBIGUOUS_STEREO)]
    )

    assert contaminated_report.hum_confidence.value > clean_report.hum_confidence.value
    assert contaminated_report.rumble_confidence.value > clean_report.rumble_confidence.value
    assert contaminated_report.clipping_fraction > 0.01
    assert contaminated_report.clipping_risk.value == 1.0
    assert contaminated_report.channel_coherence.value > 0.99
    assert independent_report.channel_coherence.value < 0.95
    assert independent_report.crosstalk_risk.value > contaminated_report.crosstalk_risk.value


def test_silence_does_not_create_positive_intervention_evidence() -> None:
    report = analyze_audio_stream([AudioBuffer(np.zeros(SR * 10, dtype=np.float32), SR)])
    evidence = report.decision_evidence(reconstruction_consent=True)

    assert report.valid is True
    assert report.bandwidth_shape == "silence"
    assert evidence.speech_dominance == 0.0
    assert evidence.rumble_confidence == 0.0
    assert evidence.band_limited_confidence == 0.0
    assert evidence.recorded_high_frequency_speech_confidence == 1.0


@pytest.mark.parametrize(
    "config",
    [
        AnalyzerConfig(frame_samples=256),
        AnalyzerConfig(frame_samples=8192),
        AnalyzerConfig(energy_bins=32),
        AnalyzerConfig(energy_bins=512),
    ],
)
def test_supported_state_sizes_remain_bounded(config: AnalyzerConfig) -> None:
    analyzer = StreamingAcousticAnalyzer(config)
    analyzer.accept(AudioBuffer(_speechlike(0.1), SR))
    report = analyzer.finish()
    assert report.state_bound_bytes < 250_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_samples": 255},
        {"frame_samples": 1000},
        {"frame_samples": 16_384},
        {"min_analysis_s": float("nan")},
        {"energy_bins": 12},
    ],
)
def test_bad_analyzer_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AnalyzerConfig(**kwargs)  # type: ignore[arg-type]
