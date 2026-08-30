"""Disk-backed Natural rendering: equivalence and bounded stage selection."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.audio.decode as decode_module
import hawavoclean.audio.encode as encode_module
import hawavoclean.pipeline as pipeline
from hawavoclean.assembly.stitch import (
    assemble_channel_timeline,
    assemble_channel_timeline_into,
)
from hawavoclean.audio.channels import classify_channels, classify_channels_bounded
from hawavoclean.audio.decode import decode_audio, decode_audio_to_memmap
from hawavoclean.audio.encode import _wav_container_format, encode_audio, encode_audio_streaming
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.config import load_config
from hawavoclean.errors import PreflightError
from hawavoclean.finishing.limiter import (
    apply_lookahead_limiter,
    apply_lookahead_limiter_to_memmap,
)
from hawavoclean.finishing.loudness import (
    measure_loudness_and_peaks,
    measure_loudness_and_peaks_streaming,
)
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.hashing import hash_bytes, hash_file, hash_numpy
from hawavoclean.paths import profile_config_path
from hawavoclean.segmentation.types import SpeechUnit
from hawavoclean.segmentation.utterances import build_speech_units

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"


@pytest.mark.unit
def test_disk_decode_and_streaming_meter_are_sample_and_metric_exact(tmp_path: Path) -> None:
    probe = probe_audio(FIXTURE)
    whole = decode_audio(probe)
    disk = decode_audio_to_memmap(probe, tmp_path / "decoded.f32", chunk_samples=7_777)

    assert isinstance(disk.data, np.memmap)
    assert np.array_equal(disk.data, whole.data)
    assert measure_loudness_and_peaks_streaming(
        disk.data, disk.sample_rate, chunk_samples=977
    ) == measure_loudness_and_peaks(whole.data, whole.sample_rate)


@pytest.mark.unit
def test_disk_decode_fails_closed_and_removes_partial_stages_when_space_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = AudioProbeResult(
        path=tmp_path / "source.wav",
        format_name="wav",
        codec_name="pcm_f32le",
        sample_rate=48_000,
        channels=1,
        duration_s=1.0,
        samples=48_000,
        bit_depth=32,
        sha256="0" * 64,
    )

    def one_chunk(*_args: object, **_kwargs: object) -> object:
        yield AudioBuffer(np.ones((1, 32), dtype=np.float32), 48_000)

    monkeypatch.setattr(decode_module, "iter_decode_audio", one_chunk)
    monkeypatch.setattr(
        "hawavoclean.audio.decode.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )
    destination = tmp_path / "decoded.f32"
    with pytest.raises(PreflightError, match="Insufficient scratch space"):
        decode_audio_to_memmap(probe, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.channel-*.tmp"))


@pytest.mark.unit
def test_memmap_assembly_limiter_and_encoder_match_allocating_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rate = 48_000
    samples = 100_003
    rng = np.random.default_rng(42)
    source = rng.normal(0.0, 0.25, size=(2, samples)).astype(np.float32)
    source[:, ::777] += np.float32(1.1)

    # One unit per channel is enough to prove the caller-owned assembly writes
    # the same timeline. Existing assembly tests cover multi-unit seam maths.
    config = load_config(profile_config_path("development"), is_production=False)
    units = build_speech_units(source[0], sample_rate, 0, config.segmentation)
    waves = [source[0, unit.start_sample : unit.end_sample] for unit in units]
    expected_channel = assemble_channel_timeline(units, waves, samples, sample_rate)
    assembled = np.memmap(
        tmp_path / "assembled.f32",
        mode="w+",
        dtype=np.float32,
        shape=(2, samples),
    )
    assemble_channel_timeline_into(
        assembled[0], units, waves, total_samples=samples, sample_rate=sample_rate
    )
    assembled[1] = source[1]
    assert np.array_equal(assembled[0], expected_channel)

    gain = np.float32(1.7)
    expected = apply_lookahead_limiter(
        np.multiply(assembled, gain, dtype=np.float32), sample_rate, ceiling_dbtp=-1.0
    )
    disk = apply_lookahead_limiter_to_memmap(
        assembled,
        sample_rate,
        tmp_path / "limited.f32",
        input_gain_linear=float(gain),
        ceiling_dbtp=-1.0,
        chunk_samples=997,
    )
    assert isinstance(disk.limited_waveform, np.memmap)
    assert disk.gain_envelope.size == 0
    assert np.array_equal(disk.limited_waveform, expected.limited_waveform)

    expected_wav = encode_audio(
        AudioBuffer(expected.limited_waveform, sample_rate),
        tmp_path / "expected.wav",
        output_bit_depth="pcm24",
        dither=True,
        seed_context="stream-equivalence",
    )
    actual_wav = encode_audio_streaming(
        AudioBuffer(disk.limited_waveform, sample_rate),
        tmp_path / "actual.wav",
        output_bit_depth="pcm24",
        dither=True,
        seed_context="stream-equivalence",
        chunk_samples=997,
    )
    assert hash_file(actual_wav) == hash_file(expected_wav)

    expected_float = encode_audio(
        AudioBuffer(expected.limited_waveform, sample_rate),
        tmp_path / "expected-float.wav",
        output_bit_depth="float32",
        dither=False,
    )
    actual_float = encode_audio_streaming(
        AudioBuffer(disk.limited_waveform, sample_rate),
        tmp_path / "actual-float.wav",
        output_bit_depth="float32",
        dither=False,
        chunk_samples=997,
    )
    assert hash_file(actual_float) == hash_file(expected_float)
    peak_at = expected_float.read_bytes().index(b"PEAK") + 8
    assert expected_float.read_bytes()[peak_at + 4 : peak_at + 8] == b"\0\0\0\0"
    assert _wav_container_format(2, 6 * 60 * 60 * 48_000, "PCM_24") == "RF64"

    monkeypatch.setattr(encode_module, "_wav_container_format", lambda *_args: "RF64")
    rf64 = encode_audio_streaming(
        AudioBuffer(disk.limited_waveform, sample_rate),
        tmp_path / "forced-rf64.wav",
        output_bit_depth="float32",
        dither=False,
        chunk_samples=997,
    )
    assert sf.info(rf64).format == "RF64"


@pytest.mark.unit
def test_maskless_segmentation_preserves_unit_boundaries_and_hashes() -> None:
    data, sample_rate = sf.read(FIXTURE, dtype="float32", always_2d=True)
    waveform = np.asarray(data[:, 0], dtype=np.float32)
    config = load_config(profile_config_path("production"), is_production=True)
    retained = build_speech_units(waveform, sample_rate, 0, config.segmentation)
    bounded = build_speech_units(
        waveform,
        sample_rate,
        0,
        config.segmentation,
        retain_speech_mask=False,
    )

    def identity(value: SpeechUnit) -> tuple[object, ...]:
        return (
            value.unit_id,
            value.channel_id,
            value.start_sample,
            value.end_sample,
            value.context_start_sample,
            value.context_end_sample,
            value.is_speech,
            value.forced_boundary,
            value.input_sha256,
        )

    assert [identity(unit) for unit in bounded] == [identity(unit) for unit in retained]
    assert all(unit.speech_mask.size == 0 for unit in bounded)


@pytest.mark.unit
def test_bounded_hash_and_channel_analysis_preserve_results() -> None:
    base = np.arange(3 * 43_219, dtype=np.float32).reshape(3, -1)
    non_contiguous = base[:, 1::3].T
    expected_bytes = np.ascontiguousarray(non_contiguous).tobytes(order="C")
    assert hash_numpy(non_contiguous) == hash_bytes(expected_bytes)

    rng = np.random.default_rng(73)
    left = rng.normal(0.0, 0.1, 90_007).astype(np.float32)
    for right in (left.copy(), rng.normal(0.0, 0.1, 90_007).astype(np.float32)):
        buffer = AudioBuffer(np.stack((left, right)), 48_000)
        assert classify_channels_bounded(buffer, chunk_samples=997) == classify_channels(buffer)
    assert pipeline._natural_worker_limit(64, streaming=True) == 2
    assert pipeline._natural_worker_limit(64, streaming=False) == 64


@pytest.mark.integration
def test_forced_streaming_natural_pipeline_is_byte_exact_and_uses_bounded_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mono, sample_rate = sf.read(FIXTURE, dtype="float32", always_2d=True)
    source = tmp_path / "dual-mono-source.wav"
    sf.write(source, np.repeat(mono, 2, axis=1), sample_rate, subtype="PCM_24")

    def whole_file_decode_used(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Natural job called the unbounded whole-file decoder")

    monkeypatch.setattr(pipeline, "decode_audio", whole_file_decode_used)
    monkeypatch.setattr(pipeline, "NATURAL_STREAMING_THRESHOLD_BYTES", 1 << 60)
    short_path = tmp_path / "short-path.wav"
    short_report = pipeline.run_pipeline(
        source,
        short_path,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    # The probe now lies about the decoded length. The actual streamed frame
    # count must still select the long route and trigger its real space check.
    actual_probe = probe_audio(source)
    underreported = replace(
        actual_probe,
        duration_s=1.0 / actual_probe.sample_rate,
        samples=1,
    )
    monkeypatch.setattr(
        pipeline,
        "probe_audio",
        lambda path, *_args, **_kwargs: replace(underreported, path=Path(path)),
    )
    actual_bytes = actual_probe.samples * actual_probe.channels * np.dtype(np.float32).itemsize
    monkeypatch.setattr(pipeline, "NATURAL_STREAMING_THRESHOLD_BYTES", actual_bytes - 1)

    def allocating_path_used(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forced streaming Natural job called an allocating stage")

    monkeypatch.setattr(pipeline, "assemble_channel_timeline", allocating_path_used)
    monkeypatch.setattr(pipeline, "apply_lookahead_limiter", allocating_path_used)
    monkeypatch.setattr(pipeline, "encode_audio", allocating_path_used)

    streamed = tmp_path / "streamed.wav"
    streamed_report = pipeline.run_pipeline(
        source,
        streamed,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    assert hash_file(streamed) == hash_file(short_path)
    assert streamed_report.input.integrated_lufs == short_report.input.integrated_lufs
    assert streamed_report.output.integrated_lufs == short_report.output.integrated_lufs
    assert streamed_report.output.true_peak_dbtp == short_report.output.true_peak_dbtp
    assert streamed_report.summary == short_report.summary
    assert [unit.output_sha256 for unit in streamed_report.units] == [
        unit.output_sha256 for unit in short_report.units
    ]
