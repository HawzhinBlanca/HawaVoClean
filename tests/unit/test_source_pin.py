"""Source identity stays bound from validation through FFmpeg decode."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.pipeline as pipeline
from hawavoclean.audio.decode import (
    DECODE_CHUNK_SAMPLES,
    decode_audio,
    decode_audio_to_memmap,
)
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.errors import MediaPreflightError, MediaPreflightReason
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.hashing import hash_file, hash_numpy
from hawavoclean.source_pin import PinnedSource, remove_source_snapshot_tree

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"
MAX_SOURCE_BYTES = 8 * 1024**3


def _close_memmap(value: np.ndarray[Any, np.dtype[np.float32]]) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


@pytest.mark.unit
def test_pin_fails_closed_when_open_source_is_rewritten_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * 4096)
    staging = tmp_path / "work"
    real_read = os.read
    changed = False

    def rewrite_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        block = real_read(descriptor, count)
        if block and not changed:
            changed = True
            source.write_bytes(b"b" * 4096)
        return block

    monkeypatch.setattr("hawavoclean.source_pin.SOURCE_COPY_CHUNK_BYTES", 128)
    monkeypatch.setattr("hawavoclean.source_pin.os.read", rewrite_after_first_read)

    with pytest.raises(MediaPreflightError) as raised:
        PinnedSource.create(
            source,
            staging_root=staging,
            max_file_size_bytes=MAX_SOURCE_BYTES,
        )

    assert raised.value.reason is MediaPreflightReason.SOURCE_CHANGED
    assert changed
    assert not list(staging.glob("source-pin-*"))


@pytest.mark.unit
def test_frozen_snapshot_adopts_into_job_and_cleans_cross_platform(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source bytes")
    pin = PinnedSource.create(
        source,
        staging_root=tmp_path / "work",
        max_file_size_bytes=MAX_SOURCE_BYTES,
    )
    workspace = tmp_path / "job"
    workspace.mkdir()

    adopted = pin.adopt(workspace)
    assert adopted == workspace / "source-snapshot" / "source.wav"
    assert adopted.read_bytes() == b"source bytes"
    pin.cleanup_unadopted()
    assert adopted.exists()

    remove_source_snapshot_tree(workspace / "source-snapshot")
    assert not adopted.exists()


@pytest.mark.unit
def test_whole_and_streaming_decoders_use_pinned_bytes_after_original_rewrite(
    tmp_path: Path,
) -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate // 4, dtype=np.float32) / np.float32(sample_rate)
    old_audio = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    new_audio = np.sin(2.0 * np.pi * 1230.0 * time).astype(np.float32)
    source = tmp_path / "source.wav"
    sf.write(source, old_audio, sample_rate, subtype="PCM_24")
    original_truth = decode_audio(probe_audio(source))

    pin = PinnedSource.create(
        source,
        staging_root=tmp_path / "work",
        max_file_size_bytes=MAX_SOURCE_BYTES,
    )
    try:
        snapshot_probe = probe_audio(pin.path)
        sf.write(source, new_audio, sample_rate, subtype="PCM_24")

        whole = decode_audio(snapshot_probe)
        streamed = decode_audio_to_memmap(snapshot_probe, tmp_path / "decoded.f32")
        try:
            assert hash_file(source) != pin.sha256
            assert np.array_equal(whole.data, original_truth.data)
            assert np.array_equal(streamed.data, original_truth.data)
        finally:
            _close_memmap(streamed.data)
    finally:
        pin.cleanup_unadopted()


@pytest.mark.integration
def test_pipeline_report_and_long_decode_remain_bound_when_original_changes_after_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(FIXTURE.read_bytes())
    old_sha = hash_file(source)
    old_decoded_sha = hash_numpy(decode_audio(probe_audio(source)).data)

    replacement = tmp_path / "replacement.wav"
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    sf.write(replacement, -audio, sample_rate, subtype="PCM_24")
    replacement_bytes = replacement.read_bytes()
    replacement_sha = hash_file(replacement)
    assert replacement_sha != old_sha

    real_probe = probe_audio
    real_decode = decode_audio_to_memmap
    observed: dict[str, object] = {}

    def probe_then_rewrite(
        path: Path | str,
        max_sample_rate: int = 48000,
        supported_sample_rates: Sequence[int] | None = None,
        *,
        max_file_size_bytes: int | None = None,
        max_duration_s: float | None = None,
        max_channels: int | None = None,
    ) -> AudioProbeResult:
        result = real_probe(
            path,
            max_sample_rate,
            supported_sample_rates,
            max_file_size_bytes=max_file_size_bytes,
            max_duration_s=max_duration_s,
            max_channels=max_channels,
        )
        source.write_bytes(replacement_bytes)
        observed["snapshot"] = Path(path)
        return result

    def capture_decode(
        probe: AudioProbeResult,
        output_path: Path | str,
        timeout_s: float = 1800.0,
        *,
        chunk_samples: int = DECODE_CHUNK_SAMPLES,
    ) -> AudioBuffer:
        assert probe.path != source
        assert hash_file(probe.path) == old_sha
        result = real_decode(
            probe,
            output_path,
            timeout_s,
            chunk_samples=chunk_samples,
        )
        observed["decoded_sha"] = hash_numpy(result.data)
        return result

    monkeypatch.setattr("hawavoclean.pipeline.probe_audio", probe_then_rewrite)
    monkeypatch.setattr("hawavoclean.pipeline.decode_audio_to_memmap", capture_decode)
    monkeypatch.setattr("hawavoclean.pipeline.NATURAL_STREAMING_THRESHOLD_BYTES", 1)

    report = pipeline.run_pipeline(
        source,
        tmp_path / "output.wav",
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    assert hash_file(source) == replacement_sha
    assert report.input.path == str(source.resolve())
    assert report.input.sha256 == old_sha
    assert observed["decoded_sha"] == old_decoded_sha
    snapshot = observed["snapshot"]
    assert isinstance(snapshot, Path)
    assert not snapshot.exists()


@pytest.mark.unit
def test_pipeline_fails_before_decode_if_private_snapshot_changes_after_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(FIXTURE.read_bytes())
    real_probe = probe_audio

    def probe_then_tamper(
        path: Path | str,
        max_sample_rate: int = 48000,
        supported_sample_rates: Sequence[int] | None = None,
        *,
        max_file_size_bytes: int | None = None,
        max_duration_s: float | None = None,
        max_channels: int | None = None,
    ) -> AudioProbeResult:
        result = real_probe(
            path,
            max_sample_rate,
            supported_sample_rates,
            max_file_size_bytes=max_file_size_bytes,
            max_duration_s=max_duration_s,
            max_channels=max_channels,
        )
        snapshot = Path(path)
        snapshot.chmod(0o600)
        with snapshot.open("r+b") as handle:
            handle.seek(-1, 2)
            handle.write(b"\x00")
        return result

    def fail_decode(
        _probe: AudioProbeResult,
        _output_path: Path | str,
        _timeout_s: float = 1800.0,
        *,
        chunk_samples: int = DECODE_CHUNK_SAMPLES,
    ) -> AudioBuffer:
        del chunk_samples
        pytest.fail("changed snapshot reached decode")

    monkeypatch.setattr("hawavoclean.pipeline.probe_audio", probe_then_tamper)
    monkeypatch.setattr("hawavoclean.pipeline.decode_audio_to_memmap", fail_decode)

    with pytest.raises(MediaPreflightError) as raised:
        pipeline.run_pipeline(
            source,
            tmp_path / "output.wav",
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )
    assert raised.value.reason is MediaPreflightReason.SOURCE_CHANGED


@pytest.mark.unit
def test_source_pin_edge_cases_and_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hawavoclean.errors import PreflightError
    from hawavoclean.source_pin import remove_source_snapshot_tree

    source = tmp_path / "source.wav"
    source.write_bytes(b"content for error branches")
    staging = tmp_path / "staging"

    # 1. os.open raises OSError
    def failing_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("Permission denied")

    monkeypatch.setattr("hawavoclean.source_pin.os.open", failing_open)
    with pytest.raises(MediaPreflightError) as exc:
        PinnedSource.create(source, staging_root=staging, max_file_size_bytes=MAX_SOURCE_BYTES)
    assert exc.value.reason is MediaPreflightReason.SOURCE_CHANGED

    # 2. Insufficient scratch space
    monkeypatch.undo()

    class FakeDiskUsage:
        free = 10  # very small

    monkeypatch.setattr("hawavoclean.source_pin.shutil.disk_usage", lambda _p: FakeDiskUsage())
    with pytest.raises(PreflightError, match="Insufficient scratch space"):
        PinnedSource.create(source, staging_root=staging, max_file_size_bytes=MAX_SOURCE_BYTES)

    # 3. Empty file
    empty_source = tmp_path / "empty.wav"
    empty_source.touch()
    with pytest.raises(MediaPreflightError) as exc_empty:
        PinnedSource.create(
            empty_source, staging_root=staging, max_file_size_bytes=MAX_SOURCE_BYTES
        )
    assert exc_empty.value.reason is MediaPreflightReason.EMPTY_FILE

    # 4. File too large
    with pytest.raises(MediaPreflightError) as exc_large:
        PinnedSource.create(source, staging_root=staging, max_file_size_bytes=5)
    assert exc_large.value.reason is MediaPreflightReason.FILE_TOO_LARGE

    # 5. remove_source_snapshot_tree handles non-existent and directories
    remove_source_snapshot_tree(tmp_path / "nonexistent_dir")

    test_dir = tmp_path / "to_clean"
    test_dir.mkdir()
    (test_dir / "file.txt").write_text("hello")
    remove_source_snapshot_tree(test_dir)
    assert not test_dir.exists()
