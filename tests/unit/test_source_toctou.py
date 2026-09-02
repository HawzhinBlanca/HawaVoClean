"""Tests for immutable source snapshot capture before queue acceptance (G0.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from hawavoclean.pipeline import run_pipeline
from hawavoclean.server.jobs import JobManager, default_command


def _create_wav(path: Path, freq: float = 440.0, duration: float = 0.5, sr: int = 16000) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sig, sr, format="WAV", subtype="PCM_16")


def test_job_submission_pins_source_snapshot_against_toctou_mutation(tmp_path: Path) -> None:
    """Modifying the source file after queue acceptance must not affect the job."""
    input_wav = tmp_path / "original_input.wav"
    output_wav = tmp_path / "output.wav"
    _create_wav(input_wav, freq=440.0)

    manager = JobManager()
    try:
        snap = manager.submit(
            input_path=input_wav,
            output_path=output_wav,
            profile="production",
            overwrite=True,
        )
        job_id = snap["job_id"]
        record = manager._jobs.get(job_id)
        assert record is not None
        assert record.source_snapshot_path is not None
        assert record.source_snapshot_path.exists()

        # Mutate the original input file after submission
        _create_wav(input_wav, freq=1000.0)

        # Execute default_command against the record
        cmd = default_command(record)
        assert "--original-input-path" in cmd
        assert str(input_wav) in cmd

        # Run pipeline directly with the snapshot and original_input_path
        report = run_pipeline(
            input_path=record.source_snapshot_path,
            output_path=output_wav,
            profile="production",
            overwrite=True,
            original_input_path=input_wav,
        )

        assert report.input.path == str(input_wav)
        assert output_wav.exists()
    finally:
        manager.shutdown()


def test_pinned_source_edge_cases_and_error_branches(tmp_path: Path) -> None:
    import pytest

    from hawavoclean.errors import MediaPreflightError, PreflightError
    from hawavoclean.source_pin import PinnedSource, remove_source_snapshot_tree

    # 1. remove_source_snapshot_tree on nonexistent path is a no-op
    remove_source_snapshot_tree(tmp_path / "nonexistent_dir")

    # 2. remove_source_snapshot_tree with nested subdirectories
    nested_dir = tmp_path / "nested_tree"
    nested_sub = nested_dir / "sub"
    nested_sub.mkdir(parents=True)
    nested_file = nested_sub / "test.txt"
    nested_file.write_text("hello")
    nested_file.chmod(0o400)
    nested_sub.chmod(0o500)
    nested_dir.chmod(0o500)
    remove_source_snapshot_tree(nested_dir)
    assert not nested_dir.exists()

    staging_root = tmp_path / "staging"
    source_file = tmp_path / "source.wav"
    _create_wav(source_file)

    # 3. Nonexistent file
    with pytest.raises(MediaPreflightError, match="does not exist"):
        PinnedSource.create(
            tmp_path / "missing.wav", staging_root=staging_root, max_file_size_bytes=10_000_000
        )

    # 4. Directory instead of regular file
    with pytest.raises(MediaPreflightError, match="not a regular file"):
        PinnedSource.create(tmp_path, staging_root=staging_root, max_file_size_bytes=10_000_000)

    # 5. Empty file
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()
    with pytest.raises(MediaPreflightError, match="empty"):
        PinnedSource.create(empty_file, staging_root=staging_root, max_file_size_bytes=10_000_000)

    # 6. File too large
    with pytest.raises(MediaPreflightError, match="maximum is 10 bytes"):
        PinnedSource.create(source_file, staging_root=staging_root, max_file_size_bytes=10)

    # 7. Context manager auto cleanup when unadopted
    with PinnedSource.create(
        source_file, staging_root=staging_root, max_file_size_bytes=10_000_000
    ) as pin:
        assert pin.path.exists()
        snap_dir = pin.directory
    assert not snap_dir.exists()

    # 8. verify() error branches
    pin2 = PinnedSource.create(
        source_file, staging_root=staging_root, max_file_size_bytes=10_000_000
    )
    try:
        # File deleted
        pin2.directory.chmod(0o700)
        pin2.path.unlink()
        with pytest.raises(MediaPreflightError, match="disappeared"):
            pin2.verify()
    finally:
        pin2.cleanup_unadopted()

    pin3 = PinnedSource.create(
        source_file, staging_root=staging_root, max_file_size_bytes=10_000_000
    )
    try:
        # File replaced by directory
        pin3.directory.chmod(0o700)
        pin3.path.unlink()
        pin3.path.mkdir()
        with pytest.raises(MediaPreflightError, match="no longer a regular file"):
            pin3.verify()
    finally:
        pin3.cleanup_unadopted()

    # 9. adopt() branches
    pin4 = PinnedSource.create(
        source_file, staging_root=staging_root, max_file_size_bytes=10_000_000
    )
    workspace = tmp_path / "ws1"
    workspace.mkdir()
    try:
        # Destination already exists
        (workspace / "source-snapshot").mkdir()
        with pytest.raises(PreflightError, match="already exists"):
            pin4.adopt(workspace)
        (workspace / "source-snapshot").rmdir()

        # Successful adoption
        adopted_path = pin4.adopt(workspace)
        assert adopted_path.exists()
        assert pin4._adopted is True

        # cleanup_unadopted() is a no-op once adopted
        pin4.cleanup_unadopted()
        assert adopted_path.exists()
    finally:
        remove_source_snapshot_tree(workspace)
