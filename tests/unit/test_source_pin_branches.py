from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from hawavoclean.errors import MediaPreflightError, MediaPreflightReason, PreflightError
from hawavoclean.source_pin import PinnedSource

SAMPLE_WAV = Path("tests/fixtures/sample_sorani_podcast.wav")
MAX_SIZE = 100 * 1024 * 1024


def test_pin_source_staging_mkdir_oserror(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    with (
        patch.object(Path, "mkdir", side_effect=OSError("denied")),
        pytest.raises(PreflightError, match="Cannot prepare source snapshot storage"),
    ):
        PinnedSource.create(SAMPLE_WAV, staging_root=staging, max_file_size_bytes=MAX_SIZE)


def test_pin_source_became_shorter(tmp_path: Path) -> None:
    staging = tmp_path / "staging"

    def fake_read(_fd: int, _n: int) -> bytes:
        return b""

    with (
        patch("os.read", side_effect=fake_read),
        pytest.raises(MediaPreflightError) as exc,
    ):
        PinnedSource.create(SAMPLE_WAV, staging_root=staging, max_file_size_bytes=MAX_SIZE)
    assert exc.value.reason is MediaPreflightReason.SOURCE_CHANGED
    assert "became shorter" in str(exc.value)


def test_pin_source_became_longer(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    real_read = os.read

    def fake_read(fd: int, n: int) -> bytes:
        if n == 1:
            return b"X"
        return real_read(fd, n)

    with (
        patch("os.read", side_effect=fake_read),
        pytest.raises(MediaPreflightError) as exc,
    ):
        PinnedSource.create(SAMPLE_WAV, staging_root=staging, max_file_size_bytes=MAX_SIZE)
    assert exc.value.reason is MediaPreflightReason.SOURCE_CHANGED
    assert "became longer" in str(exc.value)


def test_pin_source_scratch_disappeared_during_copy(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    real_disk_usage = shutil.disk_usage
    calls = 0

    def fake_usage(path: object) -> shutil._ntuple_diskusage:
        nonlocal calls
        calls += 1
        if calls > 1:
            return shutil._ntuple_diskusage(10**12, 10**12, 0)
        return real_disk_usage(path)  # type: ignore[arg-type]

    with (
        patch("shutil.disk_usage", side_effect=fake_usage),
        pytest.raises(PreflightError, match="Scratch capacity disappeared"),
    ):
        PinnedSource.create(SAMPLE_WAV, staging_root=staging, max_file_size_bytes=MAX_SIZE)


def test_pin_source_disappeared_after_copy(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    real_lstat = os.lstat
    calls = 0

    def fake_lstat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *_args: object,
        **_kwargs: object,
    ) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls > 2 and str(SAMPLE_WAV) in str(path):
            raise FileNotFoundError("simulated vanished file")
        return real_lstat(path)

    with (
        patch("os.lstat", side_effect=fake_lstat),
        pytest.raises(MediaPreflightError) as exc,
    ):
        PinnedSource.create(SAMPLE_WAV, staging_root=staging, max_file_size_bytes=MAX_SIZE)
    assert exc.value.reason is MediaPreflightReason.SOURCE_CHANGED
    assert "disappeared" in str(exc.value)
