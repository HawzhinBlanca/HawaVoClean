from __future__ import annotations

from pathlib import Path

import pytest

from hawavoclean.audio.probe import (
    MAX_METADATA_INTEGER,
    MAX_METADATA_TEXT_CHARS,
    _metadata_float,
    _metadata_int,
    _metadata_text,
    _run_bounded_capture,
    probe_audio,
)
from hawavoclean.errors import MediaPreflightError, MediaPreflightReason


def test_run_bounded_capture_invalid_limits() -> None:
    with pytest.raises(ValueError, match="probe capture limits and timeout must be positive"):
        _run_bounded_capture(["echo"], timeout_s=-1)

    with pytest.raises(ValueError, match="probe capture limits and timeout must be positive"):
        _run_bounded_capture(["echo"], timeout_s=1, stdout_limit=0)

    with pytest.raises(ValueError, match="probe capture limits and timeout must be positive"):
        _run_bounded_capture(["echo"], timeout_s=1, stderr_limit=0)


def test_metadata_int_validation() -> None:
    assert _metadata_int(42, "sample_rate") == 42
    assert _metadata_int("48000", "sample_rate") == 48000

    # Non-integer / float / bool rejected
    with pytest.raises(MediaPreflightError) as exc:
        _metadata_int(True, "field")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_int(12.34, "field")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_int("not_a_number", "field")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    # Outside supported range (< minimum or > max)
    with pytest.raises(MediaPreflightError) as exc:
        _metadata_int(-5, "field", minimum=0)
    assert exc.value.reason is MediaPreflightReason.RESOURCE_BOMB

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_int(MAX_METADATA_INTEGER + 100, "field")
    assert exc.value.reason is MediaPreflightReason.RESOURCE_BOMB


def test_metadata_float_validation() -> None:
    assert _metadata_float(12.5, "duration") == 12.5
    assert _metadata_float("3.14", "duration") == 3.14

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_float(False, "duration")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_float("invalid_float", "duration")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_float(float("nan"), "duration")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_float(float("inf"), "duration")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA


def test_metadata_text_validation() -> None:
    assert _metadata_text("pcm_s16le", "codec") == "pcm_s16le"

    # Empty or wrong type or too long
    with pytest.raises(MediaPreflightError) as exc:
        _metadata_text("", "codec")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_text(12345, "codec")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    with pytest.raises(MediaPreflightError) as exc:
        _metadata_text("a" * (MAX_METADATA_TEXT_CHARS + 1), "codec")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA

    # Control characters
    with pytest.raises(MediaPreflightError) as exc:
        _metadata_text("codec\x00bad", "codec")
    assert exc.value.reason is MediaPreflightReason.MALFORMED_METADATA


def test_probe_audio_filesystem_checks(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(MediaPreflightError) as exc:
        probe_audio(missing)
    assert exc.value.reason is MediaPreflightReason.NOT_FOUND

    # Directory
    d = tmp_path / "somedir"
    d.mkdir()
    with pytest.raises(MediaPreflightError) as exc:
        probe_audio(d)
    assert exc.value.reason is MediaPreflightReason.NOT_REGULAR_FILE

    # Empty file
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(MediaPreflightError) as exc:
        probe_audio(empty)
    assert exc.value.reason is MediaPreflightReason.EMPTY_FILE
