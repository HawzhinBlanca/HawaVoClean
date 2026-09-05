"""Targeted branch coverage tests for cross-platform matrix parity.

Directly exercises edge branches in audio/channels, assembly/validate,
assembly/stitch, finishing/eq, runtime, and platform_fs that are bypassed
when POSIX-only tests (such as POSIX process hierarchy chaos) are skipped
on Windows runners.
"""

from __future__ import annotations

import errno
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from hawavoclean.assembly.stitch import (
    assemble_channel_timeline,
    assemble_channel_timeline_into,
)
from hawavoclean.assembly.validate import validate_assembled_timeline
from hawavoclean.audio.channels import (
    classify_channels,
    classify_channels_bounded,
    handle_channel_layout,
)
from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.errors import (
    AmbiguousStereoError,
    InvalidUserInputError,
    OutputValidationError,
)
from hawavoclean.finishing.eq import (
    achieved_band_gains_db,
    apply_speech_eq,
    apply_tonal_restoration,
)
from hawavoclean.platform_fs import (
    flush_directory,
    replace_path,
    try_acquire_exclusive_file_lease,
)
from hawavoclean.runtime import (
    evict_memmap_pages,
    process_peak_rss_bytes,
)
from hawavoclean.segmentation.types import SpeechUnit

# ---------------------------------------------------------------------------
# audio.channels branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_channels_ambiguous_stereo_declaration_raises_invalid_user_input() -> None:
    buf = AudioBuffer(np.zeros((2, 100), dtype=np.float32), 48000)
    with pytest.raises(InvalidUserInputError, match="ambiguous_stereo"):
        classify_channels(buf, declared_mode="ambiguous_stereo")


@pytest.mark.unit
def test_channels_silence_correlation_branches() -> None:
    # Both channels silent -> correlation defaults to 1.0 -> DUAL_MONO_SAME
    buf = AudioBuffer(np.zeros((2, 1000), dtype=np.float32), 48000)
    assert classify_channels(buf, declared_mode="auto") == ChannelMode.DUAL_MONO_SAME

    # One silent, one active -> correlation defaults to 0.0 -> SPLIT_SPEAKERS (< 0.40)
    data = np.zeros((2, 1000), dtype=np.float32)
    data[0, :] = 0.1
    buf2 = AudioBuffer(data, 48000)
    assert classify_channels(buf2, declared_mode="auto") == ChannelMode.SPLIT_SPEAKERS


@pytest.mark.unit
def test_channels_bounded_chunk_validation_and_silence() -> None:
    buf = AudioBuffer(np.zeros((2, 1000), dtype=np.float32), 48000)
    with pytest.raises(ValueError, match="chunk_samples must be >= 1"):
        classify_channels_bounded(buf, chunk_samples=0)

    # Bounded silence correlation -> DUAL_MONO_SAME
    assert classify_channels_bounded(buf, chunk_samples=100) == ChannelMode.DUAL_MONO_SAME

    # Bounded one silent, one active -> SPLIT_SPEAKERS
    data = np.zeros((2, 1000), dtype=np.float32)
    data[0, :] = 0.1
    buf2 = AudioBuffer(data, 48000)
    assert classify_channels_bounded(buf2, chunk_samples=100) == ChannelMode.SPLIT_SPEAKERS


@pytest.mark.unit
def test_handle_channel_layout_unhandled_raises() -> None:
    buf = AudioBuffer(np.zeros((2, 100), dtype=np.float32), 48000)
    with pytest.raises(AmbiguousStereoError, match="Cannot process under unhandled channel mode"):
        handle_channel_layout(buf, ChannelMode.AMBIGUOUS_STEREO)


# ---------------------------------------------------------------------------
# assembly.validate branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_assembled_timeline_nan_and_inf_detection() -> None:
    data_nan = np.zeros((1, 100), dtype=np.float32)
    data_nan[0, 50] = np.nan
    buf_nan = AudioBuffer(data_nan, 48000)
    with pytest.raises(OutputValidationError, match="NaN or Infinite"):
        validate_assembled_timeline(buf_nan, 1, 100, 48000, [])

    data_inf = np.zeros((1, 100), dtype=np.float32)
    data_inf[0, 50] = np.inf
    buf_inf = AudioBuffer(data_inf, 48000)
    with pytest.raises(OutputValidationError, match="NaN or Infinite"):
        validate_assembled_timeline(buf_inf, 1, 100, 48000, [])


@pytest.mark.unit
def test_validate_assembled_timeline_gap_overlap_incomplete() -> None:
    buf = AudioBuffer(np.zeros((1, 100), dtype=np.float32), 48000)

    # Gap: span [0, 10] uncovered
    u_gap = [
        SpeechUnit(
            unit_id=1,
            channel_id=0,
            start_sample=10,
            end_sample=100,
            context_start_sample=10,
            context_end_sample=100,
            is_speech=True,
        )
    ]
    with pytest.raises(OutputValidationError, match="Timeline gap detected"):
        validate_assembled_timeline(buf, 1, 100, 48000, u_gap)

    # Overlap: span [40, 50] duplicated
    u_overlap = [
        SpeechUnit(
            unit_id=1,
            channel_id=0,
            start_sample=0,
            end_sample=50,
            context_start_sample=0,
            context_end_sample=50,
            is_speech=True,
        ),
        SpeechUnit(
            unit_id=2,
            channel_id=0,
            start_sample=40,
            end_sample=100,
            context_start_sample=40,
            context_end_sample=100,
            is_speech=True,
        ),
    ]
    with pytest.raises(OutputValidationError, match="Timeline overlap detected"):
        validate_assembled_timeline(buf, 1, 100, 48000, u_overlap)

    # Incomplete: covered 80 != 100
    u_incomplete = [
        SpeechUnit(
            unit_id=1,
            channel_id=0,
            start_sample=0,
            end_sample=80,
            context_start_sample=0,
            context_end_sample=80,
            is_speech=True,
        ),
    ]
    with pytest.raises(OutputValidationError, match="Timeline coverage incomplete"):
        validate_assembled_timeline(buf, 1, 100, 48000, u_incomplete)


# ---------------------------------------------------------------------------
# assembly.stitch branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assemble_channel_timeline_into_edge_cases() -> None:
    # 2D destination raises ValueError
    dest_2d = np.zeros((1, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="Assembly destination has shape"):
        assemble_channel_timeline_into(dest_2d, [], [], 100, 48000)

    # float64 destination raises ValueError
    dest_f64 = np.zeros(100, dtype=np.float64)
    with pytest.raises(ValueError, match="Assembly destination must be float32"):
        assemble_channel_timeline_into(dest_f64, [], [], 100, 48000)  # type: ignore[arg-type]

    # total_samples == 0 returns early
    dest_0 = np.zeros(0, dtype=np.float32)
    assemble_channel_timeline_into(dest_0, [], [], 0, 48000)
    assert len(assemble_channel_timeline([], [], 0, 48000)) == 0

    # Under-length wave (padded) and over-length wave (truncated)
    dest = np.zeros(100, dtype=np.float32)
    units = [
        SpeechUnit(
            unit_id=1,
            channel_id=0,
            start_sample=0,
            end_sample=50,
            context_start_sample=0,
            context_end_sample=50,
            is_speech=True,
        ),
        SpeechUnit(
            unit_id=2,
            channel_id=0,
            start_sample=50,
            end_sample=100,
            context_start_sample=50,
            context_end_sample=100,
            is_speech=True,
        ),
    ]
    waves = [
        np.ones(30, dtype=np.float32),  # under-length: 30 < 50
        np.ones(70, dtype=np.float32),  # over-length: 70 > 50
    ]
    assemble_channel_timeline_into(dest, units, waves, 100, 48000)
    assert dest.shape == (100,)
    assert dest[0] == 1.0
    assert dest[40] == 0.0  # padded with 0
    # Boundary declick diffuses the step between dest[49] (0.0) and wave[0] (1.0)
    # so dest[50] is adjusted; past the crossfade ramp, waveform is fully restored
    assert dest[70] == 1.0


# ---------------------------------------------------------------------------
# finishing.eq branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_speech_eq_air_shelf() -> None:
    # air_shelf_db != 0.0 with sample_rate > 22000 exercises high-shelf branch
    signal = np.ones(1024, dtype=np.float32) * 0.1
    eqed = apply_speech_eq(signal, 48000, air_shelf_db=2.0)
    assert eqed.shape == signal.shape
    assert np.all(np.isfinite(eqed))


@pytest.mark.unit
def test_achieved_band_gains_db_low_sample_rate() -> None:
    # At low sample rates (e.g. 8000 Hz), high-frequency target bands exceed Nyquist
    # which exercises the `if not np.any(mask): return 0.0` branch.
    deltas = achieved_band_gains_db(8000, 1.0, 1.0, 1.0)
    assert len(deltas) == 3
    assert all(isinstance(d, float) for d in deltas)


@pytest.mark.unit
def test_apply_tonal_restoration_branches() -> None:
    # Multi-channel raises ValueError
    multi = np.zeros((2, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="expects a single channel"):
        apply_tonal_restoration(multi, 48000, 1.0, 1.0, 1.0)

    # Short waveform (< 128) returned untouched
    short = np.zeros(64, dtype=np.float32)
    assert len(apply_tonal_restoration(short, 48000, 1.0, 1.0, 1.0)) == 64

    # Peak scaling branch: gain pushes unit above 0.999
    hot = np.ones(512, dtype=np.float32) * 0.99
    restored = apply_tonal_restoration(hot, 48000, 4.0, 4.0, 4.0)
    assert np.max(np.abs(restored)) <= 1.0

    # Empty sections (all 0 dB) returns input untouched
    normal = np.ones(256, dtype=np.float32) * 0.5
    restored_flat = apply_tonal_restoration(normal, 48000, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(restored_flat, normal)


# ---------------------------------------------------------------------------
# runtime branches (Windows memory & working set eviction)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_evict_memmap_pages_windows_path_and_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes

    mmap_path = tmp_path / "test_mmap.dat"
    mapping = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(1, 1024))

    # Simulate Windows platform
    monkeypatch.setattr(sys, "platform", "win32")

    # 1. Successful EmptyWorkingSet
    mock_kernel32 = MagicMock()
    mock_psapi = MagicMock()
    mock_kernel32.GetCurrentProcess.return_value = 12345
    mock_psapi.EmptyWorkingSet.return_value = 1  # TRUE

    mock_windll = types.SimpleNamespace(kernel32=mock_kernel32, psapi=mock_psapi)
    monkeypatch.setattr(ctypes, "windll", mock_windll, raising=False)

    evict_memmap_pages(mapping)
    assert mock_psapi.EmptyWorkingSet.called

    # 2. Fallback: EmptyWorkingSet absent, SetProcessWorkingSetSize called
    mock_windll_fallback = types.SimpleNamespace(
        kernel32=mock_kernel32, psapi=types.SimpleNamespace()
    )
    monkeypatch.setattr(ctypes, "windll", mock_windll_fallback, raising=False)
    evict_memmap_pages(mapping)
    assert mock_kernel32.SetProcessWorkingSetSize.called

    # 3. Exception in Windows eviction -> caught cleanly without raising
    mock_psapi.EmptyWorkingSet.side_effect = OSError("Windows API error")
    mock_windll_err = types.SimpleNamespace(kernel32=mock_kernel32, psapi=mock_psapi)
    monkeypatch.setattr(ctypes, "windll", mock_windll_err, raising=False)
    evict_memmap_pages(mapping)


@pytest.mark.unit
def test_process_peak_rss_bytes_windows_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    monkeypatch.setattr(sys, "platform", "win32")

    mock_kernel32 = MagicMock()
    mock_psapi = MagicMock()
    mock_kernel32.GetCurrentProcess.return_value = 12345

    def mock_get_info(_proc: Any, ptr: Any, _cb: Any) -> int:
        ptr._obj.PeakWorkingSetSize = 4096000
        return 1

    mock_psapi.GetProcessMemoryInfo.side_effect = mock_get_info
    mock_windll = types.SimpleNamespace(kernel32=mock_kernel32, psapi=mock_psapi)
    monkeypatch.setattr(ctypes, "windll", mock_windll, raising=False)

    rss = process_peak_rss_bytes()
    assert rss == 4096000

    # Error case: GetProcessMemoryInfo returns 0 -> raises OSError
    mock_psapi.GetProcessMemoryInfo.side_effect = None
    mock_psapi.GetProcessMemoryInfo.return_value = 0
    with pytest.raises(OSError, match="GetProcessMemoryInfo failed"):
        process_peak_rss_bytes()


# ---------------------------------------------------------------------------
# platform_fs branches (Windows sharing violation & locks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replace_path_windows_sharing_violation_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello", encoding="utf-8")

    # Simulate Windows environment
    monkeypatch.setattr("hawavoclean.platform_fs._platform_system", lambda: "windows")

    # First attempt raises winerror 32 (sharing violation), second attempt succeeds
    attempts = [0]
    real_replace = Path.replace

    def flaky_replace(self: Path, target: Path) -> Path:
        if attempts[0] == 0:
            attempts[0] += 1
            exc = PermissionError(
                13,
                "The process cannot access the file because it is being used by another process.",
            )
            exc.winerror = 32  # type: ignore[attr-defined]
            raise exc
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    replace_path(src, dst)
    assert dst.read_text(encoding="utf-8") == "hello"


@pytest.mark.unit
def test_flush_directory_windows_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hawavoclean.platform_fs._platform_name", lambda: "nt")
    # Windows flush_directory is an immediate return without error
    flush_directory(tmp_path)


@pytest.mark.unit
def test_open_safe_lock_and_lease_roundtrip(tmp_path: Path) -> None:
    lock_file = tmp_path / "test.lock"
    lease = try_acquire_exclusive_file_lease(lock_file)
    try:
        # Re-acquiring while locked raises BlockingIOError / TimeoutError
        with pytest.raises((BlockingIOError, TimeoutError)):
            try_acquire_exclusive_file_lease(lock_file)
    finally:
        lease.release()


# ---------------------------------------------------------------------------
# platform_fs extra branches (locking, Windows move, atomic renames)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_open_safe_lock_identity_mismatch_and_reparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hawavoclean.platform_fs import _open_safe_lock

    # 1. Target is a directory (not a regular file) -> raises OSError(ELOOP)
    dir_target = tmp_path / "lock_dir"
    dir_target.mkdir()
    with pytest.raises(OSError) as exc_info:
        _open_safe_lock(dir_target)
    assert exc_info.value.errno == errno.ELOOP

    # 2. If after is reparse point / symlink
    target = tmp_path / "lock.txt"
    target.write_text("ok", encoding="utf-8")
    monkeypatch.setattr("hawavoclean.platform_fs._is_reparse_or_symlink", lambda _stat: True)
    with pytest.raises(OSError) as exc_info2:
        _open_safe_lock(target)
    assert exc_info2.value.errno == errno.ELOOP


@pytest.mark.unit
def test_windows_move_failure_and_flush_rename_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes

    from hawavoclean.platform_fs import _flush_rename_directories, _windows_move

    src = tmp_path / "sub1" / "file.txt"
    dst = tmp_path / "sub2" / "file.txt"
    src.parent.mkdir()
    dst.parent.mkdir()
    src.write_text("x", encoding="utf-8")

    # _flush_rename_directories cross-directory
    monkeypatch.setattr("hawavoclean.platform_fs._platform_name", lambda: "posix")
    with monkeypatch.context() as m:
        m.setattr("hawavoclean.platform_fs.flush_directory", MagicMock())
        _flush_rename_directories(src, dst)

    # _windows_move failure raises ctypes.WinError
    mock_kernel32 = MagicMock()
    mock_kernel32.MoveFileExW.return_value = 0  # FALSE
    mock_windll = MagicMock(return_value=mock_kernel32)
    monkeypatch.setattr(ctypes, "WinDLL", mock_windll, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError", OSError, raising=False)
    mock_wintypes = types.SimpleNamespace(LPCWSTR=str, DWORD=int, BOOL=int)
    monkeypatch.setitem(sys.modules, "ctypes.wintypes", mock_wintypes)
    with pytest.raises(OSError):
        _windows_move(src, dst, replace=True)


# ---------------------------------------------------------------------------
# paths branches (cross-platform app_data_root, ffmpeg/ffprobe binaries)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_paths_app_data_root_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hawavoclean.paths import app_data_root

    # 1. State dir env override
    monkeypatch.setenv("HAWAVOCLEAN_STATE_DIR", "/custom/state")
    assert app_data_root() == Path("/custom/state").resolve()
    monkeypatch.delenv("HAWAVOCLEAN_STATE_DIR")

    # 2. Windows with LOCALAPPDATA
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\Test\\AppData\\Local")
    assert "HawaVoClean" in str(app_data_root())

    # 3. Windows without LOCALAPPDATA
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert "HawaVoClean" in str(app_data_root())

    # 4. Darwin
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "Application Support" in str(app_data_root())

    # 5. Linux with XDG_DATA_HOME
    monkeypatch.setattr(sys, "platform", "linux")
    custom_xdg = str(tmp_path / "custom_xdg")
    monkeypatch.setenv("XDG_DATA_HOME", custom_xdg)
    assert custom_xdg in str(app_data_root())

    # 6. Linux without XDG_DATA_HOME
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert ".local" in str(app_data_root())


@pytest.mark.unit
def test_paths_binary_resolution_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hawavoclean.paths import ffmpeg_bin_path, ffprobe_bin_path, resolve_calibration_file

    fake_bin = tmp_path / "fake_ffmpeg"
    fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_bin.chmod(0o755)

    # 1. Env override
    monkeypatch.setenv("HAWAVOCLEAN_FFMPEG_PATH", str(fake_bin))
    assert ffmpeg_bin_path() == str(fake_bin)
    monkeypatch.setenv("HAWAVOCLEAN_FFPROBE_PATH", str(fake_bin))
    assert ffprobe_bin_path() == str(fake_bin)

    # 2. Missing binary fallback to shutil.which
    monkeypatch.delenv("HAWAVOCLEAN_FFMPEG_PATH")
    monkeypatch.delenv("HAWAVOCLEAN_FFPROBE_PATH")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert ffmpeg_bin_path() == "/usr/bin/ffmpeg"
    assert ffprobe_bin_path() == "/usr/bin/ffprobe"

    # 3. resolve_calibration_file: absolute vs relative
    abs_p = (tmp_path / "calib.json").resolve()
    assert resolve_calibration_file(str(abs_p)) == abs_p
    rel_p = "calib.json"
    assert resolve_calibration_file(rel_p).name == "calib.json"


# ---------------------------------------------------------------------------
# audio.probe metadata validation branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audio_probe_metadata_validators() -> None:
    from hawavoclean.audio.probe import (
        MAX_METADATA_INTEGER,
        _count_samples_by_decoding,
        _metadata_float,
        _metadata_int,
        _metadata_text,
    )
    from hawavoclean.errors import MediaPreflightError

    # int validator
    assert _metadata_int(42, "field") == 42
    assert _metadata_int("100", "field") == 100
    with pytest.raises(MediaPreflightError, match="not an integer"):
        _metadata_int(True, "field")
    with pytest.raises(MediaPreflightError, match="not a valid integer"):
        _metadata_int("bad", "field")
    with pytest.raises(MediaPreflightError, match="outside the supported range"):
        _metadata_int(-1, "field", minimum=0)
    with pytest.raises(MediaPreflightError, match="outside the supported range"):
        _metadata_int(MAX_METADATA_INTEGER + 1, "field")

    # float validator
    assert _metadata_float(3.14, "field") == pytest.approx(3.14)
    with pytest.raises(MediaPreflightError, match="not a valid number"):
        _metadata_float(False, "field")
    with pytest.raises(MediaPreflightError, match="must be finite"):
        _metadata_float("nan", "field")
    with pytest.raises(MediaPreflightError, match="must be finite"):
        _metadata_float("inf", "field")

    # text validator
    assert _metadata_text("valid", "field") == "valid"
    with pytest.raises(MediaPreflightError, match="missing or malformed"):
        _metadata_text("", "field")
    with pytest.raises(MediaPreflightError, match="missing or malformed"):
        _metadata_text(123, "field")
    with pytest.raises(MediaPreflightError, match="control characters"):
        _metadata_text("line\x00break", "field")

    # _count_samples_by_decoding with None binary
    assert _count_samples_by_decoding(Path("f.wav"), None, 48000, 0, None) == 0


# ---------------------------------------------------------------------------
# audio.decode edge checks & window bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audio_decode_window_bounds_and_checks() -> None:
    from hawavoclean.audio.decode import _check_decoded, window_sample_bounds
    from hawavoclean.audio.types import AudioProbeResult

    probe = AudioProbeResult(
        path=Path("sample.wav"),
        channels=2,
        sample_rate=48000,
        samples=480000,
        duration_s=10.0,
        format_name="wav",
        codec_name="pcm_s16le",
        bit_depth=16,
        sha256="0" * 64,
    )

    # Valid window
    assert window_sample_bounds(probe, 1.0, 2.0) == (48000, 96000)

    # Invalid windows
    with pytest.raises(InvalidUserInputError, match="must be finite"):
        window_sample_bounds(probe, float("nan"), 2.0)
    with pytest.raises(InvalidUserInputError, match="must be >= 0"):
        window_sample_bounds(probe, -1.0, 2.0)
    with pytest.raises(InvalidUserInputError, match="must be greater than start_s"):
        window_sample_bounds(probe, 2.0, 1.0)
    with pytest.raises(InvalidUserInputError, match="past the end of"):
        window_sample_bounds(probe, 12.0, 15.0)

    # _check_decoded
    with pytest.raises(InvalidUserInputError, match="NaN or Infinite"):
        _check_decoded(np.array([[np.nan]], dtype=np.float32), Path("a.wav"))
    with pytest.raises(InvalidUserInputError, match="NaN or Infinite"):
        _check_decoded(np.array([[np.inf]], dtype=np.float32), Path("a.wav"))
    with pytest.raises(InvalidUserInputError, match="abnormal float amplitude"):
        _check_decoded(np.array([[100.0]], dtype=np.float32), Path("a.wav"))


# ---------------------------------------------------------------------------
# watchdog branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_watchdog_signal_and_parent_alive_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal

    from hawavoclean.watchdog import (
        _self_interrupt_signal,
        install_parent_death_watchdog,
        parent_is_alive,
    )

    # 1. Signals: Win32 returns SIGTERM
    monkeypatch.setattr(sys, "platform", "win32")
    assert _self_interrupt_signal() == signal.SIGTERM

    # POSIX with SIGINT ignored returns SIGTERM
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(signal, "getsignal", lambda _sig: signal.SIG_IGN)
    assert _self_interrupt_signal() == signal.SIGTERM

    # POSIX default returns SIGINT
    monkeypatch.setattr(signal, "getsignal", lambda _sig: signal.default_int_handler)
    assert _self_interrupt_signal() == signal.SIGINT

    # 2. parent_is_alive reparenting check
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getppid", lambda: 1)
    assert not parent_is_alive(99999)  # reparented to 1 != 99999

    # 3. install_parent_death_watchdog invalid configs
    monkeypatch.delenv("HAWAVOCLEAN_PARENT_PID", raising=False)
    assert install_parent_death_watchdog() is None

    monkeypatch.setenv("HAWAVOCLEAN_PARENT_PID", "invalid")
    assert install_parent_death_watchdog() is None

    monkeypatch.setenv("HAWAVOCLEAN_PARENT_PID", "1")  # <= 1
    assert install_parent_death_watchdog() is None

    monkeypatch.setenv("HAWAVOCLEAN_PARENT_PID", str(os.getpid()))  # self
    assert install_parent_death_watchdog() is None

    monkeypatch.setenv("HAWAVOCLEAN_PARENT_PID", "99999")
    monkeypatch.setattr(os, "getppid", lambda: 22222)
    assert install_parent_death_watchdog() is None


# ---------------------------------------------------------------------------
# audio.encode branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audio_encode_branches(tmp_path: Path) -> None:
    from hawavoclean.audio.encode import (
        _finalize_deterministic_wav,
        _wav_container_format,
        encode_audio,
        encode_audio_streaming,
    )
    from hawavoclean.audio.types import AudioBuffer

    # Container format: large sample count picks RF64
    assert _wav_container_format(2, 1000, "FLOAT") == "WAV"
    assert _wav_container_format(2, 10**9, "FLOAT") == "RF64"

    # _finalize_deterministic_wav errors
    invalid_header_file = tmp_path / "bad_hdr.wav"
    invalid_header_file.write_bytes(b"NOT_A_WAV_FILE_HEADER")
    with pytest.raises(OutputValidationError, match="not a valid WAV/RF64"):
        _finalize_deterministic_wav(invalid_header_file)

    truncated_chunk_file = tmp_path / "trunc.wav"
    truncated_chunk_file.write_bytes(b"RIFF\x24\0\0\0WAVEfmt ")  # missing chunk size
    with pytest.raises(OutputValidationError, match="truncated chunk header"):
        _finalize_deterministic_wav(truncated_chunk_file)

    malformed_peak_file = tmp_path / "malformed_peak.wav"
    malformed_peak_file.write_bytes(b"RIFF\x24\0\0\0WAVEPEAK\x04\0\0\0xxxxdata\0\0\0\0")
    with pytest.raises(OutputValidationError, match="malformed PEAK chunk"):
        _finalize_deterministic_wav(malformed_peak_file)

    # encode_audio validation: NaN and empty
    nan_buf = AudioBuffer(np.array([[np.nan]], dtype=np.float32), 48000)
    with pytest.raises(OutputValidationError, match="NaN"):
        encode_audio(nan_buf, tmp_path / "nan.wav")

    empty_buf = AudioBuffer(np.zeros((1, 0), dtype=np.float32), 48000)
    with pytest.raises(OutputValidationError, match="Cannot encode empty audio buffer"):
        encode_audio(empty_buf, tmp_path / "empty.wav")

    ok_buf = AudioBuffer(np.zeros((1, 100), dtype=np.float32), 48000)

    # Valid float32 encoding without dither
    out_f32 = encode_audio(
        ok_buf, tmp_path / "out_f32.wav", output_bit_depth="float32", dither=False
    )
    assert out_f32.is_file()

    # encode_audio_streaming validation
    with pytest.raises(ValueError, match="chunk_samples must be >= 1"):
        encode_audio_streaming(ok_buf, tmp_path / "stream_bad.wav", chunk_samples=0)

    with pytest.raises(OutputValidationError, match="Cannot encode empty audio buffer"):
        encode_audio_streaming(empty_buf, tmp_path / "stream_empty.wav")

    with pytest.raises(OutputValidationError, match="NaN"):
        encode_audio_streaming(nan_buf, tmp_path / "stream_nan.wav")

    out_stream = encode_audio_streaming(ok_buf, tmp_path / "stream_ok.wav", chunk_samples=50)
    assert out_stream.is_file()


# ---------------------------------------------------------------------------
# eval.corpus & eval.corruption branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eval_corpus_and_corruption_branches(tmp_path: Path) -> None:
    import hashlib
    import json

    from hawavoclean.eval.corpus import load_corpus_manifest, verify_corpus_audio_files
    from hawavoclean.eval.corruption import (
        corrupt_consonant_splice,
        corrupt_dropout,
        corrupt_repeated_span,
    )

    # 1. Corpus manifest loading
    missing = tmp_path / "absent.json"
    with pytest.raises(InvalidUserInputError, match="Manifest file not found"):
        load_corpus_manifest(missing)

    item_dict = {
        "id": "item1",
        "audio_path": "a.wav",
        "audio_sha256": "fake_sha",
        "duration_s": 1.0,
        "speaker_id": "spk_test",
        "dialect": "slemani",
        "gender": "male",
        "environment": "studio",
        "degradation_type": "clean",
        "transcript_sorani": "تێست",
        "split": "acceptance",
    }

    # Full JSON manifest with "items"
    full_manifest_json = tmp_path / "full.json"
    full_manifest_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "full",
                "split_name": "acceptance",
                "items_count": 1,
                "manifest_sha256": "abc",
                "items": [item_dict],
            }
        ),
        encoding="utf-8",
    )
    m_full = load_corpus_manifest(full_manifest_json)
    assert m_full.items_count == 1

    # Single-item JSON
    single_json = tmp_path / "single.json"
    single_json.write_text(json.dumps(item_dict), encoding="utf-8")
    manifest = load_corpus_manifest(single_json)
    assert manifest.items_count == 1

    # JSONL file format (not starting with '{')
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text(
        " \n" + json.dumps(item_dict) + "\n",
        encoding="utf-8",
    )
    m_jsonl = load_corpus_manifest(jsonl_file)
    assert m_jsonl.items_count == 1

    # verify_corpus_audio_files: missing file and sha mismatch
    with pytest.raises(InvalidUserInputError, match="audio missing"):
        verify_corpus_audio_files(manifest, base_dir=tmp_path)

    # Create matching file with different sha
    dummy_wav = tmp_path / "a.wav"
    dummy_wav.write_bytes(b"content")
    with pytest.raises(InvalidUserInputError, match="SHA-256 mismatch"):
        verify_corpus_audio_files(manifest, base_dir=tmp_path)

    # Successful verification with matching sha
    real_sha = hashlib.sha256(b"content").hexdigest()
    item_dict_valid = dict(item_dict, audio_sha256=real_sha)
    single_valid_json = tmp_path / "single_valid.json"
    single_valid_json.write_text(json.dumps(item_dict_valid), encoding="utf-8")
    valid_manifest = load_corpus_manifest(single_valid_json)
    verify_corpus_audio_files(valid_manifest, base_dir=tmp_path)

    # 2. Corruption out-of-bounds boundary returns copy
    wave = np.ones(100, dtype=np.float32)
    spliced = corrupt_consonant_splice(wave, 16000, start_time_s=10.0)
    np.testing.assert_array_equal(wave, spliced)

    repeated = corrupt_repeated_span(wave, 16000, start_time_s=10.0)
    np.testing.assert_array_equal(wave, repeated)

    dropout = corrupt_dropout(wave, 16000, start_time_s=0.0, duration_ms=1.0)
    assert dropout[0] == 0.0


# ---------------------------------------------------------------------------
# journal & cli branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_journal_and_cli_exit_branches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from hawavoclean.cli import exit_with_code
    from hawavoclean.errors import ExitCode
    from hawavoclean.journal import JobJournal, JournalEvent

    # 1. JobJournal read_events on missing and corrupted trailing line
    j_path = tmp_path / "job.journal"
    journal = JobJournal(j_path)
    assert journal.read_events() == []

    journal.append(JournalEvent.JOB_STARTED, {"key": "val"})
    journal.append(JournalEvent.UNIT_COMMITTED, {"unit_id": 42})
    journal.append(JournalEvent.UNIT_COMMITTED, {})  # unit_id missing

    # Append corrupt line
    with open(j_path, "a", encoding="utf-8") as f:
        f.write("{invalid_json\n")

    events = journal.read_events()
    assert len(events) == 3
    assert journal.get_committed_units() == {42}

    # 2. cli exit_with_code branches
    with pytest.raises(SystemExit) as exc1:
        exit_with_code(ExitCode.SUCCESS, "All good")
    assert exc1.value.code == 0
    captured1 = capsys.readouterr()
    assert "All good" in captured1.out

    with pytest.raises(SystemExit) as exc2:
        exit_with_code(ExitCode.PREFLIGHT_FAILURE, "Failed badly")
    assert exc2.value.code == int(ExitCode.PREFLIGHT_FAILURE)
    captured2 = capsys.readouterr()
    assert "Failed badly" in captured2.err

    with pytest.raises(SystemExit) as exc3:
        exit_with_code(0)
    assert exc3.value.code == 0


# ---------------------------------------------------------------------------
# hashing & cli diagnostic command branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hashing_branches(tmp_path: Path) -> None:
    from hawavoclean.hashing import compute_cache_key, compute_job_id, hash_bytes, hash_numpy

    # 1. hash_bytes
    assert len(hash_bytes(b"hello")) == 64

    # 2. hash_numpy with Fortran-contiguous array
    arr_c = np.ones((10, 10), dtype=np.float32)
    arr_f = np.asfortranarray(arr_c)
    assert hash_numpy(arr_c) == hash_numpy(arr_f)

    # 3. hash_numpy with memmap
    mmap_file = tmp_path / "test.mmap"
    mmap = np.memmap(mmap_file, dtype=np.float32, mode="w+", shape=(100,))
    mmap[:] = 1.0
    mmap.flush()
    assert len(hash_numpy(mmap)) == 64
    del mmap

    # 4. compute_job_id with and without restore_context
    id1 = compute_job_id("inp", "cfg", "core", "guard", "1.0.0")
    id2 = compute_job_id("inp", "cfg", "core", "guard", "1.0.0", restore_context="speaker1")
    assert id1 != id2
    assert len(id1) == 16
    assert len(id2) == 16

    # 5. compute_cache_key
    ck = compute_cache_key(b"pcm", 48000, {"m": "h"}, "guard", "cfg", "1.0")
    assert len(ck) == 64


@pytest.mark.unit
def test_cli_doctor_and_profile_commands(tmp_path: Path) -> None:
    import argparse

    from hawavoclean.cli import cmd_restore_doctor, cmd_speaker_profile
    from hawavoclean.errors import ExitCode
    from hawavoclean.paths import profiles_root

    # 1. cmd_restore_doctor execution
    res = cmd_restore_doctor(argparse.Namespace())
    assert res in (int(ExitCode.SUCCESS), int(ExitCode.PREFLIGHT_FAILURE))

    # 2. cmd_speaker_profile on empty dir -> INVALID_USER_INPUT
    empty_dir = tmp_path / "empty_profiles"
    empty_dir.mkdir()
    assert cmd_speaker_profile(argparse.Namespace(profile_target=str(empty_dir))) == int(
        ExitCode.INVALID_USER_INPUT
    )

    # 3. cmd_speaker_profile on real profiles root
    real_profiles = profiles_root()
    if real_profiles.exists():
        assert cmd_speaker_profile(argparse.Namespace(profile_target=str(real_profiles))) == int(
            ExitCode.SUCCESS
        )


# ---------------------------------------------------------------------------
# contracts, source_caps, paths fallbacks, store, bandwidth & checkpoint branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_server_contracts_validator_branches() -> None:
    from hawavoclean.server.contracts import (
        JobStatusResponseV1,
        ManualStrategyV1,
        ProcessingRecordEvidenceV1,
        ProcessingRequestV1,
        SmartSafeStrategyV1,
    )

    # 1. SmartSafeStrategyV1 disabled with profile_id
    with pytest.raises(ValueError, match="speakerProfileId is invalid"):
        SmartSafeStrategyV1(
            kind="smart_safe",
            restore_policy="disabled",
            speaker_profile_id="spk_1",
            allow_generative_reconstruction=False,
        )

    # 2. ManualStrategyV1 natural with allow_generative_reconstruction
    with pytest.raises(ValueError, match="generative reconstruction consent is invalid"):
        ManualStrategyV1(
            kind="manual",
            route="production",
            allow_generative_reconstruction=True,
        )

    # 3. ManualStrategyV1 natural with expert_cutoff_hz
    with pytest.raises(ValueError, match="expertCutoffHz is valid only"):
        ManualStrategyV1(
            kind="manual",
            route="production",
            expert_cutoff_hz=4000.0,
            allow_generative_reconstruction=False,
        )

    # 4. ProcessingRequestV1 with empty sourceId
    with pytest.raises(ValueError, match="every sourceId must contain"):
        ProcessingRequestV1(
            schema_version=1,
            source_ids=[""],
            strategy=ManualStrategyV1(kind="manual", route="production"),
            execution_policy="offline_only",
            idempotency_key="key1",
        )

    # 5. JobStatusResponseV1 bundle validations
    base_job = {
        "job_id": "job1",
        "state": "queued",
        "stage": "pending",
        "progress": 0.0,
        "message": "Queued",
        "output_path": "/out.wav",
        "report_path": "/rep.json",
        "created_at": "2026-09-05T00:00:00Z",
    }
    with pytest.raises(ValueError, match="recordBundle jobs require bundlePath"):
        JobStatusResponseV1(
            **base_job,  # type: ignore[arg-type]
            record_bundle=True,
            bundle_path=None,
        )

    with pytest.raises(ValueError, match="bundle evidence is invalid"):
        JobStatusResponseV1(
            **base_job,  # type: ignore[arg-type]
            record_bundle=False,
            bundle_path="/path/to/bundle.zip",
        )

    dummy_sha = "0" * 64
    bundle_ev = ProcessingRecordEvidenceV1(
        path="/path/b.zip",
        archive_sha256=dummy_sha,
        content_sha256=dummy_sha,
        master_sha256=dummy_sha,
        report_sha256=dummy_sha,
        summary_sha256=dummy_sha,
        total_uncompressed_bytes=100,
        internal_hashes_verified=True,
        authenticated_publisher=True,
    )
    with pytest.raises(ValueError, match="bundle evidence path must equal"):
        JobStatusResponseV1(
            **base_job,  # type: ignore[arg-type]
            record_bundle=True,
            bundle_path="/path/a.zip",
            bundle=bundle_ev,
        )


@pytest.mark.unit
def test_server_source_caps_registry_branches(tmp_path: Path) -> None:
    from hawavoclean.server.policy import PathPolicyError
    from hawavoclean.server.source_caps import (
        NativeSourceRegistry,
        resolve_native_selected_path,
    )

    # 1. resolve_native_selected_path empty or bad text
    with pytest.raises(PathPolicyError, match="selected source path is required"):
        resolve_native_selected_path("   ")

    # 2. NativeSourceRegistry resolution checks
    test_file = tmp_path / "test.wav"
    test_file.write_bytes(b"content")

    with NativeSourceRegistry(maximum=1) as registry:
        # Invalid hex or length
        assert registry.resolve_source("short") is None
        assert registry.resolve_source("z" * 32) is None
        assert registry.resolve_source("0" * 32) is None

        # Authorizes un-registered path
        assert not registry.authorizes(tmp_path / "other.wav")

        # Register valid file
        src1 = registry.register(str(test_file))
        assert registry.authorizes(test_file)
        assert registry.resolve_source(src1.source_id) == test_file.resolve()

        # Eviction at maximum capacity
        file2 = tmp_path / "file2.wav"
        file2.write_bytes(b"content2")
        src2 = registry.register(str(file2))
        assert registry.resolve_source(src2.source_id) == file2.resolve()
        # src1 was evicted
        assert registry.resolve_source(src1.source_id) is None

        # Tampered / deleted file on resolve
        file2.unlink()
        assert registry.resolve_source(src2.source_id) is None


@pytest.mark.unit
def test_paths_binary_resolution_full_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hawavoclean.paths import (
        ffmpeg_bin_path,
        restoration_checkpoint_path,
    )

    # 1. restoration_checkpoint_path: env override and packaged path
    fake_ckpt = tmp_path / "fake.pt"
    fake_ckpt.write_bytes(b"dummy")
    monkeypatch.setenv("HAWAVOCLEAN_RESTORATION_CHECKPOINT", str(fake_ckpt))
    assert restoration_checkpoint_path() == fake_ckpt.resolve()

    monkeypatch.delenv("HAWAVOCLEAN_RESTORATION_CHECKPOINT")
    pkg_dir = tmp_path / "hawarestore-kd"
    pkg_dir.mkdir()
    (pkg_dir / "hawarestore_kd.pt").write_bytes(b"dummy")
    monkeypatch.setattr("hawavoclean.paths.models_dir", lambda: tmp_path)
    assert restoration_checkpoint_path() == (pkg_dir / "hawarestore_kd.pt").resolve()

    # 2. ffmpeg fallback bins: pkg_bin and prefix_bin
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    monkeypatch.delenv("HAWAVOCLEAN_FFMPEG_PATH", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_FFPROBE_PATH", raising=False)

    fake_pkg_bin = tmp_path / "resources" / "bin" / exe_name
    fake_pkg_bin.parent.mkdir(parents=True)
    fake_pkg_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_pkg_bin.chmod(0o755)

    monkeypatch.setattr("hawavoclean.paths._PACKAGE_ROOT", tmp_path)
    assert ffmpeg_bin_path() == str(fake_pkg_bin)
    fake_pkg_bin.unlink()

    fake_prefix_bin = tmp_path / "prefix" / "bin" / exe_name
    fake_prefix_bin.parent.mkdir(parents=True)
    fake_prefix_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_prefix_bin.chmod(0o755)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    assert ffmpeg_bin_path() == str(fake_prefix_bin)


@pytest.mark.unit
def test_model_packs_store_inspect_branches(tmp_path: Path) -> None:
    from hawavoclean.model_packs.store import ModelPackStore
    from hawavoclean.model_packs.trust import TrustStore

    store = ModelPackStore(tmp_path / "packs")
    trust = TrustStore([])

    # Invalid pack_id pattern
    cap_bad = store.inspect("INVALID!PACK", trust)
    assert not cap_bad.usable
    assert cap_bad.reason_code == "invalid_pack_id"

    # Pack not installed
    cap_none = store.inspect("valid_pack_id", trust)
    assert not cap_none.usable
    assert cap_none.reason_code == "pack_not_installed"


@pytest.mark.unit
def test_bandwidth_detector_extended_branches() -> None:
    from hawavoclean.restoration.bandwidth import BandwidthDetector

    # 1. 16 kHz sample rate (no bins >= 18 kHz)
    det16 = BandwidthDetector(sample_rate=16000)
    sig16 = np.sin(2 * np.pi * 400 * np.linspace(0, 0.5, 8000)).astype(np.float32)
    est16 = det16.detect(sig16)
    assert est16.effective_cutoff_hz > 0

    # 2. Short speech mask with active frames <= 5
    mask_short = np.zeros(100, dtype=np.float32)
    mask_short[:2] = 1.0
    sig_short = np.zeros(100 * 256, dtype=np.float32)
    det_mask = BandwidthDetector(sample_rate=48000)
    est_mask = det_mask.detect(sig_short, speech_mask=mask_short)
    assert est_mask.confidence >= 0.0

    # 3. min_cutoff_hz > 16 kHz forces restore_recommended=False and fullband fallback
    det_high = BandwidthDetector(sample_rate=48000, min_cutoff_hz=17000.0, max_cutoff_hz=20000.0)
    sig_high = np.ones(48000, dtype=np.float32)
    est_high = det_high.detect(sig_high)
    assert not est_high.restore_recommended
    assert est_high.shape == "fullband"


@pytest.mark.unit
def test_checkpoint_validation_extended_branches(tmp_path: Path) -> None:
    import torch

    from hawavoclean.errors import ModelProvenanceError
    from hawavoclean.restoration.checkpoint import load_safe_checkpoint

    # 1. Missing model_state_dict
    bad_pt1 = tmp_path / "bad1.pt"
    torch.save({"config": {}}, bad_pt1)
    with pytest.raises(ModelProvenanceError, match="missing 'model_state_dict'"):
        load_safe_checkpoint(bad_pt1)

    # 2. model_state_dict not a dict
    bad_pt2 = tmp_path / "bad2.pt"
    torch.save({"model_state_dict": "not_a_dict"}, bad_pt2)
    with pytest.raises(ModelProvenanceError, match="is not a dictionary"):
        load_safe_checkpoint(bad_pt2)

    # 3. model_state_dict value not a tensor
    bad_pt3 = tmp_path / "bad3.pt"
    torch.save({"model_state_dict": {"w": [1.0, 2.0]}}, bad_pt3)
    with pytest.raises(ModelProvenanceError, match="is not a torch.Tensor"):
        load_safe_checkpoint(bad_pt3)

    # 4. model_state_dict tensor containing NaN
    bad_pt4 = tmp_path / "bad4.pt"
    torch.save(
        {"model_state_dict": {"w": torch.tensor([float("nan")], dtype=torch.float32)}},
        bad_pt4,
    )
    with pytest.raises(ModelProvenanceError, match="contains non-finite values"):
        load_safe_checkpoint(bad_pt4)

    # 5. .safetensors with corrupted metadata json sidecar
    safe_file = tmp_path / "model.safetensors"
    safe_file.write_bytes(b"dummy")
    sidecar = tmp_path / "model_metadata.json"
    sidecar.write_text("{corrupt_json", encoding="utf-8")
    with pytest.raises(
        ModelProvenanceError,
        match="Failed to load safetensors|Corrupted metadata",
    ):
        load_safe_checkpoint(safe_file)
