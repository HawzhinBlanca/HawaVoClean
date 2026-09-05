"""Targeted branch coverage tests for cross-platform matrix parity.

Directly exercises edge branches in audio/channels, assembly/validate,
assembly/stitch, finishing/eq, runtime, and platform_fs that are bypassed
when POSIX-only tests (such as POSIX process hierarchy chaos) are skipped
on Windows runners.
"""

from __future__ import annotations

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
