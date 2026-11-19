from __future__ import annotations

import ctypes
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hawavoclean.runtime import _windows_peak_rss_bytes, process_peak_rss_bytes

# --- 1. Runtime Windows Peak RSS Coverage ---


def test_windows_peak_rss_bytes_mocked() -> None:
    fake_windll = MagicMock()
    fake_kernel32 = MagicMock()
    fake_psapi = MagicMock()

    fake_windll.kernel32 = fake_kernel32
    fake_windll.psapi = fake_psapi

    def fake_get_process_memory_info(_proc: Any, counters_ref: Any, _cb: Any) -> int:
        counters = counters_ref._obj
        counters.PeakWorkingSetSize = 104857600
        return 1

    fake_psapi.GetProcessMemoryInfo.side_effect = fake_get_process_memory_info

    with patch.dict(vars(ctypes), {"windll": fake_windll}):
        peak = _windows_peak_rss_bytes()
        assert peak == 104857600

    # Test error branch: clear side_effect so return_value is used
    fake_psapi.GetProcessMemoryInfo.side_effect = None
    fake_psapi.GetProcessMemoryInfo.return_value = 0
    with (
        patch.dict(vars(ctypes), {"windll": fake_windll}),
        pytest.raises(OSError, match="GetProcessMemoryInfo failed"),
    ):
        _windows_peak_rss_bytes()


def test_process_peak_rss_bytes_windows_dispatch() -> None:
    with (
        patch("sys.platform", "win32"),
        patch("hawavoclean.runtime._windows_peak_rss_bytes", return_value=123456),
    ):
        assert process_peak_rss_bytes() == 123456
