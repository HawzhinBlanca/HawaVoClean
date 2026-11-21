"""Unit tests for subsonic filtering, de-hum notch filtering, and transient click repair."""

from __future__ import annotations

import numpy as np

from hawavoclean.finishing.repair import (
    remove_dc_subsonic,
    remove_electrical_hum,
    repair_transient_clicks,
)


def test_remove_dc_subsonic() -> None:
    # 1. Short input
    short_sig = np.zeros(32, dtype=np.float32)
    assert len(remove_dc_subsonic(short_sig, 48000)) == 32

    # 2. DC offset + 1 kHz signal
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sig_with_dc = (0.5 * np.sin(2 * np.pi * 1000 * t) + 0.3).astype(np.float32)
    filtered = remove_dc_subsonic(sig_with_dc, sr, cutoff_hz=20.0)
    assert abs(np.mean(filtered[sr // 4 : -sr // 4])) < 1e-2


def test_remove_electrical_hum() -> None:
    # 1. Short input
    short_sig = np.zeros(32, dtype=np.float32)
    assert len(remove_electrical_hum(short_sig, 48000)) == 32

    # 2. 50 Hz hum + harmonics
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    hum = (0.4 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    filtered = remove_electrical_hum(hum, sr, hum_freq_hz=50.0, num_harmonics=4)
    assert float(np.max(np.abs(filtered[sr // 4 : -sr // 4]))) < float(np.max(np.abs(hum))) * 0.2


def test_repair_transient_clicks() -> None:
    # 1. Short input
    short_sig = np.zeros(16, dtype=np.float32)
    out, count = repair_transient_clicks(short_sig)
    assert len(out) == 16
    assert count == 0

    # 2. Signal with isolated spike / click
    sig = np.zeros(1000, dtype=np.float32)
    sig[500] = 0.9  # Isolated loud transient click
    repaired, count = repair_transient_clicks(sig, threshold_sigma=3.0)
    assert count >= 1
    assert repaired[500] < 0.5
