"""Segmentation edge coverage: forced cuts, zero-crossing search, VAD edges."""

from typing import Any

import numpy as np

from voiceclean.config import SegmentationConfig
from voiceclean.segmentation.utterances import (
    build_speech_units,
    find_lowest_energy_zero_crossing,
)
from voiceclean.segmentation.vad import detect_speech_energy

SR = 48000


def _speechlike(seconds: float, seed: int = 0) -> np.ndarray[Any, np.dtype[np.float32]]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t))
    x += 0.01 * rng.standard_normal(len(t))
    return np.asarray(x, dtype=np.float32)


# ---- zero-crossing cut search -------------------------------------------


def test_zero_crossing_search_finds_low_energy_point() -> None:
    x = _speechlike(2.0)
    x[SR - 200 : SR + 200] *= 0.001  # a quiet notch near the 1s mark
    cut = find_lowest_energy_zero_crossing(x, SR, search_window_samples=SR // 2)
    assert abs(cut - SR) < SR // 2


def test_zero_crossing_search_degenerate_window() -> None:
    x = _speechlike(0.01)
    cut = find_lowest_energy_zero_crossing(x, 5, search_window_samples=0)
    assert 0 <= cut <= len(x)


def test_zero_crossing_search_no_crossings_falls_back() -> None:
    x = np.full(SR, 0.5, dtype=np.float32)  # strictly positive: no crossings
    cut = find_lowest_energy_zero_crossing(x, SR // 2, search_window_samples=1000)
    assert 0 <= cut <= SR


# ---- forced boundary on very long continuous speech ---------------------


def test_continuous_speech_beyond_hard_max_is_force_cut() -> None:
    cfg = SegmentationConfig(
        target_speech_group_s=5.0,
        hard_max_group_s=10.0,
        min_speech_ms=100,
    )
    x = _speechlike(14.0)  # continuous voiced content > hard_max
    units = build_speech_units(
        channel_waveform=x, sample_rate=SR, channel_id=0, config=cfg, start_unit_id=0
    )
    speech_units = [u for u in units if u.is_speech]
    assert len(speech_units) >= 2, "a 14s continuous span must be split"
    assert any(u.forced_boundary for u in units), "the split must be marked forced"
    # Timeline must stay contiguous and complete
    assert units[0].start_sample == 0
    assert units[-1].end_sample == len(x)
    for a, b in zip(units, units[1:], strict=False):
        assert a.end_sample == b.start_sample


def test_empty_and_silent_channels_yield_no_speech() -> None:
    cfg = SegmentationConfig()
    assert (
        build_speech_units(
            channel_waveform=np.empty(0, dtype=np.float32),
            sample_rate=SR,
            channel_id=0,
            config=cfg,
            start_unit_id=0,
        )
        == []
    )
    silent = np.zeros(SR * 2, dtype=np.float32)
    units = build_speech_units(
        channel_waveform=silent, sample_rate=SR, channel_id=0, config=cfg, start_unit_id=0
    )
    assert units and all(not u.is_speech for u in units)


# ---- VAD edges ----------------------------------------------------------


def test_vad_empty_and_tiny_inputs() -> None:
    assert detect_speech_energy(np.empty(0, dtype=np.float32), SR) == []

    tiny_loud = np.full(100, 0.5, dtype=np.float32)  # shorter than one frame
    assert detect_speech_energy(tiny_loud, SR) != []

    tiny_quiet = np.zeros(100, dtype=np.float32)
    assert detect_speech_energy(tiny_quiet, SR) == []


def test_vad_trailing_speech_and_short_burst_filtering() -> None:
    # Speech that runs to the very end exercises the tail-flush branch.
    x = np.zeros(SR, dtype=np.float32)
    x[SR // 2 :] = _speechlike(0.5)[: SR - SR // 2]
    intervals = detect_speech_energy(x, SR)
    assert intervals and intervals[-1].end_sample == SR

    # A 40ms blip is below min_speech_ms and must be discarded.
    y = np.zeros(SR, dtype=np.float32)
    y[1000 : 1000 + SR // 25] = 0.4
    assert detect_speech_energy(y, SR, min_speech_ms=150) == []
