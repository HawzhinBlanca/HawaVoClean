"""Unit tests for voice activity detection and speech unit segmentation."""

import numpy as np
import pytest

from voiceclean.config import SegmentationConfig
from voiceclean.segmentation.utterances import build_speech_units
from voiceclean.segmentation.vad import detect_speech_energy


@pytest.mark.unit
def test_vad_silence_vs_tone() -> None:
    sr = 48000
    # 1s silence, 2s tone, 1s silence
    t = np.linspace(0, 4.0, 4 * sr, endpoint=False, dtype=np.float32)
    sig = np.zeros(4 * sr, dtype=np.float32)
    sig[sr : 3 * sr] = 0.4 * np.sin(2 * np.pi * 300 * t[sr : 3 * sr])

    intervals = detect_speech_energy(
        sig,
        sr,
        energy_threshold_rel_db=-30.0,
        min_speech_ms=150,
        pause_merge_ms=200,
    )
    assert len(intervals) >= 1
    assert intervals[0].start_sample >= int(0.8 * sr)
    assert intervals[0].end_sample <= int(3.2 * sr)


@pytest.mark.unit
def test_build_speech_units_partition_complete() -> None:
    sr = 48000
    total_len = 10 * sr
    sig = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 10, total_len, endpoint=False))).astype(
        np.float32
    )

    cfg = SegmentationConfig(
        target_speech_group_s=10.0,
        hard_max_group_s=20.0,
        context_duration_s=0.5,
    )

    units = build_speech_units(sig, sample_rate=sr, channel_id=0, config=cfg)
    assert len(units) >= 1

    # Verify no gaps in coverage
    assert units[0].start_sample == 0
    assert units[-1].end_sample == total_len

    for i in range(len(units) - 1):
        assert units[i].end_sample == units[i + 1].start_sample
