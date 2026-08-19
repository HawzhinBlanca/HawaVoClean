"""DSP edge branches: stitch, resample, coherence, drift, limiter, encode."""

import numpy as np
import pytest

from voiceclean.alignment.coherence import estimate_coherence
from voiceclean.alignment.drift import analyze_local_drift
from voiceclean.assembly.stitch import assemble_channel_timeline
from voiceclean.audio.resample import resample_audio
from voiceclean.finishing.limiter import apply_lookahead_limiter
from voiceclean.segmentation.types import SpeechUnit

SR = 48000


def test_stitch_empty_timeline_and_length_coercion() -> None:
    assert assemble_channel_timeline([], [], 0, SR).size == 0

    unit = SpeechUnit(
        unit_id=0,
        channel_id=0,
        start_sample=0,
        end_sample=100,
        context_start_sample=0,
        context_end_sample=100,
        is_speech=True,
    )
    short = np.ones(60, dtype=np.float32)
    tl = assemble_channel_timeline([unit], [short], 100, SR)
    assert tl.size == 100 and tl[99] == 0.0  # padded

    long = np.ones(140, dtype=np.float32)
    tl2 = assemble_channel_timeline([unit], [long], 100, SR)
    assert tl2.size == 100  # truncated


def test_resample_same_rate_with_target() -> None:
    x = np.ones(100, dtype=np.float32)
    assert len(resample_audio(x, SR, SR, target_samples=150)) == 150
    assert len(resample_audio(x, SR, SR, target_samples=50)) == 50
    stereo = np.ones((2, 100), dtype=np.float32)
    up = resample_audio(stereo, 24000, SR)
    assert up.shape[0] == 2 and up.shape[1] == 200


def test_coherence_short_input_passes_trivially() -> None:
    x = np.ones(100, dtype=np.float32)
    res = estimate_coherence(x, x)
    assert res.passed and res.phase_coherence == 1.0


def test_coherence_detects_scrambled_phase() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    x = np.asarray(0.4 * np.sin(2 * np.pi * 300 * t), dtype=np.float32)
    scrambled = np.asarray(rng.permutation(x), dtype=np.float32)
    res = estimate_coherence(x, scrambled, expected_phase_coherent=True)
    assert not res.passed


def test_drift_measurement_paths() -> None:
    t = np.arange(SR * 2) / SR
    x = np.asarray(0.4 * np.sin(2 * np.pi * 220 * t), dtype=np.float32)
    res_same = analyze_local_drift(x, x, SR)
    assert res_same.max_window_drift_ms == pytest.approx(0.0, abs=1.0)

    short = np.ones(100, dtype=np.float32)
    res_short = analyze_local_drift(short, short, SR)
    assert res_short is not None


def test_limiter_trivial_branches() -> None:
    empty = np.zeros((1, 0), dtype=np.float32)
    res = apply_lookahead_limiter(empty, SR)
    assert res.limited_waveform.size == 0

    quiet = np.full((1, SR), 0.01, dtype=np.float32)
    res_q = apply_lookahead_limiter(quiet, SR)
    assert res_q.max_gain_reduction_db == pytest.approx(0.0, abs=0.01)

    x = np.zeros((1, SR), dtype=np.float32)
    x[0, SR // 2 : SR // 2 + 50] = 1.5
    res_nla = apply_lookahead_limiter(x, SR, lookahead_ms=0.0)
    assert res_nla.max_gain_reduction_db > 0.0
