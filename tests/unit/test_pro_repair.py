"""Unit tests for surgical dialogue repair and room-tone floor (pro_repair.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.signal

from hawavoclean.finishing.pro_repair import (
    ProRepairOptions,
    apply_declip,
    apply_dynamic_deplosive,
    apply_mouth_declick,
    apply_pro_repair_chain,
    apply_room_tone_floor,
)

SR = 48000


def _rms(x: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-18))


def _band_rms(x: np.ndarray[Any, np.dtype[np.float32]], low: float, high: float) -> float:
    sos = scipy.signal.butter(4, [low, high], btype="bandpass", fs=SR, output="sos")
    band = np.asarray(scipy.signal.sosfiltfilt(sos, x.astype(np.float64)), dtype=np.float64)
    return float(np.sqrt(np.mean(band * band) + 1e-18))


@pytest.mark.unit
def test_dynamic_deplosive_attenuates_plosive_preserves_voice() -> None:
    t = np.arange(SR, dtype=np.float64) / SR
    voice = 0.25 * np.sin(2.0 * np.pi * 320.0 * t) + 0.15 * np.sin(2.0 * np.pi * 1100.0 * t)

    # Inject a 70 ms 60 Hz plosive burst centered at 0.5 s.
    env = np.exp(-0.5 * ((t - 0.5) / 0.025) ** 2)
    plosive = 0.65 * env * np.sin(2.0 * np.pi * 60.0 * t)
    damaged = (voice + plosive).astype(np.float32)

    repaired, count = apply_dynamic_deplosive(
        damaged, SR, threshold_db=-18.0, max_reduction_db=10.0
    )

    assert repaired.shape == damaged.shape
    assert repaired.dtype == np.float32
    assert count >= 1

    # Measure plosive attenuation in 20-160 Hz around the burst window.
    s, e = int(0.42 * SR), int(0.58 * SR)
    before_low = _band_rms(damaged[s:e], 20.0, 160.0)
    after_low = _band_rms(repaired[s:e], 20.0, 160.0)
    reduction_db = 20.0 * np.log10(before_low / max(after_low, 1e-9))
    assert reduction_db >= 4.0

    # Voice fundamentals above 160 Hz must remain intact.
    before_voice = _band_rms(damaged, 220.0, 2000.0)
    after_voice = _band_rms(repaired, 220.0, 2000.0)
    voice_delta_db = abs(20.0 * np.log10(after_voice / max(before_voice, 1e-9)))
    assert voice_delta_db < 0.25


@pytest.mark.unit
def test_dynamic_deplosive_leaves_clean_voice_and_silence_untouched() -> None:
    t = np.arange(SR, dtype=np.float64) / SR
    clean_voice = (0.3 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.float32)
    out, count = apply_dynamic_deplosive(clean_voice, SR)
    assert count == 0
    assert np.allclose(out, clean_voice, atol=1e-6)

    silence = np.zeros(SR // 2, dtype=np.float32)
    out_sil, count_sil = apply_dynamic_deplosive(silence, SR)
    assert count_sil == 0
    assert np.array_equal(out_sil, silence)


@pytest.mark.unit
def test_mouth_declick_repairs_clicks_and_preserves_sibilant() -> None:
    t = np.arange(SR, dtype=np.float64) / SR
    clean = (
        0.22 * np.sin(2.0 * np.pi * 240.0 * t) + 0.10 * np.sin(2.0 * np.pi * 720.0 * t)
    ).astype(np.float32)
    damaged = clean.copy()

    # Click 1: sharp 2.5 ms damped saliva smack at 0.25 s.
    c1_start = int(0.25 * SR)
    c1_len = int(0.0025 * SR)
    ct = np.arange(c1_len, dtype=np.float64) / SR
    click1 = 0.65 * np.exp(-ct / 0.0007) * np.sin(2.0 * np.pi * 4500.0 * ct)
    damaged[c1_start : c1_start + c1_len] += click1.astype(np.float32)

    # Click 2: crisp transient click at 0.65 s.
    c2_pos = int(0.65 * SR)
    damaged[c2_pos] += np.float32(0.75)
    damaged[c2_pos + 1] -= np.float32(0.55)

    repaired, count = apply_mouth_declick(damaged, SR, sensitivity=0.5, max_click_ms=12.0)

    assert repaired.shape == damaged.shape
    assert repaired.dtype == np.float32
    assert count >= 2

    # Error around click regions should drop substantially compared to damaged signal.
    err_before = float(np.max(np.abs(damaged - clean)))
    err_after = float(np.max(np.abs(repaired - clean)))
    assert err_after < err_before * 0.35

    # Clean signal should produce zero false positives.
    _, clean_count = apply_mouth_declick(clean, SR, sensitivity=0.5)
    assert clean_count == 0

    # Sustained 70 ms 6 kHz sibilant must NOT be mistaken for a click.
    rng = np.random.default_rng(7)
    sib_len = int(0.070 * SR)
    sos_sib = scipy.signal.butter(4, [5000.0, 8000.0], btype="bandpass", fs=SR, output="sos")
    sib_noise = scipy.signal.sosfiltfilt(sos_sib, rng.standard_normal(sib_len))
    sib_env = np.hanning(sib_len)
    with_sib = clean.copy()
    s0 = int(0.4 * SR)
    with_sib[s0 : s0 + sib_len] += (0.18 * sib_env * sib_noise).astype(np.float32)

    sib_out, _ = apply_mouth_declick(with_sib, SR, sensitivity=0.5, max_click_ms=12.0)
    # Sibilant band energy must be preserved.
    e_before = _band_rms(with_sib[s0 : s0 + sib_len], 5000.0, 8000.0)
    e_after = _band_rms(sib_out[s0 : s0 + sib_len], 5000.0, 8000.0)
    assert e_after / max(e_before, 1e-9) > 0.85


@pytest.mark.unit
def test_declip_reconstructs_plateaus_and_enforces_headroom() -> None:
    t = np.arange(SR // 2, dtype=np.float64) / SR
    unclipped = (1.45 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.float32)
    clipped = np.clip(unclipped, -1.0, 1.0).astype(np.float32)

    repaired, count = apply_declip(clipped, SR, clip_threshold=0.985)

    assert repaired.shape == clipped.shape
    assert repaired.dtype == np.float32
    assert count > 10
    assert float(np.max(np.abs(repaired))) <= 0.99 + 1e-5

    # Correlation with original unclipped waveform should improve over flat-topped clip.
    corr_clipped = float(np.corrcoef(clipped, unclipped)[0, 1])
    corr_repaired = float(np.corrcoef(repaired, unclipped)[0, 1])
    assert corr_repaired > corr_clipped

    # Single-sample or 2-sample touch at 0.99 must NOT count as a clipped plateau.
    spike = np.zeros(1000, dtype=np.float32)
    spike[500] = 0.995
    spike[501] = 0.995
    out_spike, spike_count = apply_declip(spike, SR, clip_threshold=0.985)
    assert spike_count == 0
    assert np.array_equal(out_spike, spike)


@pytest.mark.unit
def test_room_tone_floor_explicit_and_proportional_modes() -> None:
    rng = np.random.default_rng(42)
    n = 2 * SR
    t = np.arange(n, dtype=np.float64) / SR

    # Speech burst in [0.5s, 1.2s], pauses in [0.0s, 0.45s] and [1.25s, 2.0s].
    speech = np.zeros(n, dtype=np.float64)
    s0, s1 = int(0.5 * SR), int(1.2 * SR)
    speech[s0:s1] = 0.25 * np.sin(2.0 * np.pi * 300.0 * t[s0:s1])

    # Colored room noise at ~ -34 dBFS.
    sos_lp = scipy.signal.butter(2, 1200.0, btype="lowpass", fs=SR, output="sos")
    room_noise = np.asarray(scipy.signal.sosfiltfilt(sos_lp, rng.standard_normal(n)))
    room_noise *= (10.0 ** (-34.0 / 20.0)) / max(float(np.sqrt(np.mean(room_noise**2))), 1e-9)

    original = (speech + room_noise).astype(np.float32)
    # Over-cleaned audio with -90 dB vacuum pauses.
    cleaned = speech.astype(np.float32)

    # Mode 1: explicit target_floor_db = -50 dB.
    filled = apply_room_tone_floor(original, cleaned, SR, target_floor_db=-50.0)
    pause_rms_db = 20.0 * np.log10(_rms(filled[: int(0.4 * SR)]))
    assert -53.0 <= pause_rms_db <= -47.0

    # Mode 2: proportional blend_ratio = 0.25 (~12 dB below -34 dBFS => ~-46 dBFS).
    blended = apply_room_tone_floor(original, cleaned, SR, target_floor_db=None, blend_ratio=0.25)
    blended_pause_db = 20.0 * np.log10(_rms(blended[: int(0.4 * SR)]))
    assert -49.0 <= blended_pause_db <= -43.0

    # Mode 3: bypass when both disabled.
    bypassed = apply_room_tone_floor(original, cleaned, SR, target_floor_db=None, blend_ratio=0.0)
    assert np.array_equal(bypassed, cleaned)

    # Pure silence stays silent.
    zeros = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(apply_room_tone_floor(zeros, zeros, SR, target_floor_db=-50.0), zeros)


@pytest.mark.unit
def test_dual_channel_and_pro_repair_chain() -> None:
    rng = np.random.default_rng(123)
    n = SR
    t = np.arange(n, dtype=np.float64) / SR

    speech = np.zeros(n, dtype=np.float64)
    s0, s1 = int(0.25 * SR), int(0.75 * SR)
    speech[s0:s1] = 0.25 * np.sin(2.0 * np.pi * 340.0 * t[s0:s1])
    room = 0.015 * rng.standard_normal(n)
    orig_mono = (speech + room).astype(np.float32)

    # Inject a clip, a plosive, and a mouth click on cleaned mono.
    damaged_mono = speech.astype(np.float32)
    damaged_mono[int(0.30 * SR) : int(0.30 * SR) + 8] = 1.0  # clipped plateau
    p_env = np.exp(-0.5 * ((t - 0.50) / 0.020) ** 2)
    damaged_mono += (0.55 * p_env * np.sin(2.0 * np.pi * 65.0 * t)).astype(np.float32)
    damaged_mono[int(0.65 * SR)] += np.float32(0.65)

    # Dual-mono stereo input (identical L/R).
    orig_stereo = np.stack([orig_mono, orig_mono], axis=0)
    damaged_stereo = np.stack([damaged_mono, damaged_mono], axis=0)

    opts = ProRepairOptions(room_tone_floor=True, room_tone_target_floor_db=-50.0)
    out, actions = apply_pro_repair_chain(damaged_stereo, SR, opts, original_waveform=orig_stereo)

    assert out.shape == damaged_stereo.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    # Dual-mono must remain bit-identical across L/R.
    assert np.array_equal(out[0], out[1])
    assert any(a.startswith("declip(") for a in actions)
    assert any(a.startswith("deplosive(") for a in actions)
    assert any(a.startswith("mouth_declick(") for a in actions)
    assert any(a.startswith("room_tone_floor(") for a in actions)


@pytest.mark.unit
def test_pro_repair_edge_cases_and_fail_closed() -> None:
    short = np.zeros(16, dtype=np.float32)
    assert apply_dynamic_deplosive(short, SR)[1] == 0
    assert apply_mouth_declick(short, SR)[1] == 0
    assert apply_declip(short, SR)[1] == 0
    assert np.array_equal(apply_room_tone_floor(short, short, SR), short)

    empty = np.empty(0, dtype=np.float32)
    assert apply_dynamic_deplosive(empty, SR)[0].size == 0
    assert apply_mouth_declick(empty, SR)[0].size == 0
    assert apply_declip(empty, SR)[0].size == 0
    assert apply_room_tone_floor(empty, empty, SR).size == 0

    bad_nan = np.array([0.0, np.nan, 0.1], dtype=np.float32)
    bad_inf = np.array([0.0, np.inf, 0.1], dtype=np.float32)
    for bad in (bad_nan, bad_inf):
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_dynamic_deplosive(bad, SR)
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_mouth_declick(bad, SR)
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_declip(bad, SR)
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_room_tone_floor(bad, np.zeros_like(bad), SR)

    with pytest.raises(ValueError, match="sample_rate"):
        apply_declip(short, 0)
    with pytest.raises(ValueError, match="sensitivity"):
        apply_mouth_declick(short, SR, sensitivity=1.5)
    with pytest.raises(ValueError, match="blend_ratio"):
        apply_room_tone_floor(short, short, SR, blend_ratio=0.75)
    with pytest.raises(ValueError, match="does not match"):
        apply_room_tone_floor(short, np.zeros(32, dtype=np.float32), SR)
