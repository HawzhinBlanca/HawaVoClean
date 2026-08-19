"""Unit tests for deterministic local finishing stages and Guard B ladder."""

import numpy as np
import pytest

from voiceclean.config import FinishingConfig, GuardConfig
from voiceclean.finishing.deess import apply_split_band_deesser
from voiceclean.finishing.repair import (
    remove_dc_subsonic,
    remove_electrical_hum,
    repair_transient_clicks,
)
from voiceclean.finishing.safe_finish import safe_finish_speech_unit
from voiceclean.guard.spectral_probe import FixedProbe
from voiceclean.guard.verdict import GuardVerdict


@pytest.mark.unit
def test_remove_dc_subsonic() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 0.2 DC offset + 5Hz rumble + 400Hz voice
    wave = 0.2 + 0.1 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)
    cleaned = remove_dc_subsonic(wave.astype(np.float32), sr, cutoff_hz=20.0)
    assert abs(float(np.mean(cleaned))) < 0.01


@pytest.mark.unit
def test_remove_electrical_hum() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    hum = 0.2 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    cleaned = remove_electrical_hum(hum, sr, hum_freq_hz=50.0)
    assert float(np.sqrt(np.mean(cleaned**2))) < 0.05


@pytest.mark.unit
def test_repair_clicks() -> None:
    wave = np.zeros(2000, dtype=np.float32)
    wave[500] = 0.8  # click spike
    repaired, count = repair_transient_clicks(wave)
    assert count >= 1
    assert abs(float(repaired[500])) < 0.1


@pytest.mark.unit
def test_deesser_gain_reduction_bounded() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    harsh_sibilance = 0.8 * np.sin(2 * np.pi * 7000 * t).astype(np.float32)
    out, max_gr = apply_split_band_deesser(harsh_sibilance, sr, max_reduction_db=4.0)
    assert max_gr <= 4.01
    assert float(np.max(np.abs(out))) < float(np.max(np.abs(harsh_sibilance)))


@pytest.mark.unit
def test_safe_finish_speech_unit_gentle_pass() -> None:
    sr = 48000
    # Use realistic multi-tone harmonic speech unit
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    wave = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 1000 * t)
        + 0.1 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)
    asr = FixedProbe()
    fin_cfg = FinishingConfig(enabled=True, preset="gentle")
    grd_cfg = GuardConfig()

    res, _ = safe_finish_speech_unit(
        wave,
        sr,
        is_speech=True,
        probe=asr,
        finishing_config=fin_cfg,
        guard_config=grd_cfg,
    )
    assert res.guard_b_verdict in (GuardVerdict.PASS, GuardVerdict.NO_SPEECH)
