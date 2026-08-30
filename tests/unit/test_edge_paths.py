"""Edge and error paths: config validation, paths overrides, calibration
loader, encode/resample branches, loudness short-file, safe-finish ladder."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean import paths
from hawavoclean.audio.encode import encode_audio
from hawavoclean.audio.resample import resample_audio
from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.config import GuardConfig, load_config
from hawavoclean.errors import CalibrationError, ConfigError
from hawavoclean.finishing.loudness import compute_static_master_gain, measure_loudness_and_peaks
from hawavoclean.finishing.safe_finish import safe_finish_speech_unit
from hawavoclean.guard.calibration import load_calibration_artifact
from hawavoclean.guard.spectral_probe import FixedProbe

SR = 48000


# ---- paths overrides ----------------------------------------------------


def test_paths_env_overrides(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CONFIG_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "w"))
    monkeypatch.setenv("HAWAVOCLEAN_STATE_DIR", str(tmp_path / "state"))
    assert paths.config_dir() == (tmp_path / "c").resolve()
    assert paths.models_dir() == (tmp_path / "m").resolve()
    assert paths.work_root() == (tmp_path / "w").resolve()
    assert paths.app_data_root() == (tmp_path / "state").resolve()
    assert paths.job_store_path() == (tmp_path / "state").resolve() / "state" / "jobs.sqlite3"
    assert paths.profile_config_path("production").name == "production.toml"
    assert paths.resolve_calibration_file("x.json") == (tmp_path / "m").resolve() / "x.json"
    absolute = tmp_path / "abs.json"
    assert paths.resolve_calibration_file(str(absolute)) == absolute


def test_paths_defaults_inside_package(monkeypatch: Any) -> None:
    monkeypatch.delenv("HAWAVOCLEAN_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_MODEL_DIR", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_WORK_DIR", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_STATE_DIR", raising=False)
    assert paths.config_dir().exists()
    assert paths.models_dir().exists()
    assert paths.work_root().name == "work"
    assert paths.job_store_path().name == "jobs.sqlite3"


def test_app_data_root_platform_contract(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv("HAWAVOCLEAN_STATE_DIR", raising=False)
    monkeypatch.setattr("hawavoclean.paths.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert paths.app_data_root() == tmp_path / "Local" / "HawaVoClean"

    monkeypatch.setattr("hawavoclean.paths.sys.platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert paths.app_data_root() == tmp_path / "xdg" / "hawavoclean"


# ---- config validation --------------------------------------------------


def test_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.toml")


def test_config_bad_toml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text("[runtime\ndevelopment = maybe")
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_invalid_field_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad2.toml"
    p.write_text("[policy]\nstrength_ladder = [2.0]\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_production_refuses_development_flag(tmp_path: Path) -> None:
    p = tmp_path / "dev.toml"
    p.write_text("[runtime]\ndevelopment = true\n")
    with pytest.raises(ConfigError):
        load_config(p, is_production=True)
    load_config(p, is_production=False)  # allowed in development


# ---- calibration loader -------------------------------------------------


def test_calibration_missing_and_invalid(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError):
        load_calibration_artifact(tmp_path / "absent.json")

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(CalibrationError):
        load_calibration_artifact(bad)

    incomplete = tmp_path / "inc.json"
    incomplete.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(CalibrationError):
        load_calibration_artifact(incomplete)


# ---- encode / resample branches ----------------------------------------


def test_encode_float32_subtype(tmp_path: Path) -> None:
    buf = AudioBuffer(
        data=(0.1 * np.ones((1, 4800), dtype=np.float32)),
        sample_rate=SR,
        channel_mode=ChannelMode.MONO,
    )
    out = encode_audio(buf, tmp_path / "f32.wav", output_bit_depth="float32", dither=False)
    info = sf.info(str(out))
    assert info.subtype == "FLOAT"
    assert info.frames == 4800


def test_resample_identity_and_target_length() -> None:
    x = np.random.default_rng(0).standard_normal(4800).astype(np.float32)
    same = resample_audio(x, SR, SR)
    assert len(same) == len(x)

    up = resample_audio(x, SR, 96000, target_samples=9600)
    assert len(up) == 9600
    down_padded = resample_audio(x, SR, 16000, target_samples=1700)
    assert len(down_padded) == 1700


# ---- loudness edge branches ---------------------------------------------


def test_loudness_short_file_branch() -> None:
    short = np.zeros((1, 1000), dtype=np.float32)
    m = measure_loudness_and_peaks(short, SR)
    assert m.integrated_lufs == -70.0

    short_loud = 0.5 * np.ones((1, 1000), dtype=np.float32)
    m2 = measure_loudness_and_peaks(short_loud, SR)
    assert m2.sample_peak_dbfs > -7.0


def test_static_gain_silence_and_backoff() -> None:
    assert compute_static_master_gain(-70.0, -19.0, -30.0) == 0.0
    # Projected peak beyond limiter budget must back off the static gain.
    g = compute_static_master_gain(
        measured_lufs=-30.0,
        target_lufs=-16.0,
        current_true_peak_dbtp=-2.0,
        true_peak_ceiling_dbtp=-1.0,
        max_limiter_reduction_db=2.5,
    )
    assert g < 14.0  # full gain would be +14 dB; the cap must reduce it


# ---- safe finish ladder branches ----------------------------------------


def _tone(seconds: float = 2.0) -> np.ndarray[Any, np.dtype[np.float32]]:
    t = np.arange(int(SR * seconds)) / SR
    x = 0.3 * np.sin(2 * np.pi * 200 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return np.asarray(x, dtype=np.float32)


def test_safe_finish_disabled_bypasses() -> None:
    from hawavoclean.config import FinishingConfig

    res, _ = safe_finish_speech_unit(
        pre_finish_waveform=_tone(),
        sample_rate=SR,
        is_speech=True,
        probe=FixedProbe(),
        finishing_config=FinishingConfig(enabled=False),
        guard_config=GuardConfig(),
    )
    assert res.preset_applied == "bypass"
    assert res.actions_taken == []


def test_safe_finish_non_speech_bypasses() -> None:
    from hawavoclean.config import FinishingConfig

    res, _ = safe_finish_speech_unit(
        pre_finish_waveform=_tone(),
        sample_rate=SR,
        is_speech=False,
        probe=FixedProbe(),
        finishing_config=FinishingConfig(enabled=True),
        guard_config=GuardConfig(),
    )
    assert res.preset_applied == "bypass"
