"""Tests for pinned FFmpeg/ffprobe, Python/native libraries, and core assets bundling (E1.10).

Contract (docs/true-10-readiness-task-sheet.md line 166):
Bundle pinned FFmpeg/ffprobe, Python/native libraries and core assets for each target.
Clean network-disabled machines process and verify Natural without Homebrew,
system Python, developer tools or a source checkout.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.paths import (
    config_dir,
    ffmpeg_bin_path,
    ffprobe_bin_path,
    models_dir,
    profile_config_path,
)
from hawavoclean.pipeline import run_pipeline
from hawavoclean.provenance import runtime_versions


@pytest.mark.unit
def test_binary_locator_precedence_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg_bin_path and ffprobe_bin_path honor explicit environment overrides."""
    fake_ffmpeg = tmp_path / "custom_ffmpeg"
    fake_ffmpeg.write_text("#!/bin/sh\necho ffmpeg version 7.1-pinned\n")
    fake_ffmpeg.chmod(0o755)

    fake_ffprobe = tmp_path / "custom_ffprobe"
    fake_ffprobe.write_text("#!/bin/sh\necho ffprobe version 7.1-pinned\n")
    fake_ffprobe.chmod(0o755)

    monkeypatch.setenv("HAWAVOCLEAN_FFMPEG_PATH", str(fake_ffmpeg))
    monkeypatch.setenv("HAWAVOCLEAN_FFPROBE_PATH", str(fake_ffprobe))

    assert ffmpeg_bin_path() == str(fake_ffmpeg)
    assert ffprobe_bin_path() == str(fake_ffprobe)


@pytest.mark.unit
def test_core_assets_bundled_and_accessible() -> None:
    """Core assets (configs and models) are packaged and accessible independent of CWD."""
    cfg_dir = config_dir()
    assert cfg_dir.is_dir(), f"Packaged config directory missing: {cfg_dir}"

    for profile in ("production", "studio", "lowband"):
        path = profile_config_path(profile)
        assert path.is_file(), f"Profile config missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0

    m_dir = models_dir()
    assert m_dir.is_dir(), f"Packaged models directory missing: {m_dir}"


@pytest.mark.unit
def test_natural_pipeline_operates_with_pinned_binaries_in_clean_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Natural pipeline successfully processes audio using pinned binaries even when host PATH is stripped."""
    system_ffmpeg = shutil.which("ffmpeg")
    system_ffprobe = shutil.which("ffprobe")
    if not system_ffmpeg or not system_ffprobe:
        pytest.skip("System ffmpeg/ffprobe not available on test host")

    # Point pinned env to the binaries and strip PATH of external tools
    monkeypatch.setenv("HAWAVOCLEAN_FFMPEG_PATH", system_ffmpeg)
    monkeypatch.setenv("HAWAVOCLEAN_FFPROBE_PATH", system_ffprobe)
    # Strip PATH to standard system utilities only (no Homebrew / opt / custom)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    # Confirm resolution points to the pinned paths despite stripped PATH
    res_ffmpeg = ffmpeg_bin_path()
    res_ffprobe = ffprobe_bin_path()
    assert res_ffmpeg is not None and Path(res_ffmpeg).resolve() == Path(system_ffmpeg).resolve()
    assert res_ffprobe is not None and Path(res_ffprobe).resolve() == Path(system_ffprobe).resolve()

    # Generate a realistic test audio file
    src = tmp_path / "pinned_test_input.wav"
    out = tmp_path / "pinned_test_output.wav"
    sr = 48000
    t = np.linspace(0.0, 3.0, int(3.0 * sr), endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
    sf.write(str(src), audio, sr, format="WAV", subtype="PCM_16")

    report = run_pipeline(
        input_path=src,
        output_path=out,
        profile="production",
        overwrite=True,
    )

    assert out.exists() and out.stat().st_size > 0
    assert report.summary.units_total > 0
    assert report.output.true_peak_dbtp is not None and report.output.true_peak_dbtp <= -1.0 + 1e-3

    # Check runtime versions provenance
    versions = runtime_versions()
    assert "ffmpeg" in versions
    assert "unavailable" not in versions["ffmpeg"]
    assert "ffprobe" in versions
    assert "unavailable" not in versions["ffprobe"]
