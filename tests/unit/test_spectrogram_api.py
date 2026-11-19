"""Unit tests for 2D STFT and delta spectrogram tile generation."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.server.spectrogram import (
    SpectrogramScale,
    compute_delta_spectrogram_tile,
    compute_spectrogram_tile,
)


def _write_tones_wav(
    path: Path,
    tones: list[tuple[float, float]],
    *,
    duration_s: float = 1.0,
    sample_rate: int = 48000,
) -> Path:
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    sig = np.zeros(n, dtype=np.float64)
    for freq_hz, amp in tones:
        sig += amp * np.sin(2.0 * math.pi * freq_hz * t)
    sf.write(str(path), np.clip(sig, -0.99, 0.99).astype(np.float32), sample_rate, subtype="PCM_24")
    return path


@pytest.mark.parametrize("scale", ["log", "mel", "linear"])
def test_compute_spectrogram_tile_scales_and_shape(tmp_path: Path, scale: SpectrogramScale) -> None:
    wav = _write_tones_wav(tmp_path / f"tone_{scale}.wav", [(1000.0, 0.5)], duration_s=1.0)
    tile = compute_spectrogram_tile(
        wav,
        start_s=0.1,
        end_s=0.9,
        width_cols=128,
        n_bins=64,
        n_fft=1024,
        scale=scale,
    )
    assert tile.n_bins == 64
    assert tile.width_cols == 128
    assert tile.scale == scale
    assert tile.sample_rate == 48000
    assert tile.f_min_hz < tile.f_max_hz <= 24000.0

    raw = base64.b64decode(tile.data_b64)
    assert len(raw) == 64 * 128
    grid = tile.decode_grid()
    assert grid.shape == (64, 128)
    assert grid.dtype == np.uint8
    # A -6 dBFS 1 kHz sine must produce bright peak pixels well above floor.
    assert int(grid.max()) > 200
    assert tile.to_dict()["scale"] == scale


def test_compute_spectrogram_tile_frequency_ordering(tmp_path: Path) -> None:
    low_wav = _write_tones_wav(tmp_path / "low.wav", [(400.0, 0.5)], duration_s=0.5)
    high_wav = _write_tones_wav(tmp_path / "high.wav", [(4000.0, 0.5)], duration_s=0.5)

    for scale in ("linear", "log", "mel"):
        low_tile = compute_spectrogram_tile(
            low_wav, start_s=0.0, end_s=0.5, width_cols=64, n_bins=128, scale=scale
        )
        high_tile = compute_spectrogram_tile(
            high_wav, start_s=0.0, end_s=0.5, width_cols=64, n_bins=128, scale=scale
        )
        low_peak_bin = int(np.argmax(low_tile.decode_grid().mean(axis=1)))
        high_peak_bin = int(np.argmax(high_tile.decode_grid().mean(axis=1)))
        assert high_peak_bin > low_peak_bin


def test_compute_spectrogram_tile_silence_and_short_window(tmp_path: Path) -> None:
    silent = _write_tones_wav(tmp_path / "silent.wav", [], duration_s=0.02)  # < n_fft
    tile = compute_spectrogram_tile(
        silent,
        start_s=0.0,
        end_s=0.5,  # intentionally exceeds duration -> clamped
        width_cols=32,
        n_bins=32,
        n_fft=2048,
        scale="log",
    )
    grid = tile.decode_grid()
    assert grid.shape == (32, 32)
    assert int(grid.max()) == 0
    assert tile.end_s <= 0.021


def test_compute_delta_spectrogram_tile_removed_energy(tmp_path: Path) -> None:
    orig = _write_tones_wav(
        tmp_path / "orig.wav",
        [(440.0, 0.3), (3000.0, 0.4)],
        duration_s=1.0,
    )
    cleaned = _write_tones_wav(
        tmp_path / "cleaned.wav",
        [(440.0, 0.3)],
        duration_s=1.0,
    )

    # Identical inputs -> zero delta everywhere.
    zero_delta = compute_delta_spectrogram_tile(
        orig, orig, start_s=0.0, end_s=1.0, width_cols=64, n_bins=64, scale="linear"
    )
    assert int(zero_delta.decode_grid().max()) == 0

    # Removing 3000 Hz produces strong delta at 3000 Hz and near-zero at 440 Hz.
    delta = compute_delta_spectrogram_tile(
        orig,
        cleaned,
        start_s=0.0,
        end_s=1.0,
        width_cols=64,
        n_bins=128,
        scale="linear",
        f_min_hz=0.0,
        f_max_hz=6000.0,
    )
    grid = delta.decode_grid()
    row_means = grid.mean(axis=1)
    peak_bin = int(np.argmax(row_means))
    # 3000 Hz is halfway between 0 and 6000 Hz -> bin ~64.
    assert 58 <= peak_bin <= 70
    assert float(row_means[peak_bin]) > 150.0
    # 440 Hz bin (~9) had no energy removed -> near zero.
    bin_440 = int(round((440.0 / 6000.0) * 127))
    assert float(row_means[bin_440]) < 15.0

    # Swapping (cleaned louder than original at 3000 Hz) clamps negative deltas to 0.
    inv_delta = compute_delta_spectrogram_tile(
        cleaned,
        orig,
        start_s=0.0,
        end_s=1.0,
        width_cols=64,
        n_bins=64,
        scale="linear",
        f_min_hz=0.0,
        f_max_hz=6000.0,
    )
    assert int(inv_delta.decode_grid().mean()) < 5


def test_compute_delta_spectrogram_tile_mismatched_sample_rates(tmp_path: Path) -> None:
    orig_16k = _write_tones_wav(
        tmp_path / "orig16.wav", [(1000.0, 0.4), (3000.0, 0.4)], duration_s=0.5, sample_rate=16000
    )
    clean_48k = _write_tones_wav(
        tmp_path / "clean48.wav", [(1000.0, 0.4)], duration_s=0.5, sample_rate=48000
    )
    tile = compute_delta_spectrogram_tile(
        orig_16k, clean_48k, start_s=0.0, end_s=0.5, width_cols=32, n_bins=32, scale="log"
    )
    assert tile.sample_rate == 48000
    assert tile.decode_grid().shape == (32, 32)
    assert int(tile.decode_grid().max()) > 100


def test_spectrogram_tile_validation(tmp_path: Path) -> None:
    wav = _write_tones_wav(tmp_path / "val.wav", [(440.0, 0.2)], duration_s=0.5)
    with pytest.raises(ValueError, match="start_s"):
        compute_spectrogram_tile(wav, start_s=-1.0, end_s=0.2)
    with pytest.raises(ValueError, match="end_s"):
        compute_spectrogram_tile(wav, start_s=0.3, end_s=0.2)
    with pytest.raises(ValueError, match="width_cols"):
        compute_spectrogram_tile(wav, start_s=0.0, end_s=0.2, width_cols=0)
    with pytest.raises(ValueError, match="n_bins"):
        compute_spectrogram_tile(wav, start_s=0.0, end_s=0.2, n_bins=0)
    with pytest.raises(ValueError, match="db_floor"):
        compute_spectrogram_tile(wav, start_s=0.0, end_s=0.2, db_floor=0.0, db_ceil=-10.0)
    with pytest.raises(ValueError, match="past the end"):
        compute_spectrogram_tile(wav, start_s=1.0, end_s=1.5)
    with pytest.raises(ValueError, match="finite"):
        compute_spectrogram_tile(wav, start_s=float("nan"), end_s=0.2)
    with pytest.raises(ValueError, match="n_fft"):
        compute_spectrogram_tile(wav, start_s=0.0, end_s=0.2, n_fft=16)
    with pytest.raises(ValueError, match="Invalid frequency bounds"):
        compute_spectrogram_tile(wav, start_s=0.0, end_s=0.2, f_min_hz=5000.0, f_max_hz=1000.0)


def test_spectrogram_tile_edge_grids_and_upsampling(tmp_path: Path) -> None:
    wav = _write_tones_wav(tmp_path / "short_up.wav", [(1000.0, 0.4)], duration_s=0.05)
    # Single frequency bin + time upsampling (n_frames < width_cols).
    tile1 = compute_spectrogram_tile(
        wav, start_s=0.0, end_s=0.05, width_cols=256, n_bins=1, n_fft=1024, scale="linear"
    )
    assert tile1.decode_grid().shape == (1, 256)

    # Dense low-freq log grid exercises narrow-bin linear interpolation branch.
    tile_dense = compute_spectrogram_tile(
        wav,
        start_s=0.0,
        end_s=0.05,
        width_cols=16,
        n_bins=128,
        n_fft=256,
        scale="log",
        f_min_hz=5.0,
        f_max_hz=100.0,
    )
    assert tile_dense.decode_grid().shape == (128, 16)

    # Reverse mismatched sample rates in delta (orig 48k, clean 16k).
    orig_48 = _write_tones_wav(
        tmp_path / "orig48.wav", [(1000.0, 0.4), (2500.0, 0.4)], duration_s=0.25, sample_rate=48000
    )
    clean_16 = _write_tones_wav(
        tmp_path / "clean16.wav", [(1000.0, 0.4)], duration_s=0.25, sample_rate=16000
    )
    delta = compute_delta_spectrogram_tile(
        orig_48, clean_16, start_s=0.0, end_s=0.25, width_cols=16, n_bins=32, scale="mel"
    )
    assert delta.sample_rate == 48000
    assert delta.decode_grid().shape == (32, 16)


def test_spectrogram_rejects_unbounded_window_duration(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from hawavoclean.server.contracts import SpectrogramRequestV1
    from hawavoclean.server.spectrogram import MAX_SPECTROGRAM_WINDOW_S

    wav = _write_tones_wav(tmp_path / "short_oom.wav", [(440.0, 0.2)], duration_s=0.5)
    assert MAX_SPECTROGRAM_WINDOW_S == 300.0

    with pytest.raises(ValueError, match="exceeds maximum"):
        compute_spectrogram_tile(wav, start_s=0.0, end_s=300.5)

    with pytest.raises(ValueError, match="exceeds maximum"):
        compute_delta_spectrogram_tile(wav, wav, start_s=10.0, end_s=310.5)

    with pytest.raises(ValidationError):
        SpectrogramRequestV1.model_validate(
            {"path": str(wav), "mode": "original", "startS": 0.0, "endS": 300.1}
        )


def test_spectrogram_internal_edge_branches(tmp_path: Path) -> None:
    from typing import cast

    from hawavoclean.server.spectrogram import (
        _compute_stft_db_grid,
        _freq_resample_matrix,
        _resample_time_axis,
    )

    wav = _write_tones_wav(tmp_path / "edge_spec.wav", [(440.0, 0.2)], duration_s=0.5)
    with pytest.raises(ValueError, match="scale must be one of"):
        compute_spectrogram_tile(
            wav,
            start_s=0.0,
            end_s=0.2,
            scale=cast(SpectrogramScale, "bogus"),
        )

    # _freq_resample_matrix boundary interpolation: fc <= stft[0], fc > stft[-1], and zero span.
    stft_freqs = np.asarray([100.0, 100.0, 200.0, 300.0], dtype=np.float64)
    target_freqs = np.asarray([50.0, 100.0, 150.0, 400.0], dtype=np.float64)
    weights = _freq_resample_matrix(stft_freqs, target_freqs, 10.0, 500.0)
    assert weights.shape == (4, 4)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(4), atol=1e-9)

    # Duplicate stft bin narrow-bin path (span <= 0.0) and right <= 0 / right >= n_stft.
    stft_dup = np.asarray([100.0, 100.0, 500.0], dtype=np.float64)
    w_dup = _freq_resample_matrix(
        stft_dup,
        np.asarray([50.0, 100.0, 600.0], dtype=np.float64),
        40.0,
        610.0,
    )
    assert w_dup[0, 0] == 1.0
    assert w_dup[2, 2] == 1.0

    # Force narrow-bin hit on duplicate stft frequency where left=0, right=1 and span==0.
    stft_same = np.asarray([100.0, 100.0], dtype=np.float64)
    w_zero_span = _freq_resample_matrix(
        stft_same,
        np.asarray([90.0, 100.0001, 110.0], dtype=np.float64),
        89.0,
        111.0,
    )
    assert w_zero_span.shape == (3, 2)

    # Exact duplicate span==0 branch: stft = [100.0, 200.0, 200.0], target bin [150..180] empty, fc=200.0?
    # Better: stft = [100.0, 100.0, 300.0], target = [100.0] with edges [100.5, 100.6] -> no mask, searchsorted=2.
    # For left=0, right=1 with span==0: searchsorted(stft, fc, side='left') == 1 requires stft[0] < fc <= stft[1]
    # which is impossible if stft[0] == stft[1], UNLESS stft is [100.0, 90.0] (non-monotonic defensive guard).
    w_defensive = _freq_resample_matrix(
        np.asarray([100.0, 90.0], dtype=np.float64),
        np.asarray([95.0, 96.0], dtype=np.float64),
        94.0,
        97.0,
    )
    assert w_defensive.shape == (2, 2)

    # _resample_time_axis: exact width match, single frame repeat, upsampling (n_frames < width_cols).
    grid_eq = np.ones((4, 8), dtype=np.float64)
    assert _resample_time_axis(grid_eq, 8) is grid_eq
    grid_one = np.full((4, 1), 2.5, dtype=np.float64)
    rep = _resample_time_axis(grid_one, 6)
    assert rep.shape == (4, 6) and np.all(rep == 2.5)
    grid_small = np.linspace(0.0, 1.0, 12, dtype=np.float64).reshape((3, 4))
    up = _resample_time_axis(grid_small, 16)
    assert up.shape == (3, 16)

    # _compute_stft_db_grid: empty mono input and long input where n_samples > n_fft * width_cols.
    db_empty = _compute_stft_db_grid(
        np.empty(0, dtype=np.float32),
        48000,
        width_cols=8,
        n_bins=4,
        n_fft=64,
        scale="linear",
        f_min_hz=0.0,
        f_max_hz=24000.0,
    )
    assert db_empty.shape == (4, 8)
    assert np.all(np.isfinite(db_empty))

    db_long = _compute_stft_db_grid(
        np.zeros(4096, dtype=np.float32),
        48000,
        width_cols=4,
        n_bins=4,
        n_fft=64,
        scale="linear",
        f_min_hz=0.0,
        f_max_hz=24000.0,
    )
    assert db_long.shape == (4, 4)
