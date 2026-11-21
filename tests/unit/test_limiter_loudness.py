"""Unit tests for BS.1770 loudness normalization and true-peak limiting."""

from pathlib import Path

import numpy as np
import pytest

from hawavoclean.finishing.limiter import apply_lookahead_limiter
from hawavoclean.finishing.loudness import compute_static_master_gain, measure_loudness_and_peaks


@pytest.mark.unit
def test_measure_loudness_and_peaks() -> None:
    sr = 48000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False, dtype=np.float32)
    # Sine wave with 0.5 amplitude
    sig = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    stereo = np.stack([sig, sig], axis=0)

    res = measure_loudness_and_peaks(stereo, sr)
    assert -15.0 <= res.integrated_lufs <= -5.0
    assert res.sample_peak_dbfs == pytest.approx(-6.02, abs=0.5)
    assert res.true_peak_dbtp <= 0.0


@pytest.mark.unit
def test_compute_static_master_gain() -> None:
    gain = compute_static_master_gain(
        measured_lufs=-22.0,
        target_lufs=-16.0,
        current_true_peak_dbtp=-5.0,
        true_peak_ceiling_dbtp=-1.0,
        max_limiter_reduction_db=2.5,
    )
    # Needed gain is +6.0 dB. Projected peak = -5.0 + 6.0 = +1.0 dBTP.
    # Max allowable projected peak = -1.0 + 2.5 = +1.5 dBTP.
    # So full +6.0 dB is allowed.
    assert gain == pytest.approx(6.0, abs=0.1)


@pytest.mark.unit
def test_limiter_enforces_ceiling() -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Hot signal exceeding 1.0
    hot = (1.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    stereo = np.stack([hot, hot], axis=0)

    res = apply_lookahead_limiter(stereo, sr, ceiling_dbtp=-1.0)
    ceiling_linear = 10.0 ** (-1.0 / 20.0)  # ~0.89125
    assert float(np.max(np.abs(res.limited_waveform))) <= ceiling_linear + 1e-4
    assert res.max_gain_reduction_db > 0.0


@pytest.mark.unit
def test_streaming_loudness_and_memmap_limiter(tmp_path: Path) -> None:
    from hawavoclean.errors import OutputValidationError
    from hawavoclean.finishing.limiter import apply_lookahead_limiter_to_memmap
    from hawavoclean.finishing.loudness import (
        StreamingLoudnessMeter,
        measure_loudness_and_peaks_streaming,
    )

    sr = 48000
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
    sig = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.stack([sig, sig], axis=0)

    # 1. Compare streaming loudness vs whole-buffer loudness
    m_direct = measure_loudness_and_peaks(stereo, sr)
    m_stream = measure_loudness_and_peaks_streaming(stereo, sr, chunk_samples=4096)
    assert m_direct.integrated_lufs == pytest.approx(m_stream.integrated_lufs, abs=0.2)
    assert m_direct.sample_peak_dbfs == pytest.approx(m_stream.sample_peak_dbfs, abs=0.1)

    # 2. Short audio (< 400ms)
    short_audio = stereo[:, : int(0.2 * sr)]
    m_short = measure_loudness_and_peaks_streaming(short_audio, sr)
    assert m_short.integrated_lufs < 0.0

    # 3. Streaming meter error cases
    meter = StreamingLoudnessMeter(sr, 2)
    with pytest.raises(ValueError, match="shape"):
        meter.push(stereo[0:1, :])
    with pytest.raises(ValueError, match="NaN or Infinite"):
        bad_chunk = stereo.copy()
        bad_chunk[0, 0] = np.nan
        meter.push(bad_chunk)

    # 4. apply_lookahead_limiter_to_memmap
    out_memmap = tmp_path / "limiter_stage.pcm"
    res_memmap = apply_lookahead_limiter_to_memmap(
        stereo,
        sr,
        out_memmap,
        ceiling_dbtp=-1.0,
        chunk_samples=2048,
    )
    assert res_memmap.ceiling_dbtp == -1.0
    assert out_memmap.is_file()

    # Destination already exists
    with pytest.raises(OutputValidationError, match="already exists"):
        apply_lookahead_limiter_to_memmap(stereo, sr, out_memmap)

    # Invalid waveform shape
    with pytest.raises(ValueError, match="shape"):
        apply_lookahead_limiter_to_memmap(stereo[0], sr, tmp_path / "stage2.pcm")
