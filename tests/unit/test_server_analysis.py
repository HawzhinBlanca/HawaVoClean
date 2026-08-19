"""``/api/analyze`` maths: waveform buckets, sine-calibrated 1/12-octave
spectrum, noise floor, loudness plumbing."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.server.analysis import (
    FLOOR_DB,
    MAX_BUCKETS,
    analyze_audio,
    average_power_spectrum,
    band_centres,
    band_integrate,
    bucket_edges,
    waveform_overview,
)

pytestmark = pytest.mark.unit


def test_band_centres_cover_40hz_to_nyquist_or_20k() -> None:
    c48 = band_centres(48000)
    assert c48[0] == pytest.approx(40.0)
    assert c48[-1] <= 20000.0 and c48[-1] * 2 ** (1 / 12) > 20000.0
    ratios = c48[1:] / c48[:-1]
    assert np.allclose(ratios, 2 ** (1 / 12))
    c16 = band_centres(16000)
    assert c16[-1] <= 8000.0 and c16[-1] * 2 ** (1 / 12) > 8000.0
    assert len(c16) < len(c48)


def test_full_scale_sine_at_band_centre_reads_zero_db() -> None:
    sr = 48000
    centres = band_centres(sr)
    idx = 56  # ~1016 Hz
    t = np.arange(sr * 2) / sr
    x = np.sin(2 * np.pi * centres[idx] * t).astype(np.float32)
    power = average_power_spectrum(x)
    bands = band_integrate(power, sr, 8192, centres)
    db = 10 * np.log10(bands)
    assert db[idx] == pytest.approx(0.0, abs=0.1)
    others = np.delete(db, [idx - 1, idx, idx + 1])
    assert np.max(others) < -40.0
    # A -20 dBFS sine reads -20 dB.
    quiet = (0.1 * x).astype(np.float32)
    db_q = 10 * np.log10(band_integrate(average_power_spectrum(quiet), sr, 8192, centres))
    assert db_q[idx] == pytest.approx(-20.0, abs=0.1)


def test_full_scale_sine_at_low_band_centres_reads_zero_db() -> None:
    """Below ~400 Hz a 1/12-octave band is narrower than the FFT main lobe;
    the integration window widens so the contract calibration rule (sine at
    a band centre = 0 dB) still holds at every band."""
    sr = 48000
    centres = band_centres(sr)
    t = np.arange(sr * 2) / sr
    for idx in (0, 12, 24):  # 40 Hz, 80 Hz, 160 Hz
        x = np.sin(2 * np.pi * centres[idx] * t).astype(np.float32)
        bands = band_integrate(average_power_spectrum(x), sr, 8192, centres)
        assert 10 * np.log10(bands[idx]) == pytest.approx(0.0, abs=0.1), centres[idx]


def test_silence_and_short_input_clamp_to_floor() -> None:
    sr = 16000
    centres = band_centres(sr)
    short = np.zeros(100, dtype=np.float32)  # shorter than one FFT frame: zero-padded
    power = average_power_spectrum(short)
    assert power.shape == (8192 // 2 + 1,)
    assert np.all(power == 0.0)
    from hawavoclean.server.analysis import _db

    assert np.all(_db(band_integrate(power, sr, 8192, centres)) == FLOOR_DB)


def test_bucket_edges_cover_every_sample_and_never_empty() -> None:
    starts, ends = bucket_edges(1000, 7)
    assert starts[0] == 0 and ends[-1] == 1000
    assert np.all(ends > starts)
    assert np.all(starts[1:] == ends[:-1])
    # Fewer samples than buckets: overlapping one-sample buckets, still n values.
    starts, ends = bucket_edges(3, 10)
    assert len(starts) == 10 and np.all(ends - starts >= 1) and np.all(ends <= 3)


def test_waveform_overview_values() -> None:
    x = np.concatenate([np.full(100, 0.5), np.full(100, -0.25), np.zeros(100)]).astype(np.float32)
    mins, maxs, rms = waveform_overview(x, 3)
    assert maxs.tolist() == [0.5, -0.25, 0.0]
    assert mins.tolist() == [0.5, -0.25, 0.0]
    assert rms[0] == pytest.approx(20 * np.log10(0.5))
    assert rms[1] == pytest.approx(20 * np.log10(0.25))
    assert rms[2] == FLOOR_DB
    # n < buckets path
    mins, maxs, rms = waveform_overview(np.asarray([0.1, -0.2], dtype=np.float32), 5)
    assert len(mins) == len(maxs) == len(rms) == 5
    assert set(maxs.tolist()) <= {0.1, -0.2} or np.allclose(np.sort(np.unique(maxs)), [-0.2, 0.1])


def _write(path: Path, data: np.ndarray, sr: int) -> Path:
    sf.write(str(path), data, sr)
    return path


def test_analyze_audio_shape_and_sanity(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr * 2) / sr
    loud = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig = np.concatenate([loud, np.zeros(sr)]).astype(np.float32)  # 2 s tone + 1 s silence
    path = _write(tmp_path / "tone.wav", sig, sr)

    a = analyze_audio(path, buckets=30)
    assert a["path"] == str(path)
    assert a["sample_rate"] == sr and a["channels"] == 1
    assert a["duration_s"] == pytest.approx(3.0, abs=1e-3)
    assert len(a["peaks"]["min"]) == len(a["peaks"]["max"]) == len(a["rms_db"]) == 30
    assert all(-1.0 <= v <= 1.0 for v in a["peaks"]["min"] + a["peaks"]["max"])
    assert max(a["peaks"]["max"]) == pytest.approx(0.5, abs=0.01)
    assert min(a["peaks"]["min"]) == pytest.approx(-0.5, abs=0.01)
    # First two thirds are the tone (~ -9 dBFS rms), last third silence (-120).
    assert a["rms_db"][0] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.2)
    assert a["rms_db"][-1] == FLOOR_DB
    assert all(FLOOR_DB <= v <= 0.0 for v in a["rms_db"])
    freqs = a["spectrum"]["freqs_hz"]
    db = a["spectrum"]["db"]
    assert len(freqs) == len(db) == len(band_centres(sr))
    assert all(FLOOR_DB <= v <= 6.0 for v in db)
    assert freqs[int(np.argmax(db))] == pytest.approx(440.0, rel=0.06)
    assert a["loudness"]["integrated_lufs"] < 0.0
    assert a["loudness"]["true_peak_dbtp"] == pytest.approx(20 * np.log10(0.5), abs=0.3)
    # Noise floor: 10th percentile of live buckets -> the tone level here
    assert a["noise_floor_db"] == pytest.approx(a["rms_db"][0], abs=0.5)


def test_analyze_audio_mono_mix_and_noise_floor(tmp_path: Path) -> None:
    sr = 16000
    rng = np.random.default_rng(3)
    n = sr * 2
    left = 0.2 * np.sin(2 * np.pi * 300 * np.arange(n) / sr)
    right = -left  # cancels in the mono mix
    stereo = np.column_stack([left, right]).astype(np.float32)
    stereo[: sr // 2] += (0.001 * rng.standard_normal((sr // 2, 2))).astype(np.float32)
    path = _write(tmp_path / "st.wav", stereo, sr)
    a = analyze_audio(path, buckets=20)
    assert a["channels"] == 2
    # Mono mix of an anti-phase pair is (almost) silent except the noisy quarter.
    assert max(a["peaks"]["max"]) < 0.01
    live = [v for v in a["rms_db"] if v > FLOOR_DB]
    assert live, "the noisy quarter must register"
    assert a["noise_floor_db"] == pytest.approx(float(np.percentile(live, 10)), abs=0.01)
    assert a["noise_floor_db"] < -50.0


def test_analyze_audio_all_silence_noise_floor_is_floor(tmp_path: Path) -> None:
    path = _write(tmp_path / "silence.wav", np.zeros(16000, dtype=np.float32), 16000)
    a = analyze_audio(path, buckets=10)
    assert a["noise_floor_db"] == FLOOR_DB
    assert set(a["rms_db"]) == {FLOOR_DB}
    assert set(a["spectrum"]["db"]) == {FLOOR_DB}


def test_analyze_audio_rejects_bad_buckets(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.wav", np.zeros(1600, dtype=np.float32), 16000)
    with pytest.raises(ValueError):
        analyze_audio(path, buckets=0)
    with pytest.raises(ValueError):
        analyze_audio(path, buckets=MAX_BUCKETS + 1)
    a = analyze_audio(path, buckets=MAX_BUCKETS)  # more buckets than samples is fine
    assert len(a["rms_db"]) == MAX_BUCKETS
