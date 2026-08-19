"""``decode_audio_window``: a time window costs a window, not a file.

The load-bearing property is that the windowed decode returns *exactly* the
same samples the full decode would have returned over the same span — on both
the ffmpeg path and the soundfile fallback — because the waveform the UI draws
after a zoom must line up with the one it drew before it.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.decode import decode_audio, decode_audio_window, window_sample_bounds
from hawavoclean.audio.probe import probe_audio
from hawavoclean.errors import InvalidUserInputError

pytestmark = pytest.mark.unit

SR = 48000


def _noise_wav(path: Path, seconds: float = 6.0, sr: int = SR, channels: int = 1) -> Path:
    """Broadband noise: every sample differs from its neighbours, so any
    off-by-one in the seek shows up immediately."""
    rng = np.random.default_rng(7)
    n = int(seconds * sr)
    data = (0.3 * rng.standard_normal((n, channels))).astype(np.float32)
    sf.write(str(path), data if channels > 1 else data[:, 0], sr, subtype="FLOAT")
    return path


@pytest.fixture()
def wav(tmp_path: Path) -> Path:
    return _noise_wav(tmp_path / "noise.wav")


def _full(path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
    return decode_audio(probe_audio(path, max_sample_rate=384000)).data


# ---------------------------------------------------------------- sample bounds


def test_window_sample_bounds_rounds_onto_the_sample_grid(wav: Path) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    assert window_sample_bounds(probe, 1.0, 2.0) == (SR, 2 * SR)
    # 1.234567 s * 48000 = 59259.216 -> 59259
    assert window_sample_bounds(probe, 1.234567, 1.734567) == (59259, 83259)
    # end beyond the file is clamped to the file length
    assert window_sample_bounds(probe, 5.0, 99.0) == (5 * SR, 6 * SR)


def test_window_sample_bounds_rejects_unusable_ranges(wav: Path) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    for start, end in [
        (float("nan"), 1.0),
        (0.0, float("inf")),
        (-0.5, 1.0),
        (2.0, 2.0),
        (2.0, 1.0),
        (6.0, 7.0),  # at the end of a 6 s file
        (60.0, 61.0),  # past it
    ]:
        with pytest.raises(InvalidUserInputError):
            window_sample_bounds(probe, start, end)


# ------------------------------------------------------- windowed == full slice


@pytest.mark.parametrize(
    ("start_s", "end_s"),
    [(0.0, 1.0), (1.0, 3.0), (1.234567, 2.5), (4.9, 6.0), (2.0, 2.001)],
)
def test_window_matches_the_full_decode_over_the_same_span(
    wav: Path, start_s: float, end_s: float
) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    full = _full(wav)
    buf = decode_audio_window(probe, start_s, end_s)
    start, end = window_sample_bounds(probe, start_s, end_s)
    assert buf.sample_rate == SR
    assert buf.samples == end - start
    np.testing.assert_array_equal(buf.data, full[:, start:end])


def test_window_matches_full_decode_on_the_soundfile_fallback(
    monkeypatch: pytest.MonkeyPatch, wav: Path
) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    full = _full(wav)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    buf = decode_audio_window(probe, 1.234567, 2.5)
    start, end = window_sample_bounds(probe, 1.234567, 2.5)
    np.testing.assert_array_equal(buf.data, full[:, start:end])
    assert buf.data.dtype == np.float32


def test_window_keeps_every_channel(tmp_path: Path) -> None:
    stereo = _noise_wav(tmp_path / "stereo.wav", seconds=3.0, channels=2)
    probe = probe_audio(stereo, max_sample_rate=384000)
    full = _full(stereo)
    buf = decode_audio_window(probe, 0.5, 1.5)
    assert buf.channels == 2
    np.testing.assert_array_equal(buf.data, full[:, SR // 2 : 3 * SR // 2])


def test_window_end_is_clamped_to_the_file(wav: Path) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    buf = decode_audio_window(probe, 5.5, 500.0)
    assert buf.samples == SR // 2
    np.testing.assert_array_equal(buf.data, _full(wav)[:, int(5.5 * SR) :])


@pytest.mark.parametrize("start_s", [0.0, 0.05, 1.111, 3.0, 7.0])
def test_window_of_a_lossy_container_matches_the_full_decode(
    tmp_path: Path, start_s: float
) -> None:
    """AAC in mp4 — the case the UI actually loads. Two traps live here: the
    first frame after a seek has no MDCT overlap partner (fixed by the
    pre-roll), and an explicit ``-ss 0`` un-trims the encoder priming samples
    (fixed by not seeking at all at the head). Both showed up as a ~0.3 to 0.7
    full-scale error before; the window must be sample-exact."""
    src = _noise_wav(tmp_path / "src.wav", seconds=8.0)
    dst = tmp_path / "src.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", "-b:a", "192k", str(dst)],
        check=True,
        capture_output=True,
    )
    probe = probe_audio(dst, max_sample_rate=384000)
    full = decode_audio(probe).data[0]
    win = decode_audio_window(probe, start_s, start_s + 1.0).data[0]
    start = int(round(start_s * probe.sample_rate))
    np.testing.assert_array_equal(win, full[start : start + win.size])


# ------------------------------------------------------------------ error paths


def test_window_rejects_nan_and_absurd_amplitude(tmp_path: Path) -> None:
    bad = tmp_path / "nan.wav"
    data = np.full(SR, np.nan, dtype=np.float32)
    sf.write(str(bad), data, SR, subtype="FLOAT")
    probe = probe_audio(bad, max_sample_rate=384000)
    with pytest.raises(InvalidUserInputError, match="NaN"):
        decode_audio_window(probe, 0.0, 0.5)

    huge = tmp_path / "huge.wav"
    sf.write(str(huge), np.full(SR, 50.0, dtype=np.float32), SR, subtype="FLOAT")
    probe = probe_audio(huge, max_sample_rate=384000)
    with pytest.raises(InvalidUserInputError, match="abnormal float amplitude"):
        decode_audio_window(probe, 0.0, 0.5)


def test_window_timeout_and_ffmpeg_failure_are_user_input_errors(
    monkeypatch: pytest.MonkeyPatch, wav: Path
) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)

    def _timeout(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=0.1)

    monkeypatch.setattr(subprocess, "run", _timeout)
    with pytest.raises(InvalidUserInputError, match="timed out"):
        decode_audio_window(probe, 0.0, 1.0, timeout_s=0.1)

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("ffmpeg vanished")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(InvalidUserInputError, match="FFmpeg failed"):
        decode_audio_window(probe, 0.0, 1.0)


def test_window_with_no_decodable_bytes_is_an_error(
    monkeypatch: pytest.MonkeyPatch, wav: Path
) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)

    class _Empty:
        stdout = b""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Empty())
    with pytest.raises(InvalidUserInputError, match="zero samples"):
        decode_audio_window(probe, 0.0, 1.0)

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        sf, "read", lambda *_a, **_k: (np.zeros((0, 1), dtype=np.float32), probe.sample_rate)
    )
    with pytest.raises(InvalidUserInputError, match="zero samples"):
        decode_audio_window(probe, 0.0, 1.0)


def test_window_fallback_rejects_a_sample_rate_mismatch(
    monkeypatch: pytest.MonkeyPatch, wav: Path
) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(sf, "read", lambda *_a, **_k: (np.zeros((10, 1), dtype=np.float32), 8000))
    with pytest.raises(InvalidUserInputError, match="does not match probe"):
        decode_audio_window(probe, 0.0, 1.0)


# -------------------------------------------------- decode_audio is unchanged


def test_full_decode_still_reads_the_whole_file(wav: Path) -> None:
    probe = probe_audio(wav, max_sample_rate=384000)
    buf = decode_audio(probe)
    assert buf.samples == 6 * SR
    assert buf.sample_rate == SR
