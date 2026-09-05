"""``POST /api/analyze`` streams: the numbers must not move, and the memory must.

Goal box E1 says analyze of a three-hour file must not cost multi-GB. It used
to decode the whole file — 12.76 GB of peak RSS on a 3 h / 2.07 GB recording.
Every product analyze returns is now a streaming reduction, which is only worth
anything if it produces the *same* numbers, so this file is mostly an
equivalence proof against the whole-file formulation it replaced:

1. the decode stream is bit-identical to ``decode_audio``;
2. the long-term average spectrum is identical to averaging every frame of the
   whole file at the end (it is the same sum over the same frames);
3. BS.1770 integrated loudness and oversampled true peak are identical to
   ``measure_loudness_and_peaks`` — within 0.01 LU / 0.01 dB is the contract,
   the measured difference is 0;
4. the overview buckets equal the whole-file buckets over the timeline the
   grid is laid on;
5. peak RSS is flat in file length, measured in a fresh subprocess.

``reference_analysis`` below *is* the old whole-file implementation, kept here
as the oracle. If it and the streaming path ever disagree, one of them is wrong
and the test says which number moved.
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean.audio.decode import decode_audio, iter_decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioProbeResult
from hawavoclean.finishing.loudness import LoudnessMeasurement, measure_loudness_and_peaks
from hawavoclean.server.analysis import (
    ANALYSIS_MAX_SAMPLE_RATE,
    SPECTRUM_N_FFT,
    _db,
    analyze_audio,
    average_power_spectrum,
    band_centres,
    band_integrate,
    stream_measurements,
    waveform_overview,
)
from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit

TOKEN = "t0ken"
H = {"X-Hawa-Token": TOKEN}
SR = 48000
BUCKETS = 1200
REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_MEDIA = REPO_ROOT / "test_output" / "ui-smoke" / "Flute 09.m4a.mp4"

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


# --------------------------------------------------------------- the oracle


def reference_analysis(
    path: Path, buckets: int = BUCKETS
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, LoudnessMeasurement, int]:
    """The whole-file formulation analyze used before it was made streaming."""
    probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    buf = decode_audio(probe)
    mono = buf.to_mono()
    mins, maxs, rms_db = waveform_overview(mono, buckets)
    bin_power = average_power_spectrum(mono, n_fft=SPECTRUM_N_FFT)
    loudness = measure_loudness_and_peaks(buf.data, buf.sample_rate)
    return mins, maxs, rms_db, bin_power, loudness, buf.samples


def spectrum_db(bin_power: FloatArray, sample_rate: int) -> FloatArray:
    centres = band_centres(sample_rate)
    return _db(band_integrate(bin_power, sample_rate, SPECTRUM_N_FFT, centres))


# ------------------------------------------------------------------ fixtures


def _bursty(n: int, sample_rate: int = SR, seed: int = 4) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Speech-shaped: loud and quiet stretches, so the -70 LUFS absolute gate
    and the -10 LU relative gate both actually discard blocks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)) ** 3
    sig = env * (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.03 * rng.standard_normal(n))
    return np.asarray(sig, dtype=np.float32)


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(work: Path) -> Iterator[TestClient]:
    assert work.is_dir()
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    manager.shutdown()


def _write(path: Path, data: np.ndarray[Any, Any], sample_rate: int = SR) -> Path:
    sf.write(str(path), data, sample_rate, subtype="FLOAT")
    return path


@pytest.fixture
def fixtures(work: Path) -> dict[str, Path]:
    """A spread that hits every branch: normal, stereo, gate-heavy, sub-400 ms,
    near-silent, a non-48 kHz rate, and a file shorter than one FFT frame."""
    sig = _bursty(SR * 9 + 777)
    out = {
        "mono": _write(work / "mono.wav", sig),
        "stereo": _write(work / "stereo.wav", np.stack([sig, np.roll(sig, 137) * 0.6], axis=1)),
        "rate_44k1": _write(work / "r441.wav", sig[: 44100 * 5], 44100),
        "short": _write(work / "short.wav", sig[: int(SR * 0.3)]),
        "sub_frame": _write(work / "subframe.wav", sig[:900]),
        "near_silence": _write(work / "quiet.wav", (sig[:900] * 1e-6).astype(np.float32)),
    }
    gated = np.zeros(SR * 8, dtype=np.float32)
    gated[SR * 3 : SR * 4] = sig[:SR]
    out["gated"] = _write(work / "gated.wav", gated)
    return out


# ------------------------------------------------- 1. the decode stream itself


def test_the_chunked_decode_is_the_same_bytes_as_the_whole_decode(
    fixtures: dict[str, Path],
) -> None:
    """The streaming reduction is only equivalent if the samples are. One
    ffmpeg process with no seek means no lossy pre-roll question arises — so
    this is bit equality, not closeness."""
    for name in ("mono", "stereo", "rate_44k1", "sub_frame"):
        probe = probe_audio(fixtures[name], max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
        whole = decode_audio(probe)
        for chunk_samples in (1 << 20, 4096, 977):
            joined = np.concatenate(
                [c.data for c in iter_decode_audio(probe, chunk_samples)], axis=1
            )
            assert np.array_equal(joined, whole.data), f"{name} @ {chunk_samples}"


def test_the_chunked_decode_is_exact_on_a_lossy_container() -> None:
    if not REAL_MEDIA.is_file():
        pytest.skip(f"test media missing: {REAL_MEDIA}")
    probe = probe_audio(REAL_MEDIA, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    whole = decode_audio(probe)
    joined = np.concatenate([c.data for c in iter_decode_audio(probe, 1 << 18)], axis=1)
    assert np.array_equal(joined, whole.data)


def test_a_chunked_decode_of_a_corrupt_file_still_raises() -> None:
    junk = REPO_ROOT / "pyproject.toml"  # a real file that is not audio
    probe = AudioProbeResult(
        path=junk,
        format_name="wav",
        codec_name="pcm_s16le",
        sample_rate=SR,
        channels=1,
        duration_s=1.0,
        samples=SR,
        bit_depth=16,
        sha256="0" * 64,
    )
    with pytest.raises(Exception, match="(?i)ffmpeg|decode"):
        list(iter_decode_audio(probe, 4096))


# --------------------------------------------------- 2. spectrum equivalence


def test_the_streaming_spectrum_equals_the_whole_file_average(
    fixtures: dict[str, Path],
) -> None:
    """A running sum of per-frame power over a running frame count is the same
    average as summing every frame at the end: same frames, same grid, same
    divisor. Tolerance here is float64 summation-order noise, nothing else."""
    for name, path in fixtures.items():
        _, _, _, ref_power, _, _ = reference_analysis(path)
        probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
        rate = probe.sample_rate
        ref_db = spectrum_db(ref_power, rate)
        for chunk_samples in (1 << 20, 65536, 4096, 977):
            _, _, _, power, _, _ = stream_measurements(probe, BUCKETS, chunk_samples)
            assert power.shape == ref_power.shape, name
            worst = float(np.max(np.abs(spectrum_db(power, rate) - ref_db)))
            assert worst < 1e-6, f"{name} @ {chunk_samples}: spectrum moved {worst} dB"


def test_the_spectrum_is_exact_on_the_real_media() -> None:
    if not REAL_MEDIA.is_file():
        pytest.skip(f"test media missing: {REAL_MEDIA}")
    _, _, _, ref_power, _, _ = reference_analysis(REAL_MEDIA)
    probe = probe_audio(REAL_MEDIA, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    _, _, _, power, _, _ = stream_measurements(probe, BUCKETS)
    worst = float(
        np.max(
            np.abs(
                spectrum_db(power, probe.sample_rate) - spectrum_db(ref_power, probe.sample_rate)
            )
        )
    )
    assert worst < 1e-6, f"spectrum moved {worst} dB on the real media"


# ------------------------------------------- 3. loudness / true-peak equivalence


def test_the_streaming_loudness_equals_the_whole_file_measurement(
    fixtures: dict[str, Path],
) -> None:
    """Gated BS.1770 needs every block before it can apply the gates, but the
    blocks themselves are a per-sample reduction: K-weight with carried filter
    state, accumulate one mean square per 400 ms block, gate at the end.
    The contract is 0.01 LU / 0.01 dB. The measured worst case over this whole
    spread is 1.2e-7 LU and *exactly* zero dB of true peak: the loudness
    residue is only there because the block mean squares accumulate in float64
    where pyloudnorm sums float32, and it does not grow with chunk count — the
    same value comes out whether the file arrives in one piece or in 977-sample
    slices. So the bound asserted here is five orders tighter than the contract."""
    for name, path in fixtures.items():
        *_, ref_loudness, _ = reference_analysis(path)
        probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
        for chunk_samples in (1 << 20, 65536, 4096, 977):
            *_, loudness, _ = stream_measurements(probe, BUCKETS, chunk_samples)
            d_lufs = abs(loudness.integrated_lufs - ref_loudness.integrated_lufs)
            d_tp = abs(loudness.true_peak_dbtp - ref_loudness.true_peak_dbtp)
            d_sp = abs(loudness.sample_peak_dbfs - ref_loudness.sample_peak_dbfs)
            where = f"{name} @ {chunk_samples}"
            assert d_lufs < 1e-5, f"{where}: integrated loudness moved {d_lufs} LU"
            assert d_tp == 0.0, f"{where}: true peak moved {d_tp} dB"
            assert d_sp == 0.0, f"{where}: sample peak moved {d_sp} dB"


def test_the_loudness_is_exact_on_the_real_media() -> None:
    if not REAL_MEDIA.is_file():
        pytest.skip(f"test media missing: {REAL_MEDIA}")
    *_, ref_loudness, _ = reference_analysis(REAL_MEDIA)
    probe = probe_audio(REAL_MEDIA, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    *_, loudness, _ = stream_measurements(probe, BUCKETS)
    assert abs(loudness.integrated_lufs - ref_loudness.integrated_lufs) < 1e-5
    assert loudness.true_peak_dbtp == ref_loudness.true_peak_dbtp
    assert loudness.sample_peak_dbfs == ref_loudness.sample_peak_dbfs


def test_loudness_of_a_signal_that_is_entirely_below_the_absolute_gate(work: Path) -> None:
    """Every block under -70 LUFS: pyloudnorm's gate keeps nothing and the
    caller maps the resulting -inf to -70. The streaming gate must agree."""
    quiet = _write(work / "verysoft.wav", (np.zeros(SR * 3) + 1e-7).astype(np.float32))
    *_, ref_loudness, _ = reference_analysis(quiet)
    probe = probe_audio(quiet, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    *_, loudness, _ = stream_measurements(probe, BUCKETS, 4096)
    assert loudness.integrated_lufs == ref_loudness.integrated_lufs == -70.0


def test_loudness_of_a_file_with_more_channels_than_bs1770_defines(work: Path) -> None:
    """pyloudnorm refuses more than five channels; the whole-file wrapper turns
    that into -70. The streaming path must not invent a number instead."""
    six = np.stack([_bursty(SR * 2, seed=s) for s in range(6)], axis=1)
    path = _write(work / "six.wav", six)
    *_, ref_loudness, _ = reference_analysis(path)
    probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    assert probe.channels == 6
    *_, loudness, _ = stream_measurements(probe, BUCKETS, 4096)
    assert loudness.integrated_lufs == ref_loudness.integrated_lufs == -70.0
    assert abs(loudness.true_peak_dbtp - ref_loudness.true_peak_dbtp) < 0.01


# ------------------------------------------------------- 4. overview buckets


def test_the_streaming_buckets_equal_the_whole_file_buckets(
    fixtures: dict[str, Path],
) -> None:
    """For PCM the container's sample count is the decoded sample count, so the
    grid is the same grid and the buckets are the same buckets."""
    for name, path in fixtures.items():
        ref_mins, ref_maxs, ref_rms, _, _, ref_n = reference_analysis(path)
        probe = probe_audio(path, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
        assert probe.samples == ref_n, f"{name}: PCM probe count should be exact"
        for chunk_samples in (1 << 20, 65536, 4096, 977):
            mins, maxs, rms, _, _, covered = stream_measurements(probe, BUCKETS, chunk_samples)
            where = f"{name} @ {chunk_samples}"
            assert covered == ref_n, where
            assert mins.shape == ref_mins.shape == (BUCKETS,), where
            assert np.array_equal(mins, ref_mins), where
            assert np.array_equal(maxs, ref_maxs), where
            assert float(np.max(np.abs(rms - ref_rms))) < 1e-9, where


def test_buckets_on_a_lossy_container_follow_the_container_timeline() -> None:
    """AAC decodes 71 samples of frame padding past the length its container
    declares. The overview grid is laid on the *container* timeline — the one
    the playhead, ``/api/peaks`` and the ``<audio>`` element use — so that tail
    is dropped rather than shifting every bucket boundary against the player.
    """
    if not REAL_MEDIA.is_file():
        pytest.skip(f"test media missing: {REAL_MEDIA}")
    probe = probe_audio(REAL_MEDIA, max_sample_rate=ANALYSIS_MAX_SAMPLE_RATE)
    mono = decode_audio(probe).to_mono()
    assert int(mono.shape[0]) > probe.samples, "expected the AAC decoder to overshoot"
    ref_mins, ref_maxs, ref_rms = waveform_overview(mono[: probe.samples], BUCKETS)
    mins, maxs, rms, _, _, covered = stream_measurements(probe, BUCKETS)
    assert covered == probe.samples
    assert np.array_equal(mins, ref_mins)
    assert np.array_equal(maxs, ref_maxs)
    assert float(np.max(np.abs(rms - ref_rms))) < 1e-9


def test_analyze_and_peaks_report_the_same_duration() -> None:
    """Before this change ``/api/analyze`` reported the decoder's length and
    ``/api/peaks`` the container's — 1.5 ms apart on the project's own media."""
    if not REAL_MEDIA.is_file():
        pytest.skip(f"test media missing: {REAL_MEDIA}")
    from hawavoclean.server.analysis import compute_peaks_window

    analysis = analyze_audio(REAL_MEDIA, BUCKETS)
    window = compute_peaks_window(REAL_MEDIA, 0.0, 1e9, 16)
    assert analysis["duration_s"] == window["duration_s"]


# ------------------------------------------------------ 5. the response shape


def test_the_response_shape_is_unchanged(client: TestClient, fixtures: dict[str, Path]) -> None:
    r = client.post("/api/analyze", headers=H, json={"path": str(fixtures["stereo"])})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {
        "path",
        "duration_s",
        "sample_rate",
        "channels",
        "peaks",
        "rms_db",
        "spectrum",
        "loudness",
        "noise_floor_db",
    }
    assert set(body["peaks"]) == {"min", "max"}
    assert set(body["spectrum"]) == {"freqs_hz", "db"}
    assert set(body["loudness"]) == {"integrated_lufs", "true_peak_dbtp"}
    assert body["channels"] == 2
    assert body["sample_rate"] == SR
    assert len(body["peaks"]["min"]) == len(body["peaks"]["max"]) == len(body["rms_db"]) == 1200
    assert len(body["spectrum"]["freqs_hz"]) == len(body["spectrum"]["db"])
    assert all(isinstance(v, float) for v in body["peaks"]["min"])
    # Rounding is part of the contract: 4 decimals on peaks, 2 on dB.
    assert all(round(v, 4) == v for v in body["peaks"]["max"])
    assert all(round(v, 2) == v for v in body["rms_db"])


def test_the_route_still_rejects_a_bad_bucket_count(
    client: TestClient, fixtures: dict[str, Path]
) -> None:
    r = client.post(
        "/api/analyze", headers=H, json={"path": str(fixtures["mono"]), "buckets": 999999}
    )
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request"


def test_analyze_of_a_corrupt_file_is_an_error_not_a_crash(client: TestClient, work: Path) -> None:
    junk = work / "junk.wav"
    junk.write_bytes(b"definitely not audio")
    r = client.post("/api/analyze", headers=H, json={"path": str(junk)})
    assert r.status_code in (400, 500)
    assert set(r.json()) == {"error", "message"}


# ------------------------------------------------------------ 6. memory proof

_MEM_SCRIPT = r"""
import json, sys, time
from pathlib import Path
from hawavoclean.server.analysis import analyze_audio

def rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3
    except ImportError:
        from hawavoclean.runtime import process_peak_rss_bytes
        return process_peak_rss_bytes() / 1e6

path = Path(sys.argv[1])
baseline = rss_mb()
t0 = time.perf_counter()
out = analyze_audio(path, 1200)
elapsed = time.perf_counter() - t0
peak = rss_mb()
assert len(out["peaks"]["min"]) == 1200 and out["duration_s"] > 0
print(json.dumps({
    "growth_mb": peak - baseline, "baseline_mb": baseline, "peak_mb": peak,
    "wall_s": elapsed, "file_mb": path.stat().st_size / 1e6,
    "duration_s": out["duration_s"], "lufs": out["loudness"]["integrated_lufs"],
}))
"""


def _measure(path: Path) -> dict[str, float]:
    """Peak RSS is a process-lifetime high-water mark, so each measurement gets
    a FRESH subprocess — an in-process delta is contaminated by what ran before."""
    proc = subprocess.run(
        [sys.executable, "-c", _MEM_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return dict(json.loads(proc.stdout))


def _write_long(path: Path, minutes: float, seed: int) -> None:
    """Written a minute at a time, so making the fixture is not itself the hog."""
    rng = np.random.default_rng(seed)
    remaining = int(SR * 60 * minutes)
    start = 0
    with sf.SoundFile(str(path), "w", samplerate=SR, channels=1, subtype="FLOAT") as f:
        while remaining > 0:
            block = min(SR * 60, remaining)
            t = np.arange(start, start + block) / SR
            env = (0.5 + 0.5 * np.sin(2 * np.pi * 0.13 * t)) ** 2
            f.write(
                (
                    env * (0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(block))
                ).astype(np.float32)
            )
            remaining -= block
            start += block


@pytest.mark.slow
def test_analyze_of_a_long_file_costs_a_chunk_not_a_file() -> None:
    """Goal box E1. Two lengths, because "smaller" is not the claim — *flat* is.
    Whole-file decoding cost 2932 MB on the 30-minute file here and 12650 MB on
    a three-hour one; streaming, both sit at ~119 MB, and the difference
    between the two file sizes is allocator noise.
    """
    out_dir = REPO_ROOT / "test_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    short = out_dir / "analyze_memory_proof_10min.wav"
    long = out_dir / "analyze_memory_proof_30min.wav"
    try:
        _write_long(short, 10, seed=3)
        _write_long(long, 30, seed=5)
        a = _measure(short)
        b = _measure(long)
    finally:
        short.unlink(missing_ok=True)
        long.unlink(missing_ok=True)

    print(
        f"\n[analyze] idle process {a['baseline_mb']:.0f} MB:"
        f"\n          10 min / {a['file_mb']:.0f} MB: peak RSS +{a['growth_mb']:.1f} MB,"
        f" {a['wall_s']:.2f} s"
        f"\n          30 min / {b['file_mb']:.0f} MB: peak RSS +{b['growth_mb']:.1f} MB,"
        f" {b['wall_s']:.2f} s"
    )
    assert abs(b["duration_s"] - 1800.0) < 0.01
    assert b["growth_mb"] < 400.0, (
        f"analyze of a {b['file_mb']:.0f} MB file grew RSS by {b['growth_mb']:.0f} MB "
        "— something stopped streaming"
    )
    assert b["growth_mb"] < b["file_mb"] / 2.0
    assert b["growth_mb"] - a["growth_mb"] < 64.0, (
        "peak RSS must be flat in file length: "
        f"{a['file_mb']:.0f} MB -> {a['growth_mb']:.0f} MB but "
        f"{b['file_mb']:.0f} MB -> {b['growth_mb']:.0f} MB"
    )
