"""``POST /api/peaks`` — windowed waveform peaks (ui-contract addendum 1).

Three things have to be true or the zoom is a lie:

1. a window's buckets equal the buckets a full-file analysis would produce over
   the same span (including a window that starts mid-bucket);
2. at ``samples_per_bucket == 1`` the response *is* the raw samples;
3. serving a window out of a huge file costs a window, not a file — measured
   as peak RSS of a fresh process, because the contract says "measured, not
   assumed".
"""

import math
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.server.analysis import (
    MAX_BUCKETS,
    PeaksWindowError,
    compute_peaks_window,
    waveform_overview,
)
from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit

TOKEN = "t0ken"
H = {"X-Hawa-Token": TOKEN}
SR = 48000
CLICKS_S = (1.0, 2.5, 3.777777, 5.0, 7.123456)
DURATION_S = 12.0


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The work dir is an allowed root for the path policy; put test media there."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(work: Path) -> Iterator[TestClient]:
    assert work.is_dir()  # the allowed-root override is active for every client test
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    manager.shutdown()


def _click_train(path: Path) -> Path:
    """Low noise floor with single-sample clicks at exactly known times: any
    misalignment moves a click into the wrong bucket and the test says so."""
    rng = np.random.default_rng(23)
    n = int(DURATION_S * SR)
    sig = (0.01 * rng.standard_normal(n)).astype(np.float32)
    for i, t in enumerate(CLICKS_S):
        sig[int(round(t * SR))] = 0.9 if i % 2 == 0 else -0.85
    sf.write(str(path), sig, SR, subtype="FLOAT")
    return path


def _mono_full(path: Path) -> np.ndarray[Any, np.dtype[np.float32]]:
    return decode_audio(probe_audio(path, max_sample_rate=384000)).to_mono()


# ------------------------------------------------------- window == full file


@pytest.mark.parametrize(
    ("start_s", "end_s", "buckets"),
    [
        (0.0, 12.0, 600),  # whole file
        (2.0, 6.0, 400),  # bucket-aligned start
        (2.5333333, 6.1777777, 377),  # deliberately off every bucket boundary
        (0.9999, 1.0001, 7),  # a hair either side of a click
        (11.5, 12.0, 1200),  # right up to the end
    ],
)
def test_window_buckets_equal_a_full_file_analysis_of_the_same_span(
    work: Path, start_s: float, end_s: float, buckets: int
) -> None:
    wav = _click_train(work / "clicks.wav")
    got = compute_peaks_window(wav, start_s, end_s, buckets)

    full = _mono_full(wav)
    start = int(round(start_s * SR))
    end = min(int(round(end_s * SR)), full.size)
    exp_min, exp_max, exp_rms = waveform_overview(full[start:end], min(buckets, end - start))

    assert got["start_s"] == pytest.approx(start / SR, abs=1e-6)
    assert got["end_s"] == pytest.approx(end / SR, abs=1e-6)
    assert len(got["peaks"]["min"]) == exp_min.size
    np.testing.assert_allclose(got["peaks"]["min"], exp_min, atol=1e-6)
    np.testing.assert_allclose(got["peaks"]["max"], exp_max, atol=1e-6)
    np.testing.assert_allclose(got["rms_db"], exp_rms, atol=0.01)


def test_clicks_land_in_the_bucket_their_timestamp_names(work: Path) -> None:
    """Absolute-time check: an off-by-one seek would put a click one bucket out."""
    wav = _click_train(work / "clicks.wav")
    start_s, end_s, buckets = 2.5333333, 8.0, 500
    got = compute_peaks_window(wav, start_s, end_s, buckets)
    span = got["end_s"] - got["start_s"]
    for t, amp in zip(CLICKS_S, (0.9, -0.85, 0.9, -0.85, 0.9), strict=True):
        if not got["start_s"] <= t < got["end_s"]:
            continue
        idx = int((t - got["start_s"]) / span * buckets)
        extreme = got["peaks"]["max"][idx] if amp > 0 else got["peaks"]["min"][idx]
        assert extreme == pytest.approx(amp, abs=1e-5), f"click at {t}s missed bucket {idx}"


def test_deep_zoom_returns_the_raw_samples(work: Path) -> None:
    """buckets clamp down to the sample count, so one bucket = one sample and
    min == max == the sample itself. This is the whole point of E3."""
    wav = _click_train(work / "clicks.wav")
    start_s, end_s = 5.0, 5.01  # 480 samples, one of them a click
    got = compute_peaks_window(wav, start_s, end_s, MAX_BUCKETS)

    n = int(round(end_s * SR)) - int(round(start_s * SR))
    assert got["samples_per_bucket"] == 1
    assert len(got["peaks"]["min"]) == len(got["peaks"]["max"]) == n

    raw = _mono_full(wav)[int(round(start_s * SR)) : int(round(end_s * SR))]
    np.testing.assert_allclose(got["peaks"]["min"], raw, atol=1e-6)
    np.testing.assert_allclose(got["peaks"]["max"], raw, atol=1e-6)
    assert got["peaks"]["min"][0] == pytest.approx(-0.85, abs=1e-5)  # the click at 5.0 s


def test_samples_per_bucket_reports_the_widest_bucket(work: Path) -> None:
    wav = _click_train(work / "clicks.wav")
    assert compute_peaks_window(wav, 0.0, 12.0, 1200)["samples_per_bucket"] == 480
    assert compute_peaks_window(wav, 12.0 - 6.5, 12.0, 1600)["samples_per_bucket"] == 195
    # 481 samples over 480 buckets: not everything is one sample yet, and the
    # report must not claim otherwise.
    win = compute_peaks_window(wav, 1.0, 1.0 + 481 / SR, 480)
    assert win["samples_per_bucket"] == 2
    assert len(win["rms_db"]) == 480


# ------------------------------------------------------------- response shape


def test_response_carries_the_contract_fields(work: Path) -> None:
    wav = _click_train(work / "clicks.wav")
    got = compute_peaks_window(wav, 3.0, 8.0, 1000)
    assert set(got) == {
        "path",
        "start_s",
        "end_s",
        "sample_rate",
        "channels",
        "duration_s",
        "samples_per_bucket",
        "peaks",
        "rms_db",
    }
    assert got["path"] == str(wav)
    assert got["sample_rate"] == SR and got["channels"] == 1
    assert got["duration_s"] == pytest.approx(DURATION_S, abs=1e-3)
    assert all(-1.0 <= v <= 1.0 for v in got["peaks"]["min"] + got["peaks"]["max"])
    assert all(-120.0 <= v <= 0.0 for v in got["rms_db"])
    assert all(lo <= hi for lo, hi in zip(got["peaks"]["min"], got["peaks"]["max"], strict=True))


def test_end_is_clamped_to_the_duration(work: Path) -> None:
    wav = _click_train(work / "clicks.wav")
    got = compute_peaks_window(wav, 11.0, 900.0, 100)
    assert got["end_s"] == pytest.approx(DURATION_S, abs=1e-3)
    assert got["duration_s"] == pytest.approx(DURATION_S, abs=1e-3)


def test_stereo_window_is_the_mono_mix(work: Path) -> None:
    left = np.zeros(SR, dtype=np.float32)
    right = np.zeros(SR, dtype=np.float32)
    left[SR // 2] = 0.8
    right[SR // 2] = 0.4
    stereo = work / "stereo.wav"
    sf.write(str(stereo), np.stack([left, right], axis=1), SR, subtype="FLOAT")
    got = compute_peaks_window(stereo, 0.0, 1.0, 10)
    assert got["channels"] == 2
    assert max(got["peaks"]["max"]) == pytest.approx(0.6, abs=1e-5)


# ------------------------------------------------------------------ bad input


@pytest.mark.parametrize(
    ("start_s", "end_s", "buckets"),
    [
        (12.0, 13.0, 100),  # start_s == duration
        (99.0, 100.0, 100),  # start_s past duration
        (5.0, 5.0, 100),  # empty window
        (5.0, 4.0, 100),  # reversed
        (-1.0, 2.0, 100),  # negative
        (float("nan"), 2.0, 100),
        (1.0, float("inf"), 100),
        (1.0, 2.0, 0),  # buckets below range
        (1.0, 2.0, MAX_BUCKETS + 1),  # buckets above range
    ],
)
def test_unusable_windows_raise_peaks_window_error(
    work: Path, start_s: float, end_s: float, buckets: int
) -> None:
    wav = _click_train(work / "clicks.wav")
    with pytest.raises(PeaksWindowError):
        compute_peaks_window(wav, start_s, end_s, buckets)


# ---------------------------------------------------------------- HTTP route


def test_route_serves_a_window(client: TestClient, work: Path) -> None:
    wav = _click_train(work / "clicks.wav")
    r = client.post(
        "/api/peaks",
        headers=H,
        json={"path": str(wav), "start_s": 2.0, "end_s": 4.0, "buckets": 500},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["path"] == str(wav)
    assert body["start_s"] == pytest.approx(2.0) and body["end_s"] == pytest.approx(4.0)
    assert body["samples_per_bucket"] == 192
    assert len(body["peaks"]["min"]) == len(body["peaks"]["max"]) == len(body["rms_db"]) == 500
    # default bucket count
    r = client.post("/api/peaks", headers=H, json={"path": str(wav), "start_s": 0.0, "end_s": 12.0})
    assert r.status_code == 200 and len(r.json()["rms_db"]) == 1200


def test_route_requires_the_token(client: TestClient, work: Path) -> None:
    wav = _click_train(work / "clicks.wav")
    body = {"path": str(wav), "start_s": 0.0, "end_s": 1.0}
    assert client.post("/api/peaks", json=body).status_code == 401
    assert client.post("/api/peaks", headers={"X-Hawa-Token": "no"}, json=body).status_code == 401
    assert client.post(f"/api/peaks?token={TOKEN}", json=body).status_code == 400
    assert client.post("/api/peaks", headers=H, json=body).status_code == 200


def test_route_path_policy_matches_analyze(client: TestClient, work: Path) -> None:
    for path, status, code in [
        ("/etc/passwd", 403, "forbidden"),
        ("relative.wav", 400, "bad_request"),
        (str(work / "nope.wav"), 404, "not_found"),
    ]:
        r = client.post("/api/peaks", headers=H, json={"path": path, "start_s": 0.0, "end_s": 1.0})
        assert r.status_code == status, r.text
        assert r.json()["error"] == code and "message" in r.json()


@pytest.mark.parametrize(
    ("with_path", "body"),
    [
        (False, {"start_s": 0.0, "end_s": 1.0}),  # no path
        (True, {"start_s": 1.0}),  # no end_s
        (True, {"end_s": 1.0}),  # no start_s
        (True, {"start_s": -1.0, "end_s": 1.0}),
        (True, {"start_s": 0.0, "end_s": 0.0}),
        (True, {"start_s": 2.0, "end_s": 1.0}),
        (True, {"start_s": 12.0, "end_s": 13.0}),  # at the end of the file
        (True, {"start_s": 99.0, "end_s": 100.0}),  # past the end of the file
        (True, {"start_s": 0.0, "end_s": 1.0, "buckets": 0}),
        (True, {"start_s": 0.0, "end_s": 1.0, "buckets": MAX_BUCKETS + 1}),
        (True, {"start_s": "soon", "end_s": 1.0}),
    ],
)
def test_route_rejects_bad_requests_with_400(
    client: TestClient, work: Path, with_path: bool, body: dict[str, Any]
) -> None:
    wav = _click_train(work / "clicks.wav")
    payload = {**({"path": str(wav)} if with_path else {}), **body}
    r = client.post("/api/peaks", headers=H, json=payload)
    assert r.status_code == 400, r.text
    assert r.json()["error"] == "bad_request"


def test_route_rejects_non_finite_bounds(client: TestClient, work: Path) -> None:
    """``json.loads`` accepts the NaN/Infinity literals, so the route has to."""
    wav = _click_train(work / "clicks.wav")
    for raw in (
        '{"path": %s, "start_s": NaN, "end_s": 1.0}',
        '{"path": %s, "start_s": 0.0, "end_s": Infinity}',
        '{"path": %s, "start_s": -Infinity, "end_s": 1.0}',
    ):
        r = client.post(
            "/api/peaks",
            headers={**H, "Content-Type": "application/json"},
            content=(raw % f'"{wav}"').encode(),
        )
        assert r.status_code == 400, r.text
        assert r.json()["error"] == "bad_request"


def test_route_reports_a_corrupt_file_without_crashing(client: TestClient, work: Path) -> None:
    junk = work / "junk.wav"
    junk.write_bytes(b"not audio at all")
    r = client.post("/api/peaks", headers=H, json={"path": str(junk), "start_s": 0.0, "end_s": 1.0})
    assert r.status_code in (400, 500)
    assert set(r.json()) == {"error", "message"}


# ------------------------------------------------------- memory proof (E3/E1)

_MEM_SCRIPT = r"""
import sys, time
from pathlib import Path
from hawavoclean.runtime import process_peak_rss_bytes
from hawavoclean.server.analysis import compute_peaks_window

def rss_mb():
    return process_peak_rss_bytes() / 1e6

path = Path(sys.argv[1])
start, end, buckets = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
baseline = rss_mb()
t0 = time.perf_counter()
out = compute_peaks_window(path, start, end, buckets)
elapsed = time.perf_counter() - t0
peak = rss_mb()
assert max(out["peaks"]["max"]) > 0.1
print(f"{peak - baseline:.2f} {baseline:.1f} {elapsed:.3f} {path.stat().st_size / 1e6:.1f} "
      f"{out['samples_per_bucket']} {len(out['peaks']['min'])}")
"""


def _measure(big: Path, start: float, end: float, buckets: int) -> tuple[float, ...]:
    """Peak RSS is a process-lifetime high-water mark, so every measurement
    gets a FRESH subprocess: an in-process delta is contaminated by whatever
    ran before it."""
    proc = subprocess.run(
        [sys.executable, "-c", _MEM_SCRIPT, str(big), str(start), str(end), str(buckets)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    return tuple(float(v) for v in proc.stdout.split())


def _write_noise(path: Path, seconds: int, seed: int) -> None:
    """Written in chunks, so generating the fixture is not itself the memory hog."""
    rng = np.random.default_rng(seed)
    remaining = SR * seconds
    with sf.SoundFile(str(path), "w", samplerate=SR, channels=1, subtype="FLOAT") as f:
        while remaining > 0:
            block = min(SR * 60, remaining)
            f.write((0.25 * rng.standard_normal(block)).astype(np.float32))
            remaining -= block


@pytest.mark.slow
def test_a_window_out_of_a_huge_file_costs_a_window() -> None:
    """Contract, addendum 1: "Peak RSS for a 5-second window out of a 3-hour
    file must stay within a few MB of the idle server (measured, not
    assumed)." 30 minutes of 48 kHz float32 mono is 345 MB on disk, so a
    whole-file decode could not possibly hide inside the budgets below.

    The second measurement is the degenerate request — the whole file asked
    for as one window, which is what a client wants for an overview. One
    decode of that span cost 8.5 GB on a 3-hour file before the reduction was
    made streaming; it must now cost a chunk.
    """
    minutes = 30
    big = Path(__file__).resolve().parents[2] / "test_output" / "peaks_memory_proof.wav"
    big.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_noise(big, minutes * 60, seed=11)
        win_growth, baseline, win_s, file_mb, spb, n_buckets = _measure(big, 1500.0, 1505.0, 1600)
        all_growth, _, all_s, _, all_spb, all_buckets = _measure(big, 0.0, minutes * 60.0, 1200)
    finally:
        big.unlink(missing_ok=True)

    print(
        f"\n[peaks] {minutes} min / {file_mb:.0f} MB file, idle process {baseline:.0f} MB:"
        f"\n        5 s window (1600 buckets): peak RSS +{win_growth:.2f} MB, {win_s * 1000:.0f} ms"
        f"\n        whole file (1200 buckets): peak RSS +{all_growth:.1f} MB, {all_s:.2f} s"
    )
    assert (spb, n_buckets) == (150, 1600)
    assert (all_spb, all_buckets) == (72000, 1200)
    assert win_growth < 32.0, (
        f"serving a 5 s window grew RSS by {win_growth:.1f} MB on a {file_mb:.0f} MB file "
        "— the handler is decoding more than the window"
    )
    assert win_growth < file_mb / 10.0
    assert win_s < 5.0, f"a 5 s window took {win_s:.2f}s to serve"
    assert all_growth < 256.0, (
        f"bucketing the whole {file_mb:.0f} MB file grew RSS by {all_growth:.0f} MB "
        "— the reduction stopped being streaming"
    )


@pytest.mark.slow
def test_serving_windows_from_a_long_file_stays_interactive(work: Path) -> None:
    """Zoom/pan re-queries land on the HTTP route, so time the route itself."""
    long_wav = work / "long.wav"
    rng = np.random.default_rng(5)
    remaining = SR * 60 * 5
    with sf.SoundFile(str(long_wav), "w", samplerate=SR, channels=1, subtype="FLOAT") as f:
        while remaining > 0:
            block = min(SR * 30, remaining)
            f.write((0.2 * rng.standard_normal(block)).astype(np.float32))
            remaining -= block

    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    timings = []
    with TestClient(app, base_url="http://127.0.0.1") as c:
        for start in (0.0, 60.0, 150.0, 275.0):
            t0 = time.perf_counter()
            r = c.post(
                "/api/peaks",
                headers=H,
                json={"path": str(long_wav), "start_s": start, "end_s": start + 5.0},
            )
            timings.append(time.perf_counter() - t0)
            assert r.status_code == 200, r.text
            assert math.isclose(r.json()["end_s"], start + 5.0, abs_tol=1e-3)
    manager.shutdown()
    print(
        f"\n[peaks] HTTP 5 s windows out of a 5 min file: "
        f"{', '.join(f'{t * 1000:.0f} ms' for t in timings)}"
    )
    assert max(timings) < 3.0


# --------------------------------------------------------------- probe cache


def test_probe_cache_serves_repeat_windows_and_notices_a_rewrite(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zoom is a burst of windows over one file; the whole-file SHA-256 inside
    ``probe_audio`` must not be paid per gesture. It must still be paid the
    moment the file on disk changes."""
    from hawavoclean.audio.probe import probe_audio as real
    from hawavoclean.server import analysis

    wav = _click_train(work / "clicks.wav")
    analysis._probe_cache.clear()
    calls = 0
    # ``analysis`` re-imports this from ``hawavoclean.audio.probe``; take the
    # function from where it is defined so the capture is the same object
    # without reaching through a module that does not re-export it.
    assert analysis.probe_audio is real  # type: ignore[attr-defined]

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis, "probe_audio", counted)

    for start in (1.0, 2.0, 3.0, 4.0):
        assert compute_peaks_window(wav, start, start + 0.5, 200)["sample_rate"] == SR
    assert calls == 1, "repeat windows over one file re-probed the file"

    # A shorter file at the same path must not be served from the stale probe.
    time.sleep(0.01)
    sf.write(str(wav), np.zeros(SR, dtype=np.float32), SR, subtype="FLOAT")
    got = compute_peaks_window(wav, 0.0, 1.0, 10)
    assert calls == 2
    assert got["duration_s"] == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(PeaksWindowError):
        compute_peaks_window(wav, 5.0, 6.0, 10)  # past the new, shorter duration


def test_probe_cache_is_bounded(work: Path) -> None:
    from hawavoclean.server import analysis

    analysis._probe_cache.clear()
    for i in range(analysis.PROBE_CACHE_SIZE + 4):
        wav = work / f"c{i}.wav"
        sf.write(str(wav), np.full(SR // 10, 0.1, dtype=np.float32), SR, subtype="FLOAT")
        compute_peaks_window(wav, 0.0, 0.05, 10)
    assert len(analysis._probe_cache) == analysis.PROBE_CACHE_SIZE


# ------------------------------------------------------------ chunked reduction


@pytest.mark.parametrize(
    ("start_s", "end_s", "buckets"),
    [
        (0.0, 1.0, 300),  # buckets much smaller than a chunk
        (0.0, 1.0, 3),  # one bucket spans several chunks
        (0.0, 1.0, 1),  # the whole window is one bucket
        (0.0, 1.0, MAX_BUCKETS),  # 6 samples per bucket
        (2.5333333, 4.7, 517),  # aligned to neither a bucket nor a chunk edge
    ],
)
def test_chunked_reduction_equals_a_single_decode(
    work: Path, monkeypatch: pytest.MonkeyPatch, start_s: float, end_s: float, buckets: int
) -> None:
    """A window too long to hold at once is reduced chunk by chunk. Shrink the
    chunk so a one-second window takes ten of them, and demand the exact same
    numbers the single-decode path produces."""
    from hawavoclean.server import analysis

    wav = _click_train(work / "clicks.wav")
    reference = compute_peaks_window(wav, start_s, end_s, buckets)

    monkeypatch.setattr(analysis, "WINDOW_CHUNK_SAMPLES", 5000)
    chunked = compute_peaks_window(wav, start_s, end_s, buckets)

    assert chunked["samples_per_bucket"] == reference["samples_per_bucket"]
    assert chunked["start_s"] == reference["start_s"]
    assert chunked["end_s"] == reference["end_s"]
    np.testing.assert_allclose(chunked["peaks"]["min"], reference["peaks"]["min"], atol=1e-6)
    np.testing.assert_allclose(chunked["peaks"]["max"], reference["peaks"]["max"], atol=1e-6)
    np.testing.assert_allclose(chunked["rms_db"], reference["rms_db"], atol=0.02)


def test_a_stream_that_ends_early_reports_what_it_actually_covered(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container can promise more samples than it delivers. The response then
    has to describe the span that was really read — no empty tail buckets, and
    an ``end_s`` the client can trust."""
    from hawavoclean.audio.decode import decode_audio_window as real
    from hawavoclean.audio.types import AudioBuffer
    from hawavoclean.server import analysis

    wav = _click_train(work / "clicks.wav")
    monkeypatch.setattr(analysis, "WINDOW_CHUNK_SAMPLES", 5000)
    # Same object ``analysis`` holds, taken from its defining module (see above).
    assert analysis.decode_audio_window is real  # type: ignore[attr-defined]

    def truncating(probe: Any, start_s: float, end_s: float, **kw: Any) -> AudioBuffer:
        buf = real(probe, start_s, end_s, **kw)
        if start_s >= 0.2:  # the stream "ends" partway into the third chunk
            return AudioBuffer(data=buf.data[:, :1500], sample_rate=buf.sample_rate)
        return buf

    monkeypatch.setattr(analysis, "decode_audio_window", truncating)
    got = compute_peaks_window(wav, 0.0, 1.0, 100)

    covered = 2 * 5000 + 1500  # two whole chunks, then a short one
    assert got["end_s"] == pytest.approx(covered / SR, abs=1e-6)
    assert len(got["peaks"]["min"]) == len(got["rms_db"]) == covered * 100 // SR + 1
    assert all(v > -120.0 for v in got["rms_db"])  # no empty tail buckets

    raw = _mono_full(wav)[:covered]
    assert max(got["peaks"]["max"]) == pytest.approx(float(raw.max()), abs=1e-6)
