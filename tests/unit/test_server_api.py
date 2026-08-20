"""Engine HTTP API (docs/ui-contract.md section 1) through FastAPI's TestClient:
auth, health, analyze, jobs + SSE + cancel, ranged audio, upload, path
policy, shutdown, UI mount, error shape."""

import json
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean import __version__
from hawavoclean.server.app import (
    create_app,
    default_output_path,
    parse_range,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord

pytestmark = pytest.mark.unit
TOKEN = "t0ken"
H = {"X-Hawa-Token": TOKEN}


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The work dir is an allowed root for the path policy; put test media there."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    return tmp_path


def _tiny_wav(path: Path, seconds: float = 1.5, sr: int = 16000) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    sig = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sig = sig + 0.01 * rng.standard_normal(t.size)
    sf.write(str(path), sig.astype(np.float32), sr)
    return path


def _py(script: str) -> list[str]:
    return [sys.executable, "-u", "-c", textwrap.dedent(script)]


_SLOW = """
import json, sys, time
sys.stdout.write(json.dumps({"event":"progress","stage":"decode","progress":0.05,
                             "message":"slow"}) + "\\n"); sys.stdout.flush()
time.sleep(60)
"""


def _fake_done_factory(record: JobRecord) -> list[str]:
    report = {"schema_version": 1, "summary": {"units_total": 2, "enhanced": 1}}
    return _py(
        f"""
        import json, sys, time
        def emit(o):
            sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        for i, (stage, p) in enumerate([("preflight", 0.02), ("decode", 0.05), ("segment", 0.08),
                                         ("enhance", 0.08), ("guard", 0.44), ("enhance", 0.44),
                                         ("guard", 0.8), ("finish", 0.85), ("publish", 0.98)]):
            ev = {{"event":"progress","stage":stage,"progress":p,"message":f"{{stage}} {{i}}"}}
            if stage in ("enhance", "guard"):
                ev["unit"] = {{"index": 1 if p < 0.5 else 2, "total": 2}}
            emit(ev); time.sleep(0.03)
        open({str(record.report_path)!r}, "w").write({json.dumps(json.dumps(report))})
        open({str(record.output_path)!r}, "wb").write(b"RIFF")
        emit({{"event":"done","progress":1.0}})
        """
    )


@pytest.fixture
def client(work: Path) -> Iterator[TestClient]:
    assert work.is_dir()  # the allowed-root override is active for every client test
    manager = JobManager(command_factory=_fake_done_factory)
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app) as c:
        yield c
    manager.shutdown()


def _wait_done(client: TestClient, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    snap: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snap = client.get(f"/api/jobs/{job_id}", headers=H).json()
        if snap["state"] in TERMINAL_STATES:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"job never finished: {snap}")


# ----------------------------------------------------------------- auth / health


def test_missing_or_wrong_token_is_401(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"
    assert "message" in r.json()
    r = client.get("/api/health", headers={"X-Hawa-Token": "wrong"})
    assert r.status_code == 401
    r = client.get("/api/health?token=wrong")
    assert r.status_code == 401
    r = client.get("/api/health?token=t%C3%A9ken")  # non-ASCII token: still a clean 401
    assert r.status_code == 401
    r = client.post("/api/jobs", json={})
    assert r.status_code == 401
    # 401 responses still carry CORS headers for a file:// (null) origin.
    r = client.get("/api/health", headers={"Origin": "null"})
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "*"


def test_token_by_header_or_query(client: TestClient) -> None:
    r = client.get("/api/health", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "version": __version__,
        "profiles": ["studio", "production"],
        "engine_pid": body["engine_pid"],
    }
    assert isinstance(body["engine_pid"], int)
    assert client.get(f"/api/health?token={TOKEN}", headers={"Origin": "null"}).status_code == 200


def test_cors_preflight_needs_no_token(client: TestClient) -> None:
    r = client.options(
        "/api/jobs",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-hawa-token,content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"
    allowed = r.headers["access-control-allow-headers"].lower()
    assert "x-hawa-token" in allowed and "content-type" in allowed
    assert "POST" in r.headers["access-control-allow-methods"]


def test_errors_are_json_with_error_and_message(client: TestClient) -> None:
    r = client.get("/api/does-not-exist", headers=H)
    assert r.status_code == 404
    assert set(r.json()) == {"error", "message"} and r.json()["error"] == "not_found"
    r = client.get("/", headers=H)  # no --ui-dir: 404 JSON
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = client.delete("/api/health", headers=H)
    assert r.status_code == 405 and r.json()["error"] == "method_not_allowed"
    r = client.post("/api/analyze", headers=H, json={"buckets": 3})
    assert r.status_code == 400
    assert r.json()["error"] == "bad_request" and "path" in r.json()["message"]
    r = client.post("/api/analyze", headers=H, json={"path": "/x", "buckets": 0})
    assert r.status_code == 400
    r = client.post("/api/analyze", headers=H, content=b"not json")
    assert r.status_code == 400


# ------------------------------------------------------------------- analyze


def test_analyze_synthetic_wav(client: TestClient, work: Path) -> None:
    wav = _tiny_wav(work / "tiny.wav")
    r = client.post("/api/analyze", headers=H, json={"path": str(wav), "buckets": 50})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["path"] == str(wav)
    assert a["sample_rate"] == 16000 and a["channels"] == 1
    assert a["duration_s"] == pytest.approx(1.5, abs=1e-3)
    assert len(a["peaks"]["min"]) == len(a["peaks"]["max"]) == len(a["rms_db"]) == 50
    assert all(-1 <= v <= 1 for v in a["peaks"]["min"] + a["peaks"]["max"])
    assert all(-120 <= v <= 0 for v in a["rms_db"])
    assert len(a["spectrum"]["freqs_hz"]) == len(a["spectrum"]["db"]) > 50
    assert a["spectrum"]["freqs_hz"][0] == pytest.approx(40.0)
    assert a["spectrum"]["freqs_hz"][-1] <= 8000.0
    assert all(-120 <= v <= 6 for v in a["spectrum"]["db"])
    assert -60 < a["loudness"]["integrated_lufs"] < 0
    assert -40 < a["loudness"]["true_peak_dbtp"] < 0
    assert -120 <= a["noise_floor_db"] <= 0
    # default bucket count
    r = client.post("/api/analyze", headers=H, json={"path": str(wav)})
    assert r.status_code == 200 and len(r.json()["rms_db"]) == 1200


def test_analyze_path_policy_and_missing(client: TestClient, work: Path) -> None:
    r = client.post("/api/analyze", headers=H, json={"path": "/etc/passwd"})
    assert r.status_code == 403 and r.json()["error"] == "forbidden"
    r = client.post("/api/analyze", headers=H, json={"path": "relative.wav"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    r = client.post("/api/analyze", headers=H, json={"path": str(work / "nope.wav")})
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    (work / "garbage.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    r = client.post("/api/analyze", headers=H, json={"path": str(work / "garbage.wav")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_user_input"


# ---------------------------------------------------------------------- jobs


def test_job_lifecycle_status_sse_and_report(client: TestClient, work: Path) -> None:
    wav = _tiny_wav(work / "Flute 09.m4a.mp4.wav")  # suffix stacking is exercised below
    src = work / "Flute 09.m4a.mp4"
    wav.rename(src)
    r = client.post("/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"})
    assert r.status_code == 202, r.text
    body = r.json()
    assert set(body) == {"job_id", "output_path", "report_path"}
    assert body["output_path"] == str(work / "Flute 09_studio.wav")
    assert body["report_path"] == str(work / "Flute 09_studio.hawavoclean.json")
    job_id = body["job_id"]

    # SSE: first a status, then changes (>=50 ms apart), then end.
    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events?token={TOKEN}") as s:
        assert s.status_code == 200
        assert s.headers["content-type"].startswith("text/event-stream")
        name = None
        for raw in s.iter_lines():
            line = raw.rstrip("\r")
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: ") and name is not None:
                events.append((name, json.loads(line[6:])))
                name = None
    assert events[0][0] == "status"
    assert events[-1] == ("end", {})
    statuses = [e[1] for e in events if e[0] == "status"]
    assert statuses[-1]["state"] == "done"
    assert statuses[-1]["progress"] == 1.0 and statuses[-1]["stage"] == "done"
    assert "report" in statuses[-1]
    progress = [s_["progress"] for s_ in statuses]
    assert progress == sorted(progress)
    seqs = [s_["seq"] for s_ in statuses]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    snap = client.get(f"/api/jobs/{job_id}", headers=H).json()
    assert snap["state"] == "done" and snap["stage"] == "done"
    assert snap["profile"] == "studio"
    assert snap["input_path"] == str(src)
    assert snap["output_path"] == body["output_path"]
    assert snap["report_path"] == body["report_path"]
    assert snap["report"]["summary"]["enhanced"] == 1
    assert snap["started_at"] and snap["finished_at"]
    assert "error" not in snap
    for key in ("job_id", "state", "stage", "progress", "message"):
        assert key in snap

    # A finished job: SSE gives status then end immediately; cancel is a no-op 200.
    with client.stream("GET", f"/api/jobs/{job_id}/events?token={TOKEN}") as s:
        text = "".join(s.iter_text())
    assert text.count("event: status") == 1 and text.rstrip().endswith("event: end\ndata: {}")
    r = client.post(f"/api/jobs/{job_id}/cancel", headers=H)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert client.get(f"/api/jobs/{job_id}", headers=H).json()["state"] == "done"


def test_job_explicit_output_overwrite_and_production_default(
    client: TestClient, work: Path
) -> None:
    src = _tiny_wav(work / "take.wav")
    out = work / "sub" / "custom.wav"
    r = client.post(
        "/api/jobs",
        headers=H,
        json={
            "input_path": str(src),
            "profile": "production",
            "output_path": str(out),
            "overwrite": True,
        },
    )
    assert r.status_code == 202
    assert r.json()["output_path"] == str(out)
    assert r.json()["report_path"] == str(work / "sub" / "custom.hawavoclean.json")
    r = client.post("/api/jobs", headers=H, json={"input_path": str(src), "profile": "production"})
    assert r.json()["output_path"] == str(work / "take_clean.wav")
    for job in (r.json()["job_id"],):
        _wait_done(client, job)


def test_job_request_validation(client: TestClient, work: Path) -> None:
    src = _tiny_wav(work / "take.wav")
    r = client.post("/api/jobs", headers=H, json={"input_path": str(src), "profile": "turbo"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    r = client.post("/api/jobs", headers=H, json={"profile": "studio"})
    assert r.status_code == 400
    r = client.post("/api/jobs", headers=H, json={"input_path": "/etc/hosts", "profile": "studio"})
    assert r.status_code == 403
    r = client.post(
        "/api/jobs", headers=H, json={"input_path": str(work / "nope.wav"), "profile": "studio"}
    )
    assert r.status_code == 404
    r = client.post(
        "/api/jobs",
        headers=H,
        json={"input_path": str(src), "profile": "studio", "output_path": "/tmp/x.wav"},
    )
    assert r.status_code == 403
    r = client.post(
        "/api/jobs",
        headers=H,
        json={"input_path": str(src), "profile": "studio", "output_path": str(work / "o.mp3")},
    )
    assert r.status_code == 400 and ".wav" in r.json()["message"]
    assert client.get("/api/jobs/j_nope", headers=H).status_code == 404
    assert client.post("/api/jobs/j_nope/cancel", headers=H).status_code == 404
    assert client.get(f"/api/jobs/j_nope/events?token={TOKEN}").status_code == 404


def test_job_cancel_running_child(work: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    src = _tiny_wav(work / "slow.wav")
    try:
        with TestClient(app) as client:
            job_id = client.post(
                "/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"}
            ).json()["job_id"]
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                snap = client.get(f"/api/jobs/{job_id}", headers=H).json()
                if snap["stage"] == "decode":
                    break
                time.sleep(0.02)
            assert snap["state"] == "running"
            r = client.post(f"/api/jobs/{job_id}/cancel", headers=H)
            assert r.status_code == 200 and r.json() == {"ok": True}
            final = _wait_done(client, job_id, timeout=10)
            assert final["state"] == "cancelled"
            assert "error" not in final
    finally:
        manager.shutdown()


def test_job_failure_surfaces_error(work: Path) -> None:
    def factory(_r: JobRecord) -> list[str]:
        return _py(
            """
            import sys
            sys.stderr.write("Traceback...\\nValueError: decoder exploded\\n")
            sys.exit(3)
            """
        )

    manager = JobManager(command_factory=factory)
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    src = _tiny_wav(work / "bad.wav")
    try:
        with TestClient(app) as client:
            job_id = client.post(
                "/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"}
            ).json()["job_id"]
            final = _wait_done(client, job_id)
            assert final["state"] == "failed" and final["stage"] == "error"
            assert final["error"] == {
                "code": "PUBLICATION_FAILURE",
                "message": "ValueError: decoder exploded",
            }
            assert "report" not in final
    finally:
        manager.shutdown()


def test_job_real_child_end_to_end(work: Path) -> None:
    """The real contract child: ``python -m hawavoclean.cli process ... --progress-json``."""
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    src = _tiny_wav(work / "real.wav")
    try:
        with TestClient(app) as client:
            r = client.post(
                "/api/jobs",
                headers=H,
                json={"input_path": str(src), "profile": "production", "overwrite": True},
            )
            assert r.status_code == 202, r.text
            job_id = r.json()["job_id"]
            final = _wait_done(client, job_id, timeout=120)
            assert final["state"] == "done", final
            assert final["progress"] == 1.0
            assert Path(final["output_path"]).exists()
            assert Path(final["report_path"]).exists()
            assert final["report"]["output"]["path"] == final["output_path"]
            assert final["report"]["summary"]["units_total"] >= 1
    finally:
        manager.shutdown()


# --------------------------------------------------------------------- audio


def test_audio_range_requests(client: TestClient, work: Path) -> None:
    wav = _tiny_wav(work / "clip.wav")
    size = wav.stat().st_size
    data = wav.read_bytes()

    r = client.get(f"/api/audio?path={wav}&token={TOKEN}")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-type"].startswith("audio/wav")
    assert int(r.headers["content-length"]) == size
    assert r.content == data

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 0-99/{size}"
    assert r.headers["content-length"] == "100"
    assert r.content == data[:100]

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": "bytes=100-"})
    assert r.status_code == 206 and r.content == data[100:]
    assert r.headers["content-range"] == f"bytes 100-{size - 1}/{size}"

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": "bytes=-10"})
    assert r.status_code == 206 and r.content == data[-10:]

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": f"bytes=0-{size + 500}"})
    assert r.status_code == 206 and r.content == data  # end clamped

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": f"bytes={size}-"})
    assert r.status_code == 416 and r.json()["error"] == "range_not_satisfiable"
    # RFC 9110: a 416 carries Content-Range: bytes */<size> (Chromium's media
    # stack needs it to recover the resource length when seeking).
    assert r.headers["content-range"] == f"bytes */{size}"

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": "items=0-1"})
    assert r.status_code == 200  # unknown unit: ignored

    r = client.get(f"/api/audio?path={wav}", headers={**H, "Range": "bytes=5-2"})
    assert r.status_code == 200  # invalid byte-range-spec: header ignored

    r = client.head(f"/api/audio?path={wav}", headers={**H, "Range": "bytes=0-9"})
    assert r.status_code == 206 and r.content == b""
    assert r.headers["content-range"] == f"bytes 0-9/{size}"

    r = client.get(f"/api/audio?path=/etc/hosts&token={TOKEN}")
    assert r.status_code == 403
    r = client.get(f"/api/audio?path={work / 'missing.wav'}&token={TOKEN}")
    assert r.status_code == 404
    r = client.get(f"/api/audio?token={TOKEN}")
    assert r.status_code == 400


def test_audio_content_types(client: TestClient, work: Path) -> None:
    expected = {
        "a.wav": "audio/wav",
        "b.m4a": "audio/mp4",
        "c.m4a.mp4": "audio/mp4",
        "d.mp3": "audio/mpeg",
        "e.flac": "audio/flac",
        "f.aac": "audio/aac",
        "g.mov": "video/quicktime",
        "h.bin": "application/octet-stream",
    }
    for name, mime in expected.items():
        p = work / name
        p.write_bytes(b"0123456789")
        r = client.get(f"/api/audio?path={p}", headers=H)
        assert r.status_code == 200, name
        assert r.headers["content-type"].split(";")[0] == mime, name
    empty = work / "empty.wav"
    empty.write_bytes(b"")
    r = client.get(f"/api/audio?path={empty}", headers={**H, "Range": "bytes=0-1"})
    assert r.status_code == 200 and r.content == b""


def test_parse_range_edge_cases() -> None:
    assert parse_range(None, 100) is None
    assert parse_range("bytes=", 100) is None
    assert parse_range("bytes=abc", 100) is None
    assert parse_range("bytes=x-y", 100) is None
    assert parse_range("bytes=0-9,20-29", 100) == (0, 9)  # first range only
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-200", 100) == (0, 99)
    assert parse_range("bytes=5-2", 100) is None  # invalid spec: ignore (RFC 9110)
    from hawavoclean.server.app import ApiError

    with pytest.raises(ApiError) as excinfo:
        parse_range("bytes=-0", 100)
    assert excinfo.value.headers == {"Content-Range": "bytes */100"}
    with pytest.raises(ApiError) as excinfo:
        parse_range("bytes=100-", 100)
    assert excinfo.value.status == 416
    assert excinfo.value.headers == {"Content-Range": "bytes */100"}


# ------------------------------------------------------------- upload / misc


def test_upload_saves_under_work_dir(client: TestClient, work: Path) -> None:
    payload = b"RIFF" + bytes(range(256)) * 10
    r = client.post(
        "/api/upload",
        headers=H,
        files={"file": ("My Clip.wav", payload, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    saved = Path(r.json()["path"])
    assert saved.is_file() and saved.read_bytes() == payload
    assert saved.name == "My Clip.wav"
    assert saved.parent.parent == work / "uploads"
    # The uploaded path is usable by the other endpoints (policy-allowed).
    r = client.get(f"/api/audio?path={saved}", headers=H)
    assert r.status_code == 200
    # Path traversal in the filename is neutralised.
    r = client.post("/api/upload", headers=H, files={"file": ("../../evil.wav", b"x")})
    assert r.status_code == 200 and Path(r.json()["path"]).name == "evil.wav"
    # Path("..").name == "..": a bare ".." filename must not target the dir itself.
    r = client.post("/api/upload", headers=H, files={"file": ("..", b"x")})
    assert r.status_code == 200 and Path(r.json()["path"]).name == "upload.bin"
    assert Path(r.json()["path"]).read_bytes() == b"x"
    r = client.post("/api/upload", headers=H)
    assert r.status_code == 400


def test_default_output_path_rule() -> None:
    assert default_output_path(Path("/a/Flute 09.m4a.mp4"), "studio") == Path(
        "/a/Flute 09_studio.wav"
    )
    assert default_output_path(Path("/a/take.WAV"), "production") == Path("/a/take_clean.wav")
    assert default_output_path(Path("/a/x.mov"), "development") == Path("/a/x_dev.wav")


def test_shutdown_responds_then_calls_hook(work: Path) -> None:
    calls: list[float] = []
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    app = create_app(
        TOKEN, None, job_manager=manager, on_shutdown=lambda: calls.append(time.time())
    )
    try:
        with TestClient(app) as client:
            t0 = time.time()
            r = client.post("/api/shutdown", headers=H)
            assert r.status_code == 200 and r.json() == {"ok": True}
            deadline = time.time() + 2.0
            while not calls and time.time() < deadline:
                time.sleep(0.02)
            assert calls, "shutdown hook was not invoked"
            assert calls[0] - t0 < 1.0
        # Lifespan exit shuts the manager down: nothing can be queued any more.
        with pytest.raises(RuntimeError):
            manager.submit(
                input_path=work / "x.wav",
                output_path=work / "y.wav",
                profile="studio",
                overwrite=False,
            )
    finally:
        manager.shutdown()


def test_ui_dir_is_served_after_api_routes(work: Path) -> None:
    ui = work / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>HawaVoClean</title>")
    (ui / "assets" / "app.js").write_text("console.log(1)")
    manager = JobManager(command_factory=_fake_done_factory)
    app = create_app(TOKEN, ui, job_manager=manager, on_shutdown=lambda: None)
    try:
        with TestClient(app) as client:
            r = client.get("/")
            assert r.status_code == 200 and "HawaVoClean" in r.text
            assert client.get("/index.html").status_code == 200
            r = client.get("/assets/app.js")
            assert r.status_code == 200 and "console.log" in r.text
            # API still wins over the static mount and still needs the token.
            assert client.get("/api/health").status_code == 401
            assert client.get("/api/health", headers=H).status_code == 200
            r = client.get("/api/nothing", headers=H)
            assert r.status_code == 404 and r.json()["error"] == "not_found"
    finally:
        manager.shutdown()


def test_ui_dir_without_index_falls_back_to_api_only(work: Path) -> None:
    ui = work / "empty-ui"
    ui.mkdir()
    manager = JobManager(command_factory=_fake_done_factory)
    app = create_app(TOKEN, ui, job_manager=manager, on_shutdown=lambda: None)
    try:
        with TestClient(app) as client:
            r = client.get("/")
            assert r.status_code == 404 and r.json()["error"] == "not_found"
    finally:
        manager.shutdown()


def test_create_app_requires_token() -> None:
    with pytest.raises(ValueError):
        create_app("", None, on_shutdown=lambda: None)


def test_unhandled_exception_is_json_500() -> None:
    manager = JobManager(command_factory=_fake_done_factory)
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("kaboom")

    manager.get_status = boom  # type: ignore[method-assign]
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/api/jobs/j_x", headers=H)
            assert r.status_code == 500
            assert r.json()["error"] == "internal_error" and "kaboom" in r.json()["message"]
    finally:
        manager.shutdown()


def test_sse_keepalive_ping_and_submit_after_shutdown(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hawavoclean.server.app as app_mod

    monkeypatch.setattr(app_mod, "SSE_PING_INTERVAL_S", 0.1)
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    src = _tiny_wav(work / "slow.wav")
    try:
        with TestClient(app) as client:
            job_id = client.post(
                "/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"}
            ).json()["job_id"]
            # TestClient buffers streamed bodies, so cancel from a timer: the
            # stream must carry keep-alive pings until then, and then end.
            import threading

            threading.Timer(0.45, manager.cancel, args=(job_id,)).start()
            with client.stream("GET", f"/api/jobs/{job_id}/events?token={TOKEN}") as s:
                joined = "".join(s.iter_text())
            assert joined.count(": ping") >= 2
            assert joined.rstrip().endswith("event: end\ndata: {}")
            assert '"state":"cancelled"' in joined
        # After lifespan shutdown the manager refuses new work -> 503 from the API.
        app2 = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
        with TestClient(app2) as client2:
            r = client2.post(
                "/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"}
            )
            assert r.status_code == 503 and r.json()["error"] == "unavailable"
    finally:
        manager.shutdown()
