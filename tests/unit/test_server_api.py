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
    PROFILES,
    SESSION_PATH,
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
    # Point the profiles root away from the checkout: health's speaker list
    # must be what a test stages, not whatever the repo's profiles/ holds.
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))
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
    report = {
        "schema_version": 1,
        "summary": {"units_total": 2, "enhanced": 1},
        "units": [
            {"unit_id": 0, "final_decision": "enhanced", "guard_a_verdict": "PASS"},
            {"unit_id": 1, "final_decision": "reverted", "guard_a_verdict": "FAIL"},
        ],
    }
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
    with TestClient(app, base_url="http://127.0.0.1") as c:
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


def test_missing_or_wrong_header_is_401_and_url_tokens_are_rejected(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"
    assert "message" in r.json()
    r = client.get("/api/health", headers={"X-Hawa-Token": "wrong"})
    assert r.status_code == 401
    r = client.get("/api/health?token=wrong")
    assert r.status_code == 400 and r.json()["error"] == "query_auth_forbidden"
    r = client.get(f"/api/health?token={TOKEN}", headers=H)
    assert r.status_code == 400  # URL credentials are refused even with a valid header
    r = client.post("/api/jobs", json={})
    assert r.status_code == 401
    # Opaque/file origins are not trusted broker clients.
    r = client.get("/api/health", headers={"Origin": "null"})
    assert r.status_code == 403 and "access-control-allow-origin" not in r.headers


def test_token_by_header_and_short_lived_session(client: TestClient) -> None:
    r = client.get("/api/health", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "version": __version__,
        "profiles": ["studio", "lowband", "production"],
        "speakers": [],  # the fixture's profiles root is empty
        "restore_available": False,
        "engine_pid": body["engine_pid"],
        "storage": {
            "managed_upload_bytes": 0,
            "managed_upload_limit_bytes": 16 * 1024 * 1024 * 1024,
            "minimum_free_bytes": 512 * 1024 * 1024,
        },
        "jobs": {
            "durable": False,
            "persistence_ok": True,
            "persistence_error": None,
        },
    }
    assert isinstance(body["engine_pid"], int)

    session = client.post(SESSION_PATH, headers=H)
    assert session.status_code == 200
    assert session.headers["cache-control"] == "no-store"
    assert "HttpOnly" in session.headers["set-cookie"]
    capability = session.json()["sessionToken"]
    client.cookies.clear()
    assert (
        client.get("/api/health", headers={"Authorization": f"Bearer {capability}"}).status_code
        == 200
    )


def test_health_lists_speakers_from_the_profiles_root(client: TestClient, work: Path) -> None:
    """`speakers` = sorted ids with a profile.json under HAWAVOCLEAN_PROFILES_DIR;
    loose research profiles never make production Restore available. Recomputed
    per request, so research diagnostics still see additions without restart."""
    body = client.get("/api/health", headers=H).json()
    assert body["speakers"] == [] and body["restore_available"] is False  # dir absent
    profiles = work / "profiles"
    for spk in ("character_02", "character_01"):  # staged out of order: response sorts
        (profiles / spk).mkdir(parents=True)
        (profiles / spk / "profile.json").write_text("{}")
    (profiles / "half_trained").mkdir()  # no profile.json: not offered to the UI
    (profiles / "schema.json").write_text("{}")  # a stray file is not a speaker
    body = client.get("/api/health", headers=H).json()
    assert body["speakers"] == ["character_01", "character_02"]
    assert body["restore_available"] is False


def test_v1_capabilities_are_maturity_bound_not_file_presence(
    client: TestClient, work: Path
) -> None:
    profile = work / "profiles" / "speaker_1"
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text("{}")

    response = client.get("/api/v1/capabilities", headers=H)
    assert response.status_code == 200
    capabilities = {item["capabilityId"]: item for item in response.json()["capabilities"]}
    assert capabilities["production"]["maturity"] == "qualified"
    assert capabilities["preserve"]["available"] is False
    assert capabilities["preserve"]["maturity"] == "blocked"
    assert "Smart Safe candidate" in capabilities["preserve"]["reason"]
    assert capabilities["smart_analysis"]["maturity"] == "experimental"
    assert capabilities["smart_analysis"]["available"] is True
    assert capabilities["smart_safe"]["maturity"] == "blocked"
    assert capabilities["restore_source"]["available"] is False
    assert capabilities["restore_enrolled"]["available"] is False
    assert client.get("/api/health", headers=H).json()["restore_available"] is False


def test_v1_job_is_batch_idempotent_after_upload_retention(client: TestClient) -> None:
    upload = client.post(
        "/api/upload",
        headers=H,
        files={"file": ("source.wav", b"source", "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]
    source_path = Path(upload.json()["path"])
    assert len(source_id) == 32

    request = {
        "schemaVersion": 1,
        "sourceIds": [source_id],
        "strategy": {
            "kind": "manual",
            "route": "production",
            "allowGenerativeReconstruction": False,
        },
        "executionPolicy": "offline_only",
        "conflictPolicy": "unique",
        "recordBundle": False,
        "idempotencyKey": "v1-desktop-request",
    }
    first = client.post("/api/v1/jobs", headers=H, json=request)
    repeated = client.post("/api/v1/jobs", headers=H, json=request)
    assert first.status_code == repeated.status_code == 202
    assert repeated.json() == first.json()
    job_id = first.json()["jobs"][0]["jobId"]
    assert _wait_done(client, job_id)["state"] == "done"
    status = client.get(f"/api/v1/jobs/{job_id}", headers=H)
    assert status.status_code == 200
    assert status.json()["state"] == "completed"
    history = client.get("/api/v1/jobs", headers=H)
    assert history.status_code == 200
    assert any(job["jobId"] == job_id for job in history.json()["jobs"])

    deadline = time.monotonic() + 5
    while source_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not source_path.exists()  # retention has removed the source and marker
    after_retention = client.post("/api/v1/jobs", headers=H, json=request)
    assert after_retention.status_code == 202
    assert after_retention.json() == first.json()

    changed = client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            **request,
            "strategy": {
                "kind": "manual",
                "route": "studio",
                "allowGenerativeReconstruction": False,
            },
        },
    )
    assert changed.status_code == 409 and "different request" in changed.json()["message"]


def test_v1_unknown_job_and_cancel_are_explicit(client: TestClient) -> None:
    assert client.get("/api/v1/jobs/j_missing", headers=H).status_code == 404
    assert client.post("/api/v1/jobs/j_missing/cancel", headers=H).status_code == 404


def test_v1_unqualified_routes_fail_closed_and_record_bundle_is_scheduled(
    client: TestClient,
) -> None:
    base = {
        "schemaVersion": 1,
        "sourceIds": ["a" * 32],
        "executionPolicy": "offline_only",
        "conflictPolicy": "unique",
        "recordBundle": False,
        "idempotencyKey": "blocked-request",
    }
    smart = client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            **base,
            "strategy": {
                "kind": "smart_safe",
                "restorePolicy": "disabled",
                "allowGenerativeReconstruction": False,
            },
        },
    )
    assert smart.status_code == 503 and smart.json()["error"] == "capability_blocked"
    upload = client.post(
        "/api/upload",
        headers=H,
        files={"file": ("bundle-source.wav", b"source", "audio/wav")},
    )
    assert upload.status_code == 200
    bundle = client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            **base,
            "sourceIds": [upload.json()["source_id"]],
            "recordBundle": True,
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
        },
    )
    assert bundle.status_code == 202
    item = bundle.json()["jobs"][0]
    assert item["recordBundle"] is True
    assert item["bundlePath"].endswith(".hawavoclean.zip")
    # This fixture intentionally speaks only the old fake child protocol, so
    # the broker must fail it rather than claim bundle completion.
    job_id = item["jobId"]
    assert _wait_done(client, job_id)["state"] == "failed"
    status = client.get(f"/api/v1/jobs/{job_id}", headers=H).json()
    assert status["state"] == "failed"
    assert "bundle" not in status


def test_cors_preflight_needs_no_token(client: TestClient) -> None:
    r = client.options(
        "/api/jobs",
        headers={
            "Origin": "hawa://app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-hawa-token,content-type",
        },
    )
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == "hawa://app"
    assert r.headers["access-control-allow-credentials"] == "true"
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


def test_unstorable_paths_are_400_on_every_endpoint_that_takes_one(
    client: TestClient, work: Path
) -> None:
    """A NUL byte or an unpaired surrogate used to reach ``lstat`` and come
    back as ``500 {"error":"internal_error","message":"ValueError: lstat:
    embedded null character in path"}`` — reachable from the UI's documented
    ``?file=`` autoload. Every path input now refuses it by design."""
    src = _tiny_wav(work / "ok.wav")
    nul = f"{work}/a\\u0000b.wav"
    surrogate = f"{work}/a\\ud800b.wav"
    hdr = {**H, "Content-Type": "application/json"}

    for body in (
        f'{{"path": "{nul}"}}',
        f'{{"path": "{surrogate}"}}',
    ):
        for route in ("/api/analyze", "/api/peaks"):
            r = client.post(route, headers=hdr, content=body.encode())
            assert r.status_code == 400, (route, body, r.status_code, r.text)
            assert r.json()["error"] == "bad_request"

    for body in (
        f'{{"input_path": "{nul}", "profile": "studio"}}',
        f'{{"input_path": "{surrogate}", "profile": "studio"}}',
        f'{{"input_path": "{src}", "profile": "studio", "output_path": "{nul}"}}',
    ):
        r = client.post("/api/jobs", headers=hdr, content=body.encode())
        assert r.status_code == 400, (body, r.status_code, r.text)
        assert r.json()["error"] == "bad_request"

    # The query-string route: a browser can percent-encode a NUL.
    r = client.get(f"/api/audio?path={work}/a%00b.wav", headers=H)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"

    # And a name a POSIX filesystem *can* hold still gets the ordinary answer.
    r = client.post("/api/analyze", headers=H, json={"path": f"{work}/a\nb.wav"})
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_upload_filename_with_a_nul_byte_is_stored_not_a_500(
    client: TestClient, work: Path
) -> None:
    """httpx percent-encodes a filename, so this needs a hand-built body. The
    NUL used to reach ``open()`` — and then the cleanup ``unlink()``, whose
    own ValueError replaced the original error in the 500."""
    boundary = "----hawa"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="a\x00b.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + b"RIFFdata"
        + f"\r\n--{boundary}--\r\n".encode()
    )
    r = client.post(
        "/api/upload",
        headers={**H, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        content=body,
    )
    assert r.status_code == 200, r.text
    saved = Path(r.json()["path"])
    assert saved.name == "ab.wav" and saved.read_bytes() == b"RIFFdata"
    assert saved.parent.parent == work / "uploads"


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
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=H) as s:
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
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=H) as s:
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


def test_job_idempotency_conflicts_unique_names_and_history(client: TestClient, work: Path) -> None:
    src = _tiny_wav(work / "durable.wav")
    output = work / "master.wav"
    request = {
        "input_path": str(src),
        "profile": "production",
        "output_path": str(output),
        "idempotency_key": "desktop-req-1",
        "conflict_policy": "fail",
    }
    first = client.post("/api/jobs", headers=H, json=request)
    repeated = client.post("/api/jobs", headers=H, json=request)
    assert first.status_code == repeated.status_code == 202
    assert repeated.json()["job_id"] == first.json()["job_id"]

    changed = client.post(
        "/api/jobs",
        headers=H,
        json={**request, "output_path": str(work / "changed.wav")},
    )
    assert changed.status_code == 409 and "different request" in changed.json()["message"]

    collision = client.post(
        "/api/jobs",
        headers=H,
        json={**request, "idempotency_key": "desktop-req-2"},
    )
    assert collision.status_code == 409

    unique = client.post(
        "/api/jobs",
        headers=H,
        json={
            **request,
            "idempotency_key": "desktop-req-3",
            "conflict_policy": "unique",
        },
    )
    assert unique.status_code == 202
    assert Path(unique.json()["output_path"]).name == "master (2).wav"

    history = client.get("/api/jobs", headers=H)
    assert history.status_code == 200
    ids = {job["job_id"] for job in history.json()["jobs"]}
    assert {first.json()["job_id"], unique.json()["job_id"]}.issubset(ids)


def test_overwrite_compatibility_cannot_contradict_conflict_policy(
    client: TestClient, work: Path
) -> None:
    src = _tiny_wav(work / "conflict.wav")
    response = client.post(
        "/api/jobs",
        headers=H,
        json={
            "input_path": str(src),
            "profile": "production",
            "overwrite": True,
            "conflict_policy": "fail",
        },
    )
    assert response.status_code == 422
    assert "overwrite=true conflicts" in response.json()["message"]


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
    assert client.get("/api/jobs/j_nope/events", headers=H).status_code == 404


def test_restore_job_request_validation(client: TestClient, work: Path) -> None:
    """Contract addendum 2: the cross-field restore rules are 422s with one
    clear message — a combination the schema knows but the contract refuses."""
    src = _tiny_wav(work / "take.wav")
    j = {"input_path": str(src), "profile": "studio"}
    r = client.post("/api/jobs", headers=H, json={**j, "mode": "restore"})
    assert r.status_code == 422 and r.json()["error"] == "bad_request"
    assert "speaker_id" in r.json()["message"]
    # An id that could not name a profile dir (or would need argv escaping)
    # is refused before it can travel into a child command line.
    for bad in ("Character_01", "char-01", "../evil", "a b", "ch;rm", "x" * 65):
        r = client.post("/api/jobs", headers=H, json={**j, "mode": "restore", "speaker_id": bad})
        assert r.status_code == 422, bad
        assert r.json() == {
            "error": "bad_request",
            "message": "speaker_id must match ^[a-z0-9_]{1,64}$",
        }
    # Restore-only fields outside restore mode: 422, never silently dropped.
    r = client.post("/api/jobs", headers=H, json={**j, "cutoff_hz": 7800.0})
    assert r.status_code == 422 and "cutoff_hz" in r.json()["message"]
    r = client.post("/api/jobs", headers=H, json={**j, "speaker_id": "character_01"})
    assert r.status_code == 422 and "speaker_id" in r.json()["message"]
    # Malformed values keep the revision-1 400 (only the cross-field rules 422).
    r = client.post(
        "/api/jobs", headers=H, json={**j, "mode": "restore", "speaker_id": "c", "cutoff_hz": -1}
    )
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_unknown_request_fields_are_422_not_silently_ignored(
    client: TestClient, work: Path
) -> None:
    """Every request model is pinned ``extra="forbid"``: a misspelled field
    used to be dropped and the request would "succeed" (audit finding)."""
    src = _tiny_wav(work / "take.wav")
    r = client.post(
        "/api/jobs",
        headers=H,
        json={"input_path": str(src), "profile": "studio", "speaker": "character_01"},
    )
    assert r.status_code == 422 and r.json()["error"] == "bad_request"
    assert "speaker" in r.json()["message"]
    r = client.post("/api/analyze", headers=H, json={"path": str(src), "bucket": 9})
    assert r.status_code == 422 and r.json()["error"] == "bad_request"
    r = client.post(
        "/api/peaks",
        headers=H,
        json={"path": str(src), "start_s": 0.0, "end_s": 1.0, "bucketz": 9},
    )
    assert r.status_code == 422 and r.json()["error"] == "bad_request"


def test_legacy_restore_is_fail_closed_even_when_a_loose_profile_exists(
    client: TestClient, work: Path
) -> None:
    src = _tiny_wav(work / "take.wav")
    # Stage the speaker the job names. /api/health is the contract for which
    # speakers exist, and a job may not name one it does not list.
    (work / "profiles" / "character_01").mkdir(parents=True)
    (work / "profiles" / "character_01" / "profile.json").write_text("{}")
    r = client.post(
        "/api/jobs",
        headers=H,
        json={
            "input_path": str(src),
            "profile": "studio",
            "mode": "restore",
            "speaker_id": "character_01",
            "cutoff_hz": 7800.0,
        },
    )
    assert r.status_code == 503, r.text
    assert r.json()["error"] == "capability_blocked"
    assert "qualified signed Sorani Restore pack" in r.json()["message"]
    assert client.get("/api/jobs", headers=H).json()["count"] == 0
    # A natural job's snapshot carries mode but no restore-only keys at all.
    r = client.post(
        "/api/jobs",
        headers=H,
        json={"input_path": str(src), "profile": "studio", "overwrite": True},
    )
    assert r.status_code == 202, r.text
    snap = client.get(f"/api/jobs/{r.json()['job_id']}", headers=H).json()
    assert snap["mode"] == "natural"
    assert "speaker_id" not in snap and "cutoff_hz" not in snap
    _wait_done(client, r.json()["job_id"])


def test_job_cancel_running_child(work: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    src = _tiny_wav(work / "slow.wav")
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        with TestClient(app, base_url="http://127.0.0.1") as client:
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

    r = client.get("/api/audio", params={"path": str(wav)}, headers=H)
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

    r = client.get("/api/audio", params={"path": "/etc/hosts"}, headers=H)
    assert r.status_code == 403
    r = client.get("/api/audio", params={"path": str(work / "missing.wav")}, headers=H)
    assert r.status_code == 404
    r = client.get("/api/audio", headers=H)
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
    assert default_output_path(Path("/a/x.mp4"), "development") == Path("/a/x_dev.wav")
    # Every offered profile needs its own suffix, or two profiles' masters
    # collide on one name and the second silently overwrites the first.
    assert default_output_path(Path("/a/take.wav"), "lowband") == Path("/a/take_lowband.wav")
    suffixes = [str(default_output_path(Path("/a/t.wav"), p)) for p in PROFILES]
    assert len(set(suffixes)) == len(suffixes), suffixes


def test_shutdown_responds_then_calls_hook(work: Path) -> None:
    calls: list[float] = []
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    app = create_app(
        TOKEN, None, job_manager=manager, on_shutdown=lambda: calls.append(time.time())
    )
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        with TestClient(app, base_url="http://127.0.0.1") as client:
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
        with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
            r = client.get("/api/jobs/j_x", headers=H)
            assert r.status_code == 500
            body = r.json()
            assert body["error"] == "internal_error"
            assert "kaboom" not in body["message"]
            assert body["request_id"] == r.headers["x-hawa-request-id"]
            assert body["request_id"].startswith("req_")
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
        with TestClient(app, base_url="http://127.0.0.1") as client:
            job_id = client.post(
                "/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"}
            ).json()["job_id"]
            # TestClient buffers streamed bodies, so cancel from a timer: the
            # stream must carry keep-alive pings until then, and then end.
            import threading

            threading.Timer(0.45, manager.cancel, args=(job_id,)).start()
            with client.stream("GET", f"/api/jobs/{job_id}/events", headers=H) as s:
                joined = "".join(s.iter_text())
            assert joined.count(": ping") >= 2
            assert joined.rstrip().endswith("event: end\ndata: {}")
            assert '"state":"cancelled"' in joined
        # After lifespan shutdown the manager refuses new work -> 503 from the API.
        app2 = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
        with TestClient(app2, base_url="http://127.0.0.1") as client2:
            r = client2.post(
                "/api/jobs", headers=H, json={"input_path": str(src), "profile": "studio"}
            )
            assert r.status_code == 503 and r.json()["error"] == "unavailable"
    finally:
        manager.shutdown()


def test_a_job_may_not_name_a_speaker_health_does_not_list(client: TestClient, work: Path) -> None:
    """An unknown speaker_id is answerable at submit, so it is answered there.

    The id used to pass the grammar check, get queued, and spawn a child that
    enhanced the entire file before restoration looked the profile up and
    failed. /api/health already publishes the installed speakers and the UI
    builds its picker from that list -- the same list answers this now.
    """
    src = _tiny_wav(work / "take.wav")
    (work / "profiles" / "character_01").mkdir(parents=True)
    (work / "profiles" / "character_01" / "profile.json").write_text("{}")

    listed = client.get("/api/health", headers=H).json()["speakers"]
    assert listed == ["character_01"]

    r = client.post(
        "/api/jobs",
        headers=H,
        json={
            "input_path": str(src),
            "profile": "studio",
            "mode": "restore",
            "speaker_id": "character_99",
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"] == "bad_request"
    assert "character_99" in r.json()["message"]


def test_v1_override_unknown_job_returns_404(client: TestClient) -> None:
    res = client.post(
        "/api/v1/jobs/unknown-job-id-9999/override",
        headers=H,
        json={"unit_index": 0, "decision": "force_original"},
    )
    assert res.status_code == 404
    assert res.json()["error"] == "not_found"


def test_v1_override_workflow_lifecycle(client: TestClient, work: Path) -> None:
    src = _tiny_wav(work / "override_source.wav")
    upload = client.post(
        "/api/upload",
        headers=H,
        files={"file": ("override.wav", src.read_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    res = client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            "schemaVersion": 1,
            "sourceIds": [source_id],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "recordBundle": False,
            "idempotencyKey": "test-override-key-1",
        },
    )
    assert res.status_code == 202
    job_id = res.json()["jobs"][0]["jobId"]
    assert _wait_done(client, job_id)["state"] == "done"

    # Out of range unit index -> 400
    bad_idx = client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unit_index": 999999, "decision": "force_original"},
    )
    assert bad_idx.status_code == 400
    assert bad_idx.json()["error"] == "invalid_unit"

    # Force original on unit 0
    override1 = client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unit_index": 0, "decision": "force_original"},
    )
    assert override1.status_code == 200
    data1 = override1.json()
    assert data1["job_id"] == job_id
    assert data1["unit_index"] == 0
    assert data1["new_decision"] == "reverted"
    assert data1["override"] == "force_original"

    # Force enhanced on unit 0
    override2 = client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unit_index": 0, "decision": "force_enhanced"},
    )
    assert override2.status_code == 200
    data2 = override2.json()
    assert data2["new_decision"] == "enhanced"
    assert data2["override"] == "force_enhanced"

    # Auto restore on unit 0
    override3 = client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unit_index": 0, "decision": "auto"},
    )
    assert override3.status_code == 200
    data3 = override3.json()
    assert data3["override"] == "auto"

