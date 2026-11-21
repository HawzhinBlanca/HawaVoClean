from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean.server.app import (
    NATIVE_SOURCE_PATH,
    SESSION_PATH,
    create_app,
    ranged_file_response,
)
from hawavoclean.server.jobs import JobManager, JobRecord

TOKEN = "t0ken-test-cache-privacy"
H = {"X-Hawa-Token": TOKEN}


def _fake_done_factory(record: JobRecord) -> list[str]:
    report = {
        "schema_version": 1,
        "summary": {"units_total": 1, "enhanced": 1},
        "units": [
            {"unit_id": 0, "final_decision": "enhanced", "guard_a_verdict": "PASS"},
        ],
    }
    return [
        sys.executable,
        "-u",
        "-c",
        textwrap.dedent(
            f"""
            import json, sys
            open({str(record.report_path)!r}, "w").write({json.dumps(json.dumps(report))})
            open({str(record.output_path)!r}, "wb").write(b"RIFF\\x24\\x00\\x00\\x00WAVE")
            sys.stdout.write(json.dumps({{"event":"done","progress":1.0}}) + "\\n")
            """
        ),
    ]


@pytest.fixture
def privacy_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))

    audio_file = tmp_path / "sample.wav"
    t = np.arange(16000) / 16000
    sig = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(audio_file), sig, 16000)

    manager = JobManager(command_factory=_fake_done_factory)
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post(SESSION_PATH, headers=H)
        bearer = {"Authorization": f"Bearer {res.json()['sessionToken']}"}
        # Register native source so renderer bearer has capability
        reg = client.post(NATIVE_SOURCE_PATH, headers=H, json={"path": str(audio_file)})
        source_id = reg.json()["sourceId"]
        yield {
            "client": client,
            "bearer": bearer,
            "audio_file": audio_file,
            "source_id": source_id,
            "work": tmp_path,
            "manager": manager,
        }
    manager.shutdown()


def test_ranged_file_response_headers(tmp_path: Path) -> None:
    test_file = tmp_path / "stream.wav"
    test_file.write_bytes(b"1234567890" * 10)

    # 200 Full response
    resp_200 = ranged_file_response(test_file, None)
    assert "no-store" in resp_200.headers["cache-control"]
    assert "no-cache" in resp_200.headers["cache-control"]
    assert resp_200.headers["pragma"] == "no-cache"
    assert resp_200.headers["accept-ranges"] == "bytes"

    # 206 Partial response
    resp_206 = ranged_file_response(test_file, "bytes=0-9")
    assert "no-store" in resp_206.headers["cache-control"]
    assert "no-cache" in resp_206.headers["cache-control"]
    assert resp_206.headers["pragma"] == "no-cache"
    assert resp_206.headers["content-range"] == f"bytes 0-9/{test_file.stat().st_size}"

    # HEAD response
    resp_head = ranged_file_response(test_file, "bytes=0-9", head_only=True)
    assert "no-store" in resp_head.headers["cache-control"]
    assert "no-cache" in resp_head.headers["cache-control"]
    assert resp_head.headers["pragma"] == "no-cache"


def test_audio_endpoint_cache_control(privacy_setup: dict[str, Any]) -> None:
    client: TestClient = privacy_setup["client"]
    audio_file: Path = privacy_setup["audio_file"]

    # Full GET 200 with root header
    r = client.get("/api/audio", params={"path": str(audio_file)}, headers=H)
    assert r.status_code == 200
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc
    assert "no-cache" in cc
    assert r.headers.get("pragma") == "no-cache"

    # Range GET 206
    r = client.get(
        f"/api/audio?path={audio_file}",
        headers={**H, "Range": "bytes=0-9"},
    )
    assert r.status_code == 206
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc
    assert "no-cache" in cc
    assert r.headers.get("pragma") == "no-cache"

    # Range HEAD 206
    r = client.head(
        f"/api/audio?path={audio_file}",
        headers={**H, "Range": "bytes=0-9"},
    )
    assert r.status_code == 206
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc
    assert "no-cache" in cc
    assert r.headers.get("pragma") == "no-cache"

    # Range 416 Unsatisfiable
    size = audio_file.stat().st_size
    r = client.get(
        f"/api/audio?path={audio_file}",
        headers={**H, "Range": f"bytes={size}-"},
    )
    assert r.status_code == 416
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc
    assert r.headers.get("pragma") == "no-cache"

    # 400 Bad request (missing path)
    r = client.get("/api/audio", headers=H)
    assert r.status_code == 400
    cc = r.headers.get("cache-control", "")
    assert "no-store" in cc
    assert r.headers.get("pragma") == "no-cache"


def test_api_routes_cache_control(privacy_setup: dict[str, Any]) -> None:
    client: TestClient = privacy_setup["client"]
    bearer: dict[str, str] = privacy_setup["bearer"]
    audio_file: Path = privacy_setup["audio_file"]

    # /api/health
    r = client.get("/api/health", headers=bearer)
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("pragma") == "no-cache"

    # /api/v1/capabilities
    r = client.get("/api/v1/capabilities", headers=bearer)
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("pragma") == "no-cache"

    # /api/peaks
    r = client.post(
        "/api/peaks",
        headers=H,
        json={"path": str(audio_file), "start_s": 0.0, "end_s": 1.0, "buckets": 10},
    )
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("pragma") == "no-cache"

    # /api/upload
    r = client.post(
        "/api/upload",
        headers=bearer,
        files={
            "file": ("test.wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00" + b"\x00" * 32)
        },
    )
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("pragma") == "no-cache"


def test_security_errors_cache_control(privacy_setup: dict[str, Any]) -> None:
    client: TestClient = privacy_setup["client"]
    client.cookies.clear()

    # 401 Unauthorized
    r = client.get("/api/audio")
    assert r.status_code == 401
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("pragma") == "no-cache"

    # 403 Forbidden (disallowed origin)
    r = client.get("/api/health", headers={"Origin": "https://malicious.invalid"})
    assert r.status_code == 403
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("pragma") == "no-cache"
