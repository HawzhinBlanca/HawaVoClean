"""Unit tests for Phase E1.6: Legacy API sunsetting contract, RFC 8594 deprecation
headers, successor links, and legacy usage telemetry tracking.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hawavoclean.server.app import (
    LEGACY_REMOVAL_RELEASE,
    LEGACY_SUNSET_DATE,
    LEGACY_SUNSET_ISO_DATE,
    create_app,
)
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit
TOKEN = "t0ken"
H = {"X-Hawa-Token": TOKEN}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    manager.shutdown()


def test_legacy_jobs_list_emits_sunset_headers(client: TestClient) -> None:
    res = client.get("/api/jobs", headers=H)
    assert res.status_code == 200
    assert res.headers["sunset"] == LEGACY_SUNSET_DATE
    assert res.headers["deprecation"] == "true"
    assert res.headers["x-hawa-sunset-date"] == LEGACY_SUNSET_ISO_DATE
    assert res.headers["x-hawa-sunset-release"] == LEGACY_REMOVAL_RELEASE
    assert '</api/v1/jobs>; rel="successor-version"' in res.headers["link"]


def test_legacy_job_detail_emits_sunset_headers(client: TestClient) -> None:
    res = client.get("/api/jobs/nonexistent-job-id", headers=H)
    assert res.status_code == 404
    assert res.headers["sunset"] == LEGACY_SUNSET_DATE
    assert res.headers["deprecation"] == "true"
    assert '</api/v1/jobs/nonexistent-job-id>; rel="successor-version"' in res.headers["link"]


def test_legacy_job_cancel_emits_sunset_headers(client: TestClient) -> None:
    res = client.post("/api/jobs/nonexistent-job-id/cancel", headers=H)
    assert res.status_code == 404
    assert res.headers["sunset"] == LEGACY_SUNSET_DATE
    assert res.headers["deprecation"] == "true"
    assert (
        '</api/v1/jobs/nonexistent-job-id/cancel>; rel="successor-version"' in res.headers["link"]
    )


def test_legacy_analyze_emits_sunset_headers(client: TestClient, tmp_path: Path) -> None:
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"RIFF....WAVEfmt ....data....")
    res = client.post("/api/analyze", json={"path": str(wav_file), "buckets": 100}, headers=H)
    # Even if error or 200, deprecation headers must be present
    assert res.headers["sunset"] == LEGACY_SUNSET_DATE
    assert res.headers["deprecation"] == "true"
    assert '</api/v1/analyze>; rel="successor-version"' in res.headers["link"]


def test_legacy_audio_emits_sunset_headers(client: TestClient, tmp_path: Path) -> None:
    res = client.get(f"/api/audio?path={tmp_path / 'missing.wav'}", headers=H)
    assert res.headers["sunset"] == LEGACY_SUNSET_DATE
    assert res.headers["deprecation"] == "true"
    assert '</api/v1/jobs>; rel="successor-version"' in res.headers["link"]


def test_v1_endpoints_do_not_emit_sunset_headers(client: TestClient) -> None:
    res = client.get("/api/v1/capabilities", headers=H)
    assert res.status_code == 200
    assert "sunset" not in res.headers
    assert "deprecation" not in res.headers
    assert "x-hawa-sunset-date" not in res.headers

    res = client.get("/api/v1/jobs", headers=H)
    assert res.status_code == 200
    assert "sunset" not in res.headers
    assert "deprecation" not in res.headers


def test_health_includes_legacy_sunset_status(client: TestClient) -> None:
    # First invoke a legacy endpoint to create telemetry
    client.get("/api/jobs", headers=H)

    res = client.get("/api/health", headers=H)
    assert res.status_code == 200
    data: dict[str, Any] = res.json()
    assert "legacy_sunset" in data
    sunset_info = data["legacy_sunset"]
    assert sunset_info["sunset_date"] == LEGACY_SUNSET_ISO_DATE
    assert sunset_info["sunset_http_date"] == LEGACY_SUNSET_DATE
    assert sunset_info["removal_release"] == LEGACY_REMOVAL_RELEASE
    assert sunset_info["total_invocations"] >= 1


def test_legacy_telemetry_endpoint(client: TestClient) -> None:
    client.get("/api/jobs", headers=H)
    client.get("/api/jobs/dummy-id", headers=H)

    res = client.get("/api/v1/telemetry/legacy", headers=H)
    assert res.status_code == 200
    data = res.json()
    assert data["schema_version"] == 1
    assert data["sunset_date"] == LEGACY_SUNSET_ISO_DATE
    assert data["removal_release"] == LEGACY_REMOVAL_RELEASE
    assert data["total_invocations"] >= 2
    assert "/api/jobs" in data["routes"]
    assert data["auth_kinds"]["root"] >= 2
    assert "/api/jobs" in data["legacy_endpoints"]
    assert data["v1_successors"]["/api/jobs"] == "/api/v1/jobs"


def test_v1_jobs_events_route_accessible(client: TestClient) -> None:
    # Verify the /api/v1/jobs/{job_id}/events route exists and returns 404 for nonexistent job
    # without legacy sunset headers
    res = client.get("/api/v1/jobs/missing-job/events", headers=H)
    assert res.status_code == 404
    assert "sunset" not in res.headers
    assert "deprecation" not in res.headers
