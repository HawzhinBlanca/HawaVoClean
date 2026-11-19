"""Tests for production health signals, observability, alerts, and privacy purity."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hawavoclean.server.app import create_app
from hawavoclean.server.health_signals import (
    RUNBOOK_ANCHORS,
    HealthSignalsService,
    sanitize_telemetry,
)
from hawavoclean.server.jobs import JobManager

TOKEN = "test-health-token-secret-12345"
AUTH_HEADER = {"X-Hawa-Token": TOKEN}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    manager = JobManager(store_path=tmp_path / "jobs.db")
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    return TestClient(app, base_url="http://127.0.0.1")


def test_nominal_health_signals(client: TestClient) -> None:
    response = client.get("/api/v1/health/signals", headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == 1
    assert data["status"] == "ok"
    assert data["healthy"] is True
    assert data["alerts"] == []
    assert set(data["signals"]) == {
        "queue_depth",
        "no_progress_timeout",
        "process_lease_loss",
        "disk_pressure",
    }
    for name, sig in data["signals"].items():
        assert sig["signal"] == name
        assert sig["status"] == "ok"
        assert sig["alert"] is None


def test_runbook_anchors_exist_in_documentation() -> None:
    operations_doc = Path("docs/operations.md").read_text(encoding="utf-8")
    for signal_name, runbook_url in RUNBOOK_ANCHORS.items():
        anchor = runbook_url.split("#")[1]
        assert f'<a id="{anchor}">' in operations_doc or f"#{anchor}" in operations_doc, (
            f"Missing anchor {anchor} for {signal_name}"
        )


def test_health_signals_alert_conditions(tmp_path: Path) -> None:
    manager = SimpleNamespace(
        _max_active_jobs=10,
        get_queue_depth=lambda: 10,
        check_no_progress_timeouts=lambda _t: [
            {"job_id": "job-1", "elapsed_seconds": 360.0, "threshold_seconds": 300.0}
        ],
        check_process_and_lease_health=lambda: (False, "Owner lease lost"),
    )
    upload_store = SimpleNamespace(
        min_free_bytes=512 * 1024 * 1024,
        root=tmp_path,
        _disk_usage=lambda _p: SimpleNamespace(free=300 * 1024 * 1024),
    )
    svc = HealthSignalsService(job_manager=manager, upload_store=upload_store)
    data = svc.evaluate_all()
    assert data["status"] == "critical"
    assert data["healthy"] is False
    alert_ids = {a["alert_id"] for a in data["alerts"]}
    assert alert_ids == {
        "QUEUE_DEPTH_EXCEEDED",
        "JOB_NO_PROGRESS_STALLED",
        "PROCESS_OR_LEASE_LOST",
        "DISK_PRESSURE_CRITICAL",
    }

    # Warning thresholds
    manager_warn = SimpleNamespace(
        _max_active_jobs=128,
        get_queue_depth=lambda: 105,
        check_no_progress_timeouts=lambda _t: [],
        check_process_and_lease_health=lambda: (True, None),
    )
    upload_warn = SimpleNamespace(
        min_free_bytes=512 * 1024 * 1024,
        root=tmp_path,
        _disk_usage=lambda _p: SimpleNamespace(free=700 * 1024 * 1024),
    )
    warn_data = HealthSignalsService(
        job_manager=manager_warn, upload_store=upload_warn
    ).evaluate_all()
    assert warn_data["status"] == "warning"
    assert {a["alert_id"] for a in warn_data["alerts"]} == {
        "QUEUE_DEPTH_WARNING",
        "DISK_PRESSURE_WARNING",
    }


def test_zero_leakage_privacy_purity() -> None:
    manager = SimpleNamespace(
        _max_active_jobs=128,
        get_queue_depth=lambda: 0,
        check_no_progress_timeouts=lambda _t: [
            {
                "job_id": "safe-uuid-1234",
                "elapsed_seconds": 400.0,
                "file_path": "/Users/hawzhin/private/confidential_interview.wav",
                "auth": "Bearer secret_token_xyz987",
                "transcript": "Classified audio spoken words",
            }
        ],
        check_process_and_lease_health=lambda: (True, None),
    )
    raw_json = json.dumps(HealthSignalsService(job_manager=manager).evaluate_all())
    assert "confidential_interview.wav" not in raw_json
    assert "/Users/hawzhin" not in raw_json
    assert "secret_token_xyz987" not in raw_json
    assert "Classified audio" not in raw_json
    assert not re.search(r"/[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+", raw_json)
    assert not re.search(r"\.(wav|mp3|m4a|flac|ogg)\b", raw_json, re.IGNORECASE)


def test_sanitize_telemetry_unit() -> None:
    dirty = {
        "job_id": "uuid-123",
        "path": "/home/user/test.wav",
        "nested": {"file": "song.flac", "count": 42, "token": "secret123"},
        "list": ["/var/log/app.log", "safe_word", 100],
    }
    clean = sanitize_telemetry(dirty)
    clean_json = json.dumps(clean)
    assert "/home/user/test.wav" not in clean_json
    assert "song.flac" not in clean_json
    assert "secret123" not in clean_json
    assert "/var/log/app.log" not in clean_json
    assert clean["job_id"] == "uuid-123"
    assert clean["nested"]["count"] == 42
