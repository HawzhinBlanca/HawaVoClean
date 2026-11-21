"""Tests for durable batch lifecycle: pause, resume, cancel, retry, batch query & events."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import JobManager, JobRecord


def _fake_command(record: JobRecord) -> list[str]:
    """Worker command that writes a deterministic output file, sidecar summary and report."""
    output = str(record.output_path)
    report = str(record.report_path)
    summary = str(record.output_path.parent / f"{record.output_path.stem}.hawavoclean.txt")
    return [
        "python3",
        "-c",
        f"""
import hashlib, json, sys, time
from pathlib import Path

out_path = Path({output!r})
out_path.parent.mkdir(parents=True, exist_ok=True)
data = b"RIFFmockwavdata"
out_path.write_bytes(data)
sha = hashlib.sha256(data).hexdigest()

sum_path = Path({summary!r})
sum_path.parent.mkdir(parents=True, exist_ok=True)
sum_path.write_text("HawaVoClean summary", encoding="utf-8")

rep_path = Path({report!r})
rep_path.parent.mkdir(parents=True, exist_ok=True)
rep_data = {{
    "metrics": {{"clean_ratio": 0.95}},
    "output": {{"sha256": sha}},
}}
rep_path.write_text(json.dumps(rep_data), encoding="utf-8")

print(json.dumps({{"event": "progress", "progress": 0.5, "stage": "processing"}}), flush=True)
print(json.dumps({{"event": "done", "status": rep_data}}), flush=True)
""",
    ]


def _fake_failing_command(_record: JobRecord) -> list[str]:
    return [
        "python3",
        "-c",
        """
import json, sys
print(json.dumps({"event": "error", "error": {"code": "CORRUPT_AUDIO", "message": "unreadable header"}}), flush=True)
sys.exit(1)
""",
    ]


@pytest.fixture
def test_env(tmp_path: Path) -> Iterator[dict[str, Any]]:
    token = "test_batch_secret_token_123"
    db_path = tmp_path / "jobs.db"
    manager = JobManager(
        command_factory=_fake_command,
        store_path=db_path,
        max_active_jobs=10,
    )
    app = create_app(token=token, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield {
            "token": token,
            "manager": manager,
            "client": client,
            "tmp_path": tmp_path,
            "db_path": db_path,
        }
    manager.shutdown()


def test_batch_submission_and_query(test_env: dict[str, Any]) -> None:
    client: TestClient = test_env["client"]
    token: str = test_env["token"]
    tmp_path: Path = test_env["tmp_path"]
    headers = {"X-Hawa-Token": token}

    f1 = tmp_path / "audio1.wav"
    f2 = tmp_path / "audio2.wav"
    f1.write_bytes(b"RIFFfake1")
    f2.write_bytes(b"RIFFfake2")

    s1 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f1)}).json()[
        "sourceId"
    ]
    s2 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f2)}).json()[
        "sourceId"
    ]

    resp = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={
            "schemaVersion": 1,
            "sourceIds": [s1, s2],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "idempotencyKey": "batch-sub-query",
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    batch_id = body["batchId"]
    assert batch_id.startswith("b_")
    assert len(body["jobs"]) == 2

    # Query list batches
    list_resp = client.get("/api/v1/batches", headers=headers)
    assert list_resp.status_code == 200
    batches = list_resp.json()["batches"]
    assert any(b["batch_id"] == batch_id for b in batches)

    # Query single batch summary
    summary_resp = client.get(f"/api/v1/batches/{batch_id}", headers=headers)
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["batch_id"] == batch_id
    assert summary["total_items"] == 2
    assert len(summary["jobs"]) == 2


def test_batch_pause_and_resume(test_env: dict[str, Any]) -> None:
    manager: JobManager = test_env["manager"]
    client: TestClient = test_env["client"]
    token: str = test_env["token"]
    tmp_path: Path = test_env["tmp_path"]
    headers = {"X-Hawa-Token": token}

    f1 = tmp_path / "pause1.wav"
    f2 = tmp_path / "pause2.wav"
    f1.write_bytes(b"RIFFfake1")
    f2.write_bytes(b"RIFFfake2")

    s1 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f1)}).json()[
        "sourceId"
    ]
    s2 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f2)}).json()[
        "sourceId"
    ]

    resp = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={
            "schemaVersion": 1,
            "sourceIds": [s1, s2],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "idempotencyKey": "batch-pause-resume",
        },
    )
    batch_id = resp.json()["batchId"]

    # Pause batch
    pause_resp = client.post(f"/api/v1/batches/{batch_id}/pause", headers=headers)
    assert pause_resp.status_code == 200
    assert pause_resp.json()["state"] == "paused"
    assert batch_id in manager._paused_batches

    # Verify summary shows paused
    summary = manager.get_batch_summary(batch_id)
    assert summary is not None
    assert summary["state"] == "paused"

    # Resume batch
    resume_resp = client.post(f"/api/v1/batches/{batch_id}/resume", headers=headers)
    assert resume_resp.status_code == 200
    assert resume_resp.json()["state"] == "running"
    assert batch_id not in manager._paused_batches


def test_batch_cancel(test_env: dict[str, Any]) -> None:
    manager: JobManager = test_env["manager"]
    client: TestClient = test_env["client"]
    token: str = test_env["token"]
    tmp_path: Path = test_env["tmp_path"]
    headers = {"X-Hawa-Token": token}

    f1 = tmp_path / "cancel1.wav"
    f2 = tmp_path / "cancel2.wav"
    f1.write_bytes(b"RIFFfake1")
    f2.write_bytes(b"RIFFfake2")

    s1 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f1)}).json()[
        "sourceId"
    ]
    s2 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f2)}).json()[
        "sourceId"
    ]

    resp = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={
            "schemaVersion": 1,
            "sourceIds": [s1, s2],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "idempotencyKey": "batch-cancel",
        },
    )
    batch_id = resp.json()["batchId"]

    # Cancel entire batch
    cancel_resp = client.post(f"/api/v1/batches/{batch_id}/cancel?wait=true", headers=headers)
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["state"] == "cancelled"

    summary = manager.get_batch_summary(batch_id)
    assert summary is not None
    for j in summary["jobs"]:
        assert j["state"] in {"cancelled", "done"}


def test_job_retry_flow(tmp_path: Path) -> None:
    token = "test_retry_token"
    db_path = tmp_path / "retry_jobs.db"
    failing_mode = True
    headers = {"X-Hawa-Token": token}

    def dynamic_command(record: JobRecord) -> list[str]:
        if failing_mode:
            return _fake_failing_command(record)
        return _fake_command(record)

    manager = JobManager(
        command_factory=dynamic_command,
        store_path=db_path,
        max_active_jobs=10,
    )
    app = create_app(token=token, job_manager=manager, on_shutdown=lambda: None)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        src = tmp_path / "retry_test.wav"
        src.write_bytes(b"RIFFsample")

        sid = client.post(
            "/api/v1/native-sources", headers=headers, json={"path": str(src)}
        ).json()["sourceId"]

        resp = client.post(
            "/api/v1/jobs",
            headers=headers,
            json={
                "schemaVersion": 1,
                "sourceIds": [sid],
                "strategy": {
                    "kind": "manual",
                    "route": "production",
                    "allowGenerativeReconstruction": False,
                },
                "executionPolicy": "offline_only",
                "conflictPolicy": "replace",
                "idempotencyKey": "job-retry-flow",
            },
        )
        assert resp.status_code == 202
        job_id = resp.json()["jobs"][0]["jobId"]

        # Wait for the job to fail
        status: dict[str, Any] = {}
        for _ in range(50):
            status = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if status["state"] == "failed":
                break
            time.sleep(0.05)
        assert status["state"] == "failed"

        # Switch to succeeding command and retry the job
        failing_mode = False
        retry_resp = client.post(f"/api/v1/jobs/{job_id}/retry", headers=headers)
        assert retry_resp.status_code == 200
        assert retry_resp.json()["state"] in {
            "queued",
            "analyzing",
            "rendering",
            "guarding",
            "publishing",
            "completed",
        }

        # Wait for it to finish successfully
        for _ in range(50):
            status = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
            if status["state"] == "completed":
                break
            time.sleep(0.05)
        assert status["state"] == "completed"
    manager.shutdown()
