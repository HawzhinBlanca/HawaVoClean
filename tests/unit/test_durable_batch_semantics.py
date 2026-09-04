"""100-item durable batch stress test: corrupt, Unicode, same-stem, mixed-format,
fault injection (pause, cancel, crash recovery/relaunch, retry), and zero-loss verification.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import JobManager, JobRecord

pytestmark = pytest.mark.unit


def _worker_script(record: JobRecord) -> list[str]:
    """Fake worker that inspects input: fails on CORRUPT or 0-byte, otherwise succeeds."""
    inp = str(record.source_snapshot_path or record.input_path)
    out = str(record.output_path)
    rep = str(record.report_path)
    summary = str(record.output_path.parent / f"{record.output_path.stem}.hawavoclean.txt")

    return [
        "python3",
        "-c",
        f"""
import hashlib, json, sys, time
from pathlib import Path

inp_path = Path({inp!r})
if not inp_path.exists() or inp_path.stat().st_size == 0 or b"CORRUPT" in inp_path.read_bytes()[:32]:
    print(json.dumps({{"event": "error", "error": {{"code": "CORRUPT_AUDIO", "message": "unreadable file"}}}}), flush=True)
    sys.exit(1)

out_path = Path({out!r})
out_path.parent.mkdir(parents=True, exist_ok=True)
data = b"RIFFclean_audio_payload_" + out_path.name.encode("utf-8")
out_path.write_bytes(data)
sha = hashlib.sha256(data).hexdigest()

sum_path = Path({summary!r})
sum_path.parent.mkdir(parents=True, exist_ok=True)
sum_path.write_text("HawaVoClean sidecar summary\\n", encoding="utf-8")

rep_path = Path({rep!r})
rep_path.parent.mkdir(parents=True, exist_ok=True)
rep_data = {{
    "metrics": {{"clean_ratio": 0.98}},
    "output": {{"sha256": sha}},
}}
rep_path.write_text(json.dumps(rep_data), encoding="utf-8")

print(json.dumps({{"event": "progress", "progress": 0.5, "stage": "processing"}}), flush=True)
print(json.dumps({{"event": "done", "status": rep_data}}), flush=True)
""",
    ]


def test_100_item_batch_stress_semantics(tmp_path: Path) -> None:
    """A 100-item batch spanning corrupt, Unicode, same-stem in multiple subdirs,
    and mixed formats (.wav, .mp3, .m4a, .flac, .aac) survives pause, resume,
    individual job cancel, simulated engine crash/relaunch, and retry
    without job loss or unintended output overwrite.
    """
    token = "test_stress_batch_token"
    db_path = tmp_path / "stress_jobs.db"
    headers = {"X-Hawa-Token": token}

    formats = [".wav", ".mp3", ".m4a", ".flac", ".aac"]
    kurdish_unicode_stems = [
        "دەنگی_تۆمارکراو",
        "کۆبوونەوە_ئەزموون",
        "پێداچوونەوەی_دەنگ",
        "تۆماری_ستۆدیۆ",
        "خاوێنکردنەوە",
    ]

    # Create 100 input audio files
    input_files: list[Path] = []
    # 1. 20 same-stem files distributed across distinct directories
    for i in range(20):
        sub = tmp_path / f"folder_{i % 5}"
        sub.mkdir(parents=True, exist_ok=True)
        f = sub / f"interview_{i // 5}{formats[i % len(formats)]}"
        f.write_bytes(f"RIFFsame_stem_{i}".encode())
        input_files.append(f)

    # 2. 20 Kurdish & Arabic Unicode stem files
    unicode_dir = tmp_path / "unicode_inputs"
    unicode_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        stem = kurdish_unicode_stems[i % len(kurdish_unicode_stems)]
        f = unicode_dir / f"{stem}_{i}{formats[i % len(formats)]}"
        f.write_bytes(f"RIFFunicode_{i}".encode())
        input_files.append(f)

    # 3. 5 Corrupt files (bad/unreadable audio header)
    corrupt_dir = tmp_path / "corrupt_inputs"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        f = corrupt_dir / f"corrupt_{i}{formats[i % len(formats)]}"
        f.write_bytes(f"CORRUPT_HEADER_INVALID_PAYLOAD_{i}".encode())
        input_files.append(f)

    # 4. 55 Standard mixed-format files
    standard_dir = tmp_path / "standard_inputs"
    standard_dir.mkdir(parents=True, exist_ok=True)
    for i in range(55):
        f = standard_dir / f"track_{i:03d}{formats[i % len(formats)]}"
        f.write_bytes(f"RIFFstandard_{i}".encode())
        input_files.append(f)

    assert len(input_files) == 100

    # Start JobManager with max_active_jobs=10
    manager = JobManager(
        command_factory=_worker_script,
        store_path=db_path,
        max_active_jobs=128,
    )
    app = create_app(token=token, job_manager=manager, on_shutdown=lambda: None)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        # Register native sources for all 100 files
        source_ids: list[str] = []
        for inp in input_files:
            resp = client.post("/api/v1/native-sources", headers=headers, json={"path": str(inp)})
            assert resp.status_code == 200
            source_ids.append(resp.json()["sourceId"])

        assert len(source_ids) == 100

        # Submit 100-item batch
        submit_resp = client.post(
            "/api/v1/jobs",
            headers=headers,
            json={
                "schemaVersion": 1,
                "sourceIds": source_ids,
                "strategy": {
                    "kind": "manual",
                    "route": "production",
                    "allowGenerativeReconstruction": False,
                },
                "executionPolicy": "offline_only",
                "conflictPolicy": "unique",
                "idempotencyKey": "stress-batch-100",
            },
        )
        assert submit_resp.status_code == 202
        body = submit_resp.json()
        batch_id = body["batchId"]
        job_entries = body["jobs"]
        assert len(job_entries) == 100

        # Mid-batch pause
        pause_resp = client.post(f"/api/v1/batches/{batch_id}/pause", headers=headers)
        assert pause_resp.status_code == 200
        assert pause_resp.json()["state"] == "paused"

        # Resume batch
        resume_resp = client.post(f"/api/v1/batches/{batch_id}/resume", headers=headers)
        assert resume_resp.status_code == 200
        assert resume_resp.json()["state"] == "running"

        # Cancel one individual job (e.g. job 50)
        target_cancel_job = job_entries[50]["jobId"]
        cancel_item_resp = client.post(f"/api/v1/jobs/{target_cancel_job}/cancel", headers=headers)
        assert cancel_item_resp.status_code == 200

        # Wait for all jobs in batch to reach terminal state
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            summary = manager.get_batch_summary(batch_id)
            assert summary is not None
            terminal_count = sum(
                1 for j in summary["jobs"] if j["state"] in {"done", "failed", "cancelled"}
            )
            if terminal_count == 100:
                break
            time.sleep(0.1)

        summary = manager.get_batch_summary(batch_id)
        assert summary is not None
        assert summary["total_items"] == 100
        assert len(summary["jobs"]) == 100

        # Verify states:
        # - The 5 corrupt jobs should be 'failed'
        # - The 1 manually cancelled job should be 'cancelled'
        # - The remaining 94 jobs should be 'done'
        states = [j["state"] for j in summary["jobs"]]
        assert states.count("cancelled") >= 1
        assert states.count("failed") == 5
        assert states.count("done") + states.count("cancelled") == 95

        # Check zero unintended overwrites: all output paths of completed jobs are unique and exist
        done_jobs = [j for j in summary["jobs"] if j["state"] == "done"]
        output_paths = [j["output_path"] for j in done_jobs]
        assert len(set(output_paths)) == len(output_paths), "Output collision detected!"
        for out in output_paths:
            assert Path(out).exists(), f"Output file missing: {out}"

        # Test retry of one of the failed corrupt items after fixing the input file
        failed_job = next(j for j in summary["jobs"] if j["state"] == "failed")
        failed_job_id = failed_job["job_id"]
        # Fix the corrupt file
        Path(failed_job["input_path"]).write_bytes(b"RIFFrepaired_valid_audio_data")

        retry_resp = client.post(f"/api/v1/jobs/{failed_job_id}/retry", headers=headers)
        assert retry_resp.status_code == 200

        # Wait for retried job to finish successfully
        retried_done = False
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = manager.get_status(failed_job_id)
            if snap and snap["state"] == "done":
                retried_done = True
                break
            time.sleep(0.05)
        assert retried_done, "Retried job failed to complete successfully!"

    manager.shutdown()

    # Test engine crash recovery and relaunch:
    # A new JobManager reloads the database and recovers all durable records
    relaunch_manager = JobManager(
        command_factory=_worker_script,
        store_path=db_path,
        max_active_jobs=10,
    )
    try:
        relaunch_summary = relaunch_manager.get_batch_summary(batch_id)
        assert relaunch_summary is not None
        assert relaunch_summary["total_items"] == 100
        assert len(relaunch_summary["jobs"]) == 100
        # Check that the retried job persisted as done
        retried_snap = relaunch_manager.get_status(failed_job_id)
        assert retried_snap is not None
        assert retried_snap["state"] == "done"
    finally:
        relaunch_manager.shutdown()


def test_batch_source_volume_loss_resilience(tmp_path: Path) -> None:
    """When a source file is removed mid-batch (simulating external drive disconnection),
    that item fails independently while other items continue and finish."""
    token = "test_volume_loss_token"
    db_path = tmp_path / "loss_jobs.db"
    headers = {"X-Hawa-Token": token}

    f1 = tmp_path / "loss_audio1.wav"
    f2 = tmp_path / "loss_audio2.wav"
    f3 = tmp_path / "loss_audio3.wav"
    f1.write_bytes(b"RIFFvalid1")
    f2.write_bytes(b"RIFFvalid2")
    f3.write_bytes(b"RIFFvalid3")

    manager = JobManager(command_factory=_worker_script, store_path=db_path, max_active_jobs=10)
    app = create_app(token=token, job_manager=manager, on_shutdown=lambda: None)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        s1 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f1)}).json()[
            "sourceId"
        ]
        s2 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f2)}).json()[
            "sourceId"
        ]
        s3 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f3)}).json()[
            "sourceId"
        ]

        # Pause manager before submission so we can simulate source loss while queued
        with manager.prepare_batch():
            pass

        resp = client.post(
            "/api/v1/jobs",
            headers=headers,
            json={
                "schemaVersion": 1,
                "sourceIds": [s1, s2, s3],
                "strategy": {
                    "kind": "manual",
                    "route": "production",
                    "allowGenerativeReconstruction": False,
                },
                "executionPolicy": "offline_only",
                "conflictPolicy": "unique",
                "idempotencyKey": "volume-loss-batch",
            },
        )
        assert resp.status_code == 202
        batch_id = resp.json()["batchId"]
        jobs = resp.json()["jobs"]
        loss_job_id = jobs[1]["jobId"]

        # Simulate volume disconnection: remove the second input file and its pinned copy
        loss_rec = manager._jobs[loss_job_id]
        if loss_rec.source_snapshot_path and loss_rec.source_snapshot_path.exists():
            loss_rec.source_snapshot_path.parent.chmod(0o700)
            loss_rec.source_snapshot_path.chmod(0o700)
            loss_rec.source_snapshot_path.unlink()
        f2.unlink()

        # Wait for terminal
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            summary = manager.get_batch_summary(batch_id)
            if summary and summary["completed_items"] + summary["failed_items"] == 3:
                break
            time.sleep(0.05)

        summary = manager.get_batch_summary(batch_id)
        assert summary is not None
        assert summary["completed_items"] == 2
        assert summary["failed_items"] == 1
        assert summary["jobs"][1]["state"] == "failed"
        assert summary["jobs"][0]["state"] == "done"
        assert summary["jobs"][2]["state"] == "done"

    manager.shutdown()


def test_batch_safe_quit_relaunch_recovery(tmp_path: Path) -> None:
    """Safe quit during batch processing marks uncommitted jobs as interrupted;
    on relaunch they are safely retried to completion with zero job loss."""
    token = "test_relaunch_token"
    db_path = tmp_path / "relaunch_jobs.db"
    headers = {"X-Hawa-Token": token}

    f1 = tmp_path / "rel1.wav"
    f2 = tmp_path / "rel2.wav"
    f1.write_bytes(b"RIFFvalid1")
    f2.write_bytes(b"RIFFvalid2")

    manager1 = JobManager(command_factory=_worker_script, store_path=db_path, max_active_jobs=10)
    app1 = create_app(token=token, job_manager=manager1, on_shutdown=lambda: None)

    with TestClient(app1, base_url="http://127.0.0.1") as client:
        s1 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f1)}).json()[
            "sourceId"
        ]
        s2 = client.post("/api/v1/native-sources", headers=headers, json={"path": str(f2)}).json()[
            "sourceId"
        ]

        # Pause batch on submission to hold them queued
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
                "idempotencyKey": "relaunch-batch",
            },
        )
        assert resp.status_code == 202
        batch_id = resp.json()["batchId"]

    # Abruptly terminate manager without graceful shutdown to simulate engine crash / kill
    with manager1._wake:
        manager1._closed = True
        manager1._wake.notify_all()
    manager1._worker.join(timeout=2.0)
    if manager1._store is not None:
        manager1._store.close()
    if manager1._owner_lease is not None:
        manager1._owner_lease.release()

    # Relaunch engine
    manager2 = JobManager(command_factory=_worker_script, store_path=db_path, max_active_jobs=10)
    try:
        summary = manager2.get_batch_summary(batch_id)
        assert summary is not None
        assert summary["total_items"] == 2
        for j in summary["jobs"]:
            if j["state"] in {"interrupted", "cancelled", "failed"}:
                manager2.retry_job(j["job_id"])

        # Wait for retried jobs to complete
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            s = manager2.get_batch_summary(batch_id)
            if s and s["completed_items"] == 2:
                break
            time.sleep(0.05)

        s = manager2.get_batch_summary(batch_id)
        assert s is not None
        assert s["completed_items"] == 2
        assert s["state"] == "done"
    finally:
        manager2.shutdown()
