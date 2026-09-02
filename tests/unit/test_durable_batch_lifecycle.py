from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from hawavoclean.server.job_store import DurableJobStore
from hawavoclean.server.jobs import JobManager


def _create_wav(path: Path, duration_s: float = 0.5, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(duration_s * sr)
    t = np.arange(samples, dtype=np.float32) / sr
    audio = 0.3 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(path, audio, sr)


def test_batch_independent_items_and_cancellation(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    manager = JobManager(store_path=db_path)

    # 1. Create a batch of 5 items with mixed names and Unicode
    files: list[Path] = []
    names = [
        "recording_1.wav",
        "کوردستان_دەنگ.wav",
        "track (session 2).wav",
        "same_stem.wav",
        "interview_final.wav",
    ]
    for name in names:
        p = tmp_path / "inputs" / name
        _create_wav(p, duration_s=0.2)
        files.append(p)

    job_ids: list[str] = []
    for f in files:
        rec = manager.submit(
            input_path=f,
            profile="production",
            output_path=tmp_path / "outputs" / f.name,
            overwrite=False,
        )
        job_ids.append(rec["job_id"])

    assert len(job_ids) == 5
    for jid in job_ids:
        snap = manager.get_status(jid)
        assert snap is not None
        assert snap["state"] in {"queued", "running", "done", "cancelled"}

    # 2. Cancel the 5th job (at the end of queue)
    cancel_target = job_ids[4]
    res = manager.cancel(cancel_target)
    assert res is True

    manager.shutdown()


def test_store_interrupted_job_recovery_on_relaunch(tmp_path: Path) -> None:
    db_path = tmp_path / "recovery_test.db"
    store = DurableJobStore(db_path)

    # Insert a job in 'running' state directly to simulate a crash/relaunch
    rec = {
        "job_id": "job_crash_sim",
        "input_path": str(tmp_path / "input.wav"),
        "output_path": str(tmp_path / "output.wav"),
        "report_path": str(tmp_path / "report.json"),
        "profile": "production",
        "overwrite": False,
        "idempotency_key": None,
        "conflict_policy": "fail",
        "request_hash": "a" * 64,
        "mode": "natural",
        "speaker_id": None,
        "cutoff_hz": None,
        "record_bundle": False,
        "bundle_path": None,
        "bundle": None,
        "source_snapshot_path": None,
        "source_snapshot_dir": None,
        "source_sha256": "b" * 64,
        "source_size_bytes": 1000,
        "state": "queued",
        "stage": "preflight",
        "progress": 0.0,
        "message": "Queued",
        "unit": None,
        "created_at": "2026-09-02T12:00:00Z",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "report": None,
        "artifact_evidence": None,
        "cancel_requested": False,
        "seq": 1,
    }
    store.reserve(
        record=rec,
        request_hash="a" * 64,
        idempotency_key=None,
        conflict_policy="fail",
    )
    rec["state"] = "running"
    rec["started_at"] = "2026-09-02T12:00:01Z"
    rec["seq"] = 2
    store.update(rec, terminal=False)

    # Close store and simulate relaunch recovery
    store.close()

    new_store = DurableJobStore(db_path)
    new_store.load_and_interrupt()
    reservation = new_store.find_job("job_crash_sim")
    assert reservation is not None
    recovered = reservation.record
    # Interrupted jobs should be transitioned to 'interrupted' state on startup
    assert recovered["state"] == "interrupted"
    assert (
        "interrupted" in str(recovered.get("message", "")).lower()
        or recovered.get("error") is not None
    )
    new_store.close()
