"""Versioned, bounded Smart Safe analysis API contract."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import hawavoclean.server.app as app_module
from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit

TOKEN = "smart-analysis-token"
HEADERS = {"X-Hawa-Token": TOKEN}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    manager = JobManager()
    app = create_app(
        TOKEN,
        None,
        job_manager=manager,
        on_shutdown=lambda: None,
        min_free_bytes=0,
        max_concurrent_analyses=1,
    )
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client
    manager.shutdown()


def _wav_bytes(path: Path, seconds: float = 9.0) -> bytes:
    sample_rate = 48_000
    timeline = np.arange(round(seconds * sample_rate), dtype=np.float64) / sample_rate
    speech_like = 0.11 * np.sin(2.0 * np.pi * 180.0 * timeline) + 0.04 * np.sin(
        2.0 * np.pi * 720.0 * timeline
    )
    sf.write(path, speech_like.astype(np.float32), sample_rate, subtype="FLOAT")
    return path.read_bytes()


def test_v1_smart_analysis_uses_an_opaque_source_and_reports_unqualified_truth(
    client: TestClient,
    tmp_path: Path,
) -> None:
    upload = client.post(
        "/api/upload",
        headers=HEADERS,
        files={"file": ("voice.wav", _wav_bytes(tmp_path / "voice.wav"), "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    response = client.post(
        "/api/v1/analyze",
        headers=HEADERS,
        json={"schemaVersion": 1, "sourceId": source_id},
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["schemaVersion"] == 1
    assert report["qualification"] == "experimental_unqualified"
    assert report["valid"] is True
    assert report["sampleRate"] == 48_000
    assert report["channels"] == 1
    assert report["durationS"] == pytest.approx(9.0)
    assert report["stateBoundBytes"] < 100_000
    assert report["speechDominance"]["direction"] == "lower"
    assert report["musicRisk"]["direction"] == "upper"
    encoded = response.text
    assert source_id not in encoded
    assert str(tmp_path) not in encoded


def test_v1_smart_analysis_rejects_unknown_or_malformed_source_ids(
    client: TestClient,
) -> None:
    malformed = client.post(
        "/api/v1/analyze",
        headers=HEADERS,
        json={"schemaVersion": 1, "sourceId": "../escape"},
    )
    assert malformed.status_code == 400
    unknown = client.post(
        "/api/v1/analyze",
        headers=HEADERS,
        json={"schemaVersion": 1, "sourceId": "a" * 32},
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"] == "not_found"


@pytest.mark.parametrize("value", [0, -1, 33])
def test_analysis_pool_bounds_fail_closed(value: int) -> None:
    with pytest.raises(ValueError, match="max_concurrent_analyses"):
        create_app(TOKEN, None, max_concurrent_analyses=value)


def test_legacy_analysis_and_peaks_share_one_bounded_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Revision-1 endpoints cannot bypass the v1 decoder/FFT budget."""

    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    source = tmp_path / "voice.wav"
    source.write_bytes(b"placeholder")
    lock = threading.Lock()
    first_entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum = 0

    def bounded_work(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            first_entered.set()
        assert release.wait(timeout=5.0)
        with lock:
            active -= 1
        return {"ok": True}

    monkeypatch.setattr(app_module, "analyze_audio", bounded_work)
    monkeypatch.setattr(app_module, "compute_peaks_window", bounded_work)
    manager = JobManager()
    app = create_app(
        TOKEN,
        None,
        job_manager=manager,
        on_shutdown=lambda: None,
        min_free_bytes=0,
        max_concurrent_analyses=1,
    )
    try:
        with (
            TestClient(app, base_url="http://127.0.0.1") as test_client,
            ThreadPoolExecutor(max_workers=2) as workers,
        ):
            analyze_future = workers.submit(
                test_client.post,
                "/api/analyze",
                headers=HEADERS,
                json={"path": str(source), "buckets": 10},
            )
            assert first_entered.wait(timeout=5.0)
            peaks_future = workers.submit(
                test_client.post,
                "/api/peaks",
                headers=HEADERS,
                json={
                    "path": str(source),
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "buckets": 10,
                },
            )
            # Give the second request time to reach the shared semaphore.
            # If either legacy route bypasses it, bounded_work observes two
            # simultaneous calls before release is set.
            time.sleep(0.2)
            with lock:
                assert maximum == 1
            release.set()
            assert analyze_future.result(timeout=5.0).status_code == 200
            assert peaks_future.result(timeout=5.0).status_code == 200
        assert maximum == 1
    finally:
        release.set()
        manager.shutdown()
