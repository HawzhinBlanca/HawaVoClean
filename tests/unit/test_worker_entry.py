"""In-process coverage of the worker subprocess entry loop."""

import multiprocessing as mp
from typing import Any

import numpy as np

from voiceclean.enhancement.production import NoOpEnhancer
from voiceclean.enhancement.worker import IsolatedEnhancementWorker, _worker_process_entry


class _ExplodingInit:
    def __init__(self, **_: Any) -> None:
        raise RuntimeError("init boom")


def test_worker_entry_serves_enhance_and_stop() -> None:
    ctx = mp.get_context("spawn")
    req: Any = ctx.Queue()
    resp: Any = ctx.Queue()
    audio = np.zeros(4800, dtype=np.float32)
    req.put({"type": "ENHANCE", "audio_bytes": audio.tobytes(), "sample_rate": 48000})
    req.put({"type": "STOP"})

    _worker_process_entry(req, resp, NoOpEnhancer, "noop", 48000)

    assert resp.get(timeout=5)["type"] == "READY"
    result = resp.get(timeout=5)
    assert result["type"] == "RESULT"
    assert len(np.frombuffer(result["audio_bytes"], dtype=np.float32)) == 4800


def test_worker_entry_reports_init_error() -> None:
    ctx = mp.get_context("spawn")
    req: Any = ctx.Queue()
    resp: Any = ctx.Queue()
    _worker_process_entry(req, resp, _ExplodingInit, "boom", 48000)
    msg = resp.get(timeout=5)
    assert msg["type"] == "INIT_ERROR"
    assert "init boom" in msg["error"]


def test_worker_entry_reports_enhance_error() -> None:
    class _EnhanceBoom(NoOpEnhancer):
        def enhance(self, _waveform: Any, _sample_rate: int) -> Any:
            raise ValueError("enhance boom")

    ctx = mp.get_context("spawn")
    req: Any = ctx.Queue()
    resp: Any = ctx.Queue()
    audio = np.zeros(480, dtype=np.float32)
    req.put({"type": "ENHANCE", "audio_bytes": audio.tobytes(), "sample_rate": 48000})
    req.put(None)  # None also terminates the loop

    _worker_process_entry(req, resp, _EnhanceBoom, "x", 48000)

    assert resp.get(timeout=5)["type"] == "READY"
    err = resp.get(timeout=5)
    assert err["type"] == "ERROR" and "enhance boom" in err["error"]


def test_worker_restarts_after_close_and_reuse() -> None:
    worker = IsolatedEnhancementWorker(timeout_s=30.0, enhancer_class=NoOpEnhancer)
    try:
        out1 = worker.enhance(np.zeros(4800, dtype=np.float32), 48000)
        assert out1.output_samples == 4800
        # Kill the process behind its back; next call must restart transparently.
        worker._kill_worker()
        out2 = worker.enhance(np.zeros(2400, dtype=np.float32), 48000)
        assert out2.output_samples == 2400
    finally:
        worker.close()
