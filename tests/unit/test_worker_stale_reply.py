"""A timed-out request's LATE reply must never be consumed by the next
request. The worker creates fresh queues on every restart, so the old
child's reply lands in an orphaned queue. Each child encodes its PID in its
output so poisoning would be provable."""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.enhancement.worker import IsolatedEnhancementWorker
from hawavoclean.errors import WorkerError

_MARK = Path("/tmp/hawavoclean-test-slow-once-marker")


class _SlowOnce:
    def __init__(self, _core_id: str = "x", sample_rate: int = 48000, **_: Any) -> None:
        self._m = EnhancerMetadata("slow-once", "0", "t", sample_rate, True)

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._m

    def warmup(self) -> None:
        pass

    def enhance(self, w: Any, sr: int) -> EnhancementResult:
        if not _MARK.exists():
            _MARK.touch()
            time.sleep(4.0)  # first-ever child misses the deadline, replies late
        return EnhancementResult(
            np.full(len(w), float(os.getpid()), np.float32), sr, 0.1, len(w), len(w)
        )


def test_stale_reply_never_poisons_next_request() -> None:
    if _MARK.exists():
        _MARK.unlink()
    wk = IsolatedEnhancementWorker(timeout_s=2.0, enhancer_class=_SlowOnce)
    try:
        try:
            wk.enhance(np.zeros(10, np.float32), 48000)
            raise AssertionError("first request should have timed out")
        except WorkerError:
            pass
        time.sleep(3.0)  # window in which the old child's late reply arrives
        out = wk.enhance(np.zeros(10, np.float32), 48000)
        assert wk.process is not None
        assert int(out.waveform[0]) == wk.process.pid, (
            "request 2 was answered by a stale reply from the killed child"
        )
    finally:
        wk.close()
        if _MARK.exists():
            _MARK.unlink()
