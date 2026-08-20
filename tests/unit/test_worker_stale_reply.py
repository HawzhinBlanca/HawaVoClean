"""A timed-out request's LATE reply must never be consumed by the next
request. The worker creates fresh queues on every restart, so the old
child's reply lands in an orphaned queue. Each child encodes its PID in its
output so poisoning would be provable.

The "first call is slow" marker travels by ENVIRONMENT VARIABLE, not by a
fixed path. The enhancer runs in a spawned child, so the two processes have
to agree on the file without passing an argument — but a constant like
``/tmp/hawavoclean-test-slow-once-marker`` is shared by every run on the
machine, and two suites running at once (a second checkout, a parallel gate)
then clear and create each other's marker: the first request does not time
out, and this test fails for a reason that has nothing to do with the
worker. Measured: 1 failure in 5 isolated runs while a second full suite was
running; 0 in 10 after this change, including 4 deliberately concurrent
pairs. The child inherits the environment, so a per-test path in ``tmp_path``
reaches it and stays inside the sandbox ``tests/conftest.py`` insists on.
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.enhancement.worker import IsolatedEnhancementWorker
from hawavoclean.errors import WorkerError

MARK_ENV = "HAWAVOCLEAN_TEST_SLOW_ONCE_MARKER"


def _mark() -> Path:
    return Path(os.environ[MARK_ENV])


class _SlowOnce:
    def __init__(self, _core_id: str = "x", sample_rate: int = 48000, **_: Any) -> None:
        self._m = EnhancerMetadata("slow-once", "0", "t", sample_rate, True)

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._m

    def warmup(self) -> None:
        pass

    def enhance(self, w: Any, sr: int) -> EnhancementResult:
        mark = _mark()
        if not mark.exists():
            mark.touch()
            time.sleep(4.0)  # first-ever child misses the deadline, replies late
        return EnhancementResult(
            np.full(len(w), float(os.getpid()), np.float32), sr, 0.1, len(w), len(w)
        )


def test_stale_reply_never_poisons_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MARK_ENV, str(tmp_path / "slow-once-marker"))
    assert not _mark().exists()
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
