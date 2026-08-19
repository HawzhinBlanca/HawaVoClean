"""Unit tests for IsolatedEnhancementWorker lifecycle and IPC communication."""

import numpy as np
import pytest

from hawavoclean.enhancement.production import NoOpEnhancer
from hawavoclean.enhancement.worker import IsolatedEnhancementWorker


@pytest.mark.unit
def test_isolated_worker_enhance_and_close() -> None:
    worker = IsolatedEnhancementWorker(
        core_id="test-noop",
        sample_rate=48000,
        enhancer_class=NoOpEnhancer,
        timeout_s=5.0,
    )
    try:
        sig = np.zeros(4800, dtype=np.float32)
        res = worker.enhance(sig, 48000)
        assert res.sample_rate == 48000
        assert len(res.waveform) == 4800
        assert np.array_equal(res.waveform, sig)
    finally:
        worker.close()
