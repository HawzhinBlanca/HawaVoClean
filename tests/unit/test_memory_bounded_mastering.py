"""Mastering (true-peak measurement + limiting) must run in bounded memory.

Found by profiling an 8-minute file: the full-file 4x/8x float64
oversampling in the limiter and loudness meter peaked at 5.5 GB RSS — a
30-minute recording would be OOM-killed. Memory must scale with a chunk,
not with the file.

Measured in a FRESH subprocess: RSS is a process-lifetime high-water mark,
so an in-process delta is contaminated by whatever ran before.
"""

import subprocess
import sys

import pytest

_SCRIPT = r"""
import resource, sys
import numpy as np
from hawavoclean.finishing.limiter import apply_lookahead_limiter
from hawavoclean.finishing.loudness import measure_loudness_and_peaks
SR = 48000; MINUTES = 8
n = SR * 60 * MINUTES
x = (0.4 * np.sin(2 * np.pi * 180 * np.arange(n) / SR)).astype(np.float32)
x[::SR] += 1.2
x = x[None, :]
def rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3
before = rss()
m = measure_loudness_and_peaks(x, SR)
res = apply_lookahead_limiter(x, SR, ceiling_dbtp=-1.0)
assert res.limited_waveform.shape == x.shape and m.true_peak_dbtp > -20
print(f"{rss() - before:.0f} {x.nbytes / 1e6:.0f}")
"""


@pytest.mark.slow
def test_limiter_and_meter_memory_is_bounded_on_long_audio() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    growth_mb, audio_mb = (float(v) for v in proc.stdout.split())
    budget = 600 + 6 * audio_mb
    assert growth_mb < budget, (
        f"mastering grew RSS by {growth_mb:.0f} MB on 8 min of mono audio "
        f"(budget {budget:.0f} MB) — full-file oversampling is back"
    )
