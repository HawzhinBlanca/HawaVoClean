"""Interrupting the CLI mid-processing must leave no partial outputs at the
destination and no orphaned worker processes — for SIGINT, SIGTERM, and
even SIGKILL of the parent (the child notices its parent died)."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

CLI = str(Path(sys.executable).with_name("hawavoclean"))


@pytest.fixture(scope="module")
def long_input(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("long") / "long.wav"
    sr = 48000
    t = np.arange(sr * 90) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    sf.write(str(p), x, sr)
    return p


def _spawn_children_of(pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


@pytest.mark.chaos
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_interrupt_leaves_no_partials_and_no_orphans(
    sig: signal.Signals, long_input: Path, tmp_path: Path
) -> None:
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    env = {**os.environ, "HAWAVOCLEAN_WORK_DIR": str(tmp_path / "work")}
    proc = subprocess.Popen(
        [CLI, "process", str(long_input), "-o", str(dest_dir / "o.wav"), "--overwrite"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Let it get past startup and into unit processing, then capture the child.
    children: list[int] = []
    deadline = time.time() + 20
    while time.time() < deadline and not children:
        time.sleep(0.25)
        children = _spawn_children_of(proc.pid)
    assert children, "worker child never appeared"

    os.kill(proc.pid, sig)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(f"{sig.name}: parent did not exit within 20s")

    # Child must notice and exit on its own (watchdog polls every 1s).
    gone = False
    for _ in range(40):
        time.sleep(0.25)
        alive = []
        for c in children:
            try:
                os.kill(c, 0)
                alive.append(c)
            except ProcessLookupError:
                pass
        if not alive:
            gone = True
            break
    for c in children:  # never leave test garbage behind
        with contextlib.suppress(ProcessLookupError):
            os.kill(c, signal.SIGKILL)
    assert gone, f"{sig.name}: orphaned worker child survived the parent"

    leftovers = [p.name for p in dest_dir.iterdir()]
    assert not leftovers, f"{sig.name}: partial outputs at destination: {leftovers}"
