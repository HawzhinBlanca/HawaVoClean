"""Interrupting the CLI mid-processing must leave no partial outputs at the
destination and no orphaned worker processes — for SIGINT, SIGTERM, and
even SIGKILL of the parent (the child notices its parent died).

FLAKE FIX (mutation-gate integrity, 2026-08-20). This test used to signal the
parent the instant ``pgrep -P`` first reported a child, and then allow the
child 10 s to disappear. Both halves were timing bets on a real subprocess:

* The worker arms its parent-death watchdog only *after*
  ``enhancer.warmup()`` (``src/hawavoclean/enhancement/worker.py``). A child
  that has merely *appeared* is still in import + model load, with no watchdog
  running, so how long it outlives a SIGKILLed parent is set by how long
  warmup happens to take on the machine that day. The old test opened its 10 s
  window right inside that unarmed gap.
* "worker child never appeared in 20 s" was a second load bet.

Both are now removed by waiting for a real ``--progress-json`` ``enhance``
event before signalling: at that point the run is genuinely *mid-processing*
(which is what the docstring always claimed), the worker exists, and its
watchdog is armed. The assertions are unchanged; only the moment they are made
is now deterministic, and the disappearance window is generous enough that
machine load can never decide the verdict.

Why it mattered beyond this file: a flaky test here short-circuits ``pytest -x``
before any mutated code runs, which is how ``scripts/mutation_gate.py`` used to
credit mutations it had never actually caught. The gate no longer credits any
test outside a mutation's declared owners, and this file is on its quarantine
list as well — but a flaky test in a repo is a broken gate somewhere, so the
race is fixed rather than merely routed around.
"""

import contextlib
import json
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

#: How long to wait for the run to reach unit enhancement. Generous on purpose:
#: it bounds a real decode + segment + model load, and is never the thing under
#: test — exceeding it means the pipeline is broken, not slow.
START_TIMEOUT_S = 180.0
#: How long an orphaned worker may take to notice its parent is gone. The
#: watchdog polls at 0.5 s, so anything over a second or two is already a bug;
#: 30 s only guarantees that machine load never decides the verdict.
ORPHAN_EXIT_TIMEOUT_S = 30.0


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


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until_enhancing(proc: "subprocess.Popen[str]") -> None:
    """Block until the run reports it is enhancing a unit.

    Reads the ``--progress-json`` stream, which is one JSON object per line on
    the original stdout. Returning means the worker subprocess exists *and* has
    finished warmup, so its parent-death watchdog is running.
    """
    assert proc.stdout is not None
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:  # the CLI exited before it ever got to a unit
            raise AssertionError(f"process exited before enhancing (rc={proc.poll()})")
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "error":
            raise AssertionError(f"pipeline failed before enhancing: {ev}")
        if ev.get("event") == "progress" and ev.get("stage") in ("enhance", "guard"):
            return
    raise AssertionError(f"pipeline never reached a unit within {START_TIMEOUT_S:.0f}s")


@pytest.mark.chaos
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_interrupt_leaves_no_partials_and_no_orphans(
    sig: signal.Signals, long_input: Path, tmp_path: Path
) -> None:
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    env = {**os.environ, "HAWAVOCLEAN_WORK_DIR": str(tmp_path / "work")}
    proc = subprocess.Popen(
        [
            CLI,
            "process",
            str(long_input),
            "-o",
            str(dest_dir / "o.wav"),
            "--overwrite",
            "--progress-json",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    children: list[int] = []
    survivors: list[int] = []
    try:
        _wait_until_enhancing(proc)
        children = _spawn_children_of(proc.pid)
        assert children, "worker child missing while a unit is being enhanced"

        os.kill(proc.pid, sig)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail(f"{sig.name}: parent did not exit within 30s")

        # Child must notice and exit on its own (watchdog polls every 0.5 s).
        deadline = time.time() + ORPHAN_EXIT_TIMEOUT_S
        survivors = [c for c in children if _alive(c)]
        while survivors and time.time() < deadline:
            time.sleep(0.1)
            survivors = [c for c in children if _alive(c)]
    finally:  # never leave test garbage behind, whichever assertion fired
        for c in {*children, *_spawn_children_of(proc.pid)}:
            with contextlib.suppress(ProcessLookupError):
                os.kill(c, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()

    assert not survivors, f"{sig.name}: orphaned worker child survived the parent: {survivors}"

    leftovers = [p.name for p in dest_dir.iterdir()]
    assert not leftovers, f"{sig.name}: partial outputs at destination: {leftovers}"
