"""Interrupting the CLI mid-processing must leave no partial outputs at the
destination and no orphaned worker processes — for SIGINT, SIGTERM, and
even SIGKILL of the parent (the child notices its parent died).

FLAKE FIX #2 (2026-08-20). The previous version waited for a real ``enhance``
progress event and then signalled, on the reasoning that the run was by then
genuinely mid-processing. It is — for about three more seconds. On this
fixture the first ``enhance`` event lands at t=1.05 s and publication at
t=4.43 s, so everything between reading that line and the signal being acted
on (the ``pgrep``, the scheduler, machine load, the reader's own position in
the pipe) ate into a 3.4 s margin that nothing enforced. Lose it and the run
has *completed*: all three artefacts are at the destination, legitimately,
and the assertion reported them as "partial outputs" — a false accusation
against production code, which is what the audit reproduced, and what turned
``scripts/run_release_checks.sh`` and ``scripts/mutation_gate.py`` red.

The margin is not widened here; it is removed. The child is SIGSTOPped
before the signal is sent (``tests/chaos/procwatch.py`` documents the
protocol), so it cannot publish between the decision and the signal, and
whether it had already published is *checked* rather than assumed. A run
that publishes before the freeze is a lost race, not a defect: the attempt
is retried, and running out of attempts is a loud failure, never a silent
pass.

Why it matters beyond this file: a flaky test here is a broken gate. The
mutation gate refuses to score a red baseline, so this one test could stop
the whole gate from producing a number.
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

from tests.chaos.procwatch import (
    alive,
    children_of,
    contents,
    describe,
    freeze,
    kill_tree,
    thaw,
    wait_all_gone,
)

CLI = str(Path(sys.executable).with_name("hawavoclean"))

#: How long to wait for the run to reach unit enhancement. Generous on purpose:
#: it bounds a real decode + segment + model load, and is never the thing under
#: test — exceeding it means the pipeline is broken, not slow.
START_TIMEOUT_S = 180.0
#: How long an orphaned worker may take to notice its parent is gone. The
#: watchdog polls at 0.5 s, so anything over a second or two is already a bug;
#: 30 s only guarantees that machine load never decides the verdict.
ORPHAN_EXIT_TIMEOUT_S = 30.0
#: Attempts allowed to catch the run before it publishes. Two is already
#: paranoid: the freeze happens ~30 ms after the enhance event, with seconds
#: of run left. Exhausting them fails the test loudly.
MAX_ATTEMPTS = 3


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


def _start_run(long_input: Path, dest_dir: Path, work_dir: Path) -> "subprocess.Popen[str]":
    return subprocess.Popen(
        [
            CLI,
            "process",
            str(long_input),
            "-o",
            str(dest_dir / "o.wav"),
            "--overwrite",
            "--progress-json",
        ],
        env={**os.environ, "HAWAVOCLEAN_WORK_DIR": str(work_dir)},
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


@pytest.mark.chaos
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_interrupt_leaves_no_partials_and_no_orphans(
    sig: signal.Signals, long_input: Path, tmp_path: Path
) -> None:
    lost_races: list[list[str]] = []

    for attempt in range(MAX_ATTEMPTS):
        dest_dir = tmp_path / f"dest{attempt}"
        dest_dir.mkdir()
        proc = _start_run(long_input, dest_dir, tmp_path / f"work{attempt}")
        children: list[int] = []
        try:
            _wait_until_enhancing(proc)

            # From here the child executes nothing: it cannot publish between
            # our decision and our signal, and what it has already written is
            # a stable fact rather than a race.
            freeze(proc.pid)
            already_written = contents(dest_dir)
            if already_written:
                # It published before we could freeze it. Nothing is wrong
                # with the code; this attempt simply did not test anything.
                lost_races.append(already_written)
                continue

            children = children_of(proc.pid)
            assert children, "worker child missing while a unit is being enhanced"
            os.kill(proc.pid, sig)  # pending while stopped
            thaw(proc.pid)  # resumes straight into it

            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pytest.fail(f"{sig.name}: parent did not exit within 30s")

            # Children must notice and exit on their own (watchdog polls at 0.5 s).
            survivors, waited = wait_all_gone(children, ORPHAN_EXIT_TIMEOUT_S)
            # Name what survived: "two pids" is not enough to tell an orphaned
            # enhancement worker from a multiprocessing resource tracker, and
            # the two call for different fixes.
            described = [f"{pid} ({describe(pid)})" for pid in survivors]
            assert not survivors, (
                f"{sig.name}: orphaned worker child survived the parent by "
                f"{waited:.1f}s: {described}"
            )
            assert not contents(dest_dir), (
                f"{sig.name}: partial outputs at destination: {contents(dest_dir)} "
                f"(the run was frozen mid-enhance with an empty destination, so "
                f"nothing here can be a completed run)"
            )
            return
        finally:  # never leave test garbage behind, whichever assertion fired
            # Only signal what is provably still ours: once Popen has reaped
            # the child, its pid belongs to the operating system again and
            # SIGKILLing it by number would be aimed at a stranger.
            if proc.poll() is None:
                kill_tree(proc.pid)
            else:
                for c in children:
                    if alive(c):  # pragma: no cover - only after a failed assertion
                        kill_tree(c)
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
            if proc.stdout is not None:
                proc.stdout.close()

    pytest.fail(
        f"{sig.name}: the run published before it could be frozen in all "
        f"{MAX_ATTEMPTS} attempts ({lost_races}). The interruption was never "
        f"exercised — this is a broken test, not a passing one."
    )
