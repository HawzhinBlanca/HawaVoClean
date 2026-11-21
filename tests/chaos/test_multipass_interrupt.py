"""Interrupting a multi-pass run mid-pass must leave no partial outputs at
the destination, no orphaned workers, and no ``multipass-*`` temp litter
under the work root.

Same freeze protocol as ``test_interrupt_cleanup.py`` (see
``tests/chaos/procwatch.py``): SIGSTOP the child before deciding anything,
check the destination is still empty, deliver the signal while frozen, thaw
into it. No timing bets.

Scope note: the per-pass pipeline workspace directories are ALLOWED to
survive an interrupt — that is the pipeline's documented crash-forensics
behaviour, unchanged by multipass. What multipass adds — the ``multipass-*``
directory holding intermediate pass outputs and their reports — must be gone
on every SIGINT/SIGTERM exit path. (SIGKILL runs no cleanup by definition;
the existing single-pass chaos test already covers orphan behaviour there.)
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
    freeze,
    kill_tree,
    thaw,
    wait_all_gone,
)

CLI = str(Path(sys.executable).with_name("hawavoclean"))
START_TIMEOUT_S = 180.0
ORPHAN_EXIT_TIMEOUT_S = 30.0
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
    assert proc.stdout is not None
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
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


def _multipass_litter(work_dir: Path) -> list[str]:
    if not work_dir.exists():
        return []
    return sorted(p.name for p in work_dir.glob("multipass-*"))


pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX process hierarchy tests (Windows covered by test_process_supervisor.py)",
    ),
]


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_multipass_interrupt_leaves_no_partials_and_no_litter(
    sig: signal.Signals, long_input: Path, tmp_path: Path
) -> None:
    lost_races: list[list[str]] = []

    for attempt in range(MAX_ATTEMPTS):
        dest_dir = tmp_path / f"dest{attempt}"
        dest_dir.mkdir()
        work_dir = tmp_path / f"work{attempt}"
        proc = subprocess.Popen(
            [
                CLI,
                "process",
                str(long_input),
                "-o",
                str(dest_dir / "o.wav"),
                "--overwrite",
                "--passes",
                "2",
                "--progress-json",
            ],
            env={**os.environ, "HAWAVOCLEAN_WORK_DIR": str(work_dir)},
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        children: list[int] = []
        try:
            _wait_until_enhancing(proc)

            freeze(proc.pid)
            already_written = contents(dest_dir)
            if already_written:  # pragma: no cover - lost race, retried
                lost_races.append(already_written)
                continue
            assert _multipass_litter(work_dir), (
                "a frozen mid-enhance multipass run must have its multipass-* "
                "temp dir on disk — without it this test would be asserting "
                "the absence of something that never existed"
            )

            children = children_of(proc.pid)
            assert children, "worker child missing while a unit is being enhanced"
            os.kill(proc.pid, sig)  # pending while stopped
            thaw(proc.pid)  # resumes straight into it

            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pytest.fail(f"{sig.name}: parent did not exit within 30s")

            survivors, waited = wait_all_gone(children, ORPHAN_EXIT_TIMEOUT_S)
            assert not survivors, (
                f"{sig.name}: orphaned worker child survived the parent by "
                f"{waited:.1f}s: {survivors}"
            )
            assert not contents(dest_dir), (
                f"{sig.name}: partial outputs at destination: {contents(dest_dir)}"
            )
            assert not _multipass_litter(work_dir), (
                f"{sig.name}: multipass temp litter under the work root: "
                f"{_multipass_litter(work_dir)}"
            )
            return
        finally:
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
