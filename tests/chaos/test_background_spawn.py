"""A background-spawned batch must still die with its parent.

The orphan watchdog's self-interrupt is SIGINT — but a batch started with
``&`` from a non-interactive shell (also nohup pipelines and most
supervisors) runs with ``SIGINT=SIG_IGN``, POSIX's way of keeping terminal
Ctrl-C away from background work. The ignore persists across exec into the
per-file child, where CPython leaves it alone — so before the escalation the
watchdog's ``os.kill(self, SIGINT)`` changed nothing and only the 5 s hard
backstop remained. Measured then: a batch SIGKILLed at +8 s of a 13 s file
left a child that ran to completion and PUBLISHED a full master 4.5 s after
its parent died. Foreground worked (0.354 s); the topology was the bug.

So this file pins the topology itself, twice: the audit's shell ``&`` repro
and a plain ``Popen`` with SIGINT ignored (the supervisor shape). In both,
the batch parent is SIGKILLed mid-file and the per-file child must be gone
fast — worker down, destination empty — through the same freeze protocol as
the engine test (see ``tests/chaos/procwatch``): the child is SIGSTOPped
before the kill so "the destination is empty" is a fact about the rest of
the test, not a stale snapshot.
"""

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.watchdog import UNWIND_GRACE_S
from tests.chaos.procwatch import (
    alive,
    children_of,
    contents,
    freeze,
    kill_tree,
    thaw,
    wait_all_gone,
)

SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX process hierarchy tests (Windows covered by test_process_supervisor.py)",
    ),
]

REPO = Path(__file__).resolve().parents[2]

#: A dead parent must be noticed and acted on well inside this. The job
#: child's watchdog polls at 0.25 s; the measured teardown in this topology
#: is ~0.1-0.2 s. The bound only keeps machine load from deciding.
ORPHAN_EXIT_TIMEOUT_S = 30.0
#: The clean-unwind proof. A dead child alone does not pin this defect: the
#: hard backstop also ends the process, at ``UNWIND_GRACE_S`` (5 s) after the
#: no-op SIGINT — which is exactly the broken state, and on a kill early in a
#: long file it too leaves the destination empty. The interrupt path is only
#: proven by a teardown that lands well before the backstop could: watchdog
#: poll 0.25 s + measured unwind ~0.2 s against a 3 s allowance.
INTERRUPT_PATH_DEADLINE_S = UNWIND_GRACE_S - 2.0
CHILD_START_TIMEOUT_S = 180.0
MAX_ATTEMPTS = 3


@pytest.fixture(scope="module")
def long_input(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("bg-spawn") / "long.wav"
    sr = 48000
    t = np.arange(sr * 90) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    sf.write(str(p), x, sr)
    return p


def test_a_background_job_really_inherits_sig_ign() -> None:
    """The premise, pinned on this machine: ``sh -c 'cmd &'`` execs the
    command with SIGINT ignored, and CPython leaves the ignore in place."""
    probe = "import signal; print(int(signal.getsignal(signal.SIGINT) is signal.SIG_IGN))"
    out = subprocess.run(
        ["/bin/sh", "-c", f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)} & wait"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    assert out == "1", f"background spawn did not inherit SIG_IGN (probe said {out!r})"


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _spawn_batch(
    topology: str, long_input: Path, dest_dir: Path
) -> tuple["subprocess.Popen[str]", int]:
    """Start ``hawavoclean batch`` in the given topology.

    Returns the direct child handle (the shell, or the batch itself) and the
    batch's pid. In both topologies the batch — and therefore the per-file
    child it spawns — runs with ``SIGINT=SIG_IGN``.
    """
    argv = [
        sys.executable,
        "-m",
        "hawavoclean.cli",
        "batch",
        str(long_input),
        "-o",
        str(dest_dir),
        "--overwrite",
    ]
    if topology == "shell_background":
        line = " ".join(shlex.quote(a) for a in argv)
        proc = subprocess.Popen(
            ["/bin/sh", "-c", f"{line} >/dev/null 2>&1 & echo $!; wait"],
            cwd=REPO,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        raw = proc.stdout.readline().strip()
        assert raw.isdigit(), f"the shell never reported the batch pid: {raw!r}"
        return proc, int(raw)
    proc = subprocess.Popen(
        argv,
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=_ignore_sigint,  # the supervisor shape: SIG_IGN across exec
    )
    return proc, proc.pid


def _wait_for_children(of_pid: int, deadline: float) -> list[int]:
    """Poll ``children_of(of_pid)`` until non-empty or the deadline."""
    while time.time() < deadline:
        kids = children_of(of_pid)
        if kids:
            return kids
        time.sleep(0.05)
    return []


@pytest.mark.chaos
@pytest.mark.parametrize("topology", ["shell_background", "popen_sigint_ignored"])
def test_sigkilled_background_batch_leaves_no_child_writing_files(
    topology: str, long_input: Path, tmp_path: Path
) -> None:
    """SIGKILL a background-spawned batch mid-file: the per-file child must
    tear down its worker and leave the destination empty, exactly as the
    foreground path does — not run to completion on the backstop's grace."""
    lost_races: list[list[str]] = []
    slow_unwinds: list[float] = []
    for attempt in range(MAX_ATTEMPTS):
        dest_dir = tmp_path / f"dest-{topology}-{attempt}"
        dest_dir.mkdir()
        handle, batch_pid = _spawn_batch(topology, long_input, dest_dir)
        job_child: int | None = None
        try:
            deadline = time.time() + CHILD_START_TIMEOUT_S
            kids = _wait_for_children(batch_pid, deadline)
            assert kids, f"the batch never spawned a per-file child (rc={handle.poll()})"
            job_child = kids[0]
            # Mid-file, provably: the per-file child has its own child (the
            # decoder or the enhancement worker). Any grandchild also means
            # the watchdog armed long ago — it arms before argument parsing.
            grandkids = _wait_for_children(job_child, deadline)
            assert grandkids, "the per-file child never started real work"

            # Freeze the job child first: from here it cannot publish, so
            # "the destination is empty" holds for the rest of the test.
            freeze(job_child)
            if contents(dest_dir):
                lost_races.append(contents(dest_dir))
                continue

            os.kill(batch_pid, SIGKILL)  # no cleanup anywhere in the batch
            if handle.pid == batch_pid:
                handle.wait(timeout=30)  # reap it: a zombie still answers kill(pid, 0)
            else:
                # The shell's `wait` reaps the batch for us; watch it vanish.
                reap_deadline = time.time() + 30.0
                while alive(batch_pid) and time.time() < reap_deadline:
                    time.sleep(0.02)
                assert not alive(batch_pid), "the batch survived SIGKILL"
            thaw(job_child)

            survivors, waited = wait_all_gone([job_child, *grandkids], ORPHAN_EXIT_TIMEOUT_S)
            assert not survivors, (
                f"[{topology}] the SIGKILLed batch left {survivors} running after "
                f"{waited:.1f}s; with SIGINT inherited ignored, the per-file child "
                f"used to outrun the backstop and publish"
            )
            # This one is a stopwatch, so one sample is not a verdict. The
            # deadline separates "the interrupt path ran" from "the backstop
            # expired", and under a loaded machine -- the full suite, under
            # coverage -- the interrupt path itself can drift past it without
            # anything being wrong. Treat a slow sample the way this test
            # already treats a lost race: retry. A genuinely broken unwind is
            # slow every time, so all MAX_ATTEMPTS samples exceed and the test
            # still fails, now with every measurement in the message.
            slow_unwinds.append(waited)
            if waited >= INTERRUPT_PATH_DEADLINE_S:
                continue
            assert not contents(dest_dir), (
                f"[{topology}] an orphaned per-file child wrote to the destination "
                f"after its batch died: {contents(dest_dir)}"
            )
            return
        finally:
            if job_child is not None and alive(job_child):
                kill_tree(job_child)
            if handle.poll() is None:
                kill_tree(handle.pid)
            with contextlib.suppress(Exception):
                handle.wait(timeout=10)
            if handle.stdout is not None:
                handle.stdout.close()

    if slow_unwinds:
        # Reaching here means no attempt ever unwound inside the deadline:
        # every sample this run produced is in the list.
        pytest.fail(
            f"[{topology}] the per-file child took "
            f"{', '.join(f'{w:.2f}s' for w in slow_unwinds)} to die, every "
            f"measurement at or past the {INTERRUPT_PATH_DEADLINE_S:.0f}s "
            f"deadline — that is the {UNWIND_GRACE_S:.0f}s hard backstop "
            f"acting, not the interrupt path; with SIGINT inherited ignored "
            f"the self-interrupt must escalate to SIGTERM and unwind at once "
            f"({len(lost_races)} further attempt(s) lost the freeze race)"
        )
    pytest.fail(
        f"[{topology}] the job published before it could be frozen in all "
        f"{MAX_ATTEMPTS} attempts ({lost_races}); the orphan case was never exercised"
    )
