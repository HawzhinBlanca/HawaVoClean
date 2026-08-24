"""Process helpers for the chaos tests, and the freeze protocol they share.

A test that signals a *running* pipeline has to answer a question first: was
the run still in flight when the signal arrived? Reading a progress event
and then signalling does not answer it — between those two statements the
child keeps running, and this pipeline publishes about three seconds after
it starts its first unit. Lose that race and the destination legitimately
holds a complete master, which the old assertion reported as "partial
outputs" — a false accusation against production code, and the flake that
kept the release gate and the mutation gate red.

So the signal is delivered under a freeze:

1. ``SIGSTOP`` the child. It cannot be caught or ignored, so from here the
   child executes nothing and its state is stable.
2. Look at the destination. Empty means publication has not happened and
   now cannot; anything there means this attempt lost the race before the
   freeze, which the caller retries instead of asserting on.
3. Send the signal under test. It stays pending.
4. ``SIGCONT``. The child resumes straight into the pending signal.

The window is now bounded by the kernel rather than by machine load.
"""

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path


def children_of(pid: int) -> list[int]:
    """Direct children of ``pid`` (empty if none, or if the pid is gone)."""
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def alive(pid: int) -> bool:
    """Whether ``pid`` still exists.

    ``kill(pid, 0)`` also succeeds for a zombie, so ask this only about
    processes nobody in this test session has to reap: the orphans here are
    reparented to init, which reaps them at once. For a direct child of the
    test process, ``Popen.wait()`` first.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not ours, but it exists
        return True
    return True


def wait_all_gone(pids: list[int], timeout_s: float) -> tuple[list[int], float]:
    """Wait for every pid to disappear. Returns (survivors, seconds waited)."""
    t0 = time.monotonic()
    survivors = [p for p in pids if alive(p)]
    while survivors and time.monotonic() - t0 < timeout_s:
        time.sleep(0.05)
        survivors = [p for p in pids if alive(p)]
    return survivors, time.monotonic() - t0


def freeze(pid: int) -> None:
    """SIGSTOP: the process runs nothing until :func:`thaw`."""
    os.kill(pid, signal.SIGSTOP)


def thaw(pid: int) -> None:
    """SIGCONT, tolerating a process that has already died (e.g. SIGKILLed)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGCONT)


def descendants(pid: int) -> list[int]:
    """``pid``'s children and their children (one extra level is all we spawn)."""
    out: list[int] = []
    for child in children_of(pid):
        out.append(child)
        out.extend(children_of(child))
    return out


def kill_tree(pid: int) -> None:
    """Best-effort SIGKILL of a pid and everything under it. Never raises."""
    for victim in [*descendants(pid), pid]:
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(victim, signal.SIGCONT)  # a stopped process cannot die of SIGTERM
            os.kill(victim, signal.SIGKILL)


def contents(directory: Path) -> list[str]:
    """Sorted names in ``directory`` — every entry, dotfiles included: a
    publish staging directory left behind is litter at the destination too."""
    return sorted(p.name for p in directory.iterdir())


def describe(pid: int) -> str:
    """Best-effort command line for ``pid``, for diagnostics in assertions.

    A bare pid says a process survived but not *what* did, and an orphaned
    enhancement worker and a lingering multiprocessing resource tracker are
    different bugs with different fixes.
    """
    out = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return out[:120] if out else "gone"
