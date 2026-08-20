"""Parent-death watchdog for processes this tool spawns.

A parent that is SIGKILLed runs no cleanup: no ``finally``, no atexit, no
``JobManager.shutdown()``. Anything it started keeps running, is reparented
to init, and finishes work nobody is waiting for any more — which is how a
killed engine could still publish a full master and report for a run its UI
had already reconciled to "failed, nothing was written".

So the *child* watches. A spawning process stamps its own pid into
:data:`PARENT_PID_ENV`; the child arms a daemon thread that polls that pid
and, the moment it is gone, raises SIGINT on itself so the ordinary
interrupt path runs — worker torn down, workspace removed, nothing
published — with a hard ``os._exit`` backstop if that unwind does not
finish. Liveness is checked two ways because either alone can lie:
``kill(pid, 0)`` (a pid can be recycled) and ``getppid()`` (which changes
to 1 on reparenting).
"""

import os
import signal
import threading

#: Environment variable naming the pid a child should watch.
PARENT_PID_ENV = "HAWAVOCLEAN_PARENT_PID"

#: Poll cadence. A dead parent is noticed within this, plus the time the
#: main thread needs to reach an interruptible point.
POLL_INTERVAL_S = 0.25

#: How long the interrupt path gets before the process is ended outright.
UNWIND_GRACE_S = 5.0

#: Exit status of a child that outlived its parent (128 + SIGINT).
ORPHAN_EXIT_CODE = 130


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """``base`` (default: this process's environment) plus our pid, for a child."""
    env = dict(base) if base is not None else dict(os.environ)
    env[PARENT_PID_ENV] = str(os.getpid())
    return env


def parent_is_alive(parent_pid: int) -> bool:
    """Whether the process that spawned us is still there."""
    if os.getppid() != parent_pid:
        return False  # reparented: our parent is gone, whoever holds that pid now
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, just not ours to signal
        return True
    return True


def install_parent_death_watchdog(
    *,
    poll_interval_s: float = POLL_INTERVAL_S,
    grace_s: float = UNWIND_GRACE_S,
) -> threading.Thread | None:
    """Arm the watchdog if this process was spawned with :data:`PARENT_PID_ENV`.

    Returns the watching thread, or ``None`` when the variable is absent or
    unusable — a hand-run CLI has no watched parent and must never exit
    because the shell that launched it went away.

    Call this as early as possible. The lesson from the enhancement worker,
    which armed its own watchdog only after model warmup, is that every
    instruction before the arming is an unwatched window.
    """
    raw = os.environ.get(PARENT_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        parent_pid = int(raw)
    except ValueError:
        return None
    if parent_pid <= 1 or parent_pid == os.getpid():
        return None

    def _watch() -> None:
        while parent_is_alive(parent_pid):
            threading.Event().wait(poll_interval_s)
        # Ask for the same unwind a Ctrl-C would produce: the process owns a
        # scratch workspace, a staging directory at the destination and a
        # worker subprocess, and all three have interrupt-safe cleanup.
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except OSError:  # pragma: no cover - only if the pid table is broken
            os._exit(ORPHAN_EXIT_CODE)
        # Backstop: a main thread stuck in a long C call (a decode, a model
        # forward) cannot raise KeyboardInterrupt. Bound the wait, then end
        # the process outright rather than leave an orphan running.
        threading.Event().wait(grace_s)
        os._exit(ORPHAN_EXIT_CODE)

    thread = threading.Thread(target=_watch, name="hawavoclean-parent-watchdog", daemon=True)
    thread.start()
    return thread
