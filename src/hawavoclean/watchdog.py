"""Parent-death watchdog for processes this tool spawns.

A parent that is SIGKILLed runs no cleanup: no ``finally``, no atexit, no
``JobManager.shutdown()``. Anything it started keeps running, is reparented
to init, and finishes work nobody is waiting for any more — which is how a
killed engine could still publish a full master and report for a run its UI
had already reconciled to "failed, nothing was written".

So the *child* watches. A spawning process stamps its own pid into
:data:`PARENT_PID_ENV`; the child arms a daemon thread that polls that pid
and, the moment it is gone, raises a signal on itself so the ordinary
interrupt path runs — worker torn down, workspace removed, nothing
published — with a hard ``os._exit`` backstop if that unwind does not
finish. Liveness is checked two ways because either alone can lie:
``kill(pid, 0)`` (a pid can be recycled) and ``getppid()`` (which changes
to 1 on reparenting).

Two topology lessons are folded in:

* The self-interrupt is SIGINT only while SIGINT can do anything. A process
  spawned from a background job (``cmd &`` in a non-interactive shell,
  nohup pipelines, most supervisors) inherits ``SIGINT=SIG_IGN`` across
  exec, and CPython leaves an inherited ignore alone — so a self-SIGINT
  there is a silent no-op and a fast child can publish inside the backstop
  grace. In that state the watchdog escalates to SIGTERM, which
  ``cli.main()`` maps onto the same KeyboardInterrupt unwind before this
  watchdog is ever armed.

* :data:`PARENT_PID_ENV` is a private contract between a spawner and its
  DIRECT child. It is only honored when ``getppid()`` still names the
  declared pid at arm time (parent death is then detected by the ppid
  *changing*), or when the child is provably the orphan of a spawner that
  died before arming (already reparented to init and the declared pid gone).
  Anything else — a stale export, a leaked shell variable — is ignored with
  a warning instead of killing an unrelated invocation at startup.
"""

import logging
import os
import signal
import sys
import threading

logger = logging.getLogger(__name__)

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


def _pid_exists(pid: int) -> bool:
    """Whether ``pid`` names a live process (not necessarily one of ours)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, just not ours to signal
        return True
    except OSError as exc:
        # On Windows, os.kill(dead_pid, 0) raises OSError with WinError 87 (ERROR_INVALID_PARAMETER)
        if getattr(exc, "winerror", None) == 87:
            return False
        raise
    return True


def _windows_get_parent_pid(pid: int) -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        kernel32 = windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == wintypes.HANDLE(-1).value or snapshot == -1:
            return None
        try:
            entry = PROCESSENTRY32()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
            if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                while True:
                    if entry.th32ProcessID == pid:
                        return int(entry.th32ParentProcessID)
                    if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                        break
        finally:
            kernel32.CloseHandle(snapshot)
    except Exception:
        pass
    return None


def parent_is_alive(parent_pid: int) -> bool:
    """Whether the process that spawned us is still there."""
    if sys.platform != "win32" and os.getppid() != parent_pid:
        return False  # reparented: our parent is gone, whoever holds that pid now
    return _pid_exists(parent_pid)


def _self_interrupt_signal() -> signal.Signals:
    """The signal whose delivery will actually unwind this process.

    SIGINT while it is catchable. A process spawned from a background job
    runs with an inherited ``SIGINT=SIG_IGN`` (POSIX keeps terminal Ctrl-C
    away from background work, and the ignore persists across exec), so a
    self-SIGINT there would change nothing; escalate to SIGTERM, which
    ``cli._install_signal_handlers`` turns into the same KeyboardInterrupt
    unwind. Checked at fire time, from the watchdog thread —
    ``signal.getsignal`` is thread-safe where ``signal.signal`` is not.
    """
    if sys.platform == "win32" or signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        return signal.SIGTERM
    return signal.SIGINT


def install_parent_death_watchdog(
    *,
    poll_interval_s: float = POLL_INTERVAL_S,
    grace_s: float = UNWIND_GRACE_S,
) -> threading.Thread | None:
    """Arm the watchdog if this process was spawned with :data:`PARENT_PID_ENV`.

    Returns the watching thread, or ``None`` when the variable is absent or
    unusable — a hand-run CLI has no watched parent and must never exit
    because the shell that launched it went away. "Unusable" includes a
    declared pid that is not this process's parent: the variable is a
    private contract between the spawner and its direct child, and a stale
    or exported value must not kill an unrelated invocation at startup.

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

    ppid = os.getppid()
    is_valid_parent = (ppid == parent_pid) or (
        sys.platform == "win32" and _windows_get_parent_pid(ppid) == parent_pid
    )
    if not is_valid_parent:
        # The declarer is not our parent. One real exception: our spawner
        # declared itself and then died in the window before this ran — that
        # child is already hanging off init (or parent died on Windows) and the declared pid is gone, and
        # it must still tear itself down or the arming gap is a free pass to
        # finish and publish. Everything else is a stale or leaked variable.
        is_spawner_death = sys.platform != "win32" and ppid == 1 and not _pid_exists(parent_pid)
        if not is_spawner_death:
            logger.warning(
                "Ignoring %s=%d: not set by this process's parent (ppid %d). "
                "The variable is a private spawner contract; a stale or "
                "exported value must not end an unrelated invocation.",
                PARENT_PID_ENV,
                parent_pid,
                ppid,
            )
            return None
        logger.warning(
            "Declared parent %d died before the watchdog armed; tearing down.",
            parent_pid,
        )
        # Fall through: the poll loop below notices at once.

    def _watch() -> None:
        while parent_is_alive(parent_pid):
            threading.Event().wait(poll_interval_s)
        # Ask for the same unwind a Ctrl-C would produce: the process owns a
        # scratch workspace, a staging directory at the destination and a
        # worker subprocess, and all three have interrupt-safe cleanup.
        try:
            os.kill(os.getpid(), _self_interrupt_signal())
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
