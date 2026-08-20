"""The parent-death watchdog: who it watches, when it refuses to arm, and
what it does when the parent is gone.

The end-to-end proof (SIGKILL a real engine, watch the job child die and
write nothing) is ``tests/chaos/test_orphan_watchdogs.py``. This file pins
the decisions that surround it, in-process.
"""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hawavoclean.watchdog import (
    PARENT_PID_ENV,
    child_env,
    install_parent_death_watchdog,
    parent_is_alive,
)

pytestmark = pytest.mark.unit


def test_child_env_names_this_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_OTHER_VAR", "kept")
    env = child_env()
    assert env[PARENT_PID_ENV] == str(os.getpid())
    assert env["SOME_OTHER_VAR"] == "kept"  # inherits the environment
    # An explicit base replaces the environment but still carries our pid.
    explicit = child_env({"ONLY": "this"})
    assert explicit == {"ONLY": "this", PARENT_PID_ENV: str(os.getpid())}


def test_parent_is_alive_answers_both_ways(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parent_is_alive(os.getppid()) is True
    # Reparented: getppid() no longer matches the pid we were told to watch.
    assert parent_is_alive(os.getppid() + 999_999) is False
    # Still our parent by getppid(), but the pid is gone (the process died and
    # the number was not reused): the kill(pid, 0) probe is what catches it.
    dead = _a_pid_that_has_exited()
    monkeypatch.setattr(os, "getppid", lambda: dead)
    assert parent_is_alive(dead) is False


def _a_pid_that_has_exited() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    return proc.pid


@pytest.mark.parametrize("value", ["", "   ", "not-a-number", "0", "1", "-4"])
def test_watchdog_refuses_to_arm_on_a_useless_pid(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PARENT_PID_ENV, value)
    assert install_parent_death_watchdog() is None


def test_watchdog_does_not_arm_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hand-run CLI must not exit because the shell that started it went away."""
    monkeypatch.delenv(PARENT_PID_ENV, raising=False)
    assert install_parent_death_watchdog() is None


def test_watchdog_does_not_arm_on_our_own_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PARENT_PID_ENV, str(os.getpid()))
    assert install_parent_death_watchdog() is None


def test_watchdog_interrupts_then_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The armed thread asks for a clean unwind first, then ends the process.

    ``os.kill`` and ``os._exit`` are recorded rather than performed — this is
    the test runner's own process — so both rungs are observable.
    """
    fired = threading.Event()
    signals: list[tuple[int, int]] = []
    exits: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        if sig == 0:
            raise ProcessLookupError  # the parent is gone
        signals.append((pid, sig))

    def fake_exit(code: int) -> None:
        # Recording and returning is enough: os._exit is the last statement
        # in the watchdog, so the thread ends on its own from here.
        exits.append(code)
        fired.set()

    monkeypatch.setattr(os, "getppid", lambda: 424242)
    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setenv(PARENT_PID_ENV, "424242")

    thread = install_parent_death_watchdog(poll_interval_s=0.01, grace_s=0.05)
    assert thread is not None
    assert fired.wait(10.0), "watchdog never acted on a dead parent"
    thread.join(timeout=5.0)

    assert signals == [(os.getpid(), signal.SIGINT)], signals
    assert exits == [130]


def test_a_child_told_to_watch_a_dead_pid_exits_at_once(tmp_path: Path) -> None:
    """End of the contract, in a real process: armed, parent gone, no work done."""
    marker = tmp_path / "still-running"
    script = (
        "import sys, time\n"
        "from hawavoclean.watchdog import install_parent_death_watchdog\n"
        "assert install_parent_death_watchdog(poll_interval_s=0.05) is not None\n"
        "time.sleep(20)\n"
        f"open({str(marker)!r}, 'w').close()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, PARENT_PID_ENV: str(_a_pid_that_has_exited())},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        rc = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - only on a broken watchdog
        proc.kill()
        pytest.fail("the watchdog never ended a process whose parent was gone")
    assert rc != 0, "an orphan must not report success"
    assert not marker.exists(), "the orphan ran on to finish its work"
