"""The parent-death watchdog: who it watches, when it refuses to arm, and
what it does when the parent is gone.

The end-to-end proof (SIGKILL a real engine, watch the job child die and
write nothing) is ``tests/chaos/test_orphan_watchdogs.py``; the background
spawn topology (``cmd &`` in a non-interactive shell, where SIGINT is
inherited ignored) is ``tests/chaos/test_background_spawn.py``. This file
pins the decisions that surround them, in-process:

* :data:`PARENT_PID_ENV` is honored only when it was set by this process's
  direct parent. A stale or exported value used to end every invocation —
  even ``--version`` — with a raw KeyboardInterrupt at startup.
* The self-interrupt escalates SIGINT → SIGTERM when SIGINT is inherited
  ignored, because in that state a self-SIGINT is a silent no-op and a fast
  child can publish inside the backstop grace.
"""

import contextlib
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
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


def test_a_declared_pid_that_is_not_our_parent_is_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A live pid that is not our parent is somebody else's contract.

    A stale exported ``HAWAVOCLEAN_PARENT_PID`` used to end every invocation
    at startup: the watchdog armed against a stranger, found it "not our
    parent", and self-interrupted even ``--version``.
    """
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setenv(PARENT_PID_ENV, str(stranger.pid))
        with caplog.at_level(logging.WARNING, logger="hawavoclean.watchdog"):
            assert install_parent_death_watchdog() is None
    finally:
        stranger.kill()
        stranger.wait(timeout=30)
    assert "not set by this process's parent" in caplog.text


def test_a_stale_dead_pid_outside_reparenting_is_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Dead declared pid, but we still hang off a real parent: stale, ignore.

    The one case a dead declared pid must act (spawner died before we armed)
    is recognizable by the reparenting to init; hanging off an ordinary
    process — a shell with a leaked export — is not that case.
    """
    dead = _a_pid_that_has_exited()
    monkeypatch.setattr(os, "getppid", lambda: 55_555)  # an interactive shell, not init
    monkeypatch.setenv(PARENT_PID_ENV, str(dead))
    with caplog.at_level(logging.WARNING, logger="hawavoclean.watchdog"):
        assert install_parent_death_watchdog() is None
    assert "not set by this process's parent" in caplog.text


class _RecordedActions:
    """os.kill / os._exit recorded rather than performed — this is the test
    runner's own process — so every rung of the watchdog is observable."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fired = threading.Event()
        self.signals: list[tuple[int, int]] = []
        self.exits: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            if sig == 0:
                raise ProcessLookupError  # every probed pid reads as gone
            self.signals.append((pid, sig))

        def fake_exit(code: int) -> None:
            # Recording and returning is enough: os._exit is the last
            # statement in the watchdog, so the thread ends on its own.
            self.exits.append(code)
            self.fired.set()

        monkeypatch.setattr(os, "kill", fake_kill)
        monkeypatch.setattr(os, "_exit", fake_exit)


def test_watchdog_interrupts_then_hard_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The armed thread asks for a clean unwind first, then ends the process."""
    actions = _RecordedActions(monkeypatch)
    monkeypatch.setattr(os, "getppid", lambda: 424242)
    monkeypatch.setenv(PARENT_PID_ENV, "424242")

    thread = install_parent_death_watchdog(poll_interval_s=0.01, grace_s=0.05)
    assert thread is not None
    assert actions.fired.wait(10.0), "watchdog never acted on a dead parent"
    thread.join(timeout=5.0)

    expected_sig = signal.SIGTERM if sys.platform == "win32" else signal.SIGINT
    assert actions.signals == [(os.getpid(), expected_sig)], actions.signals
    assert actions.exits == [130]


def test_watchdog_escalates_to_sigterm_when_sigint_is_inherited_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT=SIG_IGN (a background spawn) makes a self-SIGINT a silent no-op.

    Before the escalation, the only teardown left in that topology was the
    5 s hard backstop — which a per-file child outran, publishing a full
    master 4.5 s after its batch was SIGKILLed. The watchdog must notice the
    ignored disposition and deliver SIGTERM instead, which the CLI maps onto
    the same KeyboardInterrupt unwind.
    """
    actions = _RecordedActions(monkeypatch)
    monkeypatch.setattr(os, "getppid", lambda: 424242)
    monkeypatch.setenv(PARENT_PID_ENV, "424242")

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        thread = install_parent_death_watchdog(poll_interval_s=0.01, grace_s=0.05)
        assert thread is not None
        assert actions.fired.wait(10.0), "watchdog never acted on a dead parent"
        thread.join(timeout=5.0)
    finally:
        signal.signal(signal.SIGINT, previous)

    assert actions.signals == [(os.getpid(), signal.SIGTERM)], actions.signals
    assert actions.exits == [130]


def test_a_child_reparented_before_arming_still_tears_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawner declared itself, then died before we armed: not a stale var.

    That child is already hanging off init and the declared pid is gone; if
    the not-our-parent check ignored this too, the window between exec and
    arming would be a free pass to finish and publish.
    """
    if sys.platform == "win32":
        pytest.skip("Windows processes are never reparented to init (PID 1)")
    dead = _a_pid_that_has_exited()
    actions = _RecordedActions(monkeypatch)  # probes report the pid as gone
    monkeypatch.setattr(os, "getppid", lambda: 1)
    monkeypatch.setenv(PARENT_PID_ENV, str(dead))

    thread = install_parent_death_watchdog(poll_interval_s=0.01, grace_s=0.05)
    assert thread is not None, "a pre-arm orphan was misread as a stale variable"
    assert actions.fired.wait(10.0), "watchdog never acted on the dead spawner"
    thread.join(timeout=5.0)

    assert actions.signals == [(os.getpid(), signal.SIGINT)], actions.signals
    assert actions.exits == [130]


def test_a_process_with_a_stale_declared_pid_finishes_its_work(tmp_path: Path) -> None:
    """End of the stale-var contract, in a real process: warn, work, exit 0."""
    marker = tmp_path / "finished"
    script = (
        "from hawavoclean.watchdog import install_parent_death_watchdog\n"
        "assert install_parent_death_watchdog(poll_interval_s=0.05) is None\n"
        f"open({str(marker)!r}, 'w').close()\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, PARENT_PID_ENV: str(_a_pid_that_has_exited())},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert marker.exists(), "the stale variable stopped an unrelated invocation"
    # logging's last-resort handler puts the warning on stderr unconfigured.
    assert "not set by this process's parent" in proc.stderr


def _wait_gone(pid: int, timeout_s: float) -> float:
    """Seconds until ``pid`` disappears (the pid is init's to reap, not ours)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            return time.monotonic() - t0
        time.sleep(0.02)
    return time.monotonic() - t0


def test_a_child_whose_real_spawner_dies_is_gone_without_publishing(tmp_path: Path) -> None:
    """The honored contract, in real processes: spawner declares itself and
    dies; the child must go down with it, whether the death lands before or
    after the child finished arming."""
    marker = tmp_path / "still-running"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        textwrap.dedent(
            f"""
            import time
            from hawavoclean.watchdog import install_parent_death_watchdog
            install_parent_death_watchdog(poll_interval_s=0.05)
            time.sleep(20)
            open({str(marker)!r}, "w").close()
            """
        ),
        encoding="utf-8",
    )
    intermediate = tmp_path / "intermediate.py"
    intermediate.write_text(
        textwrap.dedent(
            f"""
            import os, subprocess, sys
            from hawavoclean.watchdog import child_env
            child = subprocess.Popen(
                [sys.executable, {str(grandchild)!r}],
                env=child_env(),
                stderr=subprocess.DEVNULL,
            )
            sys.stdout.write(str(child.pid) + "\\n")
            sys.stdout.flush()
            os._exit(0)  # no cleanup, no reaping: the child is on its own
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(intermediate)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        child_pid = int(proc.stdout.readline().strip())
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a wedged spawner
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()

    waited = _wait_gone(child_pid, timeout_s=30.0)
    try:
        os.kill(child_pid, 0)
    except (ProcessLookupError, OSError):
        pass
    else:  # pragma: no cover - only on a broken watchdog
        with contextlib.suppress(OSError):
            os.kill(child_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        pytest.fail(f"the orphan was still running {waited:.1f}s after its spawner died")
    assert not marker.exists(), "the orphan ran on to finish its work"


def test_the_interrupt_path_survives_an_inherited_sig_ign(tmp_path: Path) -> None:
    """SIG_IGN topology, real processes: the unwind still runs its cleanup.

    The grandchild is in the state a background-spawned per-file child is in:
    SIGINT ignored (inherited across exec), SIGTERM mapped to KeyboardInterrupt
    exactly as ``cli._install_signal_handlers`` does, partial output on disk.
    When its spawner dies, the escalated self-SIGTERM must unwind it — partial
    removed, nothing published — rather than leaving only the hard backstop.
    """
    partial = tmp_path / "partial-output"
    published = tmp_path / "published"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        textwrap.dedent(
            f"""
            import os, signal, sys, time
            from hawavoclean.watchdog import install_parent_death_watchdog

            signal.signal(signal.SIGINT, signal.SIG_IGN)  # the inherited state

            def _raise(_signum, _frame):
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, _raise)  # what cli.main() installs
            assert install_parent_death_watchdog(poll_interval_s=0.05) is not None
            open({str(partial)!r}, "w").close()
            try:
                time.sleep(20)
                open({str(published)!r}, "w").close()
            except KeyboardInterrupt:
                os.unlink({str(partial)!r})  # the interrupt-safe cleanup
                sys.exit(130)
            """
        ),
        encoding="utf-8",
    )
    intermediate = tmp_path / "intermediate.py"
    intermediate.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            from hawavoclean.watchdog import child_env
            child = subprocess.Popen(
                [sys.executable, {str(grandchild)!r}],
                env=child_env(),
                stderr=subprocess.DEVNULL,
            )
            sys.stdout.write(str(child.pid) + "\\n")
            sys.stdout.flush()
            time.sleep(3600)
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(intermediate)], stdout=subprocess.PIPE, text=True)
    child_pid: int | None = None
    try:
        assert proc.stdout is not None
        child_pid = int(proc.stdout.readline().strip())
        deadline = time.monotonic() + 30.0
        while not partial.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert partial.exists(), "the grandchild never armed and started work"
        proc.kill()  # no cleanup anywhere in the spawner
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a wedged spawner
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()

    assert child_pid is not None
    waited = _wait_gone(child_pid, timeout_s=30.0)
    try:
        os.kill(child_pid, 0)
    except (ProcessLookupError, OSError):
        pass
    else:  # pragma: no cover - only on a broken watchdog
        with contextlib.suppress(OSError):
            os.kill(child_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        pytest.fail(
            f"with SIGINT ignored, the orphan was still running {waited:.1f}s "
            f"after its spawner died"
        )
    assert not partial.exists(), "the unwind never ran its cleanup"
    assert not published.exists(), "the orphan published despite the watchdog"


def test_pid_exists_windows_winerror_87(monkeypatch: pytest.MonkeyPatch) -> None:
    from hawavoclean.watchdog import _pid_exists

    err = OSError("The parameter is incorrect")
    err.winerror = 87  # type: ignore[attr-defined]

    def _raise_winerror_87(_pid: int, _sig: int) -> None:
        raise err

    monkeypatch.setattr(os, "kill", _raise_winerror_87)
    assert _pid_exists(12345) is False

    other_err = OSError("Other error")
    other_err.winerror = 5  # type: ignore[attr-defined]

    def _raise_winerror_5(_pid: int, _sig: int) -> None:
        raise other_err

    monkeypatch.setattr(os, "kill", _raise_winerror_5)
    with pytest.raises(OSError):
        _pid_exists(12345)


def test_windows_get_parent_pid_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    from typing import Any

    from hawavoclean.watchdog import _windows_get_parent_pid

    # 1. windll is None
    monkeypatch.delattr(ctypes, "windll", raising=False)
    assert _windows_get_parent_pid(1234) is None

    # 2. snapshot returns -1
    class MockK32Fail:
        def CreateToolhelp32Snapshot(self, _flags: int, _pid: int) -> int:
            return -1

    monkeypatch.setattr(
        ctypes, "windll", type("MockWindll", (), {"kernel32": MockK32Fail()})(), raising=False
    )
    assert _windows_get_parent_pid(1234) is None

    # 2b. Process32FirstW returns False
    class MockK32FirstFail:
        def CreateToolhelp32Snapshot(self, _flags: int, _pid: int) -> int:
            return 100

        def Process32FirstW(self, _snap: int, _entry: Any) -> bool:
            return False

        def CloseHandle(self, _snap: int) -> bool:
            return True

    monkeypatch.setattr(
        ctypes,
        "windll",
        type("MockWindll", (), {"kernel32": MockK32FirstFail()})(),
        raising=False,
    )
    assert _windows_get_parent_pid(1234) is None

    # 3. Process iteration finds target PID
    class MockK32Success:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def CreateToolhelp32Snapshot(self, _flags: int, _pid: int) -> int:
            return 100

        def Process32FirstW(self, _snap: int, byref_entry: Any) -> bool:
            entry = byref_entry._obj
            entry.th32ProcessID = 999
            entry.th32ParentProcessID = 888
            return True

        def Process32NextW(self, _snap: int, byref_entry: Any) -> bool:
            self.calls += 1
            if self.calls == 1:
                entry = byref_entry._obj
                entry.th32ProcessID = 1234
                entry.th32ParentProcessID = 5678
                return True
            return False

        def CloseHandle(self, _snap: int) -> bool:
            self.closed = True
            return True

    mock_success = MockK32Success()
    monkeypatch.setattr(
        ctypes, "windll", type("MockWindll", (), {"kernel32": mock_success})(), raising=False
    )
    assert _windows_get_parent_pid(1234) == 5678
    assert mock_success.closed

    # 4. Target PID not found in list
    mock_not_found = MockK32Success()
    monkeypatch.setattr(
        ctypes, "windll", type("MockWindll", (), {"kernel32": mock_not_found})(), raising=False
    )
    assert _windows_get_parent_pid(99999) is None

    # 5. Exception handling
    class MockK32Crash:
        def CreateToolhelp32Snapshot(self, _flags: int, _pid: int) -> int:
            raise RuntimeError("ctypes fault")

    monkeypatch.setattr(
        ctypes, "windll", type("MockWindll", (), {"kernel32": MockK32Crash()})(), raising=False
    )
    assert _windows_get_parent_pid(1234) is None


def test_install_parent_death_watchdog_windows_parent_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hawavoclean import watchdog

    monkeypatch.setenv(watchdog.PARENT_PID_ENV, "100")
    monkeypatch.setattr(os, "getppid", lambda: 200)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        watchdog, "_windows_get_parent_pid", lambda pid: 100 if pid == 200 else None
    )
    monkeypatch.setattr(watchdog, "parent_is_alive", lambda _pid: False)

    killed: list[int] = []

    def mock_kill(pid: int, _sig: object) -> None:
        killed.append(pid)

    monkeypatch.setattr(os, "kill", mock_kill)
    monkeypatch.setattr(os, "_exit", lambda _code: None)

    t = watchdog.install_parent_death_watchdog(poll_interval_s=0.01, grace_s=0.01)
    assert t is not None
    t.join(timeout=1.0)
    assert len(killed) > 0
