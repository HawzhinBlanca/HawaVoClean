"""ProcessSupervisor owns descendants on POSIX and Windows code paths."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from hawavoclean.process_supervisor import (
    WINDOWS_CREATE_NEW_PROCESS_GROUP,
    WINDOWS_CREATE_SUSPENDED,
    WINDOWS_CTRL_BREAK_EVENT,
    ProcessSupervisor,
    ProcessSupervisorError,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A killed orphan can briefly remain as a zombie until launchd/init reaps
    # it.  It is no longer executable and therefore counts as gone.
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(status) and not status.startswith("Z")


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group qualification")
def test_posix_termination_kills_stubborn_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "grandchild.pid"
    program = "\n".join(
        [
            "import pathlib, signal, subprocess, sys, time",
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])",
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "print('ready', flush=True)",
            "time.sleep(60)",
        ]
    )
    supervisor = ProcessSupervisor.spawn(
        [sys.executable, "-c", program],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc = supervisor.process
    grandchild_pid: int | None = None
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        grandchild_pid = int(marker.read_text())
        assert os.getpgid(proc.pid) == proc.pid
        assert os.getpgid(grandchild_pid) == proc.pid

        started = time.monotonic()
        supervisor.terminate_tree(0.15)
        assert proc.wait(timeout=5.0) != 0
        assert time.monotonic() - started < 3.0

        deadline = time.monotonic() + 5.0
        while _pid_is_live(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_is_live(grandchild_pid), "grandchild survived process-group termination"
    finally:
        supervisor.close(kill_remaining=True)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
        if grandchild_pid is not None and _pid_is_live(grandchild_pid):
            os.kill(grandchild_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group qualification")
def test_job_manager_cancel_reaps_the_worker_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "job-grandchild.pid"
    program = "\n".join(
        [
            "import json, pathlib, signal, subprocess, sys, time",
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "print(json.dumps({'event':'progress','stage':'decode','progress':0.1}), flush=True)",
            "time.sleep(60)",
        ]
    )
    manager = JobManager(
        command_factory=lambda _record: [sys.executable, "-c", program],
        kill_grace_s=0.15,
    )
    grandchild_pid: int | None = None
    try:
        job_id = manager.submit(
            input_path=tmp_path / "source.wav",
            output_path=tmp_path / "master.wav",
            profile="production",
            overwrite=False,
        )["job_id"]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = manager.get_status(job_id)
            if status is not None and status["stage"] == "decode" and marker.exists():
                break
            time.sleep(0.02)
        else:
            pytest.fail("job child did not start")

        grandchild_pid = int(marker.read_text())
        assert _pid_is_live(grandchild_pid)
        assert manager.cancel(job_id) is True

        while time.monotonic() < deadline:
            status = manager.get_status(job_id)
            if status is not None and status["state"] in TERMINAL_STATES:
                break
            time.sleep(0.02)
        else:
            pytest.fail("cancelled job did not reach a terminal state")
        assert status["state"] == "cancelled"

        while _pid_is_live(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_is_live(grandchild_pid), "server cancel orphaned a worker descendant"
    finally:
        manager.shutdown(grace_s=0.1)
        if grandchild_pid is not None and _pid_is_live(grandchild_pid):
            os.kill(grandchild_pid, signal.SIGKILL)


class _FakeProcess:
    pid = 4321

    def __init__(self, *, signal_error: bool = False) -> None:
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False
        self.waited = False
        self.signal_error = signal_error

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        if self.signal_error:
            raise OSError("no console")

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        return self.returncode or 0


class _FakeWindowsJobApi:
    def __init__(self, *, fail_assignment: bool = False, fail_resume: bool = False) -> None:
        self.fail_assignment = fail_assignment
        self.fail_resume = fail_resume
        self.created = 0
        self.assigned: list[tuple[int, int]] = []
        self.resumed: list[int] = []
        self.terminated: list[tuple[int, int]] = []
        self.closed: list[int] = []
        self.events: list[str] = []

    def create_kill_on_close_job(self) -> int:
        self.created += 1
        self.events.append("job-created")
        return 9001

    def assign_process(self, job_handle: int, pid: int) -> None:
        self.events.append("assigned")
        self.assigned.append((job_handle, pid))
        if self.fail_assignment:
            raise OSError("assignment rejected")

    def resume_process(self, pid: int) -> None:
        self.events.append("resumed")
        self.resumed.append(pid)
        if self.fail_resume:
            raise OSError("resume rejected")

    def terminate_job(self, job_handle: int, exit_code: int) -> None:
        self.terminated.append((job_handle, exit_code))

    def close_handle(self, job_handle: int) -> None:
        self.closed.append(job_handle)


def test_windows_spawn_assigns_job_and_terminates_it_before_close() -> None:
    process = _FakeProcess()
    api = _FakeWindowsJobApi()
    captured: dict[str, Any] = {}

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        api.events.append("spawned-suspended")
        captured["command"] = command
        captured.update(kwargs)
        return cast(subprocess.Popen[str], process)

    supervisor = ProcessSupervisor.spawn(
        ["engine.exe", "process"],
        platform="windows",
        windows_api=api,
        popen_factory=factory,
        creationflags=0x10,
        start_new_session=True,
        text=True,
    )
    assert captured["creationflags"] == (
        0x10 | WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_SUSPENDED
    )
    assert "start_new_session" not in captured
    assert api.created == 1
    assert api.assigned == [(9001, process.pid)]
    assert api.resumed == [process.pid]
    assert api.events == ["job-created", "spawned-suspended", "assigned", "resumed"]

    supervisor.terminate_tree(0.0)
    assert process.signals == [WINDOWS_CTRL_BREAK_EVENT]
    assert api.terminated == [(9001, 1)]
    assert not process.killed, "root-only kill was used despite a working Job Object"

    supervisor.close()
    supervisor.close()
    assert api.closed == [9001]


def test_windows_ctrl_break_failure_falls_back_to_leader_then_job() -> None:
    process = _FakeProcess(signal_error=True)
    api = _FakeWindowsJobApi()

    supervisor = ProcessSupervisor.spawn(
        ["engine.exe"],
        platform="nt",
        windows_api=api,
        popen_factory=lambda *_a, **_kw: cast(subprocess.Popen[str], process),
    )
    supervisor.terminate_tree(0.0)
    assert process.terminated is True
    assert api.terminated == [(9001, 1)]
    supervisor.close()


@pytest.mark.parametrize("failure", ["assign", "resume"])
def test_windows_job_containment_failure_kills_and_reaps_child(failure: str) -> None:
    process = _FakeProcess()
    api = _FakeWindowsJobApi(
        fail_assignment=failure == "assign",
        fail_resume=failure == "resume",
    )

    with pytest.raises(ProcessSupervisorError, match="could not atomically contain child"):
        ProcessSupervisor.spawn(
            ["engine.exe"],
            platform="win32",
            windows_api=api,
            popen_factory=lambda *_a, **_kw: cast(subprocess.Popen[str], process),
        )

    assert process.killed is True
    assert process.waited is True
    assert api.terminated == [(9001, 1)]
    assert api.closed == [9001]


def test_posix_spawn_forces_a_new_session_in_factory() -> None:
    process = _FakeProcess()
    captured: dict[str, Any] = {}

    def factory(_command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        captured.update(kwargs)
        return cast(subprocess.Popen[str], process)

    supervisor = ProcessSupervisor.spawn(
        ["engine"],
        platform="posix",
        popen_factory=factory,
        start_new_session=False,
    )
    try:
        assert captured["start_new_session"] is True
        assert "creationflags" not in captured
    finally:
        # The fake pid must never be signalled on the real host.
        supervisor.close(kill_remaining=False)
