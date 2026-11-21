"""ProcessSupervisor owns descendants on POSIX and Windows code paths."""

from __future__ import annotations

import contextlib
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

REPO = Path(__file__).resolve().parents[2]
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


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
            os.kill(grandchild_pid, _SIGKILL)


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
            os.kill(grandchild_pid, _SIGKILL)


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


def test_ctypes_windows_job_api_full_qualification(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    from hawavoclean.process_supervisor import _NativeWindowsJobApi

    class _MockKernel32:
        def __init__(self) -> None:
            self.CreateJobObjectW = lambda *_a: 1001
            self.SetInformationJobObject = lambda *_a: 1
            self.OpenProcess = lambda *_a: 2001
            self.AssignProcessToJobObject = lambda *_a: 1
            self.CreateToolhelp32Snapshot = lambda *_a: 3001
            self.Thread32First = lambda *_a: True
            self.Thread32Next = lambda *_a: False
            self.OpenThread = lambda *_a: 4001
            self.ResumeThread = lambda *_a: 1
            self.TerminateJobObject = lambda *_a: 1
            self.CloseHandle = lambda *_a: 1
            self.GetLastError = lambda: 0

    mock_k32 = _MockKernel32()

    class _MockWinDLL:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            return getattr(mock_k32, name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(mock_k32, name, value)

    monkeypatch.setitem(vars(ctypes), "WinDLL", _MockWinDLL)

    api = _NativeWindowsJobApi()

    # 1. create_kill_on_close_job
    h_job = api.create_kill_on_close_job()
    assert h_job == 1001

    mock_k32.CreateJobObjectW = lambda *_a: 0
    with pytest.raises(OSError):
        api.create_kill_on_close_job()
    mock_k32.CreateJobObjectW = lambda *_a: 1001

    mock_k32.SetInformationJobObject = lambda *_a: 0
    with pytest.raises(OSError):
        api.create_kill_on_close_job()
    mock_k32.SetInformationJobObject = lambda *_a: 1

    # 2. assign_process
    api.assign_process(1001, 1234)
    mock_k32.OpenProcess = lambda *_a: 0
    with pytest.raises(OSError):
        api.assign_process(1001, 1234)
    mock_k32.OpenProcess = lambda *_a: 2001

    mock_k32.AssignProcessToJobObject = lambda *_a: 0
    with pytest.raises(OSError):
        api.assign_process(1001, 1234)
    mock_k32.AssignProcessToJobObject = lambda *_a: 1

    # 3. resume_process
    def _tfirst(_h: Any, ptr: Any) -> bool:
        entry = ptr._obj if hasattr(ptr, "_obj") else ptr
        entry.th32OwnerProcessID = 1234
        entry.th32ThreadID = 5678
        return True

    mock_k32.Thread32First = _tfirst
    mock_k32.Thread32Next = lambda *_a: False
    api.resume_process(1234)

    mock_k32.CreateToolhelp32Snapshot = lambda *_a: 0
    with pytest.raises(OSError):
        api.resume_process(1234)
    mock_k32.CreateToolhelp32Snapshot = lambda *_a: 3001

    mock_k32.OpenThread = lambda *_a: 0
    with pytest.raises(OSError):
        api.resume_process(1234)
    mock_k32.OpenThread = lambda *_a: 4001

    mock_k32.ResumeThread = lambda *_a: 0xFFFFFFFF
    with pytest.raises(OSError):
        api.resume_process(1234)
    mock_k32.ResumeThread = lambda *_a: 1

    def _tfirst_mismatch(_h: Any, ptr: Any) -> bool:
        entry = ptr._obj if hasattr(ptr, "_obj") else ptr
        entry.th32OwnerProcessID = 9999
        entry.th32ThreadID = 5678
        return True

    mock_k32.Thread32First = _tfirst_mismatch
    with pytest.raises(OSError, match="expected one suspended initial thread"):
        api.resume_process(1234)

    # 4. terminate_job
    api.terminate_job(1001, 1)
    mock_k32.TerminateJobObject = lambda *_a: 0
    with pytest.raises(OSError):
        api.terminate_job(1001, 1)
    mock_k32.TerminateJobObject = lambda *_a: 1

    # 5. close_handle
    api.close_handle(1001)
    mock_k32.CloseHandle = lambda *_a: 0
    with pytest.raises(OSError):
        api.close_handle(1001)


def test_process_supervisor_branches_and_edge_cases() -> None:
    # 1. Invalid platform
    with pytest.raises(ValueError, match="unsupported process-supervisor platform"):
        ProcessSupervisor.spawn(["cmd"], platform="invalid_platform")

    # 2. Context manager and platform property
    proc = _FakeProcess()
    with ProcessSupervisor.spawn(
        ["cmd"],
        platform="posix",
        popen_factory=lambda *_a, **_kw: cast(subprocess.Popen[str], proc),
    ) as sup:
        assert sup.platform == "posix"
        with pytest.raises(ValueError, match="grace_s must be non-negative"):
            sup.terminate_tree(-1.0)

    # 3. Factory exception cleans up windows job handle
    api = _FakeWindowsJobApi()

    def _boom_factory(*_a: Any, **_kw: Any) -> subprocess.Popen[str]:
        raise RuntimeError("spawn failed")

    with pytest.raises(RuntimeError, match="spawn failed"):
        ProcessSupervisor.spawn(
            ["cmd"],
            platform="nt",
            windows_api=api,
            popen_factory=_boom_factory,
        )
    assert api.closed == [9001]

    # 4. request_graceful_shutdown and kill_tree on closed supervisor
    sup2 = ProcessSupervisor.spawn(
        ["cmd"],
        platform="nt",
        windows_api=api,
        popen_factory=lambda *_a, **_kw: cast(subprocess.Popen[str], proc),
    )
    sup2.close()
    # Calling on closed supervisor is a safe no-op
    sup2.request_graceful_shutdown()
    sup2.kill_tree()

    # 5. kill_tree when api.terminate_job raises OSError
    class _FailingTerminateApi(_FakeWindowsJobApi):
        def terminate_job(self, _job_handle: int, _exit_code: int) -> None:
            raise OSError(5, "Access Denied")

    proc3 = _FakeProcess()
    sup3 = ProcessSupervisor.spawn(
        ["cmd"],
        platform="nt",
        windows_api=_FailingTerminateApi(),
        popen_factory=lambda *_a, **_kw: cast(subprocess.Popen[str], proc3),
    )
    sup3.kill_tree()
    assert proc3.killed is True
    sup3.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group qualification")
def test_nested_process_tree_cancellation_within_ten_seconds(tmp_path: Path) -> None:
    """Qualify 4-tier stubborn descendant tree cancellation exits within 10 seconds.

    Hierarchy: Leader -> Child -> Grandchild -> Great-Grandchild.
    All 3 descendant tiers ignore SIGTERM and attempt to keep running indefinitely.
    ProcessSupervisor.terminate_tree must eliminate the complete tree in <= 10s.
    """
    child_pid_file = tmp_path / "child.pid"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    great_grandchild_pid_file = tmp_path / "great_grandchild.pid"

    # Multi-tier script where each descendant ignores SIGTERM and spawns the next tier
    leader_script = "\n".join(
        [
            "import os, pathlib, signal, subprocess, sys, time",
            "child_code = '''",
            "import os, pathlib, signal, subprocess, sys, time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            'grandchild_code = \\"\\"\\"',
            "import os, pathlib, signal, subprocess, sys, time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            'great_code = \\"import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)\\"',
            "great = subprocess.Popen([sys.executable, '-c', great_code])",
            f"pathlib.Path({str(great_grandchild_pid_file)!r}).write_text(str(great.pid))",
            "time.sleep(120)",
            '\\"\\"\\"',
            "gc = subprocess.Popen([sys.executable, '-c', grandchild_code])",
            f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(gc.pid))",
            "time.sleep(120)",
            "'''",
            "c = subprocess.Popen([sys.executable, '-c', child_code])",
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(c.pid))",
            "print('ready', flush=True)",
            "time.sleep(120)",
        ]
    )

    supervisor = ProcessSupervisor.spawn(
        [sys.executable, "-c", leader_script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc = supervisor.process
    child_pid: int | None = None
    grandchild_pid: int | None = None
    great_grandchild_pid: int | None = None

    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        assert line == "ready"

        # Wait up to 5s for all 3 descendant tiers to initialize and write PIDs
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if (
                child_pid_file.exists()
                and grandchild_pid_file.exists()
                and great_grandchild_pid_file.exists()
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Nested child hierarchy failed to initialize PIDs")

        child_pid = int(child_pid_file.read_text().strip())
        grandchild_pid = int(grandchild_pid_file.read_text().strip())
        great_grandchild_pid = int(great_grandchild_pid_file.read_text().strip())

        # Verify all 4 tiers are alive and share the leader's process group
        for p in (proc.pid, child_pid, grandchild_pid, great_grandchild_pid):
            assert _pid_is_live(p)
            assert os.getpgid(p) == proc.pid

        # Terminate the complete tree and verify timing <= 10 seconds
        t0 = time.monotonic()
        supervisor.terminate_tree(grace_s=0.25)
        assert proc.wait(timeout=5.0) != 0

        poll_deadline = time.monotonic() + 8.0
        while time.monotonic() < poll_deadline:
            if not any(
                _pid_is_live(p) for p in (proc.pid, child_pid, grandchild_pid, great_grandchild_pid)
            ):
                break
            time.sleep(0.05)

        elapsed = time.monotonic() - t0
        assert elapsed <= 10.0, f"Process tree termination took {elapsed:.2f}s (> 10s)"

        # Verify absolutely no orphan remains across any tier
        assert not _pid_is_live(proc.pid), "Leader survived termination"
        assert not _pid_is_live(child_pid), "Child survived termination"
        assert not _pid_is_live(grandchild_pid), "Grandchild survived termination"
        assert not _pid_is_live(great_grandchild_pid), "Great-grandchild survived termination"
    finally:
        supervisor.close(kill_remaining=True)
        for cleanup_pid in (proc.pid, child_pid, grandchild_pid, great_grandchild_pid):
            if cleanup_pid is not None and _pid_is_live(cleanup_pid):
                with contextlib.suppress(OSError):
                    os.kill(cleanup_pid, _SIGKILL)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group qualification")
def test_job_manager_cancels_heavy_job_and_starts_next_within_five_seconds(
    tmp_path: Path,
) -> None:
    """Qualify cancel + next heavy job starts within 5 seconds without lock collision.

    Required completion evidence: Complete tree exits within 10s; next heavy job
    starts within five seconds; no orphan or locked artifact remains.
    """
    import soundfile as sf

    marker = tmp_path / "heavy_grandchild.pid"
    out_path = tmp_path / "master.wav"

    # Simulated heavy job 1: spawns stubborn grandchild and sleeps
    heavy_job_1_code = "\n".join(
        [
            "import json, pathlib, signal, subprocess, sys, time",
            "gc = subprocess.Popen([sys.executable, '-c', "
            "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)'])",
            f"pathlib.Path({str(marker)!r}).write_text(str(gc.pid))",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "print(json.dumps({'event':'progress','stage':'enhance','progress':0.25}), flush=True)",
            "time.sleep(120)",
        ]
    )

    # Simulated heavy job 2: writes output WAV and completes with report
    heavy_job_2_code = "\n".join(
        [
            "import json, pathlib, sys, time",
            "import numpy as np, soundfile as sf",
            f"out_file = pathlib.Path({str(out_path)!r})",
            "sf.write(str(out_file), np.zeros((4800, 1), dtype=np.float32), 48000)",
            "rep = out_file.parent / f'{out_file.stem}.hawavoclean.json'",
            "rep.write_text(json.dumps({'schema': 1, 'decision': 'pass', 'audio_sha256': '0'*64}))",
            "print(json.dumps({'event':'progress','stage':'enhance','progress':0.5}), flush=True)",
            "print(json.dumps({'event':'done','report_path':str(rep)}), flush=True)",
        ]
    )

    commands = {
        "job1": [sys.executable, "-c", heavy_job_1_code],
        "job2": [sys.executable, "-c", heavy_job_2_code],
    }

    manager = JobManager(
        command_factory=lambda record: commands[record.profile],
        kill_grace_s=0.3,
    )

    grandchild_pid: int | None = None
    try:
        # 1. Submit heavy job 1
        j1 = manager.submit(
            input_path=tmp_path / "in1.wav",
            output_path=out_path,
            profile="job1",
            overwrite=True,
        )
        j1_id = j1["job_id"]

        # Wait until job 1 is running and its stubborn grandchild is confirmed alive
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            st = manager.get_status(j1_id)
            if st is not None and st["stage"] == "enhance" and marker.exists():
                break
            time.sleep(0.02)
        else:
            pytest.fail("Heavy job 1 failed to reach enhance stage")

        grandchild_pid = int(marker.read_text().strip())
        assert _pid_is_live(grandchild_pid), "Heavy job 1 grandchild not alive"

        # 2. Cancel Job 1 and start stopwatch T0
        t0 = time.monotonic()
        cancelled_ok = manager.cancel(j1_id, wait=True, timeout_s=4.0)
        assert cancelled_ok is True

        # 3. Immediately submit heavy job 2 targeting the EXACT SAME output path
        j2 = manager.submit(
            input_path=tmp_path / "in2.wav",
            output_path=out_path,
            profile="job2",
            overwrite=True,
        )
        j2_id = j2["job_id"]

        # 4. Wait for Job 2 to transition to running stage and record T1
        run_deadline = time.monotonic() + 5.0
        while time.monotonic() < run_deadline:
            st = manager.get_status(j2_id)
            if st is not None and st["state"] in ("running", "done"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Heavy job 2 did not start within 5 seconds")

        t1 = time.monotonic()
        turnaround_time = t1 - t0
        assert turnaround_time <= 5.0, f"Next heavy job took {turnaround_time:.2f}s to start (> 5s)"

        # 5. Wait for Job 2 to finish completely
        done_deadline = time.monotonic() + 5.0
        while time.monotonic() < done_deadline:
            st = manager.get_status(j2_id)
            if st is not None and st["state"] in TERMINAL_STATES:
                break
            time.sleep(0.02)
        assert st is not None and st["state"] == "done"

        # 6. Verify Job 1 reached cancelled and no orphan process remains
        st1 = manager.get_status(j1_id)
        assert st1 is not None and st1["state"] == "cancelled"
        assert not _pid_is_live(grandchild_pid), "Grandchild from Job 1 was orphaned!"

        # 7. Verify no locked artifact remains: output is valid and readable
        assert out_path.exists()
        data, sr = sf.read(str(out_path))
        assert sr == 48000
        assert len(data) == 4800
    finally:
        manager.shutdown(grace_s=0.2)
        if grandchild_pid is not None and _pid_is_live(grandchild_pid):
            with contextlib.suppress(OSError):
                os.kill(grandchild_pid, _SIGKILL)


@pytest.mark.skipif(os.name == "nt", reason="POSIX watchdog qualification")
def test_host_crash_terminates_complete_nested_tree_via_watchdog(tmp_path: Path) -> None:
    """Qualify host crash (SIGKILL of parent) cleans up complete multi-tier child tree.

    Simulates abrupt host crash (kernel OOM / panic / SIGKILL):
    - Host process spawns child with HAWAVOCLEAN_PARENT_PID.
    - Child arms install_parent_death_watchdog and spawns grandchild and great-grandchild.
    - Host is SIGKILLed.
    - Watchdog detects host death and unwinds; zero orphan processes remain.
    """
    child_marker = tmp_path / "watchdog_child.pid"
    grandchild_marker = tmp_path / "watchdog_grandchild.pid"
    great_marker = tmp_path / "watchdog_great.pid"
    ready_marker = tmp_path / "host_ready.txt"

    src_dir = str(REPO / "src")

    great_script = tmp_path / "great.py"
    great_script.write_text(
        "\n".join(
            [
                "import os, pathlib, sys, time",
                f"sys.path.insert(0, {src_dir!r})",
                "from hawavoclean.watchdog import parent_is_alive",
                "parent_pid = os.getppid()",
                f"pathlib.Path({str(great_marker)!r}).write_text(str(os.getpid()))",
                "while parent_is_alive(parent_pid):",
                "    time.sleep(0.05)",
                "sys.exit(0)",
            ]
        ),
        encoding="utf-8",
    )

    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        "\n".join(
            [
                "import os, pathlib, subprocess, sys, time",
                f"sys.path.insert(0, {src_dir!r})",
                "from hawavoclean.watchdog import parent_is_alive",
                "ppid = os.getppid()",
                f"great = subprocess.Popen([sys.executable, {str(great_script)!r}])",
                f"pathlib.Path({str(grandchild_marker)!r}).write_text(str(os.getpid()))",
                "try:",
                "    while parent_is_alive(ppid):",
                "        time.sleep(0.05)",
                "finally:",
                "    great.kill()",
                "    great.wait()",
            ]
        ),
        encoding="utf-8",
    )

    child_script = tmp_path / "child.py"
    child_script.write_text(
        "\n".join(
            [
                "import os, pathlib, signal, subprocess, sys, time",
                f"sys.path.insert(0, {src_dir!r})",
                "from hawavoclean.watchdog import install_parent_death_watchdog",
                "def _term_handler(sig, frame):",
                "    raise KeyboardInterrupt",
                "signal.signal(signal.SIGTERM, _term_handler)",
                "install_parent_death_watchdog(poll_interval_s=0.05, grace_s=0.2)",
                f"pathlib.Path({str(child_marker)!r}).write_text(str(os.getpid()))",
                f"gc = subprocess.Popen([sys.executable, {str(grandchild_script)!r}])",
                "try:",
                "    time.sleep(120)",
                "finally:",
                "    gc.kill()",
                "    gc.wait()",
            ]
        ),
        encoding="utf-8",
    )

    host_script = tmp_path / "host.py"
    host_script.write_text(
        "\n".join(
            [
                "import os, pathlib, subprocess, sys, time",
                f"sys.path.insert(0, {src_dir!r})",
                "host_pid = os.getpid()",
                "env = dict(os.environ)",
                "env['HAWAVOCLEAN_PARENT_PID'] = str(host_pid)",
                f"env['PYTHONPATH'] = {src_dir!r}",
                f"child = subprocess.Popen([sys.executable, {str(child_script)!r}], env=env)",
                f"pathlib.Path({str(ready_marker)!r}).write_text(str(host_pid))",
                "time.sleep(120)",
            ]
        ),
        encoding="utf-8",
    )

    host_proc = subprocess.Popen(
        [sys.executable, str(host_script)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_pid: int | None = None
    grandchild_pid: int | None = None
    great_pid: int | None = None

    try:
        # Wait until all 4 processes (host, child, grandchild, great) are running
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if (
                ready_marker.exists()
                and child_marker.exists()
                and grandchild_marker.exists()
                and great_marker.exists()
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("Host hierarchy failed to initialize PIDs")

        host_pid = host_proc.pid
        child_pid = int(child_marker.read_text().strip())
        grandchild_pid = int(grandchild_marker.read_text().strip())
        great_pid = int(great_marker.read_text().strip())

        for p in (host_pid, child_pid, grandchild_pid, great_pid):
            assert _pid_is_live(p)

        # Abrupt host crash: SIGKILL the host (no atexit, no finally, no cleanup)
        os.kill(host_pid, _SIGKILL)
        host_proc.wait(timeout=5.0)
        assert not _pid_is_live(host_pid)

        # The watchdog in the child must notice within seconds and terminate
        watchdog_deadline = time.monotonic() + 10.0
        while time.monotonic() < watchdog_deadline:
            if not any(_pid_is_live(p) for p in (child_pid, grandchild_pid, great_pid)):
                break
            time.sleep(0.05)

        assert not _pid_is_live(child_pid), "Child survived host crash"
        assert not _pid_is_live(grandchild_pid), "Grandchild survived host crash"
        assert not _pid_is_live(great_pid), "Great-grandchild survived host crash"
    finally:
        if host_proc.poll() is None:
            host_proc.kill()
            host_proc.wait(timeout=2.0)
        for cleanup_pid in (child_pid, grandchild_pid, great_pid):
            if cleanup_pid is not None and _pid_is_live(cleanup_pid):
                with contextlib.suppress(OSError):
                    os.kill(cleanup_pid, _SIGKILL)


def test_windows_job_object_nested_children_and_crash_contract() -> None:
    """Qualify Windows Job Object semantics for nested trees and host crash.

    Simulates:
    1. Multi-tier hierarchy containment (Leader, Child, Grandchild) inside Job Object.
    2. JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE termination when host handle closes abruptly.
    3. Multi-thread Toolhelp snapshot enumeration and initial thread resumption.
    4. Immediate terminate_job capability across all nested members.
    """
    import ctypes

    class _MultiProcessMockKernel32:
        def __init__(self) -> None:
            self.job_limits: dict[int, int] = {}
            self.job_processes: dict[int, set[int]] = {}
            self.closed_handles: list[int] = []
            self.terminated_jobs: list[tuple[int, int]] = []
            self.resumed_threads: list[int] = []
            self.threads: list[tuple[int, int]] = []  # (thread_id, process_id)
            self._next_handle = 1000
            self._thread_idx = 0

            def _create_job(*_a: Any) -> int:
                self._next_handle += 1
                h = self._next_handle
                self.job_processes[h] = set()
                return h

            def _set_info(job: int, _cls: int, ptr: Any, _sz: int) -> int:
                info = ptr._obj if hasattr(ptr, "_obj") else ptr
                self.job_limits[job] = int(info.BasicLimitInformation.LimitFlags)
                return 1

            def _open_proc(_acc: int, _inh: int, pid: int) -> int:
                return pid + 50000

            def _assign(job: int, proc_handle: int) -> int:
                pid = proc_handle - 50000
                self.job_processes.setdefault(job, set()).add(pid)
                return 1

            def _snap(_flags: int, _pid: int) -> int:
                return 88888

            def _tnext(_s: int, ptr: Any) -> int:
                if self._thread_idx >= len(self.threads):
                    return 0
                tid, pid = self.threads[self._thread_idx]
                self._thread_idx += 1
                entry = ptr._obj if hasattr(ptr, "_obj") else ptr
                entry.th32ThreadID = tid
                entry.th32OwnerProcessID = pid
                return 1

            def _tfirst(_s: int, ptr: Any) -> int:
                if not self.threads:
                    return 0
                self._thread_idx = 0
                return _tnext(_s, ptr)

            def _openthread(_acc: int, _inh: int, tid: int) -> int:
                return tid + 70000

            def _resume(h: int) -> int:
                tid = h - 70000
                self.resumed_threads.append(tid)
                return 1

            def _term_job(job: int, code: int) -> int:
                self.terminated_jobs.append((job, code))
                self.job_processes[job] = set()
                return 1

            def _close_handle(handle: int) -> int:
                self.closed_handles.append(handle)
                if handle in self.job_limits:
                    limit = self.job_limits[handle]
                    if limit & 0x0000_2000:
                        self.job_processes[handle] = set()
                return 1

            self.CreateJobObjectW = _create_job
            self.SetInformationJobObject = _set_info
            self.OpenProcess = _open_proc
            self.AssignProcessToJobObject = _assign
            self.CreateToolhelp32Snapshot = _snap
            self.Thread32First = _tfirst
            self.Thread32Next = _tnext
            self.OpenThread = _openthread
            self.ResumeThread = _resume
            self.TerminateJobObject = _term_job
            self.CloseHandle = _close_handle
            self.GetLastError = lambda: 0

    mock_k32 = _MultiProcessMockKernel32()

    class _MockWinDLL:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            return getattr(mock_k32, name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(mock_k32, name, value)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(vars(ctypes), "WinDLL", _MockWinDLL)

    try:
        from hawavoclean.process_supervisor import _NativeWindowsJobApi

        api = _NativeWindowsJobApi()

        # 1. Create Job Object with KILL_ON_JOB_CLOSE
        job = api.create_kill_on_close_job()
        assert job in mock_k32.job_limits
        assert mock_k32.job_limits[job] == 0x0000_2000

        # 2. Assign Leader, Child, and Grandchild (nested tree)
        leader_pid = 20001
        child_pid = 20002
        grandchild_pid = 20003
        api.assign_process(job, leader_pid)
        api.assign_process(job, child_pid)
        api.assign_process(job, grandchild_pid)
        assert mock_k32.job_processes[job] == {leader_pid, child_pid, grandchild_pid}

        # 3. Test multi-thread Toolhelp snapshot: snapshot contains threads from other
        # processes and threads from child; verify resume_process picks the exact match.
        mock_k32.threads = [
            (901, 99999),  # Other process
            (902, 88888),  # Other process
            (903, leader_pid),  # Target leader process initial thread
        ]
        api.resume_process(leader_pid)
        assert mock_k32.resumed_threads == [903]

        # 4. Terminate Job Object terminates all nested members
        api.terminate_job(job, 1)
        assert mock_k32.terminated_jobs == [(job, 1)]
        assert len(mock_k32.job_processes[job]) == 0

        # 5. Emulate host crash: create job, assign nested processes, close handle abruptly.
        job2 = api.create_kill_on_close_job()
        api.assign_process(job2, 30001)
        api.assign_process(job2, 30002)
        assert len(mock_k32.job_processes[job2]) == 2
        # Host crashes -> OS closes Job Object handle -> KILL_ON_JOB_CLOSE fires
        api.close_handle(job2)
        assert len(mock_k32.job_processes[job2]) == 0
    finally:
        monkeypatch.undo()
