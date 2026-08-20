"""JobManager: child-process lifecycle, FIFO queue, cancel, failure mapping,
asyncio subscriptions. Uses fake children (python -c) so it runs in seconds."""

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from hawavoclean.server.jobs import (
    TERMINAL_STATES,
    JobManager,
    JobRecord,
    default_command,
)

pytestmark = pytest.mark.unit


def _py(script: str) -> list[str]:
    return [sys.executable, "-u", "-c", textwrap.dedent(script)]


def _wait_terminal(manager: JobManager, job_id: str, timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = manager.get_status(job_id)
        assert snap is not None
        if snap["state"] in TERMINAL_STATES:
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {manager.get_status(job_id)}")


def _wait_state(manager: JobManager, job_id: str, state: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = manager.get_status(job_id)
        if snap is not None and snap["state"] == state:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {state}: {manager.get_status(job_id)}")


def _wait_stage(manager: JobManager, job_id: str, stage: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = manager.get_status(job_id)
        if snap is not None and snap["stage"] == stage:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached stage {stage}")


def _success_factory(record: JobRecord) -> list[str]:
    """A fake child that speaks the contract: progress lines, writes the
    report sidecar, then ``done``; also chatters on stderr and stdout."""
    report = {"schema_version": 1, "job_id": "fake", "summary": {"units_total": 1}}
    return _py(
        f"""
        import json, sys, time
        def emit(o):
            sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        sys.stderr.write("[INFO] hello from stderr\\n")
        emit({{"event":"progress","stage":"preflight","progress":0.02,"message":"Preflight"}})
        print("not json at all")
        emit({{"event":"progress","stage":"enhance","progress":0.3,"message":"Enhancing unit 1/2",
              "unit":{{"index":1,"total":2}}}})
        time.sleep(0.05)
        emit({{"event":"progress","stage":"publish","progress":0.98,"message":"Publishing"}})
        open({str(record.report_path)!r}, "w").write({json.dumps(json.dumps(report))})
        open({str(record.output_path)!r}, "wb").write(b"RIFF")
        emit({{"event":"done","progress":1.0,"output_path":{str(record.output_path)!r},
              "report_path":{str(record.report_path)!r}}})
        """
    )


def test_default_command_matches_contract(tmp_path: Path) -> None:
    rec = JobRecord(
        job_id="j_x",
        input_path=tmp_path / "in.m4a.mp4",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.hawavoclean.json",
        profile="studio",
        overwrite=True,
    )
    cmd = default_command(rec)
    assert cmd[:4] == [sys.executable, "-m", "hawavoclean.cli", "process"]
    assert cmd[4:] == [
        str(tmp_path / "in.m4a.mp4"),
        "-o",
        str(tmp_path / "out.wav"),
        "--profile",
        "studio",
        "--overwrite",
        "--progress-json",
    ]
    rec.overwrite = False
    assert "--overwrite" not in default_command(rec)


def test_success_path_parses_progress_and_loads_report(tmp_path: Path) -> None:
    manager = JobManager(command_factory=_success_factory)
    try:
        snap = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "out.wav",
            profile="production",
            overwrite=False,
        )
        assert snap["state"] == "queued" and snap["stage"] == "preflight"
        assert snap["progress"] == 0.0 and snap["message"] == "Queued"
        assert snap["started_at"] is None and snap["finished_at"] is None
        assert snap["report_path"] == str(tmp_path / "out.hawavoclean.json")
        assert "error" not in snap and "report" not in snap and "unit" not in snap
        assert snap["job_id"].startswith("j_")

        final = _wait_terminal(manager, snap["job_id"])
        assert final["state"] == "done"
        assert final["stage"] == "done" and final["progress"] == 1.0
        assert final["message"] == "Done"
        assert final["report"]["summary"]["units_total"] == 1
        assert final["started_at"] is not None and final["finished_at"] is not None
        assert final["started_at"].endswith("Z")
        assert "unit" not in final and "error" not in final
        assert manager.list_jobs()[0]["job_id"] == snap["job_id"]
    finally:
        manager.shutdown()


def test_unit_is_tracked_and_cleared(tmp_path: Path) -> None:
    """While a unit is being enhanced the snapshot carries ``unit``; later
    stages without a unit clear it."""
    seen: list[dict[str, Any]] = []

    def factory(record: JobRecord) -> list[str]:
        return _py(
            f"""
            import json, sys, time
            def emit(o):
                sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
            emit({{"event":"progress","stage":"enhance","progress":0.3,"message":"Enhancing",
                  "unit":{{"index":2,"total":5}}}})
            time.sleep(0.4)
            emit({{"event":"progress","stage":"finish","progress":0.85,"message":"Finishing"}})
            time.sleep(0.4)
            open({str(record.report_path)!r}, "w").write("{{}}")
            emit({{"event":"done","progress":1.0}})
            """
        )

    manager = JobManager(command_factory=factory)
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "out.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        _wait_stage(manager, job_id, "enhance")
        snap = manager.get_status(job_id)
        assert snap is not None and snap["unit"] == {"index": 2, "total": 5}
        seen.append(snap)
        _wait_stage(manager, job_id, "finish")
        snap = manager.get_status(job_id)
        assert snap is not None and "unit" not in snap
        assert _wait_terminal(manager, job_id)["state"] == "done"
    finally:
        manager.shutdown()


def test_failure_uses_error_event_when_present(tmp_path: Path) -> None:
    def factory(_record: JobRecord) -> list[str]:
        return _py(
            """
            import json, sys
            sys.stderr.write("traceback-ish noise\\n")
            sys.stdout.write(json.dumps({"event":"error","code":"INVALID_USER_INPUT",
                                         "message":"Input audio file does not exist"}) + "\\n")
            sys.exit(4)
            """
        )

    manager = JobManager(command_factory=factory)
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "failed" and final["stage"] == "error"
        assert final["error"] == {
            "code": "INVALID_USER_INPUT",
            "message": "Input audio file does not exist",
        }
        assert final["message"] == "Input audio file does not exist"
        assert "report" not in final
    finally:
        manager.shutdown()


def test_failure_without_error_event_uses_exit_code_and_stderr_tail(tmp_path: Path) -> None:
    def factory(_record: JobRecord) -> list[str]:
        return _py(
            """
            import sys
            for i in range(80):
                sys.stderr.write(f"line {i}\\n")
            sys.stderr.write("RuntimeError: the real reason\\n\\n")
            sys.exit(2)
            """
        )

    manager = JobManager(command_factory=factory)
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "failed"
        assert final["error"]["code"] == "PREFLIGHT_FAILURE"  # ExitCode(2).name
        assert final["error"]["message"] == "RuntimeError: the real reason"
    finally:
        manager.shutdown()


def test_failure_unknown_exit_code_and_no_stderr(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py("import sys; sys.exit(7)"))
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["error"] == {"code": "EXIT_7", "message": "Worker exited with code 7"}
    finally:
        manager.shutdown()


def test_exit_zero_without_done_is_a_protocol_error(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py("print('bye')"))
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "failed"
        assert final["error"]["code"] == "PROTOCOL_ERROR"
    finally:
        manager.shutdown()


def test_done_with_unreadable_report_fails(tmp_path: Path) -> None:
    def factory(record: JobRecord) -> list[str]:
        return _py(
            f"""
            import json, sys
            open({str(record.report_path)!r}, "w").write("{{not json")
            sys.stdout.write(json.dumps({{"event":"done","progress":1.0}}) + "\\n")
            """
        )

    manager = JobManager(command_factory=factory)
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "failed"
        assert final["error"]["code"] == "PUBLICATION_FAILURE"
        assert "unreadable" in final["error"]["message"]
    finally:
        manager.shutdown()


def test_spawn_failure_is_reported(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: ["/definitely/not/a/binary"])
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "failed"
        assert final["error"]["code"] == "SPAWN_FAILED"
    finally:
        manager.shutdown()


_SLOW = """
import json, signal, sys, time
sys.stdout.write(json.dumps({"event":"progress","stage":"decode","progress":0.05,
                             "message":"slow"}) + "\\n"); sys.stdout.flush()
time.sleep(60)
"""

_STUBBORN = """
import json, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
sys.stdout.write(json.dumps({"event":"progress","stage":"decode","progress":0.05,
                             "message":"stubborn"}) + "\\n"); sys.stdout.flush()
time.sleep(60)
"""


def test_cancel_running_job_sigterm(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        _wait_stage(manager, job_id, "decode")
        t0 = time.monotonic()
        assert manager.cancel(job_id) is True
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "cancelled"
        assert time.monotonic() - t0 < 4.0  # SIGTERM honoured, no wait for SIGKILL
        assert "error" not in final and final["stage"] == "decode"
        assert final["finished_at"] is not None
        # Cancelling again is a no-op that still returns True.
        assert manager.cancel(job_id) is True
        assert manager.get_status(job_id)["state"] == "cancelled"  # type: ignore[index]
    finally:
        manager.shutdown()


def test_cancel_escalates_to_sigkill(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py(_STUBBORN), kill_grace_s=0.3)
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        _wait_stage(manager, job_id, "decode")
        t0 = time.monotonic()
        manager.cancel(job_id)
        final = _wait_terminal(manager, job_id)
        elapsed = time.monotonic() - t0
        assert final["state"] == "cancelled"
        assert 0.25 <= elapsed < 5.0
    finally:
        manager.shutdown()


def test_cancel_queued_job_and_fifo_order(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    try:
        first = manager.submit(
            input_path=tmp_path / "a.wav",
            output_path=tmp_path / "a_out.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        second = manager.submit(
            input_path=tmp_path / "b.wav",
            output_path=tmp_path / "b_out.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        third = manager.submit(
            input_path=tmp_path / "c.wav",
            output_path=tmp_path / "c_out.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        _wait_stage(manager, first, "decode")
        assert manager.get_status(second)["state"] == "queued"  # type: ignore[index]
        assert manager.get_status(third)["message"] == "Queued (position 3)"  # type: ignore[index]
        # Cancel a queued job: immediate, never runs.
        assert manager.cancel(second) is True
        snap = manager.get_status(second)
        assert snap is not None and snap["state"] == "cancelled"
        assert snap["started_at"] is None
        # Cancel the running one -> the third starts (FIFO continues).
        manager.cancel(first)
        _wait_state(manager, first, "cancelled")
        _wait_state(manager, third, "running")
        _wait_stage(manager, third, "decode")
        manager.cancel(third)
        _wait_state(manager, third, "cancelled")
    finally:
        manager.shutdown()


def test_cancel_unknown_job_returns_false() -> None:
    manager = JobManager(command_factory=lambda _r: _py(_SLOW))
    try:
        assert manager.cancel("j_nope") is False
        assert manager.get_status("j_nope") is None
    finally:
        manager.shutdown()


def test_shutdown_cancels_queue_and_kills_running(tmp_path: Path) -> None:
    manager = JobManager(command_factory=lambda _r: _py(_STUBBORN))
    running = manager.submit(
        input_path=tmp_path / "a.wav",
        output_path=tmp_path / "a_out.wav",
        profile="studio",
        overwrite=False,
    )["job_id"]
    queued = manager.submit(
        input_path=tmp_path / "b.wav",
        output_path=tmp_path / "b_out.wav",
        profile="studio",
        overwrite=False,
    )["job_id"]
    _wait_stage(manager, running, "decode")
    t0 = time.monotonic()
    manager.shutdown(grace_s=0.3)
    assert time.monotonic() - t0 < 5.0
    assert manager.get_status(queued)["state"] == "cancelled"  # type: ignore[index]
    final = _wait_terminal(manager, running, timeout=5.0)
    assert final["state"] == "cancelled"
    manager.shutdown()  # idempotent
    with pytest.raises(RuntimeError):
        manager.submit(
            input_path=tmp_path / "c.wav",
            output_path=tmp_path / "c.wav",
            profile="studio",
            overwrite=False,
        )


def test_subscribers_receive_every_change_in_order(tmp_path: Path) -> None:
    manager = JobManager(command_factory=_success_factory)

    async def collect() -> tuple[list[dict[str, Any]], str]:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "out.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        queue = manager.subscribe(job_id)
        assert queue is not None
        assert manager.subscribe("j_nope") is None
        received: list[dict[str, Any]] = []
        while True:
            snap = await asyncio.wait_for(queue.get(), timeout=15)
            received.append(snap)
            if snap["state"] in TERMINAL_STATES:
                break
        manager.unsubscribe(job_id, queue)
        manager.unsubscribe(job_id, queue)  # harmless twice
        return received, job_id

    try:
        received, _job_id = asyncio.run(collect())
        states = [s["state"] for s in received]
        assert states[0] == "running" and states[-1] == "done"
        stages = [s["stage"] for s in received]
        assert stages == ["preflight", "preflight", "enhance", "publish", "done"]
        progress = [s["progress"] for s in received]
        assert progress == sorted(progress)
        seqs = [s["seq"] for s in received]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # strictly increasing
        assert received[2]["unit"] == {"index": 1, "total": 2}
        assert "report" in received[-1]
    finally:
        manager.shutdown()


def test_subscriber_on_closed_loop_is_dropped(tmp_path: Path) -> None:
    """A subscriber whose loop is gone must be discarded, not crash the worker."""
    manager = JobManager(command_factory=_success_factory)
    job_id = manager.submit(
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "out.wav",
        profile="studio",
        overwrite=False,
    )["job_id"]
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    with manager._lock:
        manager._subscribers.setdefault(job_id, []).append((loop, queue))
    loop.close()
    try:
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "done"
        assert manager._subscribers.get(job_id) == []
    finally:
        manager.shutdown()


def test_runner_crash_marks_job_failed(tmp_path: Path) -> None:
    manager = JobManager(command_factory=_success_factory)

    def boom(_record: JobRecord) -> None:
        raise RuntimeError("runner bug")

    manager._run_job = boom  # type: ignore[assignment]
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "failed"
        assert final["error"] == {"code": "INTERNAL", "message": "runner bug"}
    finally:
        manager.shutdown()


def test_cancel_racing_the_spawn_is_honoured(tmp_path: Path) -> None:
    """Cancel arriving after the job went 'running' but before Popen returned."""
    holder: dict[str, JobManager] = {}

    def factory(record: JobRecord) -> list[str]:
        holder["m"].cancel(record.job_id)  # the factory runs just before Popen
        return _py(_SLOW)

    manager = JobManager(command_factory=factory)
    holder["m"] = manager
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        final = _wait_terminal(manager, job_id)
        assert final["state"] == "cancelled"
    finally:
        manager.shutdown()


def test_non_object_json_lines_are_ignored(tmp_path: Path) -> None:
    def factory(record: JobRecord) -> list[str]:
        return _py(
            f"""
            import json, sys
            sys.stdout.write("[1, 2, 3]\\n"); sys.stdout.write('"just a string"\\n')
            open({str(record.report_path)!r}, "w").write("{{}}")
            sys.stdout.write(json.dumps({{"event":"done","progress":1.0}}) + "\\n")
            """
        )

    manager = JobManager(command_factory=factory)
    try:
        job_id = manager.submit(
            input_path=tmp_path / "in.wav",
            output_path=tmp_path / "o.wav",
            profile="studio",
            overwrite=False,
        )["job_id"]
        assert _wait_terminal(manager, job_id)["state"] == "done"
    finally:
        manager.shutdown()
