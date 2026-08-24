"""Child-process job manager for the engine API.

Each job runs ``python -m hawavoclean.cli process IN -o OUT --profile P
[--overwrite] --progress-json`` in its own process (the same isolation the
batch command uses: a wedged decoder or model can never take the server
down). A reader thread turns the child's JSON progress lines into status
snapshots; stderr is tailed for the failure message. One job runs at a
time; the rest wait in a FIFO queue. Cancel is SIGTERM, then SIGKILL after
a grace period.

Subscribers (the SSE endpoint) register an ``asyncio.Queue`` bound to their
event loop; every status change is delivered with
``loop.call_soon_threadsafe`` so the worker thread never touches the loop.
"""

import asyncio
import contextlib
import json
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from hawavoclean.errors import ExitCode
from hawavoclean.logging import get_logger
from hawavoclean.paths import profiles_root
from hawavoclean.watchdog import child_env

logger = get_logger("server.jobs")

JobState = Literal["queued", "running", "done", "failed", "cancelled"]
TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})
STDERR_TAIL_LINES = 50
DEFAULT_KILL_GRACE_S = 5.0
DEFAULT_MAX_ACTIVE_JOBS = 8
DEFAULT_MAX_TERMINAL_JOBS = 256
DEFAULT_TERMINAL_JOB_TTL_S = 24 * 60 * 60.0


class QueueFullError(RuntimeError):
    """The bounded active-job queue has no capacity for another submission."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class JobRecord:
    """Mutable state of one job. Mutate only under ``JobManager._lock``."""

    job_id: str
    input_path: Path
    output_path: Path
    report_path: Path
    profile: str
    overwrite: bool
    mode: str = "natural"
    speaker_id: str | None = None
    cutoff_hz: float | None = None
    state: JobState = "queued"
    stage: str = "preflight"
    progress: float = 0.0
    message: str = "Queued"
    unit: dict[str, int] | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    terminal_at: float | None = None
    error: dict[str, str] | None = None
    report: dict[str, Any] | None = None
    cancel_requested: bool = False
    seq: int = 0  # bumps on every change; lets subscribers discard stale snapshots
    process: subprocess.Popen[str] | None = None
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=STDERR_TAIL_LINES))

    def snapshot(self) -> dict[str, Any]:
        """Contract ``JobStatus`` JSON."""
        out: dict[str, Any] = {
            "job_id": self.job_id,
            "seq": self.seq,
            "state": self.state,
            "stage": self.stage,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "report_path": str(self.report_path),
            "profile": self.profile,
            "mode": self.mode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.mode == "restore":
            # Natural-mode snapshots stay byte-compatible with revision 1
            # clients: the restore keys appear only when they mean something.
            out["speaker_id"] = self.speaker_id
            out["cutoff_hz"] = self.cutoff_hz
        if self.unit is not None:
            out["unit"] = dict(self.unit)
        if self.state == "failed" and self.error is not None:
            out["error"] = dict(self.error)
        if self.state == "done" and self.report is not None:
            out["report"] = self.report
        return out


CommandFactory = Callable[[JobRecord], list[str]]
TerminalCallback = Callable[[JobRecord], None]


def default_command(record: JobRecord) -> list[str]:
    """The contract child command (same isolation pattern as ``cli._run_one_isolated``)."""
    cmd = [
        sys.executable,
        "-m",
        "hawavoclean.cli",
        "process",
        str(record.input_path),
        "-o",
        str(record.output_path),
        "--profile",
        record.profile,
    ]
    if record.mode == "restore":
        # The speaker id is validated at the API boundary (``^[a-z0-9_]{1,64}$``)
        # before it may reach a child argv, and the profiles dir is resolved
        # here rather than left to the child's cwd-relative default: the engine
        # may be launched from anywhere.
        cmd += ["--mode", "restore", "--speaker-id", str(record.speaker_id)]
        cmd += ["--profiles-dir", str(profiles_root())]
        if record.cutoff_hz is not None:
            cmd += ["--cutoff-hz", str(record.cutoff_hz)]
    if record.overwrite:
        cmd.append("--overwrite")
    cmd.append("--progress-json")
    return cmd


def queue_position(queued: int, a_job_is_running: bool) -> int:
    """Where a just-submitted job stands in line, counting from 1.

    ``queued`` includes the newcomer; the job the worker is already running
    is no longer in the queue but is still ahead of it. Counting only the
    queue made the answer depend on whether the worker thread had got round
    to popping the first job yet — the same three submissions could report
    "position 3" or "position 2" from one run to the next, and the test that
    pinned it failed about one run in seven.
    """
    return queued + (1 if a_job_is_running else 0)


_Subscriber = tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]


class JobManager:
    """In-memory job table + single-worker FIFO queue over child processes."""

    def __init__(
        self,
        *,
        command_factory: CommandFactory | None = None,
        kill_grace_s: float = DEFAULT_KILL_GRACE_S,
        env: dict[str, str] | None = None,
        max_active_jobs: int = DEFAULT_MAX_ACTIVE_JOBS,
        max_terminal_jobs: int = DEFAULT_MAX_TERMINAL_JOBS,
        terminal_ttl_s: float = DEFAULT_TERMINAL_JOB_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        if max_terminal_jobs < 1:
            raise ValueError("max_terminal_jobs must be at least 1")
        if terminal_ttl_s <= 0:
            raise ValueError("terminal_ttl_s must be positive")
        self._command_factory: CommandFactory = command_factory or default_command
        self._kill_grace_s = kill_grace_s
        self._env = env
        self._max_active_jobs = max_active_jobs
        self._max_terminal_jobs = max_terminal_jobs
        self._terminal_ttl_s = terminal_ttl_s
        self._clock = clock
        self._jobs: dict[str, JobRecord] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._terminal_callbacks: list[TerminalCallback] = []
        self._pending_terminal: deque[JobRecord] = deque()
        self._closed = False
        self._worker = threading.Thread(target=self._run_loop, name="hawavoclean-jobs", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ public

    def submit(
        self,
        *,
        input_path: Path,
        output_path: Path,
        profile: str,
        overwrite: bool,
        mode: str = "natural",
        speaker_id: str | None = None,
        cutoff_hz: float | None = None,
    ) -> dict[str, Any]:
        """Queue a job; returns its initial status snapshot."""
        report_path = output_path.parent / f"{output_path.stem}.hawavoclean.json"
        record = JobRecord(
            job_id=f"j_{uuid.uuid4().hex[:16]}",
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            profile=profile,
            overwrite=overwrite,
            mode=mode,
            speaker_id=speaker_id,
            cutoff_hz=cutoff_hz,
        )
        with self._wake:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            self._prune_locked()
            active = sum(record.state not in TERMINAL_STATES for record in self._jobs.values())
            if active >= self._max_active_jobs:
                raise QueueFullError(
                    f"job queue is full ({active}/{self._max_active_jobs} active jobs)"
                )
            self._jobs[record.job_id] = record
            self._queue.append(record.job_id)
            running = any(r.state == "running" for r in self._jobs.values())
            position = queue_position(len(self._queue), running)
            record.message = "Queued" if position == 1 else f"Queued (position {position})"
            self._wake.notify_all()
            return record.snapshot()

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._prune_locked()
            record = self._jobs.get(job_id)
            return record.snapshot() if record is not None else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            return [r.snapshot() for r in self._jobs.values()]

    def active_input_paths(self) -> set[Path]:
        """Inputs that retention cleanup must not remove while queued/running."""
        with self._lock:
            return {
                record.input_path.resolve()
                for record in self._jobs.values()
                if record.state not in TERMINAL_STATES
            }

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the job lock, then run terminal cleanup *outside* it.

        Terminal callbacks are caller-supplied cleanup, and cleanup routinely
        has to ask this manager a question before it acts -- retention must
        know whether another live job still names an upload as its input
        before deleting it. Invoking callbacks from inside ``_finish_locked``
        meant asking that question re-entered a non-reentrant ``Lock`` and
        deadlocked the ``hawavoclean-jobs`` thread permanently: not an
        exception the surrounding ``try`` could log, just a thread that never
        came back, taking every later job with it.

        Callbacks also do filesystem work, which has no business holding a
        lock that every status query needs. So ``_finish_locked`` records who
        became terminal and this wrapper delivers them after the release.
        """
        try:
            with self._lock:
                yield
        finally:
            # ``finally``, not a plain suffix: a body that raises has still
            # marked jobs terminal, and their cleanup must not wait for the
            # next unrelated caller to flush the queue.
            self._drain_terminal_callbacks()

    def _drain_terminal_callbacks(self) -> None:
        """Deliver queued terminal records. Safe to call with the lock free."""
        while True:
            with self._lock:
                if not self._pending_terminal:
                    return
                record = self._pending_terminal.popleft()
                callbacks = list(self._terminal_callbacks)
            for callback in callbacks:
                try:
                    callback(record)
                except Exception as exc:  # cleanup cannot change the job verdict
                    logger.error(
                        f"Terminal cleanup for {record.job_id} failed: {exc}", exc_info=True
                    )

    def add_terminal_callback(self, callback: TerminalCallback) -> None:
        """Register idempotent, bounded cleanup invoked when any job becomes terminal."""
        with self._lock:
            self._terminal_callbacks.append(callback)

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns False for an unknown id;
        True otherwise (including the no-op on an already finished job)."""
        with self._locked():
            self._prune_locked()
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if record.state in TERMINAL_STATES:
                return True
            record.cancel_requested = True
            if record.state == "queued":
                with contextlib.suppress(ValueError):
                    self._queue.remove(job_id)
                self._finish_locked(record, "cancelled", "Cancelled before start")
                return True
            proc = record.process
        if proc is not None:
            self._terminate(proc)
        return True

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]] | None:
        """Register the calling event loop for status pushes. Returns None for
        an unknown job. Call from inside a running asyncio loop."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        with self._lock:
            self._prune_locked()
            if job_id not in self._jobs:
                return None
            self._subscribers.setdefault(job_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            remaining = [s for s in subs if s[1] is not queue]
            if remaining:
                self._subscribers[job_id] = remaining
            else:
                self._subscribers.pop(job_id, None)

    def shutdown(self, grace_s: float = 0.5) -> None:
        """Cancel everything: queued jobs are marked cancelled, the running
        child gets SIGTERM then SIGKILL after ``grace_s``. Idempotent."""
        with self._locked():
            if self._closed:
                return
            self._closed = True
            running: subprocess.Popen[str] | None = None
            for job_id in list(self._queue):
                record = self._jobs[job_id]
                record.cancel_requested = True
                self._finish_locked(record, "cancelled", "Cancelled: server shutting down")
            self._queue.clear()
            for record in self._jobs.values():
                if record.state == "running":
                    record.cancel_requested = True
                    running = record.process
            self._wake.notify_all()
        if running is not None and running.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                running.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + grace_s
            while running.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            if running.poll() is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    running.kill()
        self._worker.join(timeout=grace_s + 1.0)

    # ---------------------------------------------------------------- internal

    def _terminate(self, proc: subprocess.Popen[str]) -> None:
        """SIGTERM now; SIGKILL if still alive after the grace period."""
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.send_signal(signal.SIGTERM)

        def _reap() -> None:
            deadline = time.monotonic() + self._kill_grace_s
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if proc.poll() is None:
                logger.warning(f"Job child pid {proc.pid} ignored SIGTERM; sending SIGKILL")
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()

        threading.Thread(target=_reap, name="hawavoclean-job-kill", daemon=True).start()

    def _notify_locked(self, record: JobRecord) -> None:
        """Record a change (``seq``) and push the snapshot to every subscriber."""
        record.seq += 1
        subs = self._subscribers.get(record.job_id)
        if not subs:
            return
        snap = record.snapshot()
        alive: list[_Subscriber] = []

        def _put_latest(queue: asyncio.Queue[dict[str, Any]], value: dict[str, Any]) -> None:
            while not queue.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(value)

        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(_put_latest, queue, snap)
            except RuntimeError:
                continue  # loop closed: subscriber is gone
            alive.append((loop, queue))
        self._subscribers[record.job_id] = alive

    def _finish_locked(self, record: JobRecord, state: JobState, message: str) -> None:
        record.state = state
        record.finished_at = _now_iso()
        record.terminal_at = self._clock()
        record.message = message
        if state == "done":
            record.stage = "done"
            record.progress = 1.0
        elif state == "failed":
            record.stage = "error"
        self._notify_locked(record)
        # Queue the callbacks; ``_locked`` runs them once the lock is released.
        self._pending_terminal.append(record)

    def _prune_locked(self) -> None:
        """Expire terminal snapshots by age/count; never evict active jobs or subscribers."""
        now = self._clock()
        candidates = [
            record
            for record in self._jobs.values()
            if record.state in TERMINAL_STATES and record.terminal_at is not None
        ]
        expired = {
            record.job_id
            for record in candidates
            if now - cast(float, record.terminal_at) >= self._terminal_ttl_s
        }
        retained = [record for record in candidates if record.job_id not in expired]
        retained.sort(key=lambda record: (record.terminal_at or 0.0, record.job_id))
        overflow = max(0, len(retained) - self._max_terminal_jobs)
        expired.update(record.job_id for record in retained[:overflow])
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._subscribers.pop(job_id, None)

    def _run_loop(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                job_id = self._queue.popleft()
                record = self._jobs[job_id]
                record.state = "running"
                record.started_at = _now_iso()
                record.message = "Starting"
                self._notify_locked(record)
            try:
                self._run_job(record)
            except Exception as e:  # a bug in the runner must not kill the worker thread
                logger.error(f"Job {record.job_id} runner crashed: {e}", exc_info=True)
                with self._locked():
                    if record.state not in TERMINAL_STATES:
                        record.error = {"code": "INTERNAL", "message": str(e)}
                        self._finish_locked(record, "failed", str(e))

    def _child_env(self) -> dict[str, str]:
        """The child's environment, always carrying this engine's pid.

        ``shutdown()`` only runs when the engine exits *gracefully*. An engine
        that is SIGKILLed (crash, OOM killer, ``kill -9``) runs no cleanup at
        all, and the job child — a full ``hawavoclean process`` run — used to
        be reparented to init and go on to write a complete master and report
        for a run the UI had already reconciled to "failed, nothing was
        written". So the child watches the engine itself:
        ``hawavoclean.watchdog``.
        """
        return child_env(self._env)

    def _run_job(self, record: JobRecord) -> None:
        cmd = self._command_factory(record)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=self._child_env(),
            )
        except OSError as e:
            with self._locked():
                record.error = {"code": "SPAWN_FAILED", "message": str(e)}
                self._finish_locked(record, "failed", f"Could not start worker: {e}")
            return

        with self._lock:
            record.process = proc
            cancel_now = record.cancel_requested
        if cancel_now:
            self._terminate(proc)

        assert proc.stderr is not None and proc.stdout is not None
        stderr_stream = proc.stderr

        def _drain_stderr() -> None:
            for line in stderr_stream:
                stripped = line.rstrip("\n")
                if stripped.strip():
                    record.stderr_tail.append(stripped)

        stderr_thread = threading.Thread(
            target=_drain_stderr, name="hawavoclean-job-stderr", daemon=True
        )
        stderr_thread.start()

        done_event: dict[str, Any] | None = None
        error_event: dict[str, Any] | None = None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                logger.warning(f"Job {record.job_id}: non-JSON line on stdout: {line[:200]}")
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            if kind == "progress":
                with self._lock:
                    record.stage = str(event.get("stage", record.stage))
                    record.progress = float(event.get("progress", record.progress))
                    record.message = str(event.get("message", record.message))
                    unit = event.get("unit")
                    if isinstance(unit, dict) and "index" in unit and "total" in unit:
                        record.unit = {"index": int(unit["index"]), "total": int(unit["total"])}
                    else:
                        record.unit = None
                    self._notify_locked(record)
            elif kind == "done":
                done_event = event
            elif kind == "error":
                error_event = event

        returncode = proc.wait()
        stderr_thread.join(timeout=2.0)

        with self._locked():
            if record.state in TERMINAL_STATES:  # pragma: no cover - defensive
                return
            if done_event is not None and returncode == 0:
                report_path = Path(str(done_event.get("report_path") or record.report_path))
                try:
                    record.report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    record.error = {
                        "code": ExitCode.PUBLICATION_FAILURE.name,
                        "message": f"Published, but the report is unreadable: {e}",
                    }
                    self._finish_locked(record, "failed", record.error["message"])
                    return
                record.unit = None
                self._finish_locked(record, "done", "Done")
                return
            if record.cancel_requested:
                self._finish_locked(record, "cancelled", "Cancelled")
                return
            code, message = self._failure_details(record, returncode, error_event)
            record.error = {"code": code, "message": message}
            self._finish_locked(record, "failed", message)

    @staticmethod
    def _failure_details(
        record: JobRecord, returncode: int, error_event: dict[str, Any] | None
    ) -> tuple[str, str]:
        if error_event is not None:
            code = str(error_event.get("code") or "ERROR")
            message = str(error_event.get("message") or "Processing failed")
            return code, message
        try:
            code = ExitCode(returncode).name
        except ValueError:
            code = f"EXIT_{returncode}" if returncode >= 0 else f"SIGNAL_{-returncode}"
        if returncode == 0:
            code = "PROTOCOL_ERROR"
            return code, "Worker exited without reporting a result"
        try:
            tail = [ln for ln in record.stderr_tail if ln.strip()]
        except RuntimeError:  # pragma: no cover - drain thread still appending (an
            tail = []  # orphaned grandchild that inherited stderr can outlive the child)
        message = tail[-1][-300:] if tail else f"Worker exited with code {returncode}"
        return code, message
