"""Isolated subprocess enhancement worker with deadline enforcement and crash recovery.

The parent enforces a hard timeout on every request and restarts the worker
on crash or hang. There is no heartbeat protocol: liveness is inferred from
response deadlines only.

Speech units are independent by construction — that is what the segmentation
architecture buys — so :class:`EnhancementWorkerPool` runs several of these
workers at once. Scheduling is deliberately invisible in the result:
candidates are reassembled by unit index, never by completion order, so the
pool cannot influence a decision, the report, or a single output sample.
"""

import atexit
import contextlib
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer
from hawavoclean.errors import WorkerCrashError, WorkerTimeoutError
from hawavoclean.logging import get_logger
from hawavoclean.runtime import worker_pool_size
from hawavoclean.watchdog import parent_is_alive

logger = get_logger("worker")

#: How often an orphaned worker checks whether its parent is still there.
WATCHDOG_POLL_S = 0.5

#: Default grace given to ``STOP`` before SIGTERM, in seconds.
#:
#: **Zero, and the measurement is the reason.** ``close()`` used to put
#: ``STOP`` on the queue and terminate in the very next statement, so the
#: message was unreachable in production. Giving it a real chance was tried
#: and timed, 5 repetitions per core, on a warm worker that had just answered
#: a request:
#:
#: ===================  ==================  ==================
#: teardown path        Wiener worker       DFN3 studio worker
#: ===================  ==================  ==================
#: ``STOP`` + join      43.9 ms             182.2 ms
#: straight to SIGTERM  **2.6 ms**          **8.0 ms**
#: ===================  ==================  ==================
#:
#: A clean interpreter exit has to unwind the model, free every tensor and
#: run atexit; a signal does not. And there is nothing in a worker that needs
#: unwinding — it owns a model and two queues, all three of which die with the
#: process, which is the same reason its own orphan watchdog calls
#: ``os._exit``. So the fast path is the default and ``STOP`` is no longer
#: sent on it; ``close(grace_s=...)`` still asks politely for callers who want
#: that, and :func:`_worker_process_entry` still honours the message.
STOP_GRACE_S = 0.0

#: Fraction of physical RAM the enhancement pool may occupy. The studio core
#: loads DeepFilterNet3 into every worker, so N is capped by memory as well as
#: by cores; see :func:`memory_worker_cap`.
POOL_MEMORY_FRACTION = 0.25

#: Assumed per-worker footprint when the real one cannot be measured.
#: Deliberately pessimistic: measured 500 MB RSS for a warm DFN3 worker and
#: 133 MB for a Wiener worker on the reference machine.
FALLBACK_WORKER_RSS_BYTES = 1024 * 1024 * 1024

#: Cores left for the parent (guard, finishing, mastering) and the OS.
RESERVED_CPUS = 2

#: Override for the pool size. ``runtime.num_threads`` is the configured hint,
#: but its default of 1 means "one thread", not "one worker", so an explicit
#: escape hatch is needed until the config carries a dedicated key.
POOL_SIZE_ENV = "HAWAVOCLEAN_ENHANCE_WORKERS"


def _request_kernel_parent_death_signal() -> bool:
    """Ask Linux to SIGKILL this process the instant its parent dies.

    The polling watchdog below is portable but has two gaps a kernel
    guarantee does not: the poll interval itself, and any state in which the
    watchdog thread cannot be scheduled. ``PR_SET_PDEATHSIG`` closes both —
    the kernel delivers the signal as part of the parent's exit, so there is
    no window at all.

    It is Linux-only and inherited across ``exec`` but cleared on ``fork``,
    which is exactly right for a spawned worker. Returns whether it armed, so
    the caller keeps the portable watchdog on every other platform.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _PR_SET_PDEATHSIG = 1
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            return False
    except Exception:  # pragma: no cover - non-glibc or restricted sandbox
        return False
    # The parent can already have died between its fork and this call, which
    # would leave the signal armed against a parent that will never exit.
    if os.getppid() == 1:
        os._exit(0)
    return True


def _arm_parent_death_watchdog(poll_s: float = WATCHDOG_POLL_S) -> threading.Thread:
    """Exit this process as soon as the process that spawned it is gone.

    There is nothing to unwind here — the worker holds a model and two
    queues, both of which die with it — so the answer is an immediate
    ``os._exit``. Liveness is ``hawavoclean.watchdog.parent_is_alive``: pid
    probe *and* ``getppid()``, because a recycled pid fools the first and a
    platform that keeps the old ppid after reparenting fools the second.
    """
    parent_pid = os.getppid()

    def _watch_parent() -> None:
        while parent_is_alive(parent_pid):
            threading.Event().wait(poll_s)
        os._exit(0)

    thread = threading.Thread(target=_watch_parent, name="hawavoclean-worker-watchdog", daemon=True)
    thread.start()
    return thread


def _worker_process_entry(
    req_queue: mp.Queue,  # type: ignore[type-arg]
    resp_queue: mp.Queue,  # type: ignore[type-arg]
    enhancer_class: Any,
    core_id: str,
    sample_rate: int,
    phase_coherent: bool = True,
) -> None:
    """Entry point for the isolated worker subprocess."""
    # Parent-death watchdog FIRST, before the model exists. A parent killed
    # by SIGTERM/SIGKILL never runs its finally: blocks, and daemon=True only
    # covers a normal interpreter exit, so the child must notice on its own —
    # even mid-inference. This used to be armed after warmup, which left the
    # whole of import + construction + model load (seconds, and the most
    # likely moment for a user to give up and kill the parent) unwatched.
    #
    # Only when we really are a spawned child: `_worker_process_entry` is
    # also called in-process by the unit tests, and a watchdog thread there
    # would be watching the *test runner's* parent with an os._exit(0) on the
    # end of it.
    if mp.parent_process() is not None:
        # Both, deliberately. The kernel signal is immediate and immune to a
        # blocked interpreter but exists only on Linux; the polling thread
        # covers every other platform. A SIGKILLed parent runs no atexit
        # handler, so multiprocessing's daemon=True reaping never happens and
        # the worker is on its own — measured on Linux CI, where workers
        # outlived a SIGKILLed parent by more than thirty seconds while the
        # same test passed on macOS.
        _request_kernel_parent_death_signal()
        _arm_parent_death_watchdog()

    try:
        try:
            enhancer: Enhancer = enhancer_class(
                core_id=core_id, sample_rate=sample_rate, phase_coherent=phase_coherent
            )
        except TypeError:
            enhancer = enhancer_class()
        enhancer.warmup()
    except Exception as e:
        resp_queue.put({"type": "INIT_ERROR", "error": str(e)})
        return

    resp_queue.put({"type": "READY"})

    while True:
        try:
            msg = req_queue.get()
            if msg is None or msg.get("type") == "STOP":
                break

            if msg.get("type") == "ENHANCE":
                audio_bytes = msg["audio_bytes"]
                sr = msg["sample_rate"]
                arr = np.frombuffer(audio_bytes, dtype=np.float32)

                res = enhancer.enhance(arr, sr)
                resp_queue.put(
                    {
                        "type": "RESULT",
                        "audio_bytes": res.waveform.tobytes(),
                        "sample_rate": res.sample_rate,
                        "runtime_ms": res.model_runtime_ms,
                        "input_samples": res.input_samples,
                        "output_samples": res.output_samples,
                        "warnings": res.warnings,
                    }
                )
        except Exception as e:
            resp_queue.put({"type": "ERROR", "error": str(e)})


class IsolatedEnhancementWorker:
    """Parent controller managing the enhancement worker subprocess lifecycle."""

    def __init__(
        self,
        core_id: str = "wiener-dd-48k-v1",
        sample_rate: int = 48000,
        timeout_s: float = 120.0,
        enhancer_class: type[Enhancer] = WienerSpectralEnhancer,
        phase_coherent: bool = True,
    ) -> None:
        self.core_id = core_id
        self.sample_rate = sample_rate
        self.phase_coherent = phase_coherent
        self.timeout_s = timeout_s
        self.enhancer_class = enhancer_class
        self.process: Any = None
        self.req_queue: mp.Queue | None = None  # type: ignore[type-arg]
        self.resp_queue: mp.Queue | None = None  # type: ignore[type-arg]
        self._start_worker()

    def _start_worker(self) -> None:
        """Spawn worker subprocess and wait for READY signal."""
        ctx = mp.get_context("spawn")
        self.req_queue = ctx.Queue()
        self.resp_queue = ctx.Queue()

        self.process = ctx.Process(
            target=_worker_process_entry,
            args=(
                self.req_queue,
                self.resp_queue,
                self.enhancer_class,
                self.core_id,
                self.sample_rate,
                self.phase_coherent,
            ),
            daemon=True,
        )
        self.process.start()

        # Wait for READY signal with timeout
        try:
            msg = self.resp_queue.get(timeout=30.0)
            if msg.get("type") == "INIT_ERROR":
                raise RuntimeError(f"Worker init error: {msg.get('error')}")
            if msg.get("type") != "READY":
                raise RuntimeError(f"Unexpected worker startup message: {msg}")
        except queue.Empty as e:
            self._kill_worker()
            raise WorkerCrashError(
                "Failed to start isolated enhancement worker: no READY within 30s "
                "(model load too slow, or the worker crashed during init)"
            ) from e
        except Exception as e:
            self._kill_worker()
            raise WorkerCrashError(f"Failed to start isolated enhancement worker: {e}") from e

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        """Send audio to isolated worker with strict deadline and restart on failure."""
        if self.process is None or not self.process.is_alive():
            self._start_worker()

        assert self.req_queue is not None
        assert self.resp_queue is not None

        try:
            self.req_queue.put(
                {
                    "type": "ENHANCE",
                    "audio_bytes": waveform.astype(np.float32).tobytes(),
                    "sample_rate": sample_rate,
                }
            )

            # Poll in short slices so a child that DIES is noticed within a
            # fraction of a second instead of after the full deadline (a
            # segfaulting core would otherwise cost timeout_s per unit and be
            # misreported as a timeout).
            deadline = time.monotonic() + self.timeout_s
            msg: dict[str, Any] | None = None
            while True:
                try:
                    msg = self.resp_queue.get(timeout=0.25)
                    break
                except queue.Empty:
                    if not self.process.is_alive():
                        raise WorkerCrashError(
                            f"Enhancement worker process died (exit code "
                            f"{self.process.exitcode}) while handling a request"
                        ) from None
                    if time.monotonic() >= deadline:
                        raise queue.Empty from None

            if msg.get("type") == "RESULT":
                out_arr = np.frombuffer(msg["audio_bytes"], dtype=np.float32).copy()
                return EnhancementResult(
                    waveform=out_arr,
                    sample_rate=int(msg["sample_rate"]),
                    model_runtime_ms=float(msg["runtime_ms"]),
                    input_samples=int(msg["input_samples"]),
                    output_samples=int(msg["output_samples"]),
                    warnings=list(msg.get("warnings", [])),
                )
            elif msg.get("type") == "ERROR":
                raise WorkerCrashError(
                    f"Enhancement worker raised internal error: {msg.get('error')}"
                )
            else:
                raise WorkerCrashError(f"Unknown message type received from worker: {msg}")

        except queue.Empty as e:
            self._kill_worker()
            raise WorkerTimeoutError(f"Worker timed out after {self.timeout_s}s") from e
        except Exception as e:
            self._kill_worker()
            raise WorkerCrashError(f"Worker communication failure: {e}") from e

    def _kill_worker(self) -> None:
        """Terminate a hung/crashed worker and release its queues.

        The request queue's feeder thread can be blocked forever writing a
        multi-megabyte payload into a pipe nobody reads (child died before
        draining). Without cancel_join_thread() the interpreter joins that
        thread at exit and HANGS after printing 'finished successfully'.
        """
        if self.process is not None:
            try:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2.0)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join(timeout=2.0)
            except Exception:
                pass
            self.process = None
        for q in (self.req_queue, self.resp_queue):
            if q is not None:
                with contextlib.suppress(Exception):
                    q.cancel_join_thread()
                with contextlib.suppress(Exception):
                    q.close()
        self.req_queue = None
        self.resp_queue = None

    def pid(self) -> int | None:
        """Pid of the live worker process, or ``None``."""
        if self.process is None:
            return None
        pid = self.process.pid
        return int(pid) if pid else None

    def close(self, grace_s: float = STOP_GRACE_S) -> None:
        """Stop the worker. With ``grace_s > 0``, ask before signalling.

        ``STOP`` was dead code before this: it was put on the queue and
        ``_kill_worker`` ran in the very next statement, so no child ever read
        it. It is now either sent *and waited for*, or not sent at all — the
        pretence is gone either way. See :data:`STOP_GRACE_S` for why the
        default is the signal: politeness measured 5-22x slower, and this
        worker has nothing to unwind.
        """
        proc = self.process
        if grace_s > 0.0 and self.req_queue is not None and proc is not None and proc.is_alive():
            stopped = False
            with contextlib.suppress(Exception):
                self.req_queue.put({"type": "STOP"})
                proc.join(timeout=grace_s)
                stopped = not proc.is_alive()
            if not stopped:
                logger.debug("Worker did not stop within %.2fs; escalating to SIGTERM", grace_s)
        self._kill_worker()


def _process_rss_bytes(pid: int) -> int | None:
    """Resident set size of ``pid`` in bytes, or ``None`` if unreadable.

    ``ps`` rather than a dependency: this runs once per pool, and the number
    only has to be good enough to divide a memory budget by.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover - ps missing or unkillable
        return None
    if not out:
        return None
    try:
        return int(out.split()[0]) * 1024
    except ValueError:  # pragma: no cover - unexpected ps output
        return None


def physical_memory_bytes() -> int | None:
    """Total physical RAM, or ``None`` where the platform will not say."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
        return None


def cpu_worker_cap() -> int:
    """Workers the CPU allows, leaving :data:`RESERVED_CPUS` for the parent."""
    return max(1, (os.cpu_count() or (RESERVED_CPUS + 1)) - RESERVED_CPUS)


def memory_worker_cap(per_worker_bytes: int | None) -> int:
    """Workers that fit in :data:`POOL_MEMORY_FRACTION` of physical RAM.

    ``per_worker_bytes`` is the measured RSS of a warm worker. RSS counts
    shared pages in every process that maps them, so the marginal cost of
    worker N is lower than this — the overestimate is the safety margin.
    """
    per = per_worker_bytes or FALLBACK_WORKER_RSS_BYTES
    per = max(per, 1)
    total = physical_memory_bytes()
    if total is None:  # pragma: no cover - non-POSIX
        return 1
    return max(1, int((total * POOL_MEMORY_FRACTION) // per))


def configured_worker_hint(num_threads: int) -> int:
    """How many workers the operator has asked for, before the memory cap.

    ``runtime.num_threads`` is the configured pool size and is honoured
    through :func:`hawavoclean.runtime.worker_pool_size`, the one place that
    arithmetic lives — an explicit number is the operator's, not a suggestion.

    Its default is **1**, and 1 cannot be read literally here. That value is
    hashed into ``config_hash``, which is hashed into the job id, which seeds
    the master's deterministic dither, so raising it in a shipped profile
    would move published samples and reissue the committed reference hashes.
    The default therefore means "unset", and an unset pool is sized by the
    machine: :func:`cpu_worker_cap`, then the memory cap inside the pool.
    :data:`POOL_SIZE_ENV` overrides both, which is also how a run is pinned to
    one worker for an A/B.
    """
    raw = os.environ.get(POOL_SIZE_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("Ignoring %s=%r: not an integer", POOL_SIZE_ENV, raw)
    if num_threads > 1:
        return worker_pool_size(num_threads)
    return cpu_worker_cap()


#: How a pool builds one worker. Only the pipeline's chaos tests substitute
#: anything else; production always builds :class:`IsolatedEnhancementWorker`.
WorkerFactory = Callable[..., "IsolatedEnhancementWorker"]


@dataclass(frozen=True)
class WorkerSpec:
    """Everything that decides what a worker process *is*.

    Two pools with equal specs are interchangeable, which is what lets a
    batch keep one warm pool across files.
    """

    core_id: str
    sample_rate: int
    timeout_s: float
    enhancer_class: type[Enhancer]
    phase_coherent: bool


class EnhancementWorkerPool:
    """Several isolated workers, one deterministic result list.

    Contract:

    * ``submit`` takes the units in unit order; ``result(i)`` blocks for unit
      ``i``. Workers pull FIFO, so unit 1 is always started first and the
      caller can guard unit *i* while the pool is still enhancing *i+1* —
      but the *answers* are indexed, so completion order is unobservable.
    * A worker that crashes or times out fails only the unit it was holding:
      the failure is returned as the result for that unit (the pipeline turns
      it into original-audio passthrough) and the worker is replaced before
      the next unit, exactly as the single worker already did.
    * The strength ladder never comes back here. It is a local blend of the
      candidate the core already returned (``policy/strength.py``), so a
      guard failure costs no worker time at all.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        max_size: int,
        prewarm: int = 1,
        worker_factory: "WorkerFactory | None" = None,
        long_lived: bool = False,
    ) -> None:
        self.spec = spec
        self.worker_factory: WorkerFactory = worker_factory or IsolatedEnhancementWorker
        self._lock = threading.Lock()
        self._slot_locks: list[threading.Lock] = []
        self._workers: list[IsolatedEnhancementWorker | None] = []
        self._closed = False
        #: Set once the first worker is up and its footprint is known.
        self.measured_worker_rss_bytes: int | None = None
        self._configured_max = max(1, max_size)

        # Before any worker exists there is nothing to measure, so the first
        # ceiling is drawn with the deliberately pessimistic
        # FALLBACK_WORKER_RSS_BYTES. Starting that many at once therefore
        # cannot exhaust RAM whatever the core turns out to weigh.
        self._max_size = min(self._configured_max, memory_worker_cap(None))
        eager = min(self._max_size, max(1, prewarm))

        # Slots 1.. are started NOW, alongside slot 0, not after it. Worker 0
        # is a serial ~1 s model load on the studio core, and making the rest
        # queue behind it cost the whole pool a second of nothing (measured:
        # slot 0 ready at 1.02 s, slots 1-4 at 2.05 s).
        self._workers = [None] * eager
        warmers = [
            threading.Thread(
                target=self._prewarm_slot,
                args=(slot,),
                name=f"hawavoclean-warm-{slot}",
                daemon=True,
            )
            for slot in range(1, eager)
        ]
        for t in warmers:
            t.start()

        # Slot 0 stays synchronous: a core that cannot load has always failed
        # the job here, with this error, and it must keep doing so.
        try:
            first = self._new_worker()
        except BaseException:
            for t in warmers:
                t.join(timeout=20.0)
            self.close()
            raise
        with self._lock:
            self._workers[0] = first

        # Worth a `ps` when the pessimistic ceiling actually held us back, or
        # when this pool will outlive the file that opened it (a batch), where
        # a later file may want more workers than this one did. Otherwise
        # every slot that will ever be asked for is already running and there
        # is nothing left for a measurement to decide.
        if eager < min(self._configured_max, max(1, prewarm)) or long_lived:
            pid_of = getattr(first, "pid", None)
            pid = pid_of() if callable(pid_of) else None
            if pid is not None:
                self.measured_worker_rss_bytes = _process_rss_bytes(pid)
        # Now that one worker's real footprint is known, the ceiling for any
        # slot not already started is redrawn from the measurement.
        self._max_size = min(
            self._configured_max, memory_worker_cap(self.measured_worker_rss_bytes)
        )
        self._max_size = max(self._max_size, eager)

    def _prewarm_slot(self, slot: int) -> None:
        try:
            self.worker_at(slot)
        except Exception as e:
            logger.warning("Enhancement worker slot %d did not warm: %s", slot, e)

    @property
    def max_size(self) -> int:
        """Ceiling on live workers, after the memory cap was applied."""
        return self._max_size

    @property
    def configured_max(self) -> int:
        """Ceiling asked for, before any memory cap. This is what decides
        whether a cached pool is still the right pool: the memory cap is a
        property of the machine, identical for both, and re-deriving a pool to
        get the same number back would defeat the point of caching it."""
        return self._configured_max

    @property
    def live_workers(self) -> int:
        """Workers that have actually been started."""
        with self._lock:
            return sum(1 for w in self._workers if w is not None)

    def _new_worker(self) -> IsolatedEnhancementWorker:
        return self.worker_factory(
            core_id=self.spec.core_id,
            sample_rate=self.spec.sample_rate,
            timeout_s=self.spec.timeout_s,
            enhancer_class=self.spec.enhancer_class,
            phase_coherent=self.spec.phase_coherent,
        )

    def worker_at(self, slot: int) -> IsolatedEnhancementWorker:
        """The worker staffing ``slot``, started on first use."""
        if self._closed:
            raise WorkerCrashError("Enhancement worker pool is closed")
        with self._slot_lock(slot):
            with self._lock:
                existing = self._workers[slot] if slot < len(self._workers) else None
            if existing is not None:
                return existing
            built = self._new_worker()
            with self._lock:
                while len(self._workers) <= slot:
                    self._workers.append(None)
                self._workers[slot] = built
            return built

    def _slot_lock(self, slot: int) -> threading.Lock:
        """One lock per slot, so a prewarm thread and a dispatch thread can
        never build two processes for the same slot and orphan one."""
        with self._lock:
            while len(self._slot_locks) <= slot:
                self._slot_locks.append(threading.Lock())
            return self._slot_locks[slot]

    def begin(
        self,
        items: list[tuple[np.ndarray[Any, np.dtype[np.float32]], int]],
    ) -> "EnhancementRun":
        """Dispatch every ``(waveform, sample_rate)`` item and return a handle.

        The handle is lazy on purpose: the caller asks for unit ``i`` when it
        is ready to guard unit ``i``, and the pool keeps enhancing ``i+1..N``
        meanwhile. Guarding and enhancing therefore overlap instead of
        queueing behind each other.
        """
        run = EnhancementRun(self, items)
        run.start()
        return run

    def map_enhance(
        self,
        items: list[tuple[np.ndarray[Any, np.dtype[np.float32]], int]],
    ) -> list["UnitEnhancement"]:
        """Enhance every ``(waveform, sample_rate)`` item, answers in input order."""
        run = self.begin(items)
        try:
            return [run.result(i) for i in range(len(items))]
        finally:
            run.join()

    def close(self) -> None:
        """Stop every worker. Idempotent."""
        with self._lock:
            workers = [w for w in self._workers if w is not None]
            self._workers = []
            self._closed = True
        for w in workers:
            with contextlib.suppress(Exception):
                w.close()


@dataclass
class UnitEnhancement:
    """One unit's answer from the pool: a candidate, or the failure instead."""

    result: EnhancementResult | None
    error: BaseException | None
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.result is not None


class EnhancementRun:
    """One dispatch round over a pool. Threads here only wait on IPC."""

    def __init__(
        self,
        pool: EnhancementWorkerPool,
        items: list[tuple[np.ndarray[Any, np.dtype[np.float32]], int]],
    ) -> None:
        self._pool = pool
        self._items = items
        self._pending: queue.Queue[int] = queue.Queue()
        for i in range(len(items)):
            self._pending.put(i)
        self._results: list[UnitEnhancement | None] = [None] * len(items)
        self._events = [threading.Event() for _ in items]
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        n = min(self._pool.max_size, len(self._items))
        if n <= 1:
            # One worker means one order, and that order is the caller's. Do
            # the work in ``result``, in the caller's thread, exactly where the
            # single worker always did it — no thread, no handoff, and no
            # machinery charged to a file that cannot use it.
            return
        for slot in range(n):
            t = threading.Thread(
                target=self._drain, args=(slot,), name=f"hawavoclean-enh-{slot}", daemon=True
            )
            t.start()
            self._threads.append(t)

    def _run_one(self, worker: IsolatedEnhancementWorker, idx: int) -> None:
        t0 = time.perf_counter()
        try:
            waveform, sr = self._items[idx]
            res = worker.enhance(waveform, sr)
            out = UnitEnhancement(res, None, (time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            # Fail closed, per unit: the worker is already torn down by
            # ``enhance`` and will be replaced on its next request, so this
            # never costs more than the one unit it was holding.
            out = UnitEnhancement(None, e, (time.perf_counter() - t0) * 1000.0)
        self._results[idx] = out
        self._events[idx].set()

    def _drain(self, slot: int) -> None:
        worker: IsolatedEnhancementWorker | None = None
        while True:
            try:
                idx = self._pending.get_nowait()
            except queue.Empty:
                return
            if worker is None:
                try:
                    # Slot 0 already exists; the rest are built here, so their
                    # model load overlaps the work slot 0 is already doing.
                    worker = self._pool.worker_at(slot)
                except Exception as e:
                    # This slot cannot be staffed at all (model load failed,
                    # memory refused). Hand the unit back — a slot that IS
                    # staffed will take it — and stand this one down rather
                    # than fail a unit that nothing was wrong with.
                    logger.warning("Enhancement worker slot %d could not start: %s", slot, e)
                    self._pending.put(idx)
                    return
            self._run_one(worker, idx)

    def _rescue(self, idx: int) -> None:
        """Every thread stood down with work left. Finish it here, in the
        caller's thread, on the worker that is known to exist."""
        while not self._events[idx].is_set():
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                pending = idx
            try:
                worker = self._pool.worker_at(0)
            except Exception as e:
                self._results[pending] = UnitEnhancement(None, e, 0.0)
                self._events[pending].set()
                if pending == idx:
                    return
                continue
            self._run_one(worker, pending)

    def result(self, idx: int) -> UnitEnhancement:
        """Block for unit ``idx``. Never blocks forever: if every thread has
        stood down with this unit unanswered, it is finished here instead."""
        event = self._events[idx]
        if not self._threads:
            if not event.is_set():
                self._pending_drop(idx)
                self._run_one(self._pool.worker_at(0), idx)
            out_inline = self._results[idx]
            assert out_inline is not None
            return out_inline
        while not event.wait(0.25):
            if not any(t.is_alive() for t in self._threads):
                self._rescue(idx)
                break
        out = self._results[idx]
        assert out is not None
        return out

    def _pending_drop(self, idx: int) -> None:
        """Take ``idx`` out of the pending queue (single-worker inline path)."""
        left: list[int] = []
        while True:
            try:
                got = self._pending.get_nowait()
            except queue.Empty:
                break
            if got != idx:
                left.append(got)
        for i in left:
            self._pending.put(i)

    def join(self) -> None:
        for t in self._threads:
            t.join(timeout=1.0)


# --------------------------------------------------------------------------
# Warm-pool reuse across files (batch)
#
# A batch used to pay a full interpreter start plus a model load for every
# file, because per-file isolation was implemented as "one child process per
# file". Isolation is worth keeping; reloading the model is not. The cache
# below lets ONE process keep one warm pool across files, and it is opt-in:
# a plain `process` run never enables it, so its lifecycle is byte-for-byte
# the behaviour it always had.
# --------------------------------------------------------------------------

_cache_lock = threading.Lock()
_pool_reuse_enabled = False
_cached_pool: EnhancementWorkerPool | None = None
_cached_key: tuple[WorkerSpec, object] | None = None


def set_pool_reuse(enabled: bool) -> None:
    """Turn warm-pool reuse on or off for this process.

    Turning it off closes whatever is cached, so a caller can hand the
    workers back at any point without knowing whether it opened them.
    """
    global _pool_reuse_enabled
    with _cache_lock:
        _pool_reuse_enabled = enabled
    if not enabled:
        shutdown_pool_cache()


def pool_reuse_enabled() -> bool:
    """Whether this process keeps its enhancement pool between jobs."""
    with _cache_lock:
        return _pool_reuse_enabled


def acquire_pool(
    spec: WorkerSpec,
    max_size: int,
    prewarm: int = 1,
    worker_factory: WorkerFactory | None = None,
) -> EnhancementWorkerPool:
    """A pool for ``spec`` — the cached one when it matches, else a new one.

    A cached pool built for a different core, sample rate, deadline or phase
    setting is not interchangeable, so it is closed rather than reused: the
    point of reuse is to skip a *redundant* model load, never to run a file
    on the wrong core.
    """
    global _cached_pool, _cached_key
    key = (spec, worker_factory or IsolatedEnhancementWorker)
    # Nothing slow happens under this lock. Building a pool starts processes
    # and loads a model; doing that while holding a module-global lock is how
    # a cache turns into a stall (and, when the constructor wants to read the
    # same flag, into a deadlock).
    with _cache_lock:
        reuse = _pool_reuse_enabled
        cached, cached_key = _cached_pool, _cached_key
        if reuse and cached is not None and cached_key == key:
            if cached.configured_max >= max_size:
                return cached
        else:
            cached = None
        if reuse:
            _cached_pool, _cached_key = None, None
    if cached is not None:
        with contextlib.suppress(Exception):
            cached.close()
    pool = EnhancementWorkerPool(spec, max_size, prewarm, worker_factory, long_lived=reuse)
    if reuse:
        with _cache_lock:
            if _pool_reuse_enabled and _cached_pool is None:
                _cached_pool, _cached_key = pool, key
    return pool


def release_pool(pool: EnhancementWorkerPool | None) -> None:
    """Give a pool back. Cached pools stay warm; everything else is closed."""
    if pool is None:
        return
    with _cache_lock:
        keep = _pool_reuse_enabled and pool is _cached_pool
    if not keep:
        pool.close()


def shutdown_pool_cache() -> None:
    """Close the cached pool, if any. Safe to call more than once."""
    global _cached_pool, _cached_key
    with _cache_lock:
        pool, _cached_pool, _cached_key = _cached_pool, None, None
    if pool is not None:
        with contextlib.suppress(Exception):
            pool.close()


atexit.register(shutdown_pool_cache)
