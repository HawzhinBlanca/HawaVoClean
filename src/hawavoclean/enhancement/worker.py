"""Isolated subprocess enhancement worker with deadline enforcement and crash recovery.

The parent enforces a hard timeout on every request and restarts the worker
on crash or hang. There is no heartbeat protocol: liveness is inferred from
response deadlines only.
"""

import multiprocessing as mp
import os
import queue
import time
from typing import Any

import numpy as np

from hawavoclean.enhancement.production import WienerSpectralEnhancer
from hawavoclean.enhancement.protocol import EnhancementResult, Enhancer
from hawavoclean.errors import WorkerCrashError, WorkerTimeoutError


def _worker_process_entry(
    req_queue: mp.Queue,  # type: ignore[type-arg]
    resp_queue: mp.Queue,  # type: ignore[type-arg]
    enhancer_class: Any,
    core_id: str,
    sample_rate: int,
    phase_coherent: bool = True,
) -> None:
    """Entry point for the isolated worker subprocess."""
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

    # Parent-death watchdog on its own thread: a parent killed by
    # SIGTERM/SIGKILL never runs its finally: blocks and daemon=True only
    # covers normal interpreter exit, so the child must notice on its own —
    # even mid-inference. Liveness is checked with kill(ppid, 0), which is
    # reliable across platforms (getppid() can keep the old value after
    # reparenting on some systems).
    import threading

    parent_pid = os.getppid()

    def _watch_parent() -> None:
        while True:
            try:
                os.kill(parent_pid, 0)
            except (ProcessLookupError, PermissionError):
                os._exit(0)
            if os.getppid() != parent_pid:
                os._exit(0)
            threading.Event().wait(0.5)

    threading.Thread(target=_watch_parent, daemon=True).start()

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
            msg = self.resp_queue.get(timeout=15.0)
            if msg.get("type") == "INIT_ERROR":
                raise RuntimeError(f"Worker init error: {msg.get('error')}")
            if msg.get("type") != "READY":
                raise RuntimeError(f"Unexpected worker startup message: {msg}")
        except queue.Empty as e:
            self._kill_worker()
            raise WorkerCrashError(
                "Failed to start isolated enhancement worker: no READY within 15s "
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
                import contextlib

                with contextlib.suppress(Exception):
                    q.cancel_join_thread()
                with contextlib.suppress(Exception):
                    q.close()
        self.req_queue = None
        self.resp_queue = None

    def close(self) -> None:
        """Stop worker and clean up queues."""
        if self.req_queue is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self.req_queue.put({"type": "STOP"})
        self._kill_worker()
