"""A killed parent must not leave work running behind it.

Two orphan paths, both proved by killing a real parent with SIGKILL — the
signal that runs no cleanup anywhere, so only the child's own watchdog can
answer:

1. The engine (``hawavoclean serve`` / :class:`JobManager`) and its job child,
   a full ``hawavoclean process`` run. Before the fix this child was
   reparented to init and **ran to completion**, publishing a master and a
   report for a job the UI had already reconciled to "failed … nothing was
   written". Measured on this machine, same fixture, same moment (killed at
   "Enhancing unit 1/5"): the orphan lived 3.33 s longer and left
   ``o.wav`` + ``o.hawavoclean.json`` + ``o.hawavoclean.txt`` at the
   destination. With the watchdog: gone in 0.20 s, destination empty.

2. The pipeline and its enhancement worker, killed *during model warmup* —
   the window the worker's own watchdog used to leave unarmed, because it
   armed itself only after ``enhancer.warmup()`` returned. A warmup that
   never returns turns that window into forever.
"""

import contextlib
import json
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.chaos.procwatch import (
    alive,
    children_of,
    contents,
    descendants,
    freeze,
    kill_tree,
    thaw,
    wait_all_gone,
)

REPO = Path(__file__).resolve().parents[2]

#: A dead parent must be noticed and acted on well inside this. The watchdogs
#: poll at 0.25 s (job child) and 0.5 s (worker); the measured answers are
#: 0.20 s and under a second. The bound only keeps machine load from deciding.
ORPHAN_EXIT_TIMEOUT_S = 30.0
JOB_START_TIMEOUT_S = 180.0
MAX_ATTEMPTS = 3


@pytest.fixture(scope="module")
def long_input(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("orphan") / "long.wav"
    sr = 48000
    t = np.arange(sr * 90) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    sf.write(str(p), x, sr)
    return p


_ENGINE = """
    import json, sys, time
    from pathlib import Path
    from hawavoclean.server.jobs import JobManager

    manager = JobManager()
    snap = manager.submit(
        input_path=Path(sys.argv[1]),
        output_path=Path(sys.argv[2]),
        profile="production",
        overwrite=True,
    )
    job_id = snap["job_id"]
    while True:
        status = manager.get_status(job_id) or {}
        sys.stdout.write(json.dumps({k: status.get(k) for k in ("state", "stage")}) + "\\n")
        sys.stdout.flush()
        time.sleep(0.05)
"""


def _run_engine(script: Path, input_path: Path, output_path: Path) -> "subprocess.Popen[str]":
    return subprocess.Popen(
        [sys.executable, str(script), str(input_path), str(output_path)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _wait_until_enhancing(engine: "subprocess.Popen[str]") -> None:
    assert engine.stdout is not None
    deadline = time.time() + JOB_START_TIMEOUT_S
    while time.time() < deadline:
        line = engine.stdout.readline()
        if not line:
            raise AssertionError(f"engine exited before the job ran (rc={engine.poll()})")
        try:
            status = json.loads(line)
        except ValueError:  # pragma: no cover - the script prints only JSON
            continue
        if status.get("stage") in ("enhance", "guard"):
            return
        if status.get("state") in ("failed", "cancelled", "done"):
            raise AssertionError(f"job reached {status} before it could be interrupted")
    raise AssertionError("job never reached a unit")


@pytest.mark.chaos
def test_sigkilled_engine_leaves_no_job_child_writing_files(
    long_input: Path, tmp_path: Path
) -> None:
    """SIGKILL the engine mid-job: the child must die too, and publish nothing."""
    script = tmp_path / "engine_runner.py"
    # The engine subprocess inherits HAWAVOCLEAN_WORK_DIR from the session's
    # own isolated value (tests/conftest.py), so its scratch never escapes.
    script.write_text(textwrap.dedent(_ENGINE), encoding="utf-8")

    lost_races: list[list[str]] = []
    for attempt in range(MAX_ATTEMPTS):
        dest_dir = tmp_path / f"dest{attempt}"
        dest_dir.mkdir()
        engine = _run_engine(script, long_input, dest_dir / "o.wav")
        job_child: int | None = None
        try:
            _wait_until_enhancing(engine)
            kids = children_of(engine.pid)
            assert len(kids) == 1, f"expected exactly one job child, got {kids}"
            job_child = kids[0]

            # Freeze the job child first: from here it cannot publish, so
            # "the destination is empty" is a fact about the whole rest of
            # the test rather than a snapshot that may already be stale.
            freeze(job_child)
            if contents(dest_dir):
                lost_races.append(contents(dest_dir))
                continue
            worker_kids = children_of(job_child)
            assert worker_kids, "job child has no enhancement worker while enhancing"

            engine.kill()  # SIGKILL: no shutdown(), no finally, no cleanup
            engine.wait(timeout=30)
            thaw(job_child)

            survivors, waited = wait_all_gone([job_child, *worker_kids], ORPHAN_EXIT_TIMEOUT_S)
            assert not survivors, (
                f"the SIGKILLed engine left {survivors} running after {waited:.1f}s; "
                f"an orphaned job child finishes the run and writes files nobody "
                f"is waiting for"
            )
            assert not contents(dest_dir), (
                f"an orphaned job child published after its engine died: {contents(dest_dir)}"
            )
            return
        finally:
            # Signal only what is provably still ours: a reaped pid belongs
            # to the operating system again, and killing it by number would
            # be aimed at a stranger.
            if job_child is not None and alive(job_child):
                kill_tree(job_child)
            if engine.poll() is None:
                kill_tree(engine.pid)
            with contextlib.suppress(Exception):
                engine.wait(timeout=10)
            if engine.stdout is not None:
                engine.stdout.close()

    pytest.fail(
        f"the job published before it could be frozen in all {MAX_ATTEMPTS} attempts "
        f"({lost_races}); the orphan case was never exercised"
    )


_WARMUP_HANGS = """
    import multiprocessing as mp
    import sys
    import time
    from typing import Any

    MARKER = __MARKER__

    from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
    from hawavoclean.enhancement.worker import _worker_process_entry


    class WarmupHangs:
        \"\"\"An enhancer whose model load never finishes.\"\"\"

        def __init__(self, **_: Any) -> None:
            pass

        @property
        def metadata(self) -> EnhancerMetadata:
            return EnhancerMetadata(
                core_id="hangs", version="0", algorithm="hang",
                sample_rate=48000, phase_coherent=True,
            )

        def warmup(self) -> None:
            # Tell the test we are really inside warmup with no watchdog yet,
            # so the parent is killed at the moment under test rather than
            # while the spawn bootstrap is still unpickling.
            open(MARKER, "w").close()
            time.sleep(3600.0)

        def enhance(self, waveform: Any, sample_rate: int) -> EnhancementResult:
            raise RuntimeError("unreachable")


    if __name__ == "__main__":
        ctx = mp.get_context("spawn")
        # Hold the queues: temporaries are collected as soon as start()
        # returns, and the child then cannot rebuild their semaphores.
        req_queue, resp_queue = ctx.Queue(), ctx.Queue()
        proc = ctx.Process(
            target=_worker_process_entry,
            args=(req_queue, resp_queue, WarmupHangs, "hangs", 48000, True),
            daemon=True,
        )
        proc.start()
        sys.stdout.write(f"{proc.pid}\\n")
        sys.stdout.flush()
        time.sleep(3600.0)
"""


@pytest.mark.chaos
def test_worker_orphaned_during_warmup_exits_on_its_own(tmp_path: Path) -> None:
    """The worker's watchdog must be armed before the model is, not after.

    The worker is stuck in ``warmup()`` — the phase that used to run with no
    watchdog at all. Its parent is SIGKILLed. If the watchdog were still
    armed after warmup, this worker would sit in the process table for an
    hour holding whatever the model had allocated.
    """
    script = tmp_path / "warmup_hangs.py"
    marker = tmp_path / "in-warmup"
    script.write_text(
        textwrap.dedent(_WARMUP_HANGS).replace("__MARKER__", repr(str(marker))),
        encoding="utf-8",
    )
    parent = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    worker_pid: int | None = None
    try:
        assert parent.stdout is not None
        line = parent.stdout.readline()
        assert line.strip().isdigit(), f"parent never reported a worker pid: {line!r}"
        worker_pid = int(line.strip())

        deadline = time.time() + 60.0
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "the worker never reached warmup; nothing was tested"

        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=30)

        survivors, waited = wait_all_gone([worker_pid], ORPHAN_EXIT_TIMEOUT_S)
        assert not survivors, (
            f"the enhancement worker was still alive {waited:.1f}s after its parent "
            f"was SIGKILLed during warmup: {survivors}"
        )
    finally:
        if worker_pid is not None and alive(worker_pid):
            kill_tree(worker_pid)
        if parent.poll() is None:
            kill_tree(parent.pid)
        with contextlib.suppress(Exception):
            parent.wait(timeout=10)
        if parent.stdout is not None:
            parent.stdout.close()


@pytest.mark.chaos
def test_descendants_and_kill_tree_cover_two_levels(tmp_path: Path) -> None:
    """The helpers the two tests above rely on, exercised on a known tree."""
    script = tmp_path / "tree.py"
    script.write_text(
        textwrap.dedent(
            """
            import subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
            sys.stdout.write(f"{child.pid}\\n")
            sys.stdout.flush()
            time.sleep(3600)
            """
        ),
        encoding="utf-8",
    )
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    try:
        assert parent.stdout is not None
        grandchild = int(parent.stdout.readline().strip())
        assert grandchild in descendants(parent.pid)
        kill_tree(parent.pid)
        parent.wait(timeout=10)  # reap it: a zombie still answers kill(pid, 0)
        survivors, _ = wait_all_gone([grandchild], 10.0)
        assert not survivors, "kill_tree left the grandchild running"
    finally:
        if parent.poll() is None:
            kill_tree(parent.pid)
        with contextlib.suppress(Exception):
            parent.wait(timeout=10)
        if parent.stdout is not None:
            parent.stdout.close()
