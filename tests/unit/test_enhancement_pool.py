"""The enhancement worker pool: order, fail-closed, sizing, teardown, reuse.

The pool exists because speech units are independent, and it is only allowed
to exist because that independence is real. Every test here is about one of
the four things that could make it a lie:

1. **Order.** A candidate must belong to the unit that asked for it, whatever
   order the workers finish in.
2. **Blast radius.** A worker that dies must cost its own unit and nothing
   else, and the pool must replace it without failing the job.
3. **Size.** N must be bounded by cores, by memory and by the work there is.
4. **State.** Reusing a worker for a second unit, or a second file, must not
   change what that unit gets — which is a property of the cores, so it is
   measured on the cores themselves rather than assumed.
"""

import importlib.util
import os
import signal
import threading
import time
from typing import Any

import numpy as np
import pytest

from hawavoclean.enhancement.production import NoOpEnhancer, WienerSpectralEnhancer
from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.enhancement.worker import (
    FALLBACK_WORKER_RSS_BYTES,
    POOL_MEMORY_FRACTION,
    POOL_SIZE_ENV,
    EnhancementWorkerPool,
    IsolatedEnhancementWorker,
    WorkerSpec,
    acquire_pool,
    configured_worker_hint,
    cpu_worker_cap,
    memory_worker_cap,
    physical_memory_bytes,
    pool_reuse_enabled,
    release_pool,
    set_pool_reuse,
    shutdown_pool_cache,
)
from hawavoclean.errors import WorkerCrashError, WorkerError

SR = 48000


def _tagged(tag: int, n: int = 2400) -> np.ndarray[Any, np.dtype[np.float32]]:
    """A payload whose first sample names the unit that sent it."""
    w = np.full(n, 0.01, dtype=np.float32)
    w[0] = np.float32(tag)
    return w


class _ReverseOrderEnhancer:
    """Answers late for early units and early for late ones.

    The tag rides in sample 0, so the completion order is provably the
    reverse of the submission order and an answer that is merely "in some
    order" cannot pass.
    """

    def __init__(self, _core_id: str = "x", sample_rate: int = SR, **_: Any) -> None:
        self._meta = EnhancerMetadata("reverse", "0", "t", sample_rate, True)

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def enhance(self, waveform: Any, sample_rate: int) -> EnhancementResult:
        tag = int(waveform[0])
        time.sleep(max(0.0, 0.45 - 0.12 * tag))
        out = np.full(len(waveform), float(tag), dtype=np.float32)
        return EnhancementResult(out, sample_rate, 1.0, len(waveform), len(out))


class _DiesOnUnitOne:
    """SIGKILLs its own process for exactly one unit; identity for the rest."""

    def __init__(self, _core_id: str = "x", sample_rate: int = SR, **_: Any) -> None:
        self._meta = EnhancerMetadata("dies", "0", "t", sample_rate, True)

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def enhance(self, waveform: Any, sample_rate: int) -> EnhancementResult:
        if int(waveform[0]) == 1:
            os.kill(os.getpid(), signal.SIGKILL)
        out = np.array(waveform, dtype=np.float32, copy=True)
        return EnhancementResult(out, sample_rate, 1.0, len(waveform), len(out))


def _wait_live(pool: EnhancementWorkerPool, n: int, timeout_s: float = 30.0) -> int:
    """Prewarm is deliberately asynchronous — the constructor returns while the
    other workers are still loading — so a test that wants to see them has to
    wait for them, exactly as the dispatcher does."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and pool.live_workers < n:
        time.sleep(0.02)
    return pool.live_workers


def _spec(cls: type[Any], timeout_s: float = 30.0) -> WorkerSpec:
    return WorkerSpec(
        core_id="test",
        sample_rate=SR,
        timeout_s=timeout_s,
        enhancer_class=cls,
        phase_coherent=True,
    )


@pytest.mark.unit
def test_answers_are_indexed_by_unit_not_by_completion_order() -> None:
    """Four units answered in reverse order still land on their own indices."""
    pool = EnhancementWorkerPool(_spec(_ReverseOrderEnhancer), max_size=4, prewarm=4)
    try:
        # The pool sizes itself to the machine: `max_size` is a request, and the
        # memory cap trims it (a 16 GB CI runner staffs 3 where a 38 GB
        # workstation staffs 4). That trimming is the design, so requiring all
        # four here would assert the size of the host, not the behaviour under
        # test. Two are enough for completions to interleave, and the invariant
        # that matters holds at any size: an answer lands on its own unit's
        # index no matter which worker finished first.
        live = _wait_live(pool, 4)
        assert live >= 2, f"need at least two workers for completions to interleave, got {live}"
        items = [(_tagged(i), SR) for i in range(4)]
        out = pool.map_enhance(items)
        assert [o.ok for o in out] == [True, True, True, True]
        got = [int(o.result.waveform[0]) for o in out if o.result is not None]
        assert got == [0, 1, 2, 3], f"the pool reassembled by completion order, not by unit: {got}"
    finally:
        pool.close()


@pytest.mark.unit
def test_a_killed_worker_costs_its_own_unit_and_nothing_else() -> None:
    """One worker SIGKILLs itself mid-unit. That unit fails; the rest do not,
    and the pool staffs the empty slot again without being asked."""
    pool = EnhancementWorkerPool(_spec(_DiesOnUnitOne, timeout_s=20.0), max_size=3, prewarm=3)
    try:
        out = pool.map_enhance([(_tagged(i), SR) for i in range(5)])
        assert out[1].error is not None, "the unit whose worker died must report the failure"
        assert isinstance(out[1].error, WorkerError)
        assert out[1].result is None
        for i in (0, 2, 3, 4):
            survivor = out[i].result
            assert survivor is not None, f"unit {i} was collateral damage of unit 1's crash"
            assert int(survivor.waveform[0]) == i
        # And the pool is usable afterwards: the dead slot was replaced.
        again = pool.map_enhance([(_tagged(9), SR)])
        assert again[0].ok
    finally:
        pool.close()


@pytest.mark.unit
def test_one_worker_means_no_threads_and_the_callers_own_order() -> None:
    """A file with one speech unit must not pay for machinery it cannot use."""
    pool = EnhancementWorkerPool(_spec(NoOpEnhancer), max_size=1, prewarm=1)
    try:
        run = pool.begin([(_tagged(0), SR), (_tagged(1), SR)])
        assert run._threads == [], "a single-worker run must stay in the caller's thread"
        assert run.result(0).ok and run.result(1).ok
        run.join()
    finally:
        pool.close()


@pytest.mark.unit
def test_pool_size_is_bounded_by_cores_memory_and_work() -> None:
    cpu_cap = cpu_worker_cap()
    assert 1 <= cpu_cap <= (os.cpu_count() or 1)

    total = physical_memory_bytes()
    assert total is not None and total > 0
    # A worker that claims a quarter of RAM by itself leaves room for exactly
    # one; a tiny one is capped by cores, not by memory.
    assert memory_worker_cap(int(total * POOL_MEMORY_FRACTION)) == 1
    assert memory_worker_cap(int(total)) == 1
    assert memory_worker_cap(1024 * 1024) > cpu_cap
    # No measurement yet -> the pessimistic estimate, which is what makes an
    # unmeasured concurrent start safe.
    assert memory_worker_cap(None) == memory_worker_cap(FALLBACK_WORKER_RSS_BYTES)

    # And the pool never starts more workers than there is work: a ceiling of
    # 8 with two units to do is two processes, not eight.
    pool = EnhancementWorkerPool(_spec(NoOpEnhancer), max_size=8, prewarm=2)
    try:
        assert _wait_live(pool, 2) == 2
        pool.map_enhance([(_tagged(0), SR), (_tagged(1), SR)])
        assert pool.live_workers == 2
    finally:
        pool.close()


@pytest.mark.unit
def test_worker_hint_reads_config_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POOL_SIZE_ENV, raising=False)
    # num_threads defaults to 1 and means *threads*; taking it literally as a
    # worker count would switch the pool off for every shipped profile.
    assert configured_worker_hint(1) == cpu_worker_cap()
    assert configured_worker_hint(3) == 3
    monkeypatch.setenv(POOL_SIZE_ENV, "2")
    assert configured_worker_hint(1) == 2
    assert configured_worker_hint(9) == 2
    monkeypatch.setenv(POOL_SIZE_ENV, "not-a-number")
    assert configured_worker_hint(1) == cpu_worker_cap()


@pytest.mark.unit
def test_stop_is_honoured_and_the_signal_is_the_default() -> None:
    """``STOP`` used to be unreachable — put on the queue and then SIGTERMed
    in the next statement. Both paths are now real and distinguishable by the
    exit status the child leaves behind."""
    wk = IsolatedEnhancementWorker(enhancer_class=NoOpEnhancer, timeout_s=10.0)
    proc = wk.process
    wk.close(grace_s=10.0)
    assert proc.exitcode == 0, "STOP did not stop the worker on its own"

    wk2 = IsolatedEnhancementWorker(enhancer_class=NoOpEnhancer, timeout_s=10.0)
    proc2 = wk2.process
    wk2.close()  # default grace: straight to the signal
    assert proc2.exitcode == -signal.SIGTERM, "the default teardown is no longer the fast one"


@pytest.mark.unit
def test_pool_reuse_hands_back_the_same_processes_and_stops_when_told() -> None:
    spec = _spec(NoOpEnhancer)
    set_pool_reuse(True)
    try:
        assert pool_reuse_enabled()
        first = acquire_pool(spec, max_size=2, prewarm=1)
        pids = {first.worker_at(0).pid()}
        release_pool(first)  # a cached pool is NOT closed by release
        assert first.worker_at(0).pid() in pids

        second = acquire_pool(spec, max_size=2, prewarm=1)
        assert second is first, "a matching spec must reuse the warm pool"

        # A different core is not interchangeable, so it must not be reused.
        other = acquire_pool(_spec(WienerSpectralEnhancer), max_size=2, prewarm=1)
        assert other is not first
        release_pool(other)
    finally:
        set_pool_reuse(False)
        shutdown_pool_cache()
    assert not pool_reuse_enabled()


@pytest.mark.unit
def test_release_closes_a_pool_that_is_not_cached() -> None:
    pool = acquire_pool(_spec(NoOpEnhancer), max_size=1, prewarm=1)
    worker = pool.worker_at(0)
    proc = worker.process
    release_pool(pool)
    assert proc is None or not proc.is_alive()


def _core_hashes(enhancer: Any, waves: list[np.ndarray[Any, np.dtype[np.float32]]]) -> list[bytes]:
    return [enhancer.enhance(w, SR).waveform.tobytes() for w in waves]


@pytest.mark.unit
def test_wiener_core_is_stateless_across_calls() -> None:
    """The pool's whole licence: a unit's result must not depend on what the
    worker did before it. Same unit, first / second / alone -> same bytes."""
    rng = np.random.default_rng(11)
    a = (0.2 * np.sin(2 * np.pi * 180 * np.arange(SR) / SR)).astype(np.float32)
    a += (0.03 * rng.standard_normal(SR)).astype(np.float32)
    b = (0.2 * np.sin(2 * np.pi * 240 * np.arange(SR) / SR)).astype(np.float32)
    b += (0.03 * rng.standard_normal(SR)).astype(np.float32)

    def fresh() -> Any:
        e = WienerSpectralEnhancer()
        e.warmup()
        return e

    ab = _core_hashes(fresh(), [a, b])
    ba = _core_hashes(fresh(), [b, a])
    only_b = _core_hashes(fresh(), [b])
    assert ab[1] == ba[0] == only_b[0], "the Wiener core carries state between calls"
    assert ab[0] == ba[1]


@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("df") is None, reason="studio core needs the studio extra"
)
def test_studio_core_is_stateless_across_calls() -> None:
    """Same proof for the core that actually holds a neural model. This is the
    one that could plausibly have carried STFT or normalisation state between
    calls, which would have made a pool change the audio."""
    from hawavoclean.enhancement.studio import StudioVoiceCore

    rng = np.random.default_rng(12)
    a = (0.2 * np.sin(2 * np.pi * 180 * np.arange(SR) / SR)).astype(np.float32)
    a += (0.03 * rng.standard_normal(SR)).astype(np.float32)
    b = (0.2 * np.sin(2 * np.pi * 240 * np.arange(SR) / SR)).astype(np.float32)
    b += (0.03 * rng.standard_normal(SR)).astype(np.float32)

    def fresh() -> Any:
        e = StudioVoiceCore(sample_rate=SR, phase_coherent=False)
        e.warmup()
        return e

    ab = _core_hashes(fresh(), [a, b])
    only_b = _core_hashes(fresh(), [b])
    assert ab[1] == only_b[0], "DeepFilterNet3 carries state between calls"


class _OnlyOneStarts:
    """A factory that staffs slot 0 and refuses every other slot — what an
    out-of-memory machine looks like from inside the pool.

    Slot 0 is the one the pool builds on the calling thread; every other slot
    is built on a prewarm or dispatch thread. That is what makes the refusal
    deterministic here rather than a race for "who asked first".
    """

    def __init__(self) -> None:
        self.refusals = 0

    def __call__(self, **kwargs: Any) -> Any:
        if threading.current_thread() is not threading.main_thread():
            self.refusals += 1
            raise WorkerCrashError("no room for a second worker")
        return IsolatedEnhancementWorker(**kwargs)


@pytest.mark.unit
def test_a_slot_that_cannot_be_staffed_does_not_cost_a_unit() -> None:
    """A worker that refuses to start takes its slot out of service, not the
    run: the units it would have taken go back to a slot that works, and every
    one of them still gets a real candidate rather than a fail-closed revert."""
    factory = _OnlyOneStarts()
    pool = EnhancementWorkerPool(_spec(NoOpEnhancer), max_size=4, prewarm=4, worker_factory=factory)
    try:
        out = pool.map_enhance([(_tagged(i), SR) for i in range(4)])
        assert factory.refusals >= 1, "the test did not exercise a failing slot"
        assert [o.ok for o in out] == [True] * 4, (
            "a slot that could not start cost a unit its enhancement"
        )
    finally:
        pool.close()
