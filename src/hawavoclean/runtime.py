"""Enforcement for the ``[runtime]`` configuration section.

A configuration key that nothing reads is a promise the tool does not keep,
and this module exists so ``[runtime]`` keeps its own. It resolves
``runtime.device`` into a concrete compute device, turns ``runtime.num_threads``
into a concrete CPU budget, and publishes both into the process environment so
the spawned enhancement worker inherits them.

Why the environment and not an argument
---------------------------------------
The enhancement core is constructed inside a ``spawn``-ed subprocess whose
only inbound channel is the argument tuple of
``hawavoclean.enhancement.worker._worker_process_entry`` plus the inherited
process environment. Thread budgets (``OMP_NUM_THREADS``) have to be in place
*before* the child imports torch for them to take effect at all, so the
environment is not a workaround here — it is the mechanism that works for both
values in both processes. :data:`DEVICE_ENV_VAR` is written by
:func:`activate_runtime` (called from :func:`hawavoclean.config.load_config`)
and read back by :func:`active_device` in whichever process asks.

Why ``auto`` is CPU today
-------------------------
A GPU backend does not compute the same numbers as the CPU backend. Measured
on this reference machine (Apple M-series, 14 logical cores, torch 2.13,
DeepFilterNet3, a 20 s speech unit at 48 kHz):

===========  =========================  =====================================
device       wall time for a 20 s unit  max |Δ| vs the CPU result
===========  =========================  =====================================
cpu          339 ms  (RTF 0.017)        —
mps          731 ms  (RTF 0.037)        1.8e-08  (not bit-identical)
===========  =========================  =====================================

MPS is 2.2x *slower* here and its output differs, so promoting it under
``auto`` would trade the tool's reproducibility for a regression. ``auto``
therefore resolves through :data:`AUTO_DEVICE_PREFERENCE`, which today contains
only ``cpu``. A GPU is opt-in: write ``device = "mps"`` (or ``"cuda"``) and the
run is recorded as such in the report's ``environment.compute_device``, so a
result can never be attributed to the wrong compute path.

The device is deliberately NOT part of a core's ``params_hash`` or its
lockfile. The lockfile pins *what the model is* — architecture, parameters,
weight digests — and that is identical on every device; the device belongs to
the *environment*, alongside the platform and library versions that BLUEPRINT
invariant 8 already scopes reproducibility to. Folding it into the hash would
make ``hawavoclean audit-models`` fail on a machine merely for owning a GPU,
and would need one locked hash per device to say anything useful.
"""

import os
import sys
from ctypes import Structure, byref, c_size_t, c_ulong
from dataclasses import dataclass
from typing import Any

import numpy as np

from hawavoclean.errors import ConfigError, WorkerOOMError

#: Process-environment channel carrying the resolved device to the spawned
#: enhancement worker (and to anything else that needs to name the compute
#: path). Written by :func:`activate_runtime`, read by :func:`active_device`.
DEVICE_ENV_VAR = "HAWAVOCLEAN_DEVICE"

#: Thread-budget variables honoured by torch/OpenMP/MKL. Only ever set when
#: the configuration asks for a worker pool larger than one; a single worker
#: keeps the numeric stack's own defaults, unchanged.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")

#: Process-environment channel carrying ``runtime.worker_memory_limit_mb`` to
#: the process that holds the enhancement core, which polices itself against
#: it. Absent means unlimited, which is what a core constructed outside a
#: configured run gets.
MEMORY_LIMIT_ENV_VAR = "HAWAVOCLEAN_WORKER_MEMORY_LIMIT_MB"

#: Devices this tool knows how to name. ``cpu`` is always available.
KNOWN_DEVICES = ("cpu", "cuda", "mps")

#: What ``device = "auto"`` picks from, best first. Adding a GPU backend here
#: is a deliberate act: it changes the numbers every run produces (see the
#: module docstring), so a backend earns its place only once it is both
#: measured faster and validated against the reference hashes.
AUTO_DEVICE_PREFERENCE: tuple[str, ...] = ("cpu",)


@dataclass(frozen=True)
class DeviceResolution:
    """What was asked for, what will actually run, and why."""

    requested: str
    resolved: str
    reason: str


def device_available(name: str) -> bool:
    """Is this compute device usable in this process, right now?

    ``cpu`` is always true and never imports torch, so the default path of a
    base (non-studio) install stays torch-free.
    """
    if name == "cpu":
        return True
    if name not in KNOWN_DEVICES:
        return False
    try:
        import torch
    except ImportError:
        return False
    if name == "cuda":
        return bool(torch.cuda.is_available())
    return bool(torch.backends.mps.is_available())


def _core_runs_on_device(core_id: str) -> bool:
    """Does this core's inference actually execute on the selected device?

    A classical-DSP core is numpy on the CPU no matter what the configuration
    asks for, and the report must say so rather than repeating the request.
    """
    from hawavoclean.enhancement.factory import CORE_REGISTRY

    registration = CORE_REGISTRY.get(core_id)
    if registration is None:
        # Unknown core: resolve_core() will refuse it a moment later. Do not
        # invent a downgrade on the way to that error.
        return True
    return registration.device_aware


def resolve_device(
    requested: str,
    core_id: str | None = None,
    preference: tuple[str, ...] | None = None,
) -> DeviceResolution:
    """Turn ``runtime.device`` into the device the run will actually use.

    ``auto`` walks ``preference`` (default :data:`AUTO_DEVICE_PREFERENCE`) and
    takes the first available entry, falling back to ``cpu``. An explicit
    device that this machine cannot provide is a designed error, never a
    silent fallback — a run that quietly drops to the CPU after being told to
    use a GPU has misreported its own compute path.

    ``core_id`` (optional) downgrades the answer to ``cpu`` for cores that do
    not run on an accelerator, so the recorded device is the one that ran.
    """
    if requested not in ("auto", *KNOWN_DEVICES):
        raise ConfigError(
            f"runtime.device = {requested!r} is not a device this tool knows; "
            f"choose one of: auto, {', '.join(KNOWN_DEVICES)}"
        )

    if requested == "auto":
        ladder = AUTO_DEVICE_PREFERENCE if preference is None else preference
        chosen = next((d for d in ladder if device_available(d)), "cpu")
        reason = (
            f"auto selected {chosen!r} from preference {list(ladder)}"
            if chosen != "cpu"
            else "auto resolves to cpu: GPU backends change the numbers and are opt-in"
        )
    else:
        if not device_available(requested):
            raise ConfigError(
                f"runtime.device = {requested!r} was requested explicitly but is not "
                f"available in this process (torch reports it unusable). Install the "
                f"studio extra and a machine that provides {requested!r}, or set "
                f"runtime.device to 'auto'. Refusing to silently fall back to the CPU: "
                f"a GPU run and a CPU run do not produce the same samples."
            )
        chosen = requested
        reason = f"{requested!r} requested explicitly and available"

    if core_id is not None and chosen != "cpu" and not _core_runs_on_device(core_id):
        return DeviceResolution(
            requested=requested,
            resolved="cpu",
            reason=(f"core {core_id!r} is deterministic CPU DSP; {chosen!r} would not be used"),
        )
    return DeviceResolution(requested=requested, resolved=chosen, reason=reason)


def worker_pool_size(num_threads: int, cpu_count: int | None = None) -> int:
    """How many speech units may be enhanced concurrently.

    This is what ``runtime.num_threads`` means: the size of the enhancement
    worker pool, clamped to the machine's logical CPU count. It is emphatically
    NOT an intra-op/BLAS thread count — see :func:`threads_per_worker`.
    """
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, min(int(num_threads), max(1, cpus)))


def threads_per_worker(num_threads: int, cpu_count: int | None = None) -> int:
    """CPU threads each pooled worker may use, so a pool does not oversubscribe.

    With a pool of P workers on C cores, each worker gets ``C // P`` threads
    (at least one). At the default ``num_threads = 1`` the pool is a single
    worker and this is the whole machine, which is what the numeric stack
    already assumes — so nothing is set and nothing changes.
    """
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    pool = worker_pool_size(num_threads, cpu_count=cpus)
    return max(1, max(1, cpus) // pool)


#: What this module last published into THREAD_ENV_VARS, so a second config
#: load in the same process can revise its own value without trampling one an
#: operator set from outside.
_published_thread_budget: str | None = None


def apply_thread_budget(num_threads: int, cpu_count: int | None = None) -> int | None:
    """Publish the per-worker thread budget for this process and its children.

    Returns the budget that was published, or ``None`` when the configuration
    asks for a single worker and the numeric stack is left exactly as it was —
    the default, and the reason making this key real moved no samples.

    An operator who set ``OMP_NUM_THREADS`` themselves outranks the config and
    is never overwritten; a value this module published earlier (a long-lived
    process loading a second profile) is revised rather than left stale.
    """
    global _published_thread_budget
    if worker_pool_size(num_threads, cpu_count=cpu_count) <= 1:
        return None
    budget = str(threads_per_worker(num_threads, cpu_count=cpu_count))
    for var in THREAD_ENV_VARS:
        current = os.environ.get(var)
        if current is None or current == _published_thread_budget:
            os.environ[var] = budget
    _published_thread_budget = budget
    return int(budget)


def _windows_peak_rss_bytes() -> int:
    """Peak working set from the native process counters on Windows."""

    import ctypes

    class ProcessMemoryCounters(Structure):
        _fields_ = [
            ("cb", c_ulong),
            ("PageFaultCount", c_ulong),
            ("PeakWorkingSetSize", c_size_t),
            ("WorkingSetSize", c_size_t),
            ("QuotaPeakPagedPoolUsage", c_size_t),
            ("QuotaPagedPoolUsage", c_size_t),
            ("QuotaPeakNonPagedPoolUsage", c_size_t),
            ("QuotaNonPagedPoolUsage", c_size_t),
            ("PagefileUsage", c_size_t),
            ("PeakPagefileUsage", c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    windll = vars(ctypes)["windll"]
    windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    windll.psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    windll.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = windll.kernel32.GetCurrentProcess()
    if not windll.psapi.GetProcessMemoryInfo(process, byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.PeakWorkingSetSize)


def process_peak_rss_bytes() -> int:
    """High-water resident set of *this* process, in bytes.

    A high-water mark rather than the instantaneous figure on purpose: the
    thing worth catching is the spike that nearly took the machine down, and
    that spike is gone from the instantaneous reading by the time anyone
    looks. ``ru_maxrss`` is bytes on macOS/BSD and kibibytes on Linux.
    """
    if sys.platform == "win32":
        return _windows_peak_rss_bytes()
    # ``resource`` does not exist on Windows. Import it only inside the POSIX
    # branch so a base Windows install can import the engine at all.
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def evict_memmap_pages(
    array: np.ndarray[Any, Any],
    start_sample: int = 0,
    end_sample: int | None = None,
) -> None:
    """Evict physical page frames for a processed slice of a memory-mapped array.

    Keeps the resident set size bounded below process thresholds during long-audio
    streaming passes. Invalidates page table entries back to the OS page cache.
    No-op if the array is not backed by an mmap or on unsupported platforms.
    """
    if not isinstance(array, np.memmap):
        return
    if sys.platform == "win32":
        return

    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        pagesize = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096
        buf_ptr = int(array.__array_interface__["data"][0])
        itemsize = int(array.dtype.itemsize)
        channels = int(array.shape[0]) if array.ndim > 1 else 1
        samples = int(array.shape[-1])

        actual_end = samples if end_sample is None else min(samples, max(start_sample, end_sample))
        if actual_end <= start_sample:
            return

        madv_dontneed = 4
        ms_invalidate = 2
        has_madvise = hasattr(libc, "madvise")
        if has_madvise:
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.madvise.restype = ctypes.c_int
        has_posix_madvise = hasattr(libc, "posix_madvise")
        if has_posix_madvise:
            libc.posix_madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.posix_madvise.restype = ctypes.c_int
        has_msync = hasattr(libc, "msync")
        if has_msync:
            libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.msync.restype = ctypes.c_int

        slice_len = actual_end - start_sample
        for ch in range(channels):
            addr = buf_ptr + (ch * samples + start_sample) * itemsize
            size = slice_len * itemsize
            page_addr = addr & ~(pagesize - 1)
            page_size = ((addr + size + pagesize - 1) & ~(pagesize - 1)) - page_addr
            p_addr = ctypes.c_void_p(page_addr)
            p_size = ctypes.c_size_t(page_size)
            if has_madvise:
                libc.madvise(p_addr, p_size, madv_dontneed)
            elif has_posix_madvise:
                libc.posix_madvise(p_addr, p_size, madv_dontneed)
            if has_msync:
                libc.msync(p_addr, p_size, ms_invalidate)
    except Exception:
        pass


def memory_budget_mb() -> int | None:
    """The resident-set budget published for this process, or ``None``."""
    raw = os.environ.get(MEMORY_LIMIT_ENV_VAR, "")
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def check_memory_budget(context: str = "enhancement") -> None:
    """Refuse further work if this process has already blown its budget.

    Checked by an enhancement core *before* it accepts a unit, so nothing
    already computed is thrown away: the worker that overran stops taking
    work, the parent recycles it, and the refused unit takes the same
    fail-closed path as a crash or a timeout — ORIGINAL audio, recorded with
    its reason. This is what ``runtime.worker_memory_limit_mb`` means; see
    :class:`~hawavoclean.config.RuntimeConfig` for why it is self-policing
    rather than an ``RLIMIT_AS``.
    """
    budget = memory_budget_mb()
    if budget is None:
        return
    peak_mb = process_peak_rss_bytes() // (1024 * 1024)
    if peak_mb > budget:
        raise WorkerOOMError(
            f"{context} worker peaked at {peak_mb} MB, over the configured "
            f"runtime.worker_memory_limit_mb of {budget} MB; refusing further "
            "units so this unit falls back to original audio and the worker is recycled"
        )


def activate_runtime(
    device: str,
    core_id: str | None = None,
    num_threads: int = 1,
    memory_limit_mb: int | None = None,
) -> DeviceResolution:
    """Make a ``[runtime]`` section take effect in this process and its children.

    Called from :func:`hawavoclean.config.load_config`, so loading a profile is
    what arms it. Raises :class:`~hawavoclean.errors.ConfigError` for an
    explicitly requested device this machine cannot provide.
    """
    resolution = resolve_device(device, core_id=core_id)
    os.environ[DEVICE_ENV_VAR] = resolution.resolved
    apply_thread_budget(num_threads)
    if memory_limit_mb is not None:
        os.environ[MEMORY_LIMIT_ENV_VAR] = str(int(memory_limit_mb))
    return resolution


def active_device() -> str:
    """The compute device this process is running enhancement on.

    Reads the channel :func:`activate_runtime` publishes, so the parent (which
    writes the report) and the spawned worker (which runs the model) always
    name the same device. Unset — a config built in code that never went
    through ``load_config`` — means the default, which is ``cpu``.
    """
    name = os.environ.get(DEVICE_ENV_VAR, "")
    return name if name in KNOWN_DEVICES else "cpu"
