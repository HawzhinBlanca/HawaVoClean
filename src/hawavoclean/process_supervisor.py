"""Cross-platform ownership and termination of a complete child process tree.

Every long-running engine child is born inside an operating-system boundary:

* POSIX children start a new session, whose process-group id is the child's pid.
* Windows children start a new process group and are assigned to a Job Object
  configured with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.

Callers keep the :class:`ProcessSupervisor` alive for as long as they use the
child.  ``terminate_tree`` first requests a graceful shutdown, waits for the
bounded grace period, and then ends the complete boundary.  ``close`` is
idempotent and fail-closed: leaked descendants are killed before ownership is
released.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, cast

PlatformKind = Literal["posix", "windows"]

# These values are part of the stable Win32 API.  They are defined here rather
# than read from ``subprocess``/``signal`` because those modules do not expose
# the Windows-only names when the test suite runs on macOS.
WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x0000_0200
WINDOWS_CREATE_SUSPENDED = 0x0000_0004
WINDOWS_CTRL_BREAK_EVENT = 1

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x0000_2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_THREAD_SUSPEND_RESUME = 0x0002
_TH32CS_SNAPTHREAD = 0x0000_0004
_INVALID_DWORD = 0xFFFF_FFFF


class ProcessSupervisorError(OSError):
    """A child was spawned but could not be placed in its process boundary."""


class WindowsJobApi(Protocol):
    """Small injectable surface around the native Windows Job Object API."""

    def create_kill_on_close_job(self) -> int:
        """Return a configured Job Object handle."""

    def assign_process(self, job_handle: int, pid: int) -> None:
        """Assign ``pid`` to ``job_handle`` or raise ``OSError``."""

    def resume_process(self, pid: int) -> None:
        """Resume the initial thread of a process created suspended."""

    def terminate_job(self, job_handle: int, exit_code: int) -> None:
        """Terminate every process in the job."""

    def close_handle(self, job_handle: int) -> None:
        """Close a Job Object handle."""


class _NativeWindowsJobApi:
    """ctypes binding loaded only on Windows.

    ``OpenProcess`` avoids depending on CPython's private ``Popen._handle``.
    The retained handle is the Job Object itself; the temporary process handle
    is closed immediately after assignment.
    """

    def __init__(self) -> None:
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class _ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        win_dll = cast(Any, vars(ctypes)["WinDLL"])
        self._kernel32 = win_dll("kernel32", use_last_error=True)
        self._info_type = _ExtendedLimitInformation
        self._thread_entry_type = _ThreadEntry32
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self._kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self._kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        self._kernel32.Thread32First.restype = wintypes.BOOL
        self._kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        self._kernel32.Thread32Next.restype = wintypes.BOOL
        self._kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenThread.restype = wintypes.HANDLE
        self._kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        self._kernel32.ResumeThread.restype = wintypes.DWORD
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.GetLastError.restype = wintypes.DWORD

    def _last_error(self) -> OSError:
        code = int(self._kernel32.GetLastError())
        return OSError(code, f"Win32 error {code}")

    def create_kill_on_close_job(self) -> int:
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise self._last_error()
        job_handle = int(handle)
        info = self._info_type()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = self._kernel32.SetInformationJobObject(
            job_handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            error = self._last_error()
            self.close_handle(job_handle)
            raise error
        return job_handle

    def assign_process(self, job_handle: int, pid: int) -> None:
        process_handle = self._kernel32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
            False,
            pid,
        )
        if not process_handle:
            raise self._last_error()
        try:
            if not self._kernel32.AssignProcessToJobObject(job_handle, process_handle):
                raise self._last_error()
        finally:
            self._kernel32.CloseHandle(process_handle)

    def resume_process(self, pid: int) -> None:
        """Find and resume the only thread a CREATE_SUSPENDED child can own."""

        snapshot = self._kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
        if int(snapshot) in {0, ctypes.c_void_p(-1).value}:
            raise self._last_error()
        resumed = 0
        try:
            entry = self._thread_entry_type()
            entry.dwSize = ctypes.sizeof(entry)
            more = bool(self._kernel32.Thread32First(snapshot, ctypes.byref(entry)))
            while more:
                if int(entry.th32OwnerProcessID) == pid:
                    thread = self._kernel32.OpenThread(
                        _THREAD_SUSPEND_RESUME,
                        False,
                        int(entry.th32ThreadID),
                    )
                    if not thread:
                        raise self._last_error()
                    try:
                        if int(self._kernel32.ResumeThread(thread)) == _INVALID_DWORD:
                            raise self._last_error()
                        resumed += 1
                    finally:
                        self._kernel32.CloseHandle(thread)
                more = bool(self._kernel32.Thread32Next(snapshot, ctypes.byref(entry)))
        finally:
            self._kernel32.CloseHandle(snapshot)
        if resumed != 1:
            raise OSError(
                0,
                f"expected one suspended initial thread for pid {pid}, found {resumed}",
            )

    def terminate_job(self, job_handle: int, exit_code: int) -> None:
        if not self._kernel32.TerminateJobObject(job_handle, exit_code):
            raise self._last_error()

    def close_handle(self, job_handle: int) -> None:
        if not self._kernel32.CloseHandle(job_handle):
            raise self._last_error()


PopenFactory = Callable[..., subprocess.Popen[str]]


def _platform_kind(value: str | None) -> PlatformKind:
    if value is None:
        return "windows" if os.name == "nt" else "posix"
    if value in {"nt", "win32", "windows"}:
        return "windows"
    if value in {"posix", "darwin", "linux"}:
        return "posix"
    raise ValueError(f"unsupported process-supervisor platform {value!r}")


class ProcessSupervisor:
    """Own one child and the OS boundary containing all of its descendants."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        platform: PlatformKind,
        windows_api: WindowsJobApi | None = None,
        windows_job_handle: int | None = None,
    ) -> None:
        self.process = process
        self._platform = platform
        self._windows_api = windows_api
        self._windows_job_handle = windows_job_handle
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def spawn(
        cls,
        command: Sequence[str],
        *,
        platform: str | None = None,
        windows_api: WindowsJobApi | None = None,
        popen_factory: PopenFactory | None = None,
        env: Mapping[str, str] | None = None,
        **popen_kwargs: Any,
    ) -> ProcessSupervisor:
        """Spawn ``command`` in a new owned process-tree boundary.

        ``platform``, ``windows_api`` and ``popen_factory`` make the Win32 path
        fully testable on non-Windows CI.  Product callers leave them unset.
        """

        kind = _platform_kind(platform)
        factory = popen_factory or cast(PopenFactory, subprocess.Popen)
        kwargs = dict(popen_kwargs)
        if env is not None:
            kwargs["env"] = env

        api: WindowsJobApi | None = None
        job_handle: int | None = None
        if kind == "windows":
            # ``start_new_session`` is POSIX-only.  Preserve any caller flags
            # while adding the process-group boundary required for CTRL_BREAK.
            # CREATE_SUSPENDED is load-bearing: without it a fast child can
            # spawn a detached grandchild before Job Object assignment.
            kwargs.pop("start_new_session", None)
            kwargs["creationflags"] = (
                int(kwargs.get("creationflags", 0))
                | WINDOWS_CREATE_NEW_PROCESS_GROUP
                | WINDOWS_CREATE_SUSPENDED
            )
            api = windows_api or _NativeWindowsJobApi()
            job_handle = api.create_kill_on_close_job()
        else:
            kwargs["start_new_session"] = True

        try:
            process = factory(list(command), **kwargs)
        except BaseException:
            if api is not None and job_handle is not None:
                with contextlib.suppress(Exception):
                    api.close_handle(job_handle)
            raise
        if kind == "windows":
            try:
                assert api is not None and job_handle is not None
                api.assign_process(job_handle, process.pid)
                api.resume_process(process.pid)
            except Exception as exc:
                # A running but unowned child would violate the central
                # guarantee. The child is still suspended if assignment or
                # resume failed, so fail the spawn and reap it before return.
                if job_handle is not None:
                    assert api is not None
                    with contextlib.suppress(Exception):
                        api.terminate_job(job_handle, 1)
                    with contextlib.suppress(Exception):
                        api.close_handle(job_handle)
                with contextlib.suppress(Exception):
                    process.kill()
                with contextlib.suppress(Exception):
                    process.wait(timeout=5.0)
                raise ProcessSupervisorError(
                    f"could not atomically contain child pid {process.pid} in a Windows Job Object: {exc}"
                ) from exc

        return cls(
            process,
            platform=kind,
            windows_api=api,
            windows_job_handle=job_handle,
        )

    @property
    def platform(self) -> PlatformKind:
        return self._platform

    def request_graceful_shutdown(self) -> None:
        """Ask the complete boundary to stop without yet forcing termination."""

        with self._lock:
            if self._closed:
                return
        if self._platform == "posix":
            # start_new_session makes pgid == child pid.  Using that known id
            # still reaches surviving descendants after the leader exits.
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(self.process.pid, signal.SIGTERM)
            return

        try:
            self.process.send_signal(WINDOWS_CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            # Services and GUI applications may have no console to receive a
            # CTRL_BREAK event.  Terminating the leader is the best graceful
            # request available; the Job Object remains the hard backstop.
            if self.process.poll() is None:
                with contextlib.suppress(OSError):
                    self.process.terminate()

    def kill_tree(self) -> None:
        """Force every process in the owned boundary to exit."""

        with self._lock:
            if self._closed:
                return
            api = self._windows_api
            job_handle = self._windows_job_handle

        if self._platform == "posix":
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(self.process.pid, signal.SIGKILL)
            if self.process.poll() is None:
                with contextlib.suppress(OSError):
                    self.process.kill()
            return

        job_terminated = False
        if api is not None and job_handle is not None:
            try:
                api.terminate_job(job_handle, 1)
                job_terminated = True
            except OSError:
                job_terminated = False
        if not job_terminated and self.process.poll() is None:
            with contextlib.suppress(OSError):
                self.process.kill()

    def terminate_tree(self, grace_s: float) -> None:
        """Gracefully request shutdown, then force the entire tree to stop."""

        if grace_s < 0:
            raise ValueError("grace_s must be non-negative")
        self.request_graceful_shutdown()
        deadline = time.monotonic() + grace_s
        while self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        # Run this even when the leader exited: POSIX grandchildren can still
        # occupy its process group, and a Windows job can still contain them.
        self.kill_tree()
        if self.process.poll() is None:
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                self.process.wait(timeout=1.0)

    def close(self, *, kill_remaining: bool = True) -> None:
        """Release resources exactly once, killing leaked descendants by default."""

        if kill_remaining:
            self.kill_tree()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            api = self._windows_api
            job_handle = self._windows_job_handle
            self._windows_job_handle = None
        if api is not None and job_handle is not None:
            # KILL_ON_JOB_CLOSE is a second fail-closed backstop, including for
            # descendants still alive after a native termination error.
            with contextlib.suppress(OSError):
                api.close_handle(job_handle)

    def __enter__(self) -> ProcessSupervisor:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()
