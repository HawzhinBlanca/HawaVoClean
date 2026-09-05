"""Bounded, fail-closed retention for engine-managed upload inputs.

Only files named by a strict marker inside ``work/uploads/<uuid>/`` are ever
eligible for deletion. Outputs may share that directory; cleanup removes the
input and marker, then removes the directory only when it is empty.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from hawavoclean.platform_fs import flush_directory

DEFAULT_UPLOAD_TTL_S = 24 * 60 * 60.0
DEFAULT_MAX_UPLOAD_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MIN_FREE_BYTES = 512 * 1024 * 1024
UPLOAD_MARKER = ".hawavoclean-upload.json"
_UPLOAD_DIR = re.compile(r"^[0-9a-f]{32}$")


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


DiskUsageFactory = Callable[[Path], DiskUsage]


class StoragePressureError(RuntimeError):
    """The managed upload quota or free-space reserve would be exceeded."""


class UploadStore:
    """Marker-scoped uploads with total quota, TTL, and an emergency reserve."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_s: float = DEFAULT_UPLOAD_TTL_S,
        max_total_bytes: int = DEFAULT_MAX_UPLOAD_TOTAL_BYTES,
        min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
        clock: Callable[[], float] | None = None,
        disk_usage: DiskUsageFactory | None = None,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("upload ttl_s must be positive")
        if max_total_bytes < 1:
            raise ValueError("max_total_bytes must be at least 1")
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        self.root = root.resolve()
        self.ttl_s = ttl_s
        self.max_total_bytes = max_total_bytes
        self.min_free_bytes = min_free_bytes
        self._clock = clock or time.time
        self._disk_usage: DiskUsageFactory = disk_usage or (
            lambda path: DiskUsage(*shutil.disk_usage(path))
        )
        self._lock = threading.Lock()
        self._leases: dict[Path, int] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def _read_marker(self, directory: Path) -> tuple[Path, float] | None:
        if directory.parent != self.root or _UPLOAD_DIR.fullmatch(directory.name) is None:
            return None
        marker = directory / UPLOAD_MARKER
        if marker.is_symlink() or not marker.is_file():
            return None
        try:
            value: Any = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "created_epoch",
            "input_name",
        }:
            return None
        name = value["input_name"]
        created = value["created_epoch"]
        if (
            value["schema_version"] != 1
            or not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(created, (int, float))
            or isinstance(created, bool)
            or created < 0
        ):
            return None
        return directory / name, float(created)

    def _managed(self) -> list[tuple[Path, Path, float]]:
        records: list[tuple[Path, Path, float]] = []
        try:
            directories = list(self.root.iterdir())
        except OSError:
            return records
        for directory in directories:
            if directory.is_symlink() or not directory.is_dir():
                continue
            record = self._read_marker(directory)
            if record is not None:
                input_path, created = record
                records.append((directory, input_path, created))
        return records

    def _cleanup_record(self, directory: Path, input_path: Path) -> None:
        if input_path.parent != directory:
            return
        with contextlib.suppress(OSError):
            input_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            (directory / UPLOAD_MARKER).unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            directory.rmdir()

    def scavenge(self, active_inputs: Iterable[Path] = ()) -> int:
        """Remove expired managed inputs, never active inputs or sibling outputs."""
        active = {path.resolve() for path in active_inputs}
        removed = 0
        with self._lock:
            active.update(self._leases)
            now = self._clock()
            for directory, input_path, created in self._managed():
                if input_path.resolve() in active or now - created < self.ttl_s:
                    continue
                self._cleanup_record(directory, input_path)
                removed += 1
        return removed

    def usage_bytes(self) -> int:
        """Bytes belonging to marked upload inputs (never user outputs)."""
        with self._lock:
            total = 0
            for _directory, input_path, _created in self._managed():
                try:
                    if not input_path.is_symlink() and input_path.is_file():
                        total += input_path.stat().st_size
                except OSError:
                    continue
            return total

    def ensure_capacity(self, existing_bytes: int, additional_bytes: int) -> None:
        if existing_bytes < 0 or additional_bytes < 0:
            raise ValueError("storage byte counts cannot be negative")
        requested_total = existing_bytes + additional_bytes
        if requested_total > self.max_total_bytes:
            raise StoragePressureError(
                f"managed uploads would exceed the {self.max_total_bytes} byte total quota"
            )
        free = self._disk_usage(self.root).free
        if free - additional_bytes < self.min_free_bytes:
            raise StoragePressureError(
                f"upload would cross the {self.min_free_bytes} byte free-space reserve"
            )

    def ensure_progress(self, existing_bytes: int, written_bytes: int) -> None:
        if existing_bytes + written_bytes > self.max_total_bytes:
            raise StoragePressureError(
                f"managed uploads exceed the {self.max_total_bytes} byte total quota"
            )
        if self._disk_usage(self.root).free < self.min_free_bytes:
            raise StoragePressureError(
                f"disk free space is below the {self.min_free_bytes} byte reserve"
            )

    def stage(self, input_name: str) -> Path:
        """Allocate one marked directory. The caller writes only the returned path."""
        if not input_name or Path(input_name).name != input_name:
            raise ValueError("input_name must be one safe basename")
        with self._lock:
            directory = self.root / uuid.uuid4().hex
            directory.mkdir(mode=0o700)
            marker_path = directory / UPLOAD_MARKER
            marker_temp = directory / f"{UPLOAD_MARKER}.tmp"
            marker = {
                "schema_version": 1,
                "created_epoch": self._clock(),
                "input_name": input_name,
            }
            try:
                with marker_temp.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(marker_temp, marker_path)
                flush_directory(directory)
            except BaseException:
                with contextlib.suppress(OSError):
                    marker_temp.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    marker_path.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    directory.rmdir()
                raise
            return directory / input_name

    def source_id(self, input_path: Path) -> str:
        """Return the opaque API id for one marker-owned input."""

        candidate = input_path.resolve()
        directory = candidate.parent
        with self._lock:
            record = self._read_marker(directory)
            if record is None or record[0].resolve() != candidate:
                raise ValueError("input is not owned by this upload store")
            return directory.name

    def authorizes(self, input_path: Path) -> bool:
        """Whether ``input_path`` is the exact, live marker-owned upload.

        This is the path-form compatibility check used by legacy renderer
        endpoints.  It deliberately shares the same marker, symlink and
        regular-file validation as opaque ``source_id`` resolution.
        """

        try:
            candidate = input_path.resolve()
        except (OSError, ValueError):
            return False
        directory = candidate.parent
        with self._lock:
            record = self._read_marker(directory)
            if record is None or record[0].resolve() != candidate:
                return False
            try:
                return not candidate.is_symlink() and candidate.is_file()
            except OSError:
                return False

    def resolve_source(self, source_id: str) -> Path | None:
        """Resolve an opaque source id without accepting paths or traversal."""

        with self._lock:
            return self._resolve_source_locked(source_id)

    def _resolve_source_locked(self, source_id: str) -> Path | None:
        if _UPLOAD_DIR.fullmatch(source_id) is None:
            return None
        directory = self.root / source_id
        if directory.is_symlink() or not directory.is_dir():
            return None
        record = self._read_marker(directory)
        if record is None:
            return None
        input_path = record[0]
        try:
            if input_path.is_symlink() or not input_path.is_file():
                return None
        except OSError:
            return None
        return input_path.resolve()

    @contextmanager
    def lease_source(self, source_id: str) -> Iterator[Path | None]:
        """Keep one managed input alive for a bounded analysis/read operation.

        Resolving a source and later opening it leaves a race with TTL or job
        cleanup.  A lease closes that gap without exposing a filesystem path
        to the client.  Cleanup remains idempotent: if it observes a lease it
        leaves the input for the next scavenging pass.
        """

        leased: Path | None = None
        with self._lock:
            leased = self._resolve_source_locked(source_id)
            if leased is not None:
                self._leases[leased] = self._leases.get(leased, 0) + 1
        try:
            yield leased
        finally:
            if leased is not None:
                with self._lock:
                    remaining = self._leases.get(leased, 0) - 1
                    if remaining > 0:
                        self._leases[leased] = remaining
                    else:
                        self._leases.pop(leased, None)

    def cleanup_input(self, input_path: Path) -> bool:
        """Delete one marker-owned input. Sibling files (committed outputs) survive."""
        candidate = input_path.resolve()
        directory = candidate.parent
        with self._lock:
            if candidate in self._leases:
                return False
            record = self._read_marker(directory)
            if record is None or record[0].resolve() != candidate:
                return False
            self._cleanup_record(directory, candidate)
            return True
