"""Opaque capabilities for native files selected outside the renderer.

The desktop and Resolve main processes own the broker root secret.  They use
that authority once, immediately after an OS/Resolve selection, to register an
exact regular file.  Renderer sessions never gain a general "read a path"
capability: they can name only a path that is still backed by the same file
identity registered here, or use the opaque 128-bit source id.

Registrations are intentionally process-local.  A broker restart invalidates
them along with every renderer session, so stale capabilities cannot silently
survive a new trust boundary.
"""

from __future__ import annotations

import secrets
import stat
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from hawavoclean.server.policy import PathPolicyError, refuse_unusable_filename_text

MAX_NATIVE_SOURCES = 4096
SOURCE_ID_HEX_LENGTH = 32


def resolve_native_selected_path(raw_path: str) -> Path:
    """Resolve one root-authorized OS selection without a root allowlist.

    A native file picker can legitimately select a secondary Windows drive,
    network mount, or another OS location that is not below the home/work
    roots used for legacy browser paths.  The root-only registration call is
    the authority; subsequent renderer access remains exact-capability based.
    """

    if not raw_path or not raw_path.strip():
        raise PathPolicyError(400, "bad_request", "selected source path is required")
    refuse_unusable_filename_text(raw_path, what="selected source path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise PathPolicyError(400, "bad_request", "selected source path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise PathPolicyError(404, "not_found", "selected source file was not found") from exc
    try:
        info = resolved.stat()
    except OSError as exc:
        raise PathPolicyError(404, "not_found", "selected source file was not found") from exc
    if not stat.S_ISREG(info.st_mode):
        raise PathPolicyError(404, "not_found", "selected source is not a regular file")
    return resolved


@dataclass(frozen=True)
class NativeSource:
    """One exact native file authorized by a root-owned selection."""

    source_id: str
    path: Path
    device: int
    inode: int


class NativeSourceRegistry:
    """Bounded, thread-safe native-file capabilities.

    Both canonical path and filesystem identity are checked on every use.  A
    path swapped to a symlink, a different regular file, or a different hard
    link after selection therefore loses authority instead of inheriting it.
    In-place edits retain authority, which is necessary for media files that a
    host application is still finalising.
    """

    def __init__(self, maximum: int = MAX_NATIVE_SOURCES) -> None:
        if maximum < 1 or maximum > 65_536:
            raise ValueError("maximum native sources must be between 1 and 65536")
        self.maximum = maximum
        self._entries: OrderedDict[str, NativeSource] = OrderedDict()
        self._by_path: dict[Path, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _identity(path: Path) -> tuple[int, int] | None:
        try:
            current = path.resolve(strict=True)
            info = current.stat()
        except (OSError, ValueError):
            return None
        if current != path or not stat.S_ISREG(info.st_mode):
            return None
        return int(info.st_dev), int(info.st_ino)

    def register(self, raw_path: str) -> NativeSource:
        """Register one exact, existing regular file and return its capability."""

        path = resolve_native_selected_path(raw_path)
        identity = self._identity(path)
        if identity is None:
            raise PathPolicyError(404, "not_found", "selected source is not a regular file")
        with self._lock:
            previous_id = self._by_path.get(path)
            if previous_id is not None:
                previous = self._entries.get(previous_id)
                if previous is not None and (previous.device, previous.inode) == identity:
                    self._entries.move_to_end(previous_id)
                    return previous
                self._entries.pop(previous_id, None)
                self._by_path.pop(path, None)

            source_id = secrets.token_hex(SOURCE_ID_HEX_LENGTH // 2)
            while source_id in self._entries:  # pragma: no cover - 128-bit collision defence
                source_id = secrets.token_hex(SOURCE_ID_HEX_LENGTH // 2)
            source = NativeSource(source_id, path, *identity)
            self._entries[source_id] = source
            self._by_path[path] = source_id
            while len(self._entries) > self.maximum:
                _, evicted = self._entries.popitem(last=False)
                self._by_path.pop(evicted.path, None)
            return source

    def _valid_locked(self, source: NativeSource) -> bool:
        return self._identity(source.path) == (source.device, source.inode)

    def resolve_source(self, source_id: str) -> Path | None:
        """Resolve a 32-hex capability if the selected file is still identical."""

        if len(source_id) != SOURCE_ID_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in source_id
        ):
            return None
        with self._lock:
            source = self._entries.get(source_id)
            if source is None:
                return None
            if not self._valid_locked(source):
                self._entries.pop(source_id, None)
                self._by_path.pop(source.path, None)
                return None
            self._entries.move_to_end(source_id)
            return source.path

    def authorizes(self, path: Path) -> bool:
        """Return whether ``path`` is the still-identical registered file."""

        candidate = path.resolve()
        with self._lock:
            source_id = self._by_path.get(candidate)
            if source_id is None:
                return False
            source = self._entries.get(source_id)
            if source is None or not self._valid_locked(source):
                self._entries.pop(source_id, None)
                self._by_path.pop(candidate, None)
                return False
            self._entries.move_to_end(source_id)
            return True

    def resolve_registered_path(self, raw_path: str) -> Path | None:
        """Resolve a path-form legacy request only when root registered it."""

        try:
            candidate = resolve_native_selected_path(raw_path)
        except PathPolicyError:
            return None
        return candidate if self.authorizes(candidate) else None

    @contextmanager
    def lease_source(self, source_id: str) -> Iterator[Path | None]:
        """Match the upload-store lease interface for versioned source ids.

        Native inputs are user-owned and never scavenged by HawaVoClean, so no
        retention lease is required after the identity check.
        """

        yield self.resolve_source(source_id)
