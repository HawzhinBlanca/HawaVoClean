"""Bounded, cross-platform source snapshots for probe/decode identity.

The user-facing path is mutable. A path checked by ``stat`` and reopened by
FFmpeg later is therefore not an identity: it can name different bytes at the
two points. This module opens one regular-file object, copies it through a
bounded buffer into private application scratch, hashes those copied bytes and
freezes the snapshot before any parser sees it. Probe and decode then consume
the same application-owned file.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hawavoclean.errors import MediaPreflightError, MediaPreflightReason, PreflightError

SOURCE_COPY_CHUNK_BYTES = 1 << 20
SOURCE_SNAPSHOT_SAFETY_MARGIN_BYTES = 500 * 1024 * 1024


def _is_redirect(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(reparse_flag and attributes & reparse_flag)


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _Fingerprint:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def same_file(self, other: _Fingerprint) -> bool:
        return (self.device, self.inode) == (other.device, other.inode)


def remove_source_snapshot_tree(directory: Path) -> None:
    """Remove a frozen snapshot on POSIX and Windows."""
    if not directory.exists():
        return
    for root, directories, files in os.walk(directory, topdown=False):
        for name in files:
            with contextlib.suppress(OSError):
                Path(root, name).chmod(0o600)
        for name in directories:
            with contextlib.suppress(OSError):
                Path(root, name).chmod(0o700)
    with contextlib.suppress(OSError):
        directory.chmod(0o700)
    shutil.rmtree(directory, ignore_errors=True)


@dataclass(slots=True)
class PinnedSource:
    """Private source snapshot that can be adopted by a job workspace."""

    original_path: Path
    directory: Path
    path: Path
    sha256: str
    size_bytes: int
    _fingerprint: _Fingerprint
    _adopted: bool = False

    @classmethod
    def create(
        cls,
        source_path: Path | str,
        *,
        staging_root: Path,
        max_file_size_bytes: int,
    ) -> PinnedSource:
        source = Path(source_path)
        descriptor = -1
        directory: Path | None = None
        try:
            try:
                before_info = os.lstat(source)
            except OSError as exc:
                raise MediaPreflightError(
                    MediaPreflightReason.NOT_FOUND,
                    f"Input audio file does not exist or cannot be read: {source}",
                ) from exc
            if _is_redirect(before_info) or not stat.S_ISREG(before_info.st_mode):
                raise MediaPreflightError(
                    MediaPreflightReason.NOT_REGULAR_FILE,
                    f"Input audio source is not a regular file: {source}",
                )

            flags = os.O_RDONLY
            flags |= getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOINHERIT", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(source, flags)
            except OSError as exc:
                raise MediaPreflightError(
                    MediaPreflightReason.SOURCE_CHANGED,
                    f"Input source changed or could not be pinned safely: {source}",
                ) from exc

            opened_info = os.fstat(descriptor)
            after_open_info = os.lstat(source)
            before = _Fingerprint.from_stat(before_info)
            opened = _Fingerprint.from_stat(opened_info)
            after_open = _Fingerprint.from_stat(after_open_info)
            if (
                not stat.S_ISREG(opened_info.st_mode)
                or _is_redirect(after_open_info)
                or not before.same_file(opened)
                or not opened.same_file(after_open)
            ):
                raise MediaPreflightError(
                    MediaPreflightReason.SOURCE_CHANGED,
                    f"Input source changed identity while it was being opened: {source}",
                )
            if opened.size <= 0:
                raise MediaPreflightError(
                    MediaPreflightReason.EMPTY_FILE,
                    f"Input audio file is empty: {source}",
                )
            if opened.size > max_file_size_bytes:
                raise MediaPreflightError(
                    MediaPreflightReason.FILE_TOO_LARGE,
                    f"Input file is {opened.size:,} bytes; the maximum is "
                    f"{max_file_size_bytes:,} bytes.",
                )

            try:
                staging_root.mkdir(parents=True, exist_ok=True)
                staging_root.chmod(0o700)
                free_bytes = shutil.disk_usage(staging_root).free
            except OSError as exc:
                raise PreflightError(f"Cannot prepare source snapshot storage: {exc}") from exc
            required = opened.size + SOURCE_SNAPSHOT_SAFETY_MARGIN_BYTES
            if free_bytes < required:
                raise PreflightError(
                    "Insufficient scratch space for immutable source snapshot: "
                    f"available {free_bytes / (1024 * 1024):.1f} MiB, "
                    f"required {required / (1024 * 1024):.1f} MiB."
                )

            try:
                directory = Path(tempfile.mkdtemp(prefix="source-pin-", dir=staging_root)).resolve()
                directory.chmod(0o700)
                destination = directory / f"source{source.suffix.lower()}"
                digest = hashlib.sha256()
                copied = 0
                with open(destination, "xb", buffering=0) as output:
                    while copied < opened.size:
                        remaining = opened.size - copied
                        block = os.read(descriptor, min(SOURCE_COPY_CHUNK_BYTES, remaining))
                        if not block:
                            raise MediaPreflightError(
                                MediaPreflightReason.SOURCE_CHANGED,
                                f"Input source became shorter while it was pinned: {source}",
                            )
                        free_bytes = shutil.disk_usage(directory).free
                        if free_bytes < remaining + SOURCE_SNAPSHOT_SAFETY_MARGIN_BYTES:
                            raise PreflightError(
                                f"Scratch capacity disappeared while pinning source audio: {source}"
                            )
                        output.write(block)
                        digest.update(block)
                        copied += len(block)
                    if os.read(descriptor, 1):
                        raise MediaPreflightError(
                            MediaPreflightReason.SOURCE_CHANGED,
                            f"Input source became longer while it was pinned: {source}",
                        )
                    output.flush()
                    os.fsync(output.fileno())
            except (MediaPreflightError, PreflightError):
                raise
            except OSError as exc:
                raise PreflightError(f"Could not create source snapshot: {exc}") from exc

            opened_after_info = os.fstat(descriptor)
            try:
                path_after_info = os.lstat(source)
            except OSError as exc:
                raise MediaPreflightError(
                    MediaPreflightReason.SOURCE_CHANGED,
                    f"Input source disappeared while it was pinned: {source}",
                ) from exc
            opened_after = _Fingerprint.from_stat(opened_after_info)
            path_after = _Fingerprint.from_stat(path_after_info)
            if (
                opened_after != opened
                or path_after != opened
                or not opened.same_file(path_after)
                or _is_redirect(path_after_info)
            ):
                raise MediaPreflightError(
                    MediaPreflightReason.SOURCE_CHANGED,
                    f"Input source changed while it was pinned: {source}",
                )

            # Normal writers now fail even if they somehow discover the
            # unpredictable private path. The workspace cleanup path restores
            # permissions before removing it on Windows.
            destination.chmod(0o400)
            directory.chmod(0o500)
            frozen = _Fingerprint.from_stat(os.lstat(destination))
            return cls(
                original_path=source,
                directory=directory,
                path=destination,
                sha256=digest.hexdigest(),
                size_bytes=copied,
                _fingerprint=frozen,
            )
        except BaseException:
            if directory is not None:
                remove_source_snapshot_tree(directory)
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def verify(self) -> None:
        """Fail if the private snapshot changed identity or bytes metadata."""
        try:
            current_info = os.lstat(self.path)
        except OSError as exc:
            raise MediaPreflightError(
                MediaPreflightReason.SOURCE_CHANGED,
                "Pinned source snapshot disappeared before decode.",
            ) from exc
        current = _Fingerprint.from_stat(current_info)
        if _is_redirect(current_info) or not stat.S_ISREG(current_info.st_mode):
            raise MediaPreflightError(
                MediaPreflightReason.SOURCE_CHANGED,
                "Pinned source snapshot is no longer a regular file.",
            )
        if current != self._fingerprint:
            raise MediaPreflightError(
                MediaPreflightReason.SOURCE_CHANGED,
                "Pinned source snapshot changed before decode.",
            )

    def adopt(self, workspace_root: Path) -> Path:
        """Move the snapshot under a job so crash forensics retain its bytes."""
        self.verify()
        destination = workspace_root / "source-snapshot"
        if destination.exists():
            raise PreflightError(f"Source snapshot destination already exists: {destination}")
        try:
            # APFS and NTFS may refuse moving a read-only directory even when
            # both parents are writable. Thaw only the directory entry for the
            # atomic move; the snapshot file remains read-only throughout.
            self.directory.chmod(0o700)
            os.replace(self.directory, destination)
        except OSError as exc:
            with contextlib.suppress(OSError):
                self.directory.chmod(0o500)
            raise PreflightError(
                f"Could not attach source snapshot to job workspace: {exc}"
            ) from exc
        self.directory = destination
        self.path = destination / self.path.name
        self._adopted = True
        self.directory.chmod(0o500)
        self._fingerprint = _Fingerprint.from_stat(os.lstat(self.path))
        self.verify()
        return self.path

    def cleanup_unadopted(self) -> None:
        if not self._adopted:
            remove_source_snapshot_tree(self.directory)

    def __enter__(self) -> PinnedSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup_unadopted()
