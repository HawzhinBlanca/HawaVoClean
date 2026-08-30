"""Portable, deterministic Full Processing Record archives.

The archive contains ordinary files with fixed portable names:

``master.wav``
    The self-contained user master.
``report.json``
    The validated HawaVoClean processing report.
``summary.txt``
    The human-readable summary.
``manifest.json``
    Canonical hashes and sizes for the other three entries.

Creation is streaming and atomic. Verification accepts only this closed,
uncompressed layout, bounds metadata before reading it, hashes every byte, and
cross-checks the report's output identity against the master. This detects
corruption and accidental modification. It is not a publisher signature; a
future signed-release layer must authenticate who created the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Final

from hawavoclean.errors import PublicationError
from hawavoclean.hashing import hash_file
from hawavoclean.platform_fs import (
    exclusive_file_lock,
    flush_directory,
    is_reparse_or_symlink,
    rename_new_path,
    replace_path,
)
from hawavoclean.report.schema import HawaVoCleanReport

SCHEMA_VERSION: Final = 1
PRODUCT: Final = "hawavoclean-full-processing-record"
MASTER_NAME: Final = "master.wav"
REPORT_NAME: Final = "report.json"
SUMMARY_NAME: Final = "summary.txt"
MANIFEST_NAME: Final = "manifest.json"
ENTRY_NAMES: Final[tuple[str, ...]] = (
    MASTER_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    MANIFEST_NAME,
)
MAX_MASTER_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_REPORT_BYTES: Final = 32 * 1024 * 1024
MAX_SUMMARY_BYTES: Final = 4 * 1024 * 1024
MAX_MANIFEST_BYTES: Final = 1024 * 1024
MAX_ZIP_CONTAINER_OVERHEAD: Final = 4 * 1024 * 1024
_CHUNK_BYTES: Final = 1024 * 1024
_ZIP_DATE: Final = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    path: Path
    archive_sha256: str
    master_sha256: str
    report_sha256: str
    summary_sha256: str
    content_sha256: str
    total_uncompressed_bytes: int
    authenticated_publisher: bool = False


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _regular_file(path: Path, *, label: str, maximum_bytes: int) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PublicationError(f"Cannot read {label}: {path} ({exc})") from exc
    if is_reparse_or_symlink(path) or not stat.S_ISREG(metadata.st_mode):
        raise PublicationError(f"{label} must be a regular file, not a link or device: {path}")
    if metadata.st_size < 1:
        raise PublicationError(f"{label} is empty: {path}")
    if metadata.st_size > maximum_bytes:
        raise PublicationError(
            f"{label} exceeds the {maximum_bytes}-byte Processing Record limit: {path}"
        )
    return metadata


def _file_record(path: Path, *, role: str, maximum_bytes: int) -> dict[str, object]:
    metadata = _regular_file(path, label=role, maximum_bytes=maximum_bytes)
    return {"role": role, "sha256": hash_file(path), "size_bytes": metadata.st_size}


def _validate_wave_header(raw: bytes) -> None:
    """Reject a mislabeled payload without attempting to decode gigabytes."""

    if len(raw) < 12 or raw[:4] not in {b"RIFF", b"RF64"} or raw[8:12] != b"WAVE":
        raise PublicationError("Processing Record master is not a RIFF/RF64 WAVE file")


def _content_digest(files: dict[str, dict[str, object]]) -> str:
    return hashlib.sha256(_canonical_json(files)).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_DATE)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_STORED
    info.flag_bits = 0
    return info


def _copy_entry(
    archive: zipfile.ZipFile,
    name: str,
    source: Path,
    *,
    role: str,
    maximum_bytes: int,
) -> dict[str, object]:
    """Copy and hash the exact bytes written through one open source handle."""

    digest = hashlib.sha256()
    seen = 0
    with (
        source.open("rb") as reader,
        archive.open(_zip_info(name), "w", force_zip64=True) as writer,
    ):
        while block := reader.read(_CHUNK_BYTES):
            seen += len(block)
            if seen > maximum_bytes:
                raise PublicationError(
                    f"{role} exceeds the {maximum_bytes}-byte Processing Record limit"
                )
            writer.write(block)
            digest.update(block)
    if seen < 1:
        raise PublicationError(f"{role} became empty while creating Processing Record")
    return {"role": role, "sha256": digest.hexdigest(), "size_bytes": seen}


def create_processing_record(
    *,
    master_path: Path | str,
    report_path: Path | str,
    summary_path: Path | str,
    destination: Path | str,
    overwrite: bool = False,
) -> ProcessingRecord:
    """Create one deterministic, portable archive and publish it atomically."""

    # Keep source paths unresolved until after ``lstat`` so a caller cannot
    # smuggle a symbolic link past the regular-file check.
    master = Path(master_path).expanduser().absolute()
    report = Path(report_path).expanduser().absolute()
    summary = Path(summary_path).expanduser().absolute()
    output = Path(destination).expanduser().absolute()
    if output.suffix.lower() != ".zip":
        raise PublicationError("Full Processing Record destination must end in .zip")

    # Fail on links/devices and obvious resource bombs before creating temp
    # state. Hashes are deliberately computed later from the same open handles
    # whose bytes are written to the archive.
    _regular_file(master, label="master", maximum_bytes=MAX_MASTER_BYTES)
    _regular_file(report, label="report", maximum_bytes=MAX_REPORT_BYTES)
    _regular_file(summary, label="summary", maximum_bytes=MAX_SUMMARY_BYTES)
    if output.resolve(strict=False) in {
        master.resolve(strict=True),
        report.resolve(strict=True),
        summary.resolve(strict=True),
    }:
        raise PublicationError("Refusing to overwrite a Processing Record source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.hawavoclean-record.lock"
    with exclusive_file_lock(lock_path):
        if os.path.lexists(output) and not overwrite:
            raise PublicationError(f"Full Processing Record already exists: {output}")
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temp_path = Path(raw_temp)
        try:
            with zipfile.ZipFile(
                temp_path,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
                strict_timestamps=True,
            ) as archive:
                files = {
                    MASTER_NAME: _copy_entry(
                        archive,
                        MASTER_NAME,
                        master,
                        role="master",
                        maximum_bytes=MAX_MASTER_BYTES,
                    ),
                    REPORT_NAME: _copy_entry(
                        archive,
                        REPORT_NAME,
                        report,
                        role="report",
                        maximum_bytes=MAX_REPORT_BYTES,
                    ),
                    SUMMARY_NAME: _copy_entry(
                        archive,
                        SUMMARY_NAME,
                        summary,
                        role="summary",
                        maximum_bytes=MAX_SUMMARY_BYTES,
                    ),
                }
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "product": PRODUCT,
                    "files": files,
                    "content_sha256": _content_digest(files),
                }
                manifest_bytes = _canonical_json(manifest)
                with archive.open(
                    _zip_info(MANIFEST_NAME), "w", force_zip64=True
                ) as manifest_stream:
                    manifest_stream.write(manifest_bytes)
            with temp_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            # Never replace a valid prior record with an archive that only
            # fails verification after publication. This also closes the
            # source-mutation window because the verifier sees the exact
            # copied bytes and their manifest, not freshly reopened sources.
            verified = verify_processing_record(temp_path)
            try:
                if overwrite:
                    replace_path(temp_path, output)
                else:
                    rename_new_path(temp_path, output)
            except FileExistsError as exc:
                raise PublicationError(f"Full Processing Record already exists: {output}") from exc
            flush_directory(output.parent)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return replace(verified, path=output)


def _entry_is_regular(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_IFMT(mode) in {0, stat.S_IFREG}


def _hash_stream(stream: IO[bytes], expected_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    seen = 0
    while block := stream.read(_CHUNK_BYTES):
        seen += len(block)
        if seen > expected_size:
            raise PublicationError("Processing Record entry exceeds its declared size")
        digest.update(block)
    return digest.hexdigest(), seen


def _manifest_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PublicationError(f"Processing Record manifest is invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PublicationError("Processing Record manifest must be a JSON object")
    if raw != _canonical_json(parsed):
        raise PublicationError("Processing Record manifest is not canonical JSON")
    return parsed


def _validated_manifest(value: dict[str, Any]) -> tuple[dict[str, dict[str, object]], str]:
    if set(value) != {"schema_version", "product", "files", "content_sha256"}:
        raise PublicationError("Processing Record manifest fields differ from schema v1")
    if value["schema_version"] != SCHEMA_VERSION or value["product"] != PRODUCT:
        raise PublicationError("Processing Record manifest identity is unsupported")
    raw_files = value["files"]
    if not isinstance(raw_files, dict) or set(raw_files) != {
        MASTER_NAME,
        REPORT_NAME,
        SUMMARY_NAME,
    }:
        raise PublicationError("Processing Record manifest file inventory is incomplete")
    files: dict[str, dict[str, object]] = {}
    for name, role in (
        (MASTER_NAME, "master"),
        (REPORT_NAME, "report"),
        (SUMMARY_NAME, "summary"),
    ):
        record = raw_files[name]
        if not isinstance(record, dict) or set(record) != {"role", "sha256", "size_bytes"}:
            raise PublicationError(f"Processing Record manifest entry is malformed: {name}")
        digest = record["sha256"]
        size = record["size_bytes"]
        if record["role"] != role:
            raise PublicationError(f"Processing Record manifest role is wrong: {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise PublicationError(f"Processing Record manifest digest is invalid: {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise PublicationError(f"Processing Record manifest size is invalid: {name}")
        files[name] = {"role": role, "sha256": digest, "size_bytes": size}
    content_digest = value["content_sha256"]
    if not isinstance(content_digest, str) or content_digest != _content_digest(files):
        raise PublicationError("Processing Record content identity does not recompute")
    return files, content_digest


def verify_processing_record(path: Path | str) -> ProcessingRecord:
    """Verify the closed archive, every entry hash, and report/master binding."""

    archive_path = Path(path).expanduser().absolute()
    initial_metadata = _regular_file(
        archive_path,
        label="Full Processing Record",
        maximum_bytes=(
            MAX_MASTER_BYTES
            + MAX_REPORT_BYTES
            + MAX_SUMMARY_BYTES
            + MAX_MANIFEST_BYTES
            + MAX_ZIP_CONTAINER_OVERHEAD
        ),
    )
    try:
        # Keep one descriptor from archive parsing through archive hashing. A
        # concurrent path replacement can no longer make entry evidence come
        # from archive A while ``archive_sha256`` comes from archive B.
        with archive_path.open("rb") as archive_stream:
            opened_metadata = os.fstat(archive_stream.fileno())
            current_metadata = archive_path.lstat()
            if (
                is_reparse_or_symlink(archive_path)
                or not stat.S_ISREG(opened_metadata.st_mode)
                or not stat.S_ISREG(current_metadata.st_mode)
                or (opened_metadata.st_dev, opened_metadata.st_ino)
                != (initial_metadata.st_dev, initial_metadata.st_ino)
                or (opened_metadata.st_dev, opened_metadata.st_ino)
                != (current_metadata.st_dev, current_metadata.st_ino)
            ):
                raise PublicationError(
                    "Full Processing Record changed identity or became a link during verification"
                )
            with zipfile.ZipFile(archive_stream, mode="r", allowZip64=True) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if tuple(names) != ENTRY_NAMES or len(set(names)) != len(names):
                    raise PublicationError(
                        "Processing Record ZIP must contain exactly master.wav, report.json, "
                        "summary.txt, and manifest.json in canonical order"
                    )
                info_by_name = {info.filename: info for info in infos}
                limits = {
                    MASTER_NAME: MAX_MASTER_BYTES,
                    REPORT_NAME: MAX_REPORT_BYTES,
                    SUMMARY_NAME: MAX_SUMMARY_BYTES,
                    MANIFEST_NAME: MAX_MANIFEST_BYTES,
                }
                for info in infos:
                    if (
                        info.compress_type != zipfile.ZIP_STORED
                        or info.is_dir()
                        or not _entry_is_regular(info)
                        or info.file_size < 1
                        or info.file_size > limits[info.filename]
                    ):
                        raise PublicationError(
                            f"Processing Record ZIP entry is unsafe: {info.filename}"
                        )

                manifest_info = info_by_name[MANIFEST_NAME]
                with archive.open(manifest_info, "r") as stream:
                    manifest_raw = stream.read(MAX_MANIFEST_BYTES + 1)
                if len(manifest_raw) != manifest_info.file_size:
                    raise PublicationError("Processing Record manifest size is inconsistent")
                files, content_digest = _validated_manifest(_manifest_object(manifest_raw))

                entry_hashes: dict[str, str] = {}
                total = manifest_info.file_size
                report_raw = b""
                for name in (MASTER_NAME, REPORT_NAME, SUMMARY_NAME):
                    info = info_by_name[name]
                    expected_size = files[name]["size_bytes"]
                    if not isinstance(expected_size, int):  # Defensive after validation.
                        raise PublicationError(f"Processing Record size is invalid: {name}")
                    if info.file_size != expected_size:
                        raise PublicationError(f"Processing Record size mismatch: {name}")
                    with archive.open(info, "r") as stream:
                        if name == REPORT_NAME:
                            report_raw = stream.read(MAX_REPORT_BYTES + 1)
                            digest = hashlib.sha256(report_raw).hexdigest()
                            seen = len(report_raw)
                        else:
                            digest, seen = _hash_stream(stream, expected_size)
                    if seen != expected_size or digest != files[name]["sha256"]:
                        raise PublicationError(f"Processing Record hash mismatch: {name}")
                    entry_hashes[name] = digest
                    total += seen

                with archive.open(info_by_name[MASTER_NAME], "r") as master_stream:
                    _validate_wave_header(master_stream.read(12))

                try:
                    report = load_json_report_bytes(report_raw)
                except Exception as exc:
                    raise PublicationError(f"Processing Record report is invalid: {exc}") from exc
                if report.output.sha256 != entry_hashes[MASTER_NAME]:
                    raise PublicationError(
                        "Processing Record master does not match report.output.sha256"
                    )

            archive_stream.seek(0)
            archive_sha256, archive_size = _hash_stream(archive_stream, initial_metadata.st_size)
            if archive_size != initial_metadata.st_size:
                raise PublicationError("Full Processing Record changed size during verification")
            final_opened = os.fstat(archive_stream.fileno())
            final_path = archive_path.lstat()
            if (
                is_reparse_or_symlink(archive_path)
                or (final_opened.st_dev, final_opened.st_ino)
                != (opened_metadata.st_dev, opened_metadata.st_ino)
                or (final_path.st_dev, final_path.st_ino)
                != (opened_metadata.st_dev, opened_metadata.st_ino)
                or final_opened.st_size != initial_metadata.st_size
                or final_opened.st_mtime_ns != initial_metadata.st_mtime_ns
            ):
                raise PublicationError("Full Processing Record changed during verification")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PublicationError(f"Cannot verify Full Processing Record: {exc}") from exc

    return ProcessingRecord(
        path=archive_path,
        archive_sha256=archive_sha256,
        master_sha256=entry_hashes[MASTER_NAME],
        report_sha256=entry_hashes[REPORT_NAME],
        summary_sha256=entry_hashes[SUMMARY_NAME],
        content_sha256=content_digest,
        total_uncompressed_bytes=total,
    )


def load_json_report_bytes(raw: bytes) -> HawaVoCleanReport:
    """Validate report bytes without extracting the archive to disk."""

    return HawaVoCleanReport.model_validate_json(raw)


__all__ = [
    "ProcessingRecord",
    "create_processing_record",
    "verify_processing_record",
]
