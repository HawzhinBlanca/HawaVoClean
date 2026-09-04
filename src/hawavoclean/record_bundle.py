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
``manifest.sig`` (optional)
    Ed25519 detached publisher signature signing the domain-separated manifest.

Creation is streaming and atomic. Verification accepts only this closed,
uncompressed layout, bounds metadata before reading it, hashes every byte,
cross-checks the report's output identity against the master, and verifies
detached Ed25519 publisher signatures offline against trusted keys.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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
SIGNATURE_NAME: Final = "manifest.sig"
UNSIGNED_ENTRY_NAMES: Final[tuple[str, ...]] = (
    MASTER_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    MANIFEST_NAME,
)
SIGNED_ENTRY_NAMES: Final[tuple[str, ...]] = (
    MASTER_NAME,
    REPORT_NAME,
    SUMMARY_NAME,
    MANIFEST_NAME,
    SIGNATURE_NAME,
)
ENTRY_NAMES: Final[tuple[str, ...]] = UNSIGNED_ENTRY_NAMES
MAX_MASTER_BYTES: Final = 8 * 1024 * 1024 * 1024
MAX_REPORT_BYTES: Final = 32 * 1024 * 1024
MAX_SUMMARY_BYTES: Final = 4 * 1024 * 1024
MAX_MANIFEST_BYTES: Final = 1024 * 1024
MAX_SIGNATURE_BYTES: Final = 16 * 1024
MAX_ZIP_CONTAINER_OVERHEAD: Final = 4 * 1024 * 1024
_CHUNK_BYTES: Final = 1024 * 1024
_ZIP_DATE: Final = (1980, 1, 1, 0, 0, 0)

RECORD_SIGNATURE_DOMAIN: Final = b"HawaVoClean Full Processing Record Manifest v1\x00"
_KEY_ID_RE: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")


@dataclass(frozen=True, slots=True)
class RecordTrustedKey:
    """One offline trust-root entry for Full Processing Record verification."""

    key_id: str
    public_key_bytes: bytes
    revoked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or _KEY_ID_RE.fullmatch(self.key_id) is None:
            raise PublicationError(f"Trusted Ed25519 key_id has an invalid format: {self.key_id!r}")
        if not isinstance(self.public_key_bytes, bytes) or len(self.public_key_bytes) != 32:
            raise PublicationError(
                f"Trusted Ed25519 key {self.key_id!r} must contain exactly 32 raw bytes"
            )
        if type(self.revoked) is not bool:
            raise PublicationError(
                f"Trusted Ed25519 key {self.key_id!r} revoked flag must be boolean"
            )

    @classmethod
    def from_hex(cls, key_id: str, public_key_hex: str, revoked: bool = False) -> RecordTrustedKey:
        try:
            raw = bytes.fromhex(public_key_hex.strip())
        except ValueError as exc:
            raise PublicationError(f"Invalid public key hex for {key_id!r}") from exc
        return cls(key_id=key_id, public_key_bytes=raw, revoked=revoked)


class RecordTrustStore:
    """Immutable lookup of offline Ed25519 public keys for record publisher authentication."""

    def __init__(
        self,
        keys: tuple[RecordTrustedKey, ...] | list[RecordTrustedKey] | Sequence[RecordTrustedKey],
    ) -> None:
        indexed: dict[str, RecordTrustedKey] = {}
        for key in keys:
            if key.key_id in indexed:
                raise PublicationError(f"Duplicate trusted key id: {key.key_id}")
            indexed[key.key_id] = key
        self._keys = indexed

    def verify(self, *, key_id: str, signature: bytes, message: bytes) -> None:
        key = self._keys.get(key_id)
        if key is None:
            raise PublicationError(f"Processing Record uses unknown signing key: {key_id!r}")
        if key.revoked:
            raise PublicationError(f"Processing Record signing key is revoked: {key_id!r}")
        try:
            verifier = Ed25519PublicKey.from_public_bytes(key.public_key_bytes)
            verifier.verify(signature, message)
        except (InvalidSignature, ValueError) as exc:
            raise PublicationError(
                "Processing Record Ed25519 signature verification failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class RecordSignatureEnvelope:
    """Detached signature metadata stored in ``manifest.sig``."""

    schema_version: int
    algorithm: str
    key_id: str
    signature: bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "signature": base64.b64encode(self.signature).decode("ascii"),
        }


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
    key_id: str | None = None
    signature_sha256: str | None = None


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


def _parse_signature_envelope(raw: bytes) -> RecordSignatureEnvelope:
    if len(raw) > MAX_SIGNATURE_BYTES:
        raise PublicationError("manifest.sig exceeds the 16 KiB safety limit")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PublicationError(f"manifest.sig is invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PublicationError("manifest.sig must be a JSON object")
    expected = {"schema_version", "algorithm", "key_id", "signature"}
    if set(parsed) != expected:
        raise PublicationError("manifest.sig fields do not match the v1 schema")
    if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
        raise PublicationError("Unsupported manifest.sig schema version")
    if parsed["algorithm"] != "Ed25519":
        raise PublicationError("manifest.sig must use Ed25519")
    key_id = parsed["key_id"]
    if not isinstance(key_id, str) or _KEY_ID_RE.fullmatch(key_id) is None:
        raise PublicationError("manifest.sig key_id is invalid")
    encoded = parsed["signature"]
    if not isinstance(encoded, str):
        raise PublicationError("manifest.sig signature must be base64 text")
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PublicationError("manifest.sig signature is not canonical base64") from exc
    if len(signature) != 64:
        raise PublicationError("Ed25519 signatures must be exactly 64 bytes")
    envelope = RecordSignatureEnvelope(
        schema_version=1,
        algorithm="Ed25519",
        key_id=key_id,
        signature=signature,
    )
    if raw != _canonical_json(envelope.to_dict()):
        raise PublicationError("manifest.sig is not canonical JSON")
    return envelope


def _sign_manifest_bytes(
    manifest_bytes: bytes,
    *,
    key_id: str,
    private_key: Ed25519PrivateKey | bytes,
) -> bytes:
    if not isinstance(key_id, str) or _KEY_ID_RE.fullmatch(key_id) is None:
        raise PublicationError(f"Invalid key_id format: {key_id!r}")
    if isinstance(private_key, bytes):
        if len(private_key) != 32:
            raise PublicationError("Raw Ed25519 private key must be exactly 32 bytes")
        try:
            signer = Ed25519PrivateKey.from_private_bytes(private_key)
        except ValueError as exc:
            raise PublicationError("Invalid Ed25519 private key bytes") from exc
    elif isinstance(private_key, Ed25519PrivateKey):
        signer = private_key
    else:
        raise PublicationError("private_key must be Ed25519PrivateKey or 32 raw bytes")

    message = RECORD_SIGNATURE_DOMAIN + manifest_bytes
    signature = signer.sign(message)
    envelope = RecordSignatureEnvelope(
        schema_version=1,
        algorithm="Ed25519",
        key_id=key_id,
        signature=signature,
    )
    return _canonical_json(envelope.to_dict())


def create_processing_record(
    *,
    master_path: Path | str,
    report_path: Path | str,
    summary_path: Path | str,
    destination: Path | str,
    overwrite: bool = False,
    signing_key_id: str | None = None,
    signing_private_key: Ed25519PrivateKey | bytes | None = None,
    trust_store: RecordTrustStore | None = None,
) -> ProcessingRecord:
    """Create one deterministic, portable archive and publish it atomically."""

    master = Path(master_path).expanduser().absolute()
    report = Path(report_path).expanduser().absolute()
    summary = Path(summary_path).expanduser().absolute()
    output = Path(destination).expanduser().absolute()
    if output.suffix.lower() != ".zip":
        raise PublicationError("Full Processing Record destination must end in .zip")

    if (signing_key_id is None) != (signing_private_key is None):
        raise PublicationError(
            "Both signing_key_id and signing_private_key must be provided to sign a Processing Record"
        )

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
                if signing_key_id is not None and signing_private_key is not None:
                    sig_bytes = _sign_manifest_bytes(
                        manifest_bytes,
                        key_id=signing_key_id,
                        private_key=signing_private_key,
                    )
                    with archive.open(
                        _zip_info(SIGNATURE_NAME), "w", force_zip64=True
                    ) as sig_stream:
                        sig_stream.write(sig_bytes)
            with temp_path.open("r+b") as stream:
                os.fsync(stream.fileno())

            if trust_store is not None:
                verified = verify_processing_record(temp_path, trust_store=trust_store)
            else:
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


def sign_processing_record(
    archive_path: Path | str,
    *,
    key_id: str,
    private_key: Ed25519PrivateKey | bytes,
    destination: Path | str | None = None,
    overwrite: bool = False,
    trust_store: RecordTrustStore | None = None,
) -> ProcessingRecord:
    """Sign an existing Full Processing Record with an Ed25519 publisher key."""

    source = Path(archive_path).expanduser().absolute()
    output = Path(destination).expanduser().absolute() if destination is not None else source
    if output != source and os.path.lexists(output) and not overwrite:
        raise PublicationError(f"Full Processing Record already exists: {output}")

    # Source must verify cleanly before signing
    verify_processing_record(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.hawavoclean-record.lock"
    with exclusive_file_lock(lock_path):
        if output != source and os.path.lexists(output) and not overwrite:
            raise PublicationError(f"Full Processing Record already exists: {output}")
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temp_path = Path(raw_temp)
        try:
            with (
                zipfile.ZipFile(source, mode="r", allowZip64=True) as src_archive,
                zipfile.ZipFile(
                    temp_path,
                    mode="w",
                    compression=zipfile.ZIP_STORED,
                    allowZip64=True,
                    strict_timestamps=True,
                ) as dst_archive,
            ):
                for name in (MASTER_NAME, REPORT_NAME, SUMMARY_NAME, MANIFEST_NAME):
                    content = src_archive.read(name)
                    with dst_archive.open(_zip_info(name), "w", force_zip64=True) as writer:
                        writer.write(content)
                manifest_raw = src_archive.read(MANIFEST_NAME)
                sig_bytes = _sign_manifest_bytes(
                    manifest_raw, key_id=key_id, private_key=private_key
                )
                with dst_archive.open(_zip_info(SIGNATURE_NAME), "w", force_zip64=True) as writer:
                    writer.write(sig_bytes)

            with temp_path.open("r+b") as stream:
                os.fsync(stream.fileno())

            verified = verify_processing_record(
                temp_path, trust_store=trust_store, require_authenticated=False
            )
            if output == source or overwrite:
                replace_path(temp_path, output)
            else:
                rename_new_path(temp_path, output)
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


def verify_processing_record(
    path: Path | str,
    *,
    trust_store: RecordTrustStore | None = None,
    require_authenticated: bool = False,
) -> ProcessingRecord:
    """Verify the closed archive, every entry hash, report/master binding, and publisher signature."""

    archive_path = Path(path).expanduser().absolute()
    initial_metadata = _regular_file(
        archive_path,
        label="Full Processing Record",
        maximum_bytes=(
            MAX_MASTER_BYTES
            + MAX_REPORT_BYTES
            + MAX_SUMMARY_BYTES
            + MAX_MANIFEST_BYTES
            + MAX_SIGNATURE_BYTES
            + MAX_ZIP_CONTAINER_OVERHEAD
        ),
    )
    try:
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
                if tuple(names) not in (UNSIGNED_ENTRY_NAMES, SIGNED_ENTRY_NAMES) or len(
                    set(names)
                ) != len(names):
                    raise PublicationError(
                        "Processing Record ZIP must contain exactly master.wav, report.json, "
                        "summary.txt, and manifest.json in canonical order (with optional manifest.sig)"
                    )
                is_signed = tuple(names) == SIGNED_ENTRY_NAMES
                if require_authenticated and not is_signed:
                    raise PublicationError("Full Processing Record has no publisher signature")

                info_by_name = {info.filename: info for info in infos}
                limits = {
                    MASTER_NAME: MAX_MASTER_BYTES,
                    REPORT_NAME: MAX_REPORT_BYTES,
                    SUMMARY_NAME: MAX_SUMMARY_BYTES,
                    MANIFEST_NAME: MAX_MANIFEST_BYTES,
                    SIGNATURE_NAME: MAX_SIGNATURE_BYTES,
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

                authenticated_publisher = False
                key_id: str | None = None
                signature_sha256: str | None = None

                if is_signed:
                    sig_info = info_by_name[SIGNATURE_NAME]
                    with archive.open(sig_info, "r") as stream:
                        sig_raw = stream.read(MAX_SIGNATURE_BYTES + 1)
                    if len(sig_raw) != sig_info.file_size:
                        raise PublicationError("Processing Record signature size is inconsistent")
                    envelope = _parse_signature_envelope(sig_raw)
                    key_id = envelope.key_id
                    signature_sha256 = hashlib.sha256(sig_raw).hexdigest()
                    total += sig_info.file_size
                    if trust_store is not None:
                        trust_store.verify(
                            key_id=envelope.key_id,
                            signature=envelope.signature,
                            message=RECORD_SIGNATURE_DOMAIN + manifest_raw,
                        )
                        authenticated_publisher = True
                    else:
                        if require_authenticated:
                            raise PublicationError(
                                "No trust store provided to authenticate publisher signature"
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
        authenticated_publisher=authenticated_publisher,
        key_id=key_id,
        signature_sha256=signature_sha256,
    )


def load_json_report_bytes(raw: bytes) -> HawaVoCleanReport:
    """Validate report bytes without extracting the archive to disk."""

    return HawaVoCleanReport.model_validate_json(raw)


__all__ = [
    "ENTRY_NAMES",
    "MANIFEST_NAME",
    "MASTER_NAME",
    "ProcessingRecord",
    "RECORD_SIGNATURE_DOMAIN",
    "REPORT_NAME",
    "RecordSignatureEnvelope",
    "RecordTrustStore",
    "RecordTrustedKey",
    "SIGNED_ENTRY_NAMES",
    "SIGNATURE_NAME",
    "SUMMARY_NAME",
    "UNSIGNED_ENTRY_NAMES",
    "create_processing_record",
    "sign_processing_record",
    "verify_processing_record",
]
