"""Crash-safe publication of one audio/report/summary generation.

Three flat-file renames cannot be atomic as a set. HawaVoClean therefore stores
immutable generations in an adjacent hidden bundle and changes the three public
aliases through one shared ``current`` symlink. See ADR 0005.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from hawavoclean.errors import PublicationError

_BUNDLE_SCHEMA = 1
_OWNER_FILE = "bundle.json"
_CURRENT = "current"
_TRANSACTION = "transaction.json"
_GENERATIONS = "generations"
_FILES = {"audio": "master.wav", "json": "report.json", "txt": "summary.txt"}


@dataclass(frozen=True)
class PublicationPaths:
    """Public aliases and their adjacent private generation bundle."""

    audio: Path
    json: Path
    txt: Path
    bundle: Path
    generations: Path
    current: Path
    transaction: Path
    lock: Path

    @property
    def public(self) -> tuple[Path, Path, Path]:
        return self.audio, self.json, self.txt


def public_output_path(path: Path | str) -> Path:
    """Resolve the parent while preserving the final public output alias."""
    expanded = Path(path).expanduser()
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    return absolute.parent.resolve() / absolute.name


def publication_paths(destination_audio_path: Path | str) -> PublicationPaths:
    """Derive every path owned by one requested public output."""
    audio = public_output_path(destination_audio_path)
    json_path = audio.parent / f"{audio.stem}.hawavoclean.json"
    txt_path = audio.parent / f"{audio.stem}.hawavoclean.txt"
    bundle = audio.parent / f".{audio.name}.hawavoclean"
    return PublicationPaths(
        audio=audio,
        json=json_path,
        txt=txt_path,
        bundle=bundle,
        generations=bundle / _GENERATIONS,
        current=bundle / _CURRENT,
        transaction=bundle / _TRANSACTION,
        lock=audio.parent / f".{audio.name}.hawavoclean.lock",
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def publication_exists(destination_audio_path: Path | str) -> bool:
    """Whether any public or committed private state already owns this destination."""
    paths = publication_paths(destination_audio_path)
    if _lexists(paths.current):
        return True
    # An interrupted first publish may leave some owned aliases dangling before
    # the one authoritative pointer exists. Those are recoverable staging
    # debris, not a committed output, so a normal no-overwrite retry is safe.
    for role, public in _public_roles(paths):
        if _lexists(public) and not _owned_alias(public, _relative_alias_target(paths, role)):
            return True
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsync(path: Path, data: bytes) -> None:
    with open(path, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_fsync(source: Path, destination: Path) -> None:
    with open(source, "rb") as src, open(destination, "xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _replace_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_bytes_fsync(temporary, encoded)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _checkpoint(_name: str) -> None:
    """Fault-injection seam. Production does nothing; tests replace it."""


@contextmanager
def _publication_lock(path: Path) -> Iterator[None]:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PublicationError(f"Cannot acquire safe publication lock {path}: {exc}") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise PublicationError(f"Publication lock is not a regular file: {path}")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _owner_payload(paths: PublicationPaths) -> dict[str, object]:
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "public_names": {
            "audio": paths.audio.name,
            "json": paths.json.name,
            "txt": paths.txt.name,
        },
    }


def _ensure_bundle(paths: PublicationPaths) -> None:
    expected = _owner_payload(paths)
    if _lexists(paths.bundle):
        if paths.bundle.is_symlink() or not paths.bundle.is_dir():
            raise PublicationError(f"Publication bundle is not a real directory: {paths.bundle}")
        owner = paths.bundle / _OWNER_FILE
        try:
            actual = json.loads(owner.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationError(
                f"Publication bundle ownership is unverifiable: {owner}"
            ) from exc
        if actual != expected:
            raise PublicationError(f"Publication bundle belongs to different public paths: {owner}")
        if paths.generations.is_symlink() or not paths.generations.is_dir():
            raise PublicationError(
                f"Publication generations directory is unsafe: {paths.generations}"
            )
        return

    staging = Path(tempfile.mkdtemp(prefix=f"{paths.bundle.name}.init-", dir=paths.audio.parent))
    try:
        staging.chmod(0o700)
        generations = staging / _GENERATIONS
        generations.mkdir(mode=0o700)
        _write_bytes_fsync(
            staging / _OWNER_FILE,
            json.dumps(expected, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        )
        _fsync_directory(generations)
        _fsync_directory(staging)
        with contextlib.suppress(FileExistsError):
            os.rename(staging, paths.bundle)
            # Another serialized process can only win before our flock on
            # filesystems whose locking implementation is not process-wide.
        _fsync_directory(paths.audio.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _ensure_bundle(paths)


def _artifact_record(path: Path, filename: str) -> dict[str, object]:
    return {"filename": filename, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _validate_report_audio(report_path: Path, audio_sha256: str) -> None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        claimed = report["output"]["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PublicationError(f"Published JSON report is invalid: {report_path}") from exc
    if claimed != audio_sha256:
        raise PublicationError(
            "Refusing to publish a report that describes different audio: "
            f"report={claimed!r}, actual={audio_sha256}"
        )


def _verify_generation(generation: Path) -> dict[str, Any]:
    if generation.is_symlink() or not generation.is_dir():
        raise PublicationError(f"Generation is not a real directory: {generation}")
    manifest_path = generation / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"Generation manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _BUNDLE_SCHEMA:
        raise PublicationError(f"Unsupported generation manifest: {manifest_path}")
    if manifest.get("generation_id") != generation.name:
        raise PublicationError(f"Generation ID does not match its directory: {generation}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_FILES):
        raise PublicationError(f"Generation artifact table is invalid: {manifest_path}")
    for role, filename in _FILES.items():
        record = artifacts.get(role)
        if not isinstance(record, dict) or record.get("filename") != filename:
            raise PublicationError(f"Generation {role} record is invalid: {manifest_path}")
        artifact = generation / filename
        if artifact.is_symlink() or not artifact.is_file():
            raise PublicationError(f"Generation artifact is missing or unsafe: {artifact}")
        if record.get("size_bytes") != artifact.stat().st_size:
            raise PublicationError(f"Generation artifact size mismatch: {artifact}")
        if record.get("sha256") != _sha256_file(artifact):
            raise PublicationError(f"Generation artifact digest mismatch: {artifact}")
    payload = {"schema_version": _BUNDLE_SCHEMA, "artifacts": artifacts}
    if hashlib.sha256(_canonical_bytes(payload)).hexdigest() != generation.name:
        raise PublicationError(
            f"Generation manifest content does not derive its ID: {manifest_path}"
        )
    _validate_report_audio(generation / _FILES["json"], str(artifacts["audio"]["sha256"]))
    return manifest


def _prepare_generation(
    paths: PublicationPaths, audio_source: Path, report_bytes: bytes, summary_bytes: bytes
) -> tuple[str, dict[str, Any]]:
    staging = Path(tempfile.mkdtemp(prefix=".generation-", dir=paths.generations))
    try:
        staging.chmod(0o700)
        audio = staging / _FILES["audio"]
        report = staging / _FILES["json"]
        summary = staging / _FILES["txt"]
        _copy_fsync(audio_source, audio)
        _write_bytes_fsync(report, report_bytes)
        _write_bytes_fsync(summary, summary_bytes)
        audio_record = _artifact_record(audio, _FILES["audio"])
        _validate_report_audio(report, str(audio_record["sha256"]))
        artifacts = {
            "audio": audio_record,
            "json": _artifact_record(report, _FILES["json"]),
            "txt": _artifact_record(summary, _FILES["txt"]),
        }
        payload = {"schema_version": _BUNDLE_SCHEMA, "artifacts": artifacts}
        generation_id = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        manifest: dict[str, Any] = {**payload, "generation_id": generation_id}
        _write_bytes_fsync(
            staging / "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        )
        _fsync_directory(staging)
        _checkpoint("generation_files_durable")

        destination = paths.generations / generation_id
        if _lexists(destination):
            _verify_generation(destination)
            shutil.rmtree(staging)
        else:
            os.rename(staging, destination)
            _fsync_directory(paths.generations)
        _checkpoint("generation_committed")
        return generation_id, _verify_generation(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_current_id(paths: PublicationPaths) -> str | None:
    if not _lexists(paths.current):
        return None
    if not paths.current.is_symlink():
        raise PublicationError(f"Publication current pointer is not a symlink: {paths.current}")
    target = PurePath(os.readlink(paths.current))
    if target.is_absolute() or len(target.parts) != 2 or target.parts[0] != _GENERATIONS:
        raise PublicationError(f"Publication current pointer escapes its bundle: {paths.current}")
    generation_id = str(target.parts[1])
    if not re_full_sha256(generation_id):
        raise PublicationError(f"Publication current pointer has an invalid generation: {target}")
    _verify_generation(paths.generations / generation_id)
    return generation_id


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _relative_alias_target(paths: PublicationPaths, role: str) -> str:
    return f"{paths.bundle.name}/{_CURRENT}/{_FILES[role]}"


def _replace_symlink(path: Path, target: str) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.link"
    try:
        os.symlink(target, temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _owned_alias(path: Path, target: str) -> bool:
    return path.is_symlink() and os.readlink(path) == target


def _public_roles(paths: PublicationPaths) -> tuple[tuple[str, Path], ...]:
    return ("audio", paths.audio), ("json", paths.json), ("txt", paths.txt)


def _repair_aliases(paths: PublicationPaths, current_id: str | None) -> None:
    manifest = _verify_generation(paths.generations / current_id) if current_id else None
    for role, public in _public_roles(paths):
        expected = _relative_alias_target(paths, role)
        if _owned_alias(public, expected):
            continue
        if public.is_symlink():
            raise PublicationError(f"Refusing unexpected public output symlink: {public}")
        if _lexists(public):
            if manifest is None:
                raise PublicationError(f"Incomplete legacy output triplet at {public}")
            record = manifest["artifacts"][role]
            if not public.is_file() or _sha256_file(public) != record["sha256"]:
                raise PublicationError(
                    f"Public file differs from the committed generation; refusing overwrite: {public}"
                )
        _replace_symlink(public, expected)
        _checkpoint(f"alias_{role}_replaced")


def _replace_current(paths: PublicationPaths, generation_id: str) -> None:
    target = f"{_GENERATIONS}/{generation_id}"
    temporary = paths.bundle / f".{_CURRENT}.{uuid.uuid4().hex}.link"
    try:
        os.symlink(target, temporary)
        os.replace(temporary, paths.current)
        _checkpoint("pointer_replaced")
        _fsync_directory(paths.bundle)
        _checkpoint("pointer_durable")
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _complete_legacy_migration(paths: PublicationPaths) -> str | None:
    current = _read_current_id(paths)
    if current is not None:
        _repair_aliases(paths, current)
        return current

    states = [(_lexists(path), path.is_symlink()) for path in paths.public]
    if all(not exists for exists, _ in states):
        return None
    if any(is_link for _, is_link in states):
        # A hard kill during first publication can leave only our dangling
        # aliases. With no current pointer, no generation was authoritative.
        for role, public in _public_roles(paths):
            expected = _relative_alias_target(paths, role)
            if _lexists(public) and not _owned_alias(public, expected):
                raise PublicationError(f"Unexpected dangling output alias: {public}")
            with contextlib.suppress(FileNotFoundError):
                public.unlink()
        _fsync_directory(paths.audio.parent)
        return None
    if not all(exists for exists, _ in states):
        raise PublicationError(
            "Incomplete legacy output triplet cannot be migrated safely: "
            + ", ".join(str(path) for path in paths.public)
        )

    report_bytes = paths.json.read_bytes()
    summary_bytes = paths.txt.read_bytes()
    legacy_id, _ = _prepare_generation(paths, paths.audio, report_bytes, summary_bytes)
    _replace_json(
        paths.transaction,
        {
            "schema_version": _BUNDLE_SCHEMA,
            "phase": "migrating_legacy",
            "target_generation": legacy_id,
            "previous_generation": None,
        },
    )
    _replace_current(paths, legacy_id)
    _repair_aliases(paths, legacy_id)
    _replace_json(
        paths.transaction,
        {
            "schema_version": _BUNDLE_SCHEMA,
            "phase": "committed",
            "target_generation": legacy_id,
            "previous_generation": None,
        },
    )
    return legacy_id


def _verify_public(paths: PublicationPaths, expected_generation: str) -> None:
    if _read_current_id(paths) != expected_generation:
        raise PublicationError("Committed generation pointer changed during verification")
    manifest = _verify_generation(paths.generations / expected_generation)
    for role, public in _public_roles(paths):
        target = _relative_alias_target(paths, role)
        if not _owned_alias(public, target):
            raise PublicationError(f"Public output alias is not owned by this bundle: {public}")
        if not public.is_file() or _sha256_file(public) != manifest["artifacts"][role]["sha256"]:
            raise PublicationError(f"Public output does not match committed generation: {public}")


def resolve_committed_publication(
    destination_audio_path: Path | str,
) -> tuple[Path, Path, Path] | None:
    """Resolve one committed generation once for a multi-artifact reader.

    Returns ``None`` for a legacy flat triplet. The returned generation files
    are immutable, so a later overwrite cannot mix the caller's audio and
    report even after the publication lock is released.
    """
    paths = publication_paths(destination_audio_path)
    if not _lexists(paths.bundle):
        return None
    with _publication_lock(paths.lock):
        _ensure_bundle(paths)
        current = _complete_legacy_migration(paths)
        if current is None:
            return None
        _verify_public(paths, current)
        generation = paths.generations / current
        return (
            generation / _FILES["audio"],
            generation / _FILES["json"],
            generation / _FILES["txt"],
        )


def publish_output_generation(
    temp_audio_path: Path,
    destination_audio_path: Path,
    json_report_str: str,
    txt_summary_str: str,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Durably publish a complete generation through one atomic pointer."""
    source = Path(temp_audio_path)
    if source.is_symlink() or not source.is_file():
        raise PublicationError(f"Temporary candidate audio file missing or unsafe: {source}")
    paths = publication_paths(destination_audio_path)
    paths.audio.parent.mkdir(parents=True, exist_ok=True)

    with _publication_lock(paths.lock):
        if not overwrite and (
            _lexists(paths.current)
            or any(_lexists(public) and not public.is_symlink() for public in paths.public)
        ):
            raise PublicationError(
                f"Destination output file already exists and overwrite=False: {paths.audio}"
            )
        try:
            _ensure_bundle(paths)
            prior = _complete_legacy_migration(paths)
        except PublicationError:
            raise
        except Exception as exc:
            raise PublicationError(f"Publication recovery failed: {exc}") from exc
        if prior is not None and not overwrite:
            raise PublicationError(
                f"Destination output file already exists and overwrite=False: {paths.audio}"
            )

        try:
            generation_id, _ = _prepare_generation(
                paths,
                source,
                json_report_str.encode("utf-8"),
                txt_summary_str.encode("utf-8"),
            )
            _replace_json(
                paths.transaction,
                {
                    "schema_version": _BUNDLE_SCHEMA,
                    "phase": "prepared",
                    "target_generation": generation_id,
                    "previous_generation": prior,
                },
            )
        except PublicationError:
            raise
        except Exception as exc:
            raise PublicationError(f"Publication preparation failed: {exc}") from exc
        try:
            _repair_aliases(paths, prior)
            _checkpoint("before_pointer_commit")
            _replace_current(paths, generation_id)
            _replace_json(
                paths.transaction,
                {
                    "schema_version": _BUNDLE_SCHEMA,
                    "phase": "committed",
                    "target_generation": generation_id,
                    "previous_generation": prior,
                },
            )
            _verify_public(paths, generation_id)
        except BaseException as exc:
            # If the one authority transition happened, finish it forward.
            # This also closes the tiny interrupt window after fsync and before
            # the Python assignment to ``committed``.
            recovered = False
            try:
                if _read_current_id(paths) == generation_id:
                    _fsync_directory(paths.bundle)
                    _repair_aliases(paths, generation_id)
                    _replace_json(
                        paths.transaction,
                        {
                            "schema_version": _BUNDLE_SCHEMA,
                            "phase": "committed",
                            "target_generation": generation_id,
                            "previous_generation": prior,
                            "recovered_after": type(exc).__name__,
                        },
                    )
                    _verify_public(paths, generation_id)
                    recovered = True
            except (PublicationError, OSError):
                pass
            if recovered:
                return paths.public
            if prior is None:
                for role, public in _public_roles(paths):
                    expected = _relative_alias_target(paths, role)
                    if _owned_alias(public, expected):
                        with contextlib.suppress(OSError):
                            public.unlink()
                with contextlib.suppress(OSError):
                    _fsync_directory(paths.audio.parent)
            if isinstance(exc, Exception):
                raise PublicationError(f"Committed-generation publish failed: {exc}") from exc
            raise
        return paths.public
