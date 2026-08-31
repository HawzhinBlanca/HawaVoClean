"""Crash-safe publication of one audio/report/summary generation.

Three flat-file renames cannot be atomic as a set. HawaVoClean therefore stores
immutable generations in an adjacent hidden bundle and changes one ``current``
record atomically. Public paths are recoverable exports of that authoritative
generation and are regular, self-contained files on both POSIX and Windows.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from hawavoclean.errors import PublicationError
from hawavoclean.platform_fs import (
    exclusive_file_lock,
    flush_directory,
    is_reparse_or_symlink,
    rename_new_path,
    replace_path,
)

_BUNDLE_SCHEMA = 1
_POINTER_SCHEMA = 1
_OWNER_FILE = "bundle.json"
_CURRENT = "current"
_TRANSACTION = "transaction.json"
_GENERATIONS = "generations"
_FILES = {"audio": "master.wav", "json": "report.json", "txt": "summary.txt"}


@dataclass(frozen=True)
class PublicationPaths:
    """Public exports and their adjacent private generation bundle."""

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
    """Resolve the parent while preserving the final public output filename."""
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
    # A first publish from the legacy symlink implementation may leave owned
    # aliases dangling before its authoritative pointer exists. Those are
    # recoverable debris, not a committed output, so no-overwrite retry is safe.
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
    flush_directory(path)


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
        replace_path(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _checkpoint(_name: str) -> None:
    """Fault-injection seam. Production does nothing; tests replace it."""


@contextmanager
def _publication_lock(path: Path) -> Iterator[None]:
    lock = exclusive_file_lock(path)
    try:
        lock.__enter__()
    except OSError as exc:
        raise PublicationError(f"Cannot acquire safe publication lock {path}: {exc}") from exc
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


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
        if is_reparse_or_symlink(paths.bundle) or not paths.bundle.is_dir():
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
        if is_reparse_or_symlink(paths.generations) or not paths.generations.is_dir():
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
            rename_new_path(staging, paths.bundle)
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
    if is_reparse_or_symlink(generation) or not generation.is_dir():
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
        if is_reparse_or_symlink(artifact) or not artifact.is_file():
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
            rename_new_path(staging, destination)
        _checkpoint("generation_committed")
        return generation_id, _verify_generation(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _read_current_id(paths: PublicationPaths) -> str | None:
    if not _lexists(paths.current):
        return None
    if paths.current.is_symlink():
        target = PurePath(os.readlink(paths.current))
        if target.is_absolute() or len(target.parts) != 2 or target.parts[0] != _GENERATIONS:
            raise PublicationError(
                f"Publication current pointer escapes its bundle: {paths.current}"
            )
        generation_id = str(target.parts[1])
    else:
        if is_reparse_or_symlink(paths.current) or not paths.current.is_file():
            raise PublicationError(f"Publication current pointer is unsafe: {paths.current}")
        try:
            pointer = json.loads(paths.current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublicationError(
                f"Publication current pointer is unreadable: {paths.current}"
            ) from exc
        if (
            not isinstance(pointer, dict)
            or pointer.get("schema_version") != _POINTER_SCHEMA
            or not isinstance(pointer.get("generation_id"), str)
        ):
            raise PublicationError(f"Publication current pointer is invalid: {paths.current}")
        generation_id = str(pointer["generation_id"])
    if not re_full_sha256(generation_id):
        raise PublicationError(
            f"Publication current pointer has an invalid generation: {generation_id!r}"
        )
    _verify_generation(paths.generations / generation_id)
    return generation_id


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _relative_alias_target(paths: PublicationPaths, role: str) -> str:
    return f"{paths.bundle.name}/{_CURRENT}/{_FILES[role]}"


def _replace_regular_file(source: Path, destination: Path) -> None:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        _copy_fsync(source, temporary)
        replace_path(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _owned_alias(path: Path, target: str) -> bool:
    return path.is_symlink() and os.readlink(path) == target


def _public_roles(paths: PublicationPaths) -> tuple[tuple[str, Path], ...]:
    return ("audio", paths.audio), ("json", paths.json), ("txt", paths.txt)


def _matches_known_generation(paths: PublicationPaths, role: str, public: Path) -> bool:
    digest = _sha256_file(public)
    for candidate in paths.generations.iterdir():
        if (
            not re_full_sha256(candidate.name)
            or is_reparse_or_symlink(candidate)
            or not candidate.is_dir()
        ):
            continue
        try:
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
            record = manifest["artifacts"][role]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(record, dict) and record.get("sha256") == digest:
            _verify_generation(candidate)
            return True
    return False


def _repair_public_exports(paths: PublicationPaths, current_id: str) -> None:
    generation = paths.generations / current_id
    manifest = _verify_generation(generation)
    for role, public in _public_roles(paths):
        record = manifest["artifacts"][role]
        expected_alias = _relative_alias_target(paths, role)
        if public.is_symlink():
            if not _owned_alias(public, expected_alias):
                raise PublicationError(f"Refusing unexpected public output symlink: {public}")
        elif _lexists(public):
            if is_reparse_or_symlink(public) or not public.is_file():
                raise PublicationError(f"Public output is not a regular file: {public}")
            if _sha256_file(public) == record["sha256"]:
                continue
            if not _matches_known_generation(paths, role, public):
                raise PublicationError(
                    f"Public file differs from the committed generation; refusing overwrite: {public}"
                )
        _replace_regular_file(generation / _FILES[role], public)
        _checkpoint(f"alias_{role}_replaced")


def _replace_current(paths: PublicationPaths, generation_id: str) -> None:
    temporary = paths.bundle / f".{_CURRENT}.{uuid.uuid4().hex}.tmp"
    try:
        encoded = (
            _canonical_bytes({"schema_version": _POINTER_SCHEMA, "generation_id": generation_id})
            + b"\n"
        )
        _write_bytes_fsync(temporary, encoded)
        replace_path(temporary, paths.current)
        _checkpoint("pointer_replaced")
        _fsync_directory(paths.bundle)
        _checkpoint("pointer_durable")
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def _complete_legacy_migration(paths: PublicationPaths) -> str | None:
    current = _read_current_id(paths)
    if current is not None:
        legacy_pointer = paths.current.is_symlink()
        _repair_public_exports(paths, current)
        if legacy_pointer:
            _replace_current(paths, current)
        return current

    states = [(_lexists(path), path.is_symlink()) for path in paths.public]
    if all(not exists for exists, _ in states):
        return None
    for public in paths.public:
        if _lexists(public) and is_reparse_or_symlink(public) and not public.is_symlink():
            raise PublicationError(f"Unexpected public output reparse point: {public}")
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
    _repair_public_exports(paths, legacy_id)
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
        if is_reparse_or_symlink(public) or not public.is_file():
            raise PublicationError(f"Public output is not a regular file: {public}")
        if _sha256_file(public) != manifest["artifacts"][role]["sha256"]:
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
        try:
            _ensure_bundle(paths)
            current = _complete_legacy_migration(paths)
        except PublicationError:
            raise
        except Exception as exc:
            raise PublicationError(f"Publication recovery failed: {exc}") from exc
        if current is None:
            return None
        _verify_public(paths, current)
        generation = paths.generations / current
        return (
            generation / _FILES["audio"],
            generation / _FILES["json"],
            generation / _FILES["txt"],
        )


def resolve_immutable_publication_generation(
    destination_audio_path: Path | str,
    *,
    audio_sha256: str,
    report_sha256: str | None = None,
    summary_sha256: str | None = None,
) -> tuple[Path, Path, Path] | None:
    """Resolve the immutable generation owned by one completed job.

    ``resolve_committed_publication`` intentionally follows the current
    pointer.  A durable job artifact URL instead needs the generation that the
    job itself reported, even after a later replace-mode job advances that
    pointer. This reader selects by the available job-bound artifact digests
    and refuses an ambiguous audio-only match. It returns only verified
    generation files and never serves recoverable public exports, so a hard
    kill between their independent copy operations cannot expose a mixed
    triplet through the broker.
    """

    expected = {
        "audio": audio_sha256,
        "json": report_sha256,
        "txt": summary_sha256,
    }
    if any(value is not None and not re_full_sha256(value) for value in expected.values()):
        raise PublicationError("Expected publication digest is not a SHA-256 value")
    paths = publication_paths(destination_audio_path)
    if not _lexists(paths.bundle):
        return None
    with _publication_lock(paths.lock):
        try:
            _ensure_bundle(paths)
            # No current pointer means no generation has ever crossed the
            # atomic publication boundary, even if durable-looking staging is
            # present below ``generations``.
            if _read_current_id(paths) is None:
                return None
            matches: list[tuple[Path, Path, Path]] = []
            for generation in paths.generations.iterdir():
                if (
                    not re_full_sha256(generation.name)
                    or is_reparse_or_symlink(generation)
                    or not generation.is_dir()
                ):
                    continue
                manifest = _verify_generation(generation)
                artifacts = manifest["artifacts"]
                if all(
                    value is None or artifacts[role]["sha256"] == value
                    for role, value in expected.items()
                ):
                    matches.append(
                        (
                            generation / _FILES["audio"],
                            generation / _FILES["json"],
                            generation / _FILES["txt"],
                        )
                    )
            # Audio bytes alone are not a unique job identity: two runs may
            # publish the same master with different reports or summaries.
            # Never pick whichever directory iteration happens to see first.
            return matches[0] if len(matches) == 1 else None
        except PublicationError:
            raise
        except Exception as exc:
            raise PublicationError(f"Immutable publication lookup failed: {exc}") from exc


def publish_output_generation(
    temp_audio_path: Path,
    destination_audio_path: Path,
    json_report_str: str,
    txt_summary_str: str,
    overwrite: bool = False,
    clean_only: bool = False,
) -> tuple[Path, Path, Path]:
    """Durably publish output audio.

    If clean_only is True, emits only the destination .wav master file without
    creating public sidecars or hidden bundle directories.
    """
    source = Path(temp_audio_path)
    if is_reparse_or_symlink(source) or not source.is_file():
        raise PublicationError(f"Temporary candidate audio file missing or unsafe: {source}")
    paths = publication_paths(destination_audio_path)
    paths.audio.parent.mkdir(parents=True, exist_ok=True)

    if clean_only:
        if not overwrite and paths.audio.exists():
            raise PublicationError(
                f"Destination output file already exists and overwrite=False: {paths.audio}"
            )
        _replace_regular_file(source, paths.audio)
        for sidecar in (paths.json, paths.txt, paths.lock):
            with contextlib.suppress(OSError):
                if sidecar.is_file():
                    sidecar.unlink()
        with contextlib.suppress(OSError):
            if paths.bundle.is_dir():
                shutil.rmtree(paths.bundle)
        return paths.public

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
            if prior is not None:
                _verify_public(paths, prior)
            _checkpoint("before_pointer_commit")
            _replace_current(paths, generation_id)
            _repair_public_exports(paths, generation_id)
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
                    _repair_public_exports(paths, generation_id)
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
            if isinstance(exc, Exception):
                raise PublicationError(f"Committed-generation publish failed: {exc}") from exc
            raise
        return paths.public
