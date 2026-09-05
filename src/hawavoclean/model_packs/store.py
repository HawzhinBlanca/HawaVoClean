"""Crash-safe, rollback-resistant application store for verified model packs."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from hawavoclean.model_packs.errors import (
    ModelPackError,
    ModelPackInstallError,
    ModelPackPayloadError,
    ModelPackSignatureError,
)
from hawavoclean.model_packs.manifest import (
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    ModelPackManifest,
    SemanticVersion,
    canonical_json_bytes,
)
from hawavoclean.model_packs.rotation import PinnedRotationRoot, VerifiedKeyRotation
from hawavoclean.model_packs.rotation_store import KeyRotationStateStore
from hawavoclean.model_packs.trust import (
    ModelPackCapability,
    ModelPackQualificationPolicy,
    TrustStore,
    VerifiedModelPack,
    inspect_model_pack,
    verify_model_pack,
)
from hawavoclean.platform_fs import (
    exclusive_file_lock,
    flush_directory,
    rename_new_path,
    replace_path,
)
from hawavoclean.platform_fs import is_reparse_or_symlink as is_reparse_or_symlink
from hawavoclean.release import VERSION

_STORE_SCHEMA_VERSION: Final = 1
_STATE_FILENAME: Final = "state.json"
_PACKS_DIRECTORY: Final = "packs"
_LOCK_FILENAME: Final = ".install.lock"
_MAX_STATE_BYTES: Final = 1024 * 1024
_PACK_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_PROCESS_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class InstalledModelPack:
    """A complete installed directory and its verified signed identity."""

    path: Path
    manifest: ModelPackManifest
    manifest_sha256: str
    total_payload_bytes: int
    already_installed: bool


@dataclass(frozen=True, slots=True)
class _PackState:
    active_version: str
    highest_version: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "active_version": self.active_version,
            "highest_version": self.highest_version,
            "manifest_sha256": self.manifest_sha256,
        }


class ModelPackStore:
    """Own signed packs beneath one explicitly application-managed root.

    A pinned rotation root selects fail-closed rotation mode: committed signed
    rotation state becomes the sole pack-signing authority. Direct TrustStore
    inputs remain compatible only for stores with no committed rotation state.
    Qualification policies are application-release inputs: installation may
    retain an authentic pack without one, but production capability remains
    blocked unless the active manifest exactly matches its release policy.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        pinned_rotation_root: PinnedRotationRoot | None = None,
        qualification_policies: tuple[ModelPackQualificationPolicy, ...] = (),
    ) -> None:
        self.root = Path(root)
        if pinned_rotation_root is not None and not isinstance(
            pinned_rotation_root, PinnedRotationRoot
        ):
            raise ModelPackSignatureError(
                "pinned_rotation_root must be a PinnedRotationRoot",
                code="missing_pinned_rotation_root",
            )
        self._pinned_rotation_root = pinned_rotation_root
        policies: dict[str, ModelPackQualificationPolicy] = {}
        for policy in qualification_policies:
            if not isinstance(policy, ModelPackQualificationPolicy):
                raise TypeError(
                    "qualification_policies must contain ModelPackQualificationPolicy values"
                )
            if policy.pack_id in policies:
                raise ValueError(
                    f"duplicate qualification policy for model pack {policy.pack_id!r}"
                )
            policies[policy.pack_id] = policy
        self._qualification_policies = policies

    @classmethod
    def application_default(
        cls,
        *,
        pinned_rotation_root: PinnedRotationRoot | None = None,
        qualification_policies: tuple[ModelPackQualificationPolicy, ...] = (),
    ) -> ModelPackStore:
        """Use the platform application-data root selected by HawaVoClean."""
        from hawavoclean.paths import app_data_root

        return cls(
            app_data_root() / "model-packs",
            pinned_rotation_root=pinned_rotation_root,
            qualification_policies=qualification_policies,
        )

    def install(
        self,
        source: Path | str,
        trust_store: TrustStore,
        *,
        runtime_version: str = VERSION,
        now: datetime | None = None,
    ) -> InstalledModelPack:
        """Verify, copy, re-verify, and atomically activate one pack.

        A source is verified once to discover its signed identity and again
        under the store lock with the persisted/on-disk rollback floor. The
        copied staging directory is then verified a third time before a single
        directory rename exposes it. Existing identical versions are
        idempotent; different bytes under the same ``pack_id/version`` fail.
        """
        self._ensure_root()
        with _PROCESS_LOCK, _store_lock(self.root / _LOCK_FILENAME):
            effective_trust = self._effective_trust_store_locked(trust_store, now=now)
            initial = verify_model_pack(
                source,
                effective_trust,
                runtime_version=runtime_version,
                now=now,
            )
            state = self._load_state()
            floor = self._version_floor(initial.manifest.pack_id, state)
            verified_source = verify_model_pack(
                source,
                effective_trust,
                runtime_version=runtime_version,
                minimum_version=floor,
                now=now,
            )
            if verified_source.manifest_sha256 != initial.manifest_sha256:
                raise ModelPackInstallError(
                    "model-pack source changed while installation was starting",
                    code="source_changed_during_install",
                )
            return self._install_locked(
                verified_source,
                effective_trust,
                state=state,
                floor=floor,
                runtime_version=runtime_version,
                now=now,
            )

    def verify_and_commit_key_rotation(
        self,
        metadata_bytes: bytes,
        signature_bytes: bytes,
        pinned_root: PinnedRotationRoot,
        *,
        now: datetime | None = None,
    ) -> VerifiedKeyRotation:
        """Authenticate and durably accept a model-pack signing-key generation."""
        if not isinstance(pinned_root, PinnedRotationRoot):
            raise ModelPackSignatureError(
                "an explicit pinned rotation root is required",
                code="missing_pinned_rotation_root",
            )
        self._ensure_root()
        try:
            with _PROCESS_LOCK, _store_lock(self.root / _LOCK_FILENAME):
                verified = KeyRotationStateStore(self.root)._verify_and_commit_locked(
                    metadata_bytes,
                    signature_bytes,
                    pinned_root,
                    now=now,
                )
                self._pinned_rotation_root = pinned_root
                return verified
        except ModelPackError:
            raise
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot commit key-rotation state: {exc}",
                code="rotation_state_commit_failed",
            ) from exc

    def inspect(
        self,
        pack_id: str,
        trust_store: TrustStore,
        *,
        runtime_version: str = VERSION,
        now: datetime | None = None,
    ) -> ModelPackCapability:
        """Inspect the active installed version without trusting store metadata alone."""
        if _PACK_ID_RE.fullmatch(pack_id) is None:
            return _blocked("invalid_pack_id", "pack_id has an invalid format", pack_id=pack_id)
        try:
            self._ensure_root()
            with _PROCESS_LOCK, _store_lock(self.root / _LOCK_FILENAME):
                effective_trust = self._effective_trust_store_locked(trust_store, now=now)
                state = self._load_state()
                record = state.get(pack_id)
                if record is None:
                    return _blocked(
                        "pack_not_installed",
                        f"model pack {pack_id!r} is not installed",
                        pack_id=pack_id,
                    )
                floor = self._version_floor(pack_id, state)
                if floor is None:
                    return _blocked(
                        "missing_version_floor",
                        "active model pack has no persisted or on-disk version floor",
                        pack_id=pack_id,
                    )
                if SemanticVersion.parse(
                    record.highest_version,
                    field="store highest_version",
                ) != SemanticVersion.parse(floor, field="on-disk version floor"):
                    return _blocked(
                        "store_rollback_state_mismatch",
                        "store state omits a newer installed model-pack version",
                        pack_id=pack_id,
                        version=record.active_version,
                    )
                path = self._pack_version_path(pack_id, record.active_version)
                capability = inspect_model_pack(
                    path,
                    effective_trust,
                    runtime_version=runtime_version,
                    minimum_version=floor,
                    now=now,
                    qualification_policy=self._qualification_policies.get(pack_id),
                )
                if not capability.manifest_sha256:
                    return capability
                if capability.manifest_sha256 != record.manifest_sha256:
                    return _blocked(
                        "store_identity_mismatch",
                        "active pack does not match the store's committed manifest identity",
                        pack_id=pack_id,
                        version=record.active_version,
                    )
                return capability
        except ModelPackError as exc:
            return _blocked(exc.code, str(exc), pack_id=pack_id)

    def capabilities(
        self,
        trust_store: TrustStore,
        *,
        runtime_version: str = VERSION,
        now: datetime | None = None,
    ) -> tuple[ModelPackCapability, ...]:
        """Return deterministic capability records for every active pack."""
        try:
            self._ensure_root()
            with _PROCESS_LOCK, _store_lock(self.root / _LOCK_FILENAME):
                self._effective_trust_store_locked(trust_store, now=now)
                pack_ids = tuple(sorted(self._load_state()))
        except ModelPackError as exc:
            return (_blocked(exc.code, str(exc)),)
        return tuple(
            self.inspect(pack_id, trust_store, runtime_version=runtime_version, now=now)
            for pack_id in pack_ids
        )

    def _effective_trust_store_locked(
        self,
        legacy_trust_store: TrustStore,
        *,
        now: datetime | None,
    ) -> TrustStore:
        """Return committed effective keys, ignoring stale caller authority.

        Direct ``TrustStore`` use is retained only for stores that have never
        committed a root-signed rotation. Once rotation state exists, the
        exact persisted generation is re-verified under the pinned root and is
        the sole signing-key authority.
        """

        rotation_store = KeyRotationStateStore(self.root)
        state = rotation_store._load_state()
        if state is None:
            if self._pinned_rotation_root is not None:
                raise ModelPackInstallError(
                    "pinned rotation mode requires committed key-rotation state",
                    code="rotation_state_required",
                )
            return legacy_trust_store
        pinned_root = self._pinned_rotation_root
        if pinned_root is None:
            raise ModelPackInstallError(
                "this model-pack store has committed rotation state but no pinned root",
                code="rotation_root_required",
            )
        return rotation_store._verify_state_material(
            state,
            pinned_root,
            now=now,
        ).to_trust_store()

    def _install_locked(
        self,
        source: VerifiedModelPack,
        trust_store: TrustStore,
        *,
        state: dict[str, _PackState],
        floor: str | None,
        runtime_version: str,
        now: datetime | None,
    ) -> InstalledModelPack:
        manifest = source.manifest
        destination = self._pack_version_path(manifest.pack_id, manifest.version)
        destination_parent = destination.parent
        _mkdir_owned(destination_parent)

        if _path_exists(destination):
            existing = verify_model_pack(
                destination,
                trust_store,
                runtime_version=runtime_version,
                minimum_version=floor,
                now=now,
            )
            if existing.manifest_sha256 != source.manifest_sha256:
                raise ModelPackInstallError(
                    "a different pack already owns this pack_id/version",
                    code="pack_version_collision",
                )
            self._commit_state(state, existing)
            return _installed(existing, already_installed=True)

        staging = Path(tempfile.mkdtemp(prefix=".install-", dir=self.root))
        try:
            staging.chmod(0o700)
            _checkpoint("staging_created")
            self._copy_pack(source, staging)
            _checkpoint("payloads_copied")
            copied = verify_model_pack(
                staging,
                trust_store,
                runtime_version=runtime_version,
                minimum_version=floor,
                now=now,
            )
            if copied.manifest_sha256 != source.manifest_sha256:
                raise ModelPackInstallError(
                    "copied model pack does not match the verified source manifest",
                    code="copied_pack_identity_mismatch",
                )
            _checkpoint("staging_verified")
            try:
                rename_new_path(staging, destination)
            except OSError as exc:
                if not _path_exists(destination):
                    raise ModelPackInstallError(
                        f"cannot atomically install model pack: {exc}",
                        code="atomic_install_failed",
                    ) from exc
                raced = verify_model_pack(
                    destination,
                    trust_store,
                    runtime_version=runtime_version,
                    minimum_version=floor,
                    now=now,
                )
                if raced.manifest_sha256 != source.manifest_sha256:
                    raise ModelPackInstallError(
                        "concurrent install committed different bytes for this version",
                        code="pack_version_collision",
                    ) from exc
            flush_directory(destination_parent)
            _checkpoint("pack_installed")
            installed = verify_model_pack(
                destination,
                trust_store,
                runtime_version=runtime_version,
                minimum_version=floor,
                now=now,
            )
            self._commit_state(state, installed)
            _checkpoint("state_committed")
            return _installed(installed, already_installed=False)
        except ModelPackError:
            raise
        except Exception as exc:
            raise ModelPackInstallError(
                f"model-pack installation failed: {exc}",
                code="install_failed",
            ) from exc
        finally:
            if _path_exists(staging):
                shutil.rmtree(staging, ignore_errors=True)

    def _copy_pack(self, source: VerifiedModelPack, staging: Path) -> None:
        names = [MANIFEST_FILENAME, SIGNATURE_FILENAME]
        names.extend(payload.path for payload in source.manifest.payloads)
        for relative in names:
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _copy_regular_durable(source.path.joinpath(*PurePosixPath(relative).parts), destination)
        for directory in sorted(
            {path.parent for path in staging.rglob("*") if path.parent != staging},
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)

    def _commit_state(
        self,
        state: dict[str, _PackState],
        verified: VerifiedModelPack,
    ) -> None:
        manifest = verified.manifest
        existing = state.get(manifest.pack_id)
        highest = manifest.semantic_version
        if existing is not None:
            existing_highest = SemanticVersion.parse(
                existing.highest_version,
                field="store highest_version",
            )
            if existing_highest > highest:
                highest = existing_highest
        updated = dict(state)
        updated[manifest.pack_id] = _PackState(
            active_version=manifest.version,
            highest_version=str(highest),
            manifest_sha256=verified.manifest_sha256,
        )
        self._write_state(updated)
        state.clear()
        state.update(updated)

    def _version_floor(self, pack_id: str, state: dict[str, _PackState]) -> str | None:
        versions: list[SemanticVersion] = []
        record = state.get(pack_id)
        if record is not None:
            versions.append(
                SemanticVersion.parse(record.highest_version, field="store highest_version")
            )
        pack_root = self.root / _PACKS_DIRECTORY / pack_id
        if _path_exists(pack_root):
            _require_owned_directory(pack_root)
            try:
                entries = list(pack_root.iterdir())
            except OSError as exc:
                raise ModelPackInstallError(
                    f"cannot enumerate installed versions for {pack_id}: {exc}",
                    code="unreadable_pack_store",
                ) from exc
            for entry in entries:
                _require_owned_directory(entry)
                versions.append(
                    SemanticVersion.parse(entry.name, field="installed pack directory version")
                )
        return str(max(versions)) if versions else None

    def _pack_version_path(self, pack_id: str, version: str) -> Path:
        if _PACK_ID_RE.fullmatch(pack_id) is None:
            raise ModelPackInstallError("invalid pack_id", code="invalid_pack_id")
        canonical_version = str(SemanticVersion.parse(version, field="pack version"))
        return self.root / _PACKS_DIRECTORY / pack_id / canonical_version

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot create model-pack store {self.root}: {exc}",
                code="unwritable_pack_store",
            ) from exc
        _require_owned_directory(self.root)
        _mkdir_owned(self.root / _PACKS_DIRECTORY)

    def _load_state(self) -> dict[str, _PackState]:
        path = self.root / _STATE_FILENAME
        if not _path_exists(path):
            return {}
        try:
            metadata = path.lstat()
            if is_reparse_or_symlink(path) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("state.json is not a regular file")
            if not _trusted_store_metadata(metadata):
                raise ValueError("state.json ownership or write permissions are unsafe")
            if metadata.st_size > _MAX_STATE_BYTES:
                raise ValueError("state.json exceeds the 1 MiB safety limit")
            raw = path.read_bytes()
            value = _strict_json(raw)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ModelPackInstallError(
                f"model-pack store state is corrupt: {exc}",
                code="corrupt_pack_store_state",
            ) from exc
        if not isinstance(value, dict) or set(value) != {"schema_version", "packs"}:
            raise ModelPackInstallError(
                "model-pack store state has an invalid schema",
                code="corrupt_pack_store_state",
            )
        if value["schema_version"] != _STORE_SCHEMA_VERSION or not isinstance(value["packs"], dict):
            raise ModelPackInstallError(
                "model-pack store state has an unsupported schema",
                code="corrupt_pack_store_state",
            )
        result: dict[str, _PackState] = {}
        for pack_id, raw_record in value["packs"].items():
            if not isinstance(pack_id, str) or _PACK_ID_RE.fullmatch(pack_id) is None:
                raise ModelPackInstallError(
                    "model-pack store state contains an unsafe pack id",
                    code="corrupt_pack_store_state",
                )
            if not isinstance(raw_record, dict) or set(raw_record) != {
                "active_version",
                "highest_version",
                "manifest_sha256",
            }:
                raise ModelPackInstallError(
                    f"model-pack store state for {pack_id} is invalid",
                    code="corrupt_pack_store_state",
                )
            active = str(
                SemanticVersion.parse(raw_record["active_version"], field="active_version")
            )
            highest = str(
                SemanticVersion.parse(raw_record["highest_version"], field="highest_version")
            )
            if SemanticVersion.parse(active, field="active_version") > SemanticVersion.parse(
                highest, field="highest_version"
            ):
                raise ModelPackInstallError(
                    f"active version exceeds the rollback floor for {pack_id}",
                    code="corrupt_pack_store_state",
                )
            digest = raw_record["manifest_sha256"]
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ModelPackInstallError(
                    f"model-pack store digest for {pack_id} is invalid",
                    code="corrupt_pack_store_state",
                )
            result[pack_id] = _PackState(active, highest, digest)
        canonical = canonical_json_bytes(
            {
                "schema_version": _STORE_SCHEMA_VERSION,
                "packs": {pack_id: record.to_dict() for pack_id, record in sorted(result.items())},
            }
        )
        if raw != canonical:
            raise ModelPackInstallError(
                "model-pack store state is not canonical",
                code="corrupt_pack_store_state",
            )
        return result

    def _write_state(self, state: dict[str, _PackState]) -> None:
        payload = {
            "schema_version": _STORE_SCHEMA_VERSION,
            "packs": {pack_id: record.to_dict() for pack_id, record in sorted(state.items())},
        }
        temporary = self.root / f".{_STATE_FILENAME}.{uuid.uuid4().hex}.tmp"
        try:
            _write_new_durable(temporary, canonical_json_bytes(payload))
            replace_path(temporary, self.root / _STATE_FILENAME)
            flush_directory(self.root)
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot commit model-pack store state: {exc}",
                code="state_commit_failed",
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()


def _installed(verified: VerifiedModelPack, *, already_installed: bool) -> InstalledModelPack:
    return InstalledModelPack(
        path=verified.path,
        manifest=verified.manifest,
        manifest_sha256=verified.manifest_sha256,
        total_payload_bytes=verified.total_payload_bytes,
        already_installed=already_installed,
    )


def _blocked(
    code: str,
    reason: str,
    *,
    pack_id: str | None = None,
    version: str | None = None,
) -> ModelPackCapability:
    return ModelPackCapability(
        status="blocked",
        usable=False,
        reason_code=code,
        reason=reason,
        pack_id=pack_id,
        version=version,
    )


def _checkpoint(_name: str) -> None:
    """Fault-injection seam; production intentionally does nothing."""


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_owned_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModelPackInstallError(
            f"cannot inspect model-pack store directory {path}: {exc}",
            code="unsafe_pack_store",
        ) from exc
    if is_reparse_or_symlink(path) or not stat.S_ISDIR(metadata.st_mode):
        raise ModelPackInstallError(
            f"model-pack store path must be a real directory: {path}",
            code="unsafe_pack_store",
        )
    if not _trusted_store_metadata(metadata):
        raise ModelPackInstallError(
            f"model-pack store path has unsafe ownership or write permissions: {path}",
            code="unsafe_pack_store",
        )


def _mkdir_owned(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ModelPackInstallError(
            f"cannot create model-pack store directory {path}: {exc}",
            code="unwritable_pack_store",
        ) from exc
    _require_owned_directory(path)


def _copy_regular_durable(source: Path, destination: Path) -> None:
    try:
        metadata = source.lstat()
        if is_reparse_or_symlink(source) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("source is not a regular file")
        with open(source, "rb") as input_stream, open(destination, "xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        with contextlib.suppress(OSError):
            destination.chmod(0o600)
    except (OSError, ValueError) as exc:
        raise ModelPackPayloadError(
            f"cannot safely copy model-pack file {source.name}: {exc}",
            code="pack_copy_failed",
        ) from exc


def _write_new_durable(path: Path, data: bytes) -> None:
    with open(path, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def _fsync_directory(path: Path) -> None:
    """Compatibility wrapper for existing fault tests and call sites."""

    flush_directory(path)


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    try:
        with exclusive_file_lock(path):
            metadata = path.lstat()
            if not _trusted_store_metadata(metadata):
                raise ModelPackInstallError(
                    "model-pack install lock ownership or write permissions are unsafe",
                    code="pack_store_lock_failed",
                )
            yield
    except ModelPackInstallError:
        raise
    except OSError as exc:
        raise ModelPackInstallError(
            f"cannot acquire model-pack install lock: {exc}",
            code="pack_store_lock_failed",
        ) from exc


def _strict_json(raw: bytes) -> object:
    def pairs(pairs_value: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs_value:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _trusted_store_metadata(metadata: os.stat_result) -> bool:
    owner_getter = vars(os).get("getuid")
    if callable(owner_getter):
        getuid = cast(Callable[[], int], owner_getter)
        return metadata.st_uid == getuid() and not (
            metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        )
    return True
