"""Durable authority for root-authorized model-pack signing keys.

The store deliberately contains no trust root. A caller must provide the
application-bundled :class:`PinnedRotationRoot` whenever persisted authority is
used. The exact canonical metadata and detached signature are retained so a
restarted process can reconstruct the effective key set and verify it against
the pinned root, committed generation, and committed digest.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

from hawavoclean.model_packs.errors import (
    ModelPackError,
    ModelPackInstallError,
    ModelPackRollbackError,
    ModelPackSignatureError,
)
from hawavoclean.model_packs.manifest import canonical_json_bytes
from hawavoclean.model_packs.rotation import (
    MAX_ROTATION_GENERATION,
    MAX_ROTATION_METADATA_BYTES,
    MAX_ROTATION_SIGNATURE_BYTES,
    PinnedRotationRoot,
    VerifiedKeyRotation,
    parse_rotation_metadata_bytes,
    parse_rotation_signature_bytes,
    verify_key_rotation_metadata,
)
from hawavoclean.platform_fs import (
    exclusive_file_lock,
    flush_directory,
    is_reparse_or_symlink,
    replace_path,
)

_STATE_SCHEMA_VERSION: Final = 2
_LEGACY_STATE_SCHEMA_VERSION: Final = 1
_STATE_FILENAME: Final = "key-rotation-state.json"
# Rotation and pack publication intentionally share this lock. A signing-key
# revocation therefore cannot race a pack verification/activation boundary.
_LOCK_FILENAME: Final = ".install.lock"
_MAX_STATE_BYTES: Final = 128 * 1024
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class KeyRotationState:
    """The highest authentic key-rotation generation and its signed material."""

    root_key_id: str
    highest_generation: int
    metadata_sha256: str
    metadata_bytes: bytes | None = None
    signature_bytes: bytes | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "highest_generation": self.highest_generation,
            "metadata_sha256": self.metadata_sha256,
            "root_key_id": self.root_key_id,
        }

    @property
    def has_verification_material(self) -> bool:
        """Whether this state can reconstruct root-verified effective trust."""

        return self.metadata_bytes is not None and self.signature_bytes is not None


class KeyRotationStateStore:
    """Application-managed, cross-process-safe key-rotation authority state."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def application_default(cls) -> KeyRotationStateStore:
        """Use HawaVoClean's platform application-data model-pack directory."""
        from hawavoclean.paths import app_data_root

        return cls(app_data_root() / "model-packs")

    def current(self) -> KeyRotationState | None:
        """Read the durable summary, failing closed on unsafe or corrupt state."""
        self._ensure_root()
        try:
            with exclusive_file_lock(self.root / _LOCK_FILENAME):
                return self._load_state()
        except ModelPackInstallError:
            raise
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot read key-rotation state: {exc}",
                code="rotation_state_read_failed",
            ) from exc

    def load_verified(
        self,
        pinned_root: PinnedRotationRoot,
        *,
        now: datetime | None = None,
    ) -> VerifiedKeyRotation | None:
        """Rebuild effective trust from durable root-verified material.

        Legacy schema-v1 state retained only a digest and generation. It is
        intentionally unusable until the same root-signed generation is
        re-committed, which atomically upgrades the state to schema v2.
        """

        if not isinstance(pinned_root, PinnedRotationRoot):
            raise ModelPackSignatureError(
                "an explicit pinned rotation root is required",
                code="missing_pinned_rotation_root",
            )
        self._ensure_root()
        try:
            with exclusive_file_lock(self.root / _LOCK_FILENAME):
                state = self._load_state()
                if state is None:
                    return None
                return self._verify_state_material(state, pinned_root, now=now)
        except (ModelPackInstallError, ModelPackRollbackError):
            raise
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot read verified key-rotation authority: {exc}",
                code="rotation_state_read_failed",
            ) from exc

    def verify_and_commit(
        self,
        metadata_bytes: bytes,
        signature_bytes: bytes,
        pinned_root: PinnedRotationRoot,
        *,
        now: datetime | None = None,
    ) -> VerifiedKeyRotation:
        """Verify root-signed metadata and atomically advance its generation.

        Verification happens while holding the same inter-process lock used to
        read and replace the floor.  Equal generations are idempotent only when
        their canonical metadata digest is identical; authentic equivocation at
        one generation is rejected rather than switching trust nondeterministically.
        """
        if not isinstance(pinned_root, PinnedRotationRoot):
            raise ModelPackSignatureError(
                "an explicit pinned rotation root is required",
                code="missing_pinned_rotation_root",
            )
        self._ensure_root()
        try:
            with exclusive_file_lock(self.root / _LOCK_FILENAME):
                return self._verify_and_commit_locked(
                    metadata_bytes,
                    signature_bytes,
                    pinned_root,
                    now=now,
                )
        except (ModelPackInstallError, ModelPackRollbackError):
            raise
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot commit key-rotation state: {exc}",
                code="rotation_state_commit_failed",
            ) from exc

    def _verify_and_commit_locked(
        self,
        metadata_bytes: bytes,
        signature_bytes: bytes,
        pinned_root: PinnedRotationRoot,
        *,
        now: datetime | None,
    ) -> VerifiedKeyRotation:
        """Locked implementation shared with :class:`ModelPackStore`."""

        state = self._load_state()
        if state is not None:
            if state.root_key_id != pinned_root.key_id:
                raise ModelPackRollbackError(
                    "persisted key-rotation root does not match the pinned application root",
                    code="rotation_state_root_mismatch",
                )
            if state.has_verification_material:
                # Authenticate the already-committed bytes without requiring an
                # expired generation to remain current. This detects state-file
                # signature substitution before any update or migration.
                assert state.metadata_bytes is not None
                persisted = parse_rotation_metadata_bytes(state.metadata_bytes)
                self._verify_state_material(
                    state,
                    pinned_root,
                    now=persisted.not_before_datetime,
                )

        floor = state.highest_generation if state is not None else 0
        verified = verify_key_rotation_metadata(
            metadata_bytes,
            signature_bytes,
            pinned_root,
            minimum_generation=floor,
            now=now,
        )
        if state is not None and verified.generation == state.highest_generation:
            if verified.metadata_sha256 != state.metadata_sha256:
                raise ModelPackRollbackError(
                    "different root-signed metadata reuses the committed rotation generation",
                    code="rotation_generation_collision",
                )
            if state.has_verification_material:
                if (
                    state.metadata_bytes != metadata_bytes
                    or state.signature_bytes != signature_bytes
                ):
                    raise ModelPackRollbackError(
                        "committed rotation material changed at the same generation",
                        code="rotation_generation_collision",
                    )
                return verified

        committed = KeyRotationState(
            root_key_id=verified.root_key_id,
            highest_generation=verified.generation,
            metadata_sha256=verified.metadata_sha256,
            metadata_bytes=metadata_bytes,
            signature_bytes=signature_bytes,
        )
        self._write_state(committed)
        return verified

    def _verify_state_material(
        self,
        state: KeyRotationState,
        pinned_root: PinnedRotationRoot,
        *,
        now: datetime | None,
    ) -> VerifiedKeyRotation:
        if state.root_key_id != pinned_root.key_id:
            raise ModelPackRollbackError(
                "persisted key-rotation root does not match the pinned application root",
                code="rotation_state_root_mismatch",
            )
        if not state.has_verification_material:
            raise ModelPackInstallError(
                "legacy key-rotation state must be re-committed under the pinned root",
                code="rotation_state_upgrade_required",
            )
        assert state.metadata_bytes is not None
        assert state.signature_bytes is not None
        verified = verify_key_rotation_metadata(
            state.metadata_bytes,
            state.signature_bytes,
            pinned_root,
            minimum_generation=state.highest_generation,
            now=now,
        )
        if (
            verified.generation != state.highest_generation
            or verified.metadata_sha256 != state.metadata_sha256
            or verified.root_key_id != state.root_key_id
        ):
            raise ModelPackInstallError(
                "persisted key-rotation material does not match its committed identity",
                code="rotation_state_identity_mismatch",
            )
        return verified

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            metadata = self.root.lstat()
        except OSError as exc:
            raise ModelPackInstallError(
                f"cannot create key-rotation state directory {self.root}: {exc}",
                code="unwritable_rotation_state",
            ) from exc
        if (
            is_reparse_or_symlink(self.root)
            or not stat.S_ISDIR(metadata.st_mode)
            or not _trusted_metadata(metadata)
        ):
            raise ModelPackInstallError(
                "key-rotation state directory is not a safely owned real directory",
                code="unsafe_rotation_state",
            )

    def _load_state(self) -> KeyRotationState | None:
        path = self.root / _STATE_FILENAME
        if not os.path.lexists(path):
            return None
        try:
            metadata = path.lstat()
            if (
                is_reparse_or_symlink(path)
                or not stat.S_ISREG(metadata.st_mode)
                or not _trusted_metadata(metadata)
                or metadata.st_size > _MAX_STATE_BYTES
            ):
                raise ValueError("state file is unsafe or exceeds its size limit")
            raw = path.read_bytes()
            value = _strict_json(raw)
            state = _parse_state(value)
            expected = _canonical_state_bytes(state)
            if raw != expected:
                raise ValueError("state is not canonical JSON")
            return state
        except (
            ModelPackError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise ModelPackInstallError(
                f"key-rotation state is corrupt: {exc}",
                code="corrupt_rotation_state",
            ) from exc

    def _write_state(self, state: KeyRotationState) -> None:
        if not state.has_verification_material:
            raise ModelPackInstallError(
                "new key-rotation state requires signed verification material",
                code="rotation_state_material_missing",
            )
        payload = _canonical_state_bytes(state)
        temporary = self.root / f".{_STATE_FILENAME}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temporary, "xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            with contextlib.suppress(OSError):
                temporary.chmod(0o600)
            _rotation_checkpoint("temporary_flushed")
            replace_path(temporary, self.root / _STATE_FILENAME)
            _rotation_checkpoint("state_replaced")
            flush_directory(self.root)
            _rotation_checkpoint("directory_flushed")
        finally:
            with contextlib.suppress(OSError):
                temporary.unlink()


def _parse_state(value: object) -> KeyRotationState:
    if not isinstance(value, dict) or set(value) != {"schema_version", "state"}:
        raise ValueError("invalid state envelope")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version not in {
        _LEGACY_STATE_SCHEMA_VERSION,
        _STATE_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported state schema")
    raw = value["state"]
    base_fields = {"highest_generation", "metadata_sha256", "root_key_id"}
    expected_fields = (
        base_fields
        if schema_version == _LEGACY_STATE_SCHEMA_VERSION
        else base_fields | {"metadata_base64", "signature_base64"}
    )
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("invalid state record")
    root_key_id = raw["root_key_id"]
    generation = raw["highest_generation"]
    digest = raw["metadata_sha256"]
    if not isinstance(root_key_id, str) or _KEY_ID_RE.fullmatch(root_key_id) is None:
        raise ValueError("invalid state root key ID")
    if type(generation) is not int or not 1 <= generation <= MAX_ROTATION_GENERATION:
        raise ValueError("invalid state generation")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("invalid state metadata digest")
    if schema_version == _LEGACY_STATE_SCHEMA_VERSION:
        return KeyRotationState(root_key_id, generation, digest)

    metadata_bytes = _decode_state_material(
        raw["metadata_base64"],
        maximum=MAX_ROTATION_METADATA_BYTES,
        field="metadata_base64",
    )
    signature_bytes = _decode_state_material(
        raw["signature_base64"],
        maximum=MAX_ROTATION_SIGNATURE_BYTES,
        field="signature_base64",
    )
    if hashlib.sha256(metadata_bytes).hexdigest() != digest:
        raise ValueError("rotation metadata digest does not match committed state")
    metadata = parse_rotation_metadata_bytes(metadata_bytes)
    signature = parse_rotation_signature_bytes(signature_bytes)
    if metadata.root_key_id != root_key_id or signature.root_key_id != root_key_id:
        raise ValueError("rotation material root identity does not match committed state")
    if metadata.generation != generation:
        raise ValueError("rotation metadata generation does not match committed state")
    return KeyRotationState(
        root_key_id,
        generation,
        digest,
        metadata_bytes,
        signature_bytes,
    )


def _canonical_state_bytes(state: KeyRotationState) -> bytes:
    record = state.to_dict()
    schema_version = _LEGACY_STATE_SCHEMA_VERSION
    if state.has_verification_material:
        assert state.metadata_bytes is not None
        assert state.signature_bytes is not None
        schema_version = _STATE_SCHEMA_VERSION
        record.update(
            {
                "metadata_base64": base64.b64encode(state.metadata_bytes).decode("ascii"),
                "signature_base64": base64.b64encode(state.signature_bytes).decode("ascii"),
            }
        )
    return canonical_json_bytes({"schema_version": schema_version, "state": record})


def _decode_state_material(value: object, *, maximum: int, field: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field} is not canonical base64") from exc
    if len(decoded) > maximum:
        raise ValueError(f"{field} exceeds its size limit")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{field} is not canonical base64")
    return decoded


def _strict_json(raw: bytes) -> object:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    return cast(
        object,
        json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        ),
    )


def _trusted_metadata(metadata: os.stat_result) -> bool:
    owner_getter = vars(os).get("getuid")
    if callable(owner_getter):
        return metadata.st_uid == owner_getter() and not (
            metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        )
    return True


def _rotation_checkpoint(_name: str) -> None:
    """Fault-injection seam; production intentionally does nothing."""


__all__ = ["KeyRotationState", "KeyRotationStateStore"]
