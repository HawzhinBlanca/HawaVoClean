"""Offline Ed25519 verification and capability inspection for model packs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from hawavoclean.model_packs.errors import (
    ModelPackCompatibilityError,
    ModelPackError,
    ModelPackManifestError,
    ModelPackPayloadError,
    ModelPackRollbackError,
    ModelPackSignatureError,
)
from hawavoclean.model_packs.manifest import (
    MANIFEST_FILENAME,
    MAX_MANIFEST_BYTES,
    SIGNATURE_FILENAME,
    CoreRole,
    ModelPackManifest,
    SemanticVersion,
    canonical_json_bytes,
    parse_manifest_bytes,
)
from hawavoclean.platform_fs import is_reparse_or_symlink as is_reparse_or_symlink
from hawavoclean.release import VERSION

SIGNATURE_SCHEMA_VERSION: Final = 1
SIGNATURE_ALGORITHM: Final = "Ed25519"
MAX_SIGNATURE_BYTES: Final = 16 * 1024
MAX_LAYOUT_ENTRIES: Final = 1024
_SIGNATURE_DOMAIN: Final = b"HawaVoClean Restore Model Pack Manifest v1\x00"
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")

CapabilityState = Literal["qualified", "experimental", "blocked"]
ExecutionProvider = Literal[
    "CPUExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
]
_CAPABILITY_CORE_ROLES: Final[tuple[CoreRole, ...]] = (
    "model",
    "verifier",
    "preprocessing",
    "corpus",
    "runtime",
)
_PROVIDER_ORDER: Final[tuple[ExecutionProvider, ...]] = (
    "CPUExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
)
_PACK_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class TrustedKey:
    """One offline trust-root entry.

    Revocation is explicit and fail-closed. Entries may be application-bundled
    or converted from metadata already verified under the pinned offline
    rotation root; packs can never add their own trusted keys.
    """

    key_id: str
    public_key_bytes: bytes
    revoked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or _KEY_ID_RE.fullmatch(self.key_id) is None:
            raise ModelPackSignatureError(
                "trusted Ed25519 key_id has an invalid format",
                code="invalid_trusted_key",
            )
        if not isinstance(self.public_key_bytes, bytes) or len(self.public_key_bytes) != 32:
            raise ModelPackSignatureError(
                f"trusted Ed25519 key {self.key_id!r} must contain exactly 32 raw bytes",
                code="invalid_trusted_key",
            )
        if type(self.revoked) is not bool:
            raise ModelPackSignatureError(
                f"trusted Ed25519 key {self.key_id!r} revoked flag must be boolean",
                code="invalid_trusted_key",
            )


class TrustStore:
    """Immutable lookup of application-bundled Ed25519 public keys."""

    def __init__(self, keys: tuple[TrustedKey, ...] | list[TrustedKey]) -> None:
        indexed: dict[str, TrustedKey] = {}
        for key in keys:
            if key.key_id in indexed:
                raise ModelPackSignatureError(
                    f"duplicate trusted key id: {key.key_id}",
                    code="duplicate_trusted_key",
                )
            indexed[key.key_id] = key
        self._keys = indexed

    def verify(self, *, key_id: str, signature: bytes, message: bytes) -> None:
        key = self._keys.get(key_id)
        if key is None:
            raise ModelPackSignatureError(
                f"model pack uses unknown signing key {key_id!r}",
                code="unknown_signing_key",
            )
        if key.revoked:
            raise ModelPackSignatureError(
                f"model pack signing key {key_id!r} is revoked",
                code="revoked_signing_key",
            )
        try:
            verifier = Ed25519PublicKey.from_public_bytes(key.public_key_bytes)
            verifier.verify(signature, message)
        except (InvalidSignature, ValueError) as exc:
            raise ModelPackSignatureError(
                "model-pack Ed25519 signature verification failed",
                code="invalid_signature",
            ) from exc


@dataclass(frozen=True, slots=True)
class SignatureEnvelope:
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
class VerifiedModelPack:
    """An authentic, compatible pack whose entire payload set was hashed."""

    path: Path
    manifest: ModelPackManifest
    manifest_sha256: str
    total_payload_bytes: int


@dataclass(frozen=True, slots=True)
class ModelPackQualificationPolicy:
    """Release-owned authorization for one independently qualified pack.

    Pack signatures prove publisher authenticity, not objective quality.  A
    signed pack therefore cannot promote itself merely by writing
    ``maturity=qualified``.  The core application released alongside it must
    pin the exact canonical manifest identity and the separately qualified
    execution-provider set.  With no policy, production eligibility fails
    closed while installation and integrity inspection remain possible.
    """

    pack_id: str
    version: str
    manifest_sha256: str
    providers: tuple[ExecutionProvider, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pack_id, str) or _PACK_ID_RE.fullmatch(self.pack_id) is None:
            raise ValueError("qualification policy pack_id is invalid")
        if not isinstance(self.version, str):
            raise ValueError("qualification policy version is invalid")
        try:
            SemanticVersion.parse(self.version, field="qualification policy version")
        except ModelPackManifestError as exc:
            raise ValueError("qualification policy version is invalid") from exc
        if (
            not isinstance(self.manifest_sha256, str)
            or _SHA256_RE.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError("qualification policy manifest_sha256 is invalid")
        if not isinstance(self.providers, tuple):
            raise ValueError("qualification policy providers must be a canonical tuple")
        if not self.providers or "CPUExecutionProvider" not in self.providers:
            raise ValueError("qualification policy must include CPUExecutionProvider")
        expected = tuple(provider for provider in _PROVIDER_ORDER if provider in self.providers)
        if self.providers != expected or len(set(self.providers)) != len(self.providers):
            raise ValueError("qualification policy providers are unsupported or not canonical")

    def authorizes(self, verified: VerifiedModelPack) -> bool:
        manifest = verified.manifest
        return (
            manifest.pack_id == self.pack_id
            and manifest.version == self.version
            and verified.manifest_sha256 == self.manifest_sha256
        )


@dataclass(frozen=True, slots=True)
class ModelPackCapability:
    """Safe inspection result suitable for a future capabilities endpoint."""

    status: CapabilityState
    usable: bool
    reason_code: str
    reason: str
    pack_id: str | None = None
    version: str | None = None
    quality_tier: str | None = None
    signing_key_id: str | None = None
    manifest_sha256: str | None = None
    component_hashes: tuple[tuple[str, str], ...] = ()
    qualified_providers: tuple[ExecutionProvider, ...] = ()


def manifest_signature_message(manifest_bytes: bytes) -> bytes:
    """Domain-separate canonical manifest bytes before Ed25519 signing."""
    return _SIGNATURE_DOMAIN + manifest_bytes


def signature_envelope_bytes(*, key_id: str, signature: bytes) -> bytes:
    """Create canonical ``manifest.sig`` bytes for release tooling."""
    envelope = SignatureEnvelope(
        schema_version=SIGNATURE_SCHEMA_VERSION,
        algorithm=SIGNATURE_ALGORITHM,
        key_id=key_id,
        signature=signature,
    )
    return canonical_json_bytes(envelope.to_dict())


def parse_signature_bytes(raw: bytes, *, require_canonical: bool = True) -> SignatureEnvelope:
    """Parse the detached signature with a closed, canonical schema."""
    if len(raw) > MAX_SIGNATURE_BYTES:
        raise ModelPackManifestError(
            "manifest.sig exceeds the 16 KiB safety limit",
            code="signature_too_large",
        )
    value = _load_signature_json(raw)
    if not isinstance(value, dict):
        raise ModelPackManifestError(
            "manifest.sig must be a JSON object",
            code="invalid_signature_envelope",
        )
    expected = {"schema_version", "algorithm", "key_id", "signature"}
    if set(value) != expected:
        raise ModelPackManifestError(
            "manifest.sig fields do not match the v1 schema",
            code="invalid_signature_envelope",
        )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ModelPackManifestError(
            "unsupported manifest.sig schema version",
            code="unsupported_signature_schema",
        )
    if value["algorithm"] != SIGNATURE_ALGORITHM:
        raise ModelPackManifestError(
            "manifest.sig must use Ed25519",
            code="unsupported_signature_algorithm",
        )
    key_id = value["key_id"]
    encoded = value["signature"]
    if not isinstance(key_id, str) or _KEY_ID_RE.fullmatch(key_id) is None:
        raise ModelPackManifestError(
            "manifest.sig key_id is invalid",
            code="invalid_signature_envelope",
        )
    if not isinstance(encoded, str):
        raise ModelPackManifestError(
            "manifest.sig signature must be base64 text",
            code="invalid_signature_encoding",
        )
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ModelPackManifestError(
            "manifest.sig signature is not canonical base64",
            code="invalid_signature_encoding",
        ) from exc
    if len(signature) != 64:
        raise ModelPackManifestError(
            "Ed25519 signatures must be exactly 64 bytes",
            code="invalid_signature_encoding",
        )
    envelope = SignatureEnvelope(
        schema_version=1,
        algorithm=SIGNATURE_ALGORITHM,
        key_id=key_id,
        signature=signature,
    )
    if require_canonical and raw != canonical_json_bytes(envelope.to_dict()):
        raise ModelPackManifestError(
            "manifest.sig is not canonical JSON",
            code="noncanonical_signature",
        )
    return envelope


def verify_model_pack(
    pack_path: Path | str,
    trust_store: TrustStore,
    *,
    runtime_version: str = VERSION,
    minimum_version: str | None = None,
    now: datetime | None = None,
) -> VerifiedModelPack:
    """Verify signature, compatibility, layout, size, and SHA-256 for every payload.

    The signature is checked before potentially large payloads are read. Pack
    directories and every file/path component must be real (not symlinks), and
    undeclared files are rejected so no unauthenticated bytes hitchhike into an
    application-managed installation.
    """
    root = Path(pack_path)
    _require_real_directory(root, subject="model-pack root")
    manifest_raw = _read_small_regular(root / MANIFEST_FILENAME, MAX_MANIFEST_BYTES)
    signature_raw = _read_small_regular(root / SIGNATURE_FILENAME, MAX_SIGNATURE_BYTES)
    manifest = parse_manifest_bytes(manifest_raw)
    envelope = parse_signature_bytes(signature_raw)
    if envelope.key_id != manifest.signing_key_id:
        raise ModelPackSignatureError(
            "manifest.sig key_id does not match signed manifest signing_key_id",
            code="signing_key_mismatch",
        )
    trust_store.verify(
        key_id=manifest.signing_key_id,
        signature=envelope.signature,
        message=manifest_signature_message(manifest_raw),
    )

    checked_now = _validated_now(now)
    if checked_now < manifest.not_before_datetime:
        raise ModelPackCompatibilityError(
            f"model pack is not valid before {manifest.not_before}",
            code="pack_not_yet_valid",
        )
    if checked_now >= manifest.expires_datetime:
        raise ModelPackCompatibilityError(
            f"model pack expired at {manifest.expires_at}",
            code="pack_expired",
        )
    if not manifest.runtime_compatibility.supports(runtime_version):
        raise ModelPackCompatibilityError(
            "model pack is incompatible with runtime "
            f"{runtime_version}; supported range is "
            f"[{manifest.runtime_compatibility.min_version}, "
            f"{manifest.runtime_compatibility.max_version_exclusive})",
            code="incompatible_runtime",
        )
    if minimum_version is not None:
        minimum = SemanticVersion.parse(minimum_version, field="minimum installed version")
        if manifest.semantic_version < minimum:
            raise ModelPackRollbackError(
                f"refusing model-pack rollback from floor {minimum} to {manifest.version}",
                code="rollback_rejected",
            )

    _validate_closed_layout(root, manifest)
    total_bytes = 0
    for payload in manifest.payloads:
        payload_path = _safe_declared_file(root, payload.path)
        actual_size, actual_sha256 = _hash_regular_file(payload_path)
        if actual_size != payload.size_bytes:
            raise ModelPackPayloadError(
                f"payload size mismatch for {payload.path}: "
                f"expected {payload.size_bytes}, got {actual_size}",
                code="payload_size_mismatch",
            )
        if actual_sha256 != payload.sha256:
            raise ModelPackPayloadError(
                f"payload SHA-256 mismatch for {payload.path}",
                code="payload_hash_mismatch",
            )
        total_bytes += actual_size
    return VerifiedModelPack(
        path=root.resolve(),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        total_payload_bytes=total_bytes,
    )


def inspect_model_pack(
    pack_path: Path | str,
    trust_store: TrustStore,
    *,
    runtime_version: str = VERSION,
    minimum_version: str | None = None,
    now: datetime | None = None,
    qualification_policy: ModelPackQualificationPolicy | None = None,
) -> ModelPackCapability:
    """Return a non-throwing, fail-closed capability view of a pack."""
    try:
        verified = verify_model_pack(
            pack_path,
            trust_store,
            runtime_version=runtime_version,
            minimum_version=minimum_version,
            now=now,
        )
    except ModelPackError as exc:
        return ModelPackCapability(
            status="blocked",
            usable=False,
            reason_code=exc.code,
            reason=str(exc),
        )
    manifest = verified.manifest
    hashes = tuple((role, manifest.component(role).sha256) for role in _CAPABILITY_CORE_ROLES)
    if manifest.maturity == "qualified":
        if qualification_policy is None:
            return ModelPackCapability(
                status="blocked",
                usable=False,
                reason_code="qualification_policy_missing",
                reason=(
                    "pack is authentic and self-declares qualified, but this release does not "
                    "pin its exact independently qualified manifest identity"
                ),
                pack_id=manifest.pack_id,
                version=manifest.version,
                quality_tier=manifest.quality_tier,
                signing_key_id=manifest.signing_key_id,
                manifest_sha256=verified.manifest_sha256,
                component_hashes=hashes,
            )
        if not qualification_policy.authorizes(verified):
            return ModelPackCapability(
                status="blocked",
                usable=False,
                reason_code="qualification_identity_mismatch",
                reason=(
                    "pack signature and payloads verify, but its exact manifest identity is not "
                    "the one independently qualified by this application release"
                ),
                pack_id=manifest.pack_id,
                version=manifest.version,
                quality_tier=manifest.quality_tier,
                signing_key_id=manifest.signing_key_id,
                manifest_sha256=verified.manifest_sha256,
                component_hashes=hashes,
            )
        return ModelPackCapability(
            status="qualified",
            usable=True,
            reason_code="release_pinned_qualified_pack",
            reason=(
                "signature, compatibility, layout, payload hashes, release-pinned manifest "
                "identity, and provider qualification policy verified"
            ),
            pack_id=manifest.pack_id,
            version=manifest.version,
            quality_tier=manifest.quality_tier,
            signing_key_id=manifest.signing_key_id,
            manifest_sha256=verified.manifest_sha256,
            component_hashes=hashes,
            qualified_providers=qualification_policy.providers,
        )
    if manifest.maturity == "experimental":
        return ModelPackCapability(
            status="experimental",
            usable=False,
            reason_code="experimental_pack",
            reason="pack is authentic but is not qualified for production Restore",
            pack_id=manifest.pack_id,
            version=manifest.version,
            quality_tier=manifest.quality_tier,
            signing_key_id=manifest.signing_key_id,
            manifest_sha256=verified.manifest_sha256,
            component_hashes=hashes,
        )
    return ModelPackCapability(
        status="blocked",
        usable=False,
        reason_code="publisher_blocked_pack",
        reason="pack is authentic but its signed manifest marks it blocked",
        pack_id=manifest.pack_id,
        version=manifest.version,
        quality_tier=manifest.quality_tier,
        signing_key_id=manifest.signing_key_id,
        manifest_sha256=verified.manifest_sha256,
        component_hashes=hashes,
    )


def _validated_now(value: datetime | None) -> datetime:
    current = value if value is not None else datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ModelPackCompatibilityError(
            "verification time must be timezone-aware",
            code="invalid_verification_time",
        )
    return current.astimezone(UTC)


def _require_real_directory(path: Path, *, subject: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModelPackPayloadError(
            f"cannot inspect {subject} {path}: {exc}",
            code="unsafe_pack_layout",
        ) from exc
    if is_reparse_or_symlink(path) or not stat.S_ISDIR(metadata.st_mode):
        raise ModelPackPayloadError(
            f"{subject} must be a real directory: {path}",
            code="unsafe_pack_layout",
        )


def _safe_declared_file(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            current.lstat()
        except OSError as exc:
            raise ModelPackPayloadError(
                f"declared payload is missing: {relative}",
                code="missing_payload",
            ) from exc
        if is_reparse_or_symlink(current):
            raise ModelPackPayloadError(
                f"declared payload path contains a symlink: {relative}",
                code="unsafe_payload_symlink",
            )
    if not stat.S_ISREG(current.lstat().st_mode):
        raise ModelPackPayloadError(
            f"declared payload is not a regular file: {relative}",
            code="unsafe_payload_type",
        )
    return current


def _validate_closed_layout(root: Path, manifest: ModelPackManifest) -> None:
    expected_files = {MANIFEST_FILENAME, SIGNATURE_FILENAME}
    expected_files.update(payload.path for payload in manifest.payloads)
    expected_directories: set[str] = set()
    for filename in expected_files:
        parent = PurePosixPath(filename).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    seen_files: set[str] = set()
    entries = 0

    def on_error(error: OSError) -> None:
        raise ModelPackPayloadError(
            f"cannot enumerate model-pack layout: {error}",
            code="unsafe_pack_layout",
        ) from error

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=on_error
    ):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(root)
        for name in list(dirnames):
            entries += 1
            child = directory_path / name
            relative = (relative_directory / name).as_posix()
            metadata = child.lstat()
            if is_reparse_or_symlink(child) or not stat.S_ISDIR(metadata.st_mode):
                raise ModelPackPayloadError(
                    f"model-pack directory entry is unsafe: {relative}",
                    code="unsafe_pack_layout",
                )
            if relative not in expected_directories:
                raise ModelPackPayloadError(
                    f"model pack contains undeclared directory: {relative}",
                    code="undeclared_pack_entry",
                )
        for name in filenames:
            entries += 1
            child = directory_path / name
            relative = (relative_directory / name).as_posix()
            metadata = child.lstat()
            if is_reparse_or_symlink(child) or not stat.S_ISREG(metadata.st_mode):
                raise ModelPackPayloadError(
                    f"model-pack file entry is unsafe: {relative}",
                    code="unsafe_pack_layout",
                )
            if relative not in expected_files:
                raise ModelPackPayloadError(
                    f"model pack contains undeclared file: {relative}",
                    code="undeclared_pack_entry",
                )
            seen_files.add(relative)
        if entries > MAX_LAYOUT_ENTRIES:
            raise ModelPackPayloadError(
                f"model pack contains more than {MAX_LAYOUT_ENTRIES} filesystem entries",
                code="pack_layout_too_large",
            )
    missing = sorted(expected_files - seen_files)
    if missing:
        raise ModelPackPayloadError(
            f"model pack is missing declared files: {missing}",
            code="missing_payload",
        )


def _read_small_regular(path: Path, limit: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModelPackPayloadError(
            f"required model-pack metadata is missing: {path.name}",
            code="missing_pack_metadata",
        ) from exc
    if is_reparse_or_symlink(path) or not stat.S_ISREG(metadata.st_mode):
        raise ModelPackPayloadError(
            f"model-pack metadata must be a regular file: {path.name}",
            code="unsafe_pack_layout",
        )
    if metadata.st_size > limit:
        raise ModelPackManifestError(
            f"model-pack metadata exceeds its safety limit: {path.name}",
            code="pack_metadata_too_large",
        )
    try:
        with open(path, "rb") as stream:
            data = stream.read(limit + 1)
    except OSError as exc:
        raise ModelPackPayloadError(
            f"cannot read model-pack metadata {path.name}: {exc}",
            code="unreadable_pack_metadata",
        ) from exc
    if len(data) > limit:
        raise ModelPackManifestError(
            f"model-pack metadata exceeds its safety limit: {path.name}",
            code="pack_metadata_too_large",
        )
    return data


def _hash_regular_file(path: Path) -> tuple[int, str]:
    if is_reparse_or_symlink(path):
        raise ModelPackPayloadError(
            f"model-pack payload is a symlink or reparse point: {path}",
            code="unsafe_payload_symlink",
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ModelPackPayloadError(
            f"cannot open model-pack payload {path.name}: {exc}",
            code="unreadable_payload",
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ModelPackPayloadError(
                f"model-pack payload is not a regular file: {path}",
                code="unsafe_payload_type",
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return metadata.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_signature_json(raw: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ModelPackManifestError(
            f"manifest.sig is not strict UTF-8 JSON: {exc}",
            code="invalid_signature_envelope",
        ) from exc
