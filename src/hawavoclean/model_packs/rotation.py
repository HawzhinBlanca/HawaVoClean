"""Root-signed, offline key-rotation metadata for Restore model packs.

The application must supply a pinned Ed25519 root public key explicitly.  No
production root, private material, environment lookup, or network bootstrap is
defined here.  A root-signed generation authorizes the model-pack signing keys
that :class:`~hawavoclean.model_packs.trust.TrustStore` may use.

Low-level verification accepts an explicit ``minimum_generation``.  Production
callers should use :class:`hawavoclean.model_packs.rotation_store.KeyRotationStateStore`,
which verifies and atomically persists the maximum accepted generation beneath
application-managed storage.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from hawavoclean.model_packs.errors import (
    ModelPackCompatibilityError,
    ModelPackManifestError,
    ModelPackRollbackError,
    ModelPackSignatureError,
)
from hawavoclean.model_packs.manifest import canonical_json_bytes
from hawavoclean.model_packs.trust import TrustedKey, TrustStore

ROTATION_SCHEMA_VERSION: Final = 1
ROTATION_PRODUCT: Final = "hawavoclean-model-pack-key-rotation"
ROTATION_ALGORITHM: Final = "Ed25519"
MAX_ROTATION_METADATA_BYTES: Final = 64 * 1024
MAX_ROTATION_SIGNATURE_BYTES: Final = 16 * 1024
MAX_ROTATION_KEYS: Final = 128
MAX_ROTATION_GENERATION: Final = (1 << 63) - 1
MAX_ROTATION_VALIDITY: Final = timedelta(days=732)

_ROTATION_SIGNATURE_DOMAIN: Final = b"HawaVoClean Model Pack Key Rotation Metadata v1\x00"
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")
_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


@dataclass(frozen=True, slots=True)
class PinnedRotationRoot:
    """Application-bundled offline root identity supplied by the caller."""

    key_id: str
    public_key_bytes: bytes

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id, field="pinned root key_id")
        _validate_public_key(self.public_key_bytes, field="pinned root public key")


@dataclass(frozen=True, slots=True)
class RotationKey:
    """One root-authorized model-pack signing key."""

    key_id: str
    public_key_bytes: bytes
    revoked: bool

    def __post_init__(self) -> None:
        _validate_key_id(self.key_id, field="rotation key_id")
        _validate_public_key(self.public_key_bytes, field=f"rotation key {self.key_id!r}")
        if type(self.revoked) is not bool:
            raise _manifest_error("rotation key revoked must be boolean", "invalid_rotation_key")

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": ROTATION_ALGORITHM,
            "key_id": self.key_id,
            "public_key": base64.b64encode(self.public_key_bytes).decode("ascii"),
            "revoked": self.revoked,
        }


@dataclass(frozen=True, slots=True)
class KeyRotationMetadata:
    """Validated canonical v1 key-rotation generation."""

    schema_version: int
    product: str
    root_key_id: str
    generation: int
    issued_at: str
    not_before: str
    expires_at: str
    keys: tuple[RotationKey, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != ROTATION_SCHEMA_VERSION:
            raise _manifest_error(
                "unsupported rotation schema version", "unsupported_rotation_schema"
            )
        if self.product != ROTATION_PRODUCT:
            raise _manifest_error(
                "rotation metadata names the wrong product", "wrong_rotation_product"
            )
        _validate_key_id(self.root_key_id, field="root_key_id")
        if type(self.generation) is not int or not 1 <= self.generation <= MAX_ROTATION_GENERATION:
            raise _manifest_error(
                f"generation must be an integer between 1 and {MAX_ROTATION_GENERATION}",
                "invalid_rotation_generation",
            )
        issued = _parse_utc(self.issued_at, field="issued_at")
        not_before = _parse_utc(self.not_before, field="not_before")
        expires = _parse_utc(self.expires_at, field="expires_at")
        if issued > not_before:
            raise _manifest_error(
                "issued_at must not be after not_before", "invalid_rotation_window"
            )
        if not_before >= expires:
            raise _manifest_error("not_before must precede expires_at", "invalid_rotation_window")
        if expires - not_before > MAX_ROTATION_VALIDITY:
            raise _manifest_error(
                "rotation validity window exceeds 732 days",
                "rotation_window_too_long",
            )
        if not 1 <= len(self.keys) <= MAX_ROTATION_KEYS:
            raise _manifest_error(
                f"rotation metadata must contain 1-{MAX_ROTATION_KEYS} keys",
                "invalid_rotation_key_count",
            )
        key_ids = tuple(key.key_id for key in self.keys)
        if len(set(key_ids)) != len(key_ids):
            raise _manifest_error("rotation key IDs must be unique", "duplicate_rotation_key_id")
        if key_ids != tuple(sorted(key_ids)):
            raise _manifest_error(
                "rotation keys must be sorted by key_id",
                "noncanonical_rotation_key_order",
            )
        public_keys = tuple(key.public_key_bytes for key in self.keys)
        if len(set(public_keys)) != len(public_keys):
            raise _manifest_error(
                "a public key cannot appear under multiple rotation key IDs",
                "duplicate_rotation_public_key",
            )

    @property
    def issued_datetime(self) -> datetime:
        return _parse_utc(self.issued_at, field="issued_at")

    @property
    def not_before_datetime(self) -> datetime:
        return _parse_utc(self.not_before, field="not_before")

    @property
    def expires_datetime(self) -> datetime:
        return _parse_utc(self.expires_at, field="expires_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "product": self.product,
            "root_key_id": self.root_key_id,
            "generation": self.generation,
            "issued_at": self.issued_at,
            "not_before": self.not_before,
            "expires_at": self.expires_at,
            "keys": [key.to_dict() for key in self.keys],
        }


@dataclass(frozen=True, slots=True)
class RotationSignatureEnvelope:
    """Detached root-signature envelope for rotation metadata."""

    schema_version: int
    algorithm: str
    root_key_id: str
    signature: bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "root_key_id": self.root_key_id,
            "signature": base64.b64encode(self.signature).decode("ascii"),
        }


@dataclass(frozen=True, slots=True)
class VerifiedKeyRotation:
    """Authentic, current metadata authorized by the pinned offline root."""

    metadata: KeyRotationMetadata
    metadata_sha256: str
    root_key_id: str

    @property
    def generation(self) -> int:
        return self.metadata.generation

    def to_trust_store(self) -> TrustStore:
        """Convert only root-verified entries to immutable pack-signing trust."""

        return TrustStore(
            [
                TrustedKey(
                    key_id=key.key_id,
                    public_key_bytes=key.public_key_bytes,
                    revoked=key.revoked,
                )
                for key in self.metadata.keys
            ]
        )


def canonical_rotation_metadata_bytes(metadata: KeyRotationMetadata) -> bytes:
    """Return the only accepted serialized representation of metadata."""

    return canonical_json_bytes(metadata.to_dict())


def rotation_signature_message(metadata_bytes: bytes) -> bytes:
    """Domain-separate canonical rotation bytes before root signing."""

    return _ROTATION_SIGNATURE_DOMAIN + metadata_bytes


def rotation_signature_envelope_bytes(*, root_key_id: str, signature: bytes) -> bytes:
    """Build canonical detached root-signature bytes for release tooling."""

    _validate_key_id(root_key_id, field="root_key_id")
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ModelPackManifestError(
            "Ed25519 rotation signatures must be exactly 64 bytes",
            code="invalid_rotation_signature_encoding",
        )
    envelope = RotationSignatureEnvelope(
        schema_version=ROTATION_SCHEMA_VERSION,
        algorithm=ROTATION_ALGORITHM,
        root_key_id=root_key_id,
        signature=signature,
    )
    return canonical_json_bytes(envelope.to_dict())


def parse_rotation_metadata_bytes(
    raw: bytes,
    *,
    require_canonical: bool = True,
) -> KeyRotationMetadata:
    """Parse a bounded, closed-schema rotation generation."""

    if not isinstance(raw, bytes):
        raise _manifest_error("rotation metadata must be bytes", "invalid_rotation_metadata")
    if len(raw) > MAX_ROTATION_METADATA_BYTES:
        raise _manifest_error(
            "rotation metadata exceeds the 64 KiB safety limit",
            "rotation_metadata_too_large",
        )
    value = _load_strict_json(raw, subject="rotation metadata")
    root = _expect_mapping(value, field="rotation metadata")
    _expect_fields(
        root,
        {
            "schema_version",
            "product",
            "root_key_id",
            "generation",
            "issued_at",
            "not_before",
            "expires_at",
            "keys",
        },
        field="rotation metadata",
    )
    keys_value = root["keys"]
    if not isinstance(keys_value, list):
        raise _manifest_error("rotation keys must be an array", "invalid_rotation_keys")
    if not 1 <= len(keys_value) <= MAX_ROTATION_KEYS:
        raise _manifest_error(
            f"rotation metadata must contain 1-{MAX_ROTATION_KEYS} keys",
            "invalid_rotation_key_count",
        )
    keys = tuple(_parse_rotation_key(value, index=index) for index, value in enumerate(keys_value))
    metadata = KeyRotationMetadata(
        schema_version=_expect_int(root["schema_version"], field="schema_version"),
        product=_expect_string(root["product"], field="product", maximum=64),
        root_key_id=_expect_string(root["root_key_id"], field="root_key_id", maximum=128),
        generation=_expect_int(root["generation"], field="generation"),
        issued_at=_expect_string(root["issued_at"], field="issued_at", maximum=20),
        not_before=_expect_string(root["not_before"], field="not_before", maximum=20),
        expires_at=_expect_string(root["expires_at"], field="expires_at", maximum=20),
        keys=keys,
    )
    if require_canonical and raw != canonical_rotation_metadata_bytes(metadata):
        raise _manifest_error(
            "rotation metadata is not canonical JSON",
            "noncanonical_rotation_metadata",
        )
    return metadata


def parse_rotation_signature_bytes(
    raw: bytes,
    *,
    require_canonical: bool = True,
) -> RotationSignatureEnvelope:
    """Parse a bounded, closed-schema root-signature envelope."""

    if not isinstance(raw, bytes):
        raise _manifest_error(
            "rotation signature envelope must be bytes",
            "invalid_rotation_signature_envelope",
        )
    if len(raw) > MAX_ROTATION_SIGNATURE_BYTES:
        raise _manifest_error(
            "rotation signature exceeds the 16 KiB safety limit",
            "rotation_signature_too_large",
        )
    value = _load_strict_json(raw, subject="rotation signature")
    root = _expect_mapping(value, field="rotation signature")
    _expect_fields(
        root,
        {"schema_version", "algorithm", "root_key_id", "signature"},
        field="rotation signature",
    )
    schema_version = _expect_int(root["schema_version"], field="schema_version")
    if schema_version != ROTATION_SCHEMA_VERSION:
        raise _manifest_error(
            "unsupported rotation signature schema version",
            "unsupported_rotation_signature_schema",
        )
    algorithm = _expect_string(root["algorithm"], field="algorithm", maximum=16)
    if algorithm != ROTATION_ALGORITHM:
        raise _manifest_error(
            "rotation signature must use Ed25519",
            "unsupported_rotation_signature_algorithm",
        )
    root_key_id = _expect_string(root["root_key_id"], field="root_key_id", maximum=128)
    _validate_key_id(root_key_id, field="root_key_id")
    encoded = _expect_string(root["signature"], field="signature", maximum=128)
    signature = _decode_base64(
        encoded,
        expected_bytes=64,
        field="rotation signature",
        code="invalid_rotation_signature_encoding",
    )
    envelope = RotationSignatureEnvelope(
        schema_version=schema_version,
        algorithm=algorithm,
        root_key_id=root_key_id,
        signature=signature,
    )
    if require_canonical and raw != canonical_json_bytes(envelope.to_dict()):
        raise _manifest_error(
            "rotation signature is not canonical JSON",
            "noncanonical_rotation_signature",
        )
    return envelope


def verify_key_rotation_metadata(
    metadata_bytes: bytes,
    signature_bytes: bytes,
    pinned_root: PinnedRotationRoot,
    *,
    minimum_generation: int,
    now: datetime | None = None,
) -> VerifiedKeyRotation:
    """Authenticate one generation and enforce time and rollback policy.

    ``minimum_generation`` must come from application-bundled policy or the
    durable maximum generation previously accepted.  Equal generations are
    idempotent; smaller generations are rejected.
    """

    if not isinstance(pinned_root, PinnedRotationRoot):
        raise ModelPackSignatureError(
            "an explicit pinned rotation root is required",
            code="missing_pinned_rotation_root",
        )
    _validate_minimum_generation(minimum_generation)
    metadata = parse_rotation_metadata_bytes(metadata_bytes)
    envelope = parse_rotation_signature_bytes(signature_bytes)
    if metadata.root_key_id != pinned_root.key_id or envelope.root_key_id != pinned_root.key_id:
        raise ModelPackSignatureError(
            "rotation root key ID does not match the pinned application root",
            code="rotation_root_mismatch",
        )
    try:
        verifier = Ed25519PublicKey.from_public_bytes(pinned_root.public_key_bytes)
        verifier.verify(envelope.signature, rotation_signature_message(metadata_bytes))
    except (InvalidSignature, ValueError) as exc:
        raise ModelPackSignatureError(
            "key-rotation Ed25519 root signature verification failed",
            code="invalid_rotation_root_signature",
        ) from exc

    checked_now = _validated_now(now)
    if checked_now < metadata.not_before_datetime:
        raise ModelPackCompatibilityError(
            f"key rotation is not valid before {metadata.not_before}",
            code="rotation_not_yet_valid",
        )
    if checked_now >= metadata.expires_datetime:
        raise ModelPackCompatibilityError(
            f"key rotation expired at {metadata.expires_at}",
            code="rotation_expired",
        )
    if metadata.generation < minimum_generation:
        raise ModelPackRollbackError(
            "refusing key-rotation rollback from floor "
            f"{minimum_generation} to generation {metadata.generation}",
            code="rotation_rollback_rejected",
        )
    return VerifiedKeyRotation(
        metadata=metadata,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        root_key_id=pinned_root.key_id,
    )


def _parse_rotation_key(value: object, *, index: int) -> RotationKey:
    root = _expect_mapping(value, field=f"keys[{index}]")
    _expect_fields(
        root,
        {"algorithm", "key_id", "public_key", "revoked"},
        field=f"keys[{index}]",
    )
    algorithm = _expect_string(root["algorithm"], field=f"keys[{index}].algorithm", maximum=16)
    if algorithm != ROTATION_ALGORITHM:
        raise _manifest_error(
            f"keys[{index}] must use Ed25519",
            "unsupported_rotation_key_algorithm",
        )
    key_id = _expect_string(root["key_id"], field=f"keys[{index}].key_id", maximum=128)
    _validate_key_id(key_id, field=f"keys[{index}].key_id")
    encoded = _expect_string(
        root["public_key"],
        field=f"keys[{index}].public_key",
        maximum=64,
    )
    public_key = _decode_base64(
        encoded,
        expected_bytes=32,
        field=f"keys[{index}].public_key",
        code="invalid_rotation_public_key",
    )
    revoked = root["revoked"]
    if type(revoked) is not bool:
        raise _manifest_error(
            f"keys[{index}].revoked must be boolean",
            "invalid_rotation_key",
        )
    return RotationKey(key_id=key_id, public_key_bytes=public_key, revoked=revoked)


def _decode_base64(value: str, *, expected_bytes: int, field: str, code: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _manifest_error(f"{field} is not canonical base64", code) from exc
    if len(decoded) != expected_bytes or base64.b64encode(decoded).decode("ascii") != value:
        raise _manifest_error(
            f"{field} must encode exactly {expected_bytes} bytes in canonical base64",
            code,
        )
    return decoded


def _load_strict_json(raw: bytes, *, subject: str) -> object:
    class _DuplicateKey(ValueError):
        pass

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite numeric constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateKey(key)
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        return cast(
            object,
            json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            ),
        )
    except _DuplicateKey as exc:
        raise _manifest_error(
            f"{subject} contains a duplicate JSON key", "duplicate_rotation_field"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise _manifest_error(
            f"{subject} is not strict UTF-8 JSON", "invalid_rotation_json"
        ) from exc


def _expect_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _manifest_error(f"{field} must be a JSON object", "invalid_rotation_metadata")
    return cast(dict[str, object], value)


def _expect_fields(root: dict[str, object], expected: set[str], *, field: str) -> None:
    if set(root) != expected:
        missing = sorted(expected - set(root))
        unknown = sorted(set(root) - expected)
        raise _manifest_error(
            f"{field} fields do not match schema; missing={missing}, unknown={unknown}",
            "invalid_rotation_fields",
        )


def _expect_string(value: object, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise _manifest_error(
            f"{field} must be a non-empty string of at most {maximum} characters",
            "invalid_rotation_field",
        )
    return value


def _expect_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise _manifest_error(f"{field} must be an integer", "invalid_rotation_generation")
    return value


def _validate_key_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _KEY_ID_RE.fullmatch(value) is None:
        raise _manifest_error(
            f"{field} has an invalid format",
            "invalid_rotation_key_id",
        )
    return value


def _validate_public_key(value: object, *, field: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ModelPackSignatureError(
            f"{field} must contain exactly 32 raw Ed25519 bytes",
            code="invalid_rotation_public_key",
        )
    return value


def _validate_minimum_generation(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_ROTATION_GENERATION:
        raise ModelPackRollbackError(
            f"minimum_generation must be an integer between 0 and {MAX_ROTATION_GENERATION}",
            code="invalid_rotation_generation",
        )
    return value


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        raise _manifest_error(
            f"{field} must be canonical UTC YYYY-MM-DDTHH:MM:SSZ",
            "invalid_rotation_timestamp",
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise _manifest_error(
            f"{field} is not a real UTC timestamp", "invalid_rotation_timestamp"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise _manifest_error(f"{field} is not canonical UTC", "invalid_rotation_timestamp")
    return parsed


def _validated_now(value: datetime | None) -> datetime:
    checked = datetime.now(UTC) if value is None else value
    if checked.tzinfo is None or checked.utcoffset() is None:
        raise ModelPackCompatibilityError(
            "rotation verification time must be timezone-aware",
            code="invalid_verification_time",
        )
    timestamp = checked.timestamp()
    if not math.isfinite(timestamp):
        raise ModelPackCompatibilityError(
            "rotation verification time must be finite",
            code="invalid_verification_time",
        )
    return checked.astimezone(UTC)


def _manifest_error(message: str, code: str) -> ModelPackManifestError:
    return ModelPackManifestError(message, code=code)


__all__ = [
    "MAX_ROTATION_GENERATION",
    "MAX_ROTATION_KEYS",
    "MAX_ROTATION_METADATA_BYTES",
    "MAX_ROTATION_SIGNATURE_BYTES",
    "MAX_ROTATION_VALIDITY",
    "ROTATION_ALGORITHM",
    "ROTATION_PRODUCT",
    "ROTATION_SCHEMA_VERSION",
    "KeyRotationMetadata",
    "PinnedRotationRoot",
    "RotationKey",
    "RotationSignatureEnvelope",
    "VerifiedKeyRotation",
    "canonical_rotation_metadata_bytes",
    "parse_rotation_metadata_bytes",
    "parse_rotation_signature_bytes",
    "rotation_signature_envelope_bytes",
    "rotation_signature_message",
    "verify_key_rotation_metadata",
]
