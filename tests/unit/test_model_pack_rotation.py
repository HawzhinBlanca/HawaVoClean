"""Adversarial tests for root-signed model-pack key rotation."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hawavoclean.model_packs import (
    MAX_ROTATION_GENERATION,
    MAX_ROTATION_KEYS,
    MAX_ROTATION_METADATA_BYTES,
    MAX_ROTATION_SIGNATURE_BYTES,
    ROTATION_PRODUCT,
    ModelPackCompatibilityError,
    ModelPackError,
    ModelPackManifestError,
    ModelPackRollbackError,
    ModelPackSignatureError,
    PinnedRotationRoot,
    canonical_rotation_metadata_bytes,
    parse_rotation_metadata_bytes,
    parse_rotation_signature_bytes,
    rotation_signature_envelope_bytes,
    rotation_signature_message,
    verify_key_rotation_metadata,
)
from hawavoclean.model_packs.manifest import canonical_json_bytes

pytestmark = pytest.mark.unit

ROOT_ID = "offline-root-2026"
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _key_record(key_id: str, public_key: bytes, *, revoked: bool = False) -> dict[str, object]:
    return {
        "algorithm": "Ed25519",
        "key_id": key_id,
        "public_key": base64.b64encode(public_key).decode("ascii"),
        "revoked": revoked,
    }


def _metadata_dict(
    active_key: bytes,
    revoked_key: bytes,
    *,
    generation: int = 7,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "product": ROTATION_PRODUCT,
        "root_key_id": ROOT_ID,
        "generation": generation,
        "issued_at": "2026-08-01T00:00:00Z",
        "not_before": "2026-08-15T00:00:00Z",
        "expires_at": "2027-08-15T00:00:00Z",
        "keys": [
            _key_record("pack-active-2026", active_key),
            _key_record("pack-retired-2025", revoked_key, revoked=True),
        ],
    }


def _signed(
    root_key: Ed25519PrivateKey,
    metadata: dict[str, Any],
) -> tuple[bytes, bytes]:
    metadata_bytes = canonical_json_bytes(metadata)
    signature = root_key.sign(rotation_signature_message(metadata_bytes))
    signature_bytes = rotation_signature_envelope_bytes(
        root_key_id=str(metadata["root_key_id"]),
        signature=signature,
    )
    return metadata_bytes, signature_bytes


@pytest.fixture
def root_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def active_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def revoked_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def _root(private_key: Ed25519PrivateKey) -> PinnedRotationRoot:
    return PinnedRotationRoot(ROOT_ID, _public_bytes(private_key))


def test_root_verified_generation_converts_to_trust_store(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    metadata, signature = _signed(
        root_key,
        _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key)),
    )

    verified = verify_key_rotation_metadata(
        metadata,
        signature,
        _root(root_key),
        minimum_generation=6,
        now=NOW,
    )

    assert verified.generation == 7
    assert verified.root_key_id == ROOT_ID
    assert verified.metadata_sha256 == hashlib.sha256(metadata).hexdigest()
    assert canonical_rotation_metadata_bytes(verified.metadata) == metadata

    message = b"model-pack manifest signature message"
    trust_store = verified.to_trust_store()
    trust_store.verify(
        key_id="pack-active-2026",
        signature=active_key.sign(message),
        message=message,
    )
    with pytest.raises(ModelPackSignatureError) as revoked:
        trust_store.verify(
            key_id="pack-retired-2025",
            signature=revoked_key.sign(message),
            message=message,
        )
    assert revoked.value.code == "revoked_signing_key"


def test_domain_separation_is_mandatory(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    metadata = canonical_json_bytes(
        _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key))
    )
    undomained = rotation_signature_envelope_bytes(
        root_key_id=ROOT_ID,
        signature=root_key.sign(metadata),
    )

    with pytest.raises(ModelPackSignatureError) as caught:
        verify_key_rotation_metadata(
            metadata,
            undomained,
            _root(root_key),
            minimum_generation=0,
            now=NOW,
        )
    assert caught.value.code == "invalid_rotation_root_signature"


def test_wrong_or_implicit_root_cannot_authorize_rotation(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, signature = _signed(
        root_key,
        _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key)),
    )
    attacker = Ed25519PrivateKey.generate()
    monkeypatch.setenv(
        "HAWAVOCLEAN_MODEL_PACK_ROOT_PUBLIC_KEY",
        base64.b64encode(_public_bytes(root_key)).decode("ascii"),
    )

    with pytest.raises(ModelPackSignatureError) as wrong:
        verify_key_rotation_metadata(
            metadata,
            signature,
            PinnedRotationRoot(ROOT_ID, _public_bytes(attacker)),
            minimum_generation=0,
            now=NOW,
        )
    assert wrong.value.code == "invalid_rotation_root_signature"

    with pytest.raises(ModelPackSignatureError) as missing:
        verify_key_rotation_metadata(
            metadata,
            signature,
            None,  # type: ignore[arg-type]
            minimum_generation=0,
            now=NOW,
        )
    assert missing.value.code == "missing_pinned_rotation_root"


def test_root_identity_must_match_metadata_and_envelope(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    data = _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key))
    data["root_key_id"] = "other-root"
    metadata, signature = _signed(root_key, data)

    with pytest.raises(ModelPackSignatureError) as caught:
        verify_key_rotation_metadata(
            metadata,
            signature,
            _root(root_key),
            minimum_generation=0,
            now=NOW,
        )
    assert caught.value.code == "rotation_root_mismatch"


def test_generation_floor_rejects_rollback_and_allows_idempotency(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    metadata, signature = _signed(
        root_key,
        _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key), generation=7),
    )

    with pytest.raises(ModelPackRollbackError) as caught:
        verify_key_rotation_metadata(
            metadata,
            signature,
            _root(root_key),
            minimum_generation=8,
            now=NOW,
        )
    assert caught.value.code == "rotation_rollback_rejected"

    verified = verify_key_rotation_metadata(
        metadata,
        signature,
        _root(root_key),
        minimum_generation=7,
        now=NOW,
    )
    assert verified.generation == 7


@pytest.mark.parametrize("floor", [-1, True, MAX_ROTATION_GENERATION + 1])
def test_minimum_generation_is_strictly_bounded(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
    floor: object,
) -> None:
    metadata, signature = _signed(
        root_key,
        _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key)),
    )
    with pytest.raises(ModelPackRollbackError) as caught:
        verify_key_rotation_metadata(
            metadata,
            signature,
            _root(root_key),
            minimum_generation=floor,  # type: ignore[arg-type]
            now=NOW,
        )
    assert caught.value.code == "invalid_rotation_generation"


@pytest.mark.parametrize(
    ("mutate", "checked_at", "error_type", "code"),
    [
        (
            lambda data: data.__setitem__("not_before", "2026-09-01T00:00:00Z"),
            NOW,
            ModelPackCompatibilityError,
            "rotation_not_yet_valid",
        ),
        (
            lambda data: data.__setitem__("expires_at", "2026-08-20T00:00:00Z"),
            NOW,
            ModelPackCompatibilityError,
            "rotation_expired",
        ),
        (
            lambda data: data.__setitem__("issued_at", "2026-08-16T00:00:00Z"),
            NOW,
            ModelPackManifestError,
            "invalid_rotation_window",
        ),
        (
            lambda data: data.__setitem__("expires_at", "2029-01-01T00:00:00Z"),
            NOW,
            ModelPackManifestError,
            "rotation_window_too_long",
        ),
        (
            lambda data: data.__setitem__("generation", 0),
            NOW,
            ModelPackManifestError,
            "invalid_rotation_generation",
        ),
    ],
)
def test_validity_window_and_generation_are_enforced(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
    mutate: Callable[[dict[str, Any]], None],
    checked_at: datetime,
    error_type: type[ModelPackError],
    code: str,
) -> None:
    data = _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key))
    mutate(data)
    metadata, signature = _signed(root_key, data)
    with pytest.raises(error_type) as caught:
        verify_key_rotation_metadata(
            metadata,
            signature,
            _root(root_key),
            minimum_generation=0,
            now=checked_at,
        )
    assert caught.value.code == code


def test_naive_verification_time_is_rejected(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    metadata, signature = _signed(
        root_key,
        _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key)),
    )
    with pytest.raises(ModelPackCompatibilityError) as caught:
        verify_key_rotation_metadata(
            metadata,
            signature,
            _root(root_key),
            minimum_generation=0,
            now=datetime(2026, 8, 27, 12, 0),
        )
    assert caught.value.code == "invalid_verification_time"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda data: data["keys"].append(dict(data["keys"][0])),
            "duplicate_rotation_key_id",
        ),
        (
            lambda data: data["keys"].reverse(),
            "noncanonical_rotation_key_order",
        ),
        (
            lambda data: data["keys"][1].__setitem__("public_key", data["keys"][0]["public_key"]),
            "duplicate_rotation_public_key",
        ),
        (
            lambda data: data["keys"][0].__setitem__("key_id", "bad key id"),
            "invalid_rotation_key_id",
        ),
        (
            lambda data: data["keys"][0].__setitem__("algorithm", "RSA"),
            "unsupported_rotation_key_algorithm",
        ),
        (
            lambda data: data["keys"][0].__setitem__("public_key", "AAAA"),
            "invalid_rotation_public_key",
        ),
        (
            lambda data: data["keys"][0].__setitem__("revoked", 1),
            "invalid_rotation_key",
        ),
    ],
)
def test_key_set_is_closed_unique_sorted_and_typed(
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
    mutate: Callable[[dict[str, Any]], None],
    code: str,
) -> None:
    data = _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key))
    mutate(data)
    with pytest.raises(ModelPackManifestError) as caught:
        parse_rotation_metadata_bytes(canonical_json_bytes(data))
    assert caught.value.code == code


def test_key_count_and_metadata_size_are_bounded(
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    data = _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key))
    data["keys"] = [
        _key_record(f"key-{index:03d}", bytes([index % 251]) * 32)
        for index in range(MAX_ROTATION_KEYS + 1)
    ]
    with pytest.raises(ModelPackManifestError) as count:
        parse_rotation_metadata_bytes(canonical_json_bytes(data))
    assert count.value.code == "invalid_rotation_key_count"

    with pytest.raises(ModelPackManifestError) as size:
        parse_rotation_metadata_bytes(b" " * (MAX_ROTATION_METADATA_BYTES + 1))
    assert size.value.code == "rotation_metadata_too_large"


def test_noncanonical_unknown_and_duplicate_metadata_fields_are_rejected(
    root_key: Ed25519PrivateKey,
    active_key: Ed25519PrivateKey,
    revoked_key: Ed25519PrivateKey,
) -> None:
    data = _metadata_dict(_public_bytes(active_key), _public_bytes(revoked_key))
    canonical = canonical_json_bytes(data)
    pretty = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(ModelPackManifestError) as noncanonical:
        parse_rotation_metadata_bytes(pretty)
    assert noncanonical.value.code == "noncanonical_rotation_metadata"

    data["unexpected"] = True
    with pytest.raises(ModelPackManifestError) as unknown:
        parse_rotation_metadata_bytes(canonical_json_bytes(data))
    assert unknown.value.code == "invalid_rotation_fields"

    duplicate = canonical.replace(b'"generation":7', b'"generation":7,"generation":8', 1)
    with pytest.raises(ModelPackManifestError) as duplicated:
        parse_rotation_metadata_bytes(duplicate)
    assert duplicated.value.code == "duplicate_rotation_field"

    # A root signature over pretty JSON cannot turn it into accepted metadata.
    signature = rotation_signature_envelope_bytes(
        root_key_id=ROOT_ID,
        signature=root_key.sign(rotation_signature_message(pretty)),
    )
    with pytest.raises(ModelPackManifestError):
        verify_key_rotation_metadata(
            pretty,
            signature,
            _root(root_key),
            minimum_generation=0,
            now=NOW,
        )


def test_signature_envelope_is_canonical_bounded_and_strict(
    root_key: Ed25519PrivateKey,
) -> None:
    valid = rotation_signature_envelope_bytes(
        root_key_id=ROOT_ID,
        signature=root_key.sign(b"message"),
    )
    parsed = parse_rotation_signature_bytes(valid)
    assert parsed.root_key_id == ROOT_ID

    pretty = json.dumps(json.loads(valid), indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(ModelPackManifestError) as noncanonical:
        parse_rotation_signature_bytes(pretty)
    assert noncanonical.value.code == "noncanonical_rotation_signature"

    malformed = json.loads(valid)
    malformed["signature"] = "AAAA"
    with pytest.raises(ModelPackManifestError) as bad_encoding:
        parse_rotation_signature_bytes(canonical_json_bytes(malformed))
    assert bad_encoding.value.code == "invalid_rotation_signature_encoding"

    with pytest.raises(ModelPackManifestError) as oversized:
        parse_rotation_signature_bytes(b" " * (MAX_ROTATION_SIGNATURE_BYTES + 1))
    assert oversized.value.code == "rotation_signature_too_large"


def test_pinned_root_has_no_placeholder_or_weak_shape() -> None:
    with pytest.raises(ModelPackSignatureError) as short:
        PinnedRotationRoot(ROOT_ID, b"short")
    assert short.value.code == "invalid_rotation_public_key"

    with pytest.raises(ModelPackManifestError) as bad_id:
        PinnedRotationRoot("bad root id", b"x" * 32)
    assert bad_id.value.code == "invalid_rotation_key_id"
