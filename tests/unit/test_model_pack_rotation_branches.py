"""Branch coverage tests for hawavoclean.model_packs.rotation."""

from __future__ import annotations

from datetime import datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hawavoclean.model_packs.errors import (
    ModelPackCompatibilityError,
    ModelPackManifestError,
    ModelPackRollbackError,
    ModelPackSignatureError,
)
from hawavoclean.model_packs.manifest import canonical_json_bytes
from hawavoclean.model_packs.rotation import (
    KeyRotationMetadata,
    PinnedRotationRoot,
    RotationKey,
    RotationSignatureEnvelope,
    VerifiedKeyRotation,
    parse_rotation_metadata_bytes,
    rotation_signature_envelope_bytes,
    rotation_signature_message,
    verify_key_rotation_metadata,
)


def _gen_key() -> tuple[str, bytes]:
    priv = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization

    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return "key-1", raw


def test_rotation_key_and_envelope_to_dict() -> None:
    kid, raw_key = _gen_key()
    rk = RotationKey(key_id=kid, public_key_bytes=raw_key, revoked=False)
    d = rk.to_dict()
    assert d["key_id"] == kid
    assert d["algorithm"] == "Ed25519"
    assert d["revoked"] is False

    with pytest.raises(ModelPackManifestError, match="revoked must be boolean"):
        RotationKey(key_id=kid, public_key_bytes=raw_key, revoked="not_bool")  # type: ignore[arg-type]

    env = RotationSignatureEnvelope(
        schema_version=1,
        algorithm="Ed25519",
        root_key_id="root-1",
        signature=b"s" * 64,
    )
    env_d = env.to_dict()
    assert env_d["root_key_id"] == "root-1"
    assert env_d["schema_version"] == 1


def test_verified_key_rotation_to_trust_store() -> None:
    kid, raw_key = _gen_key()
    rk = RotationKey(key_id=kid, public_key_bytes=raw_key, revoked=False)
    meta = KeyRotationMetadata(
        schema_version=1,
        product="hawavoclean-model-pack-key-rotation",
        root_key_id="root-1",
        generation=1,
        issued_at="2026-08-01T00:00:00Z",
        not_before="2026-08-01T00:00:00Z",
        expires_at="2027-08-01T00:00:00Z",
        keys=(rk,),
    )
    vkr = VerifiedKeyRotation(
        metadata=meta,
        metadata_sha256="a" * 64,
        root_key_id="root-1",
    )
    assert vkr.generation == 1
    ts = vkr.to_trust_store()
    assert kid in ts._keys
    assert ts._keys[kid].public_key_bytes == raw_key


def test_verify_key_rotation_validation_branches() -> None:
    kid, raw_key = _gen_key()
    root_priv = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization

    root_pub_raw = root_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pinned_root = PinnedRotationRoot(key_id="root-1", public_key_bytes=root_pub_raw)

    rk = RotationKey(key_id=kid, public_key_bytes=raw_key, revoked=False)
    meta = KeyRotationMetadata(
        schema_version=1,
        product="hawavoclean-model-pack-key-rotation",
        root_key_id="root-1",
        generation=2,
        issued_at="2026-08-01T00:00:00Z",
        not_before="2026-08-01T00:00:00Z",
        expires_at="2027-08-01T00:00:00Z",
        keys=(rk,),
    )
    meta_bytes = canonical_json_bytes(meta.to_dict())
    sig = root_priv.sign(rotation_signature_message(meta_bytes))
    sig_bytes = rotation_signature_envelope_bytes(root_key_id="root-1", signature=sig)

    # 1. Invalid pinned_root type
    with pytest.raises(ModelPackSignatureError, match="pinned rotation root is required"):
        verify_key_rotation_metadata(
            meta_bytes,
            sig_bytes,
            "not_a_root",  # type: ignore[arg-type]
            minimum_generation=1,
        )

    # 2. Invalid minimum_generation
    with pytest.raises(ModelPackRollbackError, match="minimum_generation must be an integer"):
        verify_key_rotation_metadata(
            meta_bytes,
            sig_bytes,
            pinned_root,
            minimum_generation=-1,
        )

    # 3. Root key mismatch
    diff_root = PinnedRotationRoot(key_id="other-root", public_key_bytes=root_pub_raw)
    with pytest.raises(ModelPackSignatureError, match="rotation root key ID does not match"):
        verify_key_rotation_metadata(
            meta_bytes,
            sig_bytes,
            diff_root,
            minimum_generation=1,
        )

    # 4. Naive (non-timezone-aware) datetime for now
    with pytest.raises(ModelPackCompatibilityError, match="timezone-aware"):
        verify_key_rotation_metadata(
            meta_bytes,
            sig_bytes,
            pinned_root,
            minimum_generation=1,
            now=datetime(2026, 8, 15, 0, 0),  # naive
        )


def test_parse_rotation_metadata_error_branches() -> None:
    # 1. Not bytes
    with pytest.raises(ModelPackManifestError, match="must be bytes"):
        parse_rotation_metadata_bytes("string_not_bytes")  # type: ignore[arg-type]

    # 2. Not a JSON object (mapping)
    with pytest.raises(ModelPackManifestError, match="must be a JSON object"):
        parse_rotation_metadata_bytes(b"[1, 2, 3]")

    # 3. Keys not a list
    data = {
        "schema_version": 1,
        "product": "hawavoclean-model-pack-key-rotation",
        "root_key_id": "root-1",
        "generation": 1,
        "issued_at": "2026-08-01T00:00:00Z",
        "not_before": "2026-08-01T00:00:00Z",
        "expires_at": "2027-08-01T00:00:00Z",
        "keys": "not_a_list",
    }
    with pytest.raises(ModelPackManifestError, match="rotation keys must be an array"):
        parse_rotation_metadata_bytes(canonical_json_bytes(data), require_canonical=False)


def test_rotation_store_internal_parsers_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    from hawavoclean.model_packs.rotation_store import (
        _decode_state_material,
        _parse_state,
        _rotation_checkpoint,
        _strict_json,
        _trusted_metadata,
    )

    # 1. _rotation_checkpoint does nothing
    _rotation_checkpoint("any_seam")

    # 2. _strict_json duplicate keys
    with pytest.raises(ValueError, match="duplicate key"):
        _strict_json(b'{"a": 1, "a": 2}')

    # 3. _decode_state_material errors
    with pytest.raises(ValueError, match="must be base64 text"):
        _decode_state_material(123, maximum=100, field="test_field")

    with pytest.raises(ValueError, match="not canonical base64"):
        _decode_state_material("not_valid_b64!!!", maximum=100, field="test_field")

    with pytest.raises(ValueError, match="exceeds its size limit"):
        import base64

        huge_b64 = base64.b64encode(b"x" * 200).decode("ascii")
        _decode_state_material(huge_b64, maximum=50, field="test_field")

    with pytest.raises(ValueError, match="not canonical base64"):
        # Valid base64 decoding with non-canonical formatting (e.g. whitespace)
        _decode_state_material("AAAA\n", maximum=100, field="test_field")

    # 4. _parse_state envelope & field validation
    with pytest.raises(ValueError, match="invalid state envelope"):
        _parse_state([])

    with pytest.raises(ValueError, match="invalid state envelope"):
        _parse_state({"wrong": 1})

    with pytest.raises(ValueError, match="unsupported state schema"):
        _parse_state({"schema_version": 99, "state": {}})

    with pytest.raises(ValueError, match="invalid state record"):
        _parse_state({"schema_version": 1, "state": "not_a_dict"})

    with pytest.raises(ValueError, match="invalid state record"):
        _parse_state({"schema_version": 1, "state": {"unknown_field": 1}})

    with pytest.raises(ValueError, match="invalid state root key ID"):
        _parse_state(
            {
                "schema_version": 1,
                "state": {
                    "highest_generation": 1,
                    "metadata_sha256": "a" * 64,
                    "root_key_id": "invalid key id with spaces!",
                },
            }
        )

    with pytest.raises(ValueError, match="invalid state generation"):
        _parse_state(
            {
                "schema_version": 1,
                "state": {
                    "highest_generation": 0,
                    "metadata_sha256": "a" * 64,
                    "root_key_id": "valid-root-1",
                },
            }
        )

    with pytest.raises(ValueError, match="invalid state metadata digest"):
        _parse_state(
            {
                "schema_version": 1,
                "state": {
                    "highest_generation": 1,
                    "metadata_sha256": "not_a_sha256",
                    "root_key_id": "valid-root-1",
                },
            }
        )

    # Schema 2 state mismatches
    valid_meta_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            "product": "hawavoclean-model-pack-key-rotation",
            "root_key_id": "valid-root-1",
            "generation": 1,
            "issued_at": "2026-08-01T00:00:00Z",
            "not_before": "2026-08-01T00:00:00Z",
            "expires_at": "2027-08-01T00:00:00Z",
            "keys": [
                {
                    "algorithm": "Ed25519",
                    "key_id": "pack-key-1",
                    "public_key": base64.b64encode(b"k" * 32).decode("ascii"),
                    "revoked": False,
                }
            ],
        }
    )
    import hashlib

    valid_meta_b64 = base64.b64encode(valid_meta_bytes).decode("ascii")
    valid_meta_digest = hashlib.sha256(valid_meta_bytes).hexdigest()

    valid_sig_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            "algorithm": "Ed25519",
            "root_key_id": "valid-root-1",
            "signature": base64.b64encode(b"s" * 64).decode("ascii"),
        }
    )
    valid_sig_b64 = base64.b64encode(valid_sig_bytes).decode("ascii")

    # 4a. Digest mismatch
    with pytest.raises(ValueError, match="metadata digest does not match"):
        _parse_state(
            {
                "schema_version": 2,
                "state": {
                    "highest_generation": 1,
                    "metadata_sha256": "0" * 64,
                    "root_key_id": "valid-root-1",
                    "metadata_base64": valid_meta_b64,
                    "signature_base64": valid_sig_b64,
                },
            }
        )

    # 4b. Root identity mismatch
    with pytest.raises(ValueError, match="root identity does not match"):
        _parse_state(
            {
                "schema_version": 2,
                "state": {
                    "highest_generation": 1,
                    "metadata_sha256": valid_meta_digest,
                    "root_key_id": "other-root-id",
                    "metadata_base64": valid_meta_b64,
                    "signature_base64": valid_sig_b64,
                },
            }
        )

    # 4c. Generation mismatch
    with pytest.raises(ValueError, match="generation does not match"):
        _parse_state(
            {
                "schema_version": 2,
                "state": {
                    "highest_generation": 2,
                    "metadata_sha256": valid_meta_digest,
                    "root_key_id": "valid-root-1",
                    "metadata_base64": valid_meta_b64,
                    "signature_base64": valid_sig_b64,
                },
            }
        )

    # 5. _trusted_metadata permission check and non-callable getuid
    import os
    import stat
    from unittest.mock import MagicMock

    stat_mock = MagicMock(spec=os.stat_result)
    stat_mock.st_uid = os.getuid()
    stat_mock.st_mode = stat.S_IFREG | 0o666  # group and other writable
    assert _trusted_metadata(stat_mock) is False

    monkeypatch.setattr(os, "getuid", None, raising=False)
    assert _trusted_metadata(stat_mock) is True
