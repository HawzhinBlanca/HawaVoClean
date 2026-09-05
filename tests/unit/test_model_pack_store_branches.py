"""Targeted branch coverage tests for ModelPackStore and KeyRotationStateStore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hawavoclean.model_packs import (
    ModelPackInstallError,
    ModelPackSignatureError,
    ModelPackStore,
    TrustStore,
)
from hawavoclean.model_packs.rotation_store import KeyRotationStateStore


def _public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def test_store_state_loading_and_corruption_branches(tmp_path: Path) -> None:
    store = ModelPackStore(tmp_path / "store")
    store._ensure_root()
    state_file = store.root / "state.json"

    # 1. Non-dict root in state.json
    state_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ModelPackInstallError, match="invalid schema"):
        store._load_state()

    # 2. Invalid schema fields
    state_file.write_text(json.dumps({"schema_version": 1, "bad_field": 123}), encoding="utf-8")
    with pytest.raises(ModelPackInstallError, match="invalid schema"):
        store._load_state()

    # 3. Unsupported schema version
    state_file.write_text(json.dumps({"schema_version": 999, "packs": {}}), encoding="utf-8")
    with pytest.raises(ModelPackInstallError, match="unsupported schema"):
        store._load_state()

    # 4. Unsafe pack id in packs dict
    state_file.write_text(
        json.dumps({"schema_version": 1, "packs": {"invalid/pack/id": {}}}), encoding="utf-8"
    )
    with pytest.raises(ModelPackInstallError, match="unsafe pack id"):
        store._load_state()

    # 5. Invalid record schema in pack dict
    state_file.write_text(
        json.dumps({"schema_version": 1, "packs": {"sorani-clean": "not a dict"}}),
        encoding="utf-8",
    )
    with pytest.raises(ModelPackInstallError, match="is invalid"):
        store._load_state()


def test_store_path_and_root_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ModelPackStore(tmp_path / "store")

    # 1. Invalid pack_id in _pack_version_path
    with pytest.raises(ModelPackInstallError, match="invalid pack_id"):
        store._pack_version_path("invalid/id", "1.0.0")

    # 2. _ensure_root failing on mkdir
    unwritable_root = tmp_path / "store_unwritable"
    store_bad = ModelPackStore(unwritable_root)

    def failing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == unwritable_root:
            raise OSError("Read-only file system")
        real_mkdir(self, *args, **kwargs)

    real_mkdir = Path.mkdir
    monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    with pytest.raises(ModelPackInstallError, match="cannot create model-pack store"):
        store_bad._ensure_root()


def test_rotation_store_validation_branches(tmp_path: Path) -> None:
    rot_store = KeyRotationStateStore(tmp_path / "rot_store")
    rot_file = rot_store.root / "key-rotation-state.json"
    rot_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Current with no state file returns None
    assert rot_store.current() is None

    # 2. Load with corrupt JSON
    rot_file.write_text("invalid json", encoding="utf-8")
    with pytest.raises((ModelPackInstallError, ModelPackSignatureError)):
        rot_store.current()


def test_store_inspect_and_capabilities_branches(tmp_path: Path) -> None:
    store = ModelPackStore(tmp_path / "store")
    trust_store = TrustStore([])

    # 1. Invalid pack_id format in inspect
    cap = store.inspect("INVALID_PACK_ID", trust_store)
    assert cap.status == "blocked"
    assert cap.reason_code == "invalid_pack_id"

    # 2. Not installed pack_id in inspect
    cap2 = store.inspect("sorani-clean", trust_store)
    assert cap2.status == "blocked"
    assert cap2.reason_code == "pack_not_installed"

    # 3. capabilities with empty state returns empty tuple
    assert store.capabilities(trust_store) == ()

    # 4. capabilities with corrupted state returns blocked item
    store._ensure_root()
    state_file = store.root / "state.json"
    state_file.write_text("corrupted", encoding="utf-8")
    caps = store.capabilities(trust_store)
    assert len(caps) == 1
    assert caps[0].status == "blocked"


def test_store_primitives_and_strict_json(tmp_path: Path) -> None:
    from hawavoclean.model_packs.errors import ModelPackPayloadError
    from hawavoclean.model_packs.store import (
        _copy_regular_durable,
        _require_owned_directory,
        _strict_json,
        _write_new_durable,
    )

    # 1. _require_owned_directory on file
    f = tmp_path / "file.txt"
    f.touch()
    with pytest.raises(ModelPackInstallError, match="must be a real directory"):
        _require_owned_directory(f)

    # 2. _copy_regular_durable on missing file
    with pytest.raises(ModelPackPayloadError, match="cannot safely copy"):
        _copy_regular_durable(tmp_path / "missing.txt", tmp_path / "dest.txt")

    # 3. _write_new_durable
    out_file = tmp_path / "durable.bin"
    _write_new_durable(out_file, b"durable data")
    assert out_file.read_bytes() == b"durable data"

    # 4. _strict_json duplicate key
    dup_json = b'{"a": 1, "a": 2}'
    with pytest.raises(ValueError, match="duplicate key 'a'"):
        _strict_json(dup_json)


def test_rotation_store_parse_state_and_helper_branches(tmp_path: Path) -> None:
    from hawavoclean.model_packs.rotation_store import (
        KeyRotationState,
        _canonical_state_bytes,
        _decode_state_material,
        _parse_state,
        _trusted_metadata,
    )

    # 1. _parse_state invalid envelope
    with pytest.raises(ValueError, match="invalid state envelope"):
        _parse_state("not a dict")
    with pytest.raises(ValueError, match="invalid state envelope"):
        _parse_state({"schema_version": 1})

    # 2. _parse_state unsupported schema
    with pytest.raises(ValueError, match="unsupported state schema"):
        _parse_state({"schema_version": 999, "state": {}})

    # 3. _parse_state invalid fields
    with pytest.raises(ValueError, match="invalid state record"):
        _parse_state({"schema_version": 1, "state": {"unknown": 1}})

    # 4. _parse_state invalid root_key_id
    with pytest.raises(ValueError, match="invalid state root key ID"):
        _parse_state(
            {
                "schema_version": 1,
                "state": {
                    "root_key_id": "bad/key/id",
                    "highest_generation": 1,
                    "metadata_sha256": "a" * 64,
                },
            }
        )

    # 5. _parse_state invalid generation
    with pytest.raises(ValueError, match="invalid state generation"):
        _parse_state(
            {
                "schema_version": 1,
                "state": {
                    "root_key_id": "valid-key-id",
                    "highest_generation": 0,
                    "metadata_sha256": "a" * 64,
                },
            }
        )

    # 6. _parse_state invalid metadata_sha256
    with pytest.raises(ValueError, match="invalid state metadata digest"):
        _parse_state(
            {
                "schema_version": 1,
                "state": {
                    "root_key_id": "valid-key-id",
                    "highest_generation": 1,
                    "metadata_sha256": "short",
                },
            }
        )

    # 7. _decode_state_material error branches
    with pytest.raises(ValueError, match="must be base64 text"):
        _decode_state_material(123, maximum=100, field="test")
    with pytest.raises(ValueError, match="not canonical base64"):
        _decode_state_material("???", maximum=100, field="test")
    with pytest.raises(ValueError, match="exceeds its size limit"):
        _decode_state_material("AAAA", maximum=1, field="test")

    # 8. _canonical_state_bytes with and without verification material
    state_legacy = KeyRotationState("root-key", 1, "a" * 64)
    assert not state_legacy.has_verification_material
    legacy_bytes = _canonical_state_bytes(state_legacy)
    assert b'"schema_version":1' in legacy_bytes

    # 9. KeyRotationStateStore.application_default
    app_default = KeyRotationStateStore.application_default()
    assert app_default.root.name == "model-packs"

    # 10. _trusted_metadata
    st = tmp_path.stat()
    assert isinstance(_trusted_metadata(st), bool)
