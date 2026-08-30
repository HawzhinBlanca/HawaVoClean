"""Durability, rollback, and concurrency tests for key-rotation state."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import hawavoclean.model_packs.rotation_store as rotation_store_module
from hawavoclean.model_packs import (
    ModelPackInstallError,
    ModelPackQualificationPolicy,
    ModelPackRollbackError,
    ModelPackSignatureError,
    ModelPackStore,
    PinnedRotationRoot,
    TrustedKey,
    TrustStore,
    manifest_signature_message,
    rotation_signature_envelope_bytes,
    rotation_signature_message,
    signature_envelope_bytes,
)
from hawavoclean.model_packs.manifest import canonical_json_bytes
from hawavoclean.model_packs.rotation_store import KeyRotationStateStore

pytestmark = pytest.mark.unit

ROOT_ID = "offline-root-2026"
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
PACK_ID = "sorani-source-v2"


def _public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _signed_generation(
    root_key: Ed25519PrivateKey,
    pack_key: Ed25519PrivateKey,
    generation: int,
) -> tuple[bytes, bytes]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "product": "hawavoclean-model-pack-key-rotation",
        "root_key_id": ROOT_ID,
        "generation": generation,
        "issued_at": "2026-08-01T00:00:00Z",
        "not_before": "2026-08-15T00:00:00Z",
        "expires_at": "2027-08-15T00:00:00Z",
        "keys": [
            {
                "algorithm": "Ed25519",
                "key_id": f"pack-key-{generation}",
                "public_key": base64.b64encode(_public_bytes(pack_key)).decode("ascii"),
                "revoked": False,
            }
        ],
    }
    metadata_bytes = canonical_json_bytes(metadata)
    signature_bytes = rotation_signature_envelope_bytes(
        root_key_id=ROOT_ID,
        signature=root_key.sign(rotation_signature_message(metadata_bytes)),
    )
    return metadata_bytes, signature_bytes


def _signed_policy(
    root_key: Ed25519PrivateKey,
    generation: int,
    keys: list[tuple[str, Ed25519PrivateKey, bool]],
) -> tuple[bytes, bytes]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "product": "hawavoclean-model-pack-key-rotation",
        "root_key_id": ROOT_ID,
        "generation": generation,
        "issued_at": "2026-08-01T00:00:00Z",
        "not_before": "2026-08-15T00:00:00Z",
        "expires_at": "2027-08-15T00:00:00Z",
        "keys": [
            {
                "algorithm": "Ed25519",
                "key_id": key_id,
                "public_key": base64.b64encode(_public_bytes(key)).decode("ascii"),
                "revoked": revoked,
            }
            for key_id, key, revoked in sorted(keys)
        ],
    }
    metadata_bytes = canonical_json_bytes(metadata)
    return (
        metadata_bytes,
        rotation_signature_envelope_bytes(
            root_key_id=ROOT_ID,
            signature=root_key.sign(rotation_signature_message(metadata_bytes)),
        ),
    )


def _pack_record(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_pack(
    root: Path,
    signing_key: Ed25519PrivateKey,
    *,
    key_id: str,
    version: str,
) -> Path:
    payloads = {
        "model": ("payload/model.onnx", b"source-conditioned-model"),
        "verifier": ("payload/verifier.onnx", b"speaker-verifier"),
        "preprocessing": ("payload/preprocessing.json", b'{"stft":2048}'),
        "corpus": ("provenance/corpus.json", b'{"split":"speaker-disjoint"}'),
        "runtime": ("payload/runtime-contract.json", b'{"providers":["CPU"]}'),
    }
    license_path = "licenses/LICENSE.txt"
    license_bytes = b"test license"
    for path, payload in payloads.values():
        destination = root.joinpath(*Path(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    license_file = root.joinpath(*Path(license_path).parts)
    license_file.parent.mkdir(parents=True, exist_ok=True)
    license_file.write_bytes(license_bytes)
    manifest = {
        "schema_version": 1,
        "product": "hawavoclean-restore",
        "pack_id": PACK_ID,
        "version": version,
        "issued_at": "2026-01-01T00:00:00Z",
        "not_before": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "signing_key_id": key_id,
        "quality_tier": "production",
        "maturity": "qualified",
        "runtime_compatibility": {
            "min_version": "3.0.0",
            "max_version_exclusive": "4.0.0",
        },
        "components": {
            role: _pack_record(path, payload) for role, (path, payload) in payloads.items()
        },
        "assets": [{"role": "license", **_pack_record(license_path, license_bytes)}],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "manifest.sig").write_bytes(
        signature_envelope_bytes(
            key_id=key_id,
            signature=signing_key.sign(manifest_signature_message(manifest_bytes)),
        )
    )
    return root


def _process_accept(
    root: str,
    metadata: bytes,
    signature: bytes,
    root_public_key: bytes,
    results: Any,
) -> None:
    try:
        generation = (
            KeyRotationStateStore(root)
            .verify_and_commit(
                metadata,
                signature,
                PinnedRotationRoot(ROOT_ID, root_public_key),
                now=NOW,
            )
            .generation
        )
        results.put(("accepted", generation))
    except ModelPackRollbackError as exc:
        results.put((exc.code, None))


@pytest.fixture
def signed_generations() -> tuple[
    Ed25519PrivateKey,
    tuple[bytes, bytes],
    tuple[bytes, bytes],
]:
    root_key = Ed25519PrivateKey.generate()
    return (
        root_key,
        _signed_generation(root_key, Ed25519PrivateKey.generate(), 7),
        _signed_generation(root_key, Ed25519PrivateKey.generate(), 8),
    )


def test_verify_commit_persists_floor_across_relaunch_and_blocks_rollback(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
) -> None:
    root_key, generation_7, generation_8 = signed_generations
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    store = KeyRotationStateStore(tmp_path / "app-data" / "model-packs")

    accepted = store.verify_and_commit(*generation_8, pinned, now=NOW)
    relaunched = KeyRotationStateStore(store.root)

    assert accepted.generation == 8
    assert relaunched.current() is not None
    assert relaunched.current().highest_generation == 8  # type: ignore[union-attr]
    with pytest.raises(ModelPackRollbackError) as caught:
        relaunched.verify_and_commit(*generation_7, pinned, now=NOW)
    assert caught.value.code == "rotation_rollback_rejected"


def test_model_pack_store_exposes_verify_and_commit_without_implicit_root(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
) -> None:
    root_key, generation_7, _ = signed_generations
    store = ModelPackStore(tmp_path / "model-packs")

    accepted = store.verify_and_commit_key_rotation(
        *generation_7,
        PinnedRotationRoot(ROOT_ID, _public_bytes(root_key)),
        now=NOW,
    )
    assert accepted.generation == 7
    with pytest.raises(ModelPackSignatureError) as caught:
        KeyRotationStateStore(store.root).verify_and_commit(
            *generation_7,
            None,  # type: ignore[arg-type]
            now=NOW,
        )
    assert caught.value.code == "missing_pinned_rotation_root"


def test_invalid_signature_never_creates_a_floor(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
) -> None:
    root_key, generation_7, _ = signed_generations
    metadata, _signature = generation_7
    wrong_signature = _signed_generation(
        Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate(), 7
    )[1]
    store = KeyRotationStateStore(tmp_path / "model-packs")

    with pytest.raises(ModelPackSignatureError):
        store.verify_and_commit(
            metadata,
            wrong_signature,
            PinnedRotationRoot(ROOT_ID, _public_bytes(root_key)),
            now=NOW,
        )
    assert store.current() is None


def test_equal_generation_is_idempotent_but_equivocation_is_rejected(
    tmp_path: Path,
) -> None:
    root_key = Ed25519PrivateKey.generate()
    first = _signed_generation(root_key, Ed25519PrivateKey.generate(), 7)
    different = _signed_generation(root_key, Ed25519PrivateKey.generate(), 7)
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    store = KeyRotationStateStore(tmp_path / "model-packs")

    first_result = store.verify_and_commit(*first, pinned, now=NOW)
    repeated = store.verify_and_commit(*first, pinned, now=NOW)

    assert repeated.metadata_sha256 == first_result.metadata_sha256
    with pytest.raises(ModelPackRollbackError) as caught:
        store.verify_and_commit(*different, pinned, now=NOW)
    assert caught.value.code == "rotation_generation_collision"
    assert store.current() is not None
    assert store.current().metadata_sha256 == first_result.metadata_sha256  # type: ignore[union-attr]


def test_fault_before_replace_preserves_old_floor_and_retry_advances(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_key, generation_7, generation_8 = signed_generations
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    store = KeyRotationStateStore(tmp_path / "model-packs")
    store.verify_and_commit(*generation_7, pinned, now=NOW)

    def crash(name: str) -> None:
        if name == "temporary_flushed":
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(rotation_store_module, "_rotation_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.verify_and_commit(*generation_8, pinned, now=NOW)

    assert KeyRotationStateStore(store.root).current().highest_generation == 7  # type: ignore[union-attr]
    monkeypatch.setattr(rotation_store_module, "_rotation_checkpoint", lambda _name: None)
    assert store.verify_and_commit(*generation_8, pinned, now=NOW).generation == 8


def test_fault_after_atomic_replace_is_safe_on_relaunch(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_key, generation_7, generation_8 = signed_generations
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    store = KeyRotationStateStore(tmp_path / "model-packs")
    store.verify_and_commit(*generation_7, pinned, now=NOW)

    def crash(name: str) -> None:
        if name == "state_replaced":
            raise RuntimeError("simulated process death")

    monkeypatch.setattr(rotation_store_module, "_rotation_checkpoint", crash)
    with pytest.raises(RuntimeError, match="simulated process death"):
        store.verify_and_commit(*generation_8, pinned, now=NOW)

    relaunched = KeyRotationStateStore(store.root)
    assert relaunched.current().highest_generation == 8  # type: ignore[union-attr]
    with pytest.raises(ModelPackRollbackError):
        relaunched.verify_and_commit(*generation_7, pinned, now=NOW)


def test_concurrent_processes_cannot_commit_a_lower_final_floor(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-process lock test requires the fork start method")
    root_key, generation_7, generation_8 = signed_generations
    root = tmp_path / "model-packs"
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    work = [generation_7, generation_8, generation_7, generation_8]
    processes = [
        context.Process(
            target=_process_accept,
            args=(str(root), metadata, signature, _public_bytes(root_key), results),
        )
        for metadata, signature in work
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    outcomes = [results.get(timeout=2) for _ in processes]
    assert any(outcome == ("accepted", 8) for outcome in outcomes)
    assert KeyRotationStateStore(root).current().highest_generation == 8  # type: ignore[union-attr]


def test_root_mismatch_and_corrupt_state_fail_closed(
    tmp_path: Path,
    signed_generations: tuple[
        Ed25519PrivateKey,
        tuple[bytes, bytes],
        tuple[bytes, bytes],
    ],
) -> None:
    root_key, generation_7, _ = signed_generations
    store = KeyRotationStateStore(tmp_path / "model-packs")
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    store.verify_and_commit(*generation_7, pinned, now=NOW)

    other_root = PinnedRotationRoot("new-offline-root", _public_bytes(Ed25519PrivateKey.generate()))
    with pytest.raises(ModelPackRollbackError) as mismatch:
        store.verify_and_commit(*generation_7, other_root, now=NOW)
    assert mismatch.value.code == "rotation_state_root_mismatch"

    state_file = store.root / "key-rotation-state.json"
    state_file.write_text('{"schema_version":1,"state":{}}', encoding="utf-8")
    with pytest.raises(Exception) as corrupt:
        KeyRotationStateStore(store.root).current()
    assert getattr(corrupt.value, "code", None) == "corrupt_rotation_state"


def test_persisted_rotation_material_rebuilds_effective_trust_after_restart(
    tmp_path: Path,
) -> None:
    root_key = Ed25519PrivateKey.generate()
    active_key = Ed25519PrivateKey.generate()
    revoked_key = Ed25519PrivateKey.generate()
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    generation = _signed_policy(
        root_key,
        9,
        [
            ("pack-active", active_key, False),
            ("pack-revoked", revoked_key, True),
        ],
    )
    store = KeyRotationStateStore(tmp_path / "model-packs")
    accepted = store.verify_and_commit(*generation, pinned, now=NOW)

    state = KeyRotationStateStore(store.root).current()
    assert state is not None
    assert state.has_verification_material
    assert state.metadata_bytes == generation[0]
    assert state.signature_bytes == generation[1]

    relaunched = KeyRotationStateStore(store.root).load_verified(pinned, now=NOW)
    assert relaunched is not None
    assert relaunched.metadata_sha256 == accepted.metadata_sha256
    message = b"pack signature domain"
    relaunched.to_trust_store().verify(
        key_id="pack-active",
        signature=active_key.sign(message),
        message=message,
    )
    with pytest.raises(ModelPackSignatureError) as revoked:
        relaunched.to_trust_store().verify(
            key_id="pack-revoked",
            signature=revoked_key.sign(message),
            message=message,
        )
    assert revoked.value.code == "revoked_signing_key"


def test_stale_caller_trust_cannot_override_committed_revocation_after_restart(
    tmp_path: Path,
) -> None:
    root_key = Ed25519PrivateKey.generate()
    retired_key = Ed25519PrivateKey.generate()
    replacement_key = Ed25519PrivateKey.generate()
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    stale_trust = TrustStore([TrustedKey("pack-retired", _public_bytes(retired_key))])
    generation_1 = _signed_policy(
        root_key,
        1,
        [
            ("pack-current", replacement_key, False),
            ("pack-retired", retired_key, False),
        ],
    )
    generation_2 = _signed_policy(
        root_key,
        2,
        [
            ("pack-current", replacement_key, False),
            ("pack-retired", retired_key, True),
        ],
    )
    store = ModelPackStore(tmp_path / "model-packs")
    store.verify_and_commit_key_rotation(*generation_1, pinned, now=NOW)
    retired_v1 = _write_pack(
        tmp_path / "retired-v1",
        retired_key,
        key_id="pack-retired",
        version="1.0.0",
    )
    store.install(retired_v1, stale_trust, runtime_version="3.3.0", now=NOW)

    store.verify_and_commit_key_rotation(*generation_2, pinned, now=NOW)
    blocked = store.inspect(PACK_ID, stale_trust, runtime_version="3.3.0", now=NOW)
    assert blocked.status == "blocked"
    assert blocked.reason_code == "revoked_signing_key"

    retired_v2 = _write_pack(
        tmp_path / "retired-v2",
        retired_key,
        key_id="pack-retired",
        version="2.0.0",
    )
    with pytest.raises(ModelPackSignatureError) as revoked:
        store.install(retired_v2, stale_trust, runtime_version="3.3.0", now=NOW)
    assert revoked.value.code == "revoked_signing_key"

    unpinned_restart = ModelPackStore(store.root)
    missing_root = unpinned_restart.inspect(
        PACK_ID,
        stale_trust,
        runtime_version="3.3.0",
        now=NOW,
    )
    assert missing_root.status == "blocked"
    assert missing_root.reason_code == "rotation_root_required"

    restarted = ModelPackStore(store.root, pinned_rotation_root=pinned)
    still_blocked = restarted.inspect(
        PACK_ID,
        stale_trust,
        runtime_version="3.3.0",
        now=NOW,
    )
    assert still_blocked.reason_code == "revoked_signing_key"

    current_v2 = _write_pack(
        tmp_path / "current-v2",
        replacement_key,
        key_id="pack-current",
        version="2.0.0",
    )
    qualified_restart = ModelPackStore(
        store.root,
        pinned_rotation_root=pinned,
        qualification_policies=(
            ModelPackQualificationPolicy(
                pack_id=PACK_ID,
                version="2.0.0",
                manifest_sha256=hashlib.sha256(
                    (current_v2 / "manifest.json").read_bytes()
                ).hexdigest(),
                providers=("CPUExecutionProvider",),
            ),
        ),
    )
    installed = qualified_restart.install(
        current_v2,
        stale_trust,
        runtime_version="3.3.0",
        now=NOW,
    )
    assert installed.manifest.signing_key_id == "pack-current"
    assert qualified_restart.inspect(
        PACK_ID,
        stale_trust,
        runtime_version="3.3.0",
        now=NOW,
    ).usable


def test_legacy_digest_only_state_requires_root_signed_in_place_upgrade(
    tmp_path: Path,
) -> None:
    root_key = Ed25519PrivateKey.generate()
    pack_key = Ed25519PrivateKey.generate()
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    generation = _signed_policy(root_key, 7, [("pack-active", pack_key, False)])
    root = tmp_path / "model-packs"
    root.mkdir()
    (root / "key-rotation-state.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "state": {
                    "highest_generation": 7,
                    "metadata_sha256": hashlib.sha256(generation[0]).hexdigest(),
                    "root_key_id": ROOT_ID,
                },
            }
        )
    )
    legacy_trust = TrustStore([TrustedKey("pack-active", _public_bytes(pack_key))])
    store = ModelPackStore(root, pinned_rotation_root=pinned)

    blocked = store.inspect(PACK_ID, legacy_trust, runtime_version="3.3.0", now=NOW)
    assert blocked.reason_code == "rotation_state_upgrade_required"
    with pytest.raises(ModelPackInstallError) as migration:
        store.install(
            _write_pack(
                tmp_path / "source",
                pack_key,
                key_id="pack-active",
                version="1.0.0",
            ),
            legacy_trust,
            runtime_version="3.3.0",
            now=NOW,
        )
    assert migration.value.code == "rotation_state_upgrade_required"

    store.verify_and_commit_key_rotation(*generation, pinned, now=NOW)
    upgraded = KeyRotationStateStore(root).current()
    assert upgraded is not None
    assert upgraded.has_verification_material
    assert json.loads((root / "key-rotation-state.json").read_text())["schema_version"] == 2


def test_substituted_persisted_rotation_signature_fails_cryptographically(
    tmp_path: Path,
) -> None:
    root_key = Ed25519PrivateKey.generate()
    pack_key = Ed25519PrivateKey.generate()
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    generation = _signed_policy(root_key, 3, [("pack-active", pack_key, False)])
    state_store = KeyRotationStateStore(tmp_path / "model-packs")
    state_store.verify_and_commit(*generation, pinned, now=NOW)
    path = state_store.root / "key-rotation-state.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    forged_signature = rotation_signature_envelope_bytes(
        root_key_id=ROOT_ID,
        signature=b"\0" * 64,
    )
    envelope["state"]["signature_base64"] = base64.b64encode(forged_signature).decode("ascii")
    path.write_bytes(canonical_json_bytes(envelope))

    with pytest.raises(ModelPackSignatureError) as forged:
        KeyRotationStateStore(state_store.root).load_verified(pinned, now=NOW)
    assert forged.value.code == "invalid_rotation_root_signature"


def test_pinned_store_fails_closed_if_rotation_state_is_missing(
    tmp_path: Path,
) -> None:
    root_key = Ed25519PrivateKey.generate()
    pack_key = Ed25519PrivateKey.generate()
    pinned = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    generation = _signed_policy(root_key, 1, [("pack-active", pack_key, False)])
    store = ModelPackStore(tmp_path / "model-packs")
    store.verify_and_commit_key_rotation(*generation, pinned, now=NOW)
    (store.root / "key-rotation-state.json").unlink()
    stale_trust = TrustStore([TrustedKey("pack-active", _public_bytes(pack_key))])

    blocked = store.inspect(PACK_ID, stale_trust, runtime_version="3.3.0", now=NOW)
    assert blocked.reason_code == "rotation_state_required"
    with pytest.raises(ModelPackInstallError) as missing:
        store.install(
            _write_pack(
                tmp_path / "source",
                pack_key,
                key_id="pack-active",
                version="1.0.0",
            ),
            stale_trust,
            runtime_version="3.3.0",
            now=NOW,
        )
    assert missing.value.code == "rotation_state_required"
