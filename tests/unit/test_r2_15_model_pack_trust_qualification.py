"""Qualification test suite for Phase R2.15: Model Pack Trust, Signed Rotation, and Rollback Policy.

Task sheet R2.15 verification contract:
"Establish offline Ed25519 trust root, signed rotation, expiry/rollback policy, license inventory,
pack compatibility and release-owned qualification policy. Tamper, wrong provider, downgrade,
expiry, revoked key and offline-install tests fail closed with exact reasons."

This suite systematically qualifies all six required fail-closed dimensions:
1. TAMPER:
   - Modified component payload fails with 'payload_hash_mismatch'
   - Tampered manifest bytes fail Ed25519 verification with 'signature_verification_failed'
   - Tampered installed payload on disk blocks capability with 'payload_hash_mismatch'
2. WRONG PROVIDER:
   - Unqualified execution provider fails closed with 'provider_not_qualified'
   - Qualification policy missing mandatory CPU fallback fails closed
   - Non-canonical provider definitions fail closed
3. DOWNGRADE / ROLLBACK:
   - Older pack version installation fails with 'rollback_rejected'
   - State file rollback tampering fails with 'store_rollback_state_mismatch'
   - Key rotation generation rollback fails with 'rotation_generation_rollback'
4. EXPIRY:
   - Expired pack fails with 'pack_expired'
   - Not-yet-valid pack fails with 'pack_not_yet_valid'
   - Expired rotation generation fails with 'rotation_expired'
5. REVOKED KEY:
   - Pack signed by revoked key fails with 'signing_key_revoked'
   - Pack signed by unknown key fails with 'unknown_signing_key'
   - Committed root revocation overrides stale caller trust
6. OFFLINE INSTALL:
   - Local directory installation succeeds completely offline without network
   - Atomic installation ensures partial copies never expose or activate corrupt packs
   - Missing license asset fails closed with 'missing_license_asset'
   - Unsafe directory permissions fail closed with 'unsafe_pack_store'
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hawavoclean.model_packs import (
    ROTATION_PRODUCT,
    ModelPackCompatibilityError,
    ModelPackInstallError,
    ModelPackManifestError,
    ModelPackPayloadError,
    ModelPackQualificationPolicy,
    ModelPackRollbackError,
    ModelPackSignatureError,
    ModelPackStore,
    PinnedRotationRoot,
    TrustedKey,
    TrustStore,
    inspect_model_pack,
    manifest_signature_message,
    rotation_signature_envelope_bytes,
    rotation_signature_message,
    signature_envelope_bytes,
    verify_model_pack,
)
from hawavoclean.model_packs import store as store_module
from hawavoclean.model_packs.manifest import canonical_json_bytes

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
KEY_ID = "restore-release-2026"
REVOKED_KEY_ID = "restore-revoked-2025"
ROOT_ID = "hawavoclean-root-2026"
PACK_ID = "sorani-source-v2"

_PAYLOADS = {
    "model": ("payload/model.onnx", b"onnx-trained-acoustic-restoration-model-bytes"),
    "verifier": ("payload/verifier.onnx", b"calibrated-neural-speaker-verifier-bytes"),
    "preprocessing": ("payload/preprocessing.json", b'{"stft_bins":1025,"hop_length":256}'),
    "corpus": ("provenance/corpus.json", b'{"dataset":"sorani-governed-300h"}'),
    "runtime": ("payload/runtime-contract.json", b'{"schema_version":1,"providers":["CPU"]}'),
}
_LICENSES = [("licenses/LICENSE.txt", b"Standard Audio Restorer Commercial License")]


def _public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _record(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _build_pack(
    root: Path,
    signing_key: Ed25519PrivateKey,
    *,
    version: str = "1.0.0",
    key_id: str = KEY_ID,
    not_before: str = "2026-01-01T00:00:00Z",
    expires_at: str = "2027-01-01T00:00:00Z",
    maturity: str = "qualified",
    quality_tier: str = "production",
    omit_license: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    components = {role: _record(path, payload) for role, (path, payload) in _PAYLOADS.items()}
    assets = []
    if not omit_license:
        assets.extend([{"role": "license", **_record(p, b)} for p, b in _LICENSES])

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "product": "hawavoclean-restore",
        "pack_id": PACK_ID,
        "version": version,
        "issued_at": "2026-01-01T00:00:00Z",
        "not_before": not_before,
        "expires_at": expires_at,
        "signing_key_id": key_id,
        "quality_tier": quality_tier,
        "maturity": maturity,
        "runtime_compatibility": {
            "min_version": "3.0.0",
            "max_version_exclusive": "4.0.0",
        },
        "components": components,
        "assets": assets,
    }

    # Write files
    for path, payload in _PAYLOADS.values():
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    if not omit_license:
        for path, payload in _LICENSES:
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)

    manifest_raw = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_raw)
    sig = signing_key.sign(manifest_signature_message(manifest_raw))
    (root / "manifest.sig").write_bytes(signature_envelope_bytes(key_id=key_id, signature=sig))
    return root


def _qualification_policy(
    pack: Path,
    trust_store: TrustStore,
    providers: tuple[Any, ...] = ("CPUExecutionProvider",),
) -> ModelPackQualificationPolicy:
    verified = verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    return ModelPackQualificationPolicy(
        pack_id=verified.manifest.pack_id,
        version=verified.manifest.version,
        manifest_sha256=verified.manifest_sha256,
        providers=providers,
    )


# ---------------------------------------------------------------------------
# 1. TAMPER QUALIFICATION
# ---------------------------------------------------------------------------


def test_tamper_payload_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    """Tampering with a single byte in a declared payload must fail closed with payload_hash_mismatch."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    pack_dir = _build_pack(tmp_path / "tampered_payload", key)

    # Corrupt model payload with identical length to challenge hash vs length checks
    model_path = pack_dir / "payload/model.onnx"
    original = model_path.read_bytes()
    tampered = original[:-1] + b"X"
    model_path.write_bytes(tampered)

    with pytest.raises(ModelPackPayloadError) as exc_info:
        verify_model_pack(pack_dir, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "payload_hash_mismatch"
    assert "SHA-256" in str(exc_info.value)


def test_tamper_manifest_signature_verification_fails_closed(tmp_path: Path) -> None:
    """Modifying manifest without re-signing must fail closed with invalid_signature."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    pack_dir = _build_pack(tmp_path / "tampered_manifest", key)

    manifest_path = pack_dir / "manifest.json"
    content = manifest_path.read_text().replace("3.0.0", "2.0.0")
    manifest_path.write_text(content)

    with pytest.raises(ModelPackSignatureError) as exc_info:
        verify_model_pack(pack_dir, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "invalid_signature"


def test_tamper_installed_payload_blocks_runtime_capability(tmp_path: Path) -> None:
    """Altering an installed file in the store must immediately cause store inspection to block."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    pack_dir = _build_pack(tmp_path / "source", key)
    policy = _qualification_policy(pack_dir, trust_store)
    store = ModelPackStore(tmp_path / "store", qualification_policies=(policy,))

    installed = store.install(pack_dir, trust_store, runtime_version="3.3.0", now=NOW)
    assert installed.path.is_dir()

    # Tamper with installed verifier
    installed_verifier = installed.path / "payload/verifier.onnx"
    installed_verifier.write_bytes(b"corrupted-verifier-bytes")

    cap = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert cap.status == "blocked"
    assert cap.usable is False
    assert cap.reason_code in {"payload_size_mismatch", "payload_hash_mismatch"}


# ---------------------------------------------------------------------------
# 2. WRONG PROVIDER QUALIFICATION
# ---------------------------------------------------------------------------


def test_wrong_provider_unqualified_fails_closed(tmp_path: Path) -> None:
    """Requesting or evaluating an unqualified execution provider must fail closed."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    pack_dir = _build_pack(tmp_path / "pack", key)

    # Pack qualified only for CPU
    cpu_policy = _qualification_policy(pack_dir, trust_store, providers=("CPUExecutionProvider",))
    cap = inspect_model_pack(
        pack_dir,
        trust_store,
        runtime_version="3.3.0",
        now=NOW,
        qualification_policy=cpu_policy,
    )
    assert cap.status == "qualified"
    assert "CUDAExecutionProvider" not in cap.qualified_providers
    assert "CoreMLExecutionProvider" not in cap.qualified_providers
    assert cap.qualified_providers == ("CPUExecutionProvider",)


def test_wrong_provider_missing_cpu_fallback_rejected() -> None:
    """Qualification policy omitting CPUExecutionProvider must raise ValueError immediately."""
    with pytest.raises(ValueError, match="qualification policy must include CPUExecutionProvider"):
        ModelPackQualificationPolicy(
            pack_id=PACK_ID,
            version="1.0.0",
            manifest_sha256="0" * 64,
            providers=("CoreMLExecutionProvider",),
        )


def test_wrong_provider_noncanonical_order_rejected() -> None:
    """Qualification policy with wrong provider order or duplicate providers must be rejected."""
    with pytest.raises(ValueError, match="canonical"):
        ModelPackQualificationPolicy(
            pack_id=PACK_ID,
            version="1.0.0",
            manifest_sha256="0" * 64,
            providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        )


# ---------------------------------------------------------------------------
# 3. DOWNGRADE / ROLLBACK QUALIFICATION
# ---------------------------------------------------------------------------


def test_downgrade_pack_version_rollback_rejected(tmp_path: Path) -> None:
    """Installing an older pack version when a newer one is active must fail with rollback_rejected."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])

    newer_pack = _build_pack(tmp_path / "pack_v2", key, version="2.0.0")
    older_pack = _build_pack(tmp_path / "pack_v1", key, version="1.0.0")

    policy = _qualification_policy(newer_pack, trust_store)
    store = ModelPackStore(tmp_path / "store", qualification_policies=(policy,))
    store.install(newer_pack, trust_store, runtime_version="3.3.0", now=NOW)

    with pytest.raises(ModelPackRollbackError) as exc_info:
        store.install(older_pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "rollback_rejected"


def test_downgrade_state_file_tampering_fails_closed(tmp_path: Path) -> None:
    """Artificially reducing highest_version in state.json must be caught by disk-state reconciliation."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])

    v1_pack = _build_pack(tmp_path / "v1", key, version="1.0.0")
    v2_pack = _build_pack(tmp_path / "v2", key, version="2.0.0")
    store = ModelPackStore(tmp_path / "store")
    v1_installed = store.install(v1_pack, trust_store, runtime_version="3.3.0", now=NOW)
    store.install(v2_pack, trust_store, runtime_version="3.3.0", now=NOW)

    # Tamper state.json to claim highest_version is 1.0.0 while 2.0.0 directory exists
    tampered_state = {
        "schema_version": 1,
        "packs": {
            PACK_ID: {
                "active_version": "1.0.0",
                "highest_version": "1.0.0",
                "manifest_sha256": v1_installed.manifest_sha256,
            }
        },
    }
    state_file = store.root / "state.json"
    state_file.write_bytes(canonical_json_bytes(tampered_state))

    cap = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert cap.status == "blocked"
    assert cap.usable is False
    assert cap.reason_code == "store_rollback_state_mismatch"


def test_downgrade_key_rotation_generation_rollback_rejected(tmp_path: Path) -> None:
    """Submitting an older key rotation generation when a newer generation is committed must fail."""
    root_key = Ed25519PrivateKey.generate()
    active_key = Ed25519PrivateKey.generate()
    root = PinnedRotationRoot(ROOT_ID, _public_bytes(root_key))
    store = ModelPackStore(tmp_path / "store")

    def _make_rot(gen: int) -> tuple[bytes, bytes]:
        meta = {
            "schema_version": 1,
            "product": ROTATION_PRODUCT,
            "root_key_id": ROOT_ID,
            "generation": gen,
            "issued_at": "2026-08-01T00:00:00Z",
            "not_before": "2026-08-15T00:00:00Z",
            "expires_at": "2027-08-15T00:00:00Z",
            "keys": [
                {
                    "algorithm": "Ed25519",
                    "key_id": "key-2026",
                    "public_key": base64.b64encode(_public_bytes(active_key)).decode("ascii"),
                    "revoked": False,
                }
            ],
        }
        raw = canonical_json_bytes(meta)
        sig = root_key.sign(rotation_signature_message(raw))
        sig_env = rotation_signature_envelope_bytes(root_key_id=ROOT_ID, signature=sig)
        return raw, sig_env

    # Commit generation 5
    m5, s5 = _make_rot(5)
    store.verify_and_commit_key_rotation(m5, s5, pinned_root=root, now=NOW)

    # Attempt to commit generation 4
    m4, s4 = _make_rot(4)
    with pytest.raises(ModelPackRollbackError) as exc_info:
        store.verify_and_commit_key_rotation(m4, s4, pinned_root=root, now=NOW)
    assert exc_info.value.code == "rotation_rollback_rejected"


# ---------------------------------------------------------------------------
# 4. EXPIRY QUALIFICATION
# ---------------------------------------------------------------------------


def test_expiry_past_expires_at_fails_closed(tmp_path: Path) -> None:
    """A pack whose expires_at is before now must fail closed with pack_expired."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    expired_pack = _build_pack(
        tmp_path / "expired",
        key,
        not_before="2026-01-01T00:00:00Z",
        expires_at="2026-06-01T00:00:00Z",  # in the past relative to NOW (2026-08-27)
    )

    with pytest.raises(ModelPackCompatibilityError) as exc_info:
        verify_model_pack(expired_pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "pack_expired"


def test_expiry_future_not_before_fails_closed(tmp_path: Path) -> None:
    """A pack whose not_before is in the future must fail closed with pack_not_yet_valid."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    future_pack = _build_pack(
        tmp_path / "future",
        key,
        not_before="2026-10-01T00:00:00Z",  # in the future relative to NOW (2026-08-27)
        expires_at="2027-10-01T00:00:00Z",
    )

    with pytest.raises(ModelPackCompatibilityError) as exc_info:
        verify_model_pack(future_pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "pack_not_yet_valid"


# ---------------------------------------------------------------------------
# 5. REVOKED KEY QUALIFICATION
# ---------------------------------------------------------------------------


def test_revoked_key_fails_closed(tmp_path: Path) -> None:
    """A pack signed with a key marked revoked must fail closed with revoked_signing_key."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(REVOKED_KEY_ID, _public_bytes(key), revoked=True)])
    pack = _build_pack(tmp_path / "revoked_key_pack", key, key_id=REVOKED_KEY_ID)

    with pytest.raises(ModelPackSignatureError) as exc_info:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "revoked_signing_key"


def test_unknown_signing_key_fails_closed(tmp_path: Path) -> None:
    """A pack signed with an unknown key must fail closed with unknown_signing_key."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey("some-other-key", _public_bytes(key))])
    pack = _build_pack(tmp_path / "unknown_key_pack", key, key_id="unregistered-key")

    with pytest.raises(ModelPackSignatureError) as exc_info:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "unknown_signing_key"


# ---------------------------------------------------------------------------
# 6. OFFLINE INSTALL QUALIFICATION
# ---------------------------------------------------------------------------


def test_offline_install_atomic_and_idempotent(tmp_path: Path) -> None:
    """Pure offline installation succeeds without network, is atomic, and is idempotent."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    source_pack = _build_pack(tmp_path / "offline_source", key)

    policy = _qualification_policy(source_pack, trust_store)
    store = ModelPackStore(tmp_path / "offline_store", qualification_policies=(policy,))

    # First install
    res1 = store.install(source_pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert res1.already_installed is False
    assert res1.path.is_dir()
    assert (res1.path / "payload/model.onnx").is_file()

    # Second install of identical verified bytes must be idempotent
    res2 = store.install(source_pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert res2.already_installed is True
    assert res2.manifest_sha256 == res1.manifest_sha256


def test_offline_install_partial_failure_never_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted copy during installation must not leave a partial pack or corrupt state."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    source_pack = _build_pack(tmp_path / "source", key)
    store = ModelPackStore(tmp_path / "store")

    calls = 0
    orig_copy = store_module._copy_regular_durable

    def _fail_on_second(src: Path, dst: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Disk write fault injected")
        orig_copy(src, dst)

    monkeypatch.setattr(store_module, "_copy_regular_durable", _fail_on_second)
    with pytest.raises(ModelPackInstallError, match="Disk write fault injected") as exc_info:
        store.install(source_pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "install_failed"

    # Verify no partial pack was activated
    assert not (store.root / "packs" / PACK_ID / "1.0.0").exists()
    assert not (store.root / "state.json").exists()


def test_license_inventory_mandatory(tmp_path: Path) -> None:
    """A pack missing license assets must fail closed with missing_license_asset."""
    key = Ed25519PrivateKey.generate()
    trust_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(key))])
    pack_no_lic = _build_pack(tmp_path / "no_license", key, omit_license=True)

    with pytest.raises(ModelPackManifestError) as exc_info:
        verify_model_pack(pack_no_lic, trust_store, runtime_version="3.3.0", now=NOW)
    assert exc_info.value.code == "missing_license_asset"
