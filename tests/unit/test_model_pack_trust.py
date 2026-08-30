"""Adversarial tests for the signed Restore model-pack trust boundary."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hawavoclean.model_packs import (
    ExecutionProvider,
    ModelPackCompatibilityError,
    ModelPackInstallError,
    ModelPackManifestError,
    ModelPackPayloadError,
    ModelPackQualificationPolicy,
    ModelPackRollbackError,
    ModelPackSignatureError,
    ModelPackStore,
    TrustedKey,
    TrustStore,
    inspect_model_pack,
    manifest_signature_message,
    signature_envelope_bytes,
    verify_model_pack,
)
from hawavoclean.model_packs import store as store_module
from hawavoclean.model_packs import trust as trust_module
from hawavoclean.model_packs.manifest import canonical_json_bytes

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
KEY_ID = "restore-release-2026"
PACK_ID = "sorani-source-v2"

_COMPONENT_FILES = {
    "model": ("payload/model.onnx", b"source-conditioned-onnx-model"),
    "verifier": ("payload/verifier.onnx", b"speaker-verifier"),
    "preprocessing": ("payload/preprocessing.json", b'{"stft":2048}'),
    "corpus": ("provenance/corpus.json", b'{"split":"speaker-disjoint"}'),
    "runtime": ("payload/runtime-contract.json", b'{"providers":["CPU"]}'),
}
_ASSET_FILES = [("licenses/LICENSE.txt", b"test license")]


@pytest.fixture
def signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def trust_store(signing_key: Ed25519PrivateKey) -> TrustStore:
    return TrustStore([TrustedKey(KEY_ID, _public_bytes(signing_key))])


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _concurrent_install_worker(arguments: tuple[str, str, bytes]) -> bool:
    source, store_root, public_key = arguments
    result = ModelPackStore(store_root).install(
        source,
        TrustStore([TrustedKey(KEY_ID, public_key)]),
        runtime_version="3.3.0",
        now=NOW,
    )
    return result.already_installed


def _record(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_pack(
    root: Path,
    private_key: Ed25519PrivateKey,
    *,
    version: str = "1.0.0",
    maturity: str = "qualified",
    quality_tier: str = "production",
    mutate: Callable[[dict[str, Any]], None] | None = None,
    envelope_key_id: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    components = {
        role: _record(path, payload) for role, (path, payload) in _COMPONENT_FILES.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "product": "hawavoclean-restore",
        "pack_id": PACK_ID,
        "version": version,
        "issued_at": "2026-01-01T00:00:00Z",
        "not_before": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "signing_key_id": KEY_ID,
        "quality_tier": quality_tier,
        "maturity": maturity,
        "runtime_compatibility": {
            "min_version": "3.0.0",
            "max_version_exclusive": "4.0.0",
        },
        "components": components,
        "assets": [{"role": "license", **_record(path, payload)} for path, payload in _ASSET_FILES],
    }
    if mutate is not None:
        mutate(manifest)

    for path, payload in (*_COMPONENT_FILES.values(), *_ASSET_FILES):
        destination = root.joinpath(*Path(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    signature = private_key.sign(manifest_signature_message(manifest_bytes))
    (root / "manifest.sig").write_bytes(
        signature_envelope_bytes(
            key_id=envelope_key_id or str(manifest["signing_key_id"]),
            signature=signature,
        )
    )
    return root


def _qualification_policy(
    pack: Path,
    trust_store: TrustStore,
    *,
    providers: tuple[ExecutionProvider, ...] = ("CPUExecutionProvider",),
) -> ModelPackQualificationPolicy:
    verified = verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    return ModelPackQualificationPolicy(
        pack_id=verified.manifest.pack_id,
        version=verified.manifest.version,
        manifest_sha256=verified.manifest_sha256,
        providers=providers,
    )


def test_qualified_pack_requires_exact_release_policy_and_reports_only_pinned_providers(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    pack = _write_pack(tmp_path / "pack", signing_key)

    verified = verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert verified.manifest.pack_id == PACK_ID
    assert verified.total_payload_bytes == sum(
        len(payload) for _, payload in (*_COMPONENT_FILES.values(), *_ASSET_FILES)
    )
    assert (
        verified.manifest.model_sha256 == hashlib.sha256(_COMPONENT_FILES["model"][1]).hexdigest()
    )
    assert (
        verified.manifest.verifier_sha256
        == hashlib.sha256(_COMPONENT_FILES["verifier"][1]).hexdigest()
    )
    assert (
        verified.manifest.preprocessing_sha256
        == hashlib.sha256(_COMPONENT_FILES["preprocessing"][1]).hexdigest()
    )
    assert (
        verified.manifest.corpus_sha256 == hashlib.sha256(_COMPONENT_FILES["corpus"][1]).hexdigest()
    )
    assert (
        verified.manifest.runtime_sha256
        == hashlib.sha256(_COMPONENT_FILES["runtime"][1]).hexdigest()
    )

    unpinned = inspect_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert unpinned.status == "blocked"
    assert unpinned.usable is False
    assert unpinned.reason_code == "qualification_policy_missing"

    policy = _qualification_policy(
        pack,
        trust_store,
        providers=("CPUExecutionProvider", "CoreMLExecutionProvider"),
    )
    capability = inspect_model_pack(
        pack,
        trust_store,
        runtime_version="3.3.0",
        now=NOW,
        qualification_policy=policy,
    )
    assert capability.status == "qualified"
    assert capability.usable is True
    assert capability.reason_code == "release_pinned_qualified_pack"
    assert dict(capability.component_hashes)["model"] == verified.manifest.model_sha256
    assert capability.qualified_providers == (
        "CPUExecutionProvider",
        "CoreMLExecutionProvider",
    )


def test_self_declared_qualified_pack_cannot_reuse_another_signed_manifest_policy(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    qualified = _write_pack(tmp_path / "qualified", signing_key)
    policy = _qualification_policy(qualified, trust_store)
    other = _write_pack(
        tmp_path / "self-declared",
        signing_key,
        mutate=lambda data: data.__setitem__("expires_at", "2026-12-31T00:00:00Z"),
    )

    capability = inspect_model_pack(
        other,
        trust_store,
        runtime_version="3.3.0",
        now=NOW,
        qualification_policy=policy,
    )

    assert capability.status == "blocked"
    assert capability.usable is False
    assert capability.reason_code == "qualification_identity_mismatch"
    assert capability.qualified_providers == ()


def test_manifest_requires_at_least_one_signed_license_asset(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    pack = _write_pack(
        tmp_path / "no-license",
        signing_key,
        mutate=lambda data: data.__setitem__("assets", []),
    )

    with pytest.raises(ModelPackManifestError) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == "missing_license_asset"


@pytest.mark.parametrize(
    "providers",
    [
        (),
        ("CoreMLExecutionProvider",),
        ("CoreMLExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider", "BogusExecutionProvider"),
    ],
)
def test_qualification_policy_rejects_noncanonical_provider_sets(
    providers: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="qualification policy"):
        ModelPackQualificationPolicy(
            pack_id=PACK_ID,
            version="1.0.0",
            manifest_sha256="0" * 64,
            providers=providers,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "relative_path",
    [path for path, _ in (*_COMPONENT_FILES.values(), *_ASSET_FILES)],
)
def test_every_declared_payload_is_hash_verified(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    relative_path: str,
) -> None:
    pack = _write_pack(tmp_path / "pack", signing_key)
    pack.joinpath(*Path(relative_path).parts).write_bytes(b"tampered")

    with pytest.raises(
        ModelPackPayloadError,
        match="(?:size|SHA-256) mismatch",
    ) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code in {"payload_size_mismatch", "payload_hash_mismatch"}


def test_same_size_payload_tamper_reaches_sha256_check(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    pack = _write_pack(tmp_path / "pack", signing_key)
    model = pack / _COMPONENT_FILES["model"][0]
    model.write_bytes(b"x" * model.stat().st_size)

    with pytest.raises(ModelPackPayloadError) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == "payload_hash_mismatch"


def test_manifest_tamper_fails_signature_before_payload_use(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    pack = _write_pack(tmp_path / "pack", signing_key)
    manifest = json.loads((pack / "manifest.json").read_bytes())
    manifest["expires_at"] = "2028-01-01T00:00:00Z"
    (pack / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ModelPackSignatureError) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == "invalid_signature"


def test_unknown_revoked_and_mismatched_signing_keys_fail_closed(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
) -> None:
    pack = _write_pack(tmp_path / "unknown", signing_key)
    with pytest.raises(ModelPackSignatureError) as unknown:
        verify_model_pack(pack, TrustStore([]), runtime_version="3.3.0", now=NOW)
    assert unknown.value.code == "unknown_signing_key"

    revoked_store = TrustStore([TrustedKey(KEY_ID, _public_bytes(signing_key), revoked=True)])
    with pytest.raises(ModelPackSignatureError) as revoked:
        verify_model_pack(pack, revoked_store, runtime_version="3.3.0", now=NOW)
    assert revoked.value.code == "revoked_signing_key"

    mismatch = _write_pack(
        tmp_path / "mismatch",
        signing_key,
        envelope_key_id="different-release-key",
    )
    with pytest.raises(ModelPackSignatureError) as key_mismatch:
        verify_model_pack(
            mismatch,
            TrustStore([TrustedKey(KEY_ID, _public_bytes(signing_key))]),
            runtime_version="3.3.0",
            now=NOW,
        )
    assert key_mismatch.value.code == "signing_key_mismatch"


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda data: data.__setitem__("schema_version", 2), "unsupported_schema"),
        (
            lambda data: data.__setitem__("not_before", "2026-09-01T00:00:00Z"),
            "pack_not_yet_valid",
        ),
        (
            lambda data: data.__setitem__("expires_at", "2026-08-01T00:00:00Z"),
            "pack_expired",
        ),
        (
            lambda data: data.__setitem__(
                "runtime_compatibility",
                {"min_version": "3.4.0", "max_version_exclusive": "4.0.0"},
            ),
            "incompatible_runtime",
        ),
    ],
)
def test_schema_validity_and_runtime_policy_rejections(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    mutate: Callable[[dict[str, Any]], None],
    expected_code: str,
) -> None:
    pack = _write_pack(tmp_path / expected_code, signing_key, mutate=mutate)

    with pytest.raises((ModelPackManifestError, ModelPackCompatibilityError)) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == expected_code


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.onnx",
        "/absolute/model.onnx",
        "C:/windows/model.onnx",
        "C:\\windows\\model.onnx",
        "payload/../escape.onnx",
        "payload//model.onnx",
        "payload/./model.onnx",
        "payload/model:on.onnx",
        "payload/model?.onnx",
        "payload/model.onnx.",
        "payload/NUL.onnx",
        "payload/cafe\u0301.onnx",
        "manifest.json",
        "MANIFEST.JSON",
        "Manifest.Sig",
    ],
)
def test_manifest_rejects_path_traversal_and_ambiguous_paths(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    unsafe_path: str,
) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["components"]["model"]["path"] = unsafe_path

    pack = _write_pack(tmp_path / "pack", signing_key, mutate=mutate)
    with pytest.raises(ModelPackManifestError) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == "unsafe_payload_path"


def test_manifest_rejects_casefold_collisions_and_reserved_pack_id(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    def collide(data: dict[str, Any]) -> None:
        data["assets"][0]["path"] = "PAYLOAD/MODEL.ONNX"

    collision = _write_pack(tmp_path / "collision", signing_key, mutate=collide)
    with pytest.raises(ModelPackManifestError) as duplicate:
        verify_model_pack(collision, trust_store, runtime_version="3.3.0", now=NOW)
    assert duplicate.value.code == "duplicate_payload_path"

    def ancestor_collision(data: dict[str, Any]) -> None:
        data["components"]["model"]["path"] = "Foo"
        data["components"]["verifier"]["path"] = "foo/bar"

    ancestor = _write_pack(tmp_path / "ancestor", signing_key, mutate=ancestor_collision)
    with pytest.raises(ModelPackManifestError) as conflicting:
        verify_model_pack(ancestor, trust_store, runtime_version="3.3.0", now=NOW)
    assert conflicting.value.code == "conflicting_payload_path"

    reserved = _write_pack(
        tmp_path / "reserved",
        signing_key,
        mutate=lambda data: data.__setitem__("pack_id", "con"),
    )
    with pytest.raises(ModelPackManifestError) as bad_id:
        verify_model_pack(reserved, trust_store, runtime_version="3.3.0", now=NOW)
    assert bad_id.value.code == "invalid_identifier"


def test_manifest_rejects_path_surrogates_and_unbounded_semver(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    surrogate = _write_pack(tmp_path / "surrogate", signing_key)
    value = json.loads((surrogate / "manifest.json").read_bytes())
    value["components"]["model"]["path"] = "payload/\ud800.onnx"
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (surrogate / "manifest.json").write_bytes(raw)
    (surrogate / "manifest.sig").write_bytes(
        signature_envelope_bytes(
            key_id=KEY_ID,
            signature=signing_key.sign(manifest_signature_message(raw)),
        )
    )
    with pytest.raises(ModelPackManifestError) as unsafe_unicode:
        verify_model_pack(surrogate, trust_store, runtime_version="3.3.0", now=NOW)
    assert unsafe_unicode.value.code == "unsafe_payload_path"

    huge_version = _write_pack(
        tmp_path / "huge-version",
        signing_key,
        version=f"{'1' * 100}.0.0",
    )
    with pytest.raises(ModelPackManifestError) as invalid_version:
        verify_model_pack(huge_version, trust_store, runtime_version="3.3.0", now=NOW)
    assert invalid_version.value.code == "invalid_version"


def test_pack_rejects_symlinked_payload_and_undeclared_file(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    pack = _write_pack(tmp_path / "symlink", signing_key)
    model = pack / _COMPONENT_FILES["model"][0]
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(model.read_bytes())
    model.unlink()
    model.symlink_to(outside)
    with pytest.raises(ModelPackPayloadError) as symlink:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert symlink.value.code == "unsafe_pack_layout"

    closed = _write_pack(tmp_path / "extra", signing_key)
    (closed / "not-signed.bin").write_bytes(b"hitchhiker")
    with pytest.raises(ModelPackPayloadError) as undeclared:
        verify_model_pack(closed, trust_store, runtime_version="3.3.0", now=NOW)
    assert undeclared.value.code == "undeclared_pack_entry"


def test_pack_rejects_windows_reparse_points_even_when_lstat_looks_regular(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Junctions/mount points are not always reported as POSIX symlinks."""

    pack = _write_pack(tmp_path / "reparse", signing_key)
    model = pack / _COMPONENT_FILES["model"][0]
    real_check = trust_module.is_reparse_or_symlink
    monkeypatch.setattr(
        trust_module,
        "is_reparse_or_symlink",
        lambda path: Path(path) == model or real_check(Path(path)),
    )

    with pytest.raises(ModelPackPayloadError) as caught:
        verify_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == "unsafe_pack_layout"


def test_noncanonical_and_duplicate_json_are_rejected_even_when_signed(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    noncanonical = _write_pack(tmp_path / "pretty", signing_key)
    value = json.loads((noncanonical / "manifest.json").read_bytes())
    pretty = json.dumps(value, indent=2, sort_keys=True).encode()
    (noncanonical / "manifest.json").write_bytes(pretty)
    (noncanonical / "manifest.sig").write_bytes(
        signature_envelope_bytes(
            key_id=KEY_ID,
            signature=signing_key.sign(manifest_signature_message(pretty)),
        )
    )
    with pytest.raises(ModelPackManifestError) as canonical_error:
        verify_model_pack(noncanonical, trust_store, runtime_version="3.3.0", now=NOW)
    assert canonical_error.value.code == "noncanonical_manifest"

    duplicate = _write_pack(tmp_path / "duplicate", signing_key)
    original = (duplicate / "manifest.json").read_bytes()
    duplicated = b'{"schema_version":1,' + original[1:]
    (duplicate / "manifest.json").write_bytes(duplicated)
    (duplicate / "manifest.sig").write_bytes(
        signature_envelope_bytes(
            key_id=KEY_ID,
            signature=signing_key.sign(manifest_signature_message(duplicated)),
        )
    )
    with pytest.raises(ModelPackManifestError) as duplicate_error:
        verify_model_pack(duplicate, trust_store, runtime_version="3.3.0", now=NOW)
    assert duplicate_error.value.code == "invalid_json"


def test_experimental_pack_is_authentic_but_not_usable(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    pack = _write_pack(
        tmp_path / "experimental",
        signing_key,
        maturity="experimental",
        quality_tier="candidate",
    )
    capability = inspect_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)
    assert capability.status == "experimental"
    assert capability.usable is False
    assert capability.reason_code == "experimental_pack"


@pytest.mark.parametrize("quality_tier", ["research", "candidate"])
def test_qualified_metadata_with_nonproduction_quality_tier_is_blocked(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    quality_tier: str,
) -> None:
    pack = _write_pack(
        tmp_path / quality_tier,
        signing_key,
        maturity="qualified",
        quality_tier=quality_tier,
    )

    capability = inspect_model_pack(pack, trust_store, runtime_version="3.3.0", now=NOW)

    assert capability.status == "blocked"
    assert capability.usable is False
    assert capability.reason_code == "inconsistent_qualification"
    assert capability.quality_tier is None


def test_store_atomically_installs_then_idempotently_reuses_verified_bytes(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    source = _write_pack(tmp_path / "source", signing_key)
    policy = _qualification_policy(source, trust_store)
    store = ModelPackStore(
        tmp_path / "app-data" / "model-packs",
        qualification_policies=(policy,),
    )

    first = store.install(source, trust_store, runtime_version="3.3.0", now=NOW)
    assert first.path == (store.root / "packs" / PACK_ID / "1.0.0").resolve()
    assert first.already_installed is False
    assert not first.path.is_symlink()
    assert (first.path / "payload/model.onnx").is_file()
    assert not (first.path / "payload/model.onnx").is_symlink()

    second = store.install(source, trust_store, runtime_version="3.3.0", now=NOW)
    assert second.already_installed is True
    assert second.manifest_sha256 == first.manifest_sha256

    unpinned_store = ModelPackStore(store.root)
    unpinned = unpinned_store.inspect(
        PACK_ID,
        trust_store,
        runtime_version="3.3.0",
        now=NOW,
    )
    assert unpinned.status == "blocked"
    assert unpinned.reason_code == "qualification_policy_missing"

    capability = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert capability.status == "qualified"
    assert capability.usable is True
    assert capability.qualified_providers == ("CPUExecutionProvider",)
    assert store.capabilities(trust_store, runtime_version="3.3.0", now=NOW) == (capability,)


def test_store_rejects_rollback_and_leaves_newer_pack_active(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    newer = _write_pack(tmp_path / "newer", signing_key, version="2.0.0")
    older = _write_pack(tmp_path / "older", signing_key, version="1.9.9")
    store = ModelPackStore(
        tmp_path / "store",
        qualification_policies=(_qualification_policy(newer, trust_store),),
    )
    store.install(newer, trust_store, runtime_version="3.3.0", now=NOW)

    with pytest.raises(ModelPackRollbackError) as caught:
        store.install(older, trust_store, runtime_version="3.3.0", now=NOW)
    assert caught.value.code == "rollback_rejected"
    capability = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert capability.version == "2.0.0"
    assert capability.usable is True
    assert not (store.root / "packs" / PACK_ID / "1.9.9").exists()


def test_inspection_reconciles_state_floor_with_newer_on_disk_version(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    first_source = _write_pack(tmp_path / "first", signing_key, version="1.0.0")
    second_source = _write_pack(tmp_path / "second", signing_key, version="2.0.0")
    store = ModelPackStore(tmp_path / "store")
    first = store.install(first_source, trust_store, runtime_version="3.3.0", now=NOW)
    store.install(second_source, trust_store, runtime_version="3.3.0", now=NOW)

    # Reproduce a same-owner, canonical state rollback while both immutable
    # versions remain on disk. Inspection must derive the floor from disk too,
    # not trust this locally rewritten high-water mark.
    rolled_back_state = {
        "schema_version": 1,
        "packs": {
            PACK_ID: {
                "active_version": "1.0.0",
                "highest_version": "1.0.0",
                "manifest_sha256": first.manifest_sha256,
            }
        },
    }
    state_path = store.root / "state.json"
    state_path.write_bytes(canonical_json_bytes(rolled_back_state))
    state_path.chmod(0o600)

    capability = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert capability.status == "blocked"
    assert capability.usable is False
    assert capability.reason_code == "store_rollback_state_mismatch"


def test_concurrent_process_install_is_serialized_and_idempotent(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    source = _write_pack(tmp_path / "source", signing_key)
    store = ModelPackStore(
        tmp_path / "store",
        qualification_policies=(_qualification_policy(source, trust_store),),
    )
    arguments = (str(source), str(store.root), _public_bytes(signing_key))
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        results = list(executor.map(_concurrent_install_worker, [arguments] * 4))

    assert results.count(False) == 1
    assert results.count(True) == 3
    capability = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert capability.usable
    assert [path.name for path in (store.root / "packs" / PACK_ID).iterdir()] == ["1.0.0"]


def test_partial_copy_never_exposes_or_activates_a_pack(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pack(tmp_path / "source", signing_key)
    store = ModelPackStore(
        tmp_path / "store",
        qualification_policies=(_qualification_policy(source, trust_store),),
    )
    real_copy = store_module._copy_regular_durable
    calls = 0

    def fail_third_copy(source_path: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ModelPackPayloadError("injected copy failure", code="pack_copy_failed")
        real_copy(source_path, destination)

    monkeypatch.setattr(store_module, "_copy_regular_durable", fail_third_copy)
    with pytest.raises(ModelPackPayloadError, match="injected"):
        store.install(source, trust_store, runtime_version="3.3.0", now=NOW)

    assert not (store.root / "packs" / PACK_ID / "1.0.0").exists()
    assert not (store.root / "state.json").exists()
    assert list(store.root.glob(".install-*")) == []


def test_crash_after_directory_commit_recovers_idempotently_on_retry(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_pack(tmp_path / "source", signing_key)
    store = ModelPackStore(
        tmp_path / "store",
        qualification_policies=(_qualification_policy(source, trust_store),),
    )

    def crash_after_commit(name: str) -> None:
        if name == "pack_installed":
            raise RuntimeError("simulated process death after directory commit")

    monkeypatch.setattr(store_module, "_checkpoint", crash_after_commit)
    with pytest.raises(ModelPackInstallError, match="simulated process death"):
        store.install(source, trust_store, runtime_version="3.3.0", now=NOW)

    committed = store.root / "packs" / PACK_ID / "1.0.0"
    assert committed.is_dir()
    assert not (store.root / "state.json").exists()
    assert (
        verify_model_pack(
            committed,
            trust_store,
            runtime_version="3.3.0",
            now=NOW,
        ).manifest.pack_id
        == PACK_ID
    )

    monkeypatch.setattr(store_module, "_checkpoint", lambda _name: None)
    recovered = store.install(source, trust_store, runtime_version="3.3.0", now=NOW)
    assert recovered.already_installed is True
    assert store.inspect(
        PACK_ID,
        trust_store,
        runtime_version="3.3.0",
        now=NOW,
    ).usable


def test_installed_payload_tamper_blocks_capability(
    tmp_path: Path,
    signing_key: Ed25519PrivateKey,
    trust_store: TrustStore,
) -> None:
    source = _write_pack(tmp_path / "source", signing_key)
    store = ModelPackStore(tmp_path / "store")
    installed = store.install(source, trust_store, runtime_version="3.3.0", now=NOW)
    (installed.path / "payload/model.onnx").write_bytes(b"tampered")

    capability = store.inspect(PACK_ID, trust_store, runtime_version="3.3.0", now=NOW)
    assert capability.status == "blocked"
    assert capability.usable is False
    assert capability.reason_code in {"payload_size_mismatch", "payload_hash_mismatch"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission-bit policy")
def test_store_rejects_group_or_world_writable_root(
    tmp_path: Path,
    trust_store: TrustStore,
) -> None:
    root = tmp_path / "unsafe-store"
    root.mkdir()
    root.chmod(0o777)

    capability = ModelPackStore(root).inspect(PACK_ID, trust_store, now=NOW)
    assert capability.status == "blocked"
    assert capability.reason_code == "unsafe_pack_store"


def test_store_rejects_a_windows_reparse_root(
    tmp_path: Path,
    trust_store: TrustStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reparse-store"
    real_check = store_module.is_reparse_or_symlink
    monkeypatch.setattr(
        store_module,
        "is_reparse_or_symlink",
        lambda path: Path(path) == root or real_check(Path(path)),
    )

    capability = ModelPackStore(root).inspect(PACK_ID, trust_store, now=NOW)
    assert capability.status == "blocked"
    assert capability.reason_code == "unsafe_pack_store"
