"""Targeted branch coverage tests for model_packs manifest, store, and trust."""

from __future__ import annotations

import json

import pytest

from hawavoclean.model_packs.errors import ModelPackManifestError
from hawavoclean.model_packs.manifest import (
    RuntimeCompatibility,
    SemanticVersion,
    parse_manifest_bytes,
)


def test_semantic_version_parsing_edge_cases() -> None:
    with pytest.raises(ModelPackManifestError, match="must be a string"):
        SemanticVersion.parse(123, field="test_ver")

    with pytest.raises(ModelPackManifestError, match="unreasonably long"):
        SemanticVersion.parse("1" * 35, field="test_ver")

    with pytest.raises(ModelPackManifestError, match="must be canonical"):
        SemanticVersion.parse("1.0", field="test_ver")

    v = SemanticVersion.parse("1.2.3", field="test_ver")
    assert str(v) == "1.2.3"
    assert v.major == 1 and v.minor == 2 and v.patch == 3


def test_runtime_compatibility_supports() -> None:
    compat = RuntimeCompatibility(min_version="1.0.0", max_version_exclusive="2.0.0")
    assert compat.supports("1.5.0")
    assert not compat.supports("0.9.0")
    assert not compat.supports("2.0.0")


def test_parse_manifest_bytes_edge_cases() -> None:
    # 1. Overly large manifest
    with pytest.raises(ModelPackManifestError, match="exceeds the 1 MiB"):
        parse_manifest_bytes(b"x" * (1024 * 1024 + 1))

    # 2. Non-JSON or bad types
    with pytest.raises(ModelPackManifestError, match="manifest must be an object"):
        parse_manifest_bytes(b"[]")

    # 3. Wrong product
    valid_base = {
        "schema_version": 1,
        "product": "WrongProduct",
        "pack_id": "pack_1",
        "version": "1.0.0",
        "issued_at": "2026-08-01T00:00:00Z",
        "not_before": "2026-08-01T00:00:00Z",
        "expires_at": "2027-08-01T00:00:00Z",
        "signing_key_id": "key_1",
        "quality_tier": "production",
        "maturity": "qualified",
        "runtime_compatibility": {
            "min_version": "1.0.0",
            "max_version_exclusive": "2.0.0",
        },
        "components": {
            "model": {
                "path": "payload/model.onnx",
                "sha256": "a" * 64,
                "size_bytes": 100,
            },
            "verifier": {
                "path": "payload/verifier.onnx",
                "sha256": "b" * 64,
                "size_bytes": 100,
            },
            "preprocessing": {
                "path": "payload/preprocessing.json",
                "sha256": "c" * 64,
                "size_bytes": 100,
            },
            "corpus": {
                "path": "provenance/corpus.json",
                "sha256": "d" * 64,
                "size_bytes": 100,
            },
            "runtime": {
                "path": "payload/runtime-contract.json",
                "sha256": "e" * 64,
                "size_bytes": 100,
            },
        },
        "assets": [
            {
                "role": "license",
                "path": "licenses/LICENSE.txt",
                "sha256": "f" * 64,
                "size_bytes": 100,
            }
        ],
    }

    with pytest.raises(ModelPackManifestError, match="names the wrong product"):
        parse_manifest_bytes(json.dumps(valid_base).encode("utf-8"))

    # 4. Invalid validity window
    bad_dates = dict(
        valid_base,
        product="hawavoclean-restore",
        issued_at="2026-09-01T00:00:00Z",
        not_before="2026-08-01T00:00:00Z",
    )
    with pytest.raises(ModelPackManifestError, match="issued_at must not be after not_before"):
        parse_manifest_bytes(json.dumps(bad_dates).encode("utf-8"))

    bad_dates2 = dict(
        valid_base,
        product="hawavoclean-restore",
        not_before="2027-09-01T00:00:00Z",
        expires_at="2026-08-01T00:00:00Z",
    )
    with pytest.raises(ModelPackManifestError, match="not_before must be earlier than expires_at"):
        parse_manifest_bytes(json.dumps(bad_dates2).encode("utf-8"))

    # 5. Invalid quality tier or maturity
    bad_q = dict(valid_base, product="hawavoclean-restore", quality_tier="invalid_tier")
    with pytest.raises(ModelPackManifestError, match="unsupported quality_tier"):
        parse_manifest_bytes(json.dumps(bad_q).encode("utf-8"))

    bad_m = dict(valid_base, product="hawavoclean-restore", maturity="invalid_maturity")
    with pytest.raises(ModelPackManifestError, match="unsupported maturity"):
        parse_manifest_bytes(json.dumps(bad_m).encode("utf-8"))

    # Inconsistent qualification: qualified but research tier
    bad_inconsist = dict(
        valid_base, product="hawavoclean-restore", quality_tier="research", maturity="qualified"
    )
    with pytest.raises(
        ModelPackManifestError, match="qualified packs must use the production quality tier"
    ):
        parse_manifest_bytes(json.dumps(bad_inconsist).encode("utf-8"))

    # Inconsistent qualification: production tier but experimental maturity
    bad_inconsist2 = dict(
        valid_base,
        product="hawavoclean-restore",
        quality_tier="production",
        maturity="experimental",
    )
    with pytest.raises(
        ModelPackManifestError, match="production-tier packs must be qualified or blocked"
    ):
        parse_manifest_bytes(json.dumps(bad_inconsist2).encode("utf-8"))

    # 6. Invalid runtime range
    bad_range = dict(
        valid_base,
        product="hawavoclean-restore",
        runtime_compatibility={"min_version": "2.0.0", "max_version_exclusive": "1.0.0"},
    )
    with pytest.raises(ModelPackManifestError, match="runtime compatibility range is empty"):
        parse_manifest_bytes(json.dumps(bad_range).encode("utf-8"))

    # 7. Assets not an array
    bad_assets = dict(valid_base, product="hawavoclean-restore", assets="not_a_list")
    with pytest.raises(ModelPackManifestError, match="assets must be an array"):
        parse_manifest_bytes(json.dumps(bad_assets).encode("utf-8"))

    # 8. Asset bad role
    bad_asset_role = dict(
        valid_base,
        product="hawavoclean-restore",
        assets=[
            {
                "role": "invalid_role",
                "path": "licenses/LICENSE.txt",
                "sha256": "e" * 64,
                "size_bytes": 100,
            }
        ],
    )
    with pytest.raises(ModelPackManifestError, match=r"assets\[0\]\.role is unsupported"):
        parse_manifest_bytes(json.dumps(bad_asset_role).encode("utf-8"))
