"""Release SBOM invariants that do not require Docker or network access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import generate_sbom


def test_artifact_component_is_content_bound_and_path_independent(tmp_path: Path) -> None:
    first = tmp_path / "one" / "release.whl"
    second = tmp_path / "two" / "release.whl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same artifact")
    second.write_bytes(b"same artifact")

    left = generate_sbom._artifact_component("wheel", first)
    right = generate_sbom._artifact_component("wheel", second)

    assert left == right
    assert str(tmp_path) not in str(left)
    assert left["hashes"] == [
        {"alg": "SHA-256", "content": hashlib.sha256(b"same artifact").hexdigest()}
    ]


def test_vendored_model_inventory_contains_every_weight_hash() -> None:
    components, dependencies = generate_sbom._model_components()
    hashes = {
        digest["content"] for component in components for digest in component.get("hashes", [])
    }

    for weight in generate_sbom.MODEL_ROOT.glob("deepfilternet3/**/*"):
        if weight.is_file():
            assert generate_sbom._sha256(weight) in hashes
    assert dependencies["urn:hawavoclean:core:studio-dfn3-48k-v1@1.1.0"]
    assert dependencies["urn:hawavoclean:core:studio-dfn3-lowband-48k-v1@1.0.0"]


def test_inventory_references_are_deterministic_and_purls_are_preserved() -> None:
    package = {"type": "library", "name": "numpy", "version": "2", "purl": "pkg:pypi/numpy@2"}
    file_component = {
        "type": "file",
        "name": "weights.bin",
        "hashes": [{"alg": "SHA-256", "content": "a" * 64}],
    }

    assert generate_sbom._canonical_ref(package) == "pkg:pypi/numpy@2"
    assert generate_sbom._canonical_ref(file_component) == generate_sbom._canonical_ref(
        dict(reversed(list(file_component.items())))
    )


def test_contract_rejects_a_missing_release_ecosystem() -> None:
    incomplete = {
        "components": [
            {
                "type": "library",
                "purl": "pkg:pypi/hawavoclean@3.3.0",
                "properties": [{"name": "hawavoclean:artifact-name", "value": "wheel"}],
            }
        ]
    }

    with pytest.raises(generate_sbom.SbomError, match="required ecosystem"):
        generate_sbom._validate_contract(incomplete, {"wheel"})
