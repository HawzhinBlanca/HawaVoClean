"""Release SBOM invariants that do not require Docker or network access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

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


def _artifact_tree(root: Path) -> Path:
    bundle = root / "bundle"
    executable = bundle / "bin" / "run"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"exact executable\n")
    executable.chmod(0o755)
    (bundle / "empty").mkdir()
    os.symlink("bin/run", bundle / "launcher")
    return bundle


def test_directory_artifact_is_path_independent_and_binds_the_complete_tree(
    tmp_path: Path,
) -> None:
    first = _artifact_tree(tmp_path / "one")
    second = _artifact_tree(tmp_path / "two")

    left = generate_sbom._artifact_component("plugin", first)
    right = generate_sbom._artifact_component("plugin", second)

    assert left == right
    assert str(tmp_path) not in str(left)
    assert {prop["name"]: prop["value"] for prop in left["properties"]}[
        "hawavoclean:artifact-kind"
    ] == "directory-tree"

    (second / "bin" / "run").write_bytes(b"changed\n")
    assert generate_sbom._artifact_component("plugin", second) != left


def test_directory_artifact_binds_modes_and_symlink_targets(tmp_path: Path) -> None:
    bundle = _artifact_tree(tmp_path)
    original = generate_sbom._artifact_component("plugin", bundle)

    (bundle / "bin" / "run").chmod(0o644)
    assert generate_sbom._artifact_component("plugin", bundle) != original
    (bundle / "bin" / "run").chmod(0o755)

    launcher = bundle / "launcher"
    launcher.unlink()
    os.symlink("empty", launcher)
    assert generate_sbom._artifact_component("plugin", bundle) != original


def test_directory_artifact_rejects_escaping_and_dangling_symlinks(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("external", encoding="utf-8")
    os.symlink("../outside", bundle / "escape")

    with pytest.raises(generate_sbom.SbomError, match="escapes"):
        generate_sbom._artifact_component("plugin", bundle)

    (bundle / "escape").unlink()
    os.symlink("missing", bundle / "dangling")
    with pytest.raises(generate_sbom.SbomError, match="dangling"):
        generate_sbom._artifact_component("plugin", bundle)


def test_artifact_parser_accepts_real_directories_and_rejects_aliases(tmp_path: Path) -> None:
    bundle = _artifact_tree(tmp_path)
    assert generate_sbom._parse_artifacts([f"plugin={bundle}"]) == [("plugin", bundle.resolve())]
    with pytest.raises(generate_sbom.SbomError, match="invalid or duplicate"):
        generate_sbom._parse_artifacts([f"plugin={bundle}", f"alias={bundle}"])


def test_image_reference_is_resolved_and_source_labels_must_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    image_id = f"sha256:{'b' * 64}"
    inspection: list[dict[str, Any]] = [
        {
            "Id": image_id,
            "Config": {
                "Labels": {
                    "org.opencontainers.image.revision": commit,
                    "org.opencontainers.image.version": "3.3.0",
                    "org.opencontainers.image.created": "1970-01-01T00:02:03Z",
                }
            },
        }
    ]

    def fake_run(command: list[str], **_kwargs: object) -> str:
        if command[:3] == ["docker", "image", "inspect"]:
            assert command[3] == "mutable:tag"
            return json.dumps(inspection)
        assert command[:2] == ["git", "show"]
        return "123\n"

    monkeypatch.setattr(generate_sbom, "_run", fake_run)
    assert generate_sbom._resolve_image("mutable:tag", commit, "3.3.0") == image_id

    inspection[0]["Config"]["Labels"]["org.opencontainers.image.revision"] = "c" * 40
    with pytest.raises(generate_sbom.SbomError, match="does not match"):
        generate_sbom._resolve_image("mutable:tag", commit, "3.3.0")


def test_exact_locks_enrich_package_hashes_and_explicit_unknown_licenses() -> None:
    components: list[dict[str, Any]] = [
        {
            "type": "library",
            "name": "electron",
            "version": "43.4.1",
            "purl": "pkg:npm/electron@43.4.1",
        },
        {
            "type": "library",
            "name": "annotated-types",
            "version": "0.8.0",
            "purl": "pkg:pypi/annotated-types@0.8.0",
        },
        {
            "type": "library",
            "name": "pnpm",
            "version": "11.22.0",
            "purl": "pkg:npm/pnpm@11.22.0",
        },
    ]

    generate_sbom._enrich_lock_metadata(components)

    assert any(value["alg"] == "SHA-512" for value in components[0]["hashes"])
    assert any(
        value
        == {
            "alg": "SHA-256",
            "content": "13b2beaad985e05e2d6407ee4c4f35590b11f8d693a258a561055cac8f64cab7",
        }
        for value in components[1]["hashes"]
    )
    assert components[0]["licenses"] == [{"license": {"name": "NOASSERTION"}}]
    assert any(value["alg"] == "SHA-512" for value in components[2]["hashes"])


def test_source_export_contains_only_the_exact_git_tree(tmp_path: Path) -> None:
    commit = generate_sbom._run(["git", "rev-parse", "HEAD"]).strip()
    source = generate_sbom._export_source_tree(commit, tmp_path)

    assert (source / "Dockerfile").is_file()
    assert not (source / ".git").exists()
    assert not (source / ".coverage").exists()
    assert not (source / "test_output").exists()


def _minimal_complete_contract() -> dict[str, Any]:
    def package(purl: str) -> dict[str, Any]:
        return {
            "type": "library",
            "name": purl,
            "purl": purl,
            "hashes": [{"alg": "SHA-256", "content": "a" * 64}],
            "licenses": [{"license": {"name": "NOASSERTION"}}],
        }

    model_components, _dependencies = generate_sbom._model_components()
    return {
        "components": [
            package("pkg:pypi/hawavoclean@3.3.0"),
            package("pkg:npm/react@19.2.8"),
            package("pkg:npm/electron@43.4.1"),
            package("pkg:apk/wolfi/ffmpeg-7@7.1.5-r0"),
            *model_components,
            {
                "type": "file",
                "name": "release.whl",
                "hashes": [{"alg": "SHA-256", "content": "b" * 64}],
                "licenses": [{"license": {"name": "NOASSERTION"}}],
                "properties": [{"name": "hawavoclean:artifact-name", "value": "wheel"}],
            },
        ]
    }


def test_contract_requires_hashes_for_every_locked_ecosystem_component() -> None:
    complete = _minimal_complete_contract()
    generate_sbom._validate_contract(complete, {"wheel"})

    electron = next(
        component
        for component in complete["components"]
        if component.get("purl") == "pkg:npm/electron@43.4.1"
    )
    electron["hashes"] = []
    with pytest.raises(generate_sbom.SbomError, match="no cryptographic hash"):
        generate_sbom._validate_contract(complete, {"wheel"})


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
    apk_source = {"purl": "pkg:apk/wolfi/ffmpeg-7@7.1.5-r0?arch=aarch64&distro=wolfi"}
    apk_image = {"purl": "pkg:apk/wolfi/ffmpeg-7@7.1.5-r0?arch=aarch64&distro=20230201"}
    assert generate_sbom._canonical_ref(apk_source) == generate_sbom._canonical_ref(apk_image)


def test_ephemeral_trivy_scan_subject_is_removed_without_dropping_metadata() -> None:
    bom = {
        "metadata": {
            "timestamp": "stable",
            "component": {"type": "application", "name": "/tmp/random/source"},
        }
    }

    generate_sbom._remove_scan_subject(bom)

    assert bom == {"metadata": {"timestamp": "stable"}}


def test_contract_rejects_a_missing_release_ecosystem() -> None:
    incomplete = {
        "components": [
            {
                "type": "library",
                "purl": "pkg:pypi/hawavoclean@3.3.0",
                "hashes": [{"alg": "SHA-256", "content": "a" * 64}],
                "licenses": [{"license": {"name": "NOASSERTION"}}],
                "properties": [{"name": "hawavoclean:artifact-name", "value": "wheel"}],
            }
        ]
    }

    with pytest.raises(generate_sbom.SbomError, match="required ecosystem"):
        generate_sbom._validate_contract(incomplete, {"wheel"})
