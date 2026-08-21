#!/usr/bin/env python3
"""Build and validate the deterministic, release-bound CycloneDX 1.6 SBOM."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "src" / "hawavoclean" / "resources" / "models"
TRIVY_VERSION = "0.67.2"
VALIDATOR_VERSION = "0.35.0"
SCHEMAS = {
    "bom-1.6.schema.json": (
        "https://raw.githubusercontent.com/CycloneDX/specification/1.6/schema/bom-1.6.schema.json",
        "3e92dddbc30cf7f6a02b80f0942b1a4cfd4fb1c26f1dfc4310afa9d613cafb93",
    ),
    "jsf-0.82.schema.json": (
        "https://raw.githubusercontent.com/CycloneDX/specification/1.6/schema/jsf-0.82.schema.json",
        "8bae002c25e723db7ee1f26afde680ae1a2b1a8f6b4b4b0fd65dc3becb090aae",
    ),
    "spdx.schema.json": (
        "https://raw.githubusercontent.com/CycloneDX/specification/1.6/schema/spdx.schema.json",
        "baa9d3bd1ed57b6751b0887edead6b5063ff53ff7429cf85d476c6c94af0166e",
    ),
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class SbomError(RuntimeError):
    """The inventory could not be generated or did not meet the release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise SbomError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed.stdout


def _tool_preflight() -> None:
    trivy = _run(["trivy", "--version"])
    if not trivy.startswith(f"Version: {TRIVY_VERSION}\n"):
        raise SbomError(f"Trivy must be exactly {TRIVY_VERSION}; got {trivy.splitlines()[0]}")
    validator = _run(
        [
            "uvx",
            "--from",
            f"check-jsonschema=={VALIDATOR_VERSION}",
            "check-jsonschema",
            "--version",
        ]
    )
    if validator.strip() != f"check-jsonschema, version {VALIDATOR_VERSION}":
        raise SbomError(f"unexpected schema validator: {validator.strip()}")


def _source_date(commit: str) -> str:
    epoch = int(_run(["git", "show", "-s", "--format=%ct", commit]).strip())
    return (
        datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _release_version() -> str:
    value: Any = json.loads((ROOT / "src" / "hawavoclean" / "release.json").read_text())
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        raise SbomError("release.json does not contain a version")
    return cast(str, value["version"])


def _trivy_bom(arguments: list[str], output: Path) -> dict[str, Any]:
    _run(["trivy", *arguments, "--quiet", "--format", "cyclonedx", "--output", str(output)])
    value: Any = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("specVersion") != "1.6":
        raise SbomError("Trivy did not emit CycloneDX 1.6 JSON")
    return value


def _remove_scan_subject(bom: dict[str, Any]) -> None:
    """Drop Trivy's path-derived scan subject; the release root is added later."""
    metadata = bom.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("component", None)


def _remove_mutable_image_aliases(components: list[dict[str, Any]]) -> None:
    """Remove local Docker tags that are not part of an immutable image identity.

    Trivy reports every tag currently attached to an image ID. Adding a second
    tag to the same byte-identical image must not change its release SBOM; the
    immutable ImageID, RepoDigest, layers and OCI labels remain recorded.
    """
    for component in components:
        purl = component.get("purl")
        if component.get("type") != "container" and not (
            isinstance(purl, str) and purl.startswith("pkg:oci/")
        ):
            continue
        properties = component.get("properties")
        if isinstance(properties, list):
            component["properties"] = [
                prop
                for prop in properties
                if not (isinstance(prop, dict) and prop.get("name") == "aquasecurity:trivy:RepoTag")
            ]


def _canonical_ref(component: dict[str, Any]) -> str:
    purl = component.get("purl")
    if isinstance(purl, str) and purl:
        # Trivy describes the same Wolfi package from a source lock and an
        # installed image with different observational arch/distro qualifiers.
        # Name+version is the package identity; merging retains both sources'
        # hashes, licenses and properties on the installed component.
        if purl.startswith("pkg:apk/"):
            return purl.partition("?")[0]
        return purl
    hashes = component.get("hashes", [])
    identity = {
        "type": component.get("type"),
        "group": component.get("group"),
        "name": component.get("name"),
        "version": component.get("version"),
        "hashes": hashes if isinstance(hashes, list) else [],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"urn:hawavoclean:component:{digest}"


def _dedupe_objects(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = {json.dumps(value, sort_keys=True, separators=(",", ":")): value for value in values}
    return [encoded[key] for key in sorted(encoded)]


def _merge_component(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    for field in ("hashes", "licenses", "properties", "externalReferences"):
        left = existing.get(field, [])
        right = incoming.get(field, [])
        if isinstance(left, list) and isinstance(right, list) and (left or right):
            existing[field] = _dedupe_objects([*left, *right])
    for field in ("description", "publisher", "author", "scope"):
        if field not in existing and field in incoming:
            existing[field] = incoming[field]


def _normalized_inventory(
    boms: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    components: dict[str, dict[str, Any]] = {}
    ref_maps: list[dict[str, str]] = []
    bom_list = list(boms)
    for bom in bom_list:
        ref_map: dict[str, str] = {}
        for raw in bom.get("components", []):
            if not isinstance(raw, dict):
                continue
            component = copy.deepcopy(raw)
            old_ref = component.get("bom-ref")
            new_ref = _canonical_ref(component)
            component["bom-ref"] = new_ref
            if isinstance(old_ref, str):
                ref_map[old_ref] = new_ref
            if new_ref in components:
                _merge_component(components[new_ref], component)
            else:
                components[new_ref] = component
        metadata_component = bom.get("metadata", {}).get("component")
        if isinstance(metadata_component, dict):
            component = copy.deepcopy(metadata_component)
            old_ref = component.get("bom-ref")
            new_ref = _canonical_ref(component)
            component["bom-ref"] = new_ref
            if isinstance(old_ref, str):
                ref_map[old_ref] = new_ref
            if new_ref in components:
                _merge_component(components[new_ref], component)
            else:
                components[new_ref] = component
        ref_maps.append(ref_map)

    dependencies: dict[str, set[str]] = {}
    for bom, ref_map in zip(bom_list, ref_maps, strict=True):
        for raw in bom.get("dependencies", []):
            if not isinstance(raw, dict) or not isinstance(raw.get("ref"), str):
                continue
            ref = ref_map.get(raw["ref"])
            if ref is None or ref not in components:
                continue
            children = {
                mapped
                for child in raw.get("dependsOn", [])
                if isinstance(child, str)
                for mapped in [ref_map.get(child)]
                if mapped is not None and mapped in components and mapped != ref
            }
            dependencies.setdefault(ref, set()).update(children)
    dependency_rows = [
        {"ref": ref, "dependsOn": sorted(children)}
        for ref, children in sorted(dependencies.items())
    ]
    return [components[key] for key in sorted(components)], dependency_rows


def _pnpm_lock_hashes(path: Path) -> dict[str, list[dict[str, str]]]:
    """Extract registry integrity hashes from a pnpm v9 lock without a YAML dependency."""
    values: dict[str, list[dict[str, str]]] = {}
    in_packages = False
    current_purl: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "packages:":
            in_packages = True
            continue
        if in_packages and line and not line.startswith(" "):
            break
        if not in_packages:
            continue
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            key = line[2:-1].strip()
            if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
                key = key[1:-1]
            package, separator, version = key.rpartition("@")
            if not separator or not package or not version:
                raise SbomError(f"cannot parse pnpm package identity in {path}: {key}")
            current_purl = f"pkg:npm/{quote(package, safe='/')}@{version}"
            continue
        match = re.search(r"\bintegrity:\s*([A-Za-z0-9]+)-([A-Za-z0-9+/=]+)", line)
        if match is None:
            continue
        if current_purl is None:
            raise SbomError(f"pnpm integrity appears before a package identity in {path}")
        algorithm = match.group(1).lower()
        algorithm_names = {"sha256": "SHA-256", "sha384": "SHA-384", "sha512": "SHA-512"}
        if algorithm not in algorithm_names:
            raise SbomError(f"unsupported pnpm integrity algorithm in {path}: {algorithm}")
        try:
            content = base64.b64decode(match.group(2), validate=True).hex()
        except ValueError as exc:
            raise SbomError(f"invalid pnpm integrity encoding in {path}") from exc
        values.setdefault(current_purl, []).append(
            {"alg": algorithm_names[algorithm], "content": content}
        )
    if not values:
        raise SbomError(f"pnpm lock contains no package integrity hashes: {path}")
    return values


def _npm_lock_hashes(path: Path) -> dict[str, list[dict[str, str]]]:
    """Extract package integrity hashes from an npm lockfile v2/v3 package table."""
    lock: Any = json.loads(path.read_text(encoding="utf-8"))
    packages = lock.get("packages") if isinstance(lock, dict) else None
    if not isinstance(packages, dict):
        raise SbomError(f"npm lock has no package table: {path}")
    values: dict[str, list[dict[str, str]]] = {}
    for location, package in packages.items():
        if not location or not isinstance(location, str) or not isinstance(package, dict):
            continue
        name = location.rsplit("node_modules/", 1)[-1]
        version = package.get("version")
        integrity = package.get("integrity")
        if not isinstance(version, str) or not isinstance(integrity, str):
            raise SbomError(f"npm lock package lacks version or integrity: {location}")
        match = re.fullmatch(r"([A-Za-z0-9]+)-([A-Za-z0-9+/=]+)", integrity)
        if match is None:
            raise SbomError(f"invalid npm integrity value in {path}: {location}")
        algorithm = match.group(1).lower()
        algorithm_names = {"sha256": "SHA-256", "sha384": "SHA-384", "sha512": "SHA-512"}
        if algorithm not in algorithm_names:
            raise SbomError(f"unsupported npm integrity algorithm in {path}: {algorithm}")
        try:
            content = base64.b64decode(match.group(2), validate=True).hex()
        except ValueError as exc:
            raise SbomError(f"invalid npm integrity encoding in {path}: {location}") from exc
        purl = f"pkg:npm/{quote(name, safe='/')}@{version}"
        values.setdefault(purl, []).append({"alg": algorithm_names[algorithm], "content": content})
    if not values:
        raise SbomError(f"npm lock contains no package integrity hashes: {path}")
    return values


def _uv_lock_hashes(path: Path) -> dict[str, list[dict[str, str]]]:
    """Extract every registry artifact digest retained by the exact uv lock."""
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    values: dict[str, list[dict[str, str]]] = {}
    for package in lock.get("package", []):
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        purl = f"pkg:pypi/{quote(normalized, safe='')}@{version}"
        artifacts: list[Any] = []
        sdist = package.get("sdist")
        if isinstance(sdist, dict):
            artifacts.append(sdist)
        wheels = package.get("wheels")
        if isinstance(wheels, list):
            artifacts.extend(wheels)
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("hash"), str):
                continue
            algorithm, separator, content = artifact["hash"].partition(":")
            algorithms = {"sha256": "SHA-256", "sha384": "SHA-384", "sha512": "SHA-512"}
            if not separator or algorithm not in algorithms or HEX64.fullmatch(content) is None:
                raise SbomError(f"unsupported or malformed uv artifact hash for {name} {version}")
            values.setdefault(purl, []).append({"alg": algorithms[algorithm], "content": content})
    return {purl: _dedupe_objects(hashes) for purl, hashes in values.items()}


def _enrich_lock_metadata(components: list[dict[str, Any]]) -> None:
    lock_hashes: dict[str, list[dict[str, str]]] = {}
    for path in (
        ROOT / "ui" / "pnpm-lock.yaml",
        ROOT / "resolve-plugin" / "com.hawavoclean.resolve" / "pnpm-lock.yaml",
    ):
        for purl, hashes in _pnpm_lock_hashes(path).items():
            lock_hashes.setdefault(purl, []).extend(hashes)
    for purl, hashes in _npm_lock_hashes(
        ROOT / "resolve-plugin" / "toolchain" / "package-lock.json"
    ).items():
        lock_hashes.setdefault(purl, []).extend(hashes)
    for purl, hashes in _uv_lock_hashes(ROOT / "uv.lock").items():
        lock_hashes.setdefault(purl, []).extend(hashes)

    for component in components:
        component_purl = component.get("purl")
        purl_base = component_purl.partition("?")[0] if isinstance(component_purl, str) else ""
        additions = lock_hashes.get(purl_base, [])
        existing_hashes = component.get("hashes", [])
        if not isinstance(existing_hashes, list):
            raise SbomError(f"component has malformed hashes: {component.get('name')}")
        if additions:
            component["hashes"] = _dedupe_objects([*existing_hashes, *additions])
            properties = component.setdefault("properties", [])
            if not isinstance(properties, list):
                raise SbomError(f"component has malformed properties: {component.get('name')}")
            source = "pnpm-lock" if purl_base.startswith("pkg:npm/") else "uv-lock"
            properties.append({"name": "hawavoclean:integrity-source", "value": source})
            component["properties"] = _dedupe_objects(properties)
        if not component.get("licenses"):
            component["licenses"] = _license("NOASSERTION")
            properties = component.setdefault("properties", [])
            if not isinstance(properties, list):
                raise SbomError(f"component has malformed properties: {component.get('name')}")
            properties.append(
                {"name": "hawavoclean:license-metadata", "value": "not-asserted-by-source"}
            )
            component["properties"] = _dedupe_objects(properties)


def _license(value: str) -> list[dict[str, dict[str, str]]]:
    if value in {"MIT", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0"}:
        return [{"license": {"id": value}}]
    return [{"license": {"name": value}}]


def _file_component(path: Path, license_name: str, *, kind: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    digest = _sha256(path)
    return {
        "bom-ref": f"urn:hawavoclean:file:sha256:{digest}",
        "type": "file",
        "name": relative,
        "hashes": [{"alg": "SHA-256", "content": digest}],
        "licenses": _license(license_name),
        "properties": [{"name": "hawavoclean:inventory-kind", "value": kind}],
    }


def _directory_inventory(path: Path) -> dict[str, Any]:
    """Hash a directory as a path-independent, non-escaping canonical tree."""
    if path.is_symlink() or not path.is_dir():
        raise SbomError(f"directory artifact is not a real directory: {path}")
    root = path.resolve()
    records: list[dict[str, Any]] = []
    regular_files = 0
    total_size = 0
    for current, raw_directories, raw_files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories: list[str] = []
        for name in sorted(raw_directories):
            entry = current_path / name
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                target = os.readlink(entry)
                _validate_tree_symlink(root, entry, target)
                records.append({"path": relative, "target": target, "type": "symlink"})
            else:
                mode = format(stat.S_IMODE(entry.stat(follow_symlinks=False).st_mode), "04o")
                records.append({"mode": mode, "path": relative, "type": "directory"})
                directories.append(name)
        raw_directories[:] = directories
        for name in sorted(raw_files):
            entry = current_path / name
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                target = os.readlink(entry)
                _validate_tree_symlink(root, entry, target)
                records.append({"path": relative, "target": target, "type": "symlink"})
                continue
            metadata = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise SbomError(f"artifact tree contains an unsupported file type: {relative}")
            file_digest = _sha256(entry)
            records.append(
                {
                    "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                    "path": relative,
                    "sha256": file_digest,
                    "size": metadata.st_size,
                    "type": "file",
                }
            )
            regular_files += 1
            total_size += metadata.st_size
    records.sort(key=lambda value: (str(value["path"]), str(value["type"])))
    tree_digest = hashlib.sha256(b"hawavoclean-artifact-tree-v1\n")
    for record in records:
        tree_digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        tree_digest.update(b"\n")
    return {
        "digest": tree_digest.hexdigest(),
        "entries": len(records),
        "regular_files": regular_files,
        "total_size": total_size,
    }


def _validate_tree_symlink(root: Path, entry: Path, target: str) -> None:
    if Path(target).is_absolute():
        raise SbomError(f"artifact tree contains an absolute symlink: {entry.relative_to(root)}")
    resolved = (entry.parent / target).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SbomError(
            f"artifact tree symlink escapes its root: {entry.relative_to(root)}"
        ) from exc
    if not resolved.exists():
        raise SbomError(f"artifact tree contains a dangling symlink: {entry.relative_to(root)}")


def _model_components() -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    components: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, set[str]] = {}
    weight_refs: dict[str, str] = {}
    for path in sorted(MODEL_ROOT.rglob("*")):
        if not path.is_file():
            continue
        is_weight = "deepfilternet3" in path.parts
        license_name = "MIT" if is_weight else "Proprietary / All Rights Reserved"
        component = _file_component(path, license_name, kind="model-resource")
        components[component["bom-ref"]] = component
        if is_weight:
            weight_refs[path.relative_to(MODEL_ROOT).as_posix()] = component["bom-ref"]

    for lock_path in sorted(MODEL_ROOT.glob("*-core.lock.toml")):
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        core_id = lock.get("core_id")
        version = lock.get("version")
        params_hash = lock.get("params_hash")
        if not all(isinstance(value, str) and value for value in (core_id, version)):
            raise SbomError(f"invalid core identity in {lock_path.name}")
        if not isinstance(params_hash, str) or HEX64.fullmatch(params_hash) is None:
            raise SbomError(f"invalid params hash in {lock_path.name}")
        core_ref = f"urn:hawavoclean:core:{core_id}@{version}"
        core = {
            "bom-ref": core_ref,
            "type": "machine-learning-model" if "model_upstream" in lock else "library",
            "name": core_id,
            "version": version,
            "description": str(lock.get("algorithm", "HawaVoClean enhancement core")),
            "hashes": [{"alg": "SHA-256", "content": params_hash}],
            "licenses": _license(str(lock.get("code_license", "NOASSERTION"))),
            "properties": [
                {"name": "hawavoclean:implementation", "value": str(lock["implementation"])},
                {"name": "hawavoclean:lock", "value": lock_path.name},
            ],
        }
        if isinstance(lock.get("model_upstream"), str):
            core["externalReferences"] = [{"type": "model-card", "url": lock["model_upstream"]}]
        components[core_ref] = core
        dependencies[core_ref] = set()
        weights = lock.get("weight_sha256", {})
        if not isinstance(weights, dict):
            raise SbomError(f"invalid weight inventory in {lock_path.name}")
        for relative, expected in weights.items():
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise SbomError(f"invalid weight entry in {lock_path.name}")
            path = MODEL_ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                raise SbomError(f"weight mismatch: {relative}")
            dependencies[core_ref].add(weight_refs[relative])
    return [components[key] for key in sorted(components)], dependencies


def _artifact_component(name: str, path: Path) -> dict[str, Any]:
    if path.is_dir():
        inventory = _directory_inventory(path)
        digest = cast(str, inventory["digest"])
        kind = "directory-tree"
        extra_properties = [
            {"name": "hawavoclean:artifact-tree-format", "value": "canonical-jsonl-v1"},
            {"name": "hawavoclean:artifact-tree-entries", "value": str(inventory["entries"])},
            {
                "name": "hawavoclean:artifact-tree-regular-files",
                "value": str(inventory["regular_files"]),
            },
            {
                "name": "hawavoclean:artifact-tree-total-size",
                "value": str(inventory["total_size"]),
            },
        ]
    else:
        digest = _sha256(path)
        kind = "file"
        extra_properties = [
            {"name": "hawavoclean:artifact-file-size", "value": str(path.stat().st_size)}
        ]
    return {
        "bom-ref": f"urn:hawavoclean:release-artifact:{name}:sha256:{digest}",
        "type": "file",
        "name": path.name,
        "hashes": [{"alg": "SHA-256", "content": digest}],
        "licenses": _license("Proprietary / All Rights Reserved"),
        "properties": [
            {"name": "hawavoclean:artifact-name", "value": name},
            {"name": "hawavoclean:artifact-filename", "value": path.name},
            {"name": "hawavoclean:artifact-kind", "value": kind},
            *extra_properties,
        ],
    }


def _parse_artifacts(values: list[str]) -> list[tuple[str, Path]]:
    artifacts: list[tuple[str, Path]] = []
    names: set[str] = set()
    paths: set[Path] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        unresolved = Path(raw_path).expanduser()
        path = unresolved.resolve()
        valid_path = (path.is_file() or path.is_dir()) and not unresolved.is_symlink()
        if (
            not separator
            or ARTIFACT_NAME.fullmatch(name) is None
            or name in names
            or path in paths
            or not valid_path
        ):
            raise SbomError(f"invalid or duplicate artifact NAME=PATH: {value}")
        names.add(name)
        paths.add(path)
        artifacts.append((name, path))
    if not artifacts:
        raise SbomError("at least one --artifact NAME=PATH is required")
    return artifacts


def _resolve_image(image: str, commit: str, version: str) -> str:
    """Resolve a local image reference and prove it belongs to this source."""
    try:
        values: Any = json.loads(_run(["docker", "image", "inspect", image]))
    except json.JSONDecodeError as exc:
        raise SbomError("Docker returned invalid image inspection JSON") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise SbomError("Docker did not resolve exactly one local image")
    value = values[0]
    image_id = value.get("Id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise SbomError("Docker image has no immutable sha256 ID")
    config = value.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        raise SbomError("Docker image has no source identity labels")
    expected = {
        "org.opencontainers.image.revision": commit,
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.created": _source_date(commit),
    }
    for label, expected_value in expected.items():
        if labels.get(label) != expected_value:
            raise SbomError(f"Docker image label {label} does not match the source release")
    return image_id


def _download_schemas(directory: Path) -> Path:
    for name, (url, expected) in SCHEMAS.items():
        destination = directory / name
        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                payload = response.read()
        except OSError as exc:
            raise SbomError(f"cannot fetch pinned CycloneDX schema {name}: {exc}") from exc
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise SbomError(f"CycloneDX schema hash mismatch for {name}: {actual}")
        destination.write_bytes(payload)
    return directory / "bom-1.6.schema.json"


def _export_source_tree(commit: str, directory: Path) -> Path:
    """Export exactly ``commit`` so ignored/untracked state cannot pollute inventory."""
    archive = directory / "source.tar"
    source = directory / "source"
    source.mkdir()
    _run(["git", "archive", "--format=tar", f"--output={archive}", commit])
    _run(["tar", "-xf", str(archive), "-C", str(source)])
    return source


def _validate_contract(bom: dict[str, Any], artifact_names: set[str]) -> None:
    components = bom.get("components", [])
    if any(not component.get("licenses") for component in components):
        raise SbomError("inventory contains a component without explicit license metadata")
    purls = {value for component in components for value in [component.get("purl")] if value}
    required_purl_fragments = ("pkg:pypi/", "pkg:npm/react@", "pkg:npm/electron@")
    for fragment in required_purl_fragments:
        if not any(fragment in str(purl) for purl in purls):
            raise SbomError(f"inventory is missing required ecosystem component: {fragment}")
    if not any(fragment in str(purl) for fragment in ("pkg:apk/", "pkg:deb/") for purl in purls):
        raise SbomError("inventory is missing required system-package ecosystem")
    for component in components:
        purl = component.get("purl")
        if (
            isinstance(purl, str)
            and purl.startswith(("pkg:apk/", "pkg:npm/", "pkg:pypi/"))
            and not component.get("hashes")
        ):
            raise SbomError(f"locked package has no cryptographic hash: {purl}")
    recorded_artifacts = {
        prop["value"]
        for component in components
        for prop in component.get("properties", [])
        if prop.get("name") == "hawavoclean:artifact-name"
    }
    if recorded_artifacts != artifact_names:
        raise SbomError("SBOM release-artifact bindings are incomplete")
    model_hashes = {
        digest["content"]
        for component in components
        if component.get("type") == "file"
        for digest in component.get("hashes", [])
    }
    for weight in MODEL_ROOT.glob("deepfilternet3/**/*"):
        if weight.is_file() and _sha256(weight) not in model_hashes:
            raise SbomError(f"vendored model is absent from SBOM: {weight.name}")


def generate(image: str, artifact_values: list[str], output: Path) -> str:
    _tool_preflight()
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=all"])
    if dirty:
        raise SbomError("refusing to bind an SBOM to a dirty source tree")
    artifacts = _parse_artifacts(artifact_values)
    commit = _run(["git", "rev-parse", "HEAD"]).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SbomError("cannot resolve the source commit")
    version = _release_version()
    image_id = _resolve_image(image, commit, version)
    with tempfile.TemporaryDirectory(prefix="hawavoclean-sbom-") as temp_name:
        temp = Path(temp_name)
        source_tree = _export_source_tree(commit, temp)
        source_bom = _trivy_bom(
            [
                "fs",
                "--include-dev-deps",
                str(source_tree),
            ],
            temp / "source.cdx.json",
        )
        _remove_scan_subject(source_bom)
        image_bom = _trivy_bom(["image", image_id], temp / "image.cdx.json")
        components, dependency_rows = _normalized_inventory([source_bom, image_bom])
        _remove_mutable_image_aliases(components)
        model_components, model_dependencies = _model_components()
        artifact_components = [_artifact_component(name, path) for name, path in artifacts]
        custom_components = [*model_components, *artifact_components]
        existing_refs = {component["bom-ref"] for component in components}
        for component in custom_components:
            if component["bom-ref"] not in existing_refs:
                components.append(component)
                existing_refs.add(component["bom-ref"])
        _enrich_lock_metadata(components)
        dependency_map = {row["ref"]: set(row.get("dependsOn", [])) for row in dependency_rows}
        for ref, children in model_dependencies.items():
            dependency_map.setdefault(ref, set()).update(children)

        root_ref = f"pkg:pypi/hawavoclean@{version}"
        root_component = {
            "bom-ref": root_ref,
            "type": "application",
            "name": "hawavoclean",
            "version": version,
            "purl": root_ref,
            "licenses": _license("Proprietary / All Rights Reserved"),
            "externalReferences": [
                {
                    "type": "vcs",
                    "url": f"https://github.com/hawzhin/HawaVoClean/tree/{commit}",
                }
            ],
            "properties": [
                {"name": "hawavoclean:source-commit", "value": commit},
                {"name": "hawavoclean:uv-lock-sha256", "value": _sha256(ROOT / "uv.lock")},
                {
                    "name": "hawavoclean:ui-lock-sha256",
                    "value": _sha256(ROOT / "ui" / "pnpm-lock.yaml"),
                },
                {
                    "name": "hawavoclean:plugin-lock-sha256",
                    "value": _sha256(
                        ROOT / "resolve-plugin" / "com.hawavoclean.resolve" / "pnpm-lock.yaml"
                    ),
                },
                {"name": "hawavoclean:container-image", "value": image_id},
                {"name": "hawavoclean:container-image-requested", "value": image},
            ],
        }
        components = [component for component in components if component["bom-ref"] != root_ref]
        components.sort(key=lambda value: str(value["bom-ref"]))
        dependency_map[root_ref] = {str(component["bom-ref"]) for component in components}
        dependencies = [
            {"ref": ref, "dependsOn": sorted(children)}
            for ref, children in sorted(dependency_map.items())
            if ref == root_ref or ref in existing_refs
        ]
        artifact_digests = {
            name: str(component["hashes"][0]["content"])
            for (name, _path), component in zip(artifacts, artifact_components, strict=True)
        }
        seed = "|".join(
            [
                commit,
                image_id,
                *sorted(f"{name}:{digest}" for name, digest in artifact_digests.items()),
            ]
        )
        bom = {
            "$schema": "http://cyclonedx.org/schema/bom-1.6.schema.json",
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, seed)}",
            "version": 1,
            "metadata": {
                "timestamp": _source_date(commit),
                "tools": {
                    "components": [
                        {
                            "type": "application",
                            "group": "aquasecurity",
                            "name": "trivy",
                            "version": TRIVY_VERSION,
                        },
                        {
                            "type": "application",
                            "name": "hawavoclean-sbom-generator",
                            "version": version,
                            "hashes": [
                                {
                                    "alg": "SHA-256",
                                    "content": _sha256(Path(__file__).resolve()),
                                }
                            ],
                        },
                    ]
                },
                "component": root_component,
                "properties": [
                    {
                        "name": "hawavoclean:cyclonedx-schema-sha256",
                        "value": SCHEMAS["bom-1.6.schema.json"][1],
                    }
                ],
            },
            "components": components,
            "dependencies": dependencies,
            "compositions": [{"aggregate": "complete", "assemblies": [root_ref]}],
        }
        _validate_contract(bom, {name for name, _path in artifacts})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(bom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        schema = _download_schemas(temp)
        _run(
            [
                "uvx",
                "--from",
                f"check-jsonschema=={VALIDATOR_VERSION}",
                "check-jsonschema",
                "--schemafile",
                str(schema),
                str(output),
            ]
        )
    digest = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="local container reference (resolved and bound to its immutable image ID)",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="release file or directory tree to hash-bind (repeatable)",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        digest = generate(args.image, args.artifact, args.output.resolve())
    except (OSError, ValueError, json.JSONDecodeError, SbomError) as exc:
        print(f"SBOM generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"CycloneDX 1.6 SBOM: {args.output} (sha256:{digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
