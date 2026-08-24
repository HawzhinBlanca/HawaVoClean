#!/usr/bin/env python3
"""Export, assemble, sign, and verify the exact reproducible release candidate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

from scripts import generate_sbom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "candidate-manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"
SIGNATURE_NAME = f"{CHECKSUMS_NAME}.sig"
SIGNATURE_NAMESPACE = "hawavoclean-release"
REQUIRED_ASSETS = {
    "container",
    "python-runtime-lock",
    "resolve-plugin",
    "sbom",
    "sdist",
    "ui",
    "wheel",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CandidateError(ValueError):
    """Release candidate bytes, provenance, checksums, or signature are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CandidateError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise CandidateError(f"duplicate JSON key is forbidden: {key}")
            value[key] = child
        return value

    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CandidateError(f"{label} must be an object")
    return raw


def _safe_archive_name(name: str) -> PurePosixPath:
    pure = PurePosixPath(name)
    if not name or pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name:
        raise CandidateError(f"unsafe archive member: {name!r}")
    return pure


def _tar_info(name: str, *, mode: int, epoch: int, kind: bytes, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode & 0o777
    info.size = size
    info.mtime = epoch
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def normalized_directory_archive(source: Path, output: Path, *, prefix: str, epoch: int) -> None:
    """Write a deterministic gzip-compressed POSIX tree archive."""
    if source.is_symlink():
        raise CandidateError(f"archive source cannot be a symlink: {source}")
    source = source.resolve()
    if not source.is_dir():
        raise CandidateError(f"archive source is not a real directory: {source}")
    _safe_archive_name(prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        with (
            temp.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            root_stat = source.stat()
            root_info = _tar_info(
                f"{prefix}/",
                mode=stat.S_IMODE(root_stat.st_mode),
                epoch=epoch,
                kind=tarfile.DIRTYPE,
            )
            archive.addfile(root_info)
            for path in sorted(
                source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()
            ):
                relative = path.relative_to(source).as_posix()
                name = f"{prefix}/{relative}"
                _safe_archive_name(name)
                details = path.lstat()
                mode = stat.S_IMODE(details.st_mode)
                if stat.S_ISDIR(details.st_mode):
                    archive.addfile(
                        _tar_info(f"{name}/", mode=mode, epoch=epoch, kind=tarfile.DIRTYPE)
                    )
                elif stat.S_ISREG(details.st_mode):
                    info = _tar_info(
                        name,
                        mode=mode,
                        epoch=epoch,
                        kind=tarfile.REGTYPE,
                        size=details.st_size,
                    )
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                elif stat.S_ISLNK(details.st_mode):
                    link = os.readlink(path)
                    if Path(link).is_absolute():
                        raise CandidateError(f"absolute archive symlink is forbidden: {relative}")
                    resolved = (path.parent / link).resolve()
                    try:
                        resolved.relative_to(source)
                    except ValueError as exc:
                        raise CandidateError(
                            f"archive symlink escapes its source: {relative} -> {link}"
                        ) from exc
                    if not resolved.exists():
                        raise CandidateError(f"archive symlink is dangling: {relative} -> {link}")
                    info = _tar_info(name, mode=mode, epoch=epoch, kind=tarfile.SYMTYPE)
                    info.linkname = link
                    archive.addfile(info)
                else:
                    raise CandidateError(f"unsupported archive file type: {relative}")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp, output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def normalize_saved_image(source: Path, output: Path, *, epoch: int) -> None:
    """Normalize Docker's outer image-save tar while preserving content-addressed blobs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(handle)
    temp = Path(temp_name)
    try:
        with (
            tarfile.open(source, mode="r:") as incoming,
            tarfile.open(temp, mode="w", format=tarfile.PAX_FORMAT) as outgoing,
        ):
            members = incoming.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise CandidateError("Docker image archive contains duplicate members")
            for member in sorted(members, key=lambda item: item.name):
                _safe_archive_name(member.name)
                if member.isdir():
                    info = _tar_info(
                        f"{member.name.rstrip('/')}/",
                        mode=member.mode,
                        epoch=epoch,
                        kind=tarfile.DIRTYPE,
                    )
                    outgoing.addfile(info)
                elif member.isfile():
                    stream = incoming.extractfile(member)
                    if stream is None:
                        raise CandidateError(f"cannot read Docker archive member: {member.name}")
                    info = _tar_info(
                        member.name,
                        mode=member.mode,
                        epoch=epoch,
                        kind=tarfile.REGTYPE,
                        size=member.size,
                    )
                    with stream:
                        outgoing.addfile(info, stream)
                elif member.issym() or member.islnk():
                    _safe_archive_name(member.linkname)
                    kind = tarfile.SYMTYPE if member.issym() else tarfile.LNKTYPE
                    info = _tar_info(member.name, mode=member.mode, epoch=epoch, kind=kind)
                    info.linkname = member.linkname
                    outgoing.addfile(info)
                else:
                    raise CandidateError(f"unsupported Docker archive member: {member.name}")
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _asset_identity(path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "filename": path.name,
        "kind": kind,
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def export_release_assets(
    output: Path,
    *,
    epoch: int,
    wheel: Path,
    sdist: Path,
    runtime_requirements: Path,
    ui: Path,
    resolve_plugin: Path,
    sbom: Path,
    container_image: str,
) -> dict[str, dict[str, Any]]:
    """Export the distributable surfaces and runtime lock into a deterministic directory."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise CandidateError(f"release asset directory already exists: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    raw_container = stage / ".container-raw.tar"
    try:
        wheel_target = stage / wheel.name
        sdist_target = stage / sdist.name
        runtime_lock_target = stage / "hawavoclean-runtime-requirements-3.3.0.txt"
        sbom_target = stage / "hawavoclean-3.3.0.cdx.json"
        for source, target in (
            (wheel, wheel_target),
            (sdist, sdist_target),
            (runtime_requirements, runtime_lock_target),
            (sbom, sbom_target),
        ):
            if source.is_symlink() or not source.is_file():
                raise CandidateError(f"release asset source is not a real file: {source}")
            shutil.copyfile(source, target)

        ui_target = stage / "hawavoclean-ui-3.3.0.tar.gz"
        plugin_target = stage / "hawavoclean-resolve-plugin-3.3.0-macos-arm64.tar.gz"
        container_target = stage / "hawavoclean-container-3.3.0-linux-arm64.tar"
        normalized_directory_archive(ui, ui_target, prefix="hawavoclean-ui-3.3.0", epoch=epoch)
        normalized_directory_archive(
            resolve_plugin,
            plugin_target,
            prefix="hawavoclean-resolve-plugin-3.3.0-macos-arm64",
            epoch=epoch,
        )
        completed = subprocess.run(
            ["docker", "image", "save", "--output", os.fspath(raw_container), container_image],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-3000:]
            raise CandidateError(f"Docker image export failed: {detail}")
        normalize_saved_image(raw_container, container_target, epoch=epoch)
        raw_container.unlink()

        assets = {
            "container": _asset_identity(container_target, kind="oci-docker-archive"),
            "python-runtime-lock": _asset_identity(
                runtime_lock_target, kind="hash-locked-python-requirements"
            ),
            "resolve-plugin": _asset_identity(plugin_target, kind="tar-gzip-tree"),
            "sbom": _asset_identity(sbom_target, kind="cyclonedx-json"),
            "sdist": _asset_identity(sdist_target, kind="python-sdist"),
            "ui": _asset_identity(ui_target, kind="tar-gzip-tree"),
            "wheel": _asset_identity(wheel_target, kind="python-wheel"),
        }
        os.replace(stage, output)
        return dict(sorted(assets.items()))
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_gate_proof(path: Path) -> dict[str, Any]:
    proof = _load_object(path, "release-gate proof")
    claimed = proof.get("proof_sha256")
    covered = dict(proof)
    covered.pop("proof_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(covered):
        raise CandidateError("release-gate proof canonical digest does not recompute")
    if proof.get("status") != "passed" or HEX40.fullmatch(str(proof.get("source_commit"))) is None:
        raise CandidateError("release-gate proof is not passing or source-bound")
    reproducibility = proof.get("reproducibility")
    if not isinstance(reproducibility, dict) or reproducibility.get("status") != "passed":
        raise CandidateError("release-gate proof has no passing reproducibility result")
    candidate_inputs = proof.get("candidate_inputs")
    if not isinstance(candidate_inputs, dict):
        raise CandidateError("release-gate proof did not retain candidate inputs")
    assets = candidate_inputs.get("assets")
    if not isinstance(assets, dict) or set(assets) != REQUIRED_ASSETS:
        raise CandidateError("release-gate candidate asset inventory is incomplete")
    if reproducibility.get("release_asset_sha256") != {
        name: item.get("sha256") if isinstance(item, dict) else None
        for name, item in assets.items()
    }:
        raise CandidateError("release-gate asset identities are not reproducible")
    tested = reproducibility.get("artifact_sha256")
    if not isinstance(tested, dict):
        raise CandidateError("release-gate proof has no tested artifact identities")
    for release_name, tested_name in (("wheel", "wheel"), ("sdist", "sdist"), ("sbom", "sbom")):
        identity = assets[release_name]
        if not isinstance(identity, dict) or identity.get("sha256") != tested.get(tested_name):
            raise CandidateError(f"retained {release_name} differs from the tested artifact")
    return proof


def _validate_asset_map(assets: object, directory: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(assets, dict) or set(assets) != REQUIRED_ASSETS:
        raise CandidateError("candidate asset inventory is incomplete")
    validated: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for name, raw in assets.items():
        if not isinstance(raw, dict):
            raise CandidateError(f"candidate asset identity is invalid: {name}")
        filename = raw.get("filename")
        digest = raw.get("sha256")
        size = raw.get("size")
        kind = raw.get("kind")
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or not isinstance(digest, str)
            or HEX64.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(kind, str)
            or not kind
        ):
            raise CandidateError(f"candidate asset identity is malformed: {name}")
        if filename in filenames:
            raise CandidateError(f"duplicate candidate filename: {filename}")
        filenames.add(filename)
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"candidate asset is unavailable: {filename}")
        if path.stat().st_size != size or _sha256(path) != digest:
            raise CandidateError(f"candidate asset bytes differ from proof: {filename}")
        validated[str(name)] = {
            "filename": filename,
            "kind": kind,
            "sha256": digest,
            "size": size,
        }
    return dict(sorted(validated.items()))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _extract_normalized_tree(archive_path: Path, destination: Path, *, prefix: str) -> Path:
    """Extract only the regular/dir/relative-symlink subset emitted by this module."""
    destination.mkdir(parents=True, exist_ok=False)
    expected_root = destination / prefix
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members):
            raise CandidateError(f"archive contains duplicate members: {archive_path.name}")
        for member in members:
            pure = _safe_archive_name(member.name)
            if not pure.parts or pure.parts[0] != prefix:
                raise CandidateError(f"archive member is outside its release root: {member.name}")
            if not (member.isdir() or member.isfile() or member.issym()):
                raise CandidateError(f"archive contains an unsupported member: {member.name}")
        for member in sorted(members, key=lambda item: (item.issym(), item.name)):
            relative = PurePosixPath(member.name).relative_to(prefix)
            target = destination.joinpath(prefix, *relative.parts)
            try:
                target.parent.resolve().relative_to(destination.resolve())
            except ValueError as exc:
                raise CandidateError(
                    f"archive parent escapes extraction root: {member.name}"
                ) from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(member.mode & 0o777)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise CandidateError(f"cannot read archive member: {member.name}")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(member.mode & 0o777)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                link = Path(member.linkname)
                if link.is_absolute():
                    raise CandidateError(f"archive contains an absolute symlink: {member.name}")
                resolved = (target.parent / link).resolve()
                try:
                    resolved.relative_to(expected_root.resolve())
                except ValueError as exc:
                    raise CandidateError(
                        f"archive symlink escapes release root: {member.name}"
                    ) from exc
                target.symlink_to(member.linkname)
    if expected_root.is_symlink() or not expected_root.is_dir():
        raise CandidateError(f"archive did not reconstruct its release root: {archive_path.name}")
    return expected_root


def _tree_sha256(path: Path) -> str:
    component = generate_sbom._artifact_component("candidate-tree", path)
    hashes = component.get("hashes")
    if not isinstance(hashes, list):
        raise CandidateError(f"reconstructed asset has no hash inventory: {path}")
    digest = next(
        (
            item.get("content")
            for item in hashes
            if isinstance(item, dict) and item.get("alg") == "SHA-256"
        ),
        None,
    )
    if not isinstance(digest, str):
        raise CandidateError(f"reconstructed asset has no SHA-256: {path}")
    return digest


def _run_smoke(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    record, _stdout = _capture_smoke(command, cwd=cwd)
    return record


def _capture_smoke(command: list[str], *, cwd: Path | None = None) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-3000:]
        raise CandidateError(f"candidate smoke command failed: {' '.join(command)}\n{detail}")
    record = {
        "command": command,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
    }
    return record, completed.stdout


def assemble_candidate(
    proof_path: Path,
    assets_dir: Path,
    output: Path,
    *,
    signing_key: Path | None = None,
    signer_identity: str | None = None,
) -> dict[str, Any]:
    """Assemble one proof-bound candidate; signing is required for a final status."""
    proof_path = proof_path.resolve()
    if assets_dir.is_symlink():
        raise CandidateError(f"retained asset root cannot be a symlink: {assets_dir}")
    assets_dir = assets_dir.resolve()
    if not assets_dir.is_dir():
        raise CandidateError(f"retained asset root is not a directory: {assets_dir}")
    output = output.resolve()
    if output.exists():
        raise CandidateError(f"candidate output already exists: {output}")
    proof = _validate_gate_proof(proof_path)
    candidate_inputs = proof["candidate_inputs"]
    assert isinstance(candidate_inputs, dict)
    assets = _validate_asset_map(candidate_inputs["assets"], assets_dir)
    if (signing_key is None) != (signer_identity is None):
        raise CandidateError("signing key and signer identity must be supplied together")
    if signer_identity is not None and not signer_identity.strip():
        raise CandidateError("signer identity cannot be empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        candidate_assets = stage / "assets"
        candidate_assets.mkdir()
        manifest_assets: dict[str, dict[str, Any]] = {}
        for name, identity in assets.items():
            filename = str(identity["filename"])
            shutil.copyfile(assets_dir / filename, candidate_assets / filename)
            manifest_assets[name] = {**identity, "path": f"assets/{filename}"}

        source_epoch = proof.get("source_date_epoch")
        if not isinstance(source_epoch, int) or isinstance(source_epoch, bool) or source_epoch < 0:
            raise CandidateError("release-gate proof has no valid source epoch")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "release": {"product": "hawavoclean", "version": "3.3.0"},
            "status": "signed" if signing_key is not None else "unsigned_pending_signing",
            "source_commit": proof["source_commit"],
            "source_date_epoch": source_epoch,
            "source_date": datetime.fromtimestamp(source_epoch, tz=UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "release_gate": {
                "proof_file_sha256": _sha256(proof_path),
                "proof_sha256": proof["proof_sha256"],
                "toolchain_lock_sha256": proof["toolchain_lock_sha256"],
                "reproducible_passes": proof["reproducibility"].get("passes"),
                "tested_artifact_sha256": proof["reproducibility"].get("artifact_sha256"),
            },
            "assets": manifest_assets,
            "signature": {
                "algorithm": "openssh-sshsig" if signing_key is not None else None,
                "namespace": SIGNATURE_NAMESPACE if signing_key is not None else None,
                "signer_identity": signer_identity,
                "signed_file": CHECKSUMS_NAME if signing_key is not None else None,
            },
        }
        manifest["integrity"] = {"manifest_sha256": _canonical_sha256(manifest)}
        manifest_path = stage / MANIFEST_NAME
        _write_json(manifest_path, manifest)

        sum_paths = [manifest_path, *sorted(candidate_assets.iterdir(), key=lambda path: path.name)]
        checksum_lines = [
            f"{_sha256(path)}  {path.relative_to(stage).as_posix()}" for path in sum_paths
        ]
        checksums = stage / CHECKSUMS_NAME
        checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        if signing_key is not None:
            completed = subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "sign",
                    "-f",
                    os.fspath(signing_key.resolve()),
                    "-n",
                    SIGNATURE_NAMESPACE,
                    os.fspath(checksums),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0 or not (stage / SIGNATURE_NAME).is_file():
                detail = (completed.stderr or completed.stdout)[-3000:]
                raise CandidateError(f"candidate signing failed: {detail}")
        os.replace(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_candidate(
    candidate: Path,
    *,
    proof_path: Path | None = None,
    allowed_signers: Path | None = None,
    allow_unsigned: bool = False,
) -> dict[str, Any]:
    """Verify exact files, hashes, proof binding, and the required OpenSSH signature."""
    if candidate.is_symlink():
        raise CandidateError(f"candidate cannot be a symlink: {candidate}")
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise CandidateError(f"candidate is not a real directory: {candidate}")
    manifest_path = candidate / MANIFEST_NAME
    manifest = _load_object(manifest_path, "candidate manifest")
    integrity = manifest.get("integrity")
    covered = dict(manifest)
    covered.pop("integrity", None)
    if not isinstance(integrity, dict) or integrity.get("manifest_sha256") != _canonical_sha256(
        covered
    ):
        raise CandidateError("candidate manifest digest mismatch")
    if manifest.get("release") != {"product": "hawavoclean", "version": "3.3.0"}:
        raise CandidateError("candidate targets the wrong release")
    if HEX40.fullmatch(str(manifest.get("source_commit"))) is None:
        raise CandidateError("candidate source commit is malformed")

    assets_root = candidate / "assets"
    if assets_root.is_symlink() or not assets_root.is_dir():
        raise CandidateError("candidate assets path is not a real directory")
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise CandidateError("candidate manifest has no assets")
    expected_for_validation: dict[str, dict[str, Any]] = {}
    for name, raw in assets.items():
        if not isinstance(raw, dict) or raw.get("path") != f"assets/{raw.get('filename')}":
            raise CandidateError(f"candidate asset path is malformed: {name}")
        expected_for_validation[str(name)] = {
            key: raw.get(key) for key in ("filename", "kind", "sha256", "size")
        }
    validated = _validate_asset_map(expected_for_validation, assets_root)

    expected_files = {
        MANIFEST_NAME,
        CHECKSUMS_NAME,
        "assets",
    }
    status_value = manifest.get("status")
    if status_value == "signed":
        expected_files.add(SIGNATURE_NAME)
    elif status_value == "unsigned_pending_signing" and allow_unsigned:
        pass
    elif status_value == "unsigned_pending_signing":
        raise CandidateError("candidate is unsigned; explicit --allow-unsigned is required")
    else:
        raise CandidateError("candidate signing status is invalid")
    if {path.name for path in candidate.iterdir()} != expected_files:
        raise CandidateError("candidate contains missing or unexpected top-level files")
    if any(path.is_symlink() for path in candidate.iterdir()):
        raise CandidateError("candidate top-level symlinks are forbidden")
    if {path.name for path in assets_root.iterdir()} != {
        str(identity["filename"]) for identity in validated.values()
    }:
        raise CandidateError("candidate contains missing or unexpected asset files")

    checksum_paths = [manifest_path, *sorted(assets_root.iterdir(), key=lambda path: path.name)]
    expected_checksums = (
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(candidate).as_posix()}" for path in checksum_paths
        )
        + "\n"
    )
    checksums_path = candidate / CHECKSUMS_NAME
    try:
        actual_checksums = checksums_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CandidateError(f"cannot read candidate checksums: {exc}") from exc
    if actual_checksums != expected_checksums:
        raise CandidateError("candidate checksum inventory mismatch")

    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        raise CandidateError("candidate signature metadata is malformed")
    if status_value == "signed":
        if allowed_signers is None:
            raise CandidateError("signed candidate verification requires --allowed-signers")
        identity = signature.get("signer_identity")
        if (
            signature.get("algorithm") != "openssh-sshsig"
            or signature.get("namespace") != SIGNATURE_NAMESPACE
            or signature.get("signed_file") != CHECKSUMS_NAME
            or not isinstance(identity, str)
            or not identity
        ):
            raise CandidateError("candidate signature metadata differs from the contract")
        completed = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                os.fspath(allowed_signers.resolve()),
                "-I",
                identity,
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                os.fspath(candidate / SIGNATURE_NAME),
            ],
            input=actual_checksums,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-3000:]
            raise CandidateError(f"candidate signature verification failed: {detail}")

    if proof_path is not None:
        proof_path = proof_path.resolve()
        proof = _validate_gate_proof(proof_path)
        gate = manifest.get("release_gate")
        if not isinstance(gate, dict):
            raise CandidateError("candidate release-gate binding is malformed")
        if (
            manifest["source_commit"] != proof["source_commit"]
            or gate.get("proof_file_sha256") != _sha256(proof_path)
            or gate.get("proof_sha256") != proof["proof_sha256"]
            or gate.get("toolchain_lock_sha256") != proof["toolchain_lock_sha256"]
            or gate.get("tested_artifact_sha256") != proof["reproducibility"].get("artifact_sha256")
        ):
            raise CandidateError("candidate does not bind the supplied release-gate proof")
        proof_assets = proof["candidate_inputs"]["assets"]
        if {
            name: {key: identity[key] for key in ("filename", "kind", "sha256", "size")}
            for name, identity in validated.items()
        } != proof_assets:
            raise CandidateError("candidate assets differ from the supplied release-gate proof")
    return manifest


def smoke_candidate(
    candidate_root: Path,
    proof_path: Path,
    input_path: Path,
    *,
    allowed_signers: Path | None = None,
    allow_unsigned: bool = False,
) -> dict[str, Any]:
    """Install and exercise only the retained candidate runtime artifacts."""
    candidate_root = candidate_root.resolve()
    proof_path = proof_path.resolve()
    if input_path.is_symlink():
        raise CandidateError(f"candidate smoke input cannot be a symlink: {input_path}")
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise CandidateError(f"candidate smoke input is not a real file: {input_path}")
    manifest = verify_candidate(
        candidate_root,
        proof_path=proof_path,
        allowed_signers=allowed_signers,
        allow_unsigned=allow_unsigned,
    )
    proof = _validate_gate_proof(proof_path)
    toolchain = proof.get("toolchain")
    if not isinstance(toolchain, dict):
        raise CandidateError("release-gate proof has no exact smoke toolchain")
    expected_uv = toolchain.get("uv")
    expected_python = toolchain.get("resolve-engine-python")
    if not isinstance(expected_uv, str) or not isinstance(expected_python, str):
        raise CandidateError("release-gate proof omits uv or managed Python identity")
    uv_version, uv_stdout = _capture_smoke(["uv", "--version"])
    actual_uv = uv_stdout.split()[1]
    if actual_uv != expected_uv:
        raise CandidateError(f"candidate smoke uv drift: expected {expected_uv}, got {actual_uv}")

    assets = manifest["assets"]
    assert isinstance(assets, dict)

    def asset_path(name: str) -> Path:
        identity = assets.get(name)
        path_value = identity.get("path") if isinstance(identity, dict) else None
        if not isinstance(path_value, str):
            raise CandidateError(f"candidate manifest omits asset path: {name}")
        return candidate_root / path_value

    tested = proof["reproducibility"]["artifact_sha256"]
    assert isinstance(tested, dict)
    commands: list[dict[str, Any]] = [uv_version]
    with tempfile.TemporaryDirectory(prefix="hawavoclean-candidate-smoke-") as raw_temp:
        temp = Path(raw_temp)
        ui_tree = _extract_normalized_tree(
            asset_path("ui"), temp / "ui", prefix="hawavoclean-ui-3.3.0"
        )
        plugin_tree = _extract_normalized_tree(
            asset_path("resolve-plugin"),
            temp / "plugin",
            prefix="hawavoclean-resolve-plugin-3.3.0-macos-arm64",
        )
        reconstructed = {
            "ui": _tree_sha256(ui_tree),
            "resolve-plugin": _tree_sha256(plugin_tree),
        }
        for name, digest in reconstructed.items():
            if digest != tested.get(name):
                raise CandidateError(
                    f"candidate {name} archive does not reconstruct the tested tree"
                )

        venv = temp / "venv"
        commands.append(
            _run_smoke(["uv", "venv", "--python", "3.11", "--managed-python", os.fspath(venv)])
        )
        python = venv / "bin" / "python"
        cli = venv / "bin" / "hawavoclean"
        python_version, python_stdout = _capture_smoke(
            [os.fspath(python), "-c", "import platform;print(platform.python_version())"]
        )
        commands.append(python_version)
        actual_python = python_stdout.strip()
        if actual_python != expected_python:
            raise CandidateError(
                f"candidate smoke Python drift: expected {expected_python}, got {actual_python}"
            )
        commands.append(
            _run_smoke(
                [
                    "uv",
                    "pip",
                    "sync",
                    "--python",
                    os.fspath(python),
                    "--require-hashes",
                    os.fspath(asset_path("python-runtime-lock")),
                ]
            )
        )
        commands.append(
            _run_smoke(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    os.fspath(python),
                    "--no-deps",
                    os.fspath(asset_path("wheel")),
                ]
            )
        )
        commands.append(_run_smoke([os.fspath(cli), "doctor"]))
        wheel_output = temp / "wheel-smoke.wav"
        commands.append(
            _run_smoke(
                [
                    os.fspath(cli),
                    "process",
                    os.fspath(input_path),
                    "--output",
                    os.fspath(wheel_output),
                    "--profile",
                    "production",
                    "--overwrite",
                ]
            )
        )
        commands.append(
            _run_smoke(
                [
                    os.fspath(cli),
                    "verify",
                    os.fspath(wheel_output),
                    "--report",
                    os.fspath(wheel_output.with_suffix(".hawavoclean.json")),
                ]
            )
        )

        expected_image = tested.get("container-image")
        if not isinstance(expected_image, str) or HEX64.fullmatch(expected_image) is None:
            raise CandidateError("release-gate proof has no tested container image identity")
        commands.append(
            _run_smoke(["docker", "image", "load", "--input", os.fspath(asset_path("container"))])
        )
        image_id = f"sha256:{expected_image}"
        inspected, inspected_stdout = _capture_smoke(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_id],
        )
        commands.append(inspected)
        if inspected_stdout.strip() != image_id:
            raise CandidateError(
                "loaded candidate container does not have the tested image identity"
            )
        container_dir = temp / "container"
        container_dir.mkdir()
        shutil.copyfile(input_path, container_dir / "input.wav")
        container_dir.chmod(0o777)
        docker_runtime = [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "--tmpfs",
            "/cache:rw,uid=10001,gid=10001,mode=0750,size=2g",
            "--mount",
            f"type=bind,source={container_dir},target=/work",
            image_id,
        ]
        commands.append(_run_smoke([*docker_runtime, "doctor"]))
        commands.append(
            _run_smoke(
                [
                    *docker_runtime,
                    "process",
                    "/work/input.wav",
                    "--output",
                    "/work/output.wav",
                    "--profile",
                    "production",
                    "--overwrite",
                ]
            )
        )
        commands.append(
            _run_smoke(
                [
                    *docker_runtime,
                    "verify",
                    "/work/output.wav",
                    "--report",
                    "/work/output.hawavoclean.json",
                ]
            )
        )
        wheel_digest = _sha256(wheel_output)
        container_digest = _sha256(container_dir / "output.wav")
        if (
            wheel_digest != container_digest
            or wheel_digest != tested.get("wheel-smoke-audio")
            or container_digest != tested.get("container-audio")
        ):
            raise CandidateError(
                "candidate-only wheel/container outputs differ from the exact tested audio identity"
            )
        result = {
            "schema_version": 1,
            "status": "passed",
            "source_commit": manifest["source_commit"],
            "candidate_manifest_sha256": manifest["integrity"]["manifest_sha256"],
            "input_sha256": _sha256(input_path),
            "toolchain": {"uv": actual_uv, "python": actual_python},
            "reconstructed_tree_sha256": reconstructed,
            "container_image_id": image_id,
            "wheel_output_sha256": wheel_digest,
            "container_output_sha256": container_digest,
            "commands": commands,
        }
    result["proof_sha256"] = _canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble = subparsers.add_parser("assemble", help="assemble and optionally sign a candidate")
    assemble.add_argument("--gate-proof", type=Path, required=True)
    assemble.add_argument("--assets", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--signing-key", type=Path)
    assemble.add_argument("--signer-identity")
    verify = subparsers.add_parser("verify", help="verify candidate bytes and signature")
    verify.add_argument("candidate", type=Path)
    verify.add_argument("--gate-proof", type=Path)
    verify.add_argument("--allowed-signers", type=Path)
    verify.add_argument("--allow-unsigned", action="store_true")
    smoke = subparsers.add_parser("smoke", help="install and exercise candidate-only runtimes")
    smoke.add_argument("candidate", type=Path)
    smoke.add_argument("--gate-proof", type=Path, required=True)
    smoke.add_argument("--input", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--allowed-signers", type=Path)
    smoke.add_argument("--allow-unsigned", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "assemble":
            manifest = assemble_candidate(
                args.gate_proof,
                args.assets,
                args.output,
                signing_key=args.signing_key,
                signer_identity=args.signer_identity,
            )
            print(
                f"candidate assembled ({manifest['status']}): {args.output.resolve()}",
                flush=True,
            )
        elif args.command == "verify":
            manifest = verify_candidate(
                args.candidate,
                proof_path=args.gate_proof,
                allowed_signers=args.allowed_signers,
                allow_unsigned=args.allow_unsigned,
            )
            print(
                f"candidate verified ({manifest['status']}): {args.candidate.resolve()}",
                flush=True,
            )
        else:
            output = args.output.resolve()
            if output.is_relative_to(args.candidate.resolve()):
                raise CandidateError("smoke proof must be written outside the immutable candidate")
            if output.exists():
                raise CandidateError(f"candidate smoke output already exists: {output}")
            result = smoke_candidate(
                args.candidate,
                args.gate_proof,
                args.input,
                allowed_signers=args.allowed_signers,
                allow_unsigned=args.allow_unsigned,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(output, result)
            print(f"candidate artifact smoke passed: {output}", flush=True)
    except (CandidateError, OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(f"release candidate failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
