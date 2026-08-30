from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from scripts import release_candidate as candidate

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_cli_direct_entrypoint_matches_the_runbook() -> None:
    completed = subprocess.run(
        [sys.executable, os.fspath(ROOT / "scripts" / "release_candidate.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "assemble,verify,smoke" in completed.stdout


def test_candidate_smoke_launches_the_reconstructed_packaged_app(tmp_path: Path) -> None:
    app = tmp_path / "HawaVoClean.app"
    assert candidate._packaged_desktop_selftest_command(app) == [
        "node",
        os.fspath(ROOT / "desktop" / "scripts" / "packaged-selftest.cjs"),
        os.fspath(app),
    ]


def _write_gate_proof(root: Path) -> tuple[Path, Path]:
    assets = root / "retained-assets"
    assets.mkdir()
    identities: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(sorted(candidate.REQUIRED_ASSETS), start=1):
        path = assets / f"{name}.asset"
        path.write_bytes(f"asset-{index}-{name}".encode())
        identities[name] = {
            "filename": path.name,
            "kind": "test-artifact",
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
    proof: dict[str, Any] = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "source_date_epoch": 1_700_000_000,
        "status": "passed",
        "toolchain_lock_sha256": "b" * 64,
        "reproducibility": {
            "status": "passed",
            "passes": 2,
            "artifact_sha256": {
                "container-image": "c" * 64,
                "resolve-plugin": "d" * 64,
                "sbom": identities["sbom"]["sha256"],
                "sdist": identities["sdist"]["sha256"],
                "ui": "e" * 64,
                "wheel": identities["wheel"]["sha256"],
            },
            "release_asset_sha256": {
                name: identity["sha256"] for name, identity in identities.items()
            },
        },
        "candidate_inputs": {"path": "candidate-inputs", "assets": identities},
    }
    proof["proof_sha256"] = candidate._canonical_sha256(proof)
    proof_path = root / "release-gate-proof.json"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return proof_path, assets


def test_normalized_directory_archive_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        executable = root / "nested" / "run"
        executable.write_bytes(b"payload")
        executable.chmod(0o755)
        (root / "link").symlink_to("nested/run")
    os.utime(first / "nested" / "run", (10, 10))
    os.utime(second / "nested" / "run", (20, 20))

    left = tmp_path / "left.tar.gz"
    right = tmp_path / "right.tar.gz"
    candidate.normalized_directory_archive(first, left, prefix="root", epoch=123)
    candidate.normalized_directory_archive(second, right, prefix="root", epoch=123)
    assert left.read_bytes() == right.read_bytes()
    with tarfile.open(left, "r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == [
            "root",
            "root/link",
            "root/nested",
            "root/nested/run",
        ]
        assert all(member.mtime == 123 for member in archive.getmembers())
        assert all(member.uid == 0 and member.gid == 0 for member in archive.getmembers())
    reconstructed = candidate._extract_normalized_tree(left, tmp_path / "extracted", prefix="root")
    assert candidate._tree_sha256(reconstructed) == candidate._tree_sha256(first)


def test_normalized_archive_rejects_an_escaping_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (source / "escape").symlink_to(outside)
    with pytest.raises(candidate.CandidateError, match="absolute archive symlink"):
        candidate.normalized_directory_archive(
            source, tmp_path / "archive.tar.gz", prefix="root", epoch=123
        )


def _docker_like_tar(path: Path, entries: list[tuple[str, bytes]], *, epoch: int) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = epoch
            info.uid = epoch
            import io

            archive.addfile(info, io.BytesIO(payload))


def test_saved_container_outer_tar_is_normalized(tmp_path: Path) -> None:
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    entries = [("manifest.json", b"[]"), ("blobs/sha256/layer", b"layer")]
    _docker_like_tar(first, entries, epoch=10)
    _docker_like_tar(second, list(reversed(entries)), epoch=20)
    left = tmp_path / "left.tar"
    right = tmp_path / "right.tar"
    candidate.normalize_saved_image(first, left, epoch=123)
    candidate.normalize_saved_image(second, right, epoch=123)
    assert left.read_bytes() == right.read_bytes()


def test_unsigned_candidate_requires_explicit_acceptance_and_proof_binding(tmp_path: Path) -> None:
    proof, assets = _write_gate_proof(tmp_path)
    output = tmp_path / "candidate"
    manifest = candidate.assemble_candidate(proof, assets, output)
    assert manifest["status"] == "unsigned_pending_signing"
    assert manifest["schema_version"] == 2
    assert "unsigned qualification evidence" in manifest["distribution_boundary"]["desktop-proof"]
    with pytest.raises(candidate.CandidateError, match="explicit --allow-unsigned"):
        candidate.verify_candidate(output, proof_path=proof)
    verified = candidate.verify_candidate(output, proof_path=proof, allow_unsigned=True)
    assert verified["source_commit"] == "a" * 40

    asset = next((output / "assets").iterdir())
    asset.write_bytes(b"tampered")
    with pytest.raises(candidate.CandidateError, match="bytes differ from proof"):
        candidate.verify_candidate(output, proof_path=proof, allow_unsigned=True)


def test_signed_candidate_verifies_with_only_the_allowed_identity(tmp_path: Path) -> None:
    proof, assets = _write_gate_proof(tmp_path)
    key = tmp_path / "release-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", os.fspath(key)],
        check=True,
    )
    public_fields = key.with_suffix(".pub").read_text(encoding="utf-8").split()
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        f'release-owner namespaces="hawavoclean-release" {public_fields[0]} {public_fields[1]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "signed-candidate"
    manifest = candidate.assemble_candidate(
        proof,
        assets,
        output,
        signing_key=key,
        signer_identity="release-owner",
    )
    assert manifest["status"] == "signed"
    assert (
        candidate.verify_candidate(output, proof_path=proof, allowed_signers=allowed)["status"]
        == "signed"
    )

    wrong_allowed = tmp_path / "wrong_allowed_signers"
    wrong_allowed.write_text(
        f'someone-else namespaces="hawavoclean-release" {public_fields[0]} {public_fields[1]}\n',
        encoding="utf-8",
    )
    with pytest.raises(candidate.CandidateError, match="signature verification failed"):
        candidate.verify_candidate(output, proof_path=proof, allowed_signers=wrong_allowed)
