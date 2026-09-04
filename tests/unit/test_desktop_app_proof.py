from __future__ import annotations

import hashlib
import json
import plistlib
import struct
from pathlib import Path

import pytest

from scripts import validate_desktop_app as desktop_proof

pytestmark = pytest.mark.unit
SOURCE = "a" * 40


def _write_asar(path: Path, header: dict[str, object]) -> str:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = b"\x00" * ((-len(header_bytes)) % 4)
    payload = struct.pack("<i", len(header_bytes)) + header_bytes + padding
    header_pickle = struct.pack("<I", len(payload)) + payload
    size_pickle = struct.pack("<II", 4, len(header_pickle))
    path.write_bytes(size_pickle + header_pickle)
    return hashlib.sha256(header_bytes).hexdigest()


def _write_app(root: Path, *, engine_mode: str) -> Path:
    app = root / "HawaVoClean.app"
    contents = app / "Contents"
    resources = contents / "Resources"
    executable = contents / "MacOS" / "HawaVoClean"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"mach-o-placeholder")
    executable.chmod(0o755)
    (resources / "ui").mkdir(parents=True)
    (resources / "ui" / "index.html").write_text("ui", encoding="utf-8")
    asar_header_sha256 = _write_asar(resources / "app.asar", {"files": {}})
    (resources / "icon.icns").write_bytes(b"icns" + b"\x00" * 12000)
    info = {
        "CFBundleIdentifier": "com.hawavoclean.desktop",
        "CFBundleShortVersionString": "3.3.0",
        "LSMinimumSystemVersion": "14.0.0",
        "CFBundleExecutable": "HawaVoClean",
        "CFBundleIconFile": "icon.icns",
        "ElectronAsarIntegrity": {
            "Resources/app.asar": {"algorithm": "SHA256", "hash": asar_header_sha256}
        },
    }
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(info, stream)

    engine = resources / "engine"
    engine.mkdir()
    engine_manifest_sha256: str | None = None
    if engine_mode == "shell-only":
        (engine / "README.txt").write_text("intentionally absent", encoding="utf-8")
    else:
        launcher = engine / "hawavoclean-engine"
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o755)
        (engine / "payload").write_bytes(b"payload")
        (engine / "payload-link").symlink_to("payload")
        manifest = {
            "bundle_schema_version": 1,
            "artifact_type": "resolve-engine-directory",
            "product_version": "3.3.0",
            "source_revision": SOURCE,
            "platform": "macos-arm64",
        }
        manifest_path = engine / "ENGINE-MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (engine / "ENGINE-SYMLINKS").write_text("./payload-link\tpayload\n", encoding="utf-8")
        files = [
            engine / "ENGINE-MANIFEST.json",
            engine / "ENGINE-SYMLINKS",
            launcher,
            engine / "payload",
        ]
        (engine / "ENGINE-SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"./{path.relative_to(engine).as_posix()}\n"
                for path in files
            ),
            encoding="utf-8",
        )
        engine_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    provenance = {
        "schema_version": 1,
        "artifact_type": "unsigned-macos-app-proof",
        "distribution_eligible": False,
        "product": "hawavoclean",
        "product_version": "3.3.0",
        "source_revision": SOURCE,
        "target": "macos-arm64",
        "engine_mode": engine_mode,
        "engine_manifest_sha256": engine_manifest_sha256,
        "packaged_selftest_allowed": engine_mode == "full",
        "signing": {"developer_id": False, "notarized": False, "stapled": False},
    }
    (resources / desktop_proof.PROVENANCE_NAME).write_text(json.dumps(provenance), encoding="utf-8")
    return app


@pytest.mark.parametrize("engine_mode", ["shell-only", "full"])
def test_desktop_app_proof_binds_tree_source_and_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_mode: str
) -> None:
    app = _write_app(tmp_path, engine_mode=engine_mode)
    monkeypatch.setattr(
        desktop_proof,
        "_signing_classification",
        lambda _app: {
            "developer_id": False,
            "team_identifier": None,
            "classification": "adhoc_unsigned",
        },
    )
    result = desktop_proof.validate_app(app, SOURCE, engine_mode=engine_mode)
    assert result["status"] == "passed"
    assert result["distribution_eligible"] is False
    assert result["tree_sha256"]
    assert result["app_asar_header_sha256"] == desktop_proof._asar_header_sha256(
        app / "Contents" / "Resources" / "app.asar"
    )
    if engine_mode == "full":
        assert result["engine"]["regular_files"] == 4
        assert result["engine"]["symlinks"] == 1
    else:
        assert result["engine"] == {"status": "intentionally_absent"}


def test_desktop_app_proof_rejects_a_mislabelled_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _write_app(tmp_path, engine_mode="shell-only")
    monkeypatch.setattr(desktop_proof, "_signing_classification", lambda _app: {})
    with pytest.raises(desktop_proof.DesktopProofError, match="provenance differs"):
        desktop_proof.validate_app(app, "c" * 40, engine_mode="shell-only")


def test_desktop_engine_inventory_rejects_unlisted_file(tmp_path: Path) -> None:
    app = _write_app(tmp_path, engine_mode="full")
    engine = app / "Contents" / "Resources" / "engine"
    (engine / "unlisted").write_bytes(b"not checksummed")
    with pytest.raises(desktop_proof.DesktopProofError, match="inventory differs"):
        desktop_proof._verify_engine(engine, SOURCE)


def test_desktop_app_proof_rejects_arbitrary_bytes_as_asar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _write_app(tmp_path, engine_mode="shell-only")
    (app / "Contents" / "Resources" / "app.asar").write_bytes(b"arbitrary-not-an-asar")
    monkeypatch.setattr(desktop_proof, "_signing_classification", lambda _app: {})
    with pytest.raises(desktop_proof.DesktopProofError, match="ASAR"):
        desktop_proof.validate_app(app, SOURCE, engine_mode="shell-only")


def test_desktop_app_proof_rejects_mismatched_asar_header_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _write_app(tmp_path, engine_mode="shell-only")
    info_path = app / "Contents" / "Info.plist"
    with info_path.open("rb") as stream:
        info = plistlib.load(stream)
    info["ElectronAsarIntegrity"]["Resources/app.asar"]["hash"] = "b" * 64
    with info_path.open("wb") as stream:
        plistlib.dump(info, stream)
    monkeypatch.setattr(desktop_proof, "_signing_classification", lambda _app: {})
    with pytest.raises(desktop_proof.DesktopProofError, match="header hash differs"):
        desktop_proof.validate_app(app, SOURCE, engine_mode="shell-only")


def test_asar_header_parser_rejects_forged_pickle_lengths(tmp_path: Path) -> None:
    archive = tmp_path / "app.asar"
    archive.write_bytes(struct.pack("<II", 4, desktop_proof.MAX_ASAR_HEADER_BYTES + 1))
    with pytest.raises(desktop_proof.DesktopProofError, match="size is invalid"):
        desktop_proof._asar_header_sha256(archive)


def test_desktop_app_proof_rejects_stock_electron_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _write_app(tmp_path, engine_mode="full")
    info_path = app / "Contents" / "Info.plist"
    with info_path.open("rb") as stream:
        info = plistlib.load(stream)
    info["CFBundleIconFile"] = "electron.icns"
    with info_path.open("wb") as stream:
        plistlib.dump(info, stream)
    monkeypatch.setattr(desktop_proof, "_signing_classification", lambda _app: {})
    with pytest.raises(
        desktop_proof.DesktopProofError, match="retains stock/ad-hoc electron.icns identity"
    ):
        desktop_proof.validate_app(app, SOURCE, engine_mode="full")


def test_desktop_app_proof_rejects_missing_branded_icon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _write_app(tmp_path, engine_mode="full")
    (app / "Contents" / "Resources" / "icon.icns").unlink()
    monkeypatch.setattr(desktop_proof, "_signing_classification", lambda _app: {})
    with pytest.raises(
        desktop_proof.DesktopProofError, match="desktop branded icon is unavailable"
    ):
        desktop_proof.validate_app(app, SOURCE, engine_mode="full")


def test_desktop_app_proof_rejects_placeholder_engine_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _write_app(tmp_path, engine_mode="full")
    engine = app / "Contents" / "Resources" / "engine"
    (engine / "README.txt").write_text("Development placeholder only.", encoding="utf-8")
    monkeypatch.setattr(desktop_proof, "_signing_classification", lambda _app: {})
    with pytest.raises(
        desktop_proof.DesktopProofError, match="placeholder engine resource detected"
    ):
        desktop_proof.validate_app(app, SOURCE, engine_mode="full")


def test_validate_archive_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zipfile

    app = _write_app(tmp_path / "stage", engine_mode="full")
    monkeypatch.setattr(
        desktop_proof,
        "_signing_classification",
        lambda _app: {
            "developer_id": False,
            "team_identifier": None,
            "classification": "adhoc_unsigned",
        },
    )
    zip_path = tmp_path / "HawaVoClean-3.3.0-mac-arm64.zip"
    import os

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(app.rglob("*")):
            arcname = f"HawaVoClean.app/{file_path.relative_to(app)}"
            if file_path.is_symlink():
                zi = zipfile.ZipInfo(arcname)
                zi.create_system = 3
                zi.external_attr = 0o120755 << 16
                zf.writestr(zi, os.readlink(file_path))
            elif file_path.is_file():
                zi = zipfile.ZipInfo.from_file(file_path, arcname=arcname)
                zi.external_attr = (file_path.stat().st_mode & 0xFFFF) << 16
                zf.writestr(zi, file_path.read_bytes())

    result = desktop_proof.validate_archive(zip_path, SOURCE, engine_mode="full")
    assert result["status"] == "passed"
    assert result["engine"]["regular_files"] == 4


def test_validate_archive_rejects_missing_app(tmp_path: Path) -> None:
    import zipfile

    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("dummy.txt", "nothing")
    with pytest.raises(desktop_proof.DesktopProofError, match="does not contain HawaVoClean.app"):
        desktop_proof.validate_archive(empty_zip, SOURCE, engine_mode="full")
