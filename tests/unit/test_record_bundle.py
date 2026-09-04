from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import hawavoclean.record_bundle as record_bundle_module
from hawavoclean.errors import PublicationError
from hawavoclean.hashing import hash_file
from hawavoclean.record_bundle import (
    RECORD_SIGNATURE_DOMAIN,
    SIGNATURE_NAME,
    ProcessingRecord,
    RecordTrustedKey,
    RecordTrustStore,
    create_processing_record,
    sign_processing_record,
    verify_processing_record,
)
from hawavoclean.report.schema import (
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)
from hawavoclean.report.writer import serialize_json_report
from tests.support.report_provenance import build, core, environment, guard

pytestmark = pytest.mark.unit


def _sources(root: Path) -> tuple[Path, Path, Path]:
    master = root / "user master.wav"
    report_path = root / "user master.hawavoclean.json"
    summary = root / "user master.hawavoclean.txt"
    root.mkdir(parents=True, exist_ok=True)
    sf.write(master, np.zeros(48_000, dtype=np.float32), 48_000, subtype="PCM_24")
    master_digest = hash_file(master)
    output = MediaStats(
        path="master.wav",
        sha256=master_digest,
        sample_rate=48000,
        channels=1,
        samples=48000,
        duration_s=1.0,
    )
    report = HawaVoCleanReport(
        schema_version=2,
        release=current_release_metadata(),
        build=build(),
        job_id="record-bundle-test",
        config_hash="a" * 64,
        input=output.model_copy(update={"path": "input.mp3", "sha256": "b" * 64}),
        output=output,
        core=core("core", "algorithm", "c" * 64),
        guard=guard("guard", "d" * 64, "e" * 64),
        environment=environment(),
        summary=UnitSummary(),
    )
    report_path.write_text(serialize_json_report(report), encoding="utf-8")
    summary.write_text("HawaVoClean Full Processing Record\n", encoding="utf-8")
    return master, report_path, summary


def _create(root: Path, name: str = "record.zip") -> Path:
    master, report, summary = _sources(root / "sources")
    destination = root / name
    create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=destination,
    )
    return destination


def test_processing_record_is_portable_after_sources_are_removed(tmp_path: Path) -> None:
    record_path = _create(tmp_path)
    for source in (tmp_path / "sources").iterdir():
        source.unlink()
    (tmp_path / "sources").rmdir()

    verified = verify_processing_record(record_path)
    assert verified.path == record_path.resolve()
    assert verified.authenticated_publisher is False
    assert verified.total_uncompressed_bytes > 0


def test_identical_sources_produce_identical_zip_bytes(tmp_path: Path) -> None:
    first = _create(tmp_path / "first")
    second = _create(tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()


def test_archive_has_closed_portable_names_and_canonical_manifest(tmp_path: Path) -> None:
    record_path = _create(tmp_path)
    with zipfile.ZipFile(record_path) as archive:
        assert archive.namelist() == [
            "master.wav",
            "report.json",
            "summary.txt",
            "manifest.json",
        ]
        manifest_raw = archive.read("manifest.json")
        manifest = json.loads(manifest_raw)
        assert (
            manifest_raw
            == json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )


def test_master_must_match_validated_report(tmp_path: Path) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["output"]["sha256"] = "f" * 64
    report.write_text(json.dumps(report_value), encoding="utf-8")
    with pytest.raises(PublicationError, match="does not match report"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=tmp_path / "record.zip",
        )


def test_existing_record_is_not_overwritten_without_consent(tmp_path: Path) -> None:
    record_path = _create(tmp_path)
    before = record_path.read_bytes()
    master, report, summary = _sources(tmp_path / "other")
    with pytest.raises(PublicationError, match="already exists"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=record_path,
        )
    assert record_path.read_bytes() == before


def test_noncooperating_racer_wins_no_overwrite_commit_without_being_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"
    racer = b"non-cooperating winner"
    real_verify = record_bundle_module.verify_processing_record

    def install_racer_after_temp_verification(path: Path | str) -> ProcessingRecord:
        verified = real_verify(path)
        destination.write_bytes(racer)
        return verified

    monkeypatch.setattr(
        record_bundle_module,
        "verify_processing_record",
        install_racer_after_temp_verification,
    )
    with pytest.raises(PublicationError, match="already exists"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=destination,
            overwrite=False,
        )

    assert destination.read_bytes() == racer


def test_overwrite_replaces_with_one_complete_record(tmp_path: Path) -> None:
    record_path = _create(tmp_path)
    master, report, summary = _sources(tmp_path / "other")
    result = create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=record_path,
        overwrite=True,
    )
    assert result.archive_sha256 == hash_file(record_path)
    verify_processing_record(record_path)


def test_duplicate_or_extra_zip_entry_is_rejected(tmp_path: Path) -> None:
    record_path = _create(tmp_path)
    with zipfile.ZipFile(record_path, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("extra.txt", "unexpected")
    with pytest.raises(PublicationError, match="contain exactly"):
        verify_processing_record(record_path)


def test_compressed_entry_is_rejected_before_expansion(tmp_path: Path) -> None:
    source = _create(tmp_path)
    rewritten = tmp_path / "compressed.zip"
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_DEFLATED) as destination,
    ):
        for name in original.namelist():
            destination.writestr(name, original.read(name))
    with pytest.raises(PublicationError, match="entry is unsafe"):
        verify_processing_record(rewritten)


def test_payload_tamper_is_detected(tmp_path: Path) -> None:
    source = _create(tmp_path)
    tampered = tmp_path / "tampered.zip"
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_STORED) as destination,
    ):
        for name in original.namelist():
            value = original.read(name)
            if name == "summary.txt":
                value += b"tampered"
            destination.writestr(name, value)
    with pytest.raises(PublicationError, match="(?:size|hash) mismatch"):
        verify_processing_record(tampered)


def test_source_symlink_is_rejected(tmp_path: Path) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    linked = tmp_path / "linked.wav"
    linked.symlink_to(master)
    with pytest.raises(PublicationError, match="regular file"):
        create_processing_record(
            master_path=linked,
            report_path=report,
            summary_path=summary,
            destination=tmp_path / "record.zip",
        )


def test_record_symlink_is_rejected_during_verification(tmp_path: Path) -> None:
    record_path = _create(tmp_path / "record")
    linked = tmp_path / "linked.zip"
    linked.symlink_to(record_path)
    with pytest.raises(PublicationError, match="regular file"):
        verify_processing_record(linked)


def test_record_reparse_point_is_rejected_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = _create(tmp_path)
    monkeypatch.setattr(
        record_bundle_module,
        "is_reparse_or_symlink",
        lambda candidate: candidate == record_path,
    )
    with pytest.raises(PublicationError, match="regular file"):
        verify_processing_record(record_path)


@pytest.mark.skipif(os.name == "nt", reason="Windows denies replacement of an open archive")
def test_concurrent_path_replacement_cannot_mix_archive_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = _create(tmp_path / "first")
    replacement = _create(tmp_path / "second")
    original_loader = record_bundle_module.load_json_report_bytes

    def replace_during_verification(raw: bytes) -> HawaVoCleanReport:
        report = original_loader(raw)
        os.replace(replacement, record_path)
        return report

    monkeypatch.setattr(
        record_bundle_module,
        "load_json_report_bytes",
        replace_during_verification,
    )
    with pytest.raises(PublicationError, match="changed during verification"):
        verify_processing_record(record_path)


def test_destination_must_be_a_zip(tmp_path: Path) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    with pytest.raises(PublicationError, match="end in .zip"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=tmp_path / "record.wav",
        )


def test_master_must_be_an_actual_wave_container(tmp_path: Path) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    master.write_bytes(b"RIFF" + b"not-wave" * 20)
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["output"]["sha256"] = hash_file(master)
    report.write_text(json.dumps(report_value), encoding="utf-8")

    with pytest.raises(PublicationError, match="WAVE"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=tmp_path / "record.zip",
        )


def test_invalid_temp_never_replaces_valid_prior_record_when_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"
    create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=destination,
    )
    original = destination.read_bytes()
    real_copy = record_bundle_module._copy_entry

    def mutate_after_master(*args: object, **kwargs: object) -> dict[str, object]:
        copied = real_copy(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("role") == "master":
            value = json.loads(report.read_text(encoding="utf-8"))
            value["output"]["sha256"] = "f" * 64
            report.write_text(json.dumps(value), encoding="utf-8")
        return copied

    monkeypatch.setattr(record_bundle_module, "_copy_entry", mutate_after_master)
    with pytest.raises(PublicationError, match="does not match report"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=destination,
            overwrite=True,
        )
    assert destination.read_bytes() == original
    verify_processing_record(destination)


def test_manifest_parsing_and_validation_branches() -> None:
    from hawavoclean.record_bundle import _manifest_object, _validated_manifest

    # 1. Non-canonical or malformed JSON
    with pytest.raises(PublicationError, match="invalid JSON"):
        _manifest_object(b"invalid json")
    with pytest.raises(PublicationError, match="duplicate key"):
        _manifest_object(b'{"a": 1, "a": 2}')
    with pytest.raises(PublicationError, match="must be a JSON object"):
        _manifest_object(b"[1, 2, 3]")
    with pytest.raises(PublicationError, match="not canonical JSON"):
        _manifest_object(b'{"schema_version": 1, "product": "hawavoclean"} ')

    # 2. _validated_manifest schema validation
    with pytest.raises(PublicationError, match="fields differ from schema v1"):
        _validated_manifest({"wrong": 1})

    valid_manifest = {
        "schema_version": 1,
        "product": "hawavoclean-full-processing-record",
        "files": {
            "master.wav": {"role": "master", "sha256": "a" * 64, "size_bytes": 100},
            "report.json": {"role": "report", "sha256": "b" * 64, "size_bytes": 200},
            "summary.txt": {"role": "summary", "sha256": "c" * 64, "size_bytes": 300},
        },
        "content_sha256": "bad_content_sha",
    }
    with pytest.raises(PublicationError, match="content identity does not recompute"):
        _validated_manifest(valid_manifest)

    # Missing file entry
    bad_files_manifest = dict(
        valid_manifest,
        files={"master.wav": {"role": "master", "sha256": "a" * 64, "size_bytes": 100}},
    )
    with pytest.raises(PublicationError, match="file inventory is incomplete"):
        _validated_manifest(bad_files_manifest)


def test_record_bundle_file_and_header_branches(tmp_path: Path) -> None:
    from hawavoclean.record_bundle import _regular_file, _validate_wave_header

    # 1. _validate_wave_header error branches
    with pytest.raises(PublicationError, match="not a RIFF/RF64 WAVE"):
        _validate_wave_header(b"too_short")
    with pytest.raises(PublicationError, match="not a RIFF/RF64 WAVE"):
        _validate_wave_header(b"FORM" + b"\x00" * 8)
    with pytest.raises(PublicationError, match="not a RIFF/RF64 WAVE"):
        _validate_wave_header(b"RIFF" + b"\x00" * 4 + b"AIFF")

    # Valid header
    _validate_wave_header(b"RIFF" + b"\x00" * 4 + b"WAVE")
    _validate_wave_header(b"RF64" + b"\x00" * 4 + b"WAVE")

    # 2. _regular_file error branches
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    with pytest.raises(PublicationError, match="is empty"):
        _regular_file(empty_file, label="test_file", maximum_bytes=1000)

    too_large = tmp_path / "large.txt"
    too_large.write_bytes(b"a" * 100)
    with pytest.raises(PublicationError, match="exceeds the"):
        _regular_file(too_large, label="test_file", maximum_bytes=50)

    # 3. create_processing_record destination not ending in .zip
    master, report, summary = _sources(tmp_path / "src_test")
    with pytest.raises(PublicationError, match="must end in .zip"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=tmp_path / "dest.tar",
        )

    # 4. create_processing_record refusing to overwrite source file
    src_dir = tmp_path / "src_overwrite"
    src_dir.mkdir()
    fake_master = src_dir / "master.zip"
    fake_master.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"extra audio")
    fake_report = src_dir / "report.json"
    fake_report.write_bytes(b'{"output": {"sha256": "abc"}}')
    fake_summary = src_dir / "summary.txt"
    fake_summary.write_bytes(b"summary")

    with pytest.raises(PublicationError, match="Refusing to overwrite a Processing Record source"):
        create_processing_record(
            master_path=fake_master,
            report_path=fake_report,
            summary_path=fake_summary,
            destination=fake_master,
        )


def _generate_keypair() -> tuple[Ed25519PrivateKey, str, str]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes_raw()
    public_bytes = private_key.public_key().public_bytes_raw()
    return private_key, private_bytes.hex(), public_bytes.hex()


def test_record_trusted_key_validation() -> None:
    _, _, pub_hex = _generate_keypair()

    # Valid key creation
    key = RecordTrustedKey.from_hex("hawavoclean-2026", pub_hex)
    assert key.key_id == "hawavoclean-2026"
    assert not key.revoked
    assert len(key.public_key_bytes) == 32

    # Invalid key ID format
    with pytest.raises(PublicationError, match="invalid format"):
        RecordTrustedKey.from_hex("invalid key with spaces", pub_hex)
    with pytest.raises(PublicationError, match="invalid format"):
        RecordTrustedKey.from_hex("-starts-with-dash", pub_hex)
    with pytest.raises(PublicationError, match="invalid format"):
        RecordTrustedKey.from_hex("", pub_hex)

    # Invalid key length
    with pytest.raises(PublicationError, match="must contain exactly 32 raw bytes"):
        RecordTrustedKey(key_id="test-key", public_key_bytes=b"too short")

    # Invalid revoked flag
    with pytest.raises(PublicationError, match="revoked flag must be boolean"):
        RecordTrustedKey(key_id="test-key", public_key_bytes=bytes(32), revoked="no")  # type: ignore[arg-type]

    # Invalid hex
    with pytest.raises(PublicationError, match="Invalid public key hex"):
        RecordTrustedKey.from_hex("test-key", "not hex")


def test_record_trust_store_validation() -> None:
    priv1, _, pub_hex1 = _generate_keypair()
    priv2, _, pub_hex2 = _generate_keypair()

    key1 = RecordTrustedKey.from_hex("key-1", pub_hex1)
    key2 = RecordTrustedKey.from_hex("key-2", pub_hex2, revoked=True)

    # Duplicate key
    with pytest.raises(PublicationError, match="Duplicate trusted key id"):
        RecordTrustStore([key1, key1])

    store = RecordTrustStore([key1, key2])

    # Valid signature
    message = RECORD_SIGNATURE_DOMAIN + b"canonical manifest payload"
    sig1 = priv1.sign(message)
    store.verify(key_id="key-1", signature=sig1, message=message)

    # Unknown key
    with pytest.raises(PublicationError, match="unknown signing key"):
        store.verify(key_id="key-unknown", signature=sig1, message=message)

    # Revoked key
    sig2 = priv2.sign(message)
    with pytest.raises(PublicationError, match="signing key is revoked"):
        store.verify(key_id="key-2", signature=sig2, message=message)

    # Invalid signature bytes
    with pytest.raises(PublicationError, match="signature verification failed"):
        store.verify(key_id="key-1", signature=b"\x00" * 64, message=message)


def test_create_signed_processing_record_and_offline_relocation(tmp_path: Path) -> None:
    import shutil

    priv, priv_hex, pub_hex = _generate_keypair()
    trusted_key = RecordTrustedKey.from_hex("pubkey-prod-2026", pub_hex)
    trust_store = RecordTrustStore([trusted_key])

    master, report, summary = _sources(tmp_path / "orig_sources")
    record_path = tmp_path / "records" / "authenticated_record.zip"

    # Mismatched signing arguments
    with pytest.raises(PublicationError, match="Both signing_key_id and signing_private_key"):
        create_processing_record(
            master_path=master,
            report_path=report,
            summary_path=summary,
            destination=record_path,
            signing_key_id="pubkey-prod-2026",
            signing_private_key=None,
        )

    # Create signed record directly
    created = create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=record_path,
        signing_key_id="pubkey-prod-2026",
        signing_private_key=priv,
        trust_store=trust_store,
    )
    assert created.key_id == "pubkey-prod-2026"
    assert created.authenticated_publisher is True
    assert created.signature_sha256 is not None
    assert hash_file(record_path) == created.archive_sha256

    # Verify offline in place
    verified_inplace = verify_processing_record(
        record_path, trust_store=trust_store, require_authenticated=True
    )
    assert verified_inplace.authenticated_publisher is True
    assert verified_inplace.key_id == "pubkey-prod-2026"
    assert verified_inplace.signature_sha256 == created.signature_sha256

    # RELOCATION: Move archive to a completely separate folder hierarchy
    relocated_dir = tmp_path / "airgapped_relocation" / "nested" / "vault"
    relocated_dir.mkdir(parents=True)
    relocated_path = relocated_dir / "moved_record.zip"
    record_path.rename(relocated_path)

    # Delete original source directory entirely to prove offline independence
    shutil.rmtree(tmp_path / "orig_sources")
    assert not master.exists()
    assert not report.exists()

    # Verify offline relocated record
    relocated_verified = verify_processing_record(
        relocated_path, trust_store=trust_store, require_authenticated=True
    )
    assert relocated_verified.path == relocated_path.resolve()
    assert relocated_verified.archive_sha256 == created.archive_sha256
    assert relocated_verified.master_sha256 == created.master_sha256
    assert relocated_verified.authenticated_publisher is True
    assert relocated_verified.key_id == "pubkey-prod-2026"


def test_master_wav_is_ordinary_playable_pcm_wav(tmp_path: Path) -> None:
    priv, _, pub_hex = _generate_keypair()
    trust_store = RecordTrustStore([RecordTrustedKey.from_hex("wav-test-key", pub_hex)])

    master, report, summary = _sources(tmp_path / "sources")
    record_path = tmp_path / "record.zip"
    create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=record_path,
        signing_key_id="wav-test-key",
        signing_private_key=priv,
        trust_store=trust_store,
    )

    # Extract master.wav and inspect with standard soundfile / wave parser
    extracted_master = tmp_path / "extracted_master.wav"
    with zipfile.ZipFile(record_path, "r") as archive:
        extracted_master.write_bytes(archive.read("master.wav"))

    assert hash_file(extracted_master) == hash_file(master)
    audio, sr = sf.read(extracted_master)
    info = sf.info(extracted_master)
    assert sr == 48000
    assert info.channels == 1
    assert info.format == "WAV"
    assert info.subtype == "PCM_24"
    assert len(audio) == 48000


def test_sign_processing_record_inplace_and_to_destination(tmp_path: Path) -> None:
    priv, priv_hex, pub_hex = _generate_keypair()
    trust_store = RecordTrustStore([RecordTrustedKey.from_hex("signer-key", pub_hex)])

    master, report, summary = _sources(tmp_path / "sources")
    unsigned_path = tmp_path / "unsigned.zip"
    unsigned = create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=unsigned_path,
    )
    assert unsigned.key_id is None
    assert unsigned.authenticated_publisher is False

    # Sign to a new destination
    signed_dest = tmp_path / "signed_dest.zip"
    signed_new = sign_processing_record(
        unsigned_path,
        key_id="signer-key",
        private_key=bytes.fromhex(priv_hex),
        destination=signed_dest,
        trust_store=trust_store,
    )
    assert signed_new.path == signed_dest.resolve()
    assert signed_new.key_id == "signer-key"
    assert signed_new.authenticated_publisher is True
    # Original unsigned record remains intact
    assert verify_processing_record(unsigned_path).key_id is None

    # Sign in-place
    signed_inplace = sign_processing_record(
        unsigned_path,
        key_id="signer-key",
        private_key=priv,
        destination=None,
        trust_store=trust_store,
    )
    assert signed_inplace.path == unsigned_path.resolve()
    assert signed_inplace.key_id == "signer-key"
    assert signed_inplace.authenticated_publisher is True
    assert (
        verify_processing_record(unsigned_path, trust_store=trust_store).authenticated_publisher
        is True
    )

    # Re-signing with a different key replaces signature
    priv2, _, pub_hex2 = _generate_keypair()
    trust_store2 = RecordTrustStore([RecordTrustedKey.from_hex("signer-key-2", pub_hex2)])
    resigned = sign_processing_record(
        unsigned_path,
        key_id="signer-key-2",
        private_key=priv2,
        trust_store=trust_store2,
    )
    assert resigned.key_id == "signer-key-2"
    assert resigned.authenticated_publisher is True


def test_signed_processing_record_fails_on_tampering(tmp_path: Path) -> None:
    priv, _, pub_hex = _generate_keypair()
    trust_store = RecordTrustStore([RecordTrustedKey.from_hex("test-key", pub_hex)])

    master, report, summary = _sources(tmp_path / "sources")
    record_path = tmp_path / "record.zip"
    create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=record_path,
        signing_key_id="test-key",
        signing_private_key=priv,
        trust_store=trust_store,
    )

    # 1. Tamper manifest.sig: change signature
    with zipfile.ZipFile(record_path, "r") as r_zip:
        entries = {name: r_zip.read(name) for name in r_zip.namelist()}

    env = json.loads(entries[SIGNATURE_NAME].decode("utf-8"))
    import base64

    env["signature"] = base64.b64encode(b"\x00" * 64).decode("utf-8")
    tampered_sig_bytes = json.dumps(env, sort_keys=True, separators=(",", ":")).encode("utf-8")

    tampered_zip = tmp_path / "tampered_sig.zip"
    with zipfile.ZipFile(tampered_zip, "w") as w_zip:
        for name in ("master.wav", "report.json", "summary.txt", "manifest.json"):
            w_zip.writestr(name, entries[name])
        w_zip.writestr(SIGNATURE_NAME, tampered_sig_bytes)

    with pytest.raises(PublicationError, match="signature verification failed"):
        verify_processing_record(tampered_zip, trust_store=trust_store)

    # 2. Tamper manifest.json without updating manifest.sig
    tampered_manifest_zip = tmp_path / "tampered_manifest.zip"
    manifest_data = json.loads(entries["manifest.json"].decode("utf-8"))
    manifest_data["files"]["summary.txt"]["sha256"] = "0" * 64
    tampered_manifest_bytes = json.dumps(
        manifest_data, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    with zipfile.ZipFile(tampered_manifest_zip, "w") as w_zip:
        w_zip.writestr("master.wav", entries["master.wav"])
        w_zip.writestr("report.json", entries["report.json"])
        w_zip.writestr("summary.txt", entries["summary.txt"])
        w_zip.writestr("manifest.json", tampered_manifest_bytes)
        w_zip.writestr(SIGNATURE_NAME, entries[SIGNATURE_NAME])

    with pytest.raises(PublicationError):
        verify_processing_record(tampered_manifest_zip, trust_store=trust_store)

    # 3. Require authenticated fails closed on unsigned record
    unsigned_path = tmp_path / "unsigned.zip"
    create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=unsigned_path,
    )
    with pytest.raises(PublicationError, match="has no publisher signature"):
        verify_processing_record(unsigned_path, require_authenticated=True)

    # 4. Require authenticated fails closed if signed record verified without trust store
    with pytest.raises(
        PublicationError, match="No trust store provided to authenticate publisher signature"
    ):
        verify_processing_record(record_path, require_authenticated=True)


def test_signed_processing_record_fails_on_unknown_or_revoked_key(tmp_path: Path) -> None:
    priv, _, pub_hex = _generate_keypair()
    master, report, summary = _sources(tmp_path / "sources")
    record_path = tmp_path / "record.zip"
    create_processing_record(
        master_path=master,
        report_path=report,
        summary_path=summary,
        destination=record_path,
        signing_key_id="revoked-or-unknown-key",
        signing_private_key=priv,
    )

    # Unknown key in trust store
    other_priv, _, other_pub_hex = _generate_keypair()
    store_unknown = RecordTrustStore([RecordTrustedKey.from_hex("other-key", other_pub_hex)])
    with pytest.raises(PublicationError, match="unknown signing key"):
        verify_processing_record(record_path, trust_store=store_unknown)

    # Revoked key in trust store
    store_revoked = RecordTrustStore(
        [RecordTrustedKey.from_hex("revoked-or-unknown-key", pub_hex, revoked=True)]
    )
    with pytest.raises(PublicationError, match="signing key is revoked"):
        verify_processing_record(record_path, trust_store=store_revoked)


def test_parse_signature_envelope_validation() -> None:
    import base64

    # 1. exceeds MAX_SIGNATURE_BYTES
    with pytest.raises(PublicationError, match="exceeds the 16 KiB"):
        record_bundle_module._parse_signature_envelope(b"x" * 20_000)

    # 2. invalid JSON
    with pytest.raises(PublicationError, match="invalid JSON"):
        record_bundle_module._parse_signature_envelope(b"not json")

    # 3. not a dict
    with pytest.raises(PublicationError, match="must be a JSON object"):
        record_bundle_module._parse_signature_envelope(b"[1, 2, 3]")

    # 4. wrong fields
    with pytest.raises(PublicationError, match="fields do not match"):
        record_bundle_module._parse_signature_envelope(b'{"schema_version": 1}')

    # 5. unsupported schema version
    bad_ver = json.dumps(
        {"schema_version": 2, "algorithm": "Ed25519", "key_id": "k", "signature": "s"}
    ).encode()
    with pytest.raises(PublicationError, match="Unsupported manifest.sig schema version"):
        record_bundle_module._parse_signature_envelope(bad_ver)

    # 6. bad algorithm
    bad_algo = json.dumps(
        {"schema_version": 1, "algorithm": "RSA", "key_id": "k", "signature": "s"}
    ).encode()
    with pytest.raises(PublicationError, match="must use Ed25519"):
        record_bundle_module._parse_signature_envelope(bad_algo)

    # 7. invalid key_id
    bad_key = json.dumps(
        {"schema_version": 1, "algorithm": "Ed25519", "key_id": "bad key!", "signature": "s"}
    ).encode()
    with pytest.raises(PublicationError, match="key_id is invalid"):
        record_bundle_module._parse_signature_envelope(bad_key)

    # 8. signature not string
    bad_sig_type = json.dumps(
        {"schema_version": 1, "algorithm": "Ed25519", "key_id": "key1", "signature": 123}
    ).encode()
    with pytest.raises(PublicationError, match="signature must be base64 text"):
        record_bundle_module._parse_signature_envelope(bad_sig_type)

    # 9. signature not canonical base64
    bad_b64 = json.dumps(
        {
            "schema_version": 1,
            "algorithm": "Ed25519",
            "key_id": "key1",
            "signature": "not-base-64!!!",
        }
    ).encode()
    with pytest.raises(PublicationError, match="not canonical base64"):
        record_bundle_module._parse_signature_envelope(bad_b64)

    # 10. signature wrong length
    bad_len = json.dumps(
        {
            "schema_version": 1,
            "algorithm": "Ed25519",
            "key_id": "key1",
            "signature": base64.b64encode(b"short").decode(),
        }
    ).encode()
    with pytest.raises(PublicationError, match="must be exactly 64 bytes"):
        record_bundle_module._parse_signature_envelope(bad_len)

    # 11. not canonical JSON (extra spacing)
    valid_env = {
        "algorithm": "Ed25519",
        "key_id": "key1",
        "schema_version": 1,
        "signature": base64.b64encode(bytes(64)).decode(),
    }
    non_canonical = json.dumps(valid_env, indent=4).encode()
    with pytest.raises(PublicationError, match="not canonical JSON"):
        record_bundle_module._parse_signature_envelope(non_canonical)


def test_sign_manifest_bytes_validation() -> None:
    priv, priv_hex, _ = _generate_keypair()
    manifest = b'{"test": 1}'

    # Invalid key ID format
    with pytest.raises(PublicationError, match="Invalid key_id format"):
        record_bundle_module._sign_manifest_bytes(manifest, key_id="invalid key!", private_key=priv)

    # Raw bytes wrong length
    with pytest.raises(PublicationError, match="must be exactly 32 bytes"):
        record_bundle_module._sign_manifest_bytes(
            manifest, key_id="valid-key", private_key=b"short"
        )

    # Invalid raw bytes
    with pytest.raises(
        PublicationError, match="private_key must be Ed25519PrivateKey or 32 raw bytes"
    ):
        record_bundle_module._sign_manifest_bytes(
            manifest,
            key_id="valid-key",
            private_key="not bytes",  # type: ignore[arg-type]
        )

    # Valid raw bytes
    signed = record_bundle_module._sign_manifest_bytes(
        manifest, key_id="valid-key", private_key=bytes.fromhex(priv_hex)
    )
    assert len(signed) > 0
