from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.record_bundle as record_bundle_module
from hawavoclean.errors import PublicationError
from hawavoclean.hashing import hash_file
from hawavoclean.record_bundle import (
    ProcessingRecord,
    create_processing_record,
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
