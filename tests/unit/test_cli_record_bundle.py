"""CLI contract for portable Full Processing Record creation and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.cli as cli
from hawavoclean.errors import ExitCode, PublicationError
from hawavoclean.hashing import hash_file
from hawavoclean.record_bundle import verify_processing_record
from hawavoclean.report.schema import (
    HawaVoCleanReport,
    MediaStats,
    UnitSummary,
    current_release_metadata,
)
from hawavoclean.report.writer import serialize_json_report
from tests.support.report_provenance import build, core, environment, guard

pytestmark = pytest.mark.unit


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    return int(caught.value.code or 0)


def _sources(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    master = root / "Kurdish master.wav"
    report_path = root / "Kurdish master.hawavoclean.json"
    summary = root / "Kurdish master.hawavoclean.txt"
    sf.write(master, np.zeros(48_000, dtype=np.float32), 48_000, subtype="PCM_24")
    output = MediaStats(
        path="Kurdish master.wav",
        sha256=hash_file(master),
        sample_rate=48000,
        channels=1,
        samples=48000,
        duration_s=1.0,
    )
    report = HawaVoCleanReport(
        schema_version=2,
        release=current_release_metadata(),
        build=build(),
        job_id="record-cli-test",
        config_hash="a" * 64,
        input=output.model_copy(update={"path": "source.m4a", "sha256": "b" * 64}),
        output=output,
        core=core("core", "algorithm", "c" * 64),
        guard=guard("guard", "d" * 64, "e" * 64),
        environment=environment(),
        summary=UnitSummary(),
    )
    report_path.write_text(serialize_json_report(report), encoding="utf-8")
    summary.write_text("HawaVoClean processing summary\n", encoding="utf-8")
    return master, report_path, summary


def _create_args(
    master: Path, report: Path, summary: Path, destination: Path, *extra: str
) -> tuple[str, ...]:
    return (
        "record",
        "create",
        str(master),
        "--report",
        str(report),
        "--summary",
        str(summary),
        "--output",
        str(destination),
        *extra,
    )


def test_create_and_verify_emit_one_truthful_json_line(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record with spaces.zip"

    assert _run_cli(
        monkeypatch, *_create_args(master, report, summary, destination, "--json")
    ) == int(ExitCode.SUCCESS)
    captured = capsys.readouterr()
    created = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert created == {
        "schema_version": 1,
        "event": "processing_record_created",
        "operation": "create",
        "path": str(destination.resolve()),
        "archive_sha256": hash_file(destination),
        "content_sha256": created["content_sha256"],
        "master_sha256": hash_file(master),
        "report_sha256": hash_file(report),
        "summary_sha256": hash_file(summary),
        "total_uncompressed_bytes": created["total_uncompressed_bytes"],
        "internal_hashes_verified": True,
        "authenticated_publisher": False,
    }

    assert _run_cli(monkeypatch, "record", "verify", str(destination), "--json") == 0
    captured = capsys.readouterr()
    verified = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert verified["event"] == "processing_record_verified"
    assert verified["operation"] == "verify"
    assert verified["archive_sha256"] == created["archive_sha256"]
    assert verified["content_sha256"] == created["content_sha256"]
    assert verified["internal_hashes_verified"] is True
    assert verified["authenticated_publisher"] is False


def test_create_refuses_overwrite_and_reports_machine_readable_error(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"
    assert _run_cli(monkeypatch, *_create_args(master, report, summary, destination)) == 0
    capsys.readouterr()
    before = destination.read_bytes()

    assert _run_cli(
        monkeypatch, *_create_args(master, report, summary, destination, "--json")
    ) == int(ExitCode.PUBLICATION_FAILURE)
    captured = capsys.readouterr()
    error = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert error["event"] == "processing_record_error"
    assert error["operation"] == "create"
    assert error["error"]["code"] == "PUBLICATION_FAILURE"
    assert error["error"]["exit_code"] == int(ExitCode.PUBLICATION_FAILURE)
    assert "already exists" in error["error"]["message"]
    assert destination.read_bytes() == before


def test_explicit_overwrite_atomically_replaces_complete_record(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"
    assert _run_cli(monkeypatch, *_create_args(master, report, summary, destination)) == 0
    capsys.readouterr()
    original_digest = hash_file(destination)
    summary.write_text("A deliberately changed summary\n", encoding="utf-8")

    assert (
        _run_cli(
            monkeypatch,
            *_create_args(master, report, summary, destination, "--overwrite", "--json"),
        )
        == 0
    )
    replaced = json.loads(capsys.readouterr().out)
    assert replaced["archive_sha256"] == hash_file(destination)
    assert replaced["archive_sha256"] != original_digest

    assert _run_cli(monkeypatch, "record", "verify", str(destination), "--json") == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["archive_sha256"] == replaced["archive_sha256"]


def test_verify_rejects_corrupt_archive_with_json_error(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")

    assert _run_cli(monkeypatch, "record", "verify", str(corrupt), "--json") == int(
        ExitCode.PUBLICATION_FAILURE
    )
    captured = capsys.readouterr()
    error = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert error["event"] == "processing_record_error"
    assert error["operation"] == "verify"
    assert error["error"]["code"] == "PUBLICATION_FAILURE"
    assert "Cannot verify Full Processing Record" in error["error"]["message"]


def test_operating_system_failure_is_a_structured_publication_error(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"

    def fail_create(**_kwargs: object) -> object:
        raise OSError("simulated volume loss")

    monkeypatch.setattr(cli, "create_processing_record", fail_create)
    assert _run_cli(
        monkeypatch, *_create_args(master, report, summary, destination, "--json")
    ) == int(ExitCode.PUBLICATION_FAILURE)
    captured = capsys.readouterr()
    error = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    assert error["error"]["code"] == "PUBLICATION_FAILURE"
    assert error["error"]["message"] == (
        "Cannot create Full Processing Record: simulated volume loss"
    )
    assert not destination.exists()


def test_human_output_discloses_missing_publisher_authentication(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"
    assert _run_cli(monkeypatch, *_create_args(master, report, summary, destination)) == 0
    created = capsys.readouterr().out
    assert "FULL PROCESSING RECORD CREATED" in created
    assert "Internal hashes:        VERIFIED" in created
    assert "Publisher authentication: ABSENT" in created
    assert "not a signature" in created

    assert _run_cli(monkeypatch, "record", "verify", str(destination)) == 0
    verified = capsys.readouterr().out
    assert "FULL PROCESSING RECORD VERIFIED" in verified
    assert "Publisher authentication: ABSENT" in verified


def test_process_record_bundle_runs_inside_process_command_and_verifies_before_success(
    monkeypatch: Any, tmp_path: Path
) -> None:
    fixture_master, fixture_report, fixture_summary = _sources(tmp_path / "fixtures")
    output = tmp_path / "result.wav"
    bundle = tmp_path / "result.hawavoclean.zip"

    def fake_pipeline(**kwargs: object) -> None:
        destination = Path(str(kwargs["output_path"]))
        destination.write_bytes(fixture_master.read_bytes())
        destination.with_name(f"{destination.stem}.hawavoclean.json").write_bytes(
            fixture_report.read_bytes()
        )
        destination.with_name(f"{destination.stem}.hawavoclean.txt").write_bytes(
            fixture_summary.read_bytes()
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    args = argparse.Namespace(
        input=str(tmp_path / "input.wav"),
        output=str(output),
        config=None,
        profile="production",
        overwrite=False,
        progress_json=False,
        passes=1,
        mode="natural",
        speaker_id=None,
        cutoff="auto",
        cutoff_hz=None,
        profiles_dir=None,
        record_bundle=str(bundle),
    )

    assert cli.cmd_process(args) == int(ExitCode.SUCCESS)
    verified = verify_processing_record(bundle)
    assert verified.archive_sha256 == hash_file(bundle)
    assert verified.master_sha256 == hash_file(output)


def test_process_record_bundle_failure_never_returns_success(
    monkeypatch: Any, tmp_path: Path
) -> None:
    fixture_master, fixture_report, fixture_summary = _sources(tmp_path / "fixtures")
    output = tmp_path / "result.wav"

    def fake_pipeline(**kwargs: object) -> None:
        destination = Path(str(kwargs["output_path"]))
        destination.write_bytes(fixture_master.read_bytes())
        destination.with_name(f"{destination.stem}.hawavoclean.json").write_bytes(
            fixture_report.read_bytes()
        )
        destination.with_name(f"{destination.stem}.hawavoclean.txt").write_bytes(
            fixture_summary.read_bytes()
        )

    def fail_bundle(**_kwargs: object) -> object:
        raise PublicationError("simulated bundle publication failure")

    monkeypatch.setattr(cli, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(cli, "create_processing_record", fail_bundle)
    args = argparse.Namespace(
        input=str(tmp_path / "input.wav"),
        output=str(output),
        config=None,
        profile="production",
        overwrite=False,
        progress_json=False,
        passes=1,
        mode="natural",
        speaker_id=None,
        cutoff="auto",
        cutoff_hz=None,
        profiles_dir=None,
        record_bundle=str(tmp_path / "result.hawavoclean.zip"),
    )

    assert cli.cmd_process(args) == int(ExitCode.PUBLICATION_FAILURE)
    assert output.exists()  # Existing publication boundary: master commits first.
    assert not Path(args.record_bundle).exists()
