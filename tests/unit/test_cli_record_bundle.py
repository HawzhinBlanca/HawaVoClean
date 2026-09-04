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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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
        "key_id": None,
        "signature_sha256": None,
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
    assert verified["key_id"] is None
    assert verified["signature_sha256"] is None


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


def _generate_keypair() -> tuple[Ed25519PrivateKey, str, str]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes_raw()
    public_bytes = private_key.public_key().public_bytes_raw()
    return private_key, private_bytes.hex(), public_bytes.hex()


def test_cli_record_sign_inplace_and_with_json_output(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _, priv_hex, pub_hex = _generate_keypair()
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"

    # Create unsigned
    assert _run_cli(monkeypatch, *_create_args(master, report, summary, destination)) == int(
        ExitCode.SUCCESS
    )
    capsys.readouterr()

    # Sign in place with --json
    key_id = "signer-prod-2026"
    assert _run_cli(
        monkeypatch,
        "record",
        "sign",
        str(destination),
        "--key-id",
        key_id,
        "--private-key",
        priv_hex,
        "--trusted-key",
        f"{key_id}:{pub_hex}",
        "--json",
    ) == int(ExitCode.SUCCESS)

    captured = capsys.readouterr()
    signed_payload = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert signed_payload["event"] == "processing_record_signed"
    assert signed_payload["operation"] == "sign"
    assert signed_payload["key_id"] == key_id
    assert signed_payload["authenticated_publisher"] is True
    assert signed_payload["signature_sha256"] is not None


def test_cli_record_sign_to_distinct_destination_with_key_file(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _, priv_hex, pub_hex = _generate_keypair()
    master, report, summary = _sources(tmp_path / "sources")
    unsigned_path = tmp_path / "unsigned.zip"
    signed_path = tmp_path / "signed.zip"

    # Write private key to file (hex encoded)
    key_file = tmp_path / "publisher.key"
    key_file.write_text(priv_hex, encoding="utf-8")

    assert _run_cli(monkeypatch, *_create_args(master, report, summary, unsigned_path)) == int(
        ExitCode.SUCCESS
    )
    capsys.readouterr()

    # Sign to distinct destination
    key_id = "file-key-2026"
    assert _run_cli(
        monkeypatch,
        "record",
        "sign",
        str(unsigned_path),
        "--key-id",
        key_id,
        "--private-key",
        str(key_file),
        "-o",
        str(signed_path),
    ) == int(ExitCode.SUCCESS)

    human_out = capsys.readouterr().out
    assert "FULL PROCESSING RECORD SIGNED:" in human_out
    assert "Publisher authentication: UNVERIFIED" in human_out
    assert key_id in human_out
    assert signed_path.exists()

    # Verify signed_path with trusted-key
    assert _run_cli(
        monkeypatch,
        "record",
        "verify",
        str(signed_path),
        "--trusted-key",
        f"{key_id}:{pub_hex}",
        "--require-authenticated",
    ) == int(ExitCode.SUCCESS)
    verified_out = capsys.readouterr().out
    assert "Publisher authentication: VERIFIED" in verified_out


def test_cli_record_create_with_signing_flags(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _, priv_hex, pub_hex = _generate_keypair()
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "direct_signed.zip"
    key_id = "direct-key-2026"

    # Partial arguments: signing_key_id without signing_private_key fails
    assert _run_cli(
        monkeypatch,
        *_create_args(
            master,
            report,
            summary,
            destination,
            "--signing-key-id",
            key_id,
            "--json",
        ),
    ) == int(ExitCode.PUBLICATION_FAILURE)
    error_out = json.loads(capsys.readouterr().out)
    assert "Both --signing-key-id and --signing-private-key" in error_out["error"]["message"]

    # Full arguments
    assert _run_cli(
        monkeypatch,
        *_create_args(
            master,
            report,
            summary,
            destination,
            "--signing-key-id",
            key_id,
            "--signing-private-key",
            priv_hex,
            "--trusted-key",
            f"{key_id}:{pub_hex}",
            "--json",
        ),
    ) == int(ExitCode.SUCCESS)
    created_payload = json.loads(capsys.readouterr().out)
    assert created_payload["key_id"] == key_id
    assert created_payload["authenticated_publisher"] is True


def test_cli_record_verify_trust_store_json_file_and_revocation(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _, priv_hex, pub_hex = _generate_keypair()
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "record.zip"
    key_id = "trust-store-key"

    assert _run_cli(
        monkeypatch,
        *_create_args(
            master,
            report,
            summary,
            destination,
            "--signing-key-id",
            key_id,
            "--signing-private-key",
            priv_hex,
        ),
    ) == int(ExitCode.SUCCESS)
    capsys.readouterr()

    # Create JSON trust store file
    trust_store_file = tmp_path / "trust_store.json"
    trust_store_file.write_text(
        json.dumps([{"key_id": key_id, "public_key_hex": pub_hex, "revoked": False}]),
        encoding="utf-8",
    )

    # Verify using the JSON trust store file
    assert _run_cli(
        monkeypatch,
        "record",
        "verify",
        str(destination),
        "--trusted-key",
        str(trust_store_file),
        "--require-authenticated",
        "--json",
    ) == int(ExitCode.SUCCESS)
    verified = json.loads(capsys.readouterr().out)
    assert verified["authenticated_publisher"] is True

    # Revoke key via command-line argument: revoked:key_id:hex
    assert _run_cli(
        monkeypatch,
        "record",
        "verify",
        str(destination),
        "--trusted-key",
        f"revoked:{key_id}:{pub_hex}",
        "--json",
    ) == int(ExitCode.PUBLICATION_FAILURE)
    revoked_err = json.loads(capsys.readouterr().out)
    assert "is revoked" in revoked_err["error"]["message"]


def test_cli_record_verify_require_authenticated_fails_on_unsigned(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    master, report, summary = _sources(tmp_path / "sources")
    destination = tmp_path / "unsigned.zip"
    assert _run_cli(monkeypatch, *_create_args(master, report, summary, destination)) == int(
        ExitCode.SUCCESS
    )
    capsys.readouterr()

    # Verifying unsigned with --require-authenticated fails
    assert _run_cli(
        monkeypatch,
        "record",
        "verify",
        str(destination),
        "--require-authenticated",
        "--json",
    ) == int(ExitCode.PUBLICATION_FAILURE)
    err = json.loads(capsys.readouterr().out)
    assert "has no publisher signature" in err["error"]["message"]


def test_cli_key_parsing_branches(tmp_path: Path) -> None:
    priv, priv_hex, pub_hex = _generate_keypair()

    # 1. Raw 32-byte binary private key file
    raw_key_file = tmp_path / "raw_priv.bin"
    raw_key_file.write_bytes(priv.private_bytes_raw())
    assert cli._parse_private_key(str(raw_key_file)) == priv.private_bytes_raw()

    # 2. Invalid private key file
    bad_key_file = tmp_path / "bad_priv.txt"
    bad_key_file.write_text("invalid hex", encoding="utf-8")
    with pytest.raises(PublicationError, match="must contain 32 raw bytes"):
        cli._parse_private_key(str(bad_key_file))

    # 3. Invalid hex string for private key
    with pytest.raises(PublicationError, match="must be a 64-character hex string"):
        cli._parse_private_key("not a hex string")

    # 4. Raw 32-byte binary public key file
    raw_pub_file = tmp_path / "raw_pub.bin"
    raw_pub_file.write_bytes(priv.public_key().public_bytes_raw())
    items = cli._parse_trusted_key_item(f"test-key:{raw_pub_file}")
    assert len(items) == 1
    assert items[0].key_id == "test-key"
    assert items[0].public_key_bytes == priv.public_key().public_bytes_raw()

    # 5. Public key hex text file
    hex_pub_file = tmp_path / "hex_pub.txt"
    hex_pub_file.write_text(pub_hex, encoding="utf-8")
    items_hex = cli._parse_trusted_key_item(f"test-hex:{hex_pub_file}")
    assert items_hex[0].key_id == "test-hex"
    assert items_hex[0].public_key_bytes == priv.public_key().public_bytes_raw()

    # 6. Bad public key file
    bad_pub_file = tmp_path / "bad_pub.txt"
    bad_pub_file.write_text("not-hex", encoding="utf-8")
    with pytest.raises(PublicationError, match="Invalid public key in file"):
        cli._parse_trusted_key_item(f"test-bad:{bad_pub_file}")

    # 7. Invalid trusted key spec missing colon
    with pytest.raises(PublicationError, match="Invalid trusted key specification"):
        cli._parse_trusted_key_item("missing-colon-spec")

    # 8. Trust store JSON with {"keys": [...]}
    json_keys_file = tmp_path / "trust_keys_dict.json"
    json_keys_file.write_text(
        json.dumps({"keys": [{"key_id": "dict-key", "public_key_hex": pub_hex, "revoked": False}]}),
        encoding="utf-8",
    )
    dict_items = cli._parse_trusted_key_item(str(json_keys_file))
    assert len(dict_items) == 1
    assert dict_items[0].key_id == "dict-key"

    # 9. Empty trust store spec list returns None
    assert cli._parse_trust_store(None) is None
    assert cli._parse_trust_store([]) is None
