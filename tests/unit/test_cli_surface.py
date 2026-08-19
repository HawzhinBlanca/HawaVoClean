"""CLI surface coverage: every command path, in-process, with real exit codes."""

import sys
from pathlib import Path
from typing import Any

import pytest

import voiceclean.cli as cli
from voiceclean.errors import ExitCode

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["voiceclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def test_version_flag(monkeypatch: Any) -> None:
    assert _run_cli(monkeypatch, "--version") == 0


def test_doctor_via_main(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run_cli(monkeypatch, "doctor") == 0


def test_process_missing_input_is_invalid_user_input(monkeypatch: Any, tmp_path: Path) -> None:
    rc = _run_cli(monkeypatch, "process", str(tmp_path / "nope.wav"), "-o", str(tmp_path / "o.wav"))
    assert rc == int(ExitCode.INVALID_USER_INPUT)


def test_process_refuses_overwrite(monkeypatch: Any, tmp_path: Path) -> None:
    dest = tmp_path / "exists.wav"
    dest.write_bytes(b"x")
    rc = _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(dest))
    assert rc == int(ExitCode.PUBLICATION_FAILURE)


def test_verify_missing_files(monkeypatch: Any, tmp_path: Path) -> None:
    rc = _run_cli(
        monkeypatch, "verify", str(tmp_path / "missing.wav"), "-r", str(tmp_path / "m.json")
    )
    assert rc == int(ExitCode.INVALID_USER_INPUT)

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    rc = _run_cli(monkeypatch, "verify", str(audio), "-r", str(tmp_path / "m.json"))
    assert rc == int(ExitCode.INVALID_USER_INPUT)


def test_verify_checksum_mismatch(monkeypatch: Any, tmp_path: Path) -> None:
    out = tmp_path / "v.wav"
    rc = _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(out), "--overwrite")
    assert rc == 0
    report = tmp_path / "v.voiceclean.json"
    # Corrupt the audio after publication: verify must fail on checksum.
    with open(out, "r+b") as f:
        f.seek(1000)
        f.write(b"\x00\x01\x02\x03")
    rc = _run_cli(monkeypatch, "verify", str(out), "-r", str(report))
    assert rc == int(ExitCode.PUBLICATION_FAILURE)


def test_verify_happy_path(monkeypatch: Any, tmp_path: Path) -> None:
    out = tmp_path / "h.wav"
    assert _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(out), "--overwrite") == 0
    rc = _run_cli(monkeypatch, "verify", str(out), "-r", str(tmp_path / "h.voiceclean.json"))
    assert rc == 0


def test_audit_models_clean_and_tampered(monkeypatch: Any, tmp_path: Path) -> None:
    assert _run_cli(monkeypatch, "audit-models") == 0

    # Tampered params -> non-zero. Copy resources into an override dir.
    import shutil

    from voiceclean.paths import models_dir

    real_models = models_dir()  # capture BEFORE the env override
    override = tmp_path / "models"
    shutil.copytree(real_models, override)
    lock = override / "production-core.lock.toml"
    lock.write_text(lock.read_text().replace("dd_alpha = 0.95", "dd_alpha = 0.5"))
    monkeypatch.setenv("VOICECLEAN_MODEL_DIR", str(override))
    assert _run_cli(monkeypatch, "audit-models") == int(ExitCode.PREFLIGHT_FAILURE)

    # Disallowed license -> non-zero
    shutil.rmtree(override)
    shutil.copytree(real_models, override)
    lock.write_text(
        lock.read_text().replace(
            'code_license = "Proprietary / All Rights Reserved"', 'code_license = "SSPL-1.0"'
        )
    )
    assert _run_cli(monkeypatch, "audit-models") == int(ExitCode.PREFLIGHT_FAILURE)

    # Tampered calibration id -> non-zero
    shutil.rmtree(override)
    shutil.copytree(real_models, override)
    calib = override / "guard-calibration.json"
    calib.write_text(
        calib.read_text().replace('"max_timing_drift_ms": 75.0', '"max_timing_drift_ms": 200.0')
    )
    assert _run_cli(monkeypatch, "audit-models") == int(ExitCode.PREFLIGHT_FAILURE)


def test_blind_abx_command(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(REPO)
    rc = _run_cli(
        monkeypatch,
        "blind-abx",
        "--manifest",
        "data/acceptance/manifest.json",
        "--output",
        str(tmp_path / "abx.json"),
    )
    assert rc == 0
    assert (tmp_path / "abx.json").exists()


def test_benchmark_command(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(REPO)
    rc = _run_cli(
        monkeypatch,
        "benchmark",
        "--manifest",
        "data/calibration/manifest.json",
        "--output",
        str(tmp_path / "bench.json"),
    )
    assert rc == 0
    import json

    data = json.loads((tmp_path / "bench.json").read_text())
    assert data["measured"]["units_total"] > 0
