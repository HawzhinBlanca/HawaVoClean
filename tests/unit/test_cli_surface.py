"""CLI surface coverage: every command path, in-process, with real exit codes."""

import sys
from pathlib import Path
from typing import Any

import pytest

import hawavoclean.cli as cli
from hawavoclean.errors import ExitCode

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def test_version_flag(monkeypatch: Any) -> None:
    assert _run_cli(monkeypatch, "--version") == 0


def test_doctor_via_main(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run_cli(monkeypatch, "doctor") == 0


def test_every_offered_profile_is_actually_runnable() -> None:
    """The CLI's choices, the engine's profile list, the API's accepted values
    and the packaged configs must describe the same set.

    A profile offered in one place and missing in another is a run that fails
    after the user has already committed to it — and `doctor` cannot warn about
    a config nobody thought to ship.
    """
    import typing

    from hawavoclean.cli import PROFILE_CHOICES
    from hawavoclean.paths import profile_config_path
    from hawavoclean.server.app import PROFILES, JobRequest

    for name in PROFILE_CHOICES:
        assert profile_config_path(name).exists(), f"{name}: no packaged config"
    assert set(PROFILES) <= set(PROFILE_CHOICES), "engine offers a profile the CLI cannot run"
    accepted = set(typing.get_args(JobRequest.model_fields["profile"].annotation))
    assert set(PROFILES) <= accepted, "engine advertises a profile its API would reject"
    assert accepted <= set(PROFILE_CHOICES), "API accepts a profile that has no config"


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
    report = tmp_path / "v.hawavoclean.json"
    # Corrupt the audio after publication: verify must fail on checksum.
    with open(out, "r+b") as f:
        f.seek(1000)
        f.write(b"\x00\x01\x02\x03")
    rc = _run_cli(monkeypatch, "verify", str(out), "-r", str(report))
    assert rc == int(ExitCode.PUBLICATION_FAILURE)


def test_verify_happy_path(monkeypatch: Any, tmp_path: Path) -> None:
    out = tmp_path / "h.wav"
    assert _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(out), "--overwrite") == 0
    rc = _run_cli(monkeypatch, "verify", str(out), "-r", str(tmp_path / "h.hawavoclean.json"))
    assert rc == 0


def test_audit_models_clean_and_tampered(monkeypatch: Any, tmp_path: Path) -> None:
    assert _run_cli(monkeypatch, "audit-models") == 0

    # Tampered params -> non-zero. Copy resources into an override dir.
    import shutil

    from hawavoclean.paths import models_dir

    real_models = models_dir()  # capture BEFORE the env override
    override = tmp_path / "models"
    shutil.copytree(real_models, override)
    lock = override / "production-core.lock.toml"
    lock.write_text(lock.read_text().replace("dd_alpha = 0.95", "dd_alpha = 0.5"))
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(override))
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


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-500"])
def test_cutoff_hz_refuses_values_that_are_not_frequencies(monkeypatch: Any, bad: str) -> None:
    """A manual cutoff must be a real frequency, refused before any work starts.

    ``type=float`` took ``nan`` because Python's float() does. It then survived
    the detector's clamp -- every comparison against NaN is False -- and landed
    in the report as a value ``json.dumps`` will not emit, so a single typo
    bought a full enhancement run and then an unwritable report at the end of
    it. ``inf``, ``0`` and negatives were silently clamped to a bound the user
    never asked for and never told about.
    """
    code = _run_cli(
        monkeypatch,
        "process",
        str(FIXTURE),
        "-o",
        "/tmp/hawavoclean-cutoff-check.wav",
        "--cutoff",
        "manual",
        "--cutoff-hz",
        bad,
    )
    assert code != 0, f"--cutoff-hz {bad!r} was accepted"


def test_cutoff_hz_accepts_a_real_frequency() -> None:
    """The guard must not reject the values it exists to let through."""
    assert cli._cutoff_hz_arg("7800.0") == 7800.0
    assert cli._cutoff_hz_arg("2000") == 2000.0


def test_verify_says_whether_anything_anchors_its_answer(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """The same PASSED line covered two very different claims.

    With a committed publication bundle beside the audio, the generation's own
    digests anchor the check and a report edited in step with the audio is
    caught. With no bundle the report is simply taken as given -- so a pair an
    attacker had rewritten to agree printed the identical "VERIFICATION
    PASSED" and exited 0. Callers reading only the exit code still cannot tell
    those apart; callers reading the output now can.
    """
    published = tmp_path / "master.wav"
    assert _run_cli(monkeypatch, "process", str(FIXTURE), "-o", str(published)) == 0
    capsys.readouterr()

    report = published.with_suffix(".hawavoclean.json")
    assert _run_cli(monkeypatch, "verify", str(published), "--report", str(report)) == 0
    anchored = capsys.readouterr().out
    assert "the committed generation's own digests" in anchored

    # The same audio and report, moved away from their bundle.
    loose_audio = tmp_path / "moved" / "master.wav"
    loose_audio.parent.mkdir()
    loose_audio.write_bytes(published.read_bytes())
    loose_report = loose_audio.with_suffix(".hawavoclean.json")
    loose_report.write_bytes(report.read_bytes())

    assert _run_cli(monkeypatch, "verify", str(loose_audio), "--report", str(loose_report)) == 0
    loose = capsys.readouterr().out
    assert "no committed publication bundle" in loose
    assert "the committed generation's own digests" not in loose
