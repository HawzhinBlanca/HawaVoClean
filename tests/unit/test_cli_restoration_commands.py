"""Unit tests for the restoration-facing CLI surface.

Covers the cheap failure branches the integration suite leaves out:
restore-doctor preflight FAILs, speaker-profile validate targets and exit
codes, restoration-benchmark argument plumbing and summary formatting, and
the process/batch restore argument refusals.
"""

import importlib
import sys
from pathlib import Path

import pytest

from hawavoclean.cli import main
from hawavoclean.errors import ExitCode, InvalidUserInputError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_PROFILES = _REPO_ROOT / "profiles"


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    """Invoke the real argparse entry point; return the process exit code."""
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc_info:
        main()
    code = exc_info.value.code
    assert isinstance(code, int)
    return code


class _EngineStub:
    """Stand-in for the heavy F0/HawaRestore engines: fails fast on construction."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("engine deliberately stubbed for failure-branch test")


def _stub_heavy_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep restore-doctor failure tests fast: skip real F0 and restoration smoke runs."""
    monkeypatch.setattr("hawavoclean.restoration.f0.F0Extractor", _EngineStub)
    monkeypatch.setattr("hawavoclean.restoration.hawarestore_kd.HawaRestoreKD", _EngineStub)


# ---------------------------------------------------------------------------
# restore-doctor failure branches
# ---------------------------------------------------------------------------


def test_restore_doctor_survives_a_fresh_clone_without_the_research_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing vendors/universr must never gate restore preflight.

    vendors/ is an upstream clone carrying its own .git and is therefore
    gitignored, so it is absent in every fresh checkout — including every CI
    job. Nothing in restore mode imports it: HawaRestoreKD is self-contained
    and UniverSRBaseline is a research-only benchmark baseline. Hard-failing
    here reported a broken installation on machines whose restore install was
    complete, which is exactly how this broke the first CI run.
    """
    _stub_heavy_engines(monkeypatch)
    monkeypatch.setattr("hawavoclean.cli._repo_root", lambda: tmp_path)
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(_REPO_PROFILES))

    code = _run_cli(monkeypatch, "restore-doctor")

    out = capsys.readouterr().out
    assert "[INFO] Upstream UniverSR checkout absent" in out
    assert "not required for restore mode" in out
    # The absent checkout contributes no FAIL of its own; the only failures
    # here are the deliberately stubbed engines.
    assert "[FAIL] Upstream" not in out
    assert "[OK] Profile verified: character_01" in out
    assert code == int(ExitCode.PREFLIGHT_FAILURE)  # from the stubbed engines alone


def test_restore_doctor_warns_but_does_not_gate_on_a_licence_less_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A vendored checkout without its LICENSE is reported, not treated as a gate."""
    _stub_heavy_engines(monkeypatch)
    (tmp_path / "vendors" / "universr").mkdir(parents=True)
    monkeypatch.setattr("hawavoclean.cli._repo_root", lambda: tmp_path)
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(_REPO_PROFILES))

    _run_cli(monkeypatch, "restore-doctor")

    out = capsys.readouterr().out
    assert "[WARN] Upstream UniverSR checkout has no LICENSE file" in out
    assert "[FAIL] Upstream" not in out


def test_restore_doctor_reports_a_complete_research_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A vendored checkout with its LICENSE is reported as present, with the pin."""
    _stub_heavy_engines(monkeypatch)
    vendor = tmp_path / "vendors" / "universr"
    vendor.mkdir(parents=True)
    (vendor / "LICENSE").write_text("MIT License\n")
    monkeypatch.setattr("hawavoclean.cli._repo_root", lambda: tmp_path)
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(_REPO_PROFILES))

    _run_cli(monkeypatch, "restore-doctor")

    out = capsys.readouterr().out
    assert "[OK] Upstream research baseline present: UniverSR" in out
    assert "26dc21c4" in out


def test_restore_doctor_fails_on_a_profiles_root_that_cannot_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A profiles root with no speaker subdirectories IS a preflight failure.

    Resolved through paths.profiles_root(), not the working directory, so the
    doctor reports on the profiles the pipeline would really load.
    """
    _stub_heavy_engines(monkeypatch)
    empty = tmp_path / "profiles"
    empty.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(empty))

    code = _run_cli(monkeypatch, "restore-doctor")

    assert code == int(ExitCode.PREFLIGHT_FAILURE)
    out = capsys.readouterr().out
    assert "[FAIL] Profile validation failed" in out
    assert "No speaker profiles found" in out  # dynamic discovery error
    assert "ALL RESTORATION CHECKS PASSED" not in out


def test_restore_doctor_fails_when_the_profiles_root_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No profiles at all is still a hard preflight failure — restore needs them."""
    _stub_heavy_engines(monkeypatch)
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "nowhere"))

    code = _run_cli(monkeypatch, "restore-doctor")

    assert code == int(ExitCode.PREFLIGHT_FAILURE)
    assert "[FAIL] Profiles root directory missing" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# speaker-profile validate
# ---------------------------------------------------------------------------


def test_speaker_profile_validate_single_profile_json_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A direct profile.json path validates exactly that one profile."""
    target = _REPO_PROFILES / "character_01" / "profile.json"
    code = _run_cli(monkeypatch, "speaker-profile", "validate", str(target))

    assert code == int(ExitCode.SUCCESS)
    out = capsys.readouterr().out
    assert "[OK] Speaker profile valid: character_01" in out
    assert "1 speaker profile(s) validated." in out


def test_speaker_profile_validate_single_profile_directory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory holding profile.json directly validates as a single profile."""
    target = _REPO_PROFILES / "character_02"
    code = _run_cli(monkeypatch, "speaker-profile", "validate", str(target))

    assert code == int(ExitCode.SUCCESS)
    out = capsys.readouterr().out
    assert "[OK] Speaker profile valid: character_02" in out
    assert "1 speaker profile(s) validated." in out


def test_speaker_profile_validate_directory_of_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A profiles root validates every speaker subdirectory it contains."""
    root = tmp_path / "profiles"
    root.mkdir()
    for spk in ("character_01", "character_02"):
        (root / spk).symlink_to(_REPO_PROFILES / spk, target_is_directory=True)
    (root / "not_a_profile").mkdir()  # ignored: no profile.json inside

    code = _run_cli(monkeypatch, "speaker-profile", "validate", str(root))

    assert code == int(ExitCode.SUCCESS)
    out = capsys.readouterr().out
    assert "[OK] Speaker profile valid: character_01" in out
    assert "[OK] Speaker profile valid: character_02" in out
    assert "2 speaker profile(s) validated." in out


def test_speaker_profile_validate_empty_directory_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty (or mistyped) target must FAIL, never pass a CI gate as validated."""
    empty = tmp_path / "profiles"
    empty.mkdir()

    code = _run_cli(monkeypatch, "speaker-profile", "validate", str(empty))

    assert code == int(ExitCode.INVALID_USER_INPUT)
    out = capsys.readouterr().out
    assert "[FAIL] No speaker profiles found under" in out
    assert "[OK]" not in out


def test_speaker_profile_validate_invalid_profile_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A schema-invalid profile.json is refused with the documented exit code."""
    bad = tmp_path / "profile.json"
    bad.write_text('{"schema_version": "1.0", "speaker_id": "character_99"}\n')

    code = _run_cli(monkeypatch, "speaker-profile", "validate", str(bad))

    assert code == int(ExitCode.INVALID_USER_INPUT)
    out = capsys.readouterr().out
    assert "[FAIL] Speaker profile invalid" in out
    assert "validated." not in out


# ---------------------------------------------------------------------------
# restoration-benchmark
# ---------------------------------------------------------------------------


def _install_benchmark_stub(
    monkeypatch: pytest.MonkeyPatch, summary: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    """Replace research.restoration.benchmark.run_restoration_benchmark; record calls."""
    calls: list[dict[str, object]] = []

    def _stub(
        manifest_path: str | Path | None = None,
        output_json_path: str | Path | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append({"manifest_path": manifest_path, "output_json_path": output_json_path})
        return {"summary": summary}

    monkeypatch.syspath_prepend(str(_REPO_ROOT))
    bench_mod = importlib.import_module("research.restoration.benchmark")
    monkeypatch.setattr(bench_mod, "run_restoration_benchmark", _stub)
    return calls


def test_restoration_benchmark_forwards_args_and_formats_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Positional manifest and --output reach the harness; the summary is printed."""
    calls = _install_benchmark_stub(
        monkeypatch,
        {
            "hawarestore_kd": {
                "mean_fullband_lsd_db": 1.234,
                "mean_highband_lsd_db": 5.678,
                "mean_speaker_similarity": 0.912,
            }
        },
    )
    manifest = tmp_path / "manifest.json"
    out_json = tmp_path / "results.json"

    code = _run_cli(monkeypatch, "restoration-benchmark", str(manifest), "--output", str(out_json))

    assert code == int(ExitCode.SUCCESS)
    assert calls == [{"manifest_path": str(manifest), "output_json_path": str(out_json)}]
    out = capsys.readouterr().out
    assert "Benchmark complete!" in out
    assert "hawarestore_kd" in out
    assert "fullband LSD=1.23 dB" in out
    assert "highband LSD=5.68 dB" in out
    assert "spk_sim=0.912" in out
    assert f"Report written to {out_json}" in out


def test_restoration_benchmark_manifest_flag_overrides_positional(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """-m wins over the positional manifest; --output defaults to benchmark_results.json."""
    calls = _install_benchmark_stub(monkeypatch, {})

    code = _run_cli(monkeypatch, "restoration-benchmark", "positional.json", "-m", "flag.json")

    assert code == int(ExitCode.SUCCESS)
    assert calls == [{"manifest_path": "flag.json", "output_json_path": "benchmark_results.json"}]
    assert "Report written to benchmark_results.json" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# process/batch restore argument validation
# ---------------------------------------------------------------------------


def test_process_restore_without_speaker_id_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--mode restore without --speaker-id is refused before any audio is touched."""
    out_wav = tmp_path / "out.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(tmp_path / "in.wav"),
        "-o",
        str(out_wav),
        "--mode",
        "restore",
    )

    assert code == int(ExitCode.INVALID_USER_INPUT)
    assert not out_wav.exists()
    assert "Restore mode requires an explicit --speaker-id" in capsys.readouterr().err


def test_process_cutoff_manual_without_cutoff_hz_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--cutoff manual demands an explicit --cutoff-hz."""
    out_wav = tmp_path / "out.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(tmp_path / "in.wav"),
        "-o",
        str(out_wav),
        "--cutoff",
        "manual",
    )

    assert code == int(ExitCode.INVALID_USER_INPUT)
    assert not out_wav.exists()
    assert "--cutoff manual requires an explicit --cutoff-hz" in capsys.readouterr().err


@pytest.mark.parametrize("passes", ["2", "auto"])
def test_process_restore_refuses_multipass(
    passes: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Restore mode is single-pass only; combining it with --passes is refused."""
    out_wav = tmp_path / "out.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(tmp_path / "in.wav"),
        "-o",
        str(out_wav),
        "--mode",
        "restore",
        "--speaker-id",
        "character_01",
        "--passes",
        passes,
    )

    assert code == int(ExitCode.INVALID_USER_INPUT)
    assert not out_wav.exists()
    err = capsys.readouterr().err
    assert "Restore mode is single-pass only" in err
    assert "--passes 1" in err


def test_batch_restore_without_speaker_id_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """batch --mode restore without --speaker-id refuses before processing any file."""
    src = tmp_path / "in.wav"
    src.write_bytes(b"never read: validation happens first")
    out_dir = tmp_path / "outputs"

    code = _run_cli(monkeypatch, "batch", str(src), "-o", str(out_dir), "--mode", "restore")

    assert code == int(ExitCode.INVALID_USER_INPUT)
    assert "Restore mode requires an explicit --speaker-id" in capsys.readouterr().err
    assert not list(out_dir.glob("*.wav"))


def test_an_unknown_speaker_id_is_refused_before_the_audio_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad --speaker-id must cost nothing.

    The profile was looked up at restoration time -- step 10.5, after decode,
    segmentation and the enhancement of every unit -- so a typo bought the
    whole enhancement pass before the id turned out never to resolve. On the
    8 s fixture that was half the runtime; on an hour of audio it is the
    entire job.

    Decode is sabotaged rather than timed: if the speaker check still ran
    late, the failure that surfaced would be this one, and the wrong error
    message is the proof.
    """
    import hawavoclean.pipeline as pipeline

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("decode must not be reached: the speaker id is already invalid")

    monkeypatch.setattr(pipeline, "decode_audio", _explode)

    with pytest.raises(InvalidUserInputError, match="no_such_speaker"):
        pipeline.run_pipeline(
            input_path=_REPO_ROOT / "tests" / "fixtures" / "sample_sorani_podcast.wav",
            output_path=tmp_path / "out.wav",
            profile="development",
            overwrite=True,
            mode="restore",
            speaker_id="no_such_speaker",
        )


def test_pipeline_run_in_restore_mode(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    import hawavoclean.pipeline as pipeline

    sr = 48000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    sig = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    in_wav = tmp_path / "in.wav"
    sf.write(str(in_wav), sig, sr, format="WAV", subtype="PCM_16")

    out_wav = tmp_path / "out_restored.wav"
    report = pipeline.run_pipeline(
        input_path=in_wav,
        output_path=out_wav,
        profile="development",
        overwrite=True,
        mode="restore",
        speaker_id="character_01",
        profiles_dir=_REPO_PROFILES,
    )
    assert out_wav.is_file()
    assert report.restoration is not None
    assert report.restoration["mode"] == "restore"
    assert report.restoration["speaker_id"] == "character_01"
