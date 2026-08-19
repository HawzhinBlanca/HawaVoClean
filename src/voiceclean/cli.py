"""Command-Line Interface for Hawzhin VoiceClean v1."""

import argparse
import shutil
import sys
import tomllib
from pathlib import Path
from typing import NoReturn

import numpy as np
import scipy
import soundfile as sf

from voiceclean import __version__
from voiceclean.audio.probe import probe_audio
from voiceclean.config import load_config
from voiceclean.enhancement.production import wiener_params_hash
from voiceclean.errors import (
    ExitCode,
    InvalidUserInputError,
    PreflightError,
    PublicationError,
    VoiceCleanError,
)
from voiceclean.finishing.loudness import measure_loudness_and_peaks
from voiceclean.guard.calibration import load_calibration_artifact
from voiceclean.hashing import hash_file, hash_json_canonical
from voiceclean.logging import get_logger, setup_logging
from voiceclean.paths import models_dir, profile_config_path
from voiceclean.pipeline import run_pipeline
from voiceclean.report.writer import load_json_report

logger = get_logger("cli")


def exit_with_code(code: ExitCode | int, message: str | None = None) -> NoReturn:
    """Print message and terminate process with standard exit code."""
    if message:
        if int(code) != ExitCode.SUCCESS:
            sys.stderr.write(f"ERROR: {message}\n")
        else:
            sys.stdout.write(f"{message}\n")
    sys.exit(int(code))


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Run full diagnostic environment preflight checks."""
    print("================================================================================")
    print("                     HAWZHIN VOICECLEAN - SYSTEM DOCTOR                         ")
    print("================================================================================")

    all_passed = True

    # 1. Python runtime and numeric stack
    py_ver = sys.version.split()[0]
    print(f"[OK] Python version: {py_ver}")
    print(
        f"[OK] Numeric stack: numpy {np.__version__}, scipy {scipy.__version__}, "
        f"soundfile {sf.__version__}"
    )

    # 2. FFmpeg and FFprobe binaries
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path:
        print(f"[OK] FFmpeg found: {ffmpeg_path}")
    else:
        print("[WARN] FFmpeg binary not found in PATH; falling back to soundfile.")

    if ffprobe_path:
        print(f"[OK] FFprobe found: {ffprobe_path}")
    else:
        print("[WARN] FFprobe binary not found in PATH; falling back to soundfile.")

    # 3. Configuration files (packaged resources, CWD-independent)
    prod_cfg = profile_config_path("production")
    if prod_cfg.exists():
        try:
            load_config(prod_cfg, is_production=True)
            print(f"[OK] Production config valid: {prod_cfg}")
        except Exception as e:
            print(f"[FAIL] Production config invalid: {e}")
            all_passed = False
    else:
        print(f"[FAIL] Production config missing: {prod_cfg}")
        all_passed = False

    # 4. Guard calibration artifact — including integrity recompute
    calib_file = models_dir() / "guard-calibration.json"
    if calib_file.exists():
        try:
            data = load_calibration_artifact(calib_file)
            expected_id = hash_json_canonical(data["thresholds"])
            if data["calibration_id"] != expected_id:
                print(
                    f"[FAIL] Guard calibration integrity: calibration_id does not "
                    f"recompute from thresholds ({calib_file})"
                )
                all_passed = False
            else:
                print(
                    f"[OK] Guard calibration valid: {calib_file} "
                    f"(ID: {data['calibration_id'][:12]}...)"
                )
        except Exception as e:
            print(f"[FAIL] Guard calibration invalid: {e}")
            all_passed = False
    else:
        print(f"[FAIL] Guard calibration missing: {calib_file}")
        all_passed = False

    # 5. Production core lock — params hash must match the implemented core
    lock_file = models_dir() / "production-core.lock.toml"
    if lock_file.exists():
        try:
            with open(lock_file, "rb") as f:
                lock = tomllib.load(f)
            if lock.get("params_hash") != wiener_params_hash():
                print("[FAIL] Core lock params_hash does not match the implemented core.")
                all_passed = False
            else:
                print(f"[OK] Production core lock verified: {lock_file}")
        except Exception as e:
            print(f"[FAIL] Production core lock unreadable: {e}")
            all_passed = False
    else:
        print(f"[FAIL] Production core lock missing: {lock_file}")
        all_passed = False

    print("================================================================================")
    if all_passed:
        print("Doctor status: ALL CHECKS PASSED. Ready for processing.")
        return int(ExitCode.SUCCESS)
    print("Doctor status: PREFLIGHT FAILED. Address errors above before processing.")
    return int(ExitCode.PREFLIGHT_FAILURE)


def cmd_process(args: argparse.Namespace) -> int:
    """Process an audio file through the VoiceClean pipeline."""
    try:
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            profile=args.profile,
            overwrite=args.overwrite,
        )
        return int(ExitCode.SUCCESS)
    except PreflightError as e:
        logger.error(f"Preflight failure: {e}")
        return int(ExitCode.PREFLIGHT_FAILURE)
    except InvalidUserInputError as e:
        logger.error(f"Invalid user input: {e}")
        return int(ExitCode.INVALID_USER_INPUT)
    except PublicationError as e:
        logger.error(f"Publication failure: {e}")
        return int(ExitCode.PUBLICATION_FAILURE)
    except VoiceCleanError as e:
        logger.error(f"Processing error: {e}")
        return int(e.exit_code)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return int(ExitCode.PUBLICATION_FAILURE)


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify an output audio master against its immutable JSON report."""
    audio_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()

    if not audio_path.exists():
        exit_with_code(ExitCode.INVALID_USER_INPUT, f"Audio file not found: {audio_path}")

    if not report_path.exists():
        exit_with_code(ExitCode.INVALID_USER_INPUT, f"Report file not found: {report_path}")

    try:
        report = load_json_report(report_path)
    except Exception as e:
        exit_with_code(ExitCode.PUBLICATION_FAILURE, f"Failed to load or validate JSON report: {e}")

    # Check SHA-256
    actual_sha256 = hash_file(audio_path)
    if actual_sha256 != report.output.sha256:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Checksum mismatch for {audio_path}: expected {report.output.sha256}, "
            f"got {actual_sha256}",
        )

    # Check sample structure
    try:
        media = probe_audio(audio_path)
    except Exception as e:
        exit_with_code(ExitCode.PUBLICATION_FAILURE, f"Cannot probe verified audio: {e}")

    if media.samples != report.output.samples:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Sample count mismatch: expected {report.output.samples}, got {media.samples}",
        )

    if media.sample_rate != report.output.sample_rate:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Sample rate mismatch: expected {report.output.sample_rate}, got {media.sample_rate}",
        )

    if media.channels != report.output.channels:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Channels mismatch: expected {report.output.channels}, got {media.channels}",
        )

    # Check loudness and peak bounds against the -1.0 dBTP ceiling
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    loudness = measure_loudness_and_peaks(data.T, sr)

    if report.output.true_peak_dbtp is not None and loudness.true_peak_dbtp > -1.0:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"True peak {loudness.true_peak_dbtp:.3f} dBTP exceeds ceiling -1.0 dBTP",
        )

    print("================================================================================")
    print(f"VERIFICATION PASSED: {audio_path}")
    print(f"  SHA-256:             {actual_sha256}")
    print(f"  Samples:             {media.samples:,}")
    print(f"  Integrated Loudness: {loudness.integrated_lufs:.1f} LUFS")
    print(f"  True Peak:           {loudness.true_peak_dbtp:.1f} dBTP")
    print("================================================================================")
    return int(ExitCode.SUCCESS)


def cmd_audit_models(_args: argparse.Namespace) -> int:
    """Verify core provenance: params hash, license allowlist, calibration integrity.

    Exits non-zero on any verification failure — an audit that cannot fail
    is not an audit.
    """
    lock_path = models_dir() / "production-core.lock.toml"
    policy_path = models_dir() / "license-policy.toml"
    calib_path = models_dir() / "guard-calibration.json"

    failures: list[str] = []

    print("================================================================================")
    print("                      HAWZHIN VOICECLEAN - MODEL AUDIT                          ")
    print("================================================================================")

    if not lock_path.exists():
        failures.append(f"core lockfile missing: {lock_path}")
        lock = {}
    else:
        with open(lock_path, "rb") as f:
            lock = tomllib.load(f)

    if not policy_path.exists():
        failures.append(f"license policy missing: {policy_path}")
        allowed: list[str] = []
    else:
        with open(policy_path, "rb") as f:
            allowed = list(tomllib.load(f).get("allowed_licenses", []))

    if lock:
        print(f"Production core:  {lock.get('core_id')}")
        print(f"Algorithm:        {lock.get('algorithm')}")
        print(f"Code license:     {lock.get('code_license')}")

        # 1a. Parameter hash must match the core actually implemented
        actual = wiener_params_hash()
        if lock.get("params_hash") != actual:
            failures.append(
                f"params_hash mismatch: lockfile {str(lock.get('params_hash'))[:16]}... "
                f"vs implemented core {actual[:16]}..."
            )
        else:
            print(f"Params hash:      {actual[:16]}... [VERIFIED against implementation]")

        # 1b. The [params] table itself must recompute to params_hash — a
        # decorative table that can drift from its own digest is provenance
        # theater, which is exactly what this command exists to prevent.
        table_hash = hash_json_canonical(dict(lock.get("params", {})))
        if table_hash != lock.get("params_hash"):
            failures.append(
                "params table does not recompute to params_hash: the lockfile's "
                "own [params] block has been edited or drifted"
            )
        else:
            print("Params table:     recomputes to params_hash [VERIFIED]")

        # 2. License must be allowlisted
        lic = str(lock.get("code_license", ""))
        if allowed and lic not in allowed:
            failures.append(f"license {lic!r} is not in the allowlist {allowed}")
        elif allowed:
            print(f"License check:    {lic!r} allowlisted [VERIFIED]")

        # 3. Any weight digest table must resolve (none exist for DSP cores,
        # but a future neural core lands here and gets verified)
        weight_table: dict[str, str] = lock.get("weight_sha256", {}) or {}
        for fname, digest in weight_table.items():
            candidate = models_dir() / str(fname)
            if not candidate.exists():
                failures.append(f"declared weights file missing: {fname}")
            elif hash_file(candidate) != digest:
                failures.append(f"weights digest mismatch for {fname}")

    # 4. Calibration artifact integrity
    if calib_path.exists():
        try:
            calib = load_calibration_artifact(calib_path)
            if calib["calibration_id"] != hash_json_canonical(calib["thresholds"]):
                failures.append("calibration_id does not recompute from thresholds")
            else:
                print(f"Calibration:      {calib['calibration_id'][:16]}... [VERIFIED]")
        except Exception as e:
            failures.append(f"calibration artifact unreadable: {e}")
    else:
        failures.append(f"calibration artifact missing: {calib_path}")

    print("================================================================================")
    if failures:
        for fmsg in failures:
            print(f"[FAIL] {fmsg}")
        print("Audit status: FAILED")
        return int(ExitCode.PREFLIGHT_FAILURE)
    print("Audit status: ALL PROVENANCE CHECKS VERIFIED")
    return int(ExitCode.SUCCESS)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Measure guard accept/revert rates over a calibration corpus."""
    from voiceclean.eval.calibrate import run_calibration

    print(f"Measuring guard rates over manifest: {args.manifest}")
    res = run_calibration(
        manifest_path=args.manifest,
        output_calibration_path=args.output,
        corruption_profile=args.corruption_profile,
        use_fixed_probe=args.fixed_probe,
    )
    m = res["measured"]
    print(f"Calibration measured and written to: {args.output}")
    print(f"  Corpus items:        {m['item_count']}")
    print(
        f"  False accept rate:   {res['metrics']['calibration_false_accept_rate']:.4f} "
        f"({m['corrupted_accepted']}/{m['corrupted_evaluations']} corrupted accepted)"
    )
    print(
        f"  False revert rate:   {res['metrics']['calibration_false_revert_rate']:.4f} "
        f"({m['benign_rejected']}/{m['benign_evaluations']} benign rejected)"
    )
    return int(ExitCode.SUCCESS)


def cmd_eval(args: argparse.Namespace) -> int:
    """Run automated acceptance gates on an acceptance dataset."""
    from voiceclean.eval.acceptance import evaluate_acceptance_gates

    print(f"Evaluating acceptance gates against manifest: {args.manifest}")
    res = evaluate_acceptance_gates(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    passed = res["release_gate_status"] == "PASSED"
    print(f"Acceptance Evaluation Outcome: {res['release_gate_status']}")
    print(f"  Passed Items: {res['passed_items']} / {res['total_items']}")
    print(f"  Enhanced speech units: {res['speech_units_enhanced']} / {res['speech_units_total']}")
    for item in res["results"]:
        if not item["passed"]:
            for f in item["failures"]:
                print(f"  [FAIL] {item['id']}: {f}")
    for f in res["corpus_failures"]:
        print(f"  [FAIL] corpus: {f}")
    return int(ExitCode.SUCCESS if passed else ExitCode.PUBLICATION_FAILURE)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute the measured benchmark over a corpus."""
    from voiceclean.research.benchmark import run_benchmark

    print(f"Running benchmark against manifest: {args.manifest}")
    res = run_benchmark(
        manifest_path=args.manifest,
        output_report_path=args.output,
    )
    m = res["measured"]
    print(f"Benchmark complete. Report written to: {args.output}")
    print(f"  Units enhanced: {m['units_enhanced']} / {m['units_total']}")
    print(f"  Real-time factor: {m['real_time_factor']:.3f}")
    return int(ExitCode.SUCCESS)


def cmd_blind_abx(args: argparse.Namespace) -> int:
    """Generate randomized blind listening test trial sheet."""
    from voiceclean.eval.blind_abx import generate_blind_trial_manifest

    print(f"Generating Blind ABX Listening Trial Sheet from manifest: {args.manifest}")
    out = generate_blind_trial_manifest(
        system_a_manifest=args.manifest,
        system_b_manifest=args.manifest_b or args.manifest,
        output_sheet_path=args.output,
    )
    print(f"Blind ABX Session created: {out}")
    return int(ExitCode.SUCCESS)


def main() -> None:
    """CLI main entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="voiceclean",
        description=(
            "Hawzhin VoiceClean - dialogue audio cleanup (Wiener spectral denoising, "
            "spectral-change guarded finishing, BS.1770 mastering)"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    p_doctor = subparsers.add_parser(
        "doctor", help="Run system preflight and dependency diagnostics"
    )
    p_doctor.set_defaults(func=cmd_doctor)

    # process
    p_proc = subparsers.add_parser("process", help="Process an audio file through VoiceClean")
    p_proc.add_argument("input", help="Path to input audio file")
    p_proc.add_argument("--output", "-o", required=True, help="Path to output mastered WAV")
    p_proc.add_argument("--config", "-c", help="Path to custom TOML configuration")
    p_proc.add_argument(
        "--profile", "-p", choices=["production", "development"], default="production"
    )
    p_proc.add_argument(
        "--overwrite", action="store_true", help="Overwrite destination output if exists"
    )
    p_proc.set_defaults(func=cmd_process)

    # verify
    p_ver = subparsers.add_parser("verify", help="Verify output audio master against JSON report")
    p_ver.add_argument("output", help="Path to mastered WAV file")
    p_ver.add_argument("--report", "-r", required=True, help="Path to .voiceclean.json report")
    p_ver.set_defaults(func=cmd_verify)

    # audit-models
    p_audit = subparsers.add_parser(
        "audit-models", help="Verify core provenance, license allowlist, calibration integrity"
    )
    p_audit.set_defaults(func=cmd_audit_models)

    # calibrate
    p_calib = subparsers.add_parser(
        "calibrate", help="Measure guard accept/revert rates over a corpus manifest"
    )
    p_calib.add_argument(
        "--manifest", "-m", default="data/calibration/manifest.json", help="Path to corpus manifest"
    )
    p_calib.add_argument(
        "--output",
        "-o",
        default="guard-calibration.measured.json",
        help="Destination calibration JSON",
    )
    p_calib.add_argument(
        "--corruption-profile",
        choices=["mild", "standard", "severe"],
        default="standard",
        help="Corruption operator suite to measure against",
    )
    p_calib.add_argument(
        "--fixed-probe", action="store_true", help="Use the deterministic FixedProbe (tests only)"
    )
    p_calib.set_defaults(func=cmd_calibrate)

    # eval
    p_eval = subparsers.add_parser(
        "eval", help="Run hard acceptance gates against an acceptance manifest"
    )
    p_eval.add_argument(
        "--manifest", "-m", default="data/acceptance/manifest.json", help="Path to corpus manifest"
    )
    p_eval.add_argument(
        "--output-dir", "-o", default="data/acceptance/outputs", help="Output directory for results"
    )
    p_eval.set_defaults(func=cmd_eval)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run the measured pipeline benchmark")
    p_bench.add_argument(
        "--manifest", "-m", default="data/acceptance/manifest.json", help="Path to corpus manifest"
    )
    p_bench.add_argument(
        "--output", "-o", default="benchmark_results.json", help="Output report JSON"
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # blind-abx
    p_abx = subparsers.add_parser(
        "blind-abx", help="Generate randomized blind listening trial sheet"
    )
    p_abx.add_argument(
        "--manifest",
        "-m",
        default="data/acceptance/manifest.json",
        help="Path to System A manifest",
    )
    p_abx.add_argument("--manifest-b", help="Optional Path to System B manifest (comparison)")
    p_abx.add_argument(
        "--output",
        "-o",
        default="blind_abx_session.json",
        help="Destination trial sheet JSON",
    )
    p_abx.set_defaults(func=cmd_blind_abx)

    args = parser.parse_args()
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
