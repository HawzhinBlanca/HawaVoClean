"""Command-Line Interface for Hawzhin VoiceClean v1."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import NoReturn

import soundfile as sf
import torch

from voiceclean import __version__
from voiceclean.audio.probe import probe_audio
from voiceclean.config import load_config
from voiceclean.errors import (
    ExitCode,
    InvalidUserInputError,
    PreflightError,
    PublicationError,
    VoiceCleanError,
)
from voiceclean.finishing.loudness import measure_loudness_and_peaks
from voiceclean.guard.calibration import load_calibration_artifact
from voiceclean.hashing import hash_file
from voiceclean.logging import get_logger, setup_logging
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

    # 1. Python runtime
    py_ver = sys.version.split()[0]
    print(f"[OK] Python version: {py_ver}")

    # 2. PyTorch & Acceleration
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Correctness fallback)"
    print(
        f"[OK] PyTorch version: {torch_ver} (CUDA available: {cuda_avail}, Device: {device_name})"
    )

    # 3. FFmpeg and FFprobe binaries
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

    # 4. Configuration files
    prod_cfg = Path("configs/production.toml")
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

    # 5. Calibration Artifact
    calib_file = Path("models/guard-calibration.json")
    if calib_file.exists():
        try:
            data = load_calibration_artifact(calib_file)
            print(
                f"[OK] Guard calibration valid: {calib_file} (ID: {data['calibration_id'][:12]}...)"
            )
        except Exception as e:
            print(f"[FAIL] Guard calibration invalid: {e}")
            all_passed = False
    else:
        print(f"[FAIL] Guard calibration missing: {calib_file}")
        all_passed = False

    # 6. Production Core Lock
    lock_file = Path("models/production-core.lock.toml")
    if lock_file.exists():
        print(f"[OK] Production core lock present: {lock_file}")
    else:
        print(f"[FAIL] Production core lock missing: {lock_file}")
        all_passed = False

    # 7. Model Registry
    reg_file = Path("models/model-registry.toml")
    if reg_file.exists():
        print(f"[OK] Model registry present: {reg_file}")
    else:
        print(f"[FAIL] Model registry missing: {reg_file}")
        all_passed = False

    print("================================================================================")
    if all_passed:
        print("Doctor status: ALL CHECKS PASSED. Ready for production processing.")
        return int(ExitCode.SUCCESS)
    else:
        print("Doctor status: PREFLIGHT FAILED. Address errors above before running in production.")
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
            f"Checksum mismatch for {audio_path}: expected {report.output.sha256}, got {actual_sha256}",
        )

    # Check sample structure
    try:
        probe = probe_audio(audio_path)
    except Exception as e:
        exit_with_code(ExitCode.PUBLICATION_FAILURE, f"Cannot probe verified audio: {e}")

    if probe.samples != report.output.samples:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Sample count mismatch: expected {report.output.samples}, got {probe.samples}",
        )

    if probe.sample_rate != report.output.sample_rate:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Sample rate mismatch: expected {report.output.sample_rate}, got {probe.sample_rate}",
        )

    if probe.channels != report.output.channels:
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"Channels mismatch: expected {report.output.channels}, got {probe.channels}",
        )

    # Check loudness and peak bounds
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    loudness = measure_loudness_and_peaks(data.T, sr)

    if (
        report.output.true_peak_dbtp is not None and loudness.true_peak_dbtp > -0.9
    ):  # ceiling is -1.0 dBTP with 0.1dB tolerance
        exit_with_code(
            ExitCode.PUBLICATION_FAILURE,
            f"True peak {loudness.true_peak_dbtp:.2f} dBTP exceeds safety ceiling -1.0 dBTP",
        )

    print("================================================================================")
    print(f"VERIFICATION PASSED: {audio_path}")
    print(f"  SHA-256:             {actual_sha256}")
    print(f"  Samples:             {probe.samples:,}")
    print(f"  Integrated Loudness: {loudness.integrated_lufs:.1f} LUFS")
    print(f"  True Peak:           {loudness.true_peak_dbtp:.1f} dBTP")
    print("================================================================================")
    return int(ExitCode.SUCCESS)


def cmd_audit_models(_args: argparse.Namespace) -> int:
    """Audit all registered models and lockfiles for license and hash compliance."""
    import tomllib

    reg_path = Path("models/model-registry.toml")
    lock_path = Path("models/production-core.lock.toml")

    if not reg_path.exists() or not lock_path.exists():
        exit_with_code(ExitCode.PREFLIGHT_FAILURE, "Model registry or lockfile missing.")

    with open(reg_path, "rb") as f:
        registry = tomllib.load(f)
    with open(lock_path, "rb") as f:
        lock = tomllib.load(f)

    print("================================================================================")
    print("                      HAWZHIN VOICECLEAN - MODEL AUDIT                          ")
    print("================================================================================")
    print(f"Authoritative Production Core: {lock['core_id']}")
    print(f"Repository:                    {lock['repo_url']} (commit {lock['commit'][:8]})")
    print(f"Code License:                  {lock['code_license']}")
    print(f"Weights License:               {lock['weight_license']}")
    print(f"Phase Coherent:                {lock['phase_coherent']}")
    print("")
    print("Registered Benchmark Candidates:")
    for cand in registry.get("candidates", []):
        print(
            f"  - [{cand['id']}] ({cand['role']}) status={cand['status']} license={cand['code_license']}"
        )

    print("================================================================================")
    return int(ExitCode.SUCCESS)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit Guard thresholds from a calibration corpus manifest."""
    from eval.calibrate import run_calibration

    print(f"Calibrating Guard thresholds using manifest: {args.manifest}")
    res = run_calibration(
        manifest_path=args.manifest,
        output_calibration_path=args.output,
        use_fake_asr=args.fake_asr,
    )
    print(f"Calibration completed successfully. Artifact written to: {args.output}")
    print(f"  Calibration ID: {res['calibration_id']}")
    print(f"  False Accept Rate: {res['metrics']['calibration_false_accept_rate']:.4f}")
    return int(ExitCode.SUCCESS)


def cmd_eval(args: argparse.Namespace) -> int:
    """Run automated acceptance evaluation gates on an acceptance dataset."""
    from eval.acceptance import evaluate_acceptance_gates

    print(f"Evaluating acceptance gates against manifest: {args.manifest}")
    res = evaluate_acceptance_gates(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    passed = res.get("release_gate_status") == "PASSED"
    print(f"Acceptance Evaluation Outcome: {'PASSED' if passed else 'FAILED'}")
    print(f"  Passed Items: {res['passed_items']} / {res['total_items']}")
    return int(ExitCode.SUCCESS if passed else ExitCode.PUBLICATION_FAILURE)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute candidate models benchmark matrix."""
    from research.benchmark import run_benchmark

    print(f"Running candidate benchmark against manifest: {args.manifest}")
    _res = run_benchmark(
        manifest_path=args.manifest,
        output_report_path=args.output,
    )
    print(f"Benchmark completed. Report written to: {args.output}")
    return int(ExitCode.SUCCESS)


def cmd_blind_abx(args: argparse.Namespace) -> int:
    """Generate randomized blind listening test trial sheet."""
    from eval.blind_abx import generate_blind_trial_manifest

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
    # Add cwd to path for eval/ and research/ lazy imports
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    setup_logging()
    parser = argparse.ArgumentParser(
        prog="voiceclean",
        description="Hawzhin VoiceClean - Kurdish Sorani Dialogue Audio Enhancement & Linguistic Fidelity System",
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
        "audit-models", help="Audit model registry and license compliance"
    )
    p_audit.set_defaults(func=cmd_audit_models)

    # calibrate
    p_calib = subparsers.add_parser(
        "calibrate", help="Fit Guard safety thresholds on a corpus manifest"
    )
    p_calib.add_argument(
        "--manifest", "-m", default="data/calibration/manifest.json", help="Path to corpus manifest"
    )
    p_calib.add_argument(
        "--output",
        "-o",
        default="models/guard-calibration.json",
        help="Destination calibration JSON",
    )
    p_calib.add_argument(
        "--fake-asr", action="store_true", help="Use lightweight FakeSoraniASR for unit calibration"
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
    p_bench = subparsers.add_parser("benchmark", help="Run candidate multi-model benchmark matrix")
    p_bench.add_argument(
        "--manifest", "-m", default="data/acceptance/manifest.json", help="Path to corpus manifest"
    )
    p_bench.add_argument(
        "--output", "-o", default="research/benchmark_results.json", help="Output report JSON"
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
        default="eval/blind_abx_session.json",
        help="Destination trial sheet JSON",
    )
    p_abx.set_defaults(func=cmd_blind_abx)

    args = parser.parse_args()
    code = args.func(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
