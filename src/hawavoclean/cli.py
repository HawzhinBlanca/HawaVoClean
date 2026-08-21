"""Command-Line Interface for HawaVoClean v1."""

import argparse
import atexit
import contextlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import scipy
import soundfile as sf

from hawavoclean import __version__
from hawavoclean.audio.probe import probe_audio
from hawavoclean.config import load_config
from hawavoclean.enhancement.factory import CORE_REGISTRY
from hawavoclean.errors import (
    ExitCode,
    HawaVoCleanError,
    InvalidUserInputError,
    PreflightError,
    PublicationError,
)
from hawavoclean.finishing.loudness import measure_loudness_and_peaks
from hawavoclean.guard.calibration import load_calibration_artifact
from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.logging import get_logger, setup_logging
from hawavoclean.multipass import MAX_PASSES, run_multipass
from hawavoclean.paths import models_dir, profile_config_path
from hawavoclean.pipeline import run_pipeline
from hawavoclean.progress import ProgressEvent
from hawavoclean.publication import (
    public_output_path,
    publication_paths,
    resolve_committed_publication,
)
from hawavoclean.report.writer import load_json_report
from hawavoclean.watchdog import child_env, install_parent_death_watchdog

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
    print("                     HAWAVOCLEAN - SYSTEM DOCTOR                         ")
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

    # 5. Core locks — params hash must match each implemented core
    for core_id, registration in CORE_REGISTRY.items():
        lock_file = models_dir() / registration.lock_filename
        if lock_file.exists():
            try:
                with open(lock_file, "rb") as f:
                    lock = tomllib.load(f)
                if lock.get("params_hash") != registration.implementation_params_hash():
                    print(f"[FAIL] {core_id}: lock params_hash does not match implementation.")
                    all_passed = False
                else:
                    print(f"[OK] Core lock verified: {core_id} ({lock_file.name})")
            except Exception as e:
                print(f"[FAIL] {core_id}: core lock unreadable: {e}")
                all_passed = False
        else:
            print(f"[FAIL] {core_id}: core lock missing: {lock_file}")
            all_passed = False

    print("================================================================================")
    if all_passed:
        print("Doctor status: ALL CHECKS PASSED. Ready for processing.")
        return int(ExitCode.SUCCESS)
    print("Doctor status: PREFLIGHT FAILED. Address errors above before processing.")
    return int(ExitCode.PREFLIGHT_FAILURE)


class _JsonLineSink:
    """``--progress-json``: one JSON object per line on the ORIGINAL stdout.

    The real stdout is duplicated to a private descriptor for the JSON
    stream, and fd 1 is then pointed at stderr. Anything else that writes to
    stdout afterwards — a stray ``print``, a library banner, the spawned
    enhancement worker (which inherits fd 1) — lands on stderr, so the JSON
    stream is never corrupted.
    """

    def __init__(self) -> None:
        sys.stdout.flush()
        self._stdout_fd = sys.stdout.fileno()
        json_fd = os.dup(self._stdout_fd)
        os.dup2(sys.stderr.fileno(), self._stdout_fd)
        self._out = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)

    def emit(self, obj: dict[str, Any]) -> None:
        try:
            self._out.write(json.dumps(obj, separators=(",", ":")) + "\n")
            self._out.flush()
        except OSError as e:  # reader went away: keep processing, log once
            logger.warning(f"progress stream closed: {e}")

    def close(self) -> None:
        """Flush the stream and hand fd 1 back to the original stdout."""
        with contextlib.suppress(OSError, ValueError):
            self._out.flush()
            os.dup2(self._out.fileno(), self._stdout_fd)
            self._out.close()


def cmd_process(args: argparse.Namespace) -> int:
    """Process an audio file through the HawaVoClean pipeline."""
    sink = _JsonLineSink() if getattr(args, "progress_json", False) else None

    def fail(code: ExitCode, prefix: str, e: BaseException) -> int:
        logger.error(f"{prefix}: {e}")
        if sink is not None:
            sink.emit({"event": "error", "code": code.name, "message": str(e)})
        return int(code)

    on_progress = None
    if sink is not None:

        def on_progress(event: ProgressEvent) -> None:
            sink.emit(event.to_dict())

    passes: int | str = getattr(args, "passes", 1)
    try:
        if passes == 1:
            # The default path: byte-identical to a run before --passes
            # existed, including the progress event stream.
            run_pipeline(
                input_path=args.input,
                output_path=args.output,
                config_path=args.config,
                profile=args.profile,
                overwrite=args.overwrite,
                on_progress=on_progress,
            )
        else:
            run_multipass(
                input_path=args.input,
                output_path=args.output,
                passes=passes,
                config_path=args.config,
                profile=args.profile,
                overwrite=args.overwrite,
                on_progress=on_progress,
            )
        if sink is not None:
            out_path = public_output_path(args.output)
            sink.emit(
                {
                    "event": "done",
                    "progress": 1.0,
                    "output_path": str(out_path),
                    "report_path": str(out_path.parent / f"{out_path.stem}.hawavoclean.json"),
                }
            )
        return int(ExitCode.SUCCESS)
    except PreflightError as e:
        return fail(ExitCode.PREFLIGHT_FAILURE, "Preflight failure", e)
    except InvalidUserInputError as e:
        return fail(ExitCode.INVALID_USER_INPUT, "Invalid user input", e)
    except PublicationError as e:
        return fail(ExitCode.PUBLICATION_FAILURE, "Publication failure", e)
    except HawaVoCleanError as e:
        return fail(e.exit_code, "Processing error", e)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        if sink is not None:
            sink.emit(
                {
                    "event": "error",
                    "code": ExitCode.PUBLICATION_FAILURE.name,
                    "message": f"{type(e).__name__}: {e}",
                }
            )
        return int(ExitCode.PUBLICATION_FAILURE)
    finally:
        if sink is not None:
            sink.close()


def _passes_arg(value: str) -> int | str:
    """``--passes``: an integer 1..4 or 'auto'. Anything else is refused."""
    if value == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--passes must be an integer 1..{MAX_PASSES} or 'auto', got {value!r}"
        ) from None
    if not 1 <= n <= MAX_PASSES:
        raise argparse.ArgumentTypeError(
            f"--passes must be an integer 1..{MAX_PASSES} or 'auto', got {value!r}"
        )
    return n


_AUDIO_SUFFIXES = {".wav", ".mp4", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".mov", ".aiff"}


def _clean_stem(path: Path) -> str:
    """Strip stacked audio suffixes: 'Flute 09.m4a.mp4' -> 'Flute 09'."""
    name = path.name
    while True:
        stem, dot, ext = name.rpartition(".")
        if dot and f".{ext.lower()}" in _AUDIO_SUFFIXES and stem:
            name = stem
        else:
            return name


def cmd_batch_worker(_args: argparse.Namespace) -> int:
    """The long-lived side of ``batch``. Not for hand use.

    Reads one JSON request per line on stdin, publishes that file, writes one
    JSON reply per line on a private duplicate of stdout. Everything else —
    logs, tracebacks, a library banner, anything the spawned enhancement
    workers print on the fd 1 they inherit — is pushed onto stderr by
    :class:`_JsonLineSink`, which the parent captures to a file, because a pipe
    nobody drains is a deadlock waiting for a chatty run. Without that, one
    stray ``print`` three processes down would arrive as a protocol line.

    It keeps its enhancement pool warm between files (:func:`set_pool_reuse`),
    which is the whole point: the model is the same object for every file in
    the batch instead of being loaded 77 times.
    """
    from hawavoclean.enhancement.worker import set_pool_reuse, shutdown_pool_cache

    sink = _JsonLineSink()
    set_pool_reuse(True)
    try:
        sink.emit({"event": "ready"})
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            reply: dict[str, Any]
            try:
                req = json.loads(raw)
            except ValueError as e:
                sink.emit({"ok": False, "code": int(ExitCode.INVALID_USER_INPUT), "error": str(e)})
                continue
            if req.get("op") == "quit":
                break
            try:
                run_pipeline(
                    input_path=req["input"],
                    output_path=req["output"],
                    profile=req["profile"],
                    overwrite=bool(req.get("overwrite", False)),
                )
                reply = {"ok": True}
            except HawaVoCleanError as e:
                reply = {"ok": False, "code": int(e.exit_code), "error": f"{e}"}
            except Exception as e:
                reply = {
                    "ok": False,
                    "code": int(ExitCode.PUBLICATION_FAILURE),
                    "error": f"{type(e).__name__}: {e}",
                }
            sink.emit(reply)
    finally:
        shutdown_pool_cache()
        sink.close()
    return int(ExitCode.SUCCESS)


class _BatchChild:
    """One warm child process, restarted whenever it stops being trustworthy.

    The per-file child existed for isolation — a stuck decoder, a wedged
    model or a segfault must cost one file, not the batch — and that is kept
    exactly: every file still runs under a hard deadline in a process the
    parent can kill, and a breach kills the whole process group and starts a
    new child. What is dropped is the part isolation never needed, which is
    reloading the interpreter and the enhancement core for every file
    (measured at 1.5-2.0 s per file, most of it model load).

    A child is discarded, never reused, after a deadline breach or an
    unexpected exit: the next file gets a process whose state nobody has to
    reason about.
    """

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr_path: Path | None = None
        self._env = env
        self.spawns = 0

    def _spawn(self) -> None:
        self.close()
        self._stderr_path = Path(tempfile.mkstemp(prefix="hawavoclean-batch-", suffix=".log")[1])
        err = open(self._stderr_path, "w", encoding="utf-8")  # noqa: SIM115 - owned by the child
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "hawavoclean.cli", "batch-worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                # child_env: a batch killed with SIGKILL cannot reap this
                # child, so the child is told whose death to watch for.
                env=child_env(self._env),
                # Its own process group, so a deadline breach can take the
                # decoder and the enhancement workers with it rather than
                # leaving the grandchildren behind.
                start_new_session=True,
            )
        finally:
            err.close()
        self.spawns += 1
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._pump, args=(self._proc,), daemon=True)
        self._reader.start()

    def _pump(self, proc: subprocess.Popen[str]) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)  # EOF sentinel: the child is gone

    def _await_line(self, deadline: float) -> str | None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise EOFError("batch worker exited")
            line = line.strip()
            if line:
                return line

    def _await_reply(self, deadline: float) -> dict[str, Any] | None:
        """The next JSON object the child sends, or ``None`` at the deadline.

        Anything on the pipe that is not a JSON object did not come from the
        protocol — it is output that escaped onto fd 1 somewhere below — and
        it is dropped rather than allowed to fail the file it arrived during.
        """
        while True:
            line = self._await_line(deadline)
            if line is None:
                return None
            try:
                parsed = json.loads(line)
            except ValueError:
                logger.debug("batch worker wrote a non-protocol line: %.120s", line)
                continue
            if isinstance(parsed, dict):
                return parsed

    def stderr_tail(self) -> str:
        if self._stderr_path is None or not self._stderr_path.exists():
            return "no output"
        lines = [
            ln for ln in self._stderr_path.read_text(errors="replace").splitlines() if ln.strip()
        ]
        return lines[-1][-160:] if lines else "no output"

    def run(self, src: Path, dest: Path, profile: str, overwrite: bool, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        try:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
                if self._await_reply(deadline) is None:
                    raise TimeoutError
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(
                json.dumps(
                    {
                        "input": str(src),
                        "output": str(dest),
                        "profile": profile,
                        "overwrite": overwrite,
                    }
                )
                + "\n"
            )
            self._proc.stdin.flush()
            reply = self._await_reply(deadline)
            if reply is None:
                raise TimeoutError
        except TimeoutError:
            self.close()
            return f"FAILED: timed out after {timeout_s:.0f}s (killed; batch continued)"
        except (EOFError, BrokenPipeError, OSError):
            tail = self.stderr_tail()
            code = self._proc.poll() if self._proc is not None else None
            self.close()
            return f"FAILED (exit {code}): {tail}"
        if not reply.get("ok"):
            return f"FAILED (exit {reply.get('code')}): {str(reply.get('error', ''))[-160:]}"
        return _summarize_published(dest)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5.0)
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.join(timeout=2.0)
        if self._stderr_path is not None:
            path, self._stderr_path = self._stderr_path, None
            with contextlib.suppress(OSError):
                path.unlink()


#: The batch's warm child. Module-level so a batch (and the tests that call
#: ``_run_one_isolated`` directly) share one, and so atexit can end it.
_batch_child = _BatchChild()
atexit.register(_batch_child.close)


def _summarize_published(dest: Path) -> str:
    """The one-line status of a file that published successfully."""
    report_path = dest.parent / f"{dest.stem}.hawavoclean.json"
    try:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        summ = rep["summary"]
        return (
            f"ok: {summ['enhanced']}/{summ['units_total']} units enhanced, "
            f"{rep['output']['integrated_lufs']:.1f} LUFS"
        )
    except Exception as e:
        return f"FAILED: published but report unreadable ({type(e).__name__})"


def _run_one_isolated(
    src: Path, dest: Path, profile: str, overwrite: bool, timeout_s: float
) -> str:
    """Process one file in a child process under a hard deadline.

    A hung file (stuck decoder, wedged model, unresponsive disk) must never
    take the rest of the batch with it. The child is killed at the deadline
    and the batch moves on; the failure is recorded by name.

    The child is now shared with the rest of the batch and kept warm between
    files — see :class:`_BatchChild` for why that costs nothing in isolation.
    """
    return _batch_child.run(src, dest, profile, overwrite, timeout_s)


def cmd_batch(args: argparse.Namespace) -> int:
    """Process many files; one failure never aborts the rest.

    Each file gets its own pipeline run, output, and audit report. The
    summary names every failure; the exit code is non-zero if any file
    failed, so scripts cannot mistake a partial batch for a clean one.
    """
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[Path] = []
    for raw in args.inputs:
        candidate = Path(raw)
        if candidate.is_file():
            inputs.append(candidate)
        else:
            print(f"[SKIP] not a file: {candidate}")

    if not inputs:
        exit_with_code(ExitCode.INVALID_USER_INPUT, "No input files to process.")

    # Two inputs that clean to the same stem ('a.wav' and 'a.m4a') would
    # silently overwrite (or silently skip) each other. Refuse up front.
    dest_to_srcs: dict[Path, list[Path]] = {}
    for src in inputs:
        dest_to_srcs.setdefault(out_dir / f"{_clean_stem(src)}{args.suffix}.wav", []).append(src)
    collisions = {d: srcs for d, srcs in dest_to_srcs.items() if len(srcs) > 1}
    if collisions:
        lines = [
            f"  {d.name}  <-  {', '.join(x.name for x in srcs)}" for d, srcs in collisions.items()
        ]
        exit_with_code(
            ExitCode.INVALID_USER_INPUT,
            "Output name collision: these inputs would write the same file:\n"
            + "\n".join(lines)
            + "\nRename the inputs or run them in separate batches.",
        )

    results: list[tuple[str, str]] = []
    failed = 0
    try:
        for i, src in enumerate(inputs, 1):
            dest = out_dir / f"{_clean_stem(src)}{args.suffix}.wav"
            if dest.exists() and args.skip_existing:
                results.append((src.name, "skipped (exists)"))
                print(f"[{i}/{len(inputs)}] SKIP {src.name} (output exists)")
                continue
            print(f"[{i}/{len(inputs)}] {src.name} -> {dest.name}")
            status = _run_one_isolated(
                src,
                dest,
                args.profile,
                args.overwrite or args.skip_existing,
                args.per_file_timeout_s,
            )
            if not status.startswith("ok"):
                failed += 1
            results.append((src.name, status))
    finally:
        # The warm child outlives files, never the batch — including when the
        # batch ends on Ctrl-C.
        _batch_child.close()

    print("================================================================================")
    print(f"BATCH SUMMARY: {len(inputs) - failed}/{len(inputs)} succeeded")
    for name, status in results:
        print(f"  {name}: {status}")
    print("================================================================================")
    return int(ExitCode.SUCCESS if failed == 0 else ExitCode.PUBLICATION_FAILURE)


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify an output audio master against its immutable JSON report."""
    public_audio = public_output_path(args.output)
    public_report = public_output_path(args.report)
    expected_report = publication_paths(public_audio).json
    committed = (
        resolve_committed_publication(public_audio) if public_report == expected_report else None
    )
    if committed is None:
        audio_path = public_audio.resolve()
        report_path = public_report.resolve()
    else:
        audio_path, report_path, _ = committed

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
    """Verify provenance for every registered core: params hash, weights
    digests, license allowlist, calibration integrity.

    Exits non-zero on any verification failure — an audit that cannot fail
    is not an audit.
    """
    policy_path = models_dir() / "license-policy.toml"
    calib_path = models_dir() / "guard-calibration.json"

    failures: list[str] = []

    print("================================================================================")
    print("                      HAWAVOCLEAN - MODEL AUDIT                          ")
    print("================================================================================")

    if not policy_path.exists():
        failures.append(f"license policy missing: {policy_path}")
        allowed: list[str] = []
    else:
        with open(policy_path, "rb") as f:
            allowed = list(tomllib.load(f).get("allowed_licenses", []))

    for core_id, registration in CORE_REGISTRY.items():
        lock_path = models_dir() / registration.lock_filename
        print(f"Core: {core_id}")
        if not lock_path.exists():
            failures.append(f"{core_id}: lockfile missing: {lock_path}")
            continue
        with open(lock_path, "rb") as f:
            lock = tomllib.load(f)

        print(f"  Algorithm:      {lock.get('algorithm')}")
        print(f"  Code license:   {lock.get('code_license')}")

        # 1a. Params hash matches the implemented core
        actual = registration.implementation_params_hash()
        if lock.get("params_hash") != actual:
            failures.append(
                f"{core_id}: params_hash mismatch: lockfile "
                f"{str(lock.get('params_hash'))[:16]}... vs implementation {actual[:16]}..."
            )
        else:
            print(f"  Params hash:    {actual[:16]}... [VERIFIED against implementation]")

        # 1b. The lock's own tables must reconstruct params_hash
        payload: dict[str, object] = dict(lock.get("params", {}))
        weight_table = {str(k): str(v) for k, v in dict(lock.get("weight_sha256", {})).items()}
        if weight_table:
            payload["weights_sha256"] = weight_table
        if hash_json_canonical(payload) != lock.get("params_hash"):
            failures.append(
                f"{core_id}: lock tables do not recompute to params_hash "
                "(the lockfile has been hand-edited)"
            )
        else:
            print("  Lock tables:    recompute to params_hash [VERIFIED]")

        # 2. Licenses must be allowlisted
        for key in ("code_license", "model_license", "wpe_license"):
            lic = lock.get(key)
            if lic is None:
                continue
            if allowed and str(lic) not in allowed:
                failures.append(f"{core_id}: {key} {lic!r} is not in the allowlist {allowed}")
            elif allowed:
                print(f"  License check:  {key}={lic!r} allowlisted [VERIFIED]")

        # 3. Weights digests must resolve against the files on disk
        for fname, digest in weight_table.items():
            candidate = models_dir() / str(fname)
            if not candidate.exists():
                failures.append(f"{core_id}: declared weights file missing: {fname}")
            elif hash_file(candidate) != digest:
                failures.append(f"{core_id}: weights digest mismatch for {fname}")
            else:
                print(f"  Weights:        {fname} [VERIFIED]")

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
    from hawavoclean.eval.calibrate import run_calibration

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
    from hawavoclean.eval.acceptance import evaluate_acceptance_gates

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
    from hawavoclean.research.benchmark import run_benchmark

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
    from hawavoclean.eval.blind_abx import generate_blind_trial_manifest

    print(f"Generating Blind ABX Listening Trial Sheet from manifest: {args.manifest}")
    out = generate_blind_trial_manifest(
        system_a_manifest=args.manifest,
        system_b_manifest=args.manifest_b or args.manifest,
        output_sheet_path=args.output,
    )
    print(f"Blind ABX Session created: {out}")
    return int(ExitCode.SUCCESS)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the loopback HTTP engine bridge for the UI (docs/ui-contract.md)."""
    try:
        from hawavoclean.server.app import run_server
    except ImportError as e:
        exit_with_code(
            ExitCode.PREFLIGHT_FAILURE,
            f"The engine server needs the 'ui' extra ({e}). Install it with: uv sync --extra ui",
        )
    ui_dir = Path(args.ui_dir).resolve() if args.ui_dir else None
    return run_server(host=args.host, port=int(args.port), token=args.token, ui_dir=ui_dir)


def _install_signal_handlers() -> None:
    """Make SIGTERM unwind the stack like SIGINT does.

    A bare SIGTERM kills the process without running finally: blocks, which
    would orphan the isolated worker subprocess and skip workspace cleanup.
    Raising KeyboardInterrupt lets every cleanup path run; the exit code
    stays conventional (130).

    This handler is also the parent-death watchdog's escalation rung: a
    process spawned from a background job inherits ``SIGINT=SIG_IGN`` across
    exec, and in that topology the watchdog self-delivers SIGTERM instead —
    so this must be installed before the watchdog is armed.
    """
    import signal

    def _raise(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, _raise)


def main() -> None:
    """CLI entry point: one boundary turns KeyboardInterrupt into a clean exit.

    The interrupt may be a Ctrl-C mid-run, the SIGTERM handler above, or the
    parent-death watchdog's self-interrupt — which, when the spawner died
    before the child finished arming, can land anywhere in startup, including
    inside ``threading.Thread.start()``. Catching at the outermost frame is
    the only place that covers them all: "Interrupted." and the conventional
    130, never a raw traceback.
    """
    try:
        _main()
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(130)


def _main() -> None:
    """Build the parser and dispatch (may raise KeyboardInterrupt)."""
    setup_logging()
    _install_signal_handlers()
    # Before argument parsing, let alone any work: everything above an armed
    # watchdog is time a killed parent's orphan spends unwatched.
    install_parent_death_watchdog()
    parser = argparse.ArgumentParser(
        prog="hawavoclean",
        description=(
            "HawaVoClean - dialogue audio cleanup (Wiener spectral denoising, "
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
    p_proc = subparsers.add_parser("process", help="Process an audio file through HawaVoClean")
    p_proc.add_argument("input", help="Path to input audio file")
    p_proc.add_argument("--output", "-o", required=True, help="Path to output mastered WAV")
    p_proc.add_argument("--config", "-c", help="Path to custom TOML configuration")
    p_proc.add_argument(
        "--profile", "-p", choices=["production", "development", "studio"], default="production"
    )
    p_proc.add_argument(
        "--overwrite", action="store_true", help="Overwrite destination output if exists"
    )
    p_proc.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit one JSON progress object per line on stdout (logs stay on stderr)",
    )
    p_proc.add_argument(
        "--passes",
        type=_passes_arg,
        default=1,
        metavar="N|auto",
        help=(
            f"Run the pipeline N times (1..{MAX_PASSES}), each pass re-enhancing the "
            "previous pass's mastered output; 'auto' adds passes only while the guard "
            "does not regress and measured speech/floor separation improves by >= 0.5 dB, "
            "discarding a pass that fails (default: 1). "
            "process only — batch always runs single-pass jobs."
        ),
    )
    p_proc.set_defaults(func=cmd_process)

    # batch
    p_batch = subparsers.add_parser(
        "batch", help="Process many files; failures are isolated and summarized"
    )
    p_batch.add_argument("inputs", nargs="+", help="Input audio files")
    p_batch.add_argument("--output-dir", "-o", required=True, help="Directory for outputs")
    p_batch.add_argument(
        "--profile", "-p", choices=["production", "development", "studio"], default="production"
    )
    p_batch.add_argument(
        "--suffix", default="_clean", help="Appended to each stem (default: _clean)"
    )
    p_batch.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_batch.add_argument(
        "--skip-existing", action="store_true", help="Skip inputs whose output already exists"
    )
    p_batch.add_argument(
        "--per-file-timeout-s",
        type=float,
        default=1800.0,
        help="Hard deadline per file; a hung file is killed and the batch continues (default 1800)",
    )
    p_batch.set_defaults(func=cmd_batch)

    # batch-worker: the long-lived child `batch` drives. Registered so it can
    # be launched by name, hidden because it is a protocol, not a command.
    p_bw = subparsers.add_parser("batch-worker", help=argparse.SUPPRESS)
    p_bw.set_defaults(func=cmd_batch_worker)

    # serve
    p_serve = subparsers.add_parser(
        "serve", help="Run the loopback HTTP engine bridge used by the HawaVoClean UI"
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Loopback address (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=0, help="TCP port; 0 = OS-assigned (default)")
    p_serve.add_argument(
        "--token", required=True, help="Shared secret every /api request must carry"
    )
    p_serve.add_argument("--ui-dir", help="Directory with index.html + assets/ to serve at /")
    p_serve.set_defaults(func=cmd_serve)

    # verify
    p_ver = subparsers.add_parser("verify", help="Verify output audio master against JSON report")
    p_ver.add_argument("output", help="Path to mastered WAV file")
    p_ver.add_argument("--report", "-r", required=True, help="Path to .hawavoclean.json report")
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
    try:
        code = args.func(args)
    except HawaVoCleanError as e:
        # Every subcommand maps known failures to their documented exit code —
        # not just `process`. No tracebacks for user-facing errors.
        logger.error(str(e))
        sys.exit(int(e.exit_code))
    sys.exit(code)


if __name__ == "__main__":
    main()
