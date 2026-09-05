"""Child-process job manager for the engine API.

Each job runs ``python -m hawavoclean.cli process IN -o OUT --profile P
[--overwrite] --progress-json`` in its own process (the same isolation the
batch command uses: a wedged decoder or model can never take the server
down). A reader thread turns the child's JSON progress lines into status
snapshots; stderr is tailed for the failure message. One job runs at a
time; the rest wait in a FIFO queue. Every child owns a POSIX process group
or Windows Job Object; cancel requests graceful shutdown, then force-ends the
complete tree after a grace period.

Subscribers (the SSE endpoint) register an ``asyncio.Queue`` bound to their
event loop; every status change is delivered with
``loop.call_soon_threadsafe`` so the worker thread never touches the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from hawavoclean.errors import ExitCode, MediaPreflightError, MediaPreflightReason
from hawavoclean.hashing import hash_file
from hawavoclean.logging import get_logger
from hawavoclean.paths import profiles_root, work_root
from hawavoclean.platform_fs import (
    ExclusiveFileLease,
    is_reparse_or_symlink,
    try_acquire_exclusive_file_lease,
)
from hawavoclean.process_supervisor import ProcessSupervisor
from hawavoclean.publication import (
    resolve_committed_publication,
    resolve_immutable_publication_generation,
)
from hawavoclean.record_bundle import ProcessingRecord, verify_processing_record
from hawavoclean.server.job_store import (
    ConflictPolicy,
    DurableJobStore,
    IdempotencyConflictError,
    JobStoreError,
    OutputConflictError,
    Reservation,
    canonical_request_hash,
    output_key,
    processing_record_path,
    unique_candidate,
    user_artifact_paths,
)
from hawavoclean.source_pin import PinnedSource, remove_source_snapshot_tree
from hawavoclean.watchdog import child_env

MAX_INPUT_FILE_BYTES = 8 * 1024**3
logger = get_logger("server.jobs")

JobState = Literal["queued", "running", "done", "failed", "cancelled", "interrupted"]
TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled", "interrupted"})
STDERR_TAIL_LINES = 50
DEFAULT_KILL_GRACE_S = 3.0
DEFAULT_MAX_ACTIVE_JOBS = 128
DEFAULT_MAX_TERMINAL_JOBS = 256
DEFAULT_TERMINAL_JOB_TTL_S = 24 * 60 * 60.0
_ARTIFACT_EVIDENCE_SCHEMA = 1
_SHA256_LENGTH = 64
_BASE_BUNDLE_EVIDENCE_KEYS = frozenset(
    {
        "path",
        "archive_sha256",
        "content_sha256",
        "master_sha256",
        "report_sha256",
        "summary_sha256",
        "total_uncompressed_bytes",
        "internal_hashes_verified",
        "authenticated_publisher",
    }
)
_ALL_BUNDLE_EVIDENCE_KEYS = _BASE_BUNDLE_EVIDENCE_KEYS | {"key_id", "signature_sha256"}
_BUNDLE_EVIDENCE_KEYS = _BASE_BUNDLE_EVIDENCE_KEYS


class QueueFullError(RuntimeError):
    """The bounded active-job queue has no capacity for another submission."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class JobRecord:
    """Mutable state of one job. Mutate only under ``JobManager._lock``."""

    job_id: str
    input_path: Path
    output_path: Path
    report_path: Path
    profile: str
    overwrite: bool
    idempotency_key: str | None = None
    conflict_policy: ConflictPolicy = "fail"
    request_hash: str = ""
    mode: str = "natural"
    speaker_id: str | None = None
    cutoff_hz: float | None = None
    record_bundle: bool = False
    bundle_path: Path | None = None
    bundle: dict[str, Any] | None = None
    batch_id: str | None = None
    source_snapshot_path: Path | None = None
    source_snapshot_dir: Path | None = None
    source_sha256: str | None = None
    source_size_bytes: int | None = None
    state: JobState = "queued"
    stage: str = "preflight"
    progress: float = 0.0
    message: str = "Queued"
    unit: dict[str, int] | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    terminal_at: float | None = None
    error: dict[str, str] | None = None
    report: dict[str, Any] | None = None
    artifact_evidence: dict[str, Any] | None = None
    cancel_requested: bool = False
    seq: int = 0  # bumps on every change; lets subscribers discard stale snapshots
    process: subprocess.Popen[str] | None = None
    supervisor: ProcessSupervisor | None = None
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=STDERR_TAIL_LINES))

    def snapshot(self) -> dict[str, Any]:
        """Contract ``JobStatus`` JSON."""
        out: dict[str, Any] = {
            "job_id": self.job_id,
            "seq": self.seq,
            "state": self.state,
            "stage": self.stage,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "report_path": str(self.report_path),
            "profile": self.profile,
            "mode": self.mode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.mode == "restore":
            # Natural-mode snapshots stay byte-compatible with revision 1
            # clients: the restore keys appear only when they mean something.
            out["speaker_id"] = self.speaker_id
            out["cutoff_hz"] = self.cutoff_hz
        if self.unit is not None:
            out["unit"] = dict(self.unit)
        if self.idempotency_key is not None:
            out["idempotency_key"] = self.idempotency_key
        if self.batch_id is not None:
            out["batch_id"] = self.batch_id
        out["conflict_policy"] = self.conflict_policy
        out["record_bundle"] = self.record_bundle
        if self.bundle_path is not None:
            out["bundle_path"] = str(self.bundle_path)
        if self.bundle is not None:
            out["bundle"] = dict(self.bundle)
        if self.state in {"failed", "interrupted"} and self.error is not None:
            out["error"] = dict(self.error)
        if self.state == "done" and self.report is not None:
            out["report"] = self.report
        if self.state == "done" and self.artifact_evidence is not None:
            # Internal/API artifact resolution needs all three job-bound
            # digests.  Audio alone is ambiguous when deterministic renders
            # produce identical masters but different reports or summaries.
            out["artifact_evidence"] = dict(self.artifact_evidence)
        return out

    def storage_record(self) -> dict[str, Any]:
        """Complete durable shape (runtime handles and monotonic clocks excluded)."""

        return {
            "job_id": self.job_id,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "report_path": str(self.report_path),
            "profile": self.profile,
            "overwrite": self.overwrite,
            "idempotency_key": self.idempotency_key,
            "conflict_policy": self.conflict_policy,
            "request_hash": self.request_hash,
            "mode": self.mode,
            "speaker_id": self.speaker_id,
            "cutoff_hz": self.cutoff_hz,
            "record_bundle": self.record_bundle,
            "bundle_path": str(self.bundle_path) if self.bundle_path is not None else None,
            "bundle": self.bundle,
            "batch_id": self.batch_id,
            "source_snapshot_path": (
                str(self.source_snapshot_path) if self.source_snapshot_path is not None else None
            ),
            "source_snapshot_dir": (
                str(self.source_snapshot_dir) if self.source_snapshot_dir is not None else None
            ),
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "state": self.state,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "unit": self.unit,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "report": self.report,
            "artifact_evidence": self.artifact_evidence,
            "cancel_requested": self.cancel_requested,
            "seq": self.seq,
        }

    @classmethod
    def from_storage(cls, value: dict[str, Any]) -> JobRecord:
        """Rehydrate a validated record from :mod:`job_store`."""

        state = str(value.get("state", ""))
        if state not in {"queued", "running", "done", "failed", "cancelled", "interrupted"}:
            raise JobStoreError(f"durable job has unsupported state {state!r}")
        conflict_policy = str(value.get("conflict_policy", "fail"))
        if conflict_policy not in {"unique", "fail", "replace"}:
            raise JobStoreError(f"durable job has unsupported conflict policy {conflict_policy!r}")
        unit = value.get("unit")
        error = value.get("error")
        report = value.get("report")
        artifact_evidence = value.get("artifact_evidence")
        return cls(
            job_id=str(value["job_id"]),
            input_path=Path(str(value["input_path"])),
            output_path=Path(str(value["output_path"])),
            report_path=Path(str(value["report_path"])),
            profile=str(value["profile"]),
            overwrite=bool(value.get("overwrite", False)),
            idempotency_key=(
                str(value["idempotency_key"]) if value.get("idempotency_key") is not None else None
            ),
            conflict_policy=cast(ConflictPolicy, conflict_policy),
            request_hash=str(value.get("request_hash", "")),
            mode=str(value.get("mode", "natural")),
            speaker_id=(str(value["speaker_id"]) if value.get("speaker_id") is not None else None),
            cutoff_hz=(float(value["cutoff_hz"]) if value.get("cutoff_hz") is not None else None),
            record_bundle=bool(value.get("record_bundle", False)),
            bundle_path=(
                Path(str(value["bundle_path"])) if value.get("bundle_path") is not None else None
            ),
            bundle=(
                cast(dict[str, Any], value["bundle"])
                if isinstance(value.get("bundle"), dict)
                else None
            ),
            batch_id=(str(value["batch_id"]) if value.get("batch_id") is not None else None),
            source_snapshot_path=(
                Path(str(value["source_snapshot_path"]))
                if value.get("source_snapshot_path") is not None
                else None
            ),
            source_snapshot_dir=(
                Path(str(value["source_snapshot_dir"]))
                if value.get("source_snapshot_dir") is not None
                else None
            ),
            source_sha256=(
                str(value["source_sha256"]) if value.get("source_sha256") is not None else None
            ),
            source_size_bytes=(
                int(value["source_size_bytes"])
                if value.get("source_size_bytes") is not None
                else None
            ),
            state=cast(JobState, state),
            stage=str(value.get("stage", "preflight")),
            progress=float(value.get("progress", 0.0)),
            message=str(value.get("message", "")),
            unit=(cast(dict[str, int], unit) if isinstance(unit, dict) else None),
            created_at=str(value.get("created_at") or _now_iso()),
            started_at=(str(value["started_at"]) if value.get("started_at") else None),
            finished_at=(str(value["finished_at"]) if value.get("finished_at") else None),
            error=(cast(dict[str, str], error) if isinstance(error, dict) else None),
            report=(cast(dict[str, Any], report) if isinstance(report, dict) else None),
            artifact_evidence=(
                cast(dict[str, Any], artifact_evidence)
                if isinstance(artifact_evidence, dict)
                else None
            ),
            cancel_requested=bool(value.get("cancel_requested", False)),
            seq=int(value.get("seq", 0)),
        )


CommandFactory = Callable[[JobRecord], list[str]]
TerminalCallback = Callable[[JobRecord], None]


def default_command(record: JobRecord) -> list[str]:
    """The contract child command (same isolation pattern as ``cli._run_one_isolated``)."""
    input_arg = (
        str(record.source_snapshot_path)
        if record.source_snapshot_path is not None
        else str(record.input_path)
    )
    cmd = [
        sys.executable,
        "-m",
        "hawavoclean.cli",
        "process",
        input_arg,
        "-o",
        str(record.output_path),
        "--profile",
        record.profile,
    ]
    if record.source_snapshot_path is not None and str(record.input_path) != input_arg:
        cmd += ["--original-input-path", str(record.input_path)]
    if record.mode == "restore":
        # The speaker id is validated at the API boundary (``^[a-z0-9_]{1,64}$``)
        # before it may reach a child argv, and the profiles dir is resolved
        # here rather than left to the child's cwd-relative default: the engine
        # may be launched from anywhere.
        cmd += ["--mode", "restore", "--speaker-id", str(record.speaker_id)]
        cmd += ["--profiles-dir", str(profiles_root())]
        if record.cutoff_hz is not None:
            cmd += ["--cutoff-hz", str(record.cutoff_hz)]
    elif record.mode == "smart_safe":
        cmd += ["--mode", "smart_safe"]
        if record.speaker_id is not None:
            cmd += ["--speaker-id", str(record.speaker_id)]
            cmd += ["--profiles-dir", str(profiles_root())]
        if record.cutoff_hz is not None:
            cmd += ["--cutoff-hz", str(record.cutoff_hz)]
    if record.overwrite:
        cmd.append("--overwrite")
    if record.record_bundle:
        if record.bundle_path is None:  # pragma: no cover - constructor invariant
            raise ValueError("record-bundle job is missing bundle_path")
        cmd += ["--record-bundle", str(record.bundle_path)]
    cmd.append("--progress-json")
    return cmd


def _bundle_evidence(record: ProcessingRecord) -> dict[str, Any]:
    """Closed durable evidence shape returned by status endpoints."""

    evidence: dict[str, Any] = {
        "path": str(record.path),
        "archive_sha256": record.archive_sha256,
        "content_sha256": record.content_sha256,
        "master_sha256": record.master_sha256,
        "report_sha256": record.report_sha256,
        "summary_sha256": record.summary_sha256,
        "total_uncompressed_bytes": record.total_uncompressed_bytes,
        "internal_hashes_verified": True,
        "authenticated_publisher": record.authenticated_publisher,
    }
    if record.key_id is not None:
        evidence["key_id"] = record.key_id
    if record.signature_sha256 is not None:
        evidence["signature_sha256"] = record.signature_sha256
    return evidence


def queue_position(queued: int, a_job_is_running: bool) -> int:
    """Where a just-submitted job stands in line, counting from 1.

    ``queued`` includes the newcomer; the job the worker is already running
    is no longer in the queue but is still ahead of it. Counting only the
    queue made the answer depend on whether the worker thread had got round
    to popping the first job yet — the same three submissions could report
    "position 3" or "position 2" from one run to the next, and the test that
    pinned it failed about one run in seven.
    """
    return queued + (1 if a_job_is_running else 0)


_Subscriber = tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]


class JobManager:
    """Single-worker FIFO scheduler backed by an optional durable job ledger."""

    def __init__(
        self,
        *,
        command_factory: CommandFactory | None = None,
        kill_grace_s: float = DEFAULT_KILL_GRACE_S,
        env: dict[str, str] | None = None,
        max_active_jobs: int = DEFAULT_MAX_ACTIVE_JOBS,
        max_terminal_jobs: int = DEFAULT_MAX_TERMINAL_JOBS,
        terminal_ttl_s: float = DEFAULT_TERMINAL_JOB_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        store_path: Path | None = None,
    ) -> None:
        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1")
        if max_terminal_jobs < 1:
            raise ValueError("max_terminal_jobs must be at least 1")
        if terminal_ttl_s <= 0:
            raise ValueError("terminal_ttl_s must be positive")
        self._command_factory: CommandFactory = command_factory or default_command
        self._kill_grace_s = kill_grace_s
        self._env = env
        self._max_active_jobs = max_active_jobs
        self._max_terminal_jobs = max_terminal_jobs
        self._terminal_ttl_s = terminal_ttl_s
        self._clock = clock
        self._wall_clock = wall_clock
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency: dict[str, str] = {}
        self._queue: deque[str] = deque()
        self._paused_batches: set[str] = set()
        self._submitting_inputs: dict[Path, int] = {}
        # Reentrant solely so ``prepare_batch`` can hold the scheduling
        # boundary while ordinary submit calls reuse their existing locking.
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._terminal_callbacks: list[TerminalCallback] = []
        self._pending_terminal: deque[JobRecord] = deque()
        self._persistence_error: str | None = None
        self._closed = False
        self._owner_lease: ExclusiveFileLease | None = None
        self._store: DurableJobStore | None = None
        if store_path is not None:
            try:
                store_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise JobStoreError(
                    f"could not create durable job store directory {store_path.parent}: {exc}"
                ) from exc
            owner_path = Path(f"{store_path}.owner.lock")
            try:
                self._owner_lease = try_acquire_exclusive_file_lease(owner_path)
            except BlockingIOError as exc:
                raise JobStoreError(
                    f"another engine broker already owns durable job store {store_path}"
                ) from exc
            except OSError as exc:
                raise JobStoreError(
                    f"could not acquire durable job-store owner lease {owner_path}: {exc}"
                ) from exc
            try:
                self._store = DurableJobStore(store_path)
            except BaseException:
                self._owner_lease.release()
                self._owner_lease = None
                raise
        try:
            if self._store is not None:
                wall_now = self._wall_clock()
                monotonic_now = self._clock()
                for value in self._store.load_and_interrupt(
                    max_terminal_jobs=self._max_terminal_jobs,
                    terminal_ttl_s=self._terminal_ttl_s,
                    now_epoch=wall_now,
                ):
                    record = JobRecord.from_storage(value)
                    self._reconcile_completed_after_restart(record)
                    if record.state in TERMINAL_STATES:
                        terminal_epoch = value.get("_terminal_at_epoch")
                        if isinstance(terminal_epoch, (int, float)):
                            age = max(0.0, wall_now - float(terminal_epoch))
                            record.terminal_at = monotonic_now - age
                        else:
                            # A v2 terminal row without a timestamp is corrupt;
                            # make it immediately eligible for fail-closed prune.
                            record.terminal_at = monotonic_now - self._terminal_ttl_s
                    self._jobs[record.job_id] = record
                    if record.idempotency_key is not None:
                        self._idempotency[record.idempotency_key] = record.job_id
                    self._store.update(record.storage_record(), terminal=True)
                self._prune_locked()
            self._worker = threading.Thread(
                target=self._run_loop, name="hawavoclean-jobs", daemon=True
            )
            self._worker.start()
        except BaseException:
            if self._store is not None:
                self._store.close()
            if self._owner_lease is not None:
                self._owner_lease.release()
                self._owner_lease = None
            raise

    @staticmethod
    def _closed_bundle_evidence(record: JobRecord, evidence: dict[str, Any]) -> dict[str, Any]:
        """Validate the exact durable identity retained for one record ZIP."""

        keys = set(evidence)
        if record.bundle_path is None or not (
            _BASE_BUNDLE_EVIDENCE_KEYS <= keys <= _ALL_BUNDLE_EVIDENCE_KEYS
        ):
            raise JobStoreError("record-bundle job lacks closed durable evidence")
        expected_path = record.bundle_path.expanduser().absolute()
        if Path(str(evidence.get("path"))).expanduser().absolute() != expected_path:
            raise JobStoreError("Processing Record evidence path differs from its reservation")
        for digest_field in (
            "archive_sha256",
            "content_sha256",
            "master_sha256",
            "report_sha256",
            "summary_sha256",
        ):
            digest = evidence.get(digest_field)
            if (
                not isinstance(digest, str)
                or len(digest) != _SHA256_LENGTH
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise JobStoreError(f"Processing Record {digest_field} is invalid")
        size = evidence.get("total_uncompressed_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise JobStoreError("Processing Record uncompressed size evidence is invalid")
        if evidence.get("internal_hashes_verified") is not True or not isinstance(
            evidence.get("authenticated_publisher"), bool
        ):
            raise JobStoreError("Processing Record verification evidence is invalid")
        if "key_id" in evidence:
            key_id = evidence["key_id"]
            if not isinstance(key_id, str) or not key_id:
                raise JobStoreError("Processing Record key_id evidence is invalid")
        if "signature_sha256" in evidence:
            sig_digest = evidence["signature_sha256"]
            if (
                not isinstance(sig_digest, str)
                or len(sig_digest) != _SHA256_LENGTH
                or any(character not in "0123456789abcdef" for character in sig_digest)
            ):
                raise JobStoreError("Processing Record signature_sha256 evidence is invalid")
        return dict(evidence)

    @staticmethod
    def _bundle_matches_evidence(verified: ProcessingRecord, evidence: dict[str, Any]) -> bool:
        verified_evidence = _bundle_evidence(verified)
        if set(evidence) == _BASE_BUNDLE_EVIDENCE_KEYS:
            return {k: verified_evidence.get(k) for k in _BASE_BUNDLE_EVIDENCE_KEYS} == evidence
        return verified_evidence == evidence

    def _bundle_artifact_paths(
        self, record: JobRecord, evidence: dict[str, Any]
    ) -> tuple[Path, Path, Path]:
        """Resolve the exact generation described by durable ZIP evidence."""

        exact = resolve_immutable_publication_generation(
            record.output_path,
            audio_sha256=str(evidence["master_sha256"]),
            report_sha256=str(evidence["report_sha256"]),
            summary_sha256=str(evidence["summary_sha256"]),
        )
        if exact is not None:
            return exact
        current = resolve_committed_publication(record.output_path)
        if current is not None:
            raise JobStoreError("Processing Record's immutable publication generation is missing")
        legacy = (record.output_path, record.report_path, self._summary_path(record))
        if not self._artifacts_match_bundle_evidence(legacy, evidence):
            raise JobStoreError("Processing Record differs from the legacy export triplet")
        return legacy

    @staticmethod
    def _artifacts_match_bundle_evidence(
        artifacts: tuple[Path, Path, Path], evidence: dict[str, Any]
    ) -> bool:
        try:
            return all(
                hash_file(path) == str(evidence[field])
                for path, field in zip(
                    artifacts,
                    ("master_sha256", "report_sha256", "summary_sha256"),
                    strict=True,
                )
            )
        except OSError:
            return False

    def _current_publication_matches_bundle(
        self, record: JobRecord, evidence: dict[str, Any]
    ) -> bool:
        try:
            current = resolve_committed_publication(record.output_path)
        except Exception:
            return False
        artifacts = (
            current
            if current is not None
            else (record.output_path, record.report_path, self._summary_path(record))
        )
        return self._artifacts_match_bundle_evidence(artifacts, evidence)

    def _validate_bundle_artifacts(
        self,
        record: JobRecord,
        *,
        expected_evidence: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate a bundle without rebinding an old job to shared current paths."""

        if not record.record_bundle or record.bundle_path is None:
            raise JobStoreError("record-bundle job is missing its durable bundle path")
        if expected_evidence is None:
            verified = verify_processing_record(record.bundle_path)
            evidence = self._closed_bundle_evidence(record, _bundle_evidence(verified))
        else:
            evidence = self._closed_bundle_evidence(record, expected_evidence)
            try:
                shared = verify_processing_record(record.bundle_path)
            except Exception:
                shared_matches = False
            else:
                shared_matches = self._bundle_matches_evidence(shared, evidence)
            current_matches = self._current_publication_matches_bundle(record, evidence)
            if not shared_matches and current_matches:
                raise JobStoreError("the current job's Processing Record is missing or changed")

        export_audio, export_report, export_summary = self._bundle_artifact_paths(record, evidence)
        if not self._artifacts_match_bundle_evidence(
            (export_audio, export_report, export_summary), evidence
        ):
            raise JobStoreError("Processing Record evidence differs from its exact generation")
        try:
            report = json.loads(export_report.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JobStoreError(f"published report is unreadable: {exc}") from exc
        if not isinstance(report, dict):
            raise JobStoreError("published report is not a JSON object")
        if self._report_audio_sha256(cast(dict[str, Any], report)) != evidence["master_sha256"]:
            raise JobStoreError("Processing Record report describes different audio")
        return evidence, cast(dict[str, Any], report)

    @staticmethod
    def _report_audio_sha256(report: dict[str, Any]) -> str:
        output = report.get("output")
        digest = output.get("sha256") if isinstance(output, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise JobStoreError("published report lacks a canonical output SHA-256")
        return digest

    @staticmethod
    def _artifact_record(path: Path) -> dict[str, Any]:
        if is_reparse_or_symlink(path) or not path.is_file():
            raise JobStoreError(f"completed artifact is missing or unsafe: {path}")
        return {"sha256": hash_file(path), "size_bytes": path.stat().st_size}

    @staticmethod
    def _summary_path(record: JobRecord) -> Path:
        return record.output_path.parent / f"{record.output_path.stem}.hawavoclean.txt"

    def _capture_nonbundle_artifacts(self, record: JobRecord) -> None:
        """Bind a successful job to exact immutable artifact hashes."""

        if record.report is None:
            raise JobStoreError("completed job has no report object")
        expected_audio = self._report_audio_sha256(record.report)
        committed = resolve_committed_publication(record.output_path)
        if committed is None:
            artifacts = (record.output_path, record.report_path, self._summary_path(record))
            storage = "legacy_flat"
            generation_id: str | None = None
        else:
            artifacts = committed
            storage = "immutable_generation"
            generation_id = committed[0].parent.name
        audio_record, report_record, summary_record = (
            self._artifact_record(path) for path in artifacts
        )
        if audio_record["sha256"] != expected_audio:
            raise JobStoreError("published audio differs from report.output.sha256")
        try:
            report = json.loads(artifacts[1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JobStoreError(f"published report is unreadable: {exc}") from exc
        if not isinstance(report, dict):
            raise JobStoreError("published report is not a JSON object")
        if self._report_audio_sha256(cast(dict[str, Any], report)) != expected_audio:
            raise JobStoreError("resolved report describes different audio")
        record.report = cast(dict[str, Any], report)
        record.artifact_evidence = {
            "schema_version": _ARTIFACT_EVIDENCE_SCHEMA,
            "storage": storage,
            "generation_id": generation_id,
            "audio": audio_record,
            "report": report_record,
            "summary": summary_record,
        }

    def _validate_nonbundle_artifacts(self, record: JobRecord) -> None:
        """Rehydrate and verify one completed job from job-bound evidence."""

        evidence = record.artifact_evidence
        if not isinstance(evidence, dict) or set(evidence) != {
            "schema_version",
            "storage",
            "generation_id",
            "audio",
            "report",
            "summary",
        }:
            raise JobStoreError("completed job lacks closed artifact evidence")
        if evidence.get("schema_version") != _ARTIFACT_EVIDENCE_SCHEMA:
            raise JobStoreError("completed job artifact evidence has an unsupported schema")
        storage = evidence.get("storage")
        if storage not in {"immutable_generation", "legacy_flat"}:
            raise JobStoreError("completed job artifact storage kind is invalid")
        audio_evidence = evidence.get("audio")
        report_evidence = evidence.get("report")
        summary_evidence = evidence.get("summary")
        for role, expected in (
            ("audio", audio_evidence),
            ("report", report_evidence),
            ("summary", summary_evidence),
        ):
            if not isinstance(expected, dict) or set(expected) != {"sha256", "size_bytes"}:
                raise JobStoreError(f"completed job {role} evidence is invalid")
        assert isinstance(audio_evidence, dict)
        assert isinstance(report_evidence, dict)
        assert isinstance(summary_evidence, dict)
        expected_audio = audio_evidence.get("sha256")
        expected_report = report_evidence.get("sha256")
        expected_summary = summary_evidence.get("sha256")
        if not all(
            isinstance(value, str) for value in (expected_audio, expected_report, expected_summary)
        ):
            raise JobStoreError("completed job artifact digest is missing")
        committed = resolve_immutable_publication_generation(
            record.output_path,
            audio_sha256=cast(str, expected_audio),
            report_sha256=cast(str, expected_report),
            summary_sha256=cast(str, expected_summary),
        )
        if committed is None:
            if storage != "legacy_flat":
                raise JobStoreError("job's immutable publication generation is missing")
            artifacts = (record.output_path, record.report_path, self._summary_path(record))
        else:
            artifacts = committed
            expected_generation = evidence.get("generation_id")
            if (
                storage == "immutable_generation"
                and isinstance(expected_generation, str)
                and committed[0].parent.name != expected_generation
            ):
                raise JobStoreError("resolved publication generation differs from job evidence")
        for role, path in zip(("audio", "report", "summary"), artifacts, strict=True):
            expected = evidence.get(role)
            assert isinstance(expected, dict)
            if self._artifact_record(path) != expected:
                raise JobStoreError(f"completed job {role} failed digest validation")
        try:
            report = json.loads(artifacts[1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JobStoreError(f"published report is unreadable: {exc}") from exc
        if not isinstance(report, dict):
            raise JobStoreError("published report is not a JSON object")
        if self._report_audio_sha256(cast(dict[str, Any], report)) != expected_audio:
            raise JobStoreError("published report no longer describes the job audio")
        record.report = cast(dict[str, Any], report)

    @staticmethod
    def _mark_artifact_invalid(record: JobRecord, exc: Exception) -> None:
        record.bundle = None
        record.artifact_evidence = None
        record.state = "failed"
        record.stage = "error"
        record.message = "Completed artifacts failed startup validation"
        record.error = {"code": "ARTIFACT_INVALID", "message": str(exc)}
        record.finished_at = _now_iso()
        record.seq += 1

    def _reconcile_completed_after_restart(self, record: JobRecord) -> None:
        """Revalidate completed artifacts; never infer completion from paths.

        In replace mode the derived paths may still hold a perfectly valid
        *prior* export when a new job crashes before doing any work. Without
        job-bound evidence durably recorded before the terminal transition,
        promoting an interrupted row would silently attribute that old export
        to the new job. Interrupted therefore remains interrupted even when
        its paths happen to verify.
        """

        if record.state != "done":
            record.bundle = None
            return
        try:
            if record.record_bundle:
                if record.bundle is None:
                    raise JobStoreError("completed record-bundle job lacks durable evidence")
                bundle, report = self._validate_bundle_artifacts(
                    record, expected_evidence=record.bundle
                )
            else:
                self._validate_nonbundle_artifacts(record)
        except Exception as exc:
            self._mark_artifact_invalid(record, exc)
            return
        if record.record_bundle:
            record.bundle = bundle
            record.report = report

    # ------------------------------------------------------------------ public

    def submit(
        self,
        *,
        input_path: Path,
        output_path: Path,
        profile: str,
        overwrite: bool,
        mode: str = "natural",
        speaker_id: str | None = None,
        cutoff_hz: float | None = None,
        idempotency_key: str | None = None,
        conflict_policy: ConflictPolicy | None = None,
        request_context_hash: str | None = None,
        record_bundle: bool = False,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Queue a job or return its prior snapshot for an idempotent retry."""

        if idempotency_key is not None:
            if not idempotency_key.strip() or len(idempotency_key) > 128:
                raise ValueError("idempotency_key must contain 1-128 characters")
            if any(ord(char) < 0x21 or ord(char) > 0x7E for char in idempotency_key):
                raise ValueError("idempotency_key must contain visible ASCII characters only")
        norm_input = input_path.resolve()
        pinned: PinnedSource | None = None
        source_sha256: str | None = None
        source_size_bytes: int | None = None
        with self._lock:
            self._submitting_inputs[norm_input] = self._submitting_inputs.get(norm_input, 0) + 1
        try:
            policy: ConflictPolicy = conflict_policy or ("replace" if overwrite else "fail")
            if policy not in {"unique", "fail", "replace"}:
                raise ValueError(f"unsupported conflict policy: {policy}")
            if input_path.exists() and input_path.is_file():
                pinned = PinnedSource.create(
                    input_path,
                    staging_root=work_root(),
                    max_file_size_bytes=MAX_INPUT_FILE_BYTES,
                )
                source_sha256 = pinned.sha256
                source_size_bytes = pinned.size_bytes
            else:
                existing_rec: JobRecord | None = None
                if idempotency_key is not None:
                    with self._wake:
                        if idempotency_key in self._idempotency:
                            existing_rec = self._jobs.get(self._idempotency[idempotency_key])
                        elif self._store is not None:
                            durable_res = self._store.find_idempotent(idempotency_key)
                            if durable_res is not None:
                                existing_rec = JobRecord.from_storage(durable_res.record)
                if existing_rec is not None and existing_rec.state in TERMINAL_STATES:
                    source_sha256 = existing_rec.source_sha256
                    source_size_bytes = existing_rec.source_size_bytes
                elif self._command_factory is default_command:
                    raise MediaPreflightError(
                        MediaPreflightReason.NOT_FOUND,
                        f"Input audio file does not exist or cannot be read: {input_path}",
                    )

            effective_overwrite = policy == "replace"
            report_path = output_path.parent / f"{output_path.stem}.hawavoclean.json"
            bundle_path = (
                output_path.parent / f"{output_path.stem}.hawavoclean.zip"
                if record_bundle
                else None
            )
            request_payload = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "profile": profile,
                "mode": mode,
                "speaker_id": speaker_id,
                "cutoff_hz": cutoff_hz,
                "conflict_policy": policy,
                "request_context_hash": request_context_hash,
                "record_bundle": record_bundle,
                "source_sha256": source_sha256,
                "source_size_bytes": source_size_bytes,
            }
            request_hash = canonical_request_hash(request_payload)

            with self._wake:
                if self._closed:
                    raise RuntimeError("job manager is shut down")
                self._prune_locked()
                if idempotency_key is not None and idempotency_key in self._idempotency:
                    existing = self._jobs.get(self._idempotency[idempotency_key])
                    if existing is not None:
                        if existing.request_hash != request_hash:
                            raise IdempotencyConflictError(
                                "idempotency key is already bound to a different request"
                            )
                        if pinned is not None:
                            pinned.cleanup_unadopted()
                        return existing.snapshot()
                if self._store is not None and idempotency_key is not None:
                    durable_retry = self._store.find_idempotent(
                        idempotency_key, request_hash=request_hash
                    )
                    if durable_retry is not None:
                        if pinned is not None:
                            pinned.cleanup_unadopted()
                        return self._snapshot_durable_retry_locked(durable_retry)
                active = sum(r.state not in TERMINAL_STATES for r in self._jobs.values())
                if active >= self._max_active_jobs:
                    raise QueueFullError(
                        f"job queue is full ({active}/{self._max_active_jobs} active jobs)"
                    )

            record = JobRecord(
                job_id=f"j_{uuid.uuid4().hex[:16]}",
                input_path=input_path,
                output_path=output_path,
                report_path=report_path,
                profile=profile,
                overwrite=effective_overwrite,
                idempotency_key=idempotency_key,
                conflict_policy=policy,
                request_hash=request_hash,
                mode=mode,
                speaker_id=speaker_id,
                cutoff_hz=cutoff_hz,
                record_bundle=record_bundle,
                bundle_path=bundle_path,
                batch_id=batch_id,
                source_snapshot_path=pinned.path if pinned is not None else None,
                source_snapshot_dir=pinned.directory if pinned is not None else None,
                source_sha256=pinned.sha256 if pinned is not None else None,
                source_size_bytes=pinned.size_bytes if pinned is not None else None,
            )
            with self._wake:
                if self._closed:
                    raise RuntimeError("job manager is shut down")
                self._prune_locked()
                active = sum(r.state not in TERMINAL_STATES for r in self._jobs.values())
                if active >= self._max_active_jobs:
                    raise QueueFullError(
                        f"job queue is full ({active}/{self._max_active_jobs} active jobs)"
                    )
                if self._store is not None:
                    reservation = self._store.reserve(
                        record=record.storage_record(),
                        request_hash=request_hash,
                        idempotency_key=idempotency_key,
                        conflict_policy=policy,
                    )
                    record = JobRecord.from_storage(reservation.record)
                    if reservation.reused:
                        if pinned is not None:
                            pinned.cleanup_unadopted()
                        return self._snapshot_durable_retry_locked(reservation)
                else:
                    record = self._reserve_in_memory(record)
                self._jobs[record.job_id] = record
                if idempotency_key is not None:
                    self._idempotency[idempotency_key] = record.job_id
                self._queue.append(record.job_id)
                running = any(r.state == "running" for r in self._jobs.values())
                position = queue_position(len(self._queue), running)
                record.message = "Queued" if position == 1 else f"Queued (position {position})"
                self._persist_locked(record)
                self._wake.notify_all()
                return record.snapshot()
        except BaseException:
            if pinned is not None:
                pinned.cleanup_unadopted()
            raise
        finally:
            with self._lock:
                count = self._submitting_inputs.get(norm_input, 0) - 1
                if count <= 0:
                    self._submitting_inputs.pop(norm_input, None)
                else:
                    self._submitting_inputs[norm_input] = count

    def _snapshot_durable_retry_locked(self, reservation: Reservation) -> dict[str, Any]:
        """Rehydrate an exact retry without turning it into fresh history."""

        record = JobRecord.from_storage(reservation.record)
        prior_state = record.state
        self._reconcile_completed_after_restart(record)
        if prior_state == "done" and record.state == "failed" and self._store is not None:
            self._store.update(record.storage_record(), terminal=True)
        if record.state in TERMINAL_STATES and reservation.terminal_at_epoch is not None:
            age = max(0.0, self._wall_clock() - reservation.terminal_at_epoch)
            record.terminal_at = self._clock() - age
        # A terminal row absent from ``_jobs`` was already pruned (or its
        # durable prune hit an I/O failure). Returning it must not re-expand
        # bounded history. Nonterminal rows are cached defensively, though a
        # normal owned broker always has them in memory already.
        if reservation.history_visible and record.state not in TERMINAL_STATES:
            self._jobs.setdefault(record.job_id, record)
            if record.idempotency_key is not None:
                self._idempotency[record.idempotency_key] = record.job_id
            return self._jobs[record.job_id].snapshot()
        return record.snapshot()

    @contextlib.contextmanager
    def prepare_batch(self) -> Iterator[None]:
        """Prevent any new item from executing until the whole batch is accepted.

        The API validates sources first, then performs its normal ``submit``
        calls inside this boundary. If a late collision/queue/persistence
        error escapes, every newly prepared row is removed before the worker
        can observe it. Existing idempotent jobs are never rolled back.
        """

        with self._wake:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            before = set(self._jobs)
            try:
                yield
            except BaseException:
                prepared = [
                    job_id
                    for job_id in self._jobs
                    if job_id not in before and self._jobs[job_id].state == "queued"
                ]
                for job_id in prepared:
                    with contextlib.suppress(ValueError):
                        self._queue.remove(job_id)
                try:
                    if self._store is not None:
                        self._store.delete_queued(prepared)
                except JobStoreError as exc:
                    self._persistence_error = str(exc)
                    # The scheduling lock still guarantees none can publish.
                    # Leave them cancelled and durable rather than forgetting
                    # an accepted row whose deletion did not commit.
                    for job_id in prepared:
                        record = self._jobs[job_id]
                        record.state = "cancelled"
                        record.stage = "cancelled"
                        record.finished_at = _now_iso()
                        record.terminal_at = self._clock()
                        record.message = "Batch preparation rolled back"
                        self._notify_locked(record)
                    raise JobStoreError(
                        f"batch preparation failed and rollback was not durable: {exc}"
                    ) from exc
                else:
                    for job_id in prepared:
                        record = self._jobs.pop(job_id)
                        if record.idempotency_key is not None:
                            self._idempotency.pop(record.idempotency_key, None)
                        self._subscribers.pop(job_id, None)
                raise
        self._drain_terminal_callbacks()

    def _reserve_in_memory(self, record: JobRecord) -> JobRecord:
        """Apply the same conflict contract when no SQLite store is configured."""

        desired = record.output_path
        candidate = desired
        ordinal = 1
        active_keys = {
            output_key(item.output_path)
            for item in self._jobs.values()
            if item.state not in TERMINAL_STATES
        }
        while True:
            key = output_key(candidate)
            occupied = key in active_keys or any(
                os.path.lexists(path)
                for path in user_artifact_paths(candidate, record_bundle=record.record_bundle)
            )
            stale_processing_record = not record.record_bundle and os.path.lexists(
                processing_record_path(candidate)
            )
            if record.conflict_policy == "unique" and occupied:
                ordinal += 1
                candidate = unique_candidate(desired, ordinal)
                if ordinal > 100_000:
                    raise OutputConflictError("could not allocate a unique output name")
                continue
            if key in active_keys:
                raise OutputConflictError(f"output is reserved by an active job: {candidate}")
            if record.conflict_policy == "replace" and stale_processing_record:
                raise OutputConflictError(
                    "same-stem Processing Record already exists; use unique naming "
                    "or request a replacement Processing Record"
                )
            if record.conflict_policy == "fail" and occupied:
                raise OutputConflictError(f"output or sidecar already exists for: {candidate}")
            break
        record.output_path = candidate
        record.report_path = candidate.parent / f"{candidate.stem}.hawavoclean.json"
        record.bundle_path = (
            candidate.parent / f"{candidate.stem}.hawavoclean.zip" if record.record_bundle else None
        )
        return record

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._prune_locked()
            record = self._jobs.get(job_id)
            if record is not None:
                return record.snapshot()
            if self._store is None:
                return None
            resource = self._store.find_job(job_id)
            return self._snapshot_durable_retry_locked(resource) if resource is not None else None

    def get_by_idempotency(self, idempotency_key: str) -> dict[str, Any] | None:
        """Return the durable prior job for a client retry, if one exists."""

        with self._lock:
            job_id = self._idempotency.get(idempotency_key)
            record = self._jobs.get(job_id) if job_id is not None else None
            if record is not None:
                return record.snapshot()
            if self._store is None:
                return None
            reservation = self._store.find_idempotent(idempotency_key)
            return (
                self._snapshot_durable_retry_locked(reservation)
                if reservation is not None
                else None
            )

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            return [r.snapshot() for r in self._jobs.values()]

    @property
    def persistence_error(self) -> str | None:
        """Latest durable-ledger failure, exposed for health/readiness checks."""

        with self._lock:
            return self._persistence_error

    @property
    def durable(self) -> bool:
        """Whether submissions survive an engine restart."""

        return self._store is not None

    def active_input_paths(self) -> set[Path]:
        """Inputs that retention cleanup must not remove while queued/running or submitting."""
        with self._lock:
            active = {
                record.input_path.resolve()
                for record in self._jobs.values()
                if record.state not in TERMINAL_STATES
            }
            active.update(self._submitting_inputs.keys())
            return active

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold the job lock, then run terminal cleanup *outside* it.

        Terminal callbacks are caller-supplied cleanup, and cleanup routinely
        has to ask this manager a question before it acts -- retention must
        know whether another live job still names an upload as its input
        before deleting it. Invoking callbacks from inside ``_finish_locked``
        meant asking that question re-entered a non-reentrant ``Lock`` and
        deadlocked the ``hawavoclean-jobs`` thread permanently: not an
        exception the surrounding ``try`` could log, just a thread that never
        came back, taking every later job with it.

        Callbacks also do filesystem work, which has no business holding a
        lock that every status query needs. So ``_finish_locked`` records who
        became terminal and this wrapper delivers them after the release.
        """
        try:
            with self._lock:
                yield
        finally:
            # ``finally``, not a plain suffix: a body that raises has still
            # marked jobs terminal, and their cleanup must not wait for the
            # next unrelated caller to flush the queue.
            self._drain_terminal_callbacks()

    def _drain_terminal_callbacks(self) -> None:
        """Deliver queued terminal records. Safe to call with the lock free."""
        while True:
            with self._lock:
                if not self._pending_terminal:
                    return
                record = self._pending_terminal.popleft()
                callbacks = list(self._terminal_callbacks)
            for callback in callbacks:
                try:
                    callback(record)
                except Exception as exc:  # cleanup cannot change the job verdict
                    logger.error(
                        f"Terminal cleanup for {record.job_id} failed: {exc}", exc_info=True
                    )

    def add_terminal_callback(self, callback: TerminalCallback) -> None:
        """Register idempotent, bounded cleanup invoked when any job becomes terminal."""
        with self._lock:
            self._terminal_callbacks.append(callback)

    def cancel(self, job_id: str, *, wait: bool = False, timeout_s: float | None = None) -> bool:
        """Cancel a queued or running job. Returns False for an unknown id;
        True otherwise (including the no-op on an already finished job).

        When ``wait=True``, synchronously awaits termination and terminal
        state transition, releasing all output reservations before return.
        """
        with self._locked():
            self._prune_locked()
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if record.state in TERMINAL_STATES:
                return True
            record.cancel_requested = True
            if record.state == "queued":
                with contextlib.suppress(ValueError):
                    self._queue.remove(job_id)
                self._finish_locked(record, "cancelled", "Cancelled before start")
                return True
            supervisor = record.supervisor
        if supervisor is not None:
            if wait:
                supervisor.terminate_tree(self._kill_grace_s)
            else:
                self._terminate(supervisor)
        if wait:
            deadline = time.monotonic() + (
                timeout_s if timeout_s is not None else (self._kill_grace_s + 5.0)
            )
            while time.monotonic() < deadline:
                with self._lock:
                    if record.state in TERMINAL_STATES:
                        return True
                time.sleep(0.02)
        return True

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]] | None:
        """Register the calling event loop for status pushes. Returns None for
        an unknown job. Call from inside a running asyncio loop."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        with self._lock:
            self._prune_locked()
            if job_id not in self._jobs:
                return None
            self._subscribers.setdefault(job_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            remaining = [s for s in subs if s[1] is not queue]
            if remaining:
                self._subscribers[job_id] = remaining
            else:
                self._subscribers.pop(job_id, None)

    def register_batch(
        self,
        batch_id: str,
        total_items: int,
        *,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Register batch submission metadata in the durable store."""
        if self._store is not None:
            self._store.create_batch(batch_id, total_items, options=options)

    def pause_batch(self, batch_id: str) -> bool:
        """Pause scheduling of queued items in this batch."""
        with self._wake:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            self._prune_locked()
            has_batch = any(r.batch_id == batch_id for r in self._jobs.values()) or (
                self._store is not None and self._store.get_batch(batch_id) is not None
            )
            if not has_batch:
                return False
            self._paused_batches.add(batch_id)
            for record in self._jobs.values():
                if record.batch_id == batch_id and record.state == "queued":
                    record.message = "Batch paused"
                    self._notify_locked(record)
            if self._store is not None:
                self._store.update_batch_state(batch_id, "paused")
            return True

    def resume_batch(self, batch_id: str) -> bool:
        """Resume scheduling of paused items in this batch."""
        with self._wake:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            self._prune_locked()
            has_batch = any(r.batch_id == batch_id for r in self._jobs.values()) or (
                self._store is not None and self._store.get_batch(batch_id) is not None
            )
            if not has_batch:
                return False
            self._paused_batches.discard(batch_id)
            running = any(r.state == "running" for r in self._jobs.values())
            for record in self._jobs.values():
                if record.batch_id == batch_id and record.state == "queued":
                    idx = (
                        list(self._queue).index(record.job_id) + 1
                        if record.job_id in self._queue
                        else 1
                    )
                    pos = queue_position(idx, running)
                    record.message = "Queued" if pos == 1 else f"Queued (position {pos})"
                    self._notify_locked(record)
            if self._store is not None:
                self._store.update_batch_state(batch_id, "running")
            self._wake.notify_all()
            return True

    def cancel_batch(self, batch_id: str, *, wait: bool = False) -> bool:
        """Cancel all queued and running jobs belonging to this batch."""
        with self._locked():
            self._prune_locked()
            matching_ids = [r.job_id for r in self._jobs.values() if r.batch_id == batch_id]
            if not matching_ids and self._store is not None:
                db_jobs = self._store.find_batch_jobs(batch_id)
                matching_ids = [
                    str(r.record.get("job_id")) for r in db_jobs if r.record.get("job_id")
                ]
        if not matching_ids:
            return False
        for job_id in matching_ids:
            self.cancel(job_id, wait=wait)
        if self._store is not None:
            self._store.update_batch_state(batch_id, "cancelled")
        return True

    def retry_job(self, job_id: str) -> dict[str, Any]:
        """Retry a failed, interrupted, or cancelled job."""
        with self._wake:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            self._prune_locked()
            record = self._jobs.get(job_id)
            if record is None and self._store is not None:
                resource = self._store.find_job(job_id)
                if resource is not None:
                    record = JobRecord.from_storage(resource.record)
                    self._jobs[record.job_id] = record
            if record is None:
                raise KeyError(f"job {job_id} not found")
            if record.state not in TERMINAL_STATES:
                return record.snapshot()

            active = sum(r.state not in TERMINAL_STATES for r in self._jobs.values())
            if active >= self._max_active_jobs:
                raise QueueFullError(
                    f"job queue is full ({active}/{self._max_active_jobs} active jobs)"
                )

            if not record.input_path.exists() or not record.input_path.is_file():
                raise MediaPreflightError(
                    MediaPreflightReason.NOT_FOUND,
                    f"Input audio file does not exist or cannot be read: {record.input_path}",
                )

            pinned = PinnedSource.create(
                record.input_path,
                staging_root=work_root(),
                max_file_size_bytes=MAX_INPUT_FILE_BYTES,
            )
            record.source_snapshot_path = pinned.path
            record.source_snapshot_dir = pinned.directory
            record.source_sha256 = pinned.sha256
            record.source_size_bytes = pinned.size_bytes

            record.state = "queued"
            record.stage = "preflight"
            record.progress = 0.0
            record.message = "Queued (retry)"
            record.started_at = None
            record.finished_at = None
            record.terminal_at = None
            record.error = None
            record.cancel_requested = False
            record.seq += 1

            if self._store is not None:
                self._store.update(record.storage_record(), terminal=False)
                if record.batch_id is not None and record.batch_id not in self._paused_batches:
                    self._store.update_batch_state(record.batch_id, "running")

            self._queue.append(record.job_id)
            running = any(r.state == "running" for r in self._jobs.values())
            position = queue_position(len(self._queue), running)
            record.message = "Queued" if position == 1 else f"Queued (position {position})"
            self._notify_locked(record)
            self._wake.notify_all()
            return record.snapshot()

    def get_batch_summary(self, batch_id: str) -> dict[str, Any] | None:
        """Aggregate batch state across in-memory and durable jobs."""
        with self._lock:
            self._prune_locked()
            batch_jobs = [r for r in self._jobs.values() if r.batch_id == batch_id]

        db_batch = self._store.get_batch(batch_id) if self._store is not None else None
        if not batch_jobs and db_batch is None:
            return None

        if self._store is not None:
            db_job_res = self._store.find_batch_jobs(batch_id)
            known_ids = {r.job_id for r in batch_jobs}
            for res in db_job_res:
                rec = JobRecord.from_storage(res.record)
                if rec.job_id not in known_ids:
                    batch_jobs.append(rec)

        batch_jobs.sort(key=lambda j: j.created_at)
        total = db_batch["total_items"] if db_batch else len(batch_jobs)
        snapshots = [j.snapshot() for j in batch_jobs]

        completed = sum(1 for s in snapshots if s["state"] == "done")
        failed = sum(1 for s in snapshots if s["state"] in {"failed", "interrupted"})
        cancelled = sum(1 for s in snapshots if s["state"] == "cancelled")
        running = sum(1 for s in snapshots if s["state"] == "running")
        queued = sum(1 for s in snapshots if s["state"] == "queued")

        if total > 0:
            sum_progress = sum(float(s.get("progress", 0.0)) for s in snapshots)
            overall_progress = round(min(1.0, max(0.0, sum_progress / total)), 4)
        else:
            overall_progress = 0.0

        is_paused = batch_id in self._paused_batches or (
            db_batch is not None and db_batch.get("state") == "paused"
        )

        if is_paused:
            batch_state = "paused"
        elif running > 0:
            batch_state = "running"
        elif queued > 0:
            batch_state = "queued"
        elif completed + failed + cancelled >= total and total > 0:
            if cancelled == total:
                batch_state = "cancelled"
            elif failed > 0 and completed == 0:
                batch_state = "failed"
            else:
                batch_state = "done"
        else:
            batch_state = db_batch.get("state", "queued") if db_batch else "queued"

        created_at = (
            db_batch["created_at"]
            if db_batch
            else (snapshots[0]["created_at"] if snapshots else _now_iso())
        )
        updated_at = db_batch["updated_at"] if db_batch else _now_iso()

        return {
            "batch_id": batch_id,
            "state": batch_state,
            "total_items": total,
            "completed_items": completed,
            "failed_items": failed,
            "cancelled_items": cancelled,
            "running_items": running,
            "queued_items": queued,
            "progress": overall_progress,
            "created_at": created_at,
            "updated_at": updated_at,
            "jobs": snapshots,
        }

    def list_batches(self, limit: int = 50) -> list[dict[str, Any]]:
        """List active and historical batches."""
        if self._store is not None:
            return self._store.list_batches(limit=limit)
        with self._lock:
            seen_batches = {r.batch_id for r in self._jobs.values() if r.batch_id is not None}
        results = []
        for b_id in seen_batches:
            summary = self.get_batch_summary(b_id)
            if summary is not None:
                results.append(summary)
        results.sort(key=lambda b: str(b["updated_at"]), reverse=True)
        return results[:limit]

    def shutdown(self, grace_s: float = 0.5) -> None:
        """Cancel everything: queued jobs are marked cancelled, the running
        child gets SIGTERM then SIGKILL after ``grace_s``. Idempotent."""
        with self._locked():
            if self._closed:
                return
            self._closed = True
            running: ProcessSupervisor | None = None
            for job_id in list(self._queue):
                record = self._jobs[job_id]
                record.cancel_requested = True
                self._finish_locked(record, "cancelled", "Cancelled: server shutting down")
            self._queue.clear()
            for record in self._jobs.values():
                if record.state == "running":
                    record.cancel_requested = True
                    running = record.supervisor
            self._wake.notify_all()
        if running is not None:
            running.terminate_tree(grace_s)
        self._worker.join(timeout=grace_s + 1.0)
        try:
            if self._store is not None:
                self._store.close()
        finally:
            if self._owner_lease is not None:
                self._owner_lease.release()
                self._owner_lease = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown(grace_s=0.1)

    # ---------------------------------------------------------------- internal

    def _terminate(self, supervisor: ProcessSupervisor) -> None:
        """End the complete job process tree without blocking the API caller."""

        def _reap() -> None:
            supervisor.terminate_tree(self._kill_grace_s)

        threading.Thread(target=_reap, name="hawavoclean-job-kill", daemon=True).start()

    def _notify_locked(self, record: JobRecord) -> bool:
        """Record a change (``seq``) and push the snapshot to every subscriber."""
        record.seq += 1
        persisted = self._persist_locked(record)
        subs = self._subscribers.get(record.job_id)
        if not subs:
            return persisted
        snap = record.snapshot()
        alive: list[_Subscriber] = []

        def _put_latest(queue: asyncio.Queue[dict[str, Any]], value: dict[str, Any]) -> None:
            while not queue.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(value)

        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(_put_latest, queue, snap)
            except RuntimeError:
                continue  # loop closed: subscriber is gone
            alive.append((loop, queue))
        self._subscribers[record.job_id] = alive
        return persisted

    def _persist_locked(self, record: JobRecord) -> bool:
        """Persist while the caller owns the state lock; surface failures in health."""

        if self._store is None:
            return True
        try:
            self._store.update(record.storage_record(), terminal=record.state in TERMINAL_STATES)
        except JobStoreError as exc:
            self._persistence_error = str(exc)
            logger.error(f"Durable job state update failed for {record.job_id}: {exc}")
            return False
        return True

    def _finish_locked(self, record: JobRecord, state: JobState, message: str) -> None:
        record.state = state
        record.finished_at = _now_iso()
        record.terminal_at = self._clock()
        record.message = message
        if state == "done":
            record.stage = "done"
            record.progress = 1.0
        elif state == "failed":
            record.stage = "error"
        persisted = self._notify_locked(record)
        if state == "done" and not persisted:
            # A verified export that cannot make its terminal transition
            # durable must never be reported as completed. The files remain
            # recoverable, but this job verdict is explicitly failed.
            record.state = "failed"
            record.stage = "error"
            record.message = "Verified artifacts were not recorded durably"
            record.error = {
                "code": "DURABILITY_FAILURE",
                "message": self._persistence_error or "durable job update failed",
            }
            record.bundle = None
            self._notify_locked(record)
        if record.source_snapshot_dir is not None:
            remove_source_snapshot_tree(record.source_snapshot_dir)
            record.source_snapshot_dir = None
        if record.batch_id is not None and self._store is not None:
            with contextlib.suppress(Exception):
                summary = self.get_batch_summary(record.batch_id)
                if (
                    summary is not None
                    and summary["running_items"] == 0
                    and summary["queued_items"] == 0
                ):
                    self._store.update_batch_state(record.batch_id, summary["state"])
        # Queue the callbacks; ``_locked`` runs them once the lock is released.
        self._pending_terminal.append(record)

    def _prune_locked(self) -> None:
        """Expire terminal snapshots by age/count; never evict active jobs or subscribers."""
        now = self._clock()
        candidates = [
            record
            for record in self._jobs.values()
            if record.state in TERMINAL_STATES and record.terminal_at is not None
        ]
        expired = {
            record.job_id
            for record in candidates
            if now - cast(float, record.terminal_at) >= self._terminal_ttl_s
        }
        retained = [record for record in candidates if record.job_id not in expired]
        retained.sort(key=lambda record: (record.terminal_at or 0.0, record.job_id))
        overflow = max(0, len(retained) - self._max_terminal_jobs)
        expired.update(record.job_id for record in retained[:overflow])
        if self._store is not None and expired:
            try:
                self._store.prune_terminal(expired)
            except JobStoreError as exc:
                self._persistence_error = str(exc)
                logger.error(f"Durable job history prune failed: {exc}")
        for job_id in expired:
            removed = self._jobs.pop(job_id, None)
            self._subscribers.pop(job_id, None)
            if removed is not None and removed.idempotency_key is not None:
                self._idempotency.pop(removed.idempotency_key, None)

    def _run_loop(self) -> None:
        while True:
            with self._wake:
                next_job_id: str | None = None
                while not self._closed:
                    for candidate_id in self._queue:
                        rec = self._jobs.get(candidate_id)
                        if (
                            rec is not None
                            and rec.batch_id is not None
                            and rec.batch_id in self._paused_batches
                        ):
                            continue
                        next_job_id = candidate_id
                        break
                    if next_job_id is not None:
                        self._queue.remove(next_job_id)
                        break
                    self._wake.wait()
                if self._closed or next_job_id is None:
                    return
                record = self._jobs[next_job_id]
                record.state = "running"
                record.started_at = _now_iso()
                record.message = "Starting"
                self._notify_locked(record)
            try:
                self._run_job(record)
            except Exception as e:  # a bug in the runner must not kill the worker thread
                logger.error(f"Job {record.job_id} runner crashed: {e}", exc_info=True)
                with self._locked():
                    if record.state not in TERMINAL_STATES:
                        record.error = {"code": "INTERNAL", "message": str(e)}
                        self._finish_locked(record, "failed", str(e))

    def _child_env(self) -> dict[str, str]:
        """The child's environment, always carrying this engine's pid.

        ``shutdown()`` only runs when the engine exits *gracefully*. An engine
        that is SIGKILLed (crash, OOM killer, ``kill -9``) runs no cleanup at
        all, and the job child — a full ``hawavoclean process`` run — used to
        be reparented to init and go on to write a complete master and report
        for a run the UI had already reconciled to "failed, nothing was
        written". So the child watches the engine itself:
        ``hawavoclean.watchdog``.
        """
        return child_env(self._env)

    def _run_job(self, record: JobRecord) -> None:
        cmd = self._command_factory(record)
        try:
            supervisor = ProcessSupervisor.spawn(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=self._child_env(),
            )
            proc = supervisor.process
        except OSError as e:
            with self._locked():
                record.error = {"code": "SPAWN_FAILED", "message": str(e)}
                self._finish_locked(record, "failed", f"Could not start worker: {e}")
            return

        with self._lock:
            record.process = proc
            record.supervisor = supervisor
            cancel_now = record.cancel_requested
        if cancel_now:
            self._terminate(supervisor)

        assert proc.stderr is not None and proc.stdout is not None
        stderr_stream = proc.stderr

        def _drain_stderr() -> None:
            for line in stderr_stream:
                stripped = line.rstrip("\n")
                if stripped.strip():
                    record.stderr_tail.append(stripped)

        stderr_thread = threading.Thread(
            target=_drain_stderr, name="hawavoclean-job-stderr", daemon=True
        )
        stderr_thread.start()

        done_event: dict[str, Any] | None = None
        error_event: dict[str, Any] | None = None
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    logger.warning(f"Job {record.job_id}: non-JSON line on stdout: {line[:200]}")
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("event")
                if kind == "progress":
                    with self._lock:
                        record.stage = str(event.get("stage", record.stage))
                        record.progress = float(event.get("progress", record.progress))
                        record.message = str(event.get("message", record.message))
                        unit = event.get("unit")
                        if isinstance(unit, dict) and "index" in unit and "total" in unit:
                            record.unit = {
                                "index": int(unit["index"]),
                                "total": int(unit["total"]),
                            }
                        else:
                            record.unit = None
                        self._notify_locked(record)
                elif kind == "done":
                    done_event = event
                elif kind == "error":
                    error_event = event

            returncode = proc.wait()
        finally:
            # The leader's exit is not proof that every decoder/model child
            # exited.  Releasing the boundary is therefore fail-closed.
            supervisor.close(kill_remaining=True)
        stderr_thread.join(timeout=2.0)

        with self._locked():
            if record.state in TERMINAL_STATES:  # pragma: no cover - defensive
                return
            if done_event is not None and returncode == 0:
                report_path = Path(str(done_event.get("report_path") or record.report_path))
                try:
                    parsed_report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    record.error = {
                        "code": ExitCode.PUBLICATION_FAILURE.name,
                        "message": f"Published, but the report is unreadable: {e}",
                    }
                    self._finish_locked(record, "failed", record.error["message"])
                    return
                if not isinstance(parsed_report, dict):
                    record.error = {
                        "code": ExitCode.PUBLICATION_FAILURE.name,
                        "message": "Published report is not a JSON object",
                    }
                    self._finish_locked(record, "failed", record.error["message"])
                    return
                record.report = cast(dict[str, Any], parsed_report)
                if record.record_bundle:
                    try:
                        record.bundle, record.report = self._validate_bundle_artifacts(record)
                    except Exception as e:
                        record.error = {
                            "code": ExitCode.PUBLICATION_FAILURE.name,
                            "message": f"Processing Record verification failed: {e}",
                        }
                        self._finish_locked(record, "failed", record.error["message"])
                        return
                else:
                    try:
                        # Durable jobs may cross a restart, so a report alone
                        # is not sufficient evidence of which generation they
                        # completed. In-memory compatibility fakes without an
                        # output digest remain usable, but production-shaped
                        # reports are bound and validated even without SQLite.
                        self._report_audio_sha256(record.report)
                    except JobStoreError as e:
                        if self._store is not None:
                            record.error = {
                                "code": ExitCode.PUBLICATION_FAILURE.name,
                                "message": f"Completed artifact evidence is invalid: {e}",
                            }
                            self._finish_locked(record, "failed", record.error["message"])
                            return
                    else:
                        try:
                            self._capture_nonbundle_artifacts(record)
                        except Exception as e:
                            record.error = {
                                "code": ExitCode.PUBLICATION_FAILURE.name,
                                "message": f"Completed artifact verification failed: {e}",
                            }
                            self._finish_locked(record, "failed", record.error["message"])
                            return
                record.unit = None
                self._finish_locked(record, "done", "Done")
                return
            if record.cancel_requested:
                self._finish_locked(record, "cancelled", "Cancelled")
                return
            code, message = self._failure_details(record, returncode, error_event)
            record.error = {"code": code, "message": message}
            self._finish_locked(record, "failed", message)

    @staticmethod
    def _failure_details(
        record: JobRecord, returncode: int, error_event: dict[str, Any] | None
    ) -> tuple[str, str]:
        if error_event is not None:
            code = str(error_event.get("code") or "ERROR")
            message = str(error_event.get("message") or "Processing failed")
            return code, message
        try:
            code = ExitCode(returncode).name
        except ValueError:
            code = f"EXIT_{returncode}" if returncode >= 0 else f"SIGNAL_{-returncode}"
        if returncode == 0:
            code = "PROTOCOL_ERROR"
            return code, "Worker exited without reporting a result"
        try:
            tail = [ln for ln in record.stderr_tail if ln.strip()]
        except RuntimeError:  # pragma: no cover - drain thread still appending (an
            tail = []  # orphaned grandchild that inherited stderr can outlive the child)
        message = tail[-1][-300:] if tail else f"Worker exited with code {returncode}"
        return code, message
