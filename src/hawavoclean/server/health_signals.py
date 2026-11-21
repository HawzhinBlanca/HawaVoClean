"""Production health signals, observability, and actionable alert engine."""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal

SignalType = Literal[
    "queue_depth",
    "no_progress_timeout",
    "process_lease_loss",
    "disk_pressure",
]
Severity = Literal["ok", "warning", "critical"]

RUNBOOK_BASE_URL: Final[str] = "docs/operations.md"
RUNBOOK_ANCHORS: Final[dict[SignalType, str]] = {
    "queue_depth": f"{RUNBOOK_BASE_URL}#queue-depth-runbook",
    "no_progress_timeout": f"{RUNBOOK_BASE_URL}#job-timeout-runbook",
    "process_lease_loss": f"{RUNBOOK_BASE_URL}#process-lease-loss-runbook",
    "disk_pressure": f"{RUNBOOK_BASE_URL}#disk-pressure-runbook",
}

DEFAULT_NO_PROGRESS_TIMEOUT_S: Final[float] = 300.0
DEFAULT_QUEUE_WARNING_THRESHOLD: Final[int] = 100

_FORBIDDEN_PATTERNS = (
    re.compile(r"/[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+"),
    re.compile(r"[a-zA-Z]:\\[a-zA-Z0-9_\-\.]+\\"),
    re.compile(r"\.(wav|mp3|m4a|flac|ogg|aac|aiff)\b", re.IGNORECASE),
    re.compile(r"(bearer|token|secret|key|authorization)", re.IGNORECASE),
)

_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "transcript",
        "transcripts",
        "embedding",
        "embeddings",
        "audio",
        "waveform",
        "waveforms",
        "token",
        "tokens",
        "secret",
        "secrets",
        "key",
        "keys",
        "password",
        "passwords",
        "authorization",
        "auth",
        "path",
        "paths",
        "file_path",
        "filepath",
        "filename",
        "input_path",
        "output_path",
        "report_path",
        "source_path",
    }
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sanitize_telemetry(data: Any) -> Any:
    """Recursively strip or redact paths, audio filenames, transcripts, or credentials."""
    if isinstance(data, dict):
        return {
            str(k): sanitize_telemetry(v)
            for k, v in data.items()
            if str(k).lower() not in _FORBIDDEN_KEYS
            and not any(p.search(str(k).lower()) for p in _FORBIDDEN_PATTERNS)
        }
    if isinstance(data, (list, tuple, set)):
        return [sanitize_telemetry(item) for item in data]
    if isinstance(data, str) and any(p.search(data) for p in _FORBIDDEN_PATTERNS):
        return "[REDACTED]"
    return data


@dataclass(frozen=True, slots=True)
class HealthAlert:
    """Actionable alert structure."""

    alert_id: str
    signal: SignalType
    severity: Severity
    summary: str
    runbook_url: str
    triggered_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "signal": self.signal,
            "severity": self.severity,
            "summary": self.summary,
            "runbook_url": self.runbook_url,
            "triggered_at": self.triggered_at,
            "metadata": sanitize_telemetry(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SignalStatus:
    """Evaluation result for a single health signal."""

    signal: SignalType
    status: Severity
    message: str
    metrics: dict[str, Any]
    alert: HealthAlert | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "status": self.status,
            "message": self.message,
            "metrics": sanitize_telemetry(self.metrics),
            "alert": self.alert.to_dict() if self.alert is not None else None,
        }


class HealthSignalsService:
    """Evaluates live local-engine operational health signals."""

    def __init__(
        self,
        *,
        job_manager: Any | None = None,
        upload_store: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        now_iso: Callable[[], str] = _now_iso,
        no_progress_timeout_s: float = DEFAULT_NO_PROGRESS_TIMEOUT_S,
        queue_warning_threshold: int = DEFAULT_QUEUE_WARNING_THRESHOLD,
    ) -> None:
        self._manager = job_manager
        self._upload_store = upload_store
        self._clock = clock
        self._now_iso = now_iso
        self._no_progress_timeout_s = no_progress_timeout_s
        self._queue_warning_threshold = queue_warning_threshold

    def _build_status(
        self,
        signal: SignalType,
        status: Severity,
        message: str,
        metrics: dict[str, Any],
        *,
        alert_id: str | None = None,
        summary: str | None = None,
    ) -> SignalStatus:
        alert = (
            HealthAlert(
                alert_id=alert_id,
                signal=signal,
                severity=status,
                summary=summary or message,
                runbook_url=RUNBOOK_ANCHORS[signal],
                triggered_at=self._now_iso(),
                metadata=metrics,
            )
            if status != "ok" and alert_id is not None
            else None
        )
        return SignalStatus(
            signal=signal, status=status, message=message, metrics=metrics, alert=alert
        )

    def evaluate_queue_depth(self) -> SignalStatus:
        depth = 0
        max_active = 128
        if self._manager is not None:
            max_active = getattr(self._manager, "_max_active_jobs", 128)
            depth = self._manager.get_queue_depth()

        metrics = {
            "current_depth": depth,
            "max_active_jobs": max_active,
            "warning_threshold": self._queue_warning_threshold,
        }
        if depth >= max_active:
            return self._build_status(
                "queue_depth",
                "critical",
                f"Queue capacity saturated ({depth}/{max_active})",
                metrics,
                alert_id="QUEUE_DEPTH_EXCEEDED",
                summary=f"Job queue depth exceeded maximum capacity ({depth}/{max_active})",
            )
        if depth >= self._queue_warning_threshold:
            return self._build_status(
                "queue_depth",
                "warning",
                f"Queue depth is elevated ({depth}/{max_active})",
                metrics,
                alert_id="QUEUE_DEPTH_WARNING",
                summary=f"Job queue depth is near capacity ({depth}/{max_active})",
            )
        return self._build_status(
            "queue_depth",
            "ok",
            f"Queue depth within normal parameters ({depth}/{max_active})",
            metrics,
        )

    def evaluate_no_progress_timeout(self) -> SignalStatus:
        stalled_jobs: list[dict[str, Any]] = (
            self._manager.check_no_progress_timeouts(self._no_progress_timeout_s)
            if self._manager is not None
            else []
        )

        metrics = {
            "stalled_jobs_count": len(stalled_jobs),
            "timeout_seconds": self._no_progress_timeout_s,
            "stalled_jobs": stalled_jobs,
        }
        if stalled_jobs:
            elapsed_s = stalled_jobs[0].get("elapsed_seconds", self._no_progress_timeout_s)
            return self._build_status(
                "no_progress_timeout",
                "critical",
                f"{len(stalled_jobs)} running job(s) stalled without progress updates",
                metrics,
                alert_id="JOB_NO_PROGRESS_STALLED",
                summary=(
                    f"Running job has made no progress for {elapsed_s}s "
                    f"(timeout: {self._no_progress_timeout_s}s)"
                ),
            )
        return self._build_status("no_progress_timeout", "ok", "No stalled jobs detected", metrics)

    def evaluate_process_lease_loss(self) -> SignalStatus:
        lease_ok = True
        error_details: list[str] = []
        if self._manager is not None:
            healthy, reason = self._manager.check_process_and_lease_health()
            if not healthy:
                lease_ok = False
                if reason:
                    error_details.append(reason)

        metrics = {
            "lease_held": lease_ok,
            "persistence_ok": lease_ok,
            "active_processes_healthy": True,
            "error_details": error_details,
        }
        if not lease_ok:
            return self._build_status(
                "process_lease_loss",
                "critical",
                "Critical process or lease loss detected",
                metrics,
                alert_id="PROCESS_OR_LEASE_LOST",
                summary="Worker process or durable store lease was lost",
            )
        return self._build_status(
            "process_lease_loss",
            "ok",
            "Process and durable store lease are healthy",
            metrics,
        )

    def evaluate_disk_pressure(self) -> SignalStatus:
        min_free = 512 * 1024 * 1024
        free_bytes = 100 * 1024 * 1024 * 1024
        if self._upload_store is not None:
            min_free = getattr(self._upload_store, "min_free_bytes", min_free)
            root_path = getattr(self._upload_store, "root", None)
            if root_path is not None:
                disk_usage_fn = getattr(self._upload_store, "_disk_usage", None)
                free_bytes = (
                    disk_usage_fn(root_path).free
                    if disk_usage_fn is not None
                    else shutil.disk_usage(root_path).free
                )

        warning_threshold = 2 * min_free
        metrics = {
            "free_bytes": free_bytes,
            "min_free_bytes": min_free,
            "warning_threshold_bytes": warning_threshold,
        }
        free_mb = round(free_bytes / (1024 * 1024), 1)
        min_mb = round(min_free / (1024 * 1024), 1)
        if free_bytes < min_free:
            return self._build_status(
                "disk_pressure",
                "critical",
                f"Critical storage pressure ({free_mb} MB free)",
                metrics,
                alert_id="DISK_PRESSURE_CRITICAL",
                summary=f"Free disk space is below emergency reserve ({free_mb} MB < {min_mb} MB)",
            )
        if free_bytes < warning_threshold:
            return self._build_status(
                "disk_pressure",
                "warning",
                f"Elevated storage pressure ({free_mb} MB free)",
                metrics,
                alert_id="DISK_PRESSURE_WARNING",
                summary=f"Free disk space is approaching emergency reserve ({free_mb} MB free)",
            )
        return self._build_status(
            "disk_pressure",
            "ok",
            f"Sufficient free storage available ({free_mb} MB free)",
            metrics,
        )

    def evaluate_all(self) -> dict[str, Any]:
        """Evaluate all operational health signals and return structured telemetry."""
        signals_map: dict[SignalType, SignalStatus] = {
            "queue_depth": self.evaluate_queue_depth(),
            "no_progress_timeout": self.evaluate_no_progress_timeout(),
            "process_lease_loss": self.evaluate_process_lease_loss(),
            "disk_pressure": self.evaluate_disk_pressure(),
        }
        alerts = [s.alert.to_dict() for s in signals_map.values() if s.alert is not None]
        statuses = {s.status for s in signals_map.values()}
        worst_status: Severity = (
            "critical" if "critical" in statuses else ("warning" if "warning" in statuses else "ok")
        )
        return {
            "schema_version": 1,
            "status": worst_status,
            "healthy": worst_status != "critical",
            "evaluated_at": self._now_iso(),
            "signals": {k: v.to_dict() for k, v in signals_map.items()},
            "alerts": alerts,
        }
