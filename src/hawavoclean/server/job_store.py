"""Crash-safe local job ledger and output-name reservation.

The HTTP job manager deliberately runs each render in a child process, but its
state used to live only in memory.  A server restart therefore forgot history,
idempotency, and every output name another client had already reserved.  This
module is the small durable boundary underneath the in-process scheduler:

* SQLite WAL + ``synchronous=FULL`` makes accepted submissions durable;
* a partial unique index is the cross-process output lease;
* idempotency keys are bound to a canonical request hash;
* startup converts work that could not have survived the process into an
  explicit ``interrupted`` terminal record.

It stores JSON rather than importing :class:`JobRecord`, which keeps the
storage layer independent of the server's mutable runtime types.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ConflictPolicy = Literal["unique", "fail", "replace"]
NONTERMINAL_STATES: frozenset[str] = frozenset({"queued", "running"})
SCHEMA_VERSION = 2
_PRUNE_PAGE_SIZE = 128


class JobStoreError(RuntimeError):
    """The durable ledger could not uphold its contract."""


class IdempotencyConflictError(JobStoreError):
    """One idempotency key was reused for a different request."""


class OutputConflictError(JobStoreError):
    """An output is already present or reserved by active work."""


@dataclass(frozen=True)
class Reservation:
    """Result of a transactional submission reservation."""

    record: dict[str, Any]
    reused: bool
    terminal_at_epoch: float | None = None
    history_visible: bool = True


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_request_hash(payload: dict[str, Any]) -> str:
    """Stable identity for the request an idempotency key represents."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_key(path: Path) -> str:
    """Conservative key for an output lease on the current platform.

    Windows and normal macOS installations are case-insensitive.  Treating
    differently-cased names as one lease on those platforms prevents a
    correctness bug on the common filesystem even if a particular developer
    happens to use a case-sensitive volume.
    """

    absolute = os.path.abspath(os.fspath(path))
    normalized = unicodedata.normalize("NFC", absolute)
    if sys.platform in {"darwin", "win32"}:
        normalized = normalized.casefold()
    return normalized


def unique_candidate(path: Path, ordinal: int) -> Path:
    """Return the deterministic user-visible name for ``ordinal`` (1-based)."""

    if ordinal < 1:
        raise ValueError("ordinal must be positive")
    if ordinal == 1:
        return path
    return path.with_name(f"{path.stem} ({ordinal}){path.suffix}")


def user_artifact_paths(path: Path, *, record_bundle: bool) -> tuple[Path, ...]:
    """Every user-visible path owned by one output-name reservation.

    Reserving only the WAV allowed a pre-existing report, summary, or record
    ZIP to be silently replaced by a newly accepted job.  These names are a
    single logical export, so conflict policy applies to the complete set.
    """

    # The ZIP is reserved even when the new request does not ask to create
    # one. Otherwise a replace-mode Natural run could leave a same-stem ZIP
    # describing the previous audio beside the new master, which is worse
    # than a normal filename collision: it is misleading provenance.
    del record_bundle  # retained in the public signature for one-release compatibility
    return (
        path,
        path.parent / f"{path.stem}.hawavoclean.json",
        path.parent / f"{path.stem}.hawavoclean.txt",
        path.parent / f".{path.name}.hawavoclean",
        path.parent / f"{path.stem}.hawavoclean.zip",
    )


def processing_record_path(path: Path) -> Path:
    """Return the same-stem portable Processing Record path."""

    return path.parent / f"{path.stem}.hawavoclean.zip"


def _path_is_occupied(path: Path) -> bool:
    """``Path.exists`` is false for dangling symlinks; those are occupied too."""

    return os.path.lexists(path)


class DurableJobStore:
    """Thread-safe SQLite job ledger.

    Separate processes may open the same database. ``BEGIN IMMEDIATE`` plus
    the unique indexes serializes the only two decisions that must never race:
    idempotency ownership and output-name ownership.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            self._conn = conn
            self._configure()
            self._migrate()
        except (OSError, sqlite3.Error) as exc:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
            raise JobStoreError(f"could not open durable job store {path}: {exc}") from exc
        except BaseException:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
            raise

    def _configure(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 30000")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA synchronous = FULL")
            mode = str(self._conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if mode != "wal":
                raise JobStoreError(f"job store refused WAL mode (got {mode!r})")

    def _migrate(self) -> None:
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise JobStoreError(
                    f"job store schema {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version == 0:
                self._conn.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE TABLE jobs (
                        job_id TEXT PRIMARY KEY,
                        idempotency_key TEXT UNIQUE,
                        request_hash TEXT NOT NULL,
                        output_key TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        state TEXT NOT NULL,
                        terminal INTEGER NOT NULL CHECK (terminal IN (0, 1)),
                        record_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        terminal_at REAL,
                        history_visible INTEGER NOT NULL DEFAULT 1
                            CHECK (history_visible IN (0, 1))
                    );
                    CREATE UNIQUE INDEX active_output_lease
                        ON jobs(output_key) WHERE terminal = 0;
                    CREATE INDEX jobs_updated_at ON jobs(updated_at DESC);
                    CREATE INDEX visible_terminal_retention
                        ON jobs(history_visible, terminal, terminal_at DESC, created_at DESC);
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
                version = 2
            if version == 1:
                # Version 1 stored only ``updated_at``.  It is the best
                # available approximation for already-terminal rows and is
                # deliberately migrated once; later restarts never refresh it.
                self._conn.executescript(
                    """
                    BEGIN IMMEDIATE;
                    ALTER TABLE jobs ADD COLUMN terminal_at REAL;
                    ALTER TABLE jobs ADD COLUMN history_visible INTEGER NOT NULL DEFAULT 1
                        CHECK (history_visible IN (0, 1));
                    UPDATE jobs
                       SET terminal_at = COALESCE(
                           CAST(strftime('%s', updated_at) AS REAL),
                           CAST(strftime('%s', 'now') AS REAL)
                       )
                     WHERE terminal = 1;
                    CREATE INDEX visible_terminal_retention
                        ON jobs(history_visible, terminal, terminal_at DESC, created_at DESC);
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )

    def _assert_open(self) -> None:
        if self._closed:
            raise JobStoreError("durable job store is closed")

    @staticmethod
    def _encode(record: dict[str, Any]) -> str:
        try:
            return json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise JobStoreError(f"job record is not canonical JSON: {exc}") from exc

    @staticmethod
    def _decode(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise JobStoreError(f"job store contains invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise JobStoreError("job store record is not an object")
        return value

    def reserve(
        self,
        *,
        record: dict[str, Any],
        request_hash: str,
        idempotency_key: str | None,
        conflict_policy: ConflictPolicy,
    ) -> Reservation:
        """Atomically reserve idempotency and an output name.

        The returned record may contain a suffixed output/report path when the
        policy is ``unique``.  Reusing a key for the same request returns the
        original record without enqueueing new work.
        """

        if conflict_policy not in {"unique", "fail", "replace"}:
            raise ValueError(f"unsupported conflict policy: {conflict_policy}")
        desired = Path(str(record["output_path"]))
        record_bundle = bool(record.get("record_bundle", False))
        now = _now_iso()
        with self._lock:
            self._assert_open()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if idempotency_key is not None:
                    prior = self._conn.execute(
                        """
                        SELECT request_hash, record_json, terminal_at, history_visible
                          FROM jobs WHERE idempotency_key = ?
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if prior is not None:
                        if str(prior["request_hash"]) != request_hash:
                            raise IdempotencyConflictError(
                                "idempotency key is already bound to a different request"
                            )
                        value = self._decode(str(prior["record_json"]))
                        self._conn.execute("COMMIT")
                        return Reservation(
                            record=value,
                            reused=True,
                            terminal_at_epoch=(
                                float(prior["terminal_at"])
                                if prior["terminal_at"] is not None
                                else None
                            ),
                            history_visible=bool(prior["history_visible"]),
                        )

                candidate = desired
                ordinal = 1
                while True:
                    key = output_key(candidate)
                    active = self._conn.execute(
                        "SELECT job_id FROM jobs WHERE output_key = ? AND terminal = 0",
                        (key,),
                    ).fetchone()
                    occupied = active is not None or any(
                        _path_is_occupied(path)
                        for path in user_artifact_paths(candidate, record_bundle=record_bundle)
                    )
                    stale_processing_record = not record_bundle and _path_is_occupied(
                        processing_record_path(candidate)
                    )
                    if conflict_policy == "unique" and occupied:
                        ordinal += 1
                        candidate = unique_candidate(desired, ordinal)
                        if ordinal > 100_000:
                            raise OutputConflictError("could not allocate a unique output name")
                        continue
                    if active is not None:
                        raise OutputConflictError(
                            f"output is reserved by active job {active['job_id']}: {candidate}"
                        )
                    if conflict_policy == "replace" and stale_processing_record:
                        raise OutputConflictError(
                            "same-stem Processing Record already exists; use unique naming "
                            "or request a replacement Processing Record"
                        )
                    if conflict_policy == "fail" and occupied:
                        raise OutputConflictError(
                            f"output or sidecar already exists for: {candidate}"
                        )
                    break

                stored = dict(record)
                stored["output_path"] = str(candidate)
                stored["report_path"] = str(candidate.parent / f"{candidate.stem}.hawavoclean.json")
                stored["bundle_path"] = (
                    str(candidate.parent / f"{candidate.stem}.hawavoclean.zip")
                    if record_bundle
                    else None
                )
                encoded = self._encode(stored)
                self._conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, idempotency_key, request_hash, output_key,
                        output_path, state, terminal, record_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        str(stored["job_id"]),
                        idempotency_key,
                        request_hash,
                        output_key(candidate),
                        str(candidate),
                        str(stored["state"]),
                        encoded,
                        str(stored.get("created_at") or now),
                        now,
                    ),
                )
                self._conn.execute("COMMIT")
                return Reservation(record=stored, reused=False)
            except (IdempotencyConflictError, OutputConflictError):
                self._conn.execute("ROLLBACK")
                raise
            except (sqlite3.Error, JobStoreError, OSError) as exc:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                if isinstance(exc, JobStoreError):
                    raise
                raise JobStoreError(f"could not reserve durable job: {exc}") from exc

    def update(self, record: dict[str, Any], *, terminal: bool) -> None:
        """Durably replace a snapshot while preserving its first terminal time."""

        encoded = self._encode(record)
        with self._lock:
            self._assert_open()
            try:
                cursor = self._conn.execute(
                    """
                    UPDATE jobs
                       SET state = ?, terminal = ?, record_json = ?, updated_at = ?,
                           terminal_at = CASE
                               WHEN ? THEN COALESCE(terminal_at, ?)
                               ELSE NULL
                           END
                     WHERE job_id = ?
                    """,
                    (
                        str(record["state"]),
                        int(terminal),
                        encoded,
                        _now_iso(),
                        int(terminal),
                        time.time(),
                        str(record["job_id"]),
                    ),
                )
            except sqlite3.Error as exc:
                raise JobStoreError(f"could not update durable job: {exc}") from exc
            if cursor.rowcount != 1:
                raise JobStoreError(f"durable job is missing: {record.get('job_id')}")

    def delete_queued(self, job_ids: list[str]) -> None:
        """Roll back broker-prepared rows that were never eligible to execute."""

        if not job_ids:
            return
        placeholders = ",".join("?" for _ in job_ids)
        with self._lock:
            self._assert_open()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                rows = self._conn.execute(
                    f"SELECT job_id, state, terminal FROM jobs WHERE job_id IN ({placeholders})",
                    tuple(job_ids),
                ).fetchall()
                if len(rows) != len(set(job_ids)) or any(
                    str(row["state"]) != "queued" or bool(row["terminal"]) for row in rows
                ):
                    raise JobStoreError("prepared batch contains a job that is no longer queued")
                self._conn.execute(
                    f"DELETE FROM jobs WHERE job_id IN ({placeholders})", tuple(job_ids)
                )
                self._conn.execute("COMMIT")
            except (sqlite3.Error, JobStoreError) as exc:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                if isinstance(exc, JobStoreError):
                    raise
                raise JobStoreError(f"could not roll back prepared batch: {exc}") from exc

    @staticmethod
    def _compact_idempotency_receipt(record: dict[str, Any]) -> dict[str, Any]:
        """Drop large report bodies while preserving identity and evidence."""

        compact = dict(record)
        if compact.get("state") == "done":
            compact["report"] = None
        return compact

    def _prune_rows_in_transaction(self, rows: list[sqlite3.Row]) -> None:
        """Delete anonymous history and compact keyed rows into receipts."""

        for row in rows:
            job_id = str(row["job_id"])
            if row["idempotency_key"] is None:
                self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                continue
            compact = self._compact_idempotency_receipt(self._decode(str(row["record_json"])))
            self._conn.execute(
                """
                UPDATE jobs
                   SET history_visible = 0, record_json = ?
                 WHERE job_id = ? AND terminal = 1
                """,
                (self._encode(compact), job_id),
            )

    @staticmethod
    def _attach_row_metadata(record: dict[str, Any], row: sqlite3.Row) -> dict[str, Any]:
        value = dict(record)
        value["_terminal_at_epoch"] = (
            float(row["terminal_at"]) if row["terminal_at"] is not None else None
        )
        value["_history_visible"] = bool(row["history_visible"])
        return value

    def find_idempotent(
        self, idempotency_key: str, *, request_hash: str | None = None
    ) -> Reservation | None:
        """Load one exact retry on demand, including a compact pruned receipt."""

        with self._lock:
            self._assert_open()
            try:
                row = self._conn.execute(
                    """
                    SELECT request_hash, record_json, terminal_at, history_visible
                      FROM jobs WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise JobStoreError(f"could not look up durable idempotency key: {exc}") from exc
            if row is None:
                return None
            if request_hash is not None and str(row["request_hash"]) != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key is already bound to a different request"
                )
            return Reservation(
                record=self._decode(str(row["record_json"])),
                reused=True,
                terminal_at_epoch=(
                    float(row["terminal_at"]) if row["terminal_at"] is not None else None
                ),
                history_visible=bool(row["history_visible"]),
            )

    def find_job(self, job_id: str) -> Reservation | None:
        """Load one durable job resource without restoring it to list history."""

        with self._lock:
            self._assert_open()
            try:
                row = self._conn.execute(
                    """
                    SELECT record_json, terminal_at, history_visible
                      FROM jobs WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise JobStoreError(f"could not look up durable job resource: {exc}") from exc
            if row is None:
                return None
            return Reservation(
                record=self._decode(str(row["record_json"])),
                reused=True,
                terminal_at_epoch=(
                    float(row["terminal_at"]) if row["terminal_at"] is not None else None
                ),
                history_visible=bool(row["history_visible"]),
            )

    def prune_terminal(self, job_ids: set[str]) -> None:
        """Remove terminal rows from visible history without losing exact retries."""

        if not job_ids:
            return
        ordered = sorted(job_ids)
        with self._lock:
            self._assert_open()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for offset in range(0, len(ordered), _PRUNE_PAGE_SIZE):
                    page = ordered[offset : offset + _PRUNE_PAGE_SIZE]
                    placeholders = ",".join("?" for _ in page)
                    rows = self._conn.execute(
                        f"""
                        SELECT job_id, idempotency_key, record_json
                          FROM jobs
                         WHERE terminal = 1 AND history_visible = 1
                           AND job_id IN ({placeholders})
                        """,
                        tuple(page),
                    ).fetchall()
                    self._prune_rows_in_transaction(rows)
                self._conn.execute("COMMIT")
            except (sqlite3.Error, JobStoreError) as exc:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                if isinstance(exc, JobStoreError):
                    raise
                raise JobStoreError(f"could not prune durable job history: {exc}") from exc

    def load_and_interrupt(
        self,
        *,
        max_terminal_jobs: int = 256,
        terminal_ttl_s: float = 24 * 60 * 60.0,
        now_epoch: float | None = None,
    ) -> list[dict[str, Any]]:
        """Recover active work and load only the bounded visible history window.

        Full report JSON outside the window is processed in small pages and
        either deleted or compacted into an idempotency receipt.  No restart
        walks the application's lifetime history into one Python list.
        """

        if max_terminal_jobs < 1:
            raise ValueError("max_terminal_jobs must be at least 1")
        if terminal_ttl_s <= 0:
            raise ValueError("terminal_ttl_s must be positive")
        terminal_now = time.time() if now_epoch is None else now_epoch
        now_iso = _now_iso()
        cutoff = terminal_now - terminal_ttl_s
        loaded: list[dict[str, Any]] = []
        with self._lock:
            self._assert_open()
            try:
                self._conn.execute("BEGIN IMMEDIATE")

                # Active rows are bounded by the broker in normal operation,
                # but fetch in pages so even a hand-corrupted ledger cannot
                # force an unbounded allocation during recovery.
                cursor = self._conn.execute(
                    """
                    SELECT job_id, record_json FROM jobs
                     WHERE terminal = 0 ORDER BY created_at
                    """
                )
                while rows := cursor.fetchmany(_PRUNE_PAGE_SIZE):
                    for row in rows:
                        record = self._decode(str(row["record_json"]))
                        record["state"] = "interrupted"
                        record["stage"] = "interrupted"
                        record["message"] = "Interrupted by an engine restart; safe to retry"
                        record["finished_at"] = now_iso
                        record["error"] = {
                            "code": "INTERRUPTED",
                            "message": "The previous engine exited before this job completed",
                        }
                        record["seq"] = int(record.get("seq", 0)) + 1
                        self._conn.execute(
                            """
                            UPDATE jobs
                               SET state = 'interrupted', terminal = 1,
                                   record_json = ?, updated_at = ?, terminal_at = ?
                             WHERE job_id = ? AND terminal = 0
                            """,
                            (
                                self._encode(record),
                                now_iso,
                                terminal_now,
                                str(row["job_id"]),
                            ),
                        )

                keep_rows = self._conn.execute(
                    """
                    SELECT job_id
                      FROM jobs
                     WHERE terminal = 1 AND history_visible = 1
                       AND terminal_at IS NOT NULL AND terminal_at > ?
                     ORDER BY terminal_at DESC, created_at DESC, job_id DESC
                     LIMIT ?
                    """,
                    (cutoff, max_terminal_jobs),
                ).fetchall()
                keep_ids = [str(row["job_id"]) for row in keep_rows]

                where_not_kept = ""
                params: list[Any] = []
                if keep_ids:
                    placeholders = ",".join("?" for _ in keep_ids)
                    where_not_kept = f" AND job_id NOT IN ({placeholders})"
                    params.extend(keep_ids)
                while True:
                    rows = self._conn.execute(
                        f"""
                        SELECT job_id, idempotency_key, record_json
                          FROM jobs
                         WHERE terminal = 1 AND history_visible = 1
                         {where_not_kept}
                         LIMIT ?
                        """,
                        (*params, _PRUNE_PAGE_SIZE),
                    ).fetchall()
                    if not rows:
                        break
                    self._prune_rows_in_transaction(rows)

                if keep_ids:
                    placeholders = ",".join("?" for _ in keep_ids)
                    rows = self._conn.execute(
                        f"""
                        SELECT record_json, terminal_at, history_visible
                          FROM jobs WHERE job_id IN ({placeholders})
                         ORDER BY terminal_at, created_at, job_id
                        """,
                        tuple(keep_ids),
                    ).fetchall()
                    loaded = [
                        self._attach_row_metadata(self._decode(str(row["record_json"])), row)
                        for row in rows
                    ]
                self._conn.execute("COMMIT")
            except (sqlite3.Error, JobStoreError) as exc:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                if isinstance(exc, JobStoreError):
                    raise
                raise JobStoreError(f"could not recover durable jobs: {exc}") from exc
        return loaded

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True


__all__ = [
    "ConflictPolicy",
    "DurableJobStore",
    "IdempotencyConflictError",
    "JobStoreError",
    "OutputConflictError",
    "Reservation",
    "canonical_request_hash",
    "output_key",
    "processing_record_path",
    "user_artifact_paths",
    "unique_candidate",
]
