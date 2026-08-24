"""FastAPI application for the HawaVoClean engine bridge.

``create_app(token, ui_dir)`` builds the app; ``run_server(...)`` binds a
loopback socket, prints the single ``{"event":"ready",...}`` line to stdout
and serves with uvicorn. Routes, shapes and rules: ``docs/ui-contract.md``.
"""

import asyncio
import contextlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.datastructures import Headers, QueryParams
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from hawavoclean import __version__
from hawavoclean.cli import _clean_stem
from hawavoclean.errors import HawaVoCleanError, InvalidUserInputError
from hawavoclean.logging import get_logger
from hawavoclean.paths import profiles_root, work_root
from hawavoclean.server.analysis import (
    DEFAULT_BUCKETS,
    MAX_BUCKETS,
    PeaksWindowError,
    analyze_audio,
    compute_peaks_window,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord, QueueFullError
from hawavoclean.server.policy import (
    PathPolicyError,
    resolve_client_output_path,
    resolve_client_path,
)
from hawavoclean.server.retention import (
    DEFAULT_MAX_UPLOAD_TOTAL_BYTES,
    DEFAULT_MIN_FREE_BYTES,
    DEFAULT_UPLOAD_TTL_S,
    DiskUsageFactory,
    StoragePressureError,
    UploadStore,
)

logger = get_logger("server")

PROFILES: tuple[str, ...] = ("studio", "lowband", "production")
_OUTPUT_SUFFIX = {
    "studio": "_studio",
    "lowband": "_lowband",
    "production": "_clean",
    "development": "_dev",
}
_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".mov": "video/quicktime",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".webm": "audio/webm",
}
_STREAM_CHUNK = 256 * 1024
# ``POST /api/upload`` copy granularity. Starlette has already spooled the part
# to a temp file on disk (``SpooledTemporaryFile(max_size=1 MiB)``: anything
# past 1 MiB is on disk, never in RAM), so this loop is a disk-to-disk copy —
# but it must stay a *loop*, because ``await file.read()`` with no argument
# would pull the whole gigabyte into a single bytes object.
UPLOAD_CHUNK_BYTES = 1024 * 1024
# Default cap for a single upload. Generous enough for the longest real input
# (a 3-hour 48 kHz stereo WAV is 2.0 GB) and finite enough that a runaway or
# hostile client cannot fill the disk: the body is refused with 413 from its
# ``Content-Length`` before a byte is read, and again while streaming for a
# client that declares nothing. Override with ``HAWAVOCLEAN_MAX_UPLOAD_BYTES``
# (0 disables the cap) or ``create_app(max_upload_bytes=...)``.
DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_CONCURRENT_UPLOADS = 2
MAX_UPLOAD_BYTES_ENV = "HAWAVOCLEAN_MAX_UPLOAD_BYTES"
UPLOAD_PATH = "/api/upload"
SSE_MIN_INTERVAL_S = 0.05
SSE_PING_INTERVAL_S = 15.0
SHUTDOWN_DELAY_S = 0.2

_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    416: "range_not_satisfiable",
    422: "bad_request",
    500: "internal_error",
    503: "unavailable",
    507: "insufficient_storage",
}


class ApiError(Exception):
    """An endpoint-level error with the contract JSON shape."""

    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def error_response(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)


class TokenAuthMiddleware:
    """Every ``/api/*`` request must carry the token: header ``X-Hawa-Token``
    or query ``token`` (for ``EventSource`` and ``<audio src>``)."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope.get("path", "")).startswith("/api/"):
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).get("x-hawa-token")
        if supplied is None:
            supplied = QueryParams(scope.get("query_string", b"")).get("token")
        if not supplied or not hmac.compare_digest(
            supplied.encode("utf-8"), self.token.encode("utf-8")
        ):
            response = error_response(401, "unauthorized", "missing or invalid X-Hawa-Token")
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def configured_max_upload_bytes() -> int:
    """The upload cap from the environment, or the built-in default.

    A malformed or negative value is treated as "use the default" rather than
    silently disabling the cap — the failure mode of a typo must not be an
    unbounded upload.
    """
    raw = os.environ.get(MAX_UPLOAD_BYTES_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{MAX_UPLOAD_BYTES_ENV}={raw!r} is not an integer; using the default")
        return DEFAULT_MAX_UPLOAD_BYTES
    if value <= 0:
        logger.warning(f"{MAX_UPLOAD_BYTES_ENV}={raw!r} is not positive; using the default")
        return DEFAULT_MAX_UPLOAD_BYTES
    return value


class UploadSizeLimitMiddleware:
    """Refuse an over-sized ``POST /api/upload`` with 413 instead of filling the disk.

    Two checks, because either one alone has a hole. The declared
    ``Content-Length`` is refused before the body is read at all, which is the
    path every browser takes and the only one that costs nothing. A client that
    sends ``Transfer-Encoding: chunked`` declares no length, so the streamed
    bytes are counted as they arrive and the request is aborted the moment it
    passes the cap — before Starlette's spool file grows past it.

    Concurrent request bodies are bounded too: the middleware wraps multipart
    parsing, so at most ``max_concurrent`` file parts can occupy Starlette's
    spool area before the route enforces the persistent total quota.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        path: str = UPLOAD_PATH,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_UPLOADS,
    ) -> None:
        if max_bytes < 1 or max_concurrent < 1:
            raise ValueError("upload byte and concurrency limits must be positive")
        self.app = app
        self.max_bytes = max_bytes
        self.path = path
        self.max_concurrent = max_concurrent
        self._active = 0
        self._lock = asyncio.Lock()

    def _too_large(self, seen: int) -> StarletteHTTPException:
        return StarletteHTTPException(
            status_code=413,
            detail=f"upload exceeds the {self.max_bytes} byte limit ({seen} bytes)",
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or str(scope.get("path", "")) != self.path
        ):
            await self.app(scope, receive, send)
            return

        async with self._lock:
            if self._active >= self.max_concurrent:
                response = error_response(
                    503,
                    "upload_busy",
                    f"at most {self.max_concurrent} uploads may be received concurrently",
                )
                await response(scope, receive, send)
                return
            self._active += 1
        try:
            declared = Headers(scope=scope).get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
                response = error_response(
                    413,
                    "payload_too_large",
                    f"upload declares {int(declared)} bytes, over the {self.max_bytes} byte limit",
                )
                await response(scope, receive, send)
                return

            seen = 0

            async def counted() -> Any:
                nonlocal seen
                message = await receive()
                if message["type"] == "http.request":
                    seen += len(message.get("body", b""))
                    if seen > self.max_bytes:
                        raise self._too_large(seen)
                return message

            await self.app(scope, counted, send)
        finally:
            async with self._lock:
                self._active -= 1


class AnalyzeRequest(BaseModel):
    # ``extra="forbid"``: an unknown field is refused with 422, never silently
    # ignored — a client that misspells a knob must hear about it (audit
    # finding: a typo'd option used to be dropped and the request "succeed").
    model_config = ConfigDict(extra="forbid")

    path: str
    buckets: int = Field(default=DEFAULT_BUCKETS, ge=1, le=MAX_BUCKETS)


class PeaksRequest(BaseModel):
    """``POST /api/peaks`` (contract addendum 1). Non-finite bounds are refused
    here rather than downstream: ``json.loads`` happily accepts ``NaN`` and
    ``Infinity`` literals, and a NaN window would silently decode nothing."""

    model_config = ConfigDict(extra="forbid")

    path: str
    start_s: float = Field(ge=0.0, allow_inf_nan=False)
    end_s: float = Field(gt=0.0, allow_inf_nan=False)
    buckets: int = Field(default=DEFAULT_BUCKETS, ge=1, le=MAX_BUCKETS)


# The speaker id is handed to a child argv (``--speaker-id``) and joined into
# a profiles path, so it is held to the naming grammar the profile tree uses —
# anything else (spaces, ``/``, ``-``, uppercase, leading dashes) is refused.
SPEAKER_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


class JobRequest(BaseModel):
    """``POST /api/jobs`` (contract addendum 2 adds the restore-mode fields).
    The cross-field rules — restore requires ``speaker_id``, ``speaker_id``/
    ``cutoff_hz`` are restore-only — live in the submit endpoint so their 422s
    carry one clear message instead of a pydantic error list."""

    model_config = ConfigDict(extra="forbid")

    input_path: str
    profile: Literal["studio", "lowband", "production", "development"]
    output_path: str | None = None
    overwrite: bool = False
    mode: Literal["natural", "restore"] = "natural"
    speaker_id: str | None = None
    cutoff_hz: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)


def default_output_path(input_path: Path, profile: str) -> Path:
    """``<dir>/<stem>_studio.wav`` / ``<stem>_clean.wav`` with stacked audio
    suffixes stripped (``Flute 09.m4a.mp4`` -> ``Flute 09``)."""
    suffix = _OUTPUT_SUFFIX.get(profile, "_clean")
    return input_path.parent / f"{_clean_stem(input_path)}{suffix}.wav"


def available_speakers() -> list[str]:
    """Sorted speaker ids with a ``profile.json`` under :func:`profiles_root`.

    An absent or unreadable profiles tree is an empty list, not an error: a
    natural-mode-only install must still answer ``GET /api/health``, and the
    UI uses the emptiness itself (``restore_available``) to hide the restore
    control.
    """
    try:
        entries = list(profiles_root().iterdir())
    except OSError:
        return []
    return sorted(p.name for p in entries if (p / "profile.json").is_file())


def _safe_upload_name(raw: str | None) -> str:
    """A storable file name from an attacker-controlled multipart filename.

    Traversal is neutralised by taking the basename, and the two kinds of
    text that no filesystem call can accept are removed rather than allowed
    to raise ``ValueError`` out of ``open()``/``unlink()``: NUL bytes, and
    unpaired surrogates (a raw multipart header can carry either). Every
    other character a POSIX name may hold — space, newline, quote, emoji,
    non-UTF-8 bytes as surrogate escapes — is preserved.
    """
    name = (raw or "").replace("\x00", "")
    try:
        os.fsencode(name)
    except (UnicodeEncodeError, ValueError):
        name = name.encode("utf-8", "replace").decode("utf-8")
    name = Path(name).name
    if not name or name in (".", ".."):  # Path("..").name == "..": targets the dir itself
        return "upload.bin"
    return name


def content_type_for(path: Path) -> str:
    mime = _AUDIO_MIME.get(path.suffix.lower())
    if mime is None:
        guessed, _ = mimetypes.guess_type(path.name)
        mime = guessed or "application/octet-stream"
    return mime


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``bytes=`` header into an inclusive ``(start, end)``.

    Returns None when there is no usable Range header (serve the whole file);
    raises :class:`ApiError` 416 for a syntactically valid but unsatisfiable
    range."""
    if not header:
        return None
    unit, _, spec = header.strip().partition("=")
    if unit.strip().lower() != "bytes" or not spec:
        return None
    first = spec.split(",", 1)[0].strip()
    start_s, dash, end_s = first.partition("-")
    if not dash:
        return None
    unsatisfiable = {"Content-Range": f"bytes */{size}"}
    try:
        if start_s == "":
            # suffix range: last N bytes
            suffix = int(end_s)
            if suffix <= 0:
                raise ApiError(
                    416, "range_not_satisfiable", "empty suffix range", headers=unsatisfiable
                )
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_s)
            if end_s:
                end = int(end_s)
                if end < start:
                    # Invalid byte-range-spec (explicit end before start):
                    # ignore the whole header (RFC 9110 sec. 14.1.1).
                    return None
            else:
                end = size - 1
    except ValueError:
        return None
    if start >= size or start < 0:
        # A 416 must carry Content-Range: bytes */<size>; Chromium's media
        # stack uses it to recover the resource length when seeking.
        raise ApiError(
            416, "range_not_satisfiable", f"range {header!r} not satisfiable", headers=unsatisfiable
        )
    return start, min(end, size - 1)


def _file_chunks(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(_STREAM_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def ranged_file_response(
    path: Path, range_header: str | None, *, head_only: bool = False
) -> Response:
    """200 (whole file) or 206 (partial) with ``Accept-Ranges``/``Content-Range``.
    ``head_only`` answers a HEAD request: same status and headers, no body."""
    size = path.stat().st_size
    media_type = content_type_for(path)
    rng = parse_range(range_header, size) if size > 0 else None
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    if rng is None:
        status, start, end = 200, 0, size - 1
        headers["Content-Length"] = str(size)
    else:
        status, (start, end) = 206, rng
        headers["Content-Length"] = str(end - start + 1)
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    if head_only:
        return Response(status_code=status, media_type=media_type, headers=headers)
    body = _file_chunks(path, start, end) if size > 0 else iter(())
    return StreamingResponse(body, status_code=status, media_type=media_type, headers=headers)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def create_app(
    token: str,
    ui_dir: Path | None = None,
    *,
    job_manager: JobManager | None = None,
    on_shutdown: Callable[[], None] | None = None,
    max_upload_bytes: int | None = None,
    max_upload_total_bytes: int = DEFAULT_MAX_UPLOAD_TOTAL_BYTES,
    upload_ttl_s: float = DEFAULT_UPLOAD_TTL_S,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    retention_clock: Callable[[], float] | None = None,
    retention_disk_usage: DiskUsageFactory | None = None,
) -> FastAPI:
    """Build the engine app. ``on_shutdown`` runs (in a thread) shortly after
    ``POST /api/shutdown`` has been answered; when omitted the process hard-exits.
    ``max_upload_bytes`` caps ``POST /api/upload`` and must be positive; when
    omitted it comes from ``HAWAVOCLEAN_MAX_UPLOAD_BYTES`` or the default.
    Total upload bytes, age, and the emergency free-space reserve are always
    bounded; zero/negative values are rejected rather than becoming an
    accidental unlimited mode."""
    if not token:
        raise ValueError("token must be non-empty")
    manager = job_manager if job_manager is not None else JobManager()
    upload_limit = (
        configured_max_upload_bytes() if max_upload_bytes is None else int(max_upload_bytes)
    )
    if upload_limit < 1:
        raise ValueError("max_upload_bytes must be positive")
    upload_store = UploadStore(
        work_root() / "uploads",
        ttl_s=upload_ttl_s,
        max_total_bytes=max_upload_total_bytes,
        min_free_bytes=min_free_bytes,
        clock=retention_clock,
        disk_usage=retention_disk_usage,
    )
    upload_store.scavenge(manager.active_input_paths())

    def _cleanup_terminal_input(record: JobRecord) -> None:
        # One upload can back several jobs — the same file processed under two
        # profiles, or in natural and restore mode. Deleting it the moment the
        # FIRST of them finishes destroys the input the others are still going
        # to decode, so the user's upload disappears and their second job fails
        # preflight on a file they never removed. ``scavenge`` already honours
        # this set; the terminal path has to as well.
        if record.input_path.resolve() in manager.active_input_paths():
            return
        upload_store.cleanup_input(record.input_path)

    manager.add_terminal_callback(_cleanup_terminal_input)

    def _hard_exit() -> None:  # pragma: no cover - process exit
        os._exit(0)

    shutdown_hook: Callable[[], None] = on_shutdown or _hard_exit

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(title="HawaVoClean engine", version=__version__, lifespan=lifespan)
    app.state.job_manager = manager
    app.state.token = token
    app.state.ui_dir = ui_dir
    app.state.max_upload_bytes = upload_limit
    app.state.upload_store = upload_store
    app.state.upload_lock = asyncio.Lock()

    # Middleware: the last one added is outermost, so CORS wraps auth and
    # browser preflights (which carry no token) are answered before auth, and
    # the upload cap sits innermost — an unauthenticated flood is rejected by
    # the token check before its size is even considered.
    app.add_middleware(UploadSizeLimitMiddleware, max_bytes=upload_limit)
    app.add_middleware(TokenAuthMiddleware, token=token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["X-Hawa-Token", "Content-Type", "Range"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )

    # --- error shape ---------------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_req: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(PathPolicyError)
    async def _policy_error(_req: Request, exc: PathPolicyError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(detail, status_code=exc.status_code)
        return error_response(
            exc.status_code, _HTTP_CODES.get(exc.status_code, "http_error"), str(detail)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        unknown_field = False
        for err in exc.errors():
            if err.get("type") == "extra_forbidden":
                unknown_field = True
            loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
            parts.append(f"{loc}: {err.get('msg', 'invalid')}")
        # An unknown field is a well-formed request the schema rejects — 422,
        # like the restore-mode cross-field rules. Malformed values and missing
        # required fields keep the contract's historical 400.
        status = 422 if unknown_field else 400
        return error_response(status, "bad_request", "; ".join(parts) or "invalid request")

    @app.exception_handler(HawaVoCleanError)
    async def _engine_error(_req: Request, exc: HawaVoCleanError) -> JSONResponse:
        status = 400 if isinstance(exc, InvalidUserInputError) else 500
        return error_response(status, exc.exit_code.name.lower(), exc.message)

    @app.exception_handler(Exception)
    async def _unhandled(_req: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled error in engine API: {exc}", exc_info=True)
        return error_response(500, "internal_error", f"{type(exc).__name__}: {exc}")

    # --- routes ---------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        # Recomputed per request: a profile trained while the engine is up
        # (or an env-pointed tree swapped underneath it) shows without restart.
        speakers = available_speakers()
        return {
            "ok": True,
            "version": __version__,
            "profiles": list(PROFILES),
            "speakers": speakers,
            "restore_available": bool(speakers),
            "engine_pid": os.getpid(),
            "storage": {
                "managed_upload_bytes": upload_store.usage_bytes(),
                "managed_upload_limit_bytes": upload_store.max_total_bytes,
                "minimum_free_bytes": upload_store.min_free_bytes,
            },
        }

    @app.post("/api/analyze")
    async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
        path = resolve_client_path(req.path, must_exist=True)
        return await asyncio.to_thread(analyze_audio, path, req.buckets)

    @app.post("/api/peaks")
    async def peaks(req: PeaksRequest) -> dict[str, Any]:
        path = resolve_client_path(req.path, must_exist=True)
        try:
            return await asyncio.to_thread(
                compute_peaks_window, path, req.start_s, req.end_s, req.buckets
            )
        except PeaksWindowError as e:
            raise ApiError(400, "bad_request", str(e)) from e

    @app.post("/api/jobs", status_code=202)
    async def create_job(req: JobRequest) -> dict[str, Any]:
        # Restore-mode cross-field rules (contract addendum 2). 422: the JSON
        # is well-formed and every field is known — the *combination* is what
        # the contract refuses.
        if req.mode == "restore":
            if not req.speaker_id:
                raise ApiError(
                    422, "bad_request", 'mode "restore" requires speaker_id (see /api/health)'
                )
        else:
            if req.speaker_id is not None:
                raise ApiError(
                    422, "bad_request", 'speaker_id is only valid when mode is "restore"'
                )
            if req.cutoff_hz is not None:
                raise ApiError(422, "bad_request", 'cutoff_hz is only valid when mode is "restore"')
        if req.speaker_id is not None and not SPEAKER_ID_PATTERN.fullmatch(req.speaker_id):
            # The id becomes a child argv and a path segment: reject anything
            # outside the profile-tree grammar before it can travel.
            raise ApiError(422, "bad_request", "speaker_id must match ^[a-z0-9_]{1,64}$")
        if req.speaker_id is not None and req.speaker_id not in available_speakers():
            # /api/health publishes the installed speakers and the UI builds
            # its picker from that list, so a job naming one that is not there
            # is answerable now. It used to be accepted, queued, and spawned,
            # and the id was only checked once the child reached restoration --
            # after it had enhanced the whole file. Same list, same answer,
            # before any of that.
            raise ApiError(
                422,
                "bad_request",
                f"unknown speaker_id {req.speaker_id!r} (see /api/health for installed speakers)",
            )
        upload_store.scavenge(manager.active_input_paths())
        input_path = resolve_client_path(req.input_path, must_exist=True)
        if req.output_path:
            output_path = resolve_client_output_path(req.output_path)
        else:
            output_path = default_output_path(input_path, req.profile)
        if output_path.suffix.lower() != ".wav":
            raise ApiError(400, "bad_request", "output_path must end in .wav")
        try:
            snap = manager.submit(
                input_path=input_path,
                output_path=output_path,
                profile=req.profile,
                overwrite=req.overwrite,
                mode=req.mode,
                speaker_id=req.speaker_id,
                cutoff_hz=req.cutoff_hz,
            )
        except QueueFullError as e:
            raise ApiError(503, "queue_full", str(e)) from e
        except RuntimeError as e:
            raise ApiError(503, "unavailable", str(e)) from e
        return {
            "job_id": snap["job_id"],
            "output_path": snap["output_path"],
            "report_path": snap["report_path"],
        }

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        snap = manager.get_status(job_id)
        if snap is None:
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        return snap

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if manager.get_status(job_id) is None:
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        loop = asyncio.get_running_loop()

        async def stream() -> AsyncIterator[str]:
            # Subscribe inside the generator: a connection aborted before the
            # body ever starts would otherwise leak the subscription (its
            # finally block only runs once the generator has started).
            queue = manager.subscribe(job_id)
            if queue is None:  # pragma: no cover - table entries are never removed
                yield _sse("end", {})
                return
            try:
                snap = manager.get_status(job_id)
                if snap is None:  # pragma: no cover - table entries are never removed
                    yield _sse("end", {})
                    return
                yield _sse("status", snap)
                last_sent = loop.time()
                last_seq = int(snap["seq"])
                while snap["state"] not in TERMINAL_STATES:
                    try:
                        snap = await asyncio.wait_for(queue.get(), timeout=SSE_PING_INTERVAL_S)
                    except TimeoutError:
                        yield ": ping\n\n"
                        continue
                    # Coalesce: snapshots are complete states, so only the
                    # newest one matters. Throttle to >= 50 ms between sends.
                    while not queue.empty():
                        snap = queue.get_nowait()
                    wait = SSE_MIN_INTERVAL_S - (loop.time() - last_sent)
                    if wait > 0:
                        await asyncio.sleep(wait)
                        while not queue.empty():
                            snap = queue.get_nowait()
                    if int(snap["seq"]) <= last_seq:
                        continue  # queued before our initial snapshot: stale
                    yield _sse("status", snap)
                    last_sent = loop.time()
                    last_seq = int(snap["seq"])
                yield _sse("end", {})
            finally:
                manager.unsubscribe(job_id, queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        if not manager.cancel(job_id):
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        return {"ok": True}

    @app.api_route("/api/audio", methods=["GET", "HEAD"])
    async def audio(request: Request, path: str = "") -> Response:
        resolved = resolve_client_path(path, must_exist=True)
        return ranged_file_response(
            resolved, request.headers.get("range"), head_only=request.method == "HEAD"
        )

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
        """Stream the part to disk a megabyte at a time.

        Starlette has already spooled anything over 1 MiB into a temp file, so
        the body was never resident in memory; what matters here is that the
        copy stays chunked (measured: a 1.07 GB upload moves the engine's RSS
        by 5.7 MB) and that a failure part-way through does not leave a
        half-written file behind pretending to be audio.

        The part's filename is attacker-controlled text, not a name: it is
        reduced to a basename, and characters no filesystem call can accept
        (a NUL byte, an unpaired surrogate) are dropped rather than allowed
        to raise ``ValueError`` out of ``open()`` as a 500.
        """
        name = _safe_upload_name(file.filename)
        limit = int(getattr(app.state, "max_upload_bytes", 0))
        dest: Path | None = None
        try:
            # Local UI uploads are serialized. That makes the total quota an
            # exact reservation rather than a racy estimate across concurrent
            # requests, while processing jobs continue independently.
            async with app.state.upload_lock:
                upload_store.scavenge(manager.active_input_paths())
                existing = upload_store.usage_bytes()
                expected = int(file.size) if file.size is not None else upload_store.max_total_bytes
                upload_store.ensure_capacity(existing, expected)
                dest = upload_store.stage(name)
                written = 0
                with open(dest, "xb") as out:
                    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                        written += len(chunk)
                        if 0 < limit < written:
                            # Belt and braces: the middleware rejects an over-sized
                            # body before it is spooled, but a spool that arrived
                            # some other way must not be copied out in full either.
                            raise ApiError(
                                413,
                                "payload_too_large",
                                f"upload exceeds the {limit} byte limit",
                            )
                        upload_store.ensure_progress(existing, written)
                        out.write(chunk)
        except StoragePressureError as exc:
            if dest is not None:
                upload_store.cleanup_input(dest)
            raise ApiError(507, "insufficient_storage", str(exc)) from exc
        except BaseException:
            # Cleanup must never raise over the failure it is cleaning up
            # after: an unlink that itself throws would replace the real
            # error with its own.
            if dest is not None:
                with contextlib.suppress(OSError, ValueError):
                    upload_store.cleanup_input(dest)
            raise
        finally:
            await file.close()
        assert dest is not None
        return {"path": str(dest)}

    async def _shutdown_later() -> None:
        await asyncio.sleep(SHUTDOWN_DELAY_S)
        await asyncio.to_thread(shutdown_hook)

    @app.post("/api/shutdown")
    async def shutdown() -> Response:
        return JSONResponse({"ok": True}, background=BackgroundTask(_shutdown_later))

    @app.get("/api/{rest:path}")
    @app.post("/api/{rest:path}")
    async def _unknown_api(rest: str) -> Response:
        raise ApiError(404, "not_found", f"no such endpoint: /api/{rest}")

    if ui_dir is not None and (ui_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    else:
        if ui_dir is not None:
            logger.warning(f"--ui-dir {ui_dir} has no index.html; serving API only")

        @app.get("/{rest:path}")
        async def _no_ui(rest: str) -> Response:
            raise ApiError(404, "not_found", f"no UI bundle is being served (/{rest})")

    return app


def _validate_loopback(host: str) -> str:
    """The engine binds loopback only. Returns the numeric host to bind."""
    if host in ("localhost", ""):
        return "127.0.0.1"
    try:
        addr = ipaddress.ip_address(host)
    except ValueError as e:
        raise InvalidUserInputError(
            f"--host must be a loopback address (127.0.0.1), got {host!r}"
        ) from e
    if not addr.is_loopback or addr.version != 4:
        raise InvalidUserInputError(
            f"--host must be an IPv4 loopback address (127.0.0.1), got {host!r}"
        )
    return str(addr)


def bind_loopback_socket(host: str, port: int) -> socket.socket:
    """Bind + listen on ``host:port`` (port 0 = OS-assigned) and return the socket."""
    bind_host = _validate_loopback(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_host, port))
    sock.listen(128)
    return sock


def _schedule_hard_exit(delay_s: float) -> None:
    """Backstop for ``POST /api/shutdown``: the contract promises exit within
    1 s even if a client holds an SSE stream open through uvicorn's graceful
    shutdown."""
    timer = threading.Timer(delay_s, os._exit, args=(0,))
    timer.daemon = True
    timer.start()


def _redirect_stdout_to_stderr() -> None:
    """After the ready line, stdout belongs to nobody: point fd 1 at stderr so
    a stray print from any library lands where the logs go."""
    with contextlib.suppress(OSError, ValueError):
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _configure_uvicorn_logging() -> None:
    """uvicorn's own loggers: stderr only (its default config sends access
    logs to stdout, which would corrupt the ready-line protocol)."""
    uv_handler = logging.StreamHandler(sys.stderr)
    uv_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(uv_handler)
        uv_logger.setLevel(logging.INFO)
        uv_logger.propagate = False


def run_server(host: str, port: int, token: str, ui_dir: Path | None = None) -> int:
    """Serve until ``POST /api/shutdown`` or SIGINT/SIGTERM. Prints the ready
    line on stdout once the socket is listening; everything after that goes
    to stderr (stdout is re-pointed at stderr so no library can pollute it)."""
    import uvicorn

    if not token:
        raise InvalidUserInputError("--token must be non-empty")
    if ui_dir is not None and not (ui_dir / "index.html").is_file():
        raise InvalidUserInputError(f"--ui-dir has no index.html: {ui_dir}")

    sock = bind_loopback_socket(host, port)
    actual_port = int(sock.getsockname()[1])
    _configure_uvicorn_logging()

    manager = JobManager()
    server_box: dict[str, Any] = {}

    def request_shutdown() -> None:
        _schedule_hard_exit(0.7)
        manager.shutdown(grace_s=0.3)
        server = server_box.get("server")
        if server is not None:
            server.should_exit = True

    app = create_app(token, ui_dir, job_manager=manager, on_shutdown=request_shutdown)
    config = uvicorn.Config(
        app,
        host=host,
        port=actual_port,
        log_config=None,
        access_log=False,
        lifespan="on",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    server_box["server"] = server

    ready = {"event": "ready", "port": actual_port, "pid": os.getpid(), "version": __version__}
    sys.stdout.write(json.dumps(ready, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    _redirect_stdout_to_stderr()
    logger.info(f"HawaVoClean engine {__version__} listening on http://{host}:{actual_port}")

    server.run(sockets=[sock])
    manager.shutdown(grace_s=0.3)
    return 0


__all__ = [
    "ApiError",
    "JobManager",
    "bind_loopback_socket",
    "create_app",
    "parse_range",
    "ranged_file_response",
    "run_server",
]
