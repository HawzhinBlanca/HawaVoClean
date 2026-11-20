"""FastAPI application for the HawaVoClean engine bridge.

``create_app(token, ui_dir)`` builds the app; ``run_server(...)`` binds a
loopback socket, prints the single ``{"event":"ready",...}`` line to stdout
and serves with uvicorn. Routes, shapes and rules: ``docs/ui-contract.md``.
"""

import asyncio
import contextlib
import copy
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import asdict
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote_plus, urlencode, urlsplit

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from hawavoclean import __version__
from hawavoclean.audio.decode import (
    decode_audio_window,
    iter_decode_audio,
    window_sample_bounds,
)
from hawavoclean.audio.delivery import (
    EXPORT_FORMAT_EXTENSIONS,
    DeliveryPreset,
    ExportFormat,
    apply_delivery_loudness_streaming,
    export_delivery_artifact,
    extract_wav_metadata_chunks,
    inject_wav_metadata_chunks,
    measure_delivery_loudness_streaming,
)
from hawavoclean.audio.encode import (
    _finalize_deterministic_wav,
    _wav_container_format,
    encode_audio,
)
from hawavoclean.audio.multitrack import extract_polywav_tracks, inspect_polywav_tracks
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioBuffer
from hawavoclean.cli import _clean_stem
from hawavoclean.errors import (
    HawaVoCleanError,
    InvalidUserInputError,
    MediaPreflightError,
    OutputValidationError,
    PreflightError,
)
from hawavoclean.finishing.loudness import measure_loudness_and_peaks
from hawavoclean.finishing.pro_repair import apply_spectral_box_repair
from hawavoclean.hashing import hash_file
from hawavoclean.logging import get_logger
from hawavoclean.natural_contract import load_natural_route_contract
from hawavoclean.paths import config_dir, job_store_path, models_dir, profiles_root, work_root
from hawavoclean.platform_fs import is_reparse_or_symlink
from hawavoclean.publication import (
    public_output_path,
    publication_paths,
    publish_output_generation,
    resolve_immutable_publication_generation,
)
from hawavoclean.record_bundle import verify_processing_record
from hawavoclean.report.schema import HawaVoCleanReport
from hawavoclean.report.summary import generate_human_summary
from hawavoclean.report.writer import serialize_json_report
from hawavoclean.server.analysis import (
    DEFAULT_BUCKETS,
    MAX_BUCKETS,
    PeaksWindowError,
    analyze_audio,
    compute_peaks_window,
)
from hawavoclean.server.contracts import (
    MAX_SPECTROGRAM_WINDOW_S,
    CapabilitiesResponseV1,
    CapabilityStatusV1,
    DeliveryExportRequestV1,
    JobLifecycleStateV1,
    JobStatusResponseV1,
    ManualStrategyV1,
    ProcessingRequestV1,
    RegionPreviewRequestV1,
    SeparationStrategyV1,
    SmartAnalysisRequestV1,
    SmartAnalysisResponseV1,
    SmartSafeStrategyV1,
    SpectralRepairRequestV1,
    SpectrogramRequestV1,
    TrackExtractRequestV1,
    TrackInspectRequestV1,
    UnitOverrideRequestV1,
)
from hawavoclean.server.health_signals import HealthSignalsService
from hawavoclean.server.job_store import IdempotencyConflictError, OutputConflictError
from hawavoclean.server.job_store import canonical_request_hash as request_hash
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord, QueueFullError
from hawavoclean.server.policy import (
    PathPolicyError,
    refuse_unusable_filename_text,
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
from hawavoclean.server.source_caps import NativeSourceRegistry
from hawavoclean.server.spectrogram import (
    DELTA_DEFAULT_DB_CEIL,
    DELTA_DEFAULT_DB_FLOOR,
    compute_delta_spectrogram_tile,
    compute_spectrogram_tile,
)
from hawavoclean.server.unit_restitch import (
    SliceCacheBudgetError,
    UnitOverrideAction,
    UnitSliceStore,
    load_unit_slice,
    read_slice_truncated_marker,
    restitch_master_with_overrides,
)
from hawavoclean.server.warm_worker import (
    RegionPreviewRequest,
    cleanup_preview_dir,
    render_region_preview,
)
from hawavoclean.smart_safe import analyze_audio_stream

logger = get_logger("server")

PROFILES: tuple[str, ...] = ("studio", "lowband", "production")
_OUTPUT_SUFFIX = {
    "studio": "_studio",
    "studio_pro": "_studio_pro",
    "dereverb": "_dereverb",
    "lowband": "_lowband",
    "production": "_clean",
    "development": "_dev",
    "smart_safe": "_clean",
    "extract": "_extracted",
    "remove": "_removed",
    "restore_source": "_restored",
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
DEFAULT_MAX_JSON_BODY_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_EXTRACTED_TRACK_DIRS = 64
DEFAULT_EXTRACTED_TRACK_MAX_AGE_S = 3600.0
DEFAULT_MAX_CONCURRENT_UPLOADS = 2
DEFAULT_MAX_CONCURRENT_ANALYSES = 2
MAX_SMART_ANALYSIS_DURATION_S = 6 * 60 * 60.0
MAX_UPLOAD_BYTES_ENV = "HAWAVOCLEAN_MAX_UPLOAD_BYTES"
UPLOAD_PATH = "/api/upload"
SSE_MIN_INTERVAL_S = 0.05
SSE_PING_INTERVAL_S = 15.0
SHUTDOWN_DELAY_S = 0.2
SESSION_PATH = "/api/session"
NATIVE_SOURCE_PATH = "/api/v1/native-sources"
SESSION_COOKIE = "hawa_session"
DEFAULT_SESSION_TTL_S = 15 * 60.0
MAX_SESSION_TTL_S = 60 * 60.0
MAX_ACTIVE_SESSIONS = 256
TRUSTED_APP_ORIGINS = frozenset({"hawa://app"})
_CORS_METHODS = frozenset({"GET", "HEAD", "POST", "OPTIONS"})
_CORS_HEADERS = frozenset({"authorization", "content-type", "range", "x-hawa-token"})
_CORS_EXPOSE = (
    "Accept-Ranges, Content-Length, Content-Range, X-Hawa-Request-ID, "
    "Deprecation, Sunset, Link, X-Hawa-Sunset-Date, X-Hawa-Sunset-Release"
)

LEGACY_SUNSET_DATE = "Thu, 01 Oct 2026 00:00:00 GMT"
LEGACY_REMOVAL_RELEASE = "v1.0.0"
LEGACY_SUNSET_ISO_DATE = "2026-10-01"


def _legacy_successor(path: str) -> str | None:
    """Return the v1 successor path for a legacy route, or None if not legacy."""
    if path == "/api/jobs" or path == "/api/jobs/":
        return "/api/v1/jobs"
    if path.startswith("/api/jobs/"):
        suffix = path[len("/api/jobs") :]
        return f"/api/v1/jobs{suffix}"
    if path == "/api/analyze":
        return "/api/v1/analyze"
    if path == "/api/audio" or path.startswith("/api/audio/"):
        return "/api/v1/jobs"
    return None


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
    status: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
    *,
    request_id: str | None = None,
) -> JSONResponse:
    body = {"error": code, "message": message}
    response_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Pragma": "no-cache",
        **dict(headers or {}),
    }
    if request_id is not None:
        body["request_id"] = request_id
        response_headers["X-Hawa-Request-ID"] = request_id
    return JSONResponse(body, status_code=status, headers=response_headers)


def _request_id(scope: Scope) -> str:
    state = scope.setdefault("state", {})
    value = state.get("request_id")
    if isinstance(value, str):
        return value
    value = f"req_{secrets.token_hex(12)}"
    state["request_id"] = value
    return value


def _loopback_host(raw: str | None) -> bool:
    """Return whether a Host header names an exact supported loopback host.

    Parsing is intentionally narrow. Browser-tolerated oddities such as an
    embedded userinfo component, an IPv4 integer, or a trailing dot are not
    accepted at this trust boundary.
    """

    if raw is None or not raw or raw != raw.strip() or any(c in raw for c in "\r\n,@"):
        return False
    host = raw.lower()
    if host.startswith("["):
        match = re.fullmatch(r"\[([^]]+)](?::([0-9]{1,5}))?", host)
        if match is None:
            return False
        name, port = match.groups()
    else:
        if host.count(":") > 1:
            return False
        name, separator, port = host.partition(":")
        if not separator:
            port = ""
    if port and (not port.isdigit() or not 1 <= int(port) <= 65535):
        return False
    if name == "localhost":
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    return address.is_loopback


def _same_loopback_origin(origin: str, host: str) -> bool:
    """Accept the engine's exact HTTP origin (the optional server-hosted UI)."""

    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme != "http" or parsed.username or parsed.password:
        return False
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        return False
    try:
        origin_port = parsed.port
    except ValueError:
        return False
    origin_host = parsed.hostname
    if origin_host is None or not _loopback_host(parsed.netloc):
        return False

    raw_host = host.lower()
    if raw_host.startswith("["):
        end = raw_host.find("]")
        host_name = raw_host[1:end]
        raw_port = raw_host[end + 1 :]
        host_port = int(raw_port[1:]) if raw_port.startswith(":") else 80
    else:
        host_name, separator, raw_port = raw_host.partition(":")
        host_port = int(raw_port) if separator else 80
    return origin_host.lower() == host_name and (origin_port or 80) == host_port


class SessionRegistry:
    """Bounded, in-memory, short-lived client capabilities.

    Only SHA-256 digests are retained. A broker restart invalidates every
    session by design; the native shell can bootstrap another with the root
    header it already owns.
    """

    def __init__(
        self,
        ttl_s: float = DEFAULT_SESSION_TTL_S,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < ttl_s <= MAX_SESSION_TTL_S:
            raise ValueError(f"session_ttl_s must be in (0, {MAX_SESSION_TTL_S:g}]")
        self.ttl_s = float(ttl_s)
        self._clock = clock
        self._entries: dict[bytes, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8")).digest()

    def issue(self) -> tuple[str, int]:
        token = secrets.token_urlsafe(32)
        now = self._clock()
        expiry = now + self.ttl_s
        with self._lock:
            self._remove_expired(now)
            while len(self._entries) >= MAX_ACTIVE_SESSIONS:
                oldest = min(self._entries, key=self._entries.__getitem__)
                del self._entries[oldest]
            self._entries[self._digest(token)] = expiry
        return token, max(1, int(self.ttl_s))

    def valid(self, token: str) -> bool:
        if not token or len(token) > 256:
            return False
        now = self._clock()
        digest = self._digest(token)
        with self._lock:
            self._remove_expired(now)
            expiry = self._entries.get(digest)
            return expiry is not None and expiry > now

    def revoke(self, token: str) -> None:
        with self._lock:
            self._entries.pop(self._digest(token), None)

    def _remove_expired(self, now: float) -> None:
        for digest, expiry in list(self._entries.items()):
            if expiry <= now:
                del self._entries[digest]


def _cookie_session(headers: Headers) -> str | None:
    raw = headers.get("cookie")
    if raw is None or len(raw) > 4096:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(raw)
    except Exception:
        return None
    morsel = cookie.get(SESSION_COOKIE)
    return morsel.value if morsel is not None else None


def _has_query_token(raw: bytes) -> bool:
    """Detect a query credential without ever decoding or logging its value."""

    for field in raw.split(b"&"):
        key = field.partition(b"=")[0]
        try:
            decoded = unquote_plus(key.decode("ascii", "strict"))
        except (UnicodeDecodeError, ValueError):
            continue
        if decoded.casefold() == "token":
            return True
    return False


class LocalSecurityMiddleware:
    """Loopback Host/Origin validation, strict CORS, and API authentication.

    Native clients authenticate with ``X-Hawa-Token``. Browser clients may
    exchange that secret once at :data:`SESSION_PATH` for a short-lived bearer
    capability or HttpOnly cookie. Tokens in the query string are rejected so
    they cannot enter history, referrers, media URLs, or access logs.
    """

    def __init__(
        self,
        app: ASGIApp,
        token: str,
        sessions: SessionRegistry,
        trusted_origins: frozenset[str],
    ) -> None:
        self.app = app
        self.token = token
        self.sessions = sessions
        self.trusted_origins = trusted_origins

    @staticmethod
    async def _send_response(
        response: Response, scope: Scope, receive: Receive, send: Send
    ) -> None:
        await response(scope, receive, send)

    def _trusted_origin(self, origin: str, host: str) -> bool:
        return origin in self.trusted_origins or _same_loopback_origin(origin, host)

    def _cors_headers(self, origin: str) -> dict[str, str]:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": _CORS_EXPOSE,
            "Vary": "Origin",
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        headers = Headers(scope=scope)
        hosts = headers.getlist("host")
        host = hosts[0] if len(hosts) == 1 else None
        if not _loopback_host(host):
            response = error_response(
                403, "forbidden", "request Host is not an allowed loopback address"
            )
            await self._send_response(response, scope, receive, send)
            return

        origins = headers.getlist("origin")
        if len(origins) > 1:
            response = error_response(403, "forbidden", "multiple Origin headers are not allowed")
            await self._send_response(response, scope, receive, send)
            return
        origin = origins[0] if origins else None
        if origin is not None and not self._trusted_origin(origin, host or ""):
            response = error_response(403, "forbidden", "request Origin is not trusted")
            await self._send_response(response, scope, receive, send)
            return
        if origin is None and (headers.get("sec-fetch-site") or "").lower() == "cross-site":
            response = error_response(403, "forbidden", "cross-site browser request refused")
            await self._send_response(response, scope, receive, send)
            return

        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        scope.setdefault("state", {})["auth_kind"] = "none"
        if method == "OPTIONS" and path.startswith("/api/"):
            if origin is None:
                preflight_response: Response = error_response(
                    403, "forbidden", "CORS preflight requires Origin"
                )
            else:
                requested_method = (headers.get("access-control-request-method") or "").upper()
                requested_headers = {
                    item.strip().lower()
                    for item in (headers.get("access-control-request-headers") or "").split(",")
                    if item.strip()
                }
                if requested_method not in _CORS_METHODS or not requested_headers <= _CORS_HEADERS:
                    preflight_response = error_response(
                        403, "forbidden", "CORS preflight is not allowed"
                    )
                else:
                    preflight_response = Response(
                        status_code=204,
                        headers={
                            **self._cors_headers(origin),
                            "Access-Control-Allow-Methods": ", ".join(sorted(_CORS_METHODS)),
                            "Access-Control-Allow-Headers": ", ".join(sorted(_CORS_HEADERS)),
                            "Access-Control-Max-Age": "600",
                        },
                    )
            await self._send_response(preflight_response, scope, receive, send)
            return

        if path.startswith("/api/") or path == "/api":
            # Reject the old URL credential even when another credential is
            # valid. This makes accidental regressions visible immediately.
            raw_query = scope.get("query_string", b"")
            if _has_query_token(raw_query):
                response = error_response(
                    400,
                    "query_auth_forbidden",
                    "authentication tokens are not accepted in URLs",
                )
                await self._send_response(response, scope, receive, send)
                return

            roots = headers.getlist("x-hawa-token")
            if len(roots) > 1:
                response = error_response(401, "unauthorized", "multiple credentials refused")
                await self._send_response(response, scope, receive, send)
                return
            root = roots[0] if roots else None
            root_valid = root is not None and hmac.compare_digest(
                root.encode("utf-8"), self.token.encode("utf-8")
            )
            auth_kind: Literal["root", "session", "none"] = "none"
            if path == SESSION_PATH or root is not None:
                authenticated = root_valid
                if root_valid:
                    auth_kind = "root"
            else:
                authorizations = headers.getlist("authorization")
                if len(authorizations) > 1:
                    response = error_response(401, "unauthorized", "multiple credentials refused")
                    await self._send_response(response, scope, receive, send)
                    return
                authorization = authorizations[0] if authorizations else None
                if authorization is not None:
                    scheme, separator, bearer = authorization.partition(" ")
                    authenticated = (
                        separator == " "
                        and scheme.lower() == "bearer"
                        and self.sessions.valid(bearer)
                    )
                    if authenticated:
                        auth_kind = "session"
                else:
                    cookie_token = _cookie_session(headers)
                    authenticated = cookie_token is not None and self.sessions.valid(cookie_token)
                    if authenticated:
                        auth_kind = "session"
            if not authenticated:
                response = error_response(
                    401,
                    "unauthorized",
                    "missing or invalid HawaVoClean authentication",
                )
                if origin is not None:
                    for key, value in self._cors_headers(origin).items():
                        response.headers[key] = value
                await self._send_response(response, scope, receive, send)
                return
            scope.setdefault("state", {})["auth_kind"] = auth_kind

        async def secured_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                mutable = list(message.get("headers", []))
                mutable.append((b"x-hawa-request-id", request_id.encode("ascii")))
                if origin is not None:
                    cors = self._cors_headers(origin)
                    mutable.extend(
                        (key.lower().encode("ascii"), value.encode("ascii"))
                        for key, value in cors.items()
                    )
                if path.startswith("/api/") or path == "/api":
                    has_cache_control = False
                    new_headers: list[tuple[bytes, bytes]] = []
                    for name, value in mutable:
                        if name.lower() == b"cache-control":
                            has_cache_control = True
                            if b"no-store" not in value.lower():
                                new_headers.append(
                                    (name, b"no-store, no-cache, must-revalidate, private")
                                )
                            else:
                                new_headers.append((name, value))
                        else:
                            new_headers.append((name, value))
                    if not has_cache_control:
                        new_headers.append(
                            (b"cache-control", b"no-store, no-cache, must-revalidate, private")
                        )
                    if not any(name.lower() == b"pragma" for name, _ in new_headers):
                        new_headers.append((b"pragma", b"no-cache"))
                    successor = _legacy_successor(path)
                    if successor is not None:
                        if method != "OPTIONS":
                            app_obj = scope.get("app")
                            if app_obj is not None:
                                telemetry = getattr(app_obj.state, "legacy_telemetry", None)
                                if isinstance(telemetry, dict):
                                    telemetry["total_invocations"] = (
                                        int(telemetry.get("total_invocations", 0)) + 1
                                    )
                                    routes = telemetry.setdefault("routes", {})
                                    routes[path] = int(routes.get(path, 0)) + 1
                                    auth_kinds = telemetry.setdefault("auth_kinds", {})
                                    current_auth = str(
                                        scope.get("state", {}).get("auth_kind", "none")
                                    )
                                    auth_kinds[current_auth] = (
                                        int(auth_kinds.get(current_auth, 0)) + 1
                                    )
                        new_headers.append((b"deprecation", b"true"))
                        new_headers.append((b"sunset", LEGACY_SUNSET_DATE.encode("ascii")))
                        new_headers.append(
                            (b"link", f'<{successor}>; rel="successor-version"'.encode("ascii"))
                        )
                        new_headers.append(
                            (b"x-hawa-sunset-date", LEGACY_SUNSET_ISO_DATE.encode("ascii"))
                        )
                        new_headers.append(
                            (b"x-hawa-sunset-release", LEGACY_REMOVAL_RELEASE.encode("ascii"))
                        )
                    mutable = new_headers
                message["headers"] = mutable
            await send(message)

        await self.app(scope, receive, secured_send)


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
        max_json_bytes: int = DEFAULT_MAX_JSON_BODY_BYTES,
    ) -> None:
        if max_bytes < 1 or max_concurrent < 1 or max_json_bytes < 1:
            raise ValueError("upload byte and concurrency limits must be positive")
        self.app = app
        self.max_bytes = max_bytes
        self.path = path
        self.max_concurrent = max_concurrent
        self.max_json_bytes = max_json_bytes
        self._active = 0
        self._lock = asyncio.Lock()

    def _too_large(self, seen: int, *, limit: int | None = None) -> StarletteHTTPException:
        cap = self.max_bytes if limit is None else limit
        return StarletteHTTPException(
            status_code=413,
            detail=f"request body exceeds the {cap} byte limit ({seen} bytes)",
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        req_path = str(scope.get("path", ""))
        if req_path != self.path:
            if not req_path.startswith("/api/"):
                await self.app(scope, receive, send)
                return
            declared_json = Headers(scope=scope).get("content-length")
            if (
                declared_json is not None
                and declared_json.isdigit()
                and int(declared_json) > self.max_json_bytes
            ):
                response = error_response(
                    413,
                    "payload_too_large",
                    f"request body declares {int(declared_json)} bytes, over the {self.max_json_bytes} byte limit",
                )
                await response(scope, receive, send)
                return
            seen_json = 0

            async def counted_json() -> Any:
                nonlocal seen_json
                message = await receive()
                if message["type"] == "http.request":
                    seen_json += len(message.get("body", b""))
                    if seen_json > self.max_json_bytes:
                        raise self._too_large(seen_json, limit=self.max_json_bytes)
                return message

            await self.app(scope, counted_json, send)
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


class NativeSourceRequest(BaseModel):
    """One path selected by a trusted native main process."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=32_768)


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
    profile: Literal["studio", "studio_pro", "dereverb", "lowband", "production", "development"]
    output_path: str | None = None
    overwrite: bool = False
    conflict_policy: Literal["unique", "fail", "replace"] | None = None
    idempotency_key: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[\x21-\x7e]+$"
    )
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


def _redact_capability_detail(detail: str) -> str:
    """Strip local config/model/home roots from capability failure reasons."""
    for root, label in (
        (config_dir(), "<config-dir>"),
        (models_dir(), "<model-dir>"),
        (Path.home(), "<home>"),
    ):
        detail = detail.replace(str(root), label)
    return detail


def _natural_route_capability(profile: str) -> CapabilityStatusV1:
    """Inspect one route without loading an enhancer or neural checkpoint."""

    try:
        contract = load_natural_route_contract(profile, activate_runtime_config=False)
    except Exception as exc:
        # Capability inspection is a fail-closed trust boundary.  A corrupt
        # config, absent optional dependency, missing/tampered lock or weight,
        # or unexpected inspection failure must never become false readiness.
        detail = _redact_capability_detail(str(exc).strip() or type(exc).__name__)
        return CapabilityStatusV1(
            capability_id=profile,
            available=False,
            maturity="blocked",
            reason=f"{profile} route is not runnable: {detail}",
        )
    return CapabilityStatusV1(
        capability_id=profile,
        available=True,
        maturity="qualified",
        reason=(
            f"Runnable core {contract.config.enhancement.core_id!r} verified: effective "
            "config, guard calibration, implementation parameters, optional dependencies, "
            "core lock, and locked weights match the reported manifest identity"
        ),
        manifest_sha256=contract.manifest_sha256,
        providers=[contract.provider],
    )


def _restore_source_capability() -> CapabilityStatusV1:
    """Inspect guarded source-conditioned restoration (LavaSR v2) model-cold."""
    from hawavoclean.restoration.neural_bwe import (
        LAVASR_MODEL_DIR_ENV,
        probe_neural_bwe_runtime,
    )
    from hawavoclean.runtime import resolve_device

    explicit_dir = os.environ.get(LAVASR_MODEL_DIR_ENV)

    def _sanitize(text: str) -> str:
        clean = text
        if explicit_dir:
            clean = clean.replace(explicit_dir, "<lavasr-model-dir>")
            with contextlib.suppress(OSError, ValueError):
                clean = clean.replace(str(Path(explicit_dir).resolve()), "<lavasr-model-dir>")
        return _redact_capability_detail(clean)

    try:
        ok, reason = probe_neural_bwe_runtime()
    except Exception as exc:
        detail = _sanitize(str(exc).strip() or type(exc).__name__)
        return CapabilityStatusV1(
            capability_id="restore_source",
            available=False,
            maturity="blocked",
            reason=f"Source restoration runtime probe failed: {detail}",
        )
    if not ok:
        detail = _sanitize(str(reason).strip() or "LavaSR v2 runtime unavailable")
        return CapabilityStatusV1(
            capability_id="restore_source",
            available=False,
            maturity="blocked",
            reason=detail,
        )
    device = resolve_device("auto", preference=("mps", "cuda", "cpu")).resolved
    return CapabilityStatusV1(
        capability_id="restore_source",
        available=True,
        maturity="experimental",
        reason=(
            "Experimental guarded 48 kHz bandwidth extension "
            "(Apache-2.0 LavaSR v2; observed 0..cutoff band mathematically locked); "
            "requires explicit reconstruction consent."
        ),
        providers=[device],
    )


def capabilities_v1() -> CapabilitiesResponseV1:
    """Truthful runtime maturity; never infer product readiness from loose files."""

    natural_routes = [
        _natural_route_capability(profile)
        for profile in ("production", "studio", "studio_pro", "dereverb", "lowband")
    ]
    from hawavoclean.guard.perceptual_mos import perceptual_scorer_status
    from hawavoclean.separation import probe_separation_capabilities

    separation = probe_separation_capabilities(device="auto")
    scorer = perceptual_scorer_status()
    smart_safe_available = bool(scorer.get("available", True))
    raw_maturity = str(scorer.get("maturity", "experimental"))
    smart_safe_maturity: Literal["qualified", "experimental", "blocked"] = (
        "qualified"
        if raw_maturity == "qualified" and bool(scorer.get("qualified_ranker", False))
        else "blocked"
        if not smart_safe_available or raw_maturity == "blocked"
        else "experimental"
    )
    raw_providers = scorer.get("providers")
    smart_safe_providers = (
        [str(item) for item in raw_providers]
        if isinstance(raw_providers, (list, tuple)) and raw_providers
        else ["cpu"]
    )
    smart_safe_reason = (
        str(scorer.get("reason"))
        if scorer.get("reason")
        else (
            "Smart Safe remains experimental until a signed Sorani listener-calibrated ranker ships"
        )
    )

    return CapabilitiesResponseV1(
        capabilities=[
            CapabilityStatusV1(
                capability_id="separate",
                available=separation.available,
                maturity="experimental" if separation.available else "blocked",
                reason=separation.reason,
                providers=list(separation.providers),
            ),
            CapabilityStatusV1(
                capability_id="preserve",
                available=False,
                maturity="blocked",
                reason=("Preserve is an internal Smart Safe candidate, not a standalone route"),
            ),
            *natural_routes,
            CapabilityStatusV1(
                capability_id="lowband_then_production",
                available=False,
                maturity="experimental",
                reason="The route is not yet wired through the versioned job API",
            ),
            CapabilityStatusV1(
                capability_id="smart_analysis",
                available=True,
                maturity="experimental",
                reason=(
                    "Bounded streaming acoustic proxies are available, but they are not "
                    "a calibrated Sorani classifier and cannot qualify Smart Safe routing"
                ),
                providers=["cpu"],
            ),
            CapabilityStatusV1(
                capability_id="smart_safe",
                available=smart_safe_available,
                maturity=smart_safe_maturity,
                reason=smart_safe_reason,
                providers=smart_safe_providers,
            ),
            _restore_source_capability(),
            CapabilityStatusV1(
                capability_id="restore_enrolled",
                available=False,
                maturity="blocked",
                reason="No qualified signed enrolled-speaker Sorani Restore pack is installed",
            ),
            CapabilityStatusV1(
                capability_id="cloud",
                available=False,
                maturity="blocked",
                reason="Invite-only UAE cloud execution is not deployed",
            ),
        ]
    )


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
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Pragma": "no-cache",
    }
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


@contextlib.contextmanager
def _lease_source_id(
    upload_store: UploadStore,
    native_sources: NativeSourceRegistry,
    source_id: str,
) -> Iterator[Path | None]:
    """Lease either a managed upload or a root-registered native source."""

    with upload_store.lease_source(source_id) as uploaded:
        if uploaded is not None:
            yield uploaded
            return
    with native_sources.lease_source(source_id) as native:
        if native is not None:
            yield native
            return
    registered = native_sources.resolve_registered_path(source_id)
    if registered is not None:
        yield registered
        return
    try:
        candidate = Path(source_id)
        if upload_store.authorizes(candidate):
            opaque_id = upload_store.source_id(candidate)
            with upload_store.lease_source(opaque_id) as leased:
                yield leased
                return
    except Exception:
        pass
    yield None


def _analyze_smart_source(
    upload_store: UploadStore,
    native_sources: NativeSourceRegistry,
    source_id: str,
) -> SmartAnalysisResponseV1:
    """Lease, preflight, and reduce one source with bounded in-memory state."""

    with _lease_source_id(upload_store, native_sources, source_id) as source:
        if source is None:
            raise ApiError(404, "not_found", f"unknown or expired sourceId: {source_id}")
        probe = probe_audio(
            source,
            max_sample_rate=48_000,
            max_file_size_bytes=DEFAULT_MAX_UPLOAD_BYTES,
            max_duration_s=MAX_SMART_ANALYSIS_DURATION_S,
            max_channels=2,
        )
        if probe.channels not in {1, 2}:
            raise ApiError(422, "unsupported_layout", "Smart analysis supports mono or stereo only")
        if probe.duration_s > MAX_SMART_ANALYSIS_DURATION_S:
            raise ApiError(
                422,
                "resource_limit",
                "Smart analysis supports recordings up to six hours",
            )
        report = analyze_audio_stream(iter_decode_audio(probe))
        return SmartAnalysisResponseV1.model_validate({"schema_version": 1, **asdict(report)})


def _job_status_v1(snapshot: dict[str, Any]) -> JobStatusResponseV1:
    """Translate the legacy UI lifecycle into the durable public v1 states."""

    legacy_state = str(snapshot.get("state", "failed"))
    if legacy_state == "queued":
        state: JobLifecycleStateV1 = "queued"
    elif legacy_state == "running":
        running_states: dict[str, JobLifecycleStateV1] = {
            "preflight": "analyzing",
            "decode": "analyzing",
            "segment": "analyzing",
            "enhance": "rendering",
            "restore": "rendering",
            "guard": "guarding",
            "finish": "guarding",
            "publish": "publishing",
            "record_bundle": "publishing",
        }
        state = running_states.get(str(snapshot.get("stage", "")), "rendering")
    elif legacy_state == "done":
        state = "completed"
    elif legacy_state in {"cancelled", "interrupted", "failed"}:
        state = cast(JobLifecycleStateV1, legacy_state)
    else:
        state = "failed"
    return JobStatusResponseV1.model_validate(
        {
            "schema_version": 1,
            "job_id": snapshot["job_id"],
            "state": state,
            "stage": snapshot.get("stage", "unknown"),
            "progress": snapshot.get("progress", 0.0),
            "message": snapshot.get("message", ""),
            "output_path": snapshot["output_path"],
            "report_path": snapshot["report_path"],
            "record_bundle": snapshot.get("record_bundle", False),
            "bundle_path": snapshot.get("bundle_path"),
            "bundle": snapshot.get("bundle"),
            "created_at": snapshot["created_at"],
            "started_at": snapshot.get("started_at"),
            "finished_at": snapshot.get("finished_at"),
            "error": snapshot.get("error"),
            "report": snapshot.get("report"),
        }
    )


ArtifactKind = Literal["master", "report", "summary", "record"]


def _auth_kind(request: Request) -> str:
    value = getattr(request.state, "auth_kind", "none")
    return value if isinstance(value, str) else "none"


def _completed_audio_sha256(snapshot: dict[str, Any]) -> str | None:
    """Return the job-bound master digest, refusing contradictory evidence."""

    if snapshot.get("state") != "done":
        return None
    candidates: list[str] = []
    report = snapshot.get("report")
    if isinstance(report, dict):
        output = report.get("output")
        if isinstance(output, dict) and isinstance(output.get("sha256"), str):
            candidates.append(str(output["sha256"]))
    bundle = snapshot.get("bundle")
    if isinstance(bundle, dict) and isinstance(bundle.get("master_sha256"), str):
        candidates.append(str(bundle["master_sha256"]))
    artifact_evidence = snapshot.get("artifact_evidence")
    if isinstance(artifact_evidence, dict):
        audio_evidence = artifact_evidence.get("audio")
        if isinstance(audio_evidence, dict) and isinstance(audio_evidence.get("sha256"), str):
            candidates.append(str(audio_evidence["sha256"]))
    if not candidates or any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in candidates
    ):
        return None
    return candidates[0] if all(value == candidates[0] for value in candidates) else None


def _job_artifact_path(snapshot: dict[str, Any], kind: ArtifactKind) -> Path | None:
    """Resolve only immutable, job-bound completed artifacts.

    Public WAV/JSON/TXT files are recoverable convenience exports copied one
    at a time.  They are never returned here.  The authoritative generation
    is selected by the digest durably retained in the job snapshot, so a
    later replace-mode publication cannot make an older job URL serve newer
    bytes.
    """

    expected_audio = _completed_audio_sha256(snapshot)
    if expected_audio is None:
        return None
    if kind == "record":
        raw_bundle = snapshot.get("bundle_path")
        evidence = snapshot.get("bundle")
        if not isinstance(raw_bundle, str) or not isinstance(evidence, dict):
            return None
        verified = verify_processing_record(Path(raw_bundle))
        if (
            verified.archive_sha256 != evidence.get("archive_sha256")
            or verified.master_sha256 != expected_audio
        ):
            return None
        return verified.path

    bundle_evidence = snapshot.get("bundle")
    report_sha256 = (
        bundle_evidence.get("report_sha256") if isinstance(bundle_evidence, dict) else None
    )
    summary_sha256 = (
        bundle_evidence.get("summary_sha256") if isinstance(bundle_evidence, dict) else None
    )
    artifact_evidence = snapshot.get("artifact_evidence")
    if isinstance(artifact_evidence, dict):
        report_evidence = artifact_evidence.get("report")
        summary_evidence = artifact_evidence.get("summary")
        if isinstance(report_evidence, dict):
            report_sha256 = report_evidence.get("sha256")
        if isinstance(summary_evidence, dict):
            summary_sha256 = summary_evidence.get("sha256")
    generation = resolve_immutable_publication_generation(
        Path(str(snapshot["output_path"])),
        audio_sha256=expected_audio,
        report_sha256=report_sha256 if isinstance(report_sha256, str) else None,
        summary_sha256=summary_sha256 if isinstance(summary_sha256, str) else None,
    )
    if generation is None:
        return None
    role = {"master": 0, "report": 1, "summary": 2}[kind]
    return generation[role]


def _requested_job_artifact(
    manager: JobManager, requested_path: Path
) -> tuple[dict[str, Any], ArtifactKind] | None:
    """Find the newest completed job whose public artifact exactly matches."""

    for snapshot in reversed(manager.list_jobs()):
        if snapshot.get("state") != "done":
            continue
        output = public_output_path(str(snapshot["output_path"]))
        report = output.parent / f"{output.stem}.hawavoclean.json"
        summary = output.parent / f"{output.stem}.hawavoclean.txt"
        paths: dict[ArtifactKind, Path] = {
            "master": output,
            "report": report,
            "summary": summary,
        }
        raw_bundle = snapshot.get("bundle_path")
        if isinstance(raw_bundle, str):
            bundle = Path(raw_bundle)
            paths["record"] = bundle.parent.resolve() / bundle.name
        for kind, candidate in paths.items():
            if requested_path == candidate:
                return snapshot, kind
    return None


def _session_output_path(input_path: Path, profile: str, raw_output: str) -> Path:
    """Accept only the engine's derived sibling name (plus ``-N`` uniqueness)."""

    if not raw_output or not raw_output.strip():
        raise PathPolicyError(400, "bad_request", "output path is required")
    refuse_unusable_filename_text(raw_output, what="output path")
    lexical = Path(raw_output)
    if not lexical.is_absolute():
        raise PathPolicyError(400, "bad_request", "output path must be absolute")
    try:
        output = lexical.parent.resolve() / lexical.name
    except (OSError, ValueError) as exc:
        raise PathPolicyError(400, "bad_request", "output path cannot be resolved") from exc
    expected = default_output_path(input_path, profile)
    if output.parent != expected.parent:
        raise PathPolicyError(
            403,
            "forbidden",
            "renderer output must remain beside the selected input",
        )
    suffix = re.escape(expected.stem)
    unique_suffix = r"(?:-(?:[2-9]|[1-9][0-9]+))?"
    if (
        output.suffix.lower() != ".wav"
        or re.fullmatch(rf"{suffix}{unique_suffix}", output.stem) is None
    ):
        raise PathPolicyError(
            403,
            "forbidden",
            "renderer output name must be derived from the selected input and profile",
        )
    return output


_MAX_MANAGED_DELIVERY_EXPORTS = 256


class _DeliveryOutputRegistry:
    """Bounded, thread-safe registry of delivery exports authorized for non-root reads."""

    def __init__(self, maximum: int = _MAX_MANAGED_DELIVERY_EXPORTS) -> None:
        self._maximum = maximum
        self._entries: OrderedDict[Path, tuple[int, int]] = OrderedDict()
        self._lock = threading.Lock()

    def record(self, path: Path) -> None:
        """Record an exact non-symlink regular file created by ``POST /api/v1/export``."""
        if is_reparse_or_symlink(path):
            return
        try:
            canonical = path.parent.resolve() / path.name
            if is_reparse_or_symlink(canonical):
                return
            info = os.lstat(canonical)
        except (OSError, ValueError):
            return
        if not stat.S_ISREG(info.st_mode):
            return
        identity = (int(info.st_dev), int(info.st_ino))
        with self._lock:
            self._entries[canonical] = identity
            self._entries.move_to_end(canonical)
            while len(self._entries) > self._maximum:
                self._entries.popitem(last=False)

    def resolve_authorized_path(self, raw: str) -> Path | None:
        """Return the canonical path iff ``raw`` still points to a recorded regular export."""
        if not raw or not raw.strip():
            return None
        raw_path = Path(raw)
        if not raw_path.is_absolute() or is_reparse_or_symlink(raw_path):
            return None
        try:
            canonical = raw_path.parent.resolve() / raw_path.name
        except (OSError, ValueError):
            return None
        if is_reparse_or_symlink(canonical):
            return None
        with self._lock:
            expected = self._entries.get(canonical)
            if expected is None:
                return None
            try:
                info = os.lstat(canonical)
            except OSError:
                self._entries.pop(canonical, None)
                return None
            if (
                is_reparse_or_symlink(canonical)
                or not stat.S_ISREG(info.st_mode)
                or (int(info.st_dev), int(info.st_ino)) != expected
            ):
                self._entries.pop(canonical, None)
                return None
            self._entries.move_to_end(canonical)
            return canonical


def _is_system_prefix_path(candidate: Path) -> bool:
    """Return True if ``candidate`` is a host/system root alias (e.g. macOS ``/var``)."""
    for anchor in (Path.home(), Path(tempfile.gettempdir()), Path("/tmp")):
        if anchor == candidate or anchor.is_relative_to(candidate):
            return True
    return False


def _reject_export_destination_symlink(raw_path: str, dest_resolved: Path) -> None:
    """Refuse symlinks/reparse points on the final component or any non-system ancestor."""
    raw = Path(raw_path)
    if is_reparse_or_symlink(raw) or is_reparse_or_symlink(dest_resolved):
        raise PathPolicyError(
            403,
            "forbidden",
            "export destination must not be a symlink or reparse point",
        )
    for ancestor in raw.parents:
        if ancestor == ancestor.parent or _is_system_prefix_path(ancestor):
            continue
        if is_reparse_or_symlink(ancestor):
            raise PathPolicyError(
                403,
                "forbidden",
                "export destination ancestor must not be a symlink or reparse point",
            )
    for ancestor in dest_resolved.parents:
        if ancestor == ancestor.parent:
            continue
        if is_reparse_or_symlink(ancestor):
            raise PathPolicyError(
                403,
                "forbidden",
                "export destination ancestor must not be a symlink or reparse point",
            )


def _sanitize_api_error(detail: str, *paths: Path | None) -> str:
    """Strip host directory prefixes and resolved paths from client-facing error messages."""
    clean = detail
    for path in paths:
        if path is None:
            continue
        clean = clean.replace(str(path), "<path>")
        with contextlib.suppress(OSError, ValueError):
            clean = clean.replace(str(path.resolve()), "<path>")
    for root, label in (
        (work_root(), "<work-dir>"),
        (config_dir(), "<config-dir>"),
        (models_dir(), "<model-dir>"),
        (Path.home(), "<home>"),
    ):
        clean = clean.replace(str(root), label)
        with contextlib.suppress(OSError, ValueError):
            clean = clean.replace(str(root.resolve()), label)
    return clean


def cleanup_extracted_tracks_dir(
    extracted_root: Path,
    *,
    max_dirs: int = DEFAULT_MAX_EXTRACTED_TRACK_DIRS,
    max_age_s: float = DEFAULT_EXTRACTED_TRACK_MAX_AGE_S,
    protect: Path | None = None,
) -> int:
    """Prune expired or excess managed track extraction directories safely."""
    if not extracted_root.is_dir():
        return 0
    now = time.time()
    protected_name = protect.name if protect is not None else None
    protected_present = False
    candidates: list[tuple[float, str, Path]] = []
    for entry in extracted_root.iterdir():
        if is_reparse_or_symlink(entry) or not entry.is_dir():
            continue
        if protected_name is not None and entry.name == protected_name:
            protected_present = True
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, entry.name, entry))

    removed = 0
    survivors: list[tuple[float, str, Path]] = []
    for mtime, name, entry in candidates:
        if max_age_s > 0.0 and (now - mtime) > max_age_s:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        else:
            survivors.append((mtime, name, entry))

    survivors.sort(key=lambda item: (item[0], item[1]), reverse=True)
    keep_budget = max(0, max_dirs - (1 if protected_present else 0))
    for _mtime, _name, entry in survivors[keep_budget:]:
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed


def _is_managed_preview_file(raw: str, resolved: Path) -> bool:
    """Return True iff ``resolved`` is a regular managed preview/spectral WAV in ``work_root/previews``."""
    preview_dir = (work_root() / "previews").resolve()
    return (
        resolved.parent == preview_dir
        and resolved.is_file()
        and not is_reparse_or_symlink(Path(raw))
        and not is_reparse_or_symlink(resolved)
        and resolved.name.startswith(("preview_", "spectral_"))
        and resolved.suffix.lower() == ".wav"
    )


def _is_managed_extracted_track_file(raw: str, resolved: Path) -> bool:
    """Return True iff ``resolved`` is a regular non-symlink WAV under ``work_root/extracted_tracks``."""
    extracted_root = (work_root() / "extracted_tracks").resolve()
    raw_path = Path(raw)
    if not (
        resolved.is_relative_to(extracted_root)
        and resolved != extracted_root
        and resolved.is_file()
        and resolved.suffix.lower() == ".wav"
        and not is_reparse_or_symlink(raw_path)
        and not is_reparse_or_symlink(resolved)
    ):
        return False
    cur = resolved.parent
    while cur != extracted_root and cur != cur.parent:
        if is_reparse_or_symlink(cur):
            return False
        cur = cur.parent
    if raw_path.is_relative_to(extracted_root):
        raw_cur = raw_path.parent
        while raw_cur != extracted_root and raw_cur != raw_cur.parent:
            if is_reparse_or_symlink(raw_cur):
                return False
            raw_cur = raw_cur.parent
    return True


_SPECTRAL_REPAIR_MARGIN_S = 0.15
_SPECTRAL_STREAM_CHUNK_SAMPLES = 1 << 18


def stream_spectral_box_repair(
    source_path: Path,
    output_path: Path,
    *,
    start_s: float,
    end_s: float,
    f_low_hz: float,
    f_high_hz: float,
    attenuation_db: float = 24.0,
    chunk_samples: int = _SPECTRAL_STREAM_CHUNK_SAMPLES,
) -> tuple[float, float]:
    """Apply 2D spectral box repair via streaming windowed splice in bounded RAM.

    Decodes only ``[max(0, start_s - margin), min(duration_s, end_s + margin)]``
    via :func:`decode_audio_window`, applies :func:`apply_spectral_box_repair`
    on that local window, and streams the output WAV chunk-by-chunk via
    :func:`iter_decode_audio` + :class:`soundfile.SoundFile` so samples outside
    the repaired window remain bit-identical and peak RAM stays bounded even on
    multi-hour stereo files.
    """
    if (end_s - start_s) > MAX_SPECTROGRAM_WINDOW_S:
        raise ValueError(
            f"Spectral repair window duration ({end_s - start_s:.2f}s) exceeds maximum "
            f"{MAX_SPECTROGRAM_WINDOW_S:.1f}s"
        )
    if isinstance(chunk_samples, bool) or chunk_samples < 1:
        raise ValueError(f"chunk_samples must be >= 1, got {chunk_samples!r}")

    probe = probe_audio(source_path)
    sr = int(probe.sample_rate)
    channels = int(probe.channels)
    file_duration_s = probe.samples / float(sr)
    if start_s >= file_duration_s:
        raise ValueError(f"start_s {start_s:.3f}s is at or past file end ({file_duration_s:.3f}s)")
    effective_end_s = min(end_s, file_duration_s)
    if effective_end_s <= start_s:
        raise ValueError("Clamped spectral repair time window is empty")

    nyquist = 0.5 * float(sr)
    if f_high_hz > nyquist:
        raise ValueError(f"fHighHz ({f_high_hz:g} Hz) exceeds Nyquist ({nyquist:g} Hz)")

    margin_s = max(_SPECTRAL_REPAIR_MARGIN_S, (2048 + 512) / float(sr) + 0.02)
    raw_win_start_s = max(0.0, start_s - margin_s)
    raw_win_end_s = min(file_duration_s, effective_end_s + margin_s)
    win_start_idx, win_end_idx = window_sample_bounds(probe, raw_win_start_s, raw_win_end_s)
    win_start_s = win_start_idx / float(sr)
    win_end_s = win_end_idx / float(sr)

    local_buf = decode_audio_window(probe, win_start_s, win_end_s)
    actual_win_len = int(local_buf.data.shape[1])
    local_start_s = max(0.0, start_s - win_start_s)
    local_end_s = min(actual_win_len / float(sr), effective_end_s - win_start_s)
    if local_end_s <= local_start_s:
        raise ValueError("Clamped spectral repair time window is empty")

    repaired_win = apply_spectral_box_repair(
        local_buf.data,
        sr,
        start_s=local_start_s,
        end_s=local_end_s,
        f_low_hz=f_low_hz,
        f_high_hz=f_high_hz,
        attenuation_db=attenuation_db,
    )
    del local_buf
    win_splice_end = win_start_idx + int(repaired_win.shape[1])

    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(
        f".{dest_path.stem}_{os.getpid()}_{threading.get_ident()}.tmp.wav"
    )
    container = _wav_container_format(channels, max(probe.samples, 1), "PCM_24")
    try:
        written = 0
        with sf.SoundFile(
            str(tmp_path),
            mode="w",
            samplerate=sr,
            channels=channels,
            subtype="PCM_24",
            format=container,
        ) as writer:
            for chunk in iter_decode_audio(probe, chunk_samples=chunk_samples):
                chunk_len = int(chunk.data.shape[1])
                chunk_start = written
                chunk_end = written + chunk_len
                if chunk_end <= win_start_idx or chunk_start >= win_splice_end:
                    writer.write(chunk.data.T)
                else:
                    ov_start = max(chunk_start, win_start_idx)
                    ov_end = min(chunk_end, win_splice_end)
                    if ov_start > chunk_start:
                        writer.write(chunk.data[:, : ov_start - chunk_start].T)
                    writer.write(
                        repaired_win[:, ov_start - win_start_idx : ov_end - win_start_idx].T
                    )
                    if ov_end < chunk_end:
                        writer.write(chunk.data[:, ov_end - chunk_start :].T)
                written = chunk_end
        _finalize_deterministic_wav(tmp_path)
        os.replace(tmp_path, dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return start_s, effective_end_s


def _run_spectral_box_repair(
    source_path: Path,
    req: SpectralRepairRequestV1,
) -> dict[str, Any]:
    """Stream-splice 2D spectral box attenuation and write atomic ``spectral_<id>.wav``."""
    t0 = time.perf_counter()
    preview_dir = (work_root() / "previews").resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)
    key_material = (
        f"{source_path}:{req.start_s:.6f}:{req.end_s:.6f}:"
        f"{req.f_low_hz:.2f}:{req.f_high_hz:.2f}:{req.attenuation_db:.2f}:"
        f"{time.time_ns()}:{os.getpid()}:{threading.get_ident()}"
    )
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    final_path = preview_dir / f"spectral_{digest}.wav"

    start_s, effective_end_s = stream_spectral_box_repair(
        source_path,
        final_path,
        start_s=req.start_s,
        end_s=req.end_s,
        f_low_hz=req.f_low_hz,
        f_high_hz=req.f_high_hz,
        attenuation_db=req.attenuation_db,
    )

    cleanup_preview_dir(preview_dir, protect=final_path)

    output_path_str = str(final_path)
    render_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "outputPath": output_path_str,
        "audioUrl": f"/api/audio?{urlencode({'path': output_path_str})}",
        "startS": start_s,
        "endS": effective_end_s,
        "fLowHz": req.f_low_hz,
        "fHighHz": req.f_high_hz,
        "attenuationDb": req.attenuation_db,
        "renderMs": render_ms,
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON file beside ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    encoded = json.dumps(payload, indent=2) + "\n"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _default_export_dest_path(
    master_raw: str,
    master_resolved: Path,
    fmt: ExportFormat,
    preset: DeliveryPreset,
    source_resolved: Path | None,
) -> Path:
    """Derive a safe sibling destination path for ``POST /api/v1/export`` when omitted."""
    public_master = public_output_path(master_raw)
    ext_map: dict[ExportFormat, str] = {
        "wav24": ".wav",
        "wav32f": ".wav",
        "flac": ".flac",
        "aac": ".m4a",
        "video_mux": (
            source_resolved.suffix.lower()
            if source_resolved is not None
            and source_resolved.suffix.lower() in EXPORT_FORMAT_EXTENSIONS["video_mux"]
            else ".mp4"
        ),
    }
    ext = ext_map[fmt]
    preset_tag = f"_{preset}" if preset != "match_input" else ""
    fmt_tag = f"_{fmt}" if fmt in ("wav24", "wav32f") else ""
    base_stem = public_master.stem
    candidate = public_master.parent / f"{base_stem}{preset_tag}{fmt_tag}{ext}"
    if candidate in {master_resolved, public_master} or (
        source_resolved is not None and candidate == source_resolved
    ):
        candidate = public_master.parent / f"{base_stem}{preset_tag}_{fmt}_export{ext}"
    return resolve_client_output_path(str(candidate))


def _run_delivery_export(
    *,
    master_resolved: Path,
    dest_resolved: Path,
    fmt: ExportFormat,
    delivery_preset: DeliveryPreset,
    source_resolved: Path | None,
    preserve_metadata: bool,
) -> dict[str, Any]:
    """Apply optional streaming delivery loudness and export the master artifact synchronously."""
    if delivery_preset != "match_input":
        dest_resolved.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".hawa-export-", dir=dest_resolved.parent
        ) as tmp_dir:
            tmp_master = Path(tmp_dir) / "delivered.wav"
            metrics = apply_delivery_loudness_streaming(
                master_resolved,
                tmp_master,
                delivery_preset,
            )
            if preserve_metadata and fmt in ("wav24", "wav32f"):
                master_meta = extract_wav_metadata_chunks(master_resolved)
                if master_meta:
                    inject_wav_metadata_chunks(tmp_master, master_meta)
            out_path = export_delivery_artifact(
                tmp_master,
                dest_resolved,
                fmt=fmt,
                source_media_path=source_resolved,
                preserve_metadata=preserve_metadata,
            )
        lra_lu = round(float(metrics["loudness_range_lu"]), 2)
        integrated_lufs = round(float(metrics["integrated_lufs"]), 2)
        true_peak_dbtp = round(float(metrics["true_peak_dbtp"]), 2)
    else:
        out_path = export_delivery_artifact(
            master_resolved,
            dest_resolved,
            fmt=fmt,
            source_media_path=source_resolved,
            preserve_metadata=preserve_metadata,
        )
        meas, lra = measure_delivery_loudness_streaming(master_resolved)
        lra_lu = round(float(lra), 2)
        integrated_lufs = round(float(meas.integrated_lufs), 2)
        true_peak_dbtp = round(float(meas.true_peak_dbtp), 2)

    return {
        "outputPath": str(out_path),
        "format": fmt,
        "deliveryPreset": delivery_preset,
        "lraLu": lra_lu,
        "integratedLufs": integrated_lufs,
        "truePeakDbtp": true_peak_dbtp,
    }


def _normalize_restitch_decision(
    restitched_decision: str,
    *,
    auto_decision: str,
    manual_override: str | None,
    is_speech: bool,
) -> str:
    """Map ``unit_restitch`` decision tokens into lowercase report vocabulary."""
    if manual_override is None:
        return auto_decision
    upper = restitched_decision.upper()
    if upper == "ENHANCED":
        return "enhanced"
    if upper == "NO_SPEECH" or not is_speech:
        return "original_no_speech"
    return "reverted"


def _has_complete_unit_slices(job_slice_dir: Path, units: list[dict[str, Any]]) -> bool:
    """Return True when every report unit has a cached ``.npz`` slice on disk."""
    if not units or not job_slice_dir.is_dir():
        return False
    for rec in units:
        if "unit_id" not in rec or "start_sample" not in rec or "end_sample" not in rec:
            return False
        unit_id = int(rec["unit_id"])
        channel = int(rec.get("channel", 0))
        if not (job_slice_dir / f"unit_{unit_id:06d}_ch{channel}.npz").is_file():
            return False
    return True


def _live_restitch_unit_override(
    *,
    job_id: str,
    snapshot: dict[str, Any],
    report: dict[str, Any],
    units: list[dict[str, Any]],
    target_idx: int,
    req: UnitOverrideRequestV1,
    job_slice_dir: Path,
    manager: JobManager,
) -> dict[str, Any]:
    """Apply a unit override via sub-100ms cached-slice re-stitching and atomic republication."""
    t0 = time.perf_counter()
    target_unit = units[target_idx]
    target_unit_id = int(target_unit["unit_id"])
    old_decision = str(target_unit.get("final_decision", "unknown"))

    auto_units_path = job_slice_dir / "auto_units.json"
    if auto_units_path.is_file():
        loaded_auto = json.loads(auto_units_path.read_text(encoding="utf-8"))
        if (
            not isinstance(loaded_auto, list)
            or not loaded_auto
            or not all(isinstance(item, dict) for item in loaded_auto)
        ):
            raise ValueError("Corrupt auto_units metadata in slice cache")
        auto_units: list[dict[str, Any]] = loaded_auto
    else:
        auto_units = [dict(u) for u in units]
        _atomic_write_json(auto_units_path, auto_units)
    auto_by_id = {int(u["unit_id"]): u for u in auto_units}

    overrides_path = job_slice_dir / "overrides.json"
    overrides_map: dict[str, dict[str, Any]] = {}
    if overrides_path.is_file():
        loaded = json.loads(overrides_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            overrides_map = {str(k): dict(v) for k, v in loaded.items() if isinstance(v, dict)}

    if req.decision == "auto":
        overrides_map.pop(str(target_unit_id), None)
    else:
        overrides_map[str(target_unit_id)] = {
            "decision": req.decision,
            "strength": req.strength,
        }

    overrides_arg: dict[int, tuple[UnitOverrideAction, float | None]] = {}
    for uid_str, entry in overrides_map.items():
        action = cast(UnitOverrideAction, str(entry["decision"]))
        raw_strength = entry.get("strength")
        strength_val = float(raw_strength) if raw_strength is not None else None
        overrides_arg[int(uid_str)] = (action, strength_val)

    raw_output_meta = report.get("output")
    output_meta: dict[str, Any] = raw_output_meta if isinstance(raw_output_meta, dict) else {}
    sample_rate = int(output_meta.get("sample_rate", 48000))
    output_channels = int(output_meta.get("channels", 1))
    first_slice = load_unit_slice(
        job_slice_dir,
        unit_id=int(auto_units[0]["unit_id"]),
        channel=int(auto_units[0].get("channel", 0)),
    )
    duplicate_to_stereo = bool(first_slice.unit_meta.get("duplicate_to_stereo", False))
    present_channels = {int(u.get("channel", 0)) for u in auto_units}
    restitch_channels = (
        1
        if (duplicate_to_stereo or (output_channels > 1 and present_channels == {0}))
        else output_channels
    )

    out_path = public_output_path(str(snapshot["output_path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hawa-restitch-", dir=out_path.parent) as tmp_dir:
        tmp_master = Path(tmp_dir) / "restitched.wav"
        restitch_res = restitch_master_with_overrides(
            job_slice_dir,
            auto_units,
            overrides_arg,
            tmp_master,
            sample_rate=sample_rate,
            channels=restitch_channels,
            seed_context=f"restitch:{job_id}",
        )
        master_sha256 = str(restitch_res["master_sha256"])

        if restitch_channels == 1 and output_channels == 2:
            mono_wave, mono_sr = sf.read(str(tmp_master), dtype="float32", always_2d=True)
            stereo_data = np.ascontiguousarray(
                np.vstack((mono_wave[:, 0], mono_wave[:, 0])), dtype=np.float32
            )
            tmp_stereo = Path(tmp_dir) / "restitched_stereo.wav"
            encode_audio(
                AudioBuffer(data=stereo_data, sample_rate=int(mono_sr)),
                tmp_stereo,
                output_bit_depth="pcm24",
                dither=False,
                seed_context=f"restitch:{job_id}",
            )
            tmp_master = tmp_stereo
            master_sha256 = hash_file(tmp_master)

        final_audio, final_sr = sf.read(str(tmp_master), dtype="float32", always_2d=True)
        final_meas = measure_loudness_and_peaks(
            np.ascontiguousarray(final_audio.T, dtype=np.float32), int(final_sr)
        )

        updated_by_id = {int(u["unit_id"]): u for u in restitch_res["units"]}
        new_units: list[dict[str, Any]] = []
        for idx_u, orig_rec in enumerate(units):
            uid = int(orig_rec["unit_id"])
            auto_rec = auto_by_id.get(uid, orig_rec)
            restitched_rec = updated_by_id.get(uid, orig_rec)
            merged = dict(orig_rec)
            manual_ov = restitched_rec.get("manual_override")
            is_speech = bool(merged.get("is_speech", True))
            norm_decision = _normalize_restitch_decision(
                str(restitched_rec.get("final_decision", merged.get("final_decision", "reverted"))),
                auto_decision=str(auto_rec.get("final_decision", "reverted")),
                manual_override=manual_ov if isinstance(manual_ov, str) else None,
                is_speech=is_speech,
            )
            merged["final_decision"] = norm_decision
            merged["chosen_strength"] = float(
                restitched_rec.get("chosen_strength", merged.get("chosen_strength", 0.0))
            )
            merged["manual_override"] = manual_ov
            if idx_u == target_idx:
                if req.decision == "auto":
                    merged["decision_reason"] = str(
                        auto_rec.get("decision_reason", "auto restored")
                    )
                elif req.decision == "custom_strength":
                    merged["decision_reason"] = (
                        f"manual override: custom_strength={req.strength} (was {old_decision})"
                    )
                else:
                    merged["decision_reason"] = (
                        f"manual override: {req.decision} (was {old_decision})"
                    )
            new_units.append(merged)

        updated_report = dict(report)
        updated_report["units"] = new_units
        if isinstance(updated_report.get("output"), dict):
            updated_output = dict(updated_report["output"])
            updated_output["sha256"] = master_sha256
            updated_output["integrated_lufs"] = float(final_meas.integrated_lufs)
            updated_output["true_peak_dbtp"] = float(final_meas.true_peak_dbtp)
            updated_report["output"] = updated_output
        if isinstance(updated_report.get("summary"), dict):
            updated_summary = dict(updated_report["summary"])
            updated_summary["enhanced"] = sum(
                1 for u in new_units if u.get("final_decision") == "enhanced"
            )
            updated_summary["reverted"] = sum(
                1 for u in new_units if u.get("final_decision") in ("reverted", "original_reverted")
            )
            updated_summary["no_speech"] = sum(
                1 for u in new_units if u.get("final_decision") == "original_no_speech"
            )
            updated_report["summary"] = updated_summary

        try:
            validated_rep = HawaVoCleanReport.model_validate(updated_report)
            json_str = serialize_json_report(validated_rep)
            txt_str = generate_human_summary(validated_rep)
            updated_report = json.loads(json_str)
        except Exception:
            json_str = json.dumps(updated_report, indent=2) + "\n"
            txt_path = out_path.parent / f"{out_path.stem}.hawavoclean.txt"
            txt_str = txt_path.read_text(encoding="utf-8") if txt_path.is_file() else ""

        pub = publication_paths(out_path)
        if pub.bundle.is_dir():
            publish_output_generation(
                tmp_master,
                out_path,
                json_str,
                txt_str,
                overwrite=True,
            )
        else:
            os.replace(tmp_master, out_path)
            raw_report_path = snapshot.get("report_path")
            if isinstance(raw_report_path, str) and raw_report_path:
                _atomic_write_json(Path(raw_report_path), updated_report)

    _atomic_write_json(overrides_path, overrides_map)
    manager.update_completed_override(
        job_id,
        updated_report,
        recapture_artifacts=not bool(snapshot.get("record_bundle", False)),
    )

    new_decision = str(new_units[target_idx]["final_decision"])
    restitch_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "job_id": job_id,
        "unit_index": target_idx,
        "unit_id": target_unit_id,
        "old_decision": old_decision,
        "new_decision": new_decision,
        "override": req.decision,
        "restitched": True,
        "restitchMs": restitch_ms,
        "masterSha256": master_sha256,
    }


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
    session_ttl_s: float = DEFAULT_SESSION_TTL_S,
    session_clock: Callable[[], float] = time.monotonic,
    trusted_origins: frozenset[str] = TRUSTED_APP_ORIGINS,
    max_concurrent_analyses: int = DEFAULT_MAX_CONCURRENT_ANALYSES,
    native_source_registry: NativeSourceRegistry | None = None,
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
    if max_concurrent_analyses < 1 or max_concurrent_analyses > 32:
        raise ValueError("max_concurrent_analyses must be between 1 and 32")
    if not trusted_origins or any(
        origin in {"*", "null"} or "\r" in origin or "\n" in origin for origin in trusted_origins
    ):
        raise ValueError("trusted_origins must be explicit non-null origins")
    manager = job_manager if job_manager is not None else JobManager()
    sessions = SessionRegistry(session_ttl_s, clock=session_clock)
    native_sources = native_source_registry or NativeSourceRegistry()
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
        # Interactive V1 studio sessions retain uploaded sources under
        # UploadStore's bounded TTL/quota/scavenge lifecycle so post-Process
        # Preview (P), Spectrogram, Multi-Take, A/B Original playback, and
        # Video Mux stay alive for the session. Legacy /api/jobs retains
        # immediate post-terminal upload cleanup.
        if record.retain_input_on_terminal:
            return
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

    health_signals = HealthSignalsService(
        job_manager=manager,
        upload_store=upload_store,
    )

    delivery_exports = _DeliveryOutputRegistry()

    app = FastAPI(title="HawaVoClean engine", version=__version__, lifespan=lifespan)
    app.state.job_manager = manager
    app.state.health_signals = health_signals
    app.state.token = token
    app.state.ui_dir = ui_dir
    app.state.max_upload_bytes = upload_limit
    app.state.upload_store = upload_store
    app.state.upload_lock = asyncio.Lock()
    app.state.sessions = sessions
    app.state.native_sources = native_sources
    app.state.delivery_exports = delivery_exports
    app.state.analysis_slots = asyncio.Semaphore(max_concurrent_analyses)
    app.state.preview_slots = asyncio.Semaphore(2)
    app.state.assistant_slots = asyncio.Semaphore(1)
    app.state.legacy_telemetry = {
        "total_invocations": 0,
        "routes": {},
        "auth_kinds": {},
        "sunset_date": LEGACY_SUNSET_ISO_DATE,
        "sunset_http_date": LEGACY_SUNSET_DATE,
        "removal_release": LEGACY_REMOVAL_RELEASE,
    }

    def _resolve_read_path(request: Request, raw: str) -> Path:
        """Apply the renderer capability boundary to one legacy path read."""

        if _auth_kind(request) == "root":
            # One-release native/CLI compatibility. The root secret never
            # crosses the preload bridge or browser cookie boundary.
            return resolve_client_path(raw, must_exist=True)

        native = native_sources.resolve_registered_path(raw)
        if native is not None:
            return native

        # Durable outputs may live beside a root-registered source on a
        # secondary drive, outside the legacy home/work allowlist. Match the
        # exact public name before applying that old path policy, then return
        # only the job-bound immutable generation.
        if not raw or not raw.strip():
            raise PathPolicyError(400, "bad_request", "path is required")
        refuse_unusable_filename_text(raw)
        lexical = Path(raw)
        if not lexical.is_absolute():
            raise PathPolicyError(400, "bad_request", "path must be absolute")
        requested = public_output_path(raw)
        match = _requested_job_artifact(manager, requested)
        if match is not None:
            snapshot, kind = match
            try:
                artifact = _job_artifact_path(snapshot, kind)
            except (HawaVoCleanError, OSError, ValueError, KeyError):
                artifact = None
            if artifact is not None:
                return artifact

        delivery_export = delivery_exports.resolve_authorized_path(raw)
        if delivery_export is not None:
            return delivery_export

        resolved = resolve_client_path(raw, must_exist=True)
        if (
            upload_store.authorizes(resolved)
            or _is_managed_preview_file(raw, resolved)
            or _is_managed_extracted_track_file(raw, resolved)
        ):
            return resolved
        raise PathPolicyError(
            403,
            "forbidden",
            "renderer session has no capability for this file",
        )

    # The last middleware added is outermost. Boundary security therefore
    # rejects hostile Host/Origin/auth before upload parsing or route code.
    app.add_middleware(UploadSizeLimitMiddleware, max_bytes=upload_limit)
    app.add_middleware(
        LocalSecurityMiddleware,
        token=token,
        sessions=sessions,
        trusted_origins=trusted_origins,
    )

    # --- error shape ---------------------------------------------------------
    def _opaque_internal_error(
        req: Request, exc: BaseException, code: str | None = None
    ) -> JSONResponse:
        request_id = _request_id(req.scope)
        logger.error(
            "Engine API failure request_id=%s type=%s code=%s",
            request_id,
            type(exc).__name__,
            code or "unhandled",
        )
        return error_response(
            500,
            "internal_error",
            "The engine could not complete the request. Use the request ID for diagnostics.",
            request_id=request_id,
        )

    @app.exception_handler(ApiError)
    async def _api_error(req: Request, exc: ApiError) -> JSONResponse:
        if exc.status == 500:
            return _opaque_internal_error(req, exc, exc.code)
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(PathPolicyError)
    async def _policy_error(_req: Request, exc: PathPolicyError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(req: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 500:
            return _opaque_internal_error(req, exc, "http_error")
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
    async def _engine_error(req: Request, exc: HawaVoCleanError) -> JSONResponse:
        status = 400 if isinstance(exc, InvalidUserInputError) else 500
        if status == 500:
            return _opaque_internal_error(req, exc, exc.exit_code.name)
        return error_response(status, exc.exit_code.name.lower(), exc.message)

    @app.exception_handler(Exception)
    async def _unhandled(req: Request, exc: Exception) -> JSONResponse:
        # Do not put arbitrary exception messages or tracebacks in broker logs:
        # they may contain paths, tokens, model inputs, or third-party detail.
        return _opaque_internal_error(req, exc)

    # --- routes ---------------------------------------------------------------
    @app.post(SESSION_PATH)
    async def create_session() -> JSONResponse:
        session_token, max_age = sessions.issue()
        response = JSONResponse(
            {
                "sessionToken": session_token,
                "expiresInSeconds": max_age,
                "tokenType": "Bearer",
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )
        response.set_cookie(
            SESSION_COOKIE,
            session_token,
            max_age=max_age,
            httponly=True,
            samesite="strict",
            path="/api",
        )
        return response

    @app.post(NATIVE_SOURCE_PATH)
    async def register_native_source(request: Request, req: NativeSourceRequest) -> JSONResponse:
        """Bind one OS/Resolve selection to an opaque source capability.

        This route is deliberately root-only. A renderer bearer/cookie cannot
        turn a guessed path into authority by calling the registration API.
        """

        if _auth_kind(request) != "root":
            raise ApiError(403, "forbidden", "native source registration requires root authority")
        source = native_sources.register(req.path)
        return JSONResponse(
            {"sourceId": source.source_id, "path": str(source.path)},
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

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
            # Loose profile.json files prove only that research identities are
            # present. Product availability comes from the signed capability
            # contract and remains false until a qualified Restore pack exists.
            "restore_available": False,
            "engine_pid": os.getpid(),
            "storage": {
                "managed_upload_bytes": upload_store.usage_bytes(),
                "managed_upload_limit_bytes": upload_store.max_total_bytes,
                "minimum_free_bytes": upload_store.min_free_bytes,
            },
            "jobs": {
                "durable": manager.durable,
                "persistence_ok": manager.persistence_error is None,
                "persistence_error": manager.persistence_error,
            },
            "legacy_sunset": {
                "sunset_date": LEGACY_SUNSET_ISO_DATE,
                "sunset_http_date": LEGACY_SUNSET_DATE,
                "removal_release": LEGACY_REMOVAL_RELEASE,
                "total_invocations": int(
                    getattr(app.state, "legacy_telemetry", {}).get("total_invocations", 0)
                ),
            },
        }

    @app.get("/api/v1/health/signals")
    async def health_signals_endpoint(_request: Request) -> dict[str, Any]:
        return health_signals.evaluate_all()

    @app.get("/api/v1/telemetry/legacy")
    async def legacy_telemetry(_request: Request) -> dict[str, Any]:
        telemetry = getattr(app.state, "legacy_telemetry", {})
        return {
            "schema_version": 1,
            "sunset_date": LEGACY_SUNSET_ISO_DATE,
            "sunset_http_date": LEGACY_SUNSET_DATE,
            "removal_release": LEGACY_REMOVAL_RELEASE,
            "total_invocations": int(telemetry.get("total_invocations", 0)),
            "routes": dict(telemetry.get("routes", {})),
            "auth_kinds": dict(telemetry.get("auth_kinds", {})),
            "legacy_endpoints": [
                "/api/jobs",
                "/api/jobs/{job_id}",
                "/api/jobs/{job_id}/cancel",
                "/api/jobs/{job_id}/events",
                "/api/analyze",
                "/api/audio",
            ],
            "v1_successors": {
                "/api/jobs": "/api/v1/jobs",
                "/api/jobs/{job_id}": "/api/v1/jobs/{job_id}",
                "/api/jobs/{job_id}/cancel": "/api/v1/jobs/{job_id}/cancel",
                "/api/jobs/{job_id}/events": "/api/v1/jobs/{job_id}/events",
                "/api/analyze": "/api/v1/analyze",
                "/api/audio": "/api/v1/jobs/{job_id}/artifacts/master",
            },
        }

    @app.post("/api/analyze")
    async def analyze(request: Request, req: AnalyzeRequest) -> dict[str, Any]:
        path = _resolve_read_path(request, req.path)
        # Legacy clients share the same bounded worker budget as the v1
        # analyzer.  Without this, a renderer could bypass the v1 semaphore
        # simply by issuing many revision-1 waveform requests and create an
        # unbounded collection of decoder/FFT threads.
        async with app.state.analysis_slots:
            result = await asyncio.to_thread(analyze_audio, path, req.buckets)
        pub = public_output_path(req.path)
        if _requested_job_artifact(manager, pub) is not None:
            result = dict(result)
            result["path"] = str(pub)
        return result

    @app.post("/api/v1/analyze")
    async def analyze_v1(req: SmartAnalysisRequestV1) -> dict[str, Any]:
        # The analyzer is fixed-state, but decoders and FFT work are not free.
        # Bound concurrent whole-file reductions so a batch cannot turn the
        # desktop broker into an unbounded thread/process fan-out.
        async with app.state.analysis_slots:
            result = await asyncio.to_thread(
                _analyze_smart_source, upload_store, native_sources, req.source_id
            )
        return result.model_dump(by_alias=True, mode="json", exclude_none=True)

    @app.get("/api/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return capabilities_v1().model_dump(by_alias=True, mode="json", exclude_none=True)

    @app.get("/api/v1/jobs")
    async def list_v1_jobs() -> dict[str, Any]:
        jobs = [
            _job_status_v1(snapshot).model_dump(by_alias=True, mode="json", exclude_none=True)
            for snapshot in manager.list_jobs()
        ]
        return {"schemaVersion": 1, "jobs": jobs}

    @app.get("/api/v1/jobs/{job_id}")
    async def get_v1_job(job_id: str) -> dict[str, Any]:
        snapshot = manager.get_status(job_id)
        if snapshot is None:
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        return _job_status_v1(snapshot).model_dump(by_alias=True, mode="json", exclude_none=True)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel_v1_job(job_id: str) -> dict[str, Any]:
        if not manager.cancel(job_id):
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        snapshot = manager.get_status(job_id)
        if snapshot is None:  # pragma: no cover - manager retains the just-cancelled record
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        return _job_status_v1(snapshot).model_dump(by_alias=True, mode="json", exclude_none=True)

    @app.post("/api/v1/jobs/{job_id}/retry")
    async def retry_v1_job(job_id: str) -> dict[str, Any]:
        try:
            snapshot = manager.retry_job(job_id)
        except KeyError as exc:
            raise ApiError(404, "not_found", f"unknown job: {job_id}") from exc
        except OutputConflictError as exc:
            raise ApiError(409, "conflict", str(exc)) from exc
        except QueueFullError as exc:
            raise ApiError(503, "queue_full", str(exc)) from exc
        except MediaPreflightError as exc:
            raise ApiError(400, exc.reason.value, str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(503, "unavailable", str(exc)) from exc
        return _job_status_v1(snapshot).model_dump(by_alias=True, mode="json", exclude_none=True)

    @app.get("/api/v1/batches")
    async def list_v1_batches(limit: int = 50) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "batches": manager.list_batches(limit=limit),
        }

    @app.get("/api/v1/batches/{batch_id}")
    async def get_v1_batch(batch_id: str) -> dict[str, Any]:
        summary = manager.get_batch_summary(batch_id)
        if summary is None:
            raise ApiError(404, "not_found", f"unknown batch: {batch_id}")
        return summary

    @app.post("/api/v1/batches/{batch_id}/pause")
    async def pause_v1_batch(batch_id: str) -> dict[str, Any]:
        if not manager.pause_batch(batch_id):
            raise ApiError(404, "not_found", f"unknown batch: {batch_id}")
        return {"ok": True, "batchId": batch_id, "state": "paused"}

    @app.post("/api/v1/batches/{batch_id}/resume")
    async def resume_v1_batch(batch_id: str) -> dict[str, Any]:
        if not manager.resume_batch(batch_id):
            raise ApiError(404, "not_found", f"unknown batch: {batch_id}")
        return {"ok": True, "batchId": batch_id, "state": "running"}

    @app.post("/api/v1/batches/{batch_id}/cancel")
    async def cancel_v1_batch(batch_id: str, wait: bool = False) -> dict[str, Any]:
        if not manager.cancel_batch(batch_id, wait=wait):
            raise ApiError(404, "not_found", f"unknown batch: {batch_id}")
        return {"ok": True, "batchId": batch_id, "state": "cancelled"}

    @app.get("/api/v1/batches/{batch_id}/events")
    async def batch_events(batch_id: str) -> StreamingResponse:
        summary = manager.get_batch_summary(batch_id)
        if summary is None:
            raise ApiError(404, "not_found", f"unknown batch: {batch_id}")

        async def stream() -> AsyncIterator[str]:
            last_json = ""
            while True:
                current = manager.get_batch_summary(batch_id)
                if current is None:
                    yield _sse("end", {})
                    return
                current_json = json.dumps(current, sort_keys=True)
                if current_json != last_json:
                    yield _sse("batch_status", current)
                    last_json = current_json
                if current["state"] in {"done", "failed", "cancelled"}:
                    yield _sse("end", {})
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, private",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/v1/preview")
    async def preview_region_v1(request: Request, req: RegionPreviewRequestV1) -> dict[str, Any]:
        """Render a sub-second windowed enhancement preview using the persistent warm core cache."""
        source_path = _resolve_read_path(request, req.path)
        preview_req = RegionPreviewRequest(
            source_path=source_path,
            start_s=req.start_s,
            end_s=req.end_s,
            profile=req.profile,
            strength=req.strength,
            ambience_floor_db=req.ambience_floor_db,
            include_perceptual_mos=req.include_perceptual_mos,
        )
        try:
            async with app.state.preview_slots:
                result = await asyncio.to_thread(render_region_preview, preview_req)
        except ValueError as exc:
            raise ApiError(400, "bad_request", _sanitize_api_error(str(exc), source_path)) from exc
        except FileNotFoundError as exc:
            raise ApiError(404, "not_found", _sanitize_api_error(str(exc), source_path)) from exc
        except MediaPreflightError as exc:
            raise ApiError(
                400, exc.reason.value, _sanitize_api_error(str(exc), source_path)
            ) from exc
        except sf.SoundFileError as exc:
            raise ApiError(400, "bad_request", "audio file could not be decoded") from exc

        metrics_before: dict[str, Any] = {
            "lufs": result.before_lufs,
            "noiseFloorDb": result.before_noise_floor_db,
        }
        metrics_after: dict[str, Any] = {
            "lufs": result.after_lufs,
            "noiseFloorDb": result.after_noise_floor_db,
        }
        if result.before_mos is not None:
            metrics_before["mos"] = result.before_mos
        if result.after_mos is not None:
            metrics_after["mos"] = result.after_mos

        preview_path_str = str(result.preview_path)
        audio_url = f"/api/audio?{urlencode({'path': preview_path_str})}"
        return {
            "previewPath": preview_path_str,
            "audioUrl": audio_url,
            "renderMs": result.render_ms,
            "warmHit": result.warm_hit,
            "metricsBefore": metrics_before,
            "metricsAfter": metrics_after,
            "durationS": result.duration_s,
        }

    @app.post("/api/v1/spectrogram")
    async def spectrogram_tile_v1(request: Request, req: SpectrogramRequestV1) -> dict[str, Any]:
        """Compute a 2D STFT or energy-removed delta spectrogram tile over ``[startS, endS]``."""
        orig_path = _resolve_read_path(request, req.path)
        cleaned_path = (
            _resolve_read_path(request, req.cleaned_path) if req.cleaned_path is not None else None
        )
        try:
            async with app.state.analysis_slots:
                if req.mode == "delta":
                    assert cleaned_path is not None
                    db_floor = req.db_floor if req.db_floor is not None else DELTA_DEFAULT_DB_FLOOR
                    db_ceil = req.db_ceil if req.db_ceil is not None else DELTA_DEFAULT_DB_CEIL
                    tile = await asyncio.to_thread(
                        compute_delta_spectrogram_tile,
                        orig_path,
                        cleaned_path,
                        start_s=req.start_s,
                        end_s=req.end_s,
                        width_cols=req.width_cols,
                        n_bins=req.n_bins,
                        n_fft=req.n_fft,
                        scale=req.scale,
                        db_floor=db_floor,
                        db_ceil=db_ceil,
                    )
                else:
                    target_path = (
                        cleaned_path if (req.mode == "cleaned" and cleaned_path) else orig_path
                    )
                    db_floor = req.db_floor if req.db_floor is not None else -90.0
                    db_ceil = req.db_ceil if req.db_ceil is not None else 0.0
                    tile = await asyncio.to_thread(
                        compute_spectrogram_tile,
                        target_path,
                        start_s=req.start_s,
                        end_s=req.end_s,
                        width_cols=req.width_cols,
                        n_bins=req.n_bins,
                        n_fft=req.n_fft,
                        scale=req.scale,
                        db_floor=db_floor,
                        db_ceil=db_ceil,
                    )
        except ValueError as exc:
            raise ApiError(
                400, "bad_request", _sanitize_api_error(str(exc), orig_path, cleaned_path)
            ) from exc
        except FileNotFoundError as exc:
            raise ApiError(
                404, "not_found", _sanitize_api_error(str(exc), orig_path, cleaned_path)
            ) from exc
        except MediaPreflightError as exc:
            raise ApiError(
                400, exc.reason.value, _sanitize_api_error(str(exc), orig_path, cleaned_path)
            ) from exc
        except sf.SoundFileError as exc:
            raise ApiError(400, "bad_request", "audio file could not be decoded") from exc
        return tile.to_dict()

    @app.post("/api/v1/spectral-repair")
    async def spectral_repair_v1(request: Request, req: SpectralRepairRequestV1) -> dict[str, Any]:
        """Apply 2D time×frequency surgical STFT attenuation and serve via managed ``/api/audio``."""
        source_path = _resolve_read_path(request, req.path)
        try:
            async with app.state.preview_slots:
                return await asyncio.to_thread(_run_spectral_box_repair, source_path, req)
        except ValueError as exc:
            raise ApiError(400, "bad_request", _sanitize_api_error(str(exc), source_path)) from exc
        except FileNotFoundError as exc:
            raise ApiError(404, "not_found", _sanitize_api_error(str(exc), source_path)) from exc
        except MediaPreflightError as exc:
            raise ApiError(
                400, exc.reason.value, _sanitize_api_error(str(exc), source_path)
            ) from exc
        except sf.SoundFileError as exc:
            raise ApiError(400, "bad_request", "audio file could not be decoded") from exc

    @app.post("/api/v1/export")
    async def export_delivery_v1(request: Request, req: DeliveryExportRequestV1) -> dict[str, Any]:
        """Export a master WAV to broadcast/streaming loudness and target delivery format."""
        master_resolved = _resolve_read_path(request, req.master_path)
        source_resolved = (
            _resolve_read_path(request, req.source_media_path)
            if req.source_media_path is not None
            else None
        )
        effective_preset: DeliveryPreset = req.delivery_preset or "match_input"
        if req.dest_path is not None:
            dest_resolved = resolve_client_output_path(req.dest_path)
            _reject_export_destination_symlink(req.dest_path, dest_resolved)
            if _auth_kind(request) != "root":
                exports_root = (work_root() / "exports").resolve()
                # Never authorize writing into an internal immutable generation directory;
                # only allow the public output directory, source directory, or managed exports root.
                allowed_parents = {public_output_path(req.master_path).parent}
                if source_resolved is not None:
                    allowed_parents.add(source_resolved.parent)
                if dest_resolved.parent not in allowed_parents and not (
                    dest_resolved.parent == exports_root
                    or dest_resolved.parent.is_relative_to(exports_root)
                ):
                    raise PathPolicyError(
                        403,
                        "forbidden",
                        "renderer export destination must remain beside the master/source or under exports",
                    )
        else:
            dest_resolved = _default_export_dest_path(
                req.master_path,
                master_resolved,
                req.format,
                effective_preset,
                source_resolved,
            )
            _reject_export_destination_symlink(str(dest_resolved), dest_resolved)

        allowed_exts = EXPORT_FORMAT_EXTENSIONS[req.format]
        if dest_resolved.suffix.lower() not in allowed_exts:
            raise ApiError(
                400,
                "bad_request",
                f"export destination extension {dest_resolved.suffix!r} is not valid for "
                f"format {req.format!r}; expected one of {sorted(allowed_exts)}",
            )
        if os.path.lexists(dest_resolved):
            try:
                dest_stat = os.lstat(dest_resolved)
            except OSError as exc:
                raise ApiError(400, "bad_request", "export destination is not accessible") from exc
            if not stat.S_ISREG(dest_stat.st_mode):
                raise ApiError(
                    400,
                    "bad_request",
                    "export destination exists and is not a regular file",
                )

        try:
            async with app.state.analysis_slots:
                result = await asyncio.to_thread(
                    _run_delivery_export,
                    master_resolved=master_resolved,
                    dest_resolved=dest_resolved,
                    fmt=req.format,
                    delivery_preset=effective_preset,
                    source_resolved=source_resolved,
                    preserve_metadata=req.preserve_metadata,
                )
        except ValueError as exc:
            raise ApiError(
                400,
                "bad_request",
                _sanitize_api_error(str(exc), master_resolved, dest_resolved, source_resolved),
            ) from exc
        except FileNotFoundError as exc:
            raise ApiError(
                404,
                "not_found",
                _sanitize_api_error(str(exc), master_resolved, dest_resolved, source_resolved),
            ) from exc
        except MediaPreflightError as exc:
            raise ApiError(
                400,
                exc.reason.value,
                _sanitize_api_error(str(exc), master_resolved, dest_resolved, source_resolved),
            ) from exc
        except sf.SoundFileError as exc:
            raise ApiError(400, "bad_request", "audio file could not be decoded") from exc
        except PreflightError as exc:
            raise ApiError(
                503,
                "unavailable",
                _sanitize_api_error(str(exc), master_resolved, dest_resolved, source_resolved),
            ) from exc
        except OutputValidationError as exc:
            raise ApiError(
                422,
                "export_failed",
                _sanitize_api_error(str(exc), master_resolved, dest_resolved, source_resolved),
            ) from exc

        delivery_exports.record(Path(str(result["outputPath"])))
        return result

    @app.post("/api/v1/tracks/inspect")
    async def inspect_tracks_v1(request: Request, req: TrackInspectRequestV1) -> dict[str, Any]:
        """Inspect tracks inside a 1-16 channel Poly-WAV/BWF file."""
        resolved = _resolve_read_path(request, req.path)
        try:
            async with app.state.analysis_slots:
                tracks = await asyncio.to_thread(inspect_polywav_tracks, resolved)
        except ValueError as exc:
            raise ApiError(400, "bad_request", _sanitize_api_error(str(exc), resolved)) from exc
        except FileNotFoundError as exc:
            raise ApiError(404, "not_found", _sanitize_api_error(str(exc), resolved)) from exc
        except sf.SoundFileError as exc:
            raise ApiError(400, "bad_request", "audio file could not be decoded") from exc
        return {
            "path": str(resolved),
            "trackCount": len(tracks),
            "tracks": [track.to_dict() for track in tracks],
        }

    @app.post("/api/v1/tracks/extract")
    async def extract_tracks_v1(request: Request, req: TrackExtractRequestV1) -> dict[str, Any]:
        """Stream-extract selected 0-based tracks from a Poly-WAV/BWF file into mono WAVs."""
        resolved = _resolve_read_path(request, req.path)
        extracted_root = (work_root() / "extracted_tracks").resolve()
        if req.out_dir is not None:
            target_dir = resolve_client_path(req.out_dir, must_exist=False)
            _reject_export_destination_symlink(req.out_dir, target_dir)
            if os.path.lexists(target_dir) and not target_dir.is_dir():
                raise ApiError(
                    400,
                    "bad_request",
                    "extract outDir exists and is not a directory",
                )
            if _auth_kind(request) != "root" and not (
                target_dir == resolved.parent
                or target_dir.is_relative_to(resolved.parent)
                or target_dir == extracted_root
                or target_dir.is_relative_to(extracted_root)
            ):
                raise PathPolicyError(
                    403,
                    "forbidden",
                    "renderer track extraction outDir must remain beside the source or under extracted_tracks",
                )
        else:
            target_dir = extracted_root / f"{resolved.stem}_{secrets.token_hex(4)}"

        try:
            async with app.state.analysis_slots:
                extracted = await asyncio.to_thread(
                    extract_polywav_tracks, resolved, req.track_indices, target_dir
                )
        except ValueError as exc:
            raise ApiError(
                400, "bad_request", _sanitize_api_error(str(exc), resolved, target_dir)
            ) from exc
        except FileNotFoundError as exc:
            raise ApiError(
                404, "not_found", _sanitize_api_error(str(exc), resolved, target_dir)
            ) from exc
        except sf.SoundFileError as exc:
            raise ApiError(400, "bad_request", "audio file could not be decoded") from exc

        if req.out_dir is None or (
            target_dir != extracted_root and target_dir.parent == extracted_root
        ):
            cleanup_extracted_tracks_dir(extracted_root, protect=target_dir)

        source_ids = [native_sources.register(str(p)).source_id for p in extracted]
        return {
            "path": str(resolved),
            "outDir": str(target_dir),
            "extractedPaths": [str(p) for p in extracted],
            "sourceIds": source_ids,
        }

    def _apply_unit_override_sync(job_id: str, req: UnitOverrideRequestV1) -> dict[str, Any]:
        with manager.override_lock(job_id):
            snapshot = manager.get_status(job_id)
            if snapshot is None:
                raise ApiError(404, "not_found", f"unknown job: {job_id}")
            state = snapshot.get("state")
            if state not in {"done", "completed"}:
                raise ApiError(409, "invalid_state", f"job {job_id} is {state}, not completed")
            if bool(snapshot.get("record_bundle", False)):
                raise ApiError(
                    409,
                    "sealed_record_bundle",
                    "sealed processing records cannot be mutated in-place; "
                    "run without record bundle to edit units",
                )
            report = snapshot.get("report")
            report_file: Path | None = None
            if report is None:
                raw_path = snapshot.get("report_path")
                if raw_path:
                    p = Path(raw_path)
                    if p.is_file():
                        try:
                            report = json.loads(p.read_text())
                            report_file = p
                        except Exception as e:
                            logger.warning(f"Could not load report from {p}: {e}")
            if report is None or not isinstance(report, dict):
                raise ApiError(409, "invalid_state", f"job {job_id} has no report")

            units = report.get("units", [])
            if not isinstance(units, list):
                raise ApiError(409, "invalid_state", f"job {job_id} report has no units list")

            target_idx: int | None = None
            if req.unit_index is not None:
                if req.unit_index >= len(units):
                    raise ApiError(
                        400,
                        "invalid_unit",
                        f"unit_index {req.unit_index} out of range (0..{len(units) - 1})",
                    )
                target_idx = req.unit_index
            elif req.unit_id is not None:
                for idx_u, candidate_u in enumerate(units):
                    if isinstance(candidate_u, dict) and candidate_u.get("unit_id") == req.unit_id:
                        target_idx = idx_u
                        break
                if target_idx is None:
                    raise ApiError(
                        400,
                        "invalid_unit",
                        f"unit_id {req.unit_id} not found in job {job_id}",
                    )
            assert target_idx is not None

            candidate_dir: Path | None = None
            with contextlib.suppress(ValueError):
                candidate_dir = UnitSliceStore(work_root()).job_dir(job_id)
            if candidate_dir is not None and read_slice_truncated_marker(candidate_dir) is not None:
                raise ApiError(
                    409,
                    "slice_cache_budget_exceeded",
                    "unit slice cache exceeded disk budget; live re-stitching is unavailable for this job",
                )
            job_slice_dir: Path | None = (
                candidate_dir
                if candidate_dir is not None and _has_complete_unit_slices(candidate_dir, units)
                else None
            )

            if job_slice_dir is not None:
                try:
                    result = _live_restitch_unit_override(
                        job_id=job_id,
                        snapshot=snapshot,
                        report=report,
                        units=units,
                        target_idx=target_idx,
                        req=req,
                        job_slice_dir=job_slice_dir,
                        manager=manager,
                    )
                except SliceCacheBudgetError as exc:
                    raise ApiError(
                        409,
                        "slice_cache_budget_exceeded",
                        _sanitize_api_error(str(exc), job_slice_dir),
                    ) from exc
                except ValueError as exc:
                    raise ApiError(
                        400,
                        "invalid_override",
                        _sanitize_api_error(str(exc), job_slice_dir),
                    ) from exc
                except FileNotFoundError as exc:
                    raise ApiError(
                        409,
                        "slice_cache_unavailable",
                        "cached unit slice is missing",
                    ) from exc
                logger.info(
                    "Unit %d of job %s live-restitched: %s -> %s (override=%s, %.1f ms)",
                    target_idx,
                    job_id,
                    result["old_decision"],
                    result["new_decision"],
                    req.decision,
                    result["restitchMs"],
                )
                return result

            updated_report = copy.deepcopy(report)
            updated_units = updated_report.get("units", [])
            assert isinstance(updated_units, list)
            unit = updated_units[target_idx]
            unit_id_val = int(unit.get("unit_id", target_idx))
            old_decision = unit.get("final_decision", "unknown")

            if req.decision == "auto":
                # Restore the automatic guard decision (remove override)
                unit.pop("manual_override", None)
                # Restore original decision from guard verdict
                verdict = unit.get("guard_a_verdict", "FAIL")
                if verdict == "PASS":
                    unit["final_decision"] = "enhanced"
                    unit["decision_reason"] = "guard PASS (auto restored)"
                else:
                    unit["final_decision"] = "reverted"
                    unit["decision_reason"] = f"guard {verdict} (auto restored)"
            elif req.decision == "force_original":
                unit["final_decision"] = "reverted"
                unit["manual_override"] = "force_original"
                unit["decision_reason"] = f"manual override: force_original (was {old_decision})"
            elif req.decision == "force_enhanced":
                unit["final_decision"] = "enhanced"
                unit["manual_override"] = "force_enhanced"
                unit["decision_reason"] = f"manual override: force_enhanced (was {old_decision})"
            elif req.decision == "custom_strength":
                assert req.strength is not None
                unit["chosen_strength"] = round(float(req.strength), 4)
                unit["final_decision"] = "reverted" if req.strength == 0.0 else "enhanced"
                unit["manual_override"] = f"custom_strength:{req.strength:.3f}"
                unit["decision_reason"] = (
                    f"manual override: custom_strength={req.strength} (was {old_decision})"
                )

            # Update the summary counts
            enhanced_count = sum(1 for u in updated_units if u.get("final_decision") == "enhanced")
            reverted_count = sum(1 for u in updated_units if u.get("final_decision") == "reverted")
            if "summary" in updated_report and isinstance(updated_report["summary"], dict):
                updated_report["summary"]["enhanced"] = enhanced_count
                updated_report["summary"]["reverted"] = reverted_count

            if report_file is not None:
                try:
                    _atomic_write_json(report_file, updated_report)
                except Exception as e:
                    logger.warning(f"Could not persist updated report to {report_file}: {e}")

            manager.update_completed_override(job_id, updated_report, recapture_artifacts=False)

            logger.info(
                "Unit %d of job %s: %s → %s (override=%s)",
                target_idx,
                job_id,
                old_decision,
                unit["final_decision"],
                req.decision,
            )

            return {
                "job_id": job_id,
                "unit_index": target_idx,
                "unit_id": unit_id_val,
                "old_decision": old_decision,
                "new_decision": unit["final_decision"],
                "override": req.decision,
                "restitched": False,
            }

    @app.post("/api/v1/jobs/{job_id}/override")
    async def override_v1_unit(job_id: str, req: UnitOverrideRequestV1) -> dict[str, Any]:
        """Manually override a guard decision on a single unit.

        When cached per-unit slices exist for ``job_id``, atomically re-stitches
        the actual master WAV in <100 ms and updates the report and immutable
        artifact hashes. When slices are absent (legacy or mock jobs), falls back
        to metadata-only report override with ``restitched: false``.
        """
        return await asyncio.to_thread(_apply_unit_override_sync, job_id, req)

    @app.get("/api/v1/assistant/status")
    async def audio_assistant_status() -> dict[str, Any]:
        from hawavoclean.assistant import assistant_status

        return assistant_status()

    # Planning only: no audio bytes, source paths, or processing authority are
    # passed to a language model. Execution still uses the closed job contract.
    from hawavoclean.assistant import AudioPlanRequest

    @app.post("/api/v1/assistant/plan")
    async def audio_assistant_plan(req: AudioPlanRequest) -> dict[str, Any]:
        from hawavoclean.assistant import plan_audio

        try:
            async with app.state.assistant_slots:
                plan = await asyncio.to_thread(plan_audio, req)
        except (ValueError, RuntimeError) as exc:
            raise ApiError(503, "assistant_unavailable", str(exc)) from exc
        return plan.model_dump(mode="json")

    @app.post("/api/v1/jobs", status_code=202)
    async def create_v1_jobs(req: ProcessingRequestV1) -> dict[str, Any]:
        """Versioned, idempotent batch submission for currently qualified local routes."""

        if req.execution_policy == "cloud_allowed":
            raise ApiError(503, "capability_blocked", "cloud execution is not deployed")
        separation_options: dict[str, Any] | None = None
        submit_cutoff_hz: float | None = None
        submit_strength: float | None = None
        submit_time_range: tuple[float, float] | None = None
        submit_ambience_floor_db: float | None = None
        submit_delivery_preset: str | None = None
        submit_pro_repair: dict[str, bool] | None = None
        if isinstance(req.strategy, SmartSafeStrategyV1):
            smart_cap = next(
                (c for c in capabilities_v1().capabilities if c.capability_id == "smart_safe"),
                None,
            )
            if smart_cap is None or not smart_cap.available or smart_cap.maturity == "blocked":
                raise ApiError(
                    503,
                    "capability_blocked",
                    smart_cap.reason
                    if smart_cap and smart_cap.reason
                    else "Smart Safe is unavailable",
                )
            submit_mode = "smart_safe"
            submit_profile = "production"
            submit_speaker_id = req.strategy.speaker_profile_id
            output_profile = "smart_safe"
        elif isinstance(req.strategy, SeparationStrategyV1):
            from hawavoclean.separation import probe_runtime

            status = await asyncio.to_thread(
                probe_runtime,
                model_size=req.strategy.model_size,
                device=req.strategy.device,
                prompt=req.strategy.prompt,
                operation=req.strategy.operation,
            )
            if not status.available:
                raise ApiError(503, "capability_blocked", status.reason)
            if req.strategy.cleanup_profile is not None:
                cleanup = _natural_route_capability(req.strategy.cleanup_profile)
                if not cleanup.available:
                    raise ApiError(
                        503, "capability_blocked", cleanup.reason or "Cleanup unavailable"
                    )
            separation_options = req.strategy.model_dump(mode="json")
            submit_mode = "separate"
            submit_profile = req.strategy.cleanup_profile or "production"
            submit_speaker_id = None
            output_profile = "extract" if req.strategy.operation == "extract" else "remove"
        elif isinstance(req.strategy, ManualStrategyV1):
            if req.strategy.route == "restore_source":
                restore_cap = _restore_source_capability()
                if not restore_cap.available or restore_cap.maturity == "blocked":
                    raise ApiError(
                        503,
                        "capability_blocked",
                        restore_cap.reason or "restore_source capability is unavailable",
                    )
                clean_profile = "production"
                for candidate_profile in ("studio_pro", "studio"):
                    candidate_cap = _natural_route_capability(candidate_profile)
                    if candidate_cap.available and candidate_cap.maturity == "qualified":
                        clean_profile = candidate_profile
                        break
                submit_mode = "restore"
                submit_profile = clean_profile
                submit_speaker_id = None
                submit_cutoff_hz = req.strategy.expert_cutoff_hz
                submit_strength = req.strategy.strength
                submit_time_range = req.strategy.time_range
                submit_ambience_floor_db = req.strategy.ambience_floor_db
                submit_delivery_preset = req.strategy.delivery_preset
                submit_pro_repair = req.strategy.pro_repair
                output_profile = "restore_source"
            elif req.strategy.route in {
                "production",
                "studio",
                "studio_pro",
                "dereverb",
                "lowband",
            }:
                route_capability = _natural_route_capability(req.strategy.route)
                if not route_capability.available or route_capability.maturity != "qualified":
                    reason = (
                        route_capability.reason
                        if route_capability.reason
                        else f"route {req.strategy.route!r} is not runnable"
                    )
                    raise ApiError(503, "capability_blocked", reason)
                submit_mode = "natural"
                submit_profile = req.strategy.route
                submit_speaker_id = req.strategy.speaker_profile_id
                submit_strength = req.strategy.strength
                submit_time_range = req.strategy.time_range
                submit_ambience_floor_db = req.strategy.ambience_floor_db
                submit_delivery_preset = req.strategy.delivery_preset
                submit_pro_repair = req.strategy.pro_repair
                output_profile = req.strategy.route
            else:
                raise ApiError(
                    503,
                    "capability_blocked",
                    f"route {req.strategy.route!r} is not production-qualified",
                )
        else:
            raise ApiError(
                503,
                "capability_blocked",
                "Unsupported processing strategy",
            )

        wire_request = req.model_dump(by_alias=True, mode="json", exclude_none=True)
        batch_hash = request_hash(wire_request)
        item_keys = [
            "v1-"
            + request_hash(
                {
                    "idempotencyKey": req.idempotency_key,
                    "index": index,
                }
            )
            for index in range(len(req.source_ids))
        ]
        submitted: list[dict[str, Any]] = []
        existing_job_ids = {str(job["job_id"]) for job in manager.list_jobs()}
        newly_submitted: list[str] = []

        def _cancel_newly_submitted() -> None:
            for job_id in newly_submitted:
                manager.cancel(job_id)

        # A source lease closes the gap between opaque-id resolution and the
        # durable job row taking ownership. Without it, a different job's
        # terminal cleanup could delete a shared upload in that window.
        with contextlib.ExitStack() as source_leases:
            sources: list[Path] = []
            for index, source_id in enumerate(req.source_ids):
                prior = manager.get_by_idempotency(item_keys[index])
                if prior is not None:
                    # A response-lost retry must still work after retention has
                    # removed the upload. ``submit`` below binds this prior item
                    # to the full batch hash and rejects a changed request.
                    sources.append(Path(str(prior["input_path"])))
                    continue
                source = source_leases.enter_context(
                    _lease_source_id(upload_store, native_sources, source_id)
                )
                if source is None:
                    raise ApiError(404, "not_found", f"unknown or expired sourceId: {source_id}")
                if separation_options is not None:
                    from hawavoclean.separation import MAX_DURATION_S, MAX_FILE_SIZE_BYTES

                    # Reject unsupported media before source pinning or model load.
                    try:
                        await asyncio.to_thread(
                            probe_audio,
                            source,
                            max_sample_rate=192000,
                            max_duration_s=MAX_DURATION_S,
                            max_channels=2,
                            max_file_size_bytes=MAX_FILE_SIZE_BYTES,
                        )
                    except HawaVoCleanError as exc:
                        raise ApiError(422, "unsupported_media", str(exc)) from exc
                sources.append(source)

            batch_id = f"b_{batch_hash[:16]}"
            try:
                # The scheduling boundary holds every new row out of the live
                # worker queue until the whole batch has reserved its names and
                # committed durably. A late collision therefore cannot return
                # total failure after an earlier item has already published.
                manager.register_batch(batch_id, len(sources), options=wire_request)
                with manager.prepare_batch():
                    for index, source in enumerate(sources):
                        source_id = req.source_ids[index]
                        snap = manager.submit(
                            input_path=source,
                            output_path=default_output_path(source, output_profile),
                            profile=submit_profile,
                            overwrite=req.conflict_policy == "replace",
                            mode=submit_mode,
                            separation_options=separation_options,
                            strength=submit_strength,
                            time_range=submit_time_range,
                            ambience_floor_db=submit_ambience_floor_db,
                            delivery_preset=submit_delivery_preset,
                            pro_repair=submit_pro_repair,
                            speaker_id=submit_speaker_id,
                            cutoff_hz=submit_cutoff_hz,
                            idempotency_key=item_keys[index],
                            conflict_policy=req.conflict_policy,
                            request_context_hash=batch_hash,
                            record_bundle=req.record_bundle,
                            retain_input_on_terminal=True,
                            batch_id=batch_id,
                        )
                        if str(snap["job_id"]) not in existing_job_ids:
                            newly_submitted.append(str(snap["job_id"]))
                        item: dict[str, Any] = {
                            "sourceId": source_id,
                            "jobId": snap["job_id"],
                            "outputPath": snap["output_path"],
                            "reportPath": snap["report_path"],
                            "recordBundle": snap.get("record_bundle", False),
                        }
                        if snap.get("bundle_path") is not None:
                            item["bundlePath"] = snap["bundle_path"]
                        if snap.get("bundle") is not None:
                            item["bundle"] = snap["bundle"]
                        submitted.append(item)
            except (IdempotencyConflictError, OutputConflictError) as exc:
                _cancel_newly_submitted()
                raise ApiError(409, "conflict", str(exc)) from exc
            except QueueFullError as exc:
                _cancel_newly_submitted()
                raise ApiError(503, "queue_full", str(exc)) from exc
            except RuntimeError as exc:
                _cancel_newly_submitted()
                raise ApiError(503, "unavailable", str(exc)) from exc
            except BaseException:
                # ``prepare_batch`` has already removed every newly reserved
                # queued row before releasing the worker scheduling lock.
                _cancel_newly_submitted()
                raise
        return {
            "schemaVersion": 1,
            "batchId": f"b_{batch_hash[:16]}",
            "execution": "local",
            "jobs": submitted,
        }

    @app.post("/api/peaks")
    async def peaks(request: Request, req: PeaksRequest) -> dict[str, Any]:
        path = _resolve_read_path(request, req.path)
        try:
            async with app.state.analysis_slots:
                result = await asyncio.to_thread(
                    compute_peaks_window, path, req.start_s, req.end_s, req.buckets
                )
        except PeaksWindowError as e:
            raise ApiError(400, "bad_request", str(e)) from e
        pub = public_output_path(req.path)
        if _requested_job_artifact(manager, pub) is not None and "path" in result:
            result = dict(result)
            result["path"] = str(pub)
        return result

    @app.post("/api/jobs", status_code=202)
    async def create_job(request: Request, req: JobRequest) -> dict[str, Any]:
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
        if req.mode == "restore":
            # Loose research profiles and the retired HawaRestore checkpoint
            # are not production capabilities.  The versioned capability
            # endpoint already reports Restore as blocked; the one-release
            # legacy adapter must enforce the same truth instead of spawning
            # an unqualified source-checkout path behind the UI's back.
            raise ApiError(
                503,
                "capability_blocked",
                "Legacy Restore is blocked until a qualified signed Sorani Restore pack is installed",
            )
        if req.conflict_policy is not None and req.overwrite and req.conflict_policy != "replace":
            raise ApiError(
                422,
                "bad_request",
                "overwrite=true conflicts with conflict_policy other than replace",
            )
        upload_store.scavenge(manager.active_input_paths())
        input_path = _resolve_read_path(request, req.input_path)
        # Output authority follows the user-facing selected path, not an
        # immutable private generation path returned by the safe reader.
        selected_input_path = (
            input_path if _auth_kind(request) == "root" else public_output_path(req.input_path)
        )
        if req.output_path:
            output_path = (
                resolve_client_output_path(req.output_path)
                if _auth_kind(request) == "root"
                else _session_output_path(selected_input_path, req.profile, req.output_path)
            )
        else:
            output_path = default_output_path(selected_input_path, req.profile)
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
                idempotency_key=req.idempotency_key,
                conflict_policy=req.conflict_policy,
            )
        except (IdempotencyConflictError, OutputConflictError) as e:
            raise ApiError(409, "conflict", str(e)) from e
        except QueueFullError as e:
            raise ApiError(503, "queue_full", str(e)) from e
        except RuntimeError as e:
            raise ApiError(503, "unavailable", str(e)) from e
        return {
            "job_id": snap["job_id"],
            "output_path": snap["output_path"],
            "report_path": snap["report_path"],
        }

    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        jobs = manager.list_jobs()
        return {"jobs": jobs, "count": len(jobs)}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        snap = manager.get_status(job_id)
        if snap is None:
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        return snap

    @app.get("/api/v1/jobs/{job_id}/events")
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
                "Cache-Control": "no-store, no-cache, must-revalidate, private",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        if not manager.cancel(job_id):
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        return {"ok": True}

    @app.api_route(
        "/api/v1/jobs/{job_id}/artifacts/{kind}",
        methods=["GET", "HEAD"],
    )
    async def job_artifact(
        request: Request,
        job_id: str,
        kind: ArtifactKind,
    ) -> Response:
        snapshot = manager.get_status(job_id)
        if snapshot is None:
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        try:
            artifact = _job_artifact_path(snapshot, kind)
        except (HawaVoCleanError, OSError, ValueError, KeyError) as exc:
            raise ApiError(
                409,
                "artifact_unavailable",
                "the completed job artifact failed authoritative verification",
            ) from exc
        if artifact is None:
            raise ApiError(
                409,
                "artifact_unavailable",
                "this job has no verified immutable artifact of that kind",
            )
        return ranged_file_response(
            artifact,
            request.headers.get("range"),
            head_only=request.method == "HEAD",
        )

    @app.api_route("/api/audio", methods=["GET", "HEAD"])
    async def audio(request: Request, path: str = "") -> Response:
        resolved = _resolve_read_path(request, path)
        return ranged_file_response(
            resolved, request.headers.get("range"), head_only=request.method == "HEAD"
        )

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
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
        return JSONResponse(
            {"path": str(dest), "source_id": upload_store.source_id(dest)},
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, private",
                "Pragma": "no-cache",
            },
        )

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

    manager = JobManager(store_path=job_store_path())
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
