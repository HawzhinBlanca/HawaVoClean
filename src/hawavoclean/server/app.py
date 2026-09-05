"""FastAPI application for the HawaVoClean engine bridge.

``create_app(token, ui_dir)`` builds the app; ``run_server(...)`` binds a
loopback socket, prints the single ``{"event":"ready",...}`` line to stdout
and serves with uvicorn. Routes, shapes and rules: ``docs/ui-contract.md``.
"""

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import re
import secrets
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import asdict
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote_plus, urlsplit

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
from hawavoclean.audio.decode import iter_decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.cli import _clean_stem
from hawavoclean.errors import HawaVoCleanError, InvalidUserInputError, MediaPreflightError
from hawavoclean.logging import get_logger
from hawavoclean.natural_contract import load_natural_route_contract
from hawavoclean.paths import config_dir, job_store_path, models_dir, profiles_root, work_root
from hawavoclean.publication import (
    public_output_path,
    resolve_immutable_publication_generation,
)
from hawavoclean.record_bundle import verify_processing_record
from hawavoclean.server.analysis import (
    DEFAULT_BUCKETS,
    MAX_BUCKETS,
    PeaksWindowError,
    analyze_audio,
    compute_peaks_window,
)
from hawavoclean.server.contracts import (
    CapabilitiesResponseV1,
    CapabilityStatusV1,
    JobLifecycleStateV1,
    JobStatusResponseV1,
    ManualStrategyV1,
    ProcessingRequestV1,
    SmartAnalysisRequestV1,
    SmartAnalysisResponseV1,
    SmartSafeStrategyV1,
    UnitOverrideRequestV1,
)
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
from hawavoclean.smart_safe import analyze_audio_stream

logger = get_logger("server")

PROFILES: tuple[str, ...] = ("studio", "lowband", "production")
_OUTPUT_SUFFIX = {
    "studio": "_studio",
    "lowband": "_lowband",
    "production": "_clean",
    "development": "_dev",
    "smart_safe": "_clean",
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
    profile: Literal["studio", "lowband", "production", "development"]
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


def _natural_route_capability(profile: str) -> CapabilityStatusV1:
    """Inspect one route without loading an enhancer or neural checkpoint."""

    try:
        contract = load_natural_route_contract(profile, activate_runtime_config=False)
    except Exception as exc:
        # Capability inspection is a fail-closed trust boundary.  A corrupt
        # config, absent optional dependency, missing/tampered lock or weight,
        # or unexpected inspection failure must never become false readiness.
        detail = str(exc).strip() or type(exc).__name__
        for root, label in (
            (config_dir(), "<config-dir>"),
            (models_dir(), "<model-dir>"),
        ):
            detail = detail.replace(str(root), label)
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


def capabilities_v1() -> CapabilitiesResponseV1:
    """Truthful runtime maturity; never infer product readiness from loose files."""

    natural_routes = [
        _natural_route_capability(profile) for profile in ("production", "studio", "lowband")
    ]

    return CapabilitiesResponseV1(
        capabilities=[
            CapabilityStatusV1(
                capability_id="preserve",
                available=False,
                maturity="blocked",
                reason=(
                    "Preserve is a Smart Safe candidate, but qualified Smart Safe routing "
                    "is not yet available through the versioned job API"
                ),
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
                available=True,
                maturity="qualified",
                providers=["cpu"],
            ),
            CapabilityStatusV1(
                capability_id="restore_source",
                available=False,
                maturity="blocked",
                reason="No qualified signed source-conditioned Sorani Restore pack is installed",
            ),
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
    app.state.sessions = sessions
    app.state.native_sources = native_sources
    app.state.analysis_slots = asyncio.Semaphore(max_concurrent_analyses)
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

        resolved = resolve_client_path(raw, must_exist=True)
        if upload_store.authorizes(resolved):
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
            return await asyncio.to_thread(analyze_audio, path, req.buckets)

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

    @app.post("/api/v1/jobs/{job_id}/override")
    async def override_v1_unit(job_id: str, req: UnitOverrideRequestV1) -> dict[str, Any]:
        """Manually override a guard decision on a single unit.

        The override is recorded in the report's audit trail so it is
        transparent — never silent.  The published WAV is NOT regenerated
        (that would require re-decoding and re-assembling); only the report
        metadata changes.  A future version may regenerate the master.
        """
        snapshot = manager.get_status(job_id)
        if snapshot is None:
            raise ApiError(404, "not_found", f"unknown job: {job_id}")
        state = snapshot.get("state")
        if state not in {"done", "completed"}:
            raise ApiError(409, "invalid_state", f"job {job_id} is {state}, not completed")
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
        if req.unit_index >= len(units):
            raise ApiError(
                400,
                "invalid_unit",
                f"unit_index {req.unit_index} out of range (0..{len(units) - 1})",
            )

        unit = units[req.unit_index]
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

        # Update the summary counts
        enhanced_count = sum(1 for u in units if u.get("final_decision") == "enhanced")
        reverted_count = sum(1 for u in units if u.get("final_decision") == "reverted")
        if "summary" in report and isinstance(report["summary"], dict):
            report["summary"]["enhanced"] = enhanced_count
            report["summary"]["reverted"] = reverted_count

        if report_file is not None:
            try:
                report_file.write_text(json.dumps(report, indent=2) + "\n")
            except Exception as e:
                logger.warning(f"Could not persist updated report to {report_file}: {e}")

        logger.info(
            "Unit %d of job %s: %s → %s (override=%s)",
            req.unit_index,
            job_id,
            old_decision,
            unit["final_decision"],
            req.decision,
        )

        return {
            "job_id": job_id,
            "unit_index": req.unit_index,
            "old_decision": old_decision,
            "new_decision": unit["final_decision"],
            "override": req.decision,
        }

    @app.post("/api/v1/jobs", status_code=202)
    async def create_v1_jobs(req: ProcessingRequestV1) -> dict[str, Any]:
        """Versioned, idempotent batch submission for currently qualified local routes."""

        if req.execution_policy == "cloud_allowed":
            raise ApiError(503, "capability_blocked", "cloud execution is not deployed")
        if isinstance(req.strategy, SmartSafeStrategyV1):
            smart_cap = next(
                (c for c in capabilities_v1().capabilities if c.capability_id == "smart_safe"),
                None,
            )
            if smart_cap is None or not smart_cap.available or smart_cap.maturity != "qualified":
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
        elif isinstance(req.strategy, ManualStrategyV1):
            if req.strategy.route not in {"production", "studio", "lowband"}:
                raise ApiError(
                    503,
                    "capability_blocked",
                    f"route {req.strategy.route!r} is not production-qualified",
                )
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
            output_profile = req.strategy.route
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
                            speaker_id=submit_speaker_id,
                            idempotency_key=item_keys[index],
                            conflict_policy=req.conflict_policy,
                            request_context_hash=batch_hash,
                            record_bundle=req.record_bundle,
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
                return await asyncio.to_thread(
                    compute_peaks_window, path, req.start_s, req.end_s, req.buckets
                )
        except PeaksWindowError as e:
            raise ApiError(400, "bad_request", str(e)) from e

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
