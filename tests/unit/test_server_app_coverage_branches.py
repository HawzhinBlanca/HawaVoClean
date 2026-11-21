from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response

from hawavoclean.server.app import (
    ApiError,
    _file_chunks,
    _job_artifact_path,
    _requested_job_artifact,
    _safe_upload_name,
    _session_output_path,
    create_app,
)
from hawavoclean.server.policy import PathPolicyError


def test_clean_filename_edge_cases() -> None:
    # Surrogate that triggers UnicodeEncodeError on os.fsencode
    surrogate_name = "test_\ud800_name.wav"
    cleaned = _safe_upload_name(surrogate_name)
    assert "test_" in cleaned

    # Empty or dots
    assert _safe_upload_name("") == "upload.bin"
    assert _safe_upload_name(".") == "upload.bin"
    assert _safe_upload_name("..") == "upload.bin"
    assert _safe_upload_name("normal.wav") == "normal.wav"


def test_file_chunks_truncated(tmp_path: Path) -> None:
    test_file = tmp_path / "short.bin"
    test_file.write_bytes(b"12345")
    # Request 100 bytes from a 5-byte file
    chunks = list(_file_chunks(test_file, start=0, end=100))
    assert b"".join(chunks) == b"12345"


def test_session_output_path_validations(tmp_path: Path) -> None:
    in_path = tmp_path / "in.wav"

    # Empty
    with pytest.raises(PathPolicyError, match="output path is required"):
        _session_output_path(in_path, "production", "")

    # Relative
    with pytest.raises(PathPolicyError, match="output path must be absolute"):
        _session_output_path(in_path, "production", "rel/out.wav")

    # Resolve OSError
    with (
        patch.object(Path, "resolve", side_effect=OSError("symlink error")),
        pytest.raises(PathPolicyError, match="output path cannot be resolved"),
    ):
        _session_output_path(in_path, "production", str(tmp_path / "out.wav"))


def test_resolve_verified_job_artifact_branches(tmp_path: Path) -> None:
    # 1. Missing completed audio
    snapshot_no_audio: dict[str, Any] = {"state": "done"}
    assert _job_artifact_path(snapshot_no_audio, "master") is None

    # 2. Record kind with invalid raw_bundle or evidence
    snapshot_record: dict[str, Any] = {
        "state": "done",
        "report": {"output": {"sha256": "a" * 64}},
        "bundle_path": 123,  # not a string
        "bundle": None,
    }
    assert _job_artifact_path(snapshot_record, "record") is None

    # 3. Record kind with digest mismatch
    fake_verified = MagicMock()
    fake_verified.archive_sha256 = "b" * 64
    fake_verified.master_sha256 = "c" * 64
    snapshot_record_mismatch: dict[str, Any] = {
        "state": "done",
        "report": {"output": {"sha256": "a" * 64}},
        "bundle_path": "/fake/bundle.zip",
        "bundle": {"archive_sha256": "a" * 64},
    }
    with patch("hawavoclean.server.app.verify_processing_record", return_value=fake_verified):
        assert _job_artifact_path(snapshot_record_mismatch, "record") is None

    # 4. Master kind when resolve_immutable_publication_generation returns None
    snapshot_master: dict[str, Any] = {
        "state": "done",
        "output_path": str(tmp_path / "out.wav"),
        "report": {"output": {"sha256": "a" * 64}},
    }
    with patch(
        "hawavoclean.server.app.resolve_immutable_publication_generation", return_value=None
    ):
        assert _job_artifact_path(snapshot_master, "master") is None


def test_requested_job_artifact_branches(tmp_path: Path) -> None:
    fake_mgr = MagicMock()
    out = tmp_path / "done.wav"
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK")

    fake_mgr.list_jobs.return_value = [
        {"state": "running", "output_path": str(tmp_path / "run.wav")},
        {"state": "done", "output_path": str(out), "bundle_path": str(bundle)},
    ]

    # Matching bundle path
    res = _requested_job_artifact(fake_mgr, bundle.resolve())
    assert res is not None
    assert res[1] == "record"

    # Non matching
    assert _requested_job_artifact(fake_mgr, tmp_path / "unknown.wav") is None


@pytest.mark.anyio
async def test_server_app_middleware_edge_cases() -> None:
    app = create_app(
        token="test_token_1234567890",
        trusted_origins=frozenset(["http://localhost:3000"]),
    )

    scope_base: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 50000),
        "scheme": "http",
    }

    async def run_req(
        method: str,
        path: str,
        headers: list[tuple[bytes, bytes]],
    ) -> Response:
        scope = dict(scope_base)
        scope["method"] = method
        scope["path"] = path
        scope["headers"] = headers

        response_messages: list[dict[str, Any]] = []

        async def fake_receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def fake_send(msg: MutableMapping[str, Any]) -> None:
            response_messages.append(dict(msg))

        await app(scope, fake_receive, fake_send)

        status = 200
        resp_headers: dict[str, str] = {}
        body = b""
        for msg in response_messages:
            if msg["type"] == "http.response.start":
                status = msg["status"]
                resp_headers = {
                    k.decode("latin1"): v.decode("latin1") for k, v in msg.get("headers", [])
                }
            elif msg["type"] == "http.response.body":
                body += msg.get("body", b"")

        return Response(content=body, status_code=status, headers=resp_headers)

    # 1. Multiple Origin headers -> 403
    resp = await run_req(
        "GET",
        "/api/jobs",
        [
            (b"host", b"127.0.0.1:8000"),
            (b"origin", b"http://localhost:3000"),
            (b"origin", b"http://evil.com"),
        ],
    )
    assert resp.status_code == 403
    assert b"multiple Origin headers are not allowed" in resp.body

    # 2. CORS preflight requires Origin -> 403
    resp = await run_req(
        "OPTIONS",
        "/api/jobs",
        [(b"host", b"127.0.0.1:8000")],
    )
    assert resp.status_code == 403
    assert b"CORS preflight requires Origin" in resp.body

    # 3. CORS preflight invalid method -> 403
    resp = await run_req(
        "OPTIONS",
        "/api/jobs",
        [
            (b"host", b"127.0.0.1:8000"),
            (b"origin", b"http://localhost:3000"),
            (b"access-control-request-method", b"DELETE"),
        ],
    )
    assert resp.status_code == 403
    assert b"CORS preflight is not allowed" in resp.body

    # 4. Multiple x-hawa-token headers -> 401
    resp = await run_req(
        "GET",
        "/api/jobs",
        [
            (b"host", b"127.0.0.1:8000"),
            (b"x-hawa-token", b"token1"),
            (b"x-hawa-token", b"token2"),
        ],
    )
    assert resp.status_code == 401
    assert b"multiple credentials refused" in resp.body

    # 5. Multiple authorization headers -> 401
    resp = await run_req(
        "GET",
        "/api/jobs",
        [
            (b"host", b"127.0.0.1:8000"),
            (b"authorization", b"Bearer a"),
            (b"authorization", b"Bearer b"),
        ],
    )
    assert resp.status_code == 401
    assert b"multiple credentials refused" in resp.body

    # 6. Unauthorized with Origin -> CORS headers on 401 response
    resp = await run_req(
        "GET",
        "/api/jobs",
        [
            (b"host", b"127.0.0.1:8000"),
            (b"origin", b"http://localhost:3000"),
        ],
    )
    assert resp.status_code == 401
    assert "access-control-allow-origin" in resp.headers


@pytest.mark.anyio
async def test_exception_handlers_internal_and_http() -> None:
    app = create_app(token="tok")

    fake_request = MagicMock(spec=Request)
    fake_request.scope = {"state": {"request_id": "req-12345"}}

    # 1. ApiError 500
    handler_api = cast(Callable[..., Awaitable[Response]], app.exception_handlers[ApiError])
    resp = await handler_api(fake_request, ApiError(500, "crash", "boom"))
    assert resp.status_code == 500
    assert b"internal_error" in resp.body

    # 2. StarletteHTTPException 500
    handler_http = cast(
        Callable[..., Awaitable[Response]], app.exception_handlers[StarletteHTTPException]
    )
    resp = await handler_http(fake_request, StarletteHTTPException(status_code=500, detail="boom"))
    assert resp.status_code == 500
    assert b"internal_error" in resp.body

    # 3. StarletteHTTPException with dict detail
    resp = await handler_http(
        fake_request,
        StarletteHTTPException(status_code=400, detail=cast(Any, {"error": "custom_bad_request"})),
    )
    assert resp.status_code == 400
    assert b"custom_bad_request" in resp.body
