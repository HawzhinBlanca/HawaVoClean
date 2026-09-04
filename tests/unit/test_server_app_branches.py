"""Targeted branch coverage tests for server/app.py helper functions and routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.datastructures import Headers

from hawavoclean.server.app import (
    MAX_ACTIVE_SESSIONS,
    MAX_SESSION_TTL_S,
    SessionRegistry,
    _completed_audio_sha256,
    _cookie_session,
    _has_query_token,
    _job_artifact_path,
    _job_status_v1,
    _loopback_host,
    _safe_upload_name,
    _same_loopback_origin,
    _session_output_path,
    content_type_for,
    parse_range,
)
from hawavoclean.server.policy import PathPolicyError


def test_loopback_host_branches() -> None:
    assert not _loopback_host(None)
    assert not _loopback_host("")
    assert not _loopback_host(" localhost ")
    assert not _loopback_host("localhost\r\n")
    assert not _loopback_host("user@localhost")
    assert not _loopback_host("localhost,evil.com")
    assert not _loopback_host("[::1]:99999")
    assert not _loopback_host("[::1]:0")
    assert not _loopback_host("[::1]:abc")
    assert not _loopback_host("[not_ipv6]")
    assert not _loopback_host("127.0.0.1:80:80")
    assert not _loopback_host("not_a_valid_host")
    assert not _loopback_host("8.8.8.8")
    assert _loopback_host("localhost")
    assert _loopback_host("localhost:8000")
    assert _loopback_host("127.0.0.1")
    assert _loopback_host("127.0.0.1:8080")
    assert _loopback_host("[::1]")
    assert _loopback_host("[::1]:9000")


def test_same_loopback_origin_branches() -> None:
    assert not _same_loopback_origin("https://localhost:8000", "localhost:8000")
    assert not _same_loopback_origin("http://user:pass@localhost:8000", "localhost:8000")
    assert not _same_loopback_origin("http://localhost:8000/some/path", "localhost:8000")
    assert not _same_loopback_origin("http://localhost:8000?query=1", "localhost:8000")
    assert not _same_loopback_origin("http://localhost:8000#fragment", "localhost:8000")
    assert not _same_loopback_origin("http://evil.com:8000", "localhost:8000")
    assert not _same_loopback_origin("http://localhost:8001", "localhost:8000")
    assert _same_loopback_origin("http://localhost:8000", "localhost:8000")
    assert _same_loopback_origin("http://localhost", "localhost")
    assert _same_loopback_origin("http://[::1]:8000", "[::1]:8000")
    assert _same_loopback_origin("http://[::1]", "[::1]")


def test_session_registry_branches() -> None:
    with pytest.raises(ValueError, match="session_ttl_s"):
        SessionRegistry(ttl_s=0)
    with pytest.raises(ValueError, match="session_ttl_s"):
        SessionRegistry(ttl_s=MAX_SESSION_TTL_S + 1)

    now = 1000.0
    registry = SessionRegistry(ttl_s=60.0, clock=lambda: now)
    assert not registry.valid("")
    assert not registry.valid("a" * 300)
    assert not registry.valid("unknown_token")

    token, ttl = registry.issue()
    assert ttl == 60
    assert registry.valid(token)

    # Revoke
    registry.revoke(token)
    assert not registry.valid(token)

    # Expiry
    token2, _ = registry.issue()
    assert registry.valid(token2)
    now += 61.0
    assert not registry.valid(token2)

    # Capacity eviction
    registry2 = SessionRegistry(ttl_s=60.0, clock=lambda: 1000.0)
    issued = [registry2.issue()[0] for _ in range(MAX_ACTIVE_SESSIONS + 5)]
    assert len(issued) == MAX_ACTIVE_SESSIONS + 5
    assert len(registry2._entries) <= MAX_ACTIVE_SESSIONS


def test_cookie_session_and_query_token_branches() -> None:
    assert _cookie_session(Headers({})) is None
    assert _cookie_session(Headers({"cookie": "x" * 5000})) is None
    assert _cookie_session(Headers({"cookie": "hawa_session=abc123"})) == "abc123"

    assert not _has_query_token(b"")
    assert not _has_query_token(b"foo=bar")
    assert _has_query_token(b"token=secret")
    assert _has_query_token(b"TOKEN=secret")
    assert _has_query_token(b"foo=bar&token=secret")


def test_safe_upload_name_branches() -> None:
    assert _safe_upload_name("") == "upload.bin"
    assert _safe_upload_name(None) == "upload.bin"
    assert _safe_upload_name("..") == "upload.bin"
    assert _safe_upload_name(".") == "upload.bin"
    assert _safe_upload_name("hello\x00world.wav") == "helloworld.wav"
    assert _safe_upload_name("/path/to/my_audio.wav") == "my_audio.wav"


def test_content_type_and_parse_range_branches() -> None:
    assert content_type_for(Path("track.wav")) == "audio/wav"
    assert content_type_for(Path("track.flac")) == "audio/flac"
    assert content_type_for(Path("track.unknown_ext_xyz")) == "application/octet-stream"

    assert parse_range(None, 1000) is None
    assert parse_range("", 1000) is None
    assert parse_range("seconds=0-10", 1000) is None
    assert parse_range("bytes=100-200", 1000) == (100, 200)
    assert parse_range("bytes=100-", 1000) == (100, 999)
    assert parse_range("bytes=-200", 1000) == (800, 999)


def test_job_status_v1_and_artifact_branches(tmp_path: Path) -> None:
    # 1. Job status legacy mapping
    queued_snap = {
        "job_id": "j1",
        "state": "queued",
        "output_path": "/out.wav",
        "report_path": "/out.json",
        "created_at": "2026-08-01T00:00:00Z",
    }
    status = _job_status_v1(queued_snap)
    assert status.state == "queued"

    unknown_snap = {
        "job_id": "j2",
        "state": "weird_state",
        "output_path": "/out.wav",
        "report_path": "/out.json",
        "created_at": "2026-08-01T00:00:00Z",
    }
    assert _job_status_v1(unknown_snap).state == "failed"

    # 2. _completed_audio_sha256
    assert _completed_audio_sha256({"state": "queued"}) is None
    assert (
        _completed_audio_sha256({"state": "done", "report": {"output": {"sha256": "invalid_sha"}}})
        is None
    )
    valid_sha = "a" * 64
    assert (
        _completed_audio_sha256({"state": "done", "report": {"output": {"sha256": valid_sha}}})
        == valid_sha
    )
    # Contradictory evidence
    contradictory = {
        "state": "done",
        "report": {"output": {"sha256": "a" * 64}},
        "bundle": {"master_sha256": "b" * 64},
    }
    assert _completed_audio_sha256(contradictory) is None

    # 3. _job_artifact_path for invalid bundle
    bad_bundle_snap = {
        "state": "done",
        "report": {"output": {"sha256": valid_sha}},
        "bundle_path": None,
        "bundle": {"archive_sha256": "c" * 64, "master_sha256": valid_sha},
    }
    assert _job_artifact_path(bad_bundle_snap, "record") is None

    # 4. _session_output_path validation
    in_file = tmp_path / "source.wav"
    in_file.touch()
    with pytest.raises(PathPolicyError, match="output path is required"):
        _session_output_path(in_file, "production", "")

    with pytest.raises(PathPolicyError, match="output path must be absolute"):
        _session_output_path(in_file, "production", "relative/out.wav")

    with pytest.raises(PathPolicyError, match="must remain beside"):
        _session_output_path(in_file, "production", "/different/dir/out.wav")


def test_lease_source_id_branches(tmp_path: Path) -> None:
    from hawavoclean.server.app import _lease_source_id
    from hawavoclean.server.retention import UploadStore
    from hawavoclean.server.source_caps import NativeSourceRegistry

    upload_root = tmp_path / "uploads"
    upload_store = UploadStore(upload_root)
    native_registry = NativeSourceRegistry()

    # 1. Managed upload by opaque source_id
    uploaded_file = upload_store.stage("test.wav")
    uploaded_file.write_bytes(b"data")
    opaque_upload_id = upload_store.source_id(uploaded_file)
    with _lease_source_id(upload_store, native_registry, opaque_upload_id) as leased:
        assert leased == uploaded_file.resolve()

    # 2. Managed upload by authorized path string
    with _lease_source_id(upload_store, native_registry, str(uploaded_file)) as leased:
        assert leased == uploaded_file.resolve()

    # 3. Native source by registered opaque source_id
    native_file = tmp_path / "native.wav"
    native_file.write_bytes(b"data")
    native_cap = native_registry.register(str(native_file))
    with _lease_source_id(upload_store, native_registry, native_cap.source_id) as leased:
        assert leased == native_file.resolve()

    # 4. Native source by raw registered path
    with _lease_source_id(upload_store, native_registry, str(native_file)) as leased:
        assert leased == native_file.resolve()

    # 5. Unknown source
    with _lease_source_id(upload_store, native_registry, "nonexistent_source_id") as leased:
        assert leased is None
