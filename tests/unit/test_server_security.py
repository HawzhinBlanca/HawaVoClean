"""Adversarial checks for the local engine broker trust boundary."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hawavoclean.publication import (
    publication_paths,
    publish_output_generation,
    resolve_committed_publication,
)
from hawavoclean.server.app import SESSION_PATH, create_app
from hawavoclean.server.jobs import JobManager

pytestmark = pytest.mark.unit

TOKEN = "native-bootstrap-secret"
ROOT_HEADER = {"X-Hawa-Token": TOKEN}


def _wav(path: Path, frames: int = 1600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(
            b"".join(struct.pack("<h", (index % 100) - 50) for index in range(frames))
        )
    return path


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def secured_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, Clock, JobManager]]:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    clock = Clock()
    manager = JobManager()
    app = create_app(
        TOKEN,
        None,
        job_manager=manager,
        on_shutdown=lambda: None,
        min_free_bytes=0,
        session_ttl_s=2,
        session_clock=clock,
    )
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client, clock, manager


def test_host_and_origin_are_exactly_confined_to_the_local_broker(
    secured_client: tuple[TestClient, Clock, JobManager],
) -> None:
    client, _, _ = secured_client

    hostile_host = client.get("/api/health", headers={**ROOT_HEADER, "Host": "attacker.example"})
    assert hostile_host.status_code == 403
    assert hostile_host.json()["error"] == "forbidden"

    for origin in ("https://attacker.example", "null", "file://", "http://127.0.0.1:9999"):
        response = client.get("/api/health", headers={**ROOT_HEADER, "Origin": origin})
        assert response.status_code == 403, origin
        assert "access-control-allow-origin" not in response.headers

    trusted = client.get("/api/health", headers={**ROOT_HEADER, "Origin": "hawa://app"})
    assert trusted.status_code == 200
    assert trusted.headers["access-control-allow-origin"] == "hawa://app"
    assert trusted.headers["access-control-allow-credentials"] == "true"
    assert trusted.headers["vary"] == "Origin"

    same_origin = client.get(
        "/api/health", headers={**ROOT_HEADER, "Origin": "http://127.0.0.1:8765"}
    )
    assert same_origin.status_code == 200
    assert same_origin.headers["access-control-allow-origin"] == "http://127.0.0.1:8765"

    cross_site_without_origin = client.get(
        "/api/health", headers={**ROOT_HEADER, "Sec-Fetch-Site": "cross-site"}
    )
    assert cross_site_without_origin.status_code == 403


def test_missing_auth_and_any_query_token_fail_closed(
    secured_client: tuple[TestClient, Clock, JobManager],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _, _ = secured_client
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Hawa-Token": "wrong"}).status_code == 401

    for query in (
        f"token={TOKEN}",
        f"TOKEN={TOKEN}",
        f"%74oken={TOKEN}",
        "token",
    ):
        response = client.get(f"/api/health?{query}", headers=ROOT_HEADER)
        assert response.status_code == 400, query
        assert response.json() == {
            "error": "query_auth_forbidden",
            "message": "authentication tokens are not accepted in URLs",
        }
        assert TOKEN not in response.text
    assert TOKEN not in caplog.text


def test_short_lived_cookie_and_bearer_sessions_expire(
    secured_client: tuple[TestClient, Clock, JobManager],
) -> None:
    client, clock, _ = secured_client

    wrong_bootstrap = client.post(SESSION_PATH, headers={"X-Hawa-Token": "wrong"})
    assert wrong_bootstrap.status_code == 401
    issued = client.post(SESSION_PATH, headers=ROOT_HEADER)
    assert issued.status_code == 200
    assert issued.headers["cache-control"] == "no-store"
    assert "HttpOnly" in issued.headers["set-cookie"]
    assert "SameSite=strict" in issued.headers["set-cookie"]
    token = issued.json()["sessionToken"]
    assert issued.json()["expiresInSeconds"] == 2

    # TestClient retained the HttpOnly cookie, so no JavaScript-readable root
    # secret or Authorization header is needed for ordinary same-site calls.
    assert client.get("/api/health").status_code == 200
    client.cookies.clear()
    assert (
        client.get("/api/health", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    )

    # A session capability cannot mint another capability; only the native
    # bootstrap header can do that.
    assert (
        client.post(SESSION_PATH, headers={"Authorization": f"Bearer {token}"}).status_code == 401
    )
    clock.now += 2.1
    assert (
        client.get("/api/health", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    )
    assert client.get("/api/health", headers=ROOT_HEADER).status_code == 200


def test_unhandled_errors_are_opaque_in_response_and_sanitized_in_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    manager = JobManager()
    secret_detail = "root-secret-in-exception /Users/private/audio.wav"

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(secret_detail)

    manager.get_status = boom  # type: ignore[method-assign]
    app = create_app(TOKEN, job_manager=manager, on_shutdown=lambda: None, min_free_bytes=0)
    caplog.set_level(logging.ERROR, logger="hawavoclean.server")
    # Deliberately do not surface the server exception so the public response
    # generated by FastAPI's error handler can be inspected.
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/jobs/j_failure", headers=ROOT_HEADER)

        assert response.status_code == 500
        body = response.json()
        assert set(body) == {"error", "message", "request_id"}
        assert body["error"] == "internal_error"
        assert secret_detail not in response.text
        assert re.fullmatch(r"req_[0-9a-f]{24}", body["request_id"])
        assert response.headers["x-hawa-request-id"] == body["request_id"]
        assert secret_detail not in caplog.text
        assert "RuntimeError" in caplog.text and body["request_id"] in caplog.text


def test_wildcard_and_opaque_trusted_origin_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicit non-null"):
        create_app(TOKEN, trusted_origins=frozenset({"*"}))
    with pytest.raises(ValueError, match="explicit non-null"):
        create_app(TOKEN, trusted_origins=frozenset({"null"}))


def test_renderer_session_cannot_turn_arbitrary_home_paths_into_authority(
    secured_client: tuple[TestClient, Clock, JobManager],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, manager = secured_client
    fake_home = tmp_path / "home"
    secret = _wav(fake_home / ".ssh" / "id_ed25519.wav")
    selected = _wav(fake_home / "Music" / "selected.wav")
    submitted: list[dict[str, Any]] = []

    def fake_submit(**kwargs: Any) -> dict[str, Any]:
        submitted.append(kwargs)
        output = Path(kwargs["output_path"])
        return {
            "job_id": f"j_{len(submitted)}",
            "output_path": str(output),
            "report_path": str(output.with_suffix(".hawavoclean.json")),
        }

    monkeypatch.setattr(manager, "submit", fake_submit)
    issued = client.post(SESSION_PATH, headers=ROOT_HEADER)
    bearer = {"Authorization": f"Bearer {issued.json()['sessionToken']}"}
    client.cookies.clear()

    assert client.get("/api/audio", headers=bearer, params={"path": str(secret)}).status_code == 403
    assert (
        client.post("/api/analyze", headers=bearer, json={"path": str(secret)}).status_code == 403
    )
    assert (
        client.post(
            "/api/peaks",
            headers=bearer,
            json={"path": str(secret), "start_s": 0, "end_s": 0.01},
        ).status_code
        == 403
    )
    refused_job = client.post(
        "/api/jobs",
        headers=bearer,
        json={"input_path": str(secret), "profile": "production"},
    )
    assert refused_job.status_code == 403
    assert submitted == []
    assert (
        client.post(
            "/api/v1/native-sources", headers=bearer, json={"path": str(secret)}
        ).status_code
        == 403
    )
    # A rejected create-job did not register or otherwise launder the path.
    assert client.get("/api/audio", headers=bearer, params={"path": str(secret)}).status_code == 403

    registration = client.post(
        "/api/v1/native-sources", headers=ROOT_HEADER, json={"path": str(selected)}
    )
    assert registration.status_code == 200
    assert registration.headers["cache-control"] == "no-store"
    source_id = registration.json()["sourceId"]
    assert re.fullmatch(r"[0-9a-f]{32}", source_id)
    assert registration.json()["path"] == str(selected.resolve())

    assert (
        client.get("/api/audio", headers=bearer, params={"path": str(selected)}).status_code == 200
    )
    assert (
        client.post("/api/analyze", headers=bearer, json={"path": str(selected)}).status_code == 200
    )
    assert (
        client.post(
            "/api/peaks",
            headers=bearer,
            json={"path": str(selected), "start_s": 0, "end_s": 0.01},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/analyze", headers=bearer, json={"schemaVersion": 1, "sourceId": source_id}
        ).status_code
        == 200
    )

    arbitrary_output = selected.parent / "unrelated.wav"
    assert (
        client.post(
            "/api/jobs",
            headers=bearer,
            json={
                "input_path": str(selected),
                "output_path": str(arbitrary_output),
                "profile": "production",
            },
        ).status_code
        == 403
    )
    safe_output = selected.parent / "selected_clean-2.wav"
    accepted = client.post(
        "/api/jobs",
        headers=bearer,
        json={
            "input_path": str(selected),
            "output_path": str(safe_output),
            "profile": "production",
        },
    )
    assert accepted.status_code == 202
    assert submitted[-1]["input_path"] == selected.resolve()
    assert submitted[-1]["output_path"] == safe_output.resolve()


def test_managed_upload_is_a_renderer_capability(
    secured_client: tuple[TestClient, Clock, JobManager],
    tmp_path: Path,
) -> None:
    client, _, _ = secured_client
    source = _wav(tmp_path / "browser.wav")
    issued = client.post(SESSION_PATH, headers=ROOT_HEADER)
    bearer = {"Authorization": f"Bearer {issued.json()['sessionToken']}"}
    client.cookies.clear()
    upload = client.post(
        "/api/upload",
        headers=bearer,
        files={"file": (source.name, source.read_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    managed = upload.json()["path"]
    assert client.get("/api/audio", headers=bearer, params={"path": managed}).status_code == 200
    assert client.post("/api/analyze", headers=bearer, json={"path": managed}).status_code == 200


def test_session_artifacts_are_job_bound_immutable_generations_not_public_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "work"))
    manager = JobManager()
    output = tmp_path / "work" / "result.wav"
    candidate = tmp_path / "candidate.wav"
    authentic = b"RIFF-authentic-generation"
    candidate.write_bytes(authentic)
    digest = hashlib.sha256(authentic).hexdigest()
    report_value = {"output": {"sha256": digest}, "summary": {"units_total": 1, "enhanced": 1}}
    audio, report, summary = publish_output_generation(
        candidate,
        output,
        json.dumps(report_value),
        "authentic summary",
    )
    first_generation = resolve_committed_publication(output)
    assert first_generation is not None

    def artifact_evidence(generation: tuple[Path, Path, Path]) -> dict[str, Any]:
        roles = ("audio", "report", "summary")
        return {
            "schema_version": 1,
            "storage": "immutable_generation",
            "generation_id": generation[0].parent.name,
            **{
                role: {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
                for role, path in zip(roles, generation, strict=True)
            },
        }

    snapshot: dict[str, Any] = {
        "job_id": "j_artifact",
        "seq": 1,
        "state": "done",
        "stage": "done",
        "progress": 1.0,
        "message": "Done",
        "input_path": str(tmp_path / "selected.wav"),
        "output_path": str(audio),
        "report_path": str(report),
        "profile": "production",
        "mode": "natural",
        "created_at": "2026-08-27T00:00:00Z",
        "started_at": "2026-08-27T00:00:01Z",
        "finished_at": "2026-08-27T00:00:02Z",
        "conflict_policy": "replace",
        "record_bundle": False,
        "report": report_value,
        "artifact_evidence": artifact_evidence(first_generation),
    }
    snapshots = {"j_artifact": snapshot}
    monkeypatch.setattr(
        manager,
        "get_status",
        lambda job_id: snapshots.get(job_id),
    )
    monkeypatch.setattr(manager, "list_jobs", lambda: list(snapshots.values()))
    app = create_app(TOKEN, job_manager=manager, on_shutdown=lambda: None, min_free_bytes=0)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        issued = client.post(SESSION_PATH, headers=ROOT_HEADER)
        bearer = {"Authorization": f"Bearer {issued.json()['sessionToken']}"}
        client.cookies.clear()

        # Simulate a hard kill between independent convenience-export copies.
        paths = publication_paths(output)
        paths.audio.write_bytes(b"new-public-audio")
        paths.json.write_text('{"mixed":"old-report"}', encoding="utf-8")
        paths.txt.write_text("mixed summary", encoding="utf-8")

        legacy_master = client.get("/api/audio", headers=bearer, params={"path": str(output)})
        exact_master = client.get("/api/v1/jobs/j_artifact/artifacts/master", headers=bearer)
        exact_report = client.get("/api/v1/jobs/j_artifact/artifacts/report", headers=bearer)
        assert legacy_master.status_code == exact_master.status_code == 200
        assert legacy_master.content == exact_master.content == authentic
        assert exact_report.json() == report_value
        assert (
            client.get(
                "/api/audio",
                headers=bearer,
                params={
                    "path": str(
                        paths.generations / next(paths.generations.iterdir()).name / "master.wav"
                    )
                },
            ).status_code
            == 403
        )
        assert summary.read_text(encoding="utf-8") == "mixed summary"

        # Two deterministic runs can produce identical audio while their
        # reports and summaries differ. Persisted sidecar evidence binds each
        # job to its own exact generation rather than forcing a false 409.
        paths.audio.write_bytes(authentic)
        paths.json.write_text(json.dumps(report_value), encoding="utf-8")
        paths.txt.write_text("authentic summary", encoding="utf-8")
        second_report = {**report_value, "other_run": True}
        publish_output_generation(
            candidate,
            output,
            json.dumps(second_report),
            "different summary",
            overwrite=True,
        )
        second_generation = resolve_committed_publication(output)
        assert second_generation is not None
        snapshots["j_artifact_new"] = {
            **snapshot,
            "job_id": "j_artifact_new",
            "report": second_report,
            "artifact_evidence": artifact_evidence(second_generation),
        }

        old_master = client.get("/api/v1/jobs/j_artifact/artifacts/master", headers=bearer)
        old_report = client.get("/api/v1/jobs/j_artifact/artifacts/report", headers=bearer)
        old_summary = client.get("/api/v1/jobs/j_artifact/artifacts/summary", headers=bearer)
        new_report = client.get("/api/v1/jobs/j_artifact_new/artifacts/report", headers=bearer)
        new_summary = client.get("/api/v1/jobs/j_artifact_new/artifacts/summary", headers=bearer)
        assert old_master.status_code == old_report.status_code == old_summary.status_code == 200
        assert old_master.content == authentic
        assert old_report.json() == report_value
        assert old_summary.text == "authentic summary"
        assert new_report.status_code == new_summary.status_code == 200
        assert new_report.json() == second_report
        assert new_summary.text == "different summary"

        # Job-bound digests enable exact history; they do not relax integrity.
        # If an application-managed generation is modified, the endpoint
        # keeps the explicit 409 contract and serves no bytes.
        first_generation[1].chmod(0o600)
        first_generation[1].write_text('{"tampered":true}', encoding="utf-8")
        tampered = client.get("/api/v1/jobs/j_artifact/artifacts/report", headers=bearer)
        assert tampered.status_code == 409
        assert tampered.json()["error"] == "artifact_unavailable"
