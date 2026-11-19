"""Separation routing and real publication plumbing, using explicit test stems.

These tests establish integration behavior, not SAM-Audio inference quality.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

import hawavoclean.audio_edit as edits
from hawavoclean.audio.types import AudioBuffer
from hawavoclean.audio_edit import EditRequest, run_audio_edit
from hawavoclean.hashing import hash_file
from hawavoclean.publication import resolve_committed_publication
from hawavoclean.report.writer import load_json_report
from hawavoclean.server.app import create_app
from hawavoclean.server.contracts import SeparationStrategyV1
from hawavoclean.server.jobs import JobManager, JobRecord, default_command


def _source(path: Path, rate: int = 44100) -> Path:
    time = np.arange(rate // 2) / rate
    data = np.stack(
        [0.2 * np.sin(2 * np.pi * 220 * time), 0.1 * np.sin(2 * np.pi * 440 * time)], axis=-1
    ).astype(np.float32)
    sf.write(path, data, rate, subtype="FLOAT")
    return path


@pytest.mark.parametrize("cleanup", [None, "production"])
def test_edit_publishes_original_lineage_and_selected_model_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: Literal["production"] | None,
) -> None:
    source = _source(tmp_path / "source.wav")
    original_hash = hash_file(source)
    selected: list[str] = []

    def separate(src: Path, dst: Path, **options: object) -> dict[str, object]:
        assert options["operation"] == "remove"
        data, rate = sf.read(src, always_2d=True, dtype="float32")
        # A deliberately chosen residual; never source-minus-target.
        sf.write(dst, data * 0.3, rate, subtype="FLOAT")
        selected.append(hash_file(dst))
        return {
            "engine": "test-stem",
            "qualification": "experimental_unqualified",
            "model_sha256": "1" * 64,
        }

    monkeypatch.setattr(edits, "separate_audio", separate)
    target = tmp_path / "result.wav"
    request = EditRequest(
        input_path=str(source),
        original_input_path=str(source),
        output_path=str(target),
        options=SeparationStrategyV1(
            kind="separate", operation="remove", prompt="background music", cleanup_profile=cleanup
        ),
    )
    report = run_audio_edit(request)
    reloaded = load_json_report(target.with_suffix(".hawavoclean.json"))
    assert hash_file(source) == original_hash == report.input.sha256
    assert report.input.path == str(source)
    assert report.output.sha256 == hash_file(target) == reloaded.output.sha256
    assert report.output.samples == report.input.samples == 22050
    assert report.output.sample_rate == 44100
    assert report.output.channels == 2
    assert report.separation is not None
    assert report.separation["selected_stem_sha256"] == selected[0]
    assert "signal-integrity-only" in report.separation["guard_scope"]
    assert resolve_committed_publication(target) is not None


def test_failed_separation_cannot_publish_or_modify_prior_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path / "source.wav")
    target = tmp_path / "existing.wav"
    target.write_bytes(b"previous output")

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("Model is missing")

    monkeypatch.setattr(edits, "separate_audio", fail)
    with pytest.raises(RuntimeError, match="missing"):
        run_audio_edit(
            EditRequest(
                input_path=str(source),
                original_input_path=str(source),
                output_path=str(target),
                overwrite=True,
                options=SeparationStrategyV1(kind="separate", operation="extract", prompt="speech"),
            )
        )
    assert target.read_bytes() == b"previous output"
    assert not target.with_suffix(".hawavoclean.json").exists()
    assert not list(tmp_path.glob(".hawa-separate-*"))


def test_actual_decode_ceiling_closes_stream_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path / "source.wav")
    monkeypatch.setattr(edits, "MAX_DURATION_S", 1.0)
    closed = []

    def lying_decode(*_args: object, **_kwargs: object) -> Iterator[AudioBuffer]:
        try:
            yield AudioBuffer(np.zeros((2, 44101), dtype=np.float32), 44100)
            raise AssertionError("decoder was consumed after its frame limit")
        finally:
            closed.append(True)

    monkeypatch.setattr(edits, "iter_decode_audio", lying_decode)
    with pytest.raises(ValueError, match="Decoded recording exceeds"):
        run_audio_edit(
            EditRequest(
                input_path=str(source),
                original_input_path=str(source),
                output_path=str(tmp_path / "out.wav"),
                options=SeparationStrategyV1(kind="separate", operation="extract", prompt="speech"),
            )
        )
    assert closed == [True]


def test_output_symlink_cannot_redirect_publication(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.wav")
    digest = hash_file(source)
    alias = tmp_path / "alias.wav"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="linked destination"):
        run_audio_edit(
            EditRequest(
                input_path=str(source),
                original_input_path=str(source),
                output_path=str(alias),
                overwrite=True,
                options=SeparationStrategyV1(kind="separate", operation="extract", prompt="speech"),
            )
        )
    assert hash_file(source) == digest


def test_separation_settings_survive_durable_roundtrip_without_prompt_in_argv(
    tmp_path: Path,
) -> None:
    options = SeparationStrategyV1(
        kind="separate", operation="extract", prompt="a dog barking"
    ).model_dump()
    record = JobRecord(
        job_id="j_separate",
        input_path=tmp_path / "input.wav",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.hawavoclean.json",
        profile="production",
        overwrite=False,
        mode="separate",
        separation_options=options,
    )
    restored = JobRecord.from_storage(json.loads(json.dumps(record.storage_record())))
    assert restored.separation_options == options
    assert "a dog barking" not in " ".join(default_command(restored))
    assert default_command(restored)[-1] == "hawavoclean.audio_edit"


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt": " "},
        {"operation": "shell"},
        {"seed": -1},
        {"device": "remote"},
        {"arbitrary": True},
    ],
)
def test_closed_separation_contract(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SeparationStrategyV1.model_validate(
            {"kind": "separate", "operation": "extract", "prompt": "speech", **changes}
        )


def test_assistant_and_model_unavailability_use_authenticated_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "rules")
    monkeypatch.delenv("HAWAVOCLEAN_AUDIOGHOST_PYTHON", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_ROFORMER_PYTHON", raising=False)
    monkeypatch.setenv("HAWAVOCLEAN_ROFORMER_MODEL_DIR", str(tmp_path / "no-roformer-models"))
    manager = JobManager()
    app = create_app("audio-edit-test", None, job_manager=manager, min_free_bytes=0)
    headers = {"X-Hawa-Token": "audio-edit-test"}
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            denied = client.post("/api/v1/assistant/plan", json={"message": "clean the voice"})
            assert denied.status_code == 401
            plan = client.post(
                "/api/v1/assistant/plan",
                headers=headers,
                json={"message": "remove background music"},
            )
            assert plan.status_code == 200, plan.text
            assert plan.json()["operation"] == "remove"
            capabilities = client.get("/api/v1/capabilities", headers=headers).json()
            separation = next(
                c for c in capabilities["capabilities"] if c["capabilityId"] == "separate"
            )
            assert separation["available"] is False
            blocked = client.post(
                "/api/v1/jobs",
                headers=headers,
                json={
                    "schemaVersion": 1,
                    "sourceIds": ["not-resolved-before-readiness"],
                    "strategy": {"kind": "separate", "operation": "extract", "prompt": "speech"},
                    "executionPolicy": "offline_only",
                    "idempotencyKey": "separation-1",
                },
            )
            assert blocked.status_code == 503
            assert blocked.json()["error"] == "capability_blocked"
            assert manager.list_jobs() == []
    finally:
        manager.shutdown()
