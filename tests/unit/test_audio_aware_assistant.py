"""Phase 5 audio-aware assistant, truthful capabilities, and strength/time-range tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

import hawavoclean.assistant as assistant
import hawavoclean.cli as cli
from hawavoclean.assistant import (
    AcousticContext,
    AssistantError,
    AudioPlanRequest,
    PlanPayload,
    TimeRange,
    assistant_status,
    plan_audio,
)
from hawavoclean.hashing import hash_file
from hawavoclean.report.writer import load_json_report
from hawavoclean.separation import RuntimeStatus
from hawavoclean.server.app import capabilities_v1, create_app
from hawavoclean.server.contracts import ManualStrategyV1, SeparationStrategyV1
from hawavoclean.server.jobs import JobManager, JobRecord, default_command

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _rules_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "rules")
    monkeypatch.delenv("HAWAVOCLEAN_OLLAMA_URL", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_CHAT_MODEL", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_OLLAMA_KEEP_ALIVE", raising=False)
    monkeypatch.setattr(assistant, "_LOCAL_ROOT", tmp_path)


# --- Privacy & schema boundaries --------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    [
        {"path": "/Users/me/secret.wav"},
        {"file_path": "/tmp/a.wav"},
        {"filename": "interview.wav"},
        {"transcript": "spoken Kurdish words"},
        {"audio": [0.1, 0.2]},
        {"noise_risk": True},
        {"sample_rate": True},
        {"channels": False},
        {"duration_s": -1.0},
        {"noise_risk": 1.2},
        {"noise_risk": float("nan")},
        {"estimated_cutoff_hz": 0.0},
        {"bandwidth_shape": "ultrasonic"},
        {"available_profiles": ["nonexistent"]},
        {"separation_available": "yes"},
    ],
)
def test_acoustic_context_rejects_paths_transcripts_audio_and_invalid_bounds(
    forbidden: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        AcousticContext.model_validate(forbidden)


@pytest.mark.parametrize(
    "bad_range",
    [
        {"start_s": 10.0, "end_s": 10.0},
        {"start_s": 20.0, "end_s": 10.0},
        {"start_s": -1.0, "end_s": 5.0},
        {"start_s": float("nan"), "end_s": 5.0},
        {"start_s": True, "end_s": 5.0},
        {"start_s": 1.0, "end_s": 5.0, "extra": 1},
    ],
)
def test_time_range_rejects_invalid_bounds(bad_range: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TimeRange.model_validate(bad_range)


def test_plan_payload_enforces_closed_operations_strength_and_range() -> None:
    with pytest.raises(ValidationError):
        PlanPayload(operation="restore", target="speech", explanation="Restore")
    with pytest.raises(ValidationError):
        PlanPayload(operation="restore", cleanup_profile="studio", explanation="Restore")
    with pytest.raises(ValidationError):
        PlanPayload(
            operation="clarify",
            time_range=TimeRange(start_s=1.0, end_s=2.0),
            explanation="Clarify?",
        )
    with pytest.raises(ValidationError):
        PlanPayload(operation="clean", cleanup_profile="studio", strength=0.05, explanation="Clean")
    with pytest.raises(ValidationError):
        PlanPayload(
            operation="clean",
            cleanup_profile="studio",
            strength=True,
            explanation="Clean",
        )


def test_generation_schema_covers_all_five_operations_without_dangling_refs() -> None:
    schema = assistant._schema()
    rendered = json.dumps(schema)
    assert "$ref" not in rendered
    by_op = {b["properties"]["operation"]["const"]: b for b in schema["anyOf"]}
    assert set(by_op) == {
        "clean",
        "smart_safe",
        "restore",
        "extract",
        "remove",
        "spectral_repair",
        "preview",
        "playback",
        "export",
        "undo",
        "redo",
        "unit_override",
        "reset",
        "ui_action",
        "clarify",
    }
    assert "studio_pro" in by_op["clean"]["properties"]["cleanup_profile"]["enum"]
    assert by_op["restore"]["properties"]["cleanup_profile"] == {"type": "null"}
    assert by_op["clarify"]["properties"]["time_range"] == {"type": "null"}
    assert all(branch["additionalProperties"] is False for branch in schema["anyOf"])


# --- Audio-aware telemetry grounding (rules) --------------------------------


def test_rules_recommends_extract_speech_and_studio_pro_when_music_risk_high() -> None:
    ctx = AcousticContext(
        duration_s=45.0,
        sample_rate=48000,
        channels=2,
        music_risk=0.82,
        noise_risk=0.65,
        available_profiles=["production", "studio", "studio_pro", "lowband"],
        separation_available=True,
    )
    plan = plan_audio(AudioPlanRequest(message="Clean this recording", acoustic_context=ctx))
    assert plan.operation == "extract"
    assert plan.target == "speech"
    assert plan.cleanup_profile == "studio_pro"
    assert "82%" in plan.explanation


def test_rules_falls_back_to_clean_when_music_risk_high_but_separation_unavailable() -> None:
    ctx = AcousticContext(
        music_risk=0.85,
        noise_risk=0.60,
        available_profiles=["production", "studio", "studio_pro"],
        separation_available=False,
    )
    plan = plan_audio(AudioPlanRequest(message="What do you recommend?", acoustic_context=ctx))
    assert plan.operation == "clean"
    assert plan.cleanup_profile == "studio_pro"


def test_rules_recommends_restore_on_steep_lowpass_bandwidth() -> None:
    ctx = AcousticContext(
        bandwidth_shape="steep_lowpass",
        estimated_cutoff_hz=3800.0,
        noise_risk=0.2,
        available_profiles=["production", "studio", "studio_pro"],
    )
    plan = plan_audio(AudioPlanRequest(message="Fix this audio", acoustic_context=ctx))
    assert plan.operation == "restore"
    assert plan.target is None
    assert plan.cleanup_profile is None
    assert "3800 Hz" in plan.explanation


def test_rules_recommends_studio_pro_or_studio_on_high_noise_or_reverb() -> None:
    ctx_pro = AcousticContext(
        noise_risk=0.7,
        reverberation_risk=0.5,
        available_profiles=["production", "studio", "studio_pro"],
    )
    plan_pro = plan_audio(AudioPlanRequest(message="Enhance this clip", acoustic_context=ctx_pro))
    assert plan_pro.operation == "clean"
    assert plan_pro.cleanup_profile == "studio_pro"

    ctx_legacy = AcousticContext(
        reverberation_risk=0.65,
        available_profiles=["production", "studio"],
    )
    plan_legacy = plan_audio(
        AudioPlanRequest(message="Clean this clip", acoustic_context=ctx_legacy)
    )
    assert plan_legacy.operation == "clean"
    assert plan_legacy.cleanup_profile == "studio"


def test_rules_recommends_lowband_on_rumble_and_production_on_clean_hum_audio() -> None:
    ctx_rumble = AcousticContext(noise_risk=0.1, reverberation_risk=0.1, rumble_confidence=0.75)
    assert (
        plan_audio(
            AudioPlanRequest(message="Improve the voice", acoustic_context=ctx_rumble)
        ).cleanup_profile
        == "lowband"
    )

    ctx_clean = AcousticContext(
        noise_risk=0.05,
        reverberation_risk=0.10,
        hum_confidence=0.7,
        available_profiles=["production", "studio_pro"],
    )
    assert (
        plan_audio(
            AudioPlanRequest(message="Clean my voice", acoustic_context=ctx_clean)
        ).cleanup_profile
        == "production"
    )

    ctx_ui_noisy = AcousticContext(
        noise_floor_db=-42.0, available_profiles=["production", "studio_pro"]
    )
    assert (
        plan_audio(
            AudioPlanRequest(message="Clean my voice", acoustic_context=ctx_ui_noisy)
        ).cleanup_profile
        == "studio_pro"
    )

    ctx_ui_clean = AcousticContext(
        noise_floor_db=-66.0, available_profiles=["production", "studio_pro"]
    )
    assert (
        plan_audio(
            AudioPlanRequest(message="Enhance this", acoustic_context=ctx_ui_clean)
        ).cleanup_profile
        == "production"
    )

    ctx_minimal = AcousticContext(duration_s=10.0, available_profiles=["production", "studio_pro"])
    assert (
        plan_audio(
            AudioPlanRequest(message="Clean this", acoustic_context=ctx_minimal)
        ).cleanup_profile
        == "studio_pro"
    )


def test_rules_explicit_studio_pro_and_restore_commands() -> None:
    plan_pro = plan_audio(AudioPlanRequest(message="Clean with studio pro"))
    assert plan_pro.operation == "clean" and plan_pro.cleanup_profile == "studio_pro"

    plan_restore = plan_audio(AudioPlanRequest(message="Restore missing high frequencies"))
    assert plan_restore.operation == "restore" and plan_restore.cleanup_profile is None


# --- Strength & time-range parsing ------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected_strength"),
    [
        ("Clean the voice and keep 20% natural room tone", 0.8),
        ("Clean my voice at 50% strength", 0.5),
        ("Clean my voice at half strength", 0.5),
        ("Gently clean my voice", 0.6),
        ("Clean my voice and preserve natural room tone", 0.75),
    ],
)
def test_rules_parses_strength_phrases(message: str, expected_strength: float) -> None:
    plan = plan_audio(AudioPlanRequest(message=message))
    assert plan.operation == "clean"
    assert plan.strength == pytest.approx(expected_strength)


@pytest.mark.parametrize(
    "bad_strength_msg",
    [
        "Clean my voice at 0% strength",
        "Clean my voice at 150% strength",
        "Clean my voice and keep 100% natural room tone",
    ],
)
def test_rules_clarifies_invalid_strength_percentages(bad_strength_msg: str) -> None:
    plan = plan_audio(AudioPlanRequest(message=bad_strength_msg))
    assert plan.operation == "clarify"


@pytest.mark.parametrize(
    ("message", "op", "target", "start_s", "end_s", "strength"),
    [
        ("Remove the siren from 0:15 to 0:28", "remove", "siren", 15.0, 28.0, 1.0),
        (
            "Remove background music between 15 and 28 seconds",
            "remove",
            "background music",
            15.0,
            28.0,
            1.0,
        ),
        ("Clean my voice from 1:02:03 to 1:02:10", "clean", None, 3723.0, 3730.0, 1.0),
        ("Clean my voice between 1 and 2 minutes", "clean", None, 60.0, 120.0, 1.0),
        ("Remove dog barking at 0:10-0:20 at 50%", "remove", "dog barking", 10.0, 20.0, 0.5),
    ],
)
def test_rules_parses_time_spans_and_strength_together(
    message: str,
    op: str,
    target: str | None,
    start_s: float,
    end_s: float,
    strength: float,
) -> None:
    plan = plan_audio(AudioPlanRequest(message=message))
    assert plan.operation == op
    assert plan.target == target
    assert plan.time_range == TimeRange(start_s=start_s, end_s=end_s)
    assert plan.strength == pytest.approx(strength)


def test_rules_validates_time_ranges_against_order_and_duration() -> None:
    assert (
        plan_audio(AudioPlanRequest(message="Remove siren from 0:28 to 0:15")).operation
        == "clarify"
    )
    assert (
        plan_audio(AudioPlanRequest(message="Remove siren in the last 10 seconds")).operation
        == "clarify"
    )

    ctx = AcousticContext(duration_s=30.0)
    assert (
        plan_audio(
            AudioPlanRequest(message="Remove siren from 0:35 to 0:40", acoustic_context=ctx)
        ).operation
        == "clarify"
    )
    assert (
        plan_audio(
            AudioPlanRequest(message="Remove siren from 0:10 to 0:40", acoustic_context=ctx)
        ).operation
        == "clarify"
    )
    # Within 0.5s rounding tolerance clamps cleanly to duration_s.
    clamped = plan_audio(
        AudioPlanRequest(message="Remove siren from 0:10 to 0:30.2", acoustic_context=ctx)
    )
    assert clamped.operation == "remove"
    assert clamped.time_range == TimeRange(start_s=10.0, end_s=30.0)


# --- Ollama keep_alive & telemetry forwarding -------------------------------


def test_ollama_keep_alive_env_and_telemetry_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "ollama")
    observed: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        observed.update(url=url, payload=payload, headers=headers)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "operation": "clean",
                        "target": None,
                        "cleanup_profile": "studio_pro",
                        "model_size": "large",
                        "quality": "high",
                        "strength": 0.8,
                        "time_range": {"start_s": 5.0, "end_s": 15.0},
                        "explanation": "Clean dialogue with studio_pro keeping 20% room tone.",
                    }
                )
            }
        }

    monkeypatch.setattr(assistant, "_post_json", fake_post)
    ctx = AcousticContext(
        duration_s=40.0,
        sample_rate=48000,
        channels=1,
        noise_risk=0.78,
        reverberation_risk=0.45,
        bandwidth_shape="fullband",
        estimated_cutoff_hz=19000.0,
        noise_floor_db=-41.5,
        integrated_lufs=-20.0,
        available_profiles=["production", "studio", "studio_pro"],
        separation_available=True,
    )
    plan = plan_audio(
        AudioPlanRequest(
            message="Clean from 0:05 to 0:15 and keep 20% natural room tone",
            acoustic_context=ctx,
        )
    )
    assert plan.provider == "ollama"
    assert plan.cleanup_profile == "studio_pro"
    assert plan.strength == pytest.approx(0.8)
    assert plan.time_range == TimeRange(start_s=5.0, end_s=15.0)
    assert observed["payload"]["keep_alive"] == "5m"
    sys_msg = observed["payload"]["messages"][0]["content"]
    assert "noise_risk: 0.78" in sys_msg
    assert "available_profiles: production, studio, studio_pro" in sys_msg

    monkeypatch.setenv("HAWAVOCLEAN_OLLAMA_KEEP_ALIVE", "0")
    plan_audio(AudioPlanRequest(message="Clean"))
    assert observed["payload"]["keep_alive"] == 0

    monkeypatch.setenv("HAWAVOCLEAN_OLLAMA_KEEP_ALIVE", "10m")
    plan_audio(AudioPlanRequest(message="Clean"))
    assert observed["payload"]["keep_alive"] == "10m"


@pytest.mark.parametrize("bad", ["", "5m; rm -rf", "-2", "999999", "forever"])
def test_invalid_keep_alive_fails_closed(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_OLLAMA_KEEP_ALIVE", bad)
    with pytest.raises(AssistantError, match="keep_alive"):
        assistant._ollama_keep_alive()
    status = assistant_status()
    assert status["available"] is False


def test_ollama_hallucinated_out_of_bounds_time_range_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "ollama")
    monkeypatch.setattr(
        assistant,
        "_post_json",
        lambda *_args: {
            "message": {
                "content": json.dumps(
                    {
                        "operation": "clean",
                        "target": None,
                        "cleanup_profile": "studio",
                        "model_size": "large",
                        "quality": "high",
                        "strength": 1.0,
                        "time_range": {"start_s": 50.0, "end_s": 60.0},
                        "explanation": "Clean.",
                    }
                )
            }
        },
    )
    plan = plan_audio(
        AudioPlanRequest(
            message="Clean from 0:50 to 1:00",
            acoustic_context=AcousticContext(duration_s=30.0),
        )
    )
    assert plan.operation == "clarify"
    assert plan.time_range is None


# --- Server contracts, truthful capabilities & CLI strength/time-range ------


def test_v1_strategy_contracts_validate_studio_pro_device_strength_and_range() -> None:
    manual = ManualStrategyV1(
        kind="manual",
        route="studio_pro",
        strength=0.7,
        time_range=(2.0, 8.0),
    )
    assert manual.strength == 0.7 and manual.time_range == (2.0, 8.0)

    sep = SeparationStrategyV1(
        kind="separate",
        operation="extract",
        prompt="speech",
        cleanup_profile="studio_pro",
        strength=0.6,
        time_range=(1.0, 5.0),
    )
    assert sep.device == "auto" and sep.cleanup_profile == "studio_pro"

    for bad in (
        {"strength": 0.05},
        {"strength": 1.5},
        {"strength": True},
        {"time_range": (5.0, 5.0)},
        {"time_range": (5.0, 2.0)},
        {"time_range": (-1.0, 2.0)},
        {"time_range": (float("nan"), 2.0)},
        {"time_range": (True, 2.0)},
    ):
        with pytest.raises(ValidationError):
            ManualStrategyV1.model_validate({"kind": "manual", "route": "production", **bad})
        with pytest.raises(ValidationError):
            SeparationStrategyV1.model_validate(
                {"kind": "separate", "operation": "extract", "prompt": "speech", **bad}
            )


def test_capabilities_v1_truthful_smart_safe_studio_pro_and_hybrid_separation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hawavoclean.guard.perceptual_mos as pm
    import hawavoclean.separation as sep_mod

    monkeypatch.setattr(
        pm,
        "perceptual_scorer_status",
        lambda: {
            "available": True,
            "backend": "acoustic_proxy",
            "calibrated": False,
            "qualified_ranker": False,
            "maturity": "experimental",
            "providers": ["cpu"],
            "reason": "Uncalibrated proxy; Smart Safe is experimental.",
        },
    )

    def fake_probe(device: str = "auto") -> RuntimeStatus:
        assert device == "auto"
        return RuntimeStatus(
            available=True,
            reason="Vocal/music separation ready via MLX RoFormer.",
            providers=("mlx",),
        )

    monkeypatch.setattr(sep_mod, "probe_separation_capabilities", fake_probe)
    caps = {c.capability_id: c for c in capabilities_v1().capabilities}
    assert "studio_pro" in caps
    assert caps["smart_safe"].available is True
    assert caps["smart_safe"].maturity == "experimental"
    assert caps["smart_safe"].reason == "Uncalibrated proxy; Smart Safe is experimental."
    assert caps["separate"].available is True
    assert caps["separate"].maturity == "experimental"
    assert caps["separate"].providers == ["mlx"]
    assert "RoFormer" in (caps["separate"].reason or "")


def test_server_plan_endpoint_accepts_acoustic_context_and_rejects_paths() -> None:
    manager = JobManager()
    app = create_app("tok", None, job_manager=manager, min_free_bytes=0)
    headers = {"X-Hawa-Token": "tok"}
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            ok = client.post(
                "/api/v1/assistant/plan",
                headers=headers,
                json={
                    "message": "Clean this recording from 0:05 to 0:15",
                    "acoustic_context": {
                        "duration_s": 30.0,
                        "noise_risk": 0.75,
                        "available_profiles": ["production", "studio", "studio_pro"],
                    },
                },
            )
            assert ok.status_code == 200, ok.text
            body = ok.json()
            assert body["operation"] == "clean"
            assert body["cleanup_profile"] == "studio_pro"
            assert body["time_range"] == {"start_s": 5.0, "end_s": 15.0}

            rejected = client.post(
                "/api/v1/assistant/plan",
                headers=headers,
                json={
                    "message": "Clean",
                    "acoustic_context": {"path": "/etc/passwd"},
                },
            )
            assert rejected.status_code == 422
    finally:
        manager.shutdown()


def test_job_record_roundtrips_strength_and_time_range_into_cli_argv(tmp_path: Path) -> None:
    record = JobRecord(
        job_id="j_blend",
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.hawavoclean.json",
        profile="production",
        overwrite=True,
        mode="natural",
        strength=0.6,
        time_range=(2.5, 7.5),
    )
    restored = JobRecord.from_storage(json.loads(json.dumps(record.storage_record())))
    assert restored.strength == 0.6
    assert restored.time_range == (2.5, 7.5)
    argv = default_command(restored)
    assert "--strength" in argv and "0.6" in argv
    assert "--time-range" in argv and "2.5" in argv and "7.5" in argv
    snap = restored.snapshot()
    assert snap["strength"] == 0.6 and snap["time_range"] == [2.5, 7.5]


def test_cli_process_strength_and_time_range_preserves_outside_samples_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    wave = (
        0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.03 * np.sin(2.0 * np.pi * 4000.0 * t)
    ).astype(np.float32)
    src = tmp_path / "clip.wav"
    sf.write(src, wave, sr, subtype="PCM_24")
    orig_pcm, _ = sf.read(src, dtype="float32", always_2d=True)

    # Surgical time-range + strength run on PCM_24: exact bit-identity outside [start_idx, end_idx)
    start_idx = int(round(0.25 * sr))
    end_idx = int(round(0.75 * sr))
    out_range = tmp_path / "out_range.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hawavoclean",
            "process",
            str(src),
            "-o",
            str(out_range),
            "--profile",
            "production",
            "--strength",
            "0.5",
            "--time-range",
            "0.25",
            "0.75",
            "--overwrite",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert out_range.is_file()
    report = load_json_report(out_range.with_suffix(".hawavoclean.json"))
    assert report.output.sha256 == hash_file(out_range)
    assert report.input.sha256 == hash_file(src)
    out_pcm, out_sr = sf.read(out_range, dtype="float32", always_2d=True)
    assert out_sr == sr and out_pcm.shape == orig_pcm.shape
    assert np.array_equal(out_pcm[:start_idx], orig_pcm[:start_idx])
    assert np.array_equal(out_pcm[end_idx:], orig_pcm[end_idx:])

    # Surgical time-range on FLOAT WAV anchored at clip start (0.0 -> 0.5) preserves float32 bit-identity
    src_float = tmp_path / "clip_float.wav"
    sf.write(src_float, wave, sr, subtype="FLOAT")
    orig_float, _ = sf.read(src_float, dtype="float32", always_2d=True)
    out_float = tmp_path / "out_float_range.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hawavoclean",
            "process",
            str(src_float),
            "-o",
            str(out_float),
            "--profile",
            "production",
            "--time-range",
            "0.0",
            "0.5",
            "--overwrite",
        ],
    )
    with pytest.raises(SystemExit) as exc_float:
        cli.main()
    assert exc_float.value.code == 0
    assert sf.info(str(out_float)).subtype == "FLOAT"
    out_float_pcm, _ = sf.read(out_float, dtype="float32", always_2d=True)
    half_idx = int(round(0.5 * sr))
    assert np.array_equal(out_float_pcm[half_idx:], orig_float[half_idx:])

    # Whole-clip strength blend run
    out_strength = tmp_path / "out_strength.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hawavoclean",
            "process",
            str(src),
            "-o",
            str(out_strength),
            "--profile",
            "production",
            "--strength",
            "0.6",
            "--overwrite",
        ],
    )
    with pytest.raises(SystemExit) as exc2:
        cli.main()
    assert exc2.value.code == 0
    report2 = load_json_report(out_strength.with_suffix(".hawavoclean.json"))
    assert report2.output.sha256 == hash_file(out_strength)

    # Validation failures fail closed with non-zero exit code.
    for bad_args in (
        ["--strength", "0.05"],
        ["--strength", "nan"],
        ["--time-range", "0.8", "0.2"],
        ["--strength", "0.5", "--passes", "2"],
        ["--time-range", "5.0", "6.0"],
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hawavoclean",
                "process",
                str(src),
                "-o",
                str(tmp_path / "bad.wav"),
                "--profile",
                "production",
                *bad_args,
            ],
        )
        with pytest.raises(SystemExit) as bad_exc:
            cli.main()
        assert bad_exc.value.code != 0


def test_assistant_handles_out_of_range_timestamps_and_rejects_control_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "rules")

    # Timestamps exceeding 24h (86400s) return a clarification rather than ValidationError.
    for msg in (
        "Remove siren from 90000s to 95000s",
        "Remove siren from 2000 minutes to 3000 minutes",
        "Remove siren from 99:99:99 to 99:99:99.5",
    ):
        plan = plan_audio(AudioPlanRequest(message=msg))
        assert plan.operation == "clarify"
        assert plan.needs_clarification is True

    # Non-whitespace ASCII control characters fail closed at AudioPlanRequest validation (HTTP 422).
    for bad in ("Remove dog\x00barking", "Remove dog\x07barking", "Clean\x1b[31mvoice"):
        with pytest.raises(ValidationError, match="control characters"):
            AudioPlanRequest(message=bad)

    # Normal whitespace (\n, \t) remains valid.
    multiline = plan_audio(AudioPlanRequest(message="Remove dog\nbarking"))
    assert multiline.operation == "remove"
    assert multiline.target == "dog barking"


def test_universal_chat_controls_all_studio_operations_and_modifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "rules")
    ctx = AcousticContext(
        duration_s=120.0,
        sample_rate=48000,
        channels=2,
        available_profiles=["production", "studio", "studio_pro", "dereverb", "lowband"],
        separation_available=True,
    )

    # 1. Compound full-chain DSP command in one sentence
    compound = plan_audio(
        AudioPlanRequest(
            message=(
                "Remove room echo, fix harsh S sounds, mouth clicks, clipping, plosives, "
                "warm up voice, leveler, cut rumble below 90 Hz, room tone at -48 dB, "
                "preserve stereo M/S, and master for Podcast"
            ),
            acoustic_context=ctx,
        )
    )
    assert compound.operation == "clean"
    assert compound.cleanup_profile == "dereverb"
    assert compound.hpf_enabled is True and compound.hpf_cutoff_hz == 90.0
    assert compound.pro_repair is not None
    assert compound.pro_repair.deess is True
    assert compound.pro_repair.declick is True
    assert compound.pro_repair.declip is True
    assert compound.pro_repair.deplosive is True
    assert compound.pro_repair.tonal_restoration is True
    assert compound.pro_repair.level_rider is True
    assert compound.ambience_floor_db == -48.0
    assert compound.stereo_mode == "true_stereo_ms"
    assert compound.delivery_preset == "podcast"

    # 2. HPF disable, delivery presets & stereo modes
    hpf_off = plan_audio(AudioPlanRequest(message="Clean voice, disable HPF, YouTube -14 LUFS"))
    assert hpf_off.hpf_enabled is False and hpf_off.delivery_preset == "streaming"
    netflix = plan_audio(AudioPlanRequest(message="Master voice for Netflix dialogue gated"))
    assert netflix.delivery_preset == "dialogue_gated"
    ebu = plan_audio(AudioPlanRequest(message="Broadcast EBU R128 and split speakers"))
    assert ebu.delivery_preset == "ebu_r128" and ebu.stereo_mode == "split_speakers"

    # 3. 2D Spectral box repair from UI [Band: ...] chip AND explicit Hz + time range
    band = plan_audio(
        AudioPlanRequest(
            message="Attenuate this hum by 18 dB [Band: 140–220 Hz · 0:04.2–0:06.1]",
            acoustic_context=ctx,
        )
    )
    assert band.operation == "spectral_repair" and band.spectral_repair is not None
    assert band.spectral_repair.f_low_hz == 140.0 and band.spectral_repair.f_high_hz == 220.0
    assert band.spectral_repair.attenuation_db == 18.0

    explicit_spec = plan_audio(
        AudioPlanRequest(
            message="Notch 2.5 to 4 kHz from 0:10 to 0:15 by 24 dB",
            acoustic_context=ctx,
        )
    )
    assert (
        explicit_spec.operation == "spectral_repair" and explicit_spec.spectral_repair is not None
    )
    assert explicit_spec.spectral_repair.f_low_hz == 2500.0
    assert explicit_spec.spectral_repair.f_high_hz == 4000.0

    # 4. Warm preview & Smart Safe
    prev = plan_audio(
        AudioPlanRequest(message="Preview 0:10 to 0:18 with studio pro", acoustic_context=ctx)
    )
    assert prev.operation == "preview" and prev.cleanup_profile == "studio_pro"
    assert prev.time_range is not None and prev.time_range.start_s == 10.0

    ss = plan_audio(AudioPlanRequest(message="Auto Smart Safe with Podcast loudness"))
    assert ss.operation == "smart_safe" and ss.delivery_preset == "podcast"

    # 5. Playback, A/B, Delta, Seek, Loop, Mix
    assert (
        plan_audio(AudioPlanRequest(message="Hear what was removed")).playback_action
        == "hear_removed"
    )
    assert plan_audio(AudioPlanRequest(message="Play original")).playback_action == "play_original"
    assert plan_audio(AudioPlanRequest(message="Play cleaned")).playback_action == "play_cleaned"
    assert plan_audio(AudioPlanRequest(message="Pause")).playback_action == "pause"
    assert plan_audio(AudioPlanRequest(message="Play")).playback_action == "play"
    assert plan_audio(AudioPlanRequest(message="Loop on")).playback_action == "loop_on"
    assert plan_audio(AudioPlanRequest(message="Loop off")).playback_action == "loop_off"
    seek_p = plan_audio(AudioPlanRequest(message="Seek to 0:45"))
    assert (
        seek_p.operation == "playback"
        and seek_p.playback_action == "seek"
        and seek_p.seek_s == 45.0
    )
    mix_p = plan_audio(AudioPlanRequest(message="Set mix to 65%"))
    assert (
        mix_p.operation == "playback"
        and mix_p.playback_action == "set_mix"
        and mix_p.wet_dry_mix == 0.65
    )

    # 6. Undo / Redo / Reset / Unit Override / Export
    assert plan_audio(AudioPlanRequest(message="Undo")).operation == "undo"
    assert plan_audio(AudioPlanRequest(message="Redo")).operation == "redo"
    assert plan_audio(AudioPlanRequest(message="Reset all settings")).operation == "reset"
    uo = plan_audio(AudioPlanRequest(message="Revert unit 3"))
    assert uo.operation == "unit_override" and uo.unit_override is not None
    assert uo.unit_override.unit_id == 3 and uo.unit_override.action == "force_original"
    exp = plan_audio(AudioPlanRequest(message="Export as FLAC for Podcast"))
    assert (
        exp.operation == "export"
        and exp.export_format == "flac"
        and exp.delivery_preset == "podcast"
    )

    # 7. Sorani Kurdish commands
    ckb_derev = plan_audio(AudioPlanRequest(message="دەنگدانەوەی ژوور لاببە", acoustic_context=ctx))
    assert ckb_derev.operation == "clean" and ckb_derev.cleanup_profile == "dereverb"
    ckb_music = plan_audio(AudioPlanRequest(message="مۆسیقا لاببە"))
    assert ckb_music.operation == "remove" and ckb_music.target == "background music"
    ckb_clean = plan_audio(AudioPlanRequest(message="ژاوەژاو لاببە"))
    assert ckb_clean.operation == "clean"
    ckb_delta = plan_audio(AudioPlanRequest(message="بیستنی لابراوەکان"))
    assert ckb_delta.operation == "playback" and ckb_delta.playback_action == "hear_removed"
    ckb_undo = plan_audio(AudioPlanRequest(message="گەڕانەوە"))
    assert ckb_undo.operation == "undo"
