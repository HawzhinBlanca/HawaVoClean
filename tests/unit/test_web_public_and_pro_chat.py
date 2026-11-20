"""Tests for production web-public deployment lockdown and Pro conversational chat."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean import assistant, cli
from hawavoclean.assistant import AcousticContext, AudioPlanRequest, PlanPayload, _rules_plan
from hawavoclean.server import app as app_mod
from hawavoclean.server.app import NATIVE_SOURCE_PATH, create_app
from hawavoclean.server.policy import (
    PathPolicyError,
    allowed_roots,
    is_web_public_mode,
    resolve_client_output_path,
    resolve_client_path,
)

AUTH_TOKEN = "web-public-secret-token-123456"
ROOT_HEADERS = {"X-Hawa-Token": AUTH_TOKEN}


def _write_wav(path: Path, seconds: float = 0.5, sr: int = 16000) -> Path:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wave = (0.15 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(path, wave, sr, subtype="PCM_16")
    return path


def _plan_offline(message: str, ctx: AcousticContext | None = None) -> PlanPayload:
    return _rules_plan(
        AudioPlanRequest(
            message=message,
            acoustic_context=ctx,
            allow_cloud_text=False,
        )
    )


# ---------------------------------------------------------------------------
# Pillar 1: Pro Conversational Audio Mastery (Diagnosis, UI Actions, Presets)
# ---------------------------------------------------------------------------


def test_plan_payload_requires_ui_action_field() -> None:
    with pytest.raises(ValueError, match="ui_action"):
        PlanPayload(operation="ui_action", explanation="Missing action")


def test_acoustic_diagnosis_with_no_context_en_and_kurdish() -> None:
    en = _plan_offline("What's wrong with this audio?")
    assert en.operation == "ui_action"
    assert en.ui_action == "diagnose"
    assert "Load an audio or video clip first" in en.explanation

    ckb = _plan_offline("کێشەی ئەم دەنگە چییە؟")
    assert ckb.operation == "ui_action"
    assert ckb.ui_action == "diagnose"
    assert "تۆمارێک بار بکە" in ckb.explanation


def test_acoustic_diagnosis_reports_all_measured_flags_and_clean_state() -> None:
    noisy_ctx = AcousticContext(
        duration_s=12.5,
        sample_rate=48000,
        channels=2,
        integrated_lufs=-21.4,
        noise_floor_db=-44.0,
        noise_risk=0.65,
        reverberation_risk=0.55,
        music_risk=0.72,
        clipping_risk=0.035,
        rumble_confidence=0.8,
        bandwidth_shape="steep_lowpass",
    )
    en = _plan_offline("Diagnose this audio", noisy_ctx)
    assert en.operation == "ui_action"
    assert en.ui_action == "diagnose"
    for expected in (
        "12.5s",
        "48 kHz",
        "2ch",
        "-21.4 LUFS",
        "-44.0 dB",
        "elevated noise",
        "room reverb",
        "background music",
        "clipping",
        "low-frequency rumble",
        "narrowband high-frequency rolloff",
    ):
        assert expected in en.explanation

    ckb = _plan_offline("کێشەی ئەم دەنگە چییە؟", noisy_ctx)
    assert ckb.ui_action == "diagnose"
    assert "کێشە دۆزراوەکان" in ckb.explanation

    # Fallback noise_floor_db > -55 when noise_risk is None
    floor_only = AcousticContext(
        duration_s=4.0, sample_rate=16000, channels=1, noise_floor_db=-48.0
    )
    en_floor = _plan_offline("How loud is this?", floor_only)
    assert "mono" in en_floor.explanation
    assert "noticeable background noise" in en_floor.explanation

    # Clean clip branch (EN + Kurdish)
    clean_ctx = AcousticContext(
        duration_s=8.0,
        sample_rate=48000,
        channels=1,
        integrated_lufs=-16.0,
        noise_floor_db=-72.0,
    )
    en_clean = _plan_offline("Check the levels", clean_ctx)
    assert "looks clean" in en_clean.explanation
    ckb_clean = _plan_offline("کێشەی دەنگە چییە", clean_ctx)
    assert "تۆمارەکە پاکە" in ckb_clean.explanation


@pytest.mark.parametrize(
    ("prompt", "expected_action"),
    [
        ("Switch to Pro Studio", "switch_pro"),
        ("ستۆدیۆی پرۆ", "switch_pro"),
        ("Switch to Simple mode", "switch_simple"),
        ("شێوازی سادە", "switch_simple"),
        ("Switch to Kurdish", "switch_kurdish"),
        ("بگۆڕە بۆ کوردی", "switch_kurdish"),
        ("Switch to English", "switch_english"),
        ("بگۆڕە بۆ ئینگلیزی", "switch_english"),
        ("Open models", "open_models"),
        ("مۆدێلەکان بکەوە", "open_models"),
    ],
)
def test_ui_action_commands_en_and_kurdish(prompt: str, expected_action: str) -> None:
    plan = _plan_offline(prompt)
    assert plan.operation == "ui_action"
    assert plan.ui_action == expected_action


def test_pro_studio_natural_presets_and_relative_refinements() -> None:
    ctx = AcousticContext(available_profiles=["production", "studio", "studio_pro", "dereverb"])

    # Relative refinements
    softer = _plan_offline("Dial it back, too processed", ctx)
    assert softer.operation == "clean"
    assert softer.cleanup_profile == "studio"
    assert softer.strength == pytest.approx(0.55)
    assert softer.ambience_floor_db == pytest.approx(-48.0)

    stronger = _plan_offline("Make it stronger, still noisy", ctx)
    assert stronger.operation == "clean"
    assert stronger.cleanup_profile == "studio_pro"
    assert stronger.strength == pytest.approx(1.0)

    natural = _plan_offline("Keep it more natural with room tone", ctx)
    assert natural.operation == "clean"
    assert natural.ambience_floor_db == pytest.approx(-45.0)
    assert natural.strength == pytest.approx(0.65)

    less_harsh = _plan_offline("Make it less harsh with softer s", ctx)
    assert less_harsh.operation == "clean"
    assert less_harsh.pro_repair is not None
    assert less_harsh.pro_repair.deess is True
    assert less_harsh.pro_repair.tonal_restoration is True

    # Natural studio style presets
    radio = _plan_offline("Give it a radio voice", ctx)
    assert radio.operation == "clean"
    assert radio.delivery_preset == "podcast"
    assert radio.pro_repair is not None
    assert radio.pro_repair.tonal_restoration is True
    assert radio.pro_repair.deplosive is True
    assert radio.pro_repair.level_rider is True

    bright = _plan_offline("Make it brighter and less muddy", ctx)
    assert bright.operation == "clean"
    assert bright.hpf_enabled is True
    assert bright.hpf_cutoff_hz == pytest.approx(100.0)
    assert bright.pro_repair is not None
    assert bright.pro_repair.tonal_restoration is True
    assert bright.pro_repair.deess is True

    louder = _plan_offline("Make it louder, it's too quiet", ctx)
    assert louder.operation == "clean"
    assert louder.delivery_preset == "podcast"
    assert louder.pro_repair is not None
    assert louder.pro_repair.level_rider is True

    phone = _plan_offline("Rescue this phone recording", ctx)
    assert phone.operation == "clean"
    assert phone.hpf_enabled is True
    assert phone.hpf_cutoff_hz == pytest.approx(85.0)
    assert phone.pro_repair is not None
    assert phone.pro_repair.declick is True
    assert phone.pro_repair.declip is True

    film = _plan_offline("Prep this for film dialogue", ctx)
    assert film.operation == "clean"
    assert film.cleanup_profile == "dereverb"
    assert film.delivery_preset == "dialogue_gated"
    assert film.ambience_floor_db == pytest.approx(-48.0)


def test_schema_includes_ui_action_enum_without_refs() -> None:
    schema = assistant._schema()
    ui_branch = next(
        b for b in schema["anyOf"] if b["properties"]["operation"]["const"] == "ui_action"
    )
    assert set(ui_branch["properties"]["ui_action"]["enum"]) == {
        "switch_pro",
        "switch_simple",
        "switch_kurdish",
        "switch_english",
        "open_models",
        "diagnose",
    }
    assert ui_branch["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Pillar 2: Production Website Deployment Mode (--web-public + CORS + Lockdown)
# ---------------------------------------------------------------------------


def test_policy_web_public_locks_allowed_roots_to_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(work_dir))
    monkeypatch.delenv("HAWAVOCLEAN_WEB_PUBLIC", raising=False)

    assert is_web_public_mode() is False
    assert is_web_public_mode(True) is True
    assert is_web_public_mode(False) is False

    monkeypatch.setenv("HAWAVOCLEAN_WEB_PUBLIC", "1")
    assert is_web_public_mode() is True
    assert allowed_roots() == [work_dir.resolve()]

    # Path inside user home (outside work dir) must be forbidden in web_public mode
    home_file = Path.home() / ".hawavoclean_web_public_test_nonexistent.wav"
    with pytest.raises(PathPolicyError) as exc_read:
        resolve_client_path(str(home_file), must_exist=False, web_public=True)
    assert exc_read.value.status == 403
    assert "web-public" in exc_read.value.message

    with pytest.raises(PathPolicyError) as exc_out:
        resolve_client_output_path(str(home_file), web_public=True)
    assert exc_out.value.status == 403
    assert "web-public" in exc_out.value.message

    # Path inside work_root succeeds
    ok_file = _write_wav(work_dir / "uploaded.wav")
    assert resolve_client_path(str(ok_file), must_exist=True, web_public=True) == ok_file.resolve()
    assert (
        resolve_client_output_path(str(work_dir / "out.wav"), web_public=True)
        == (work_dir / "out.wav").resolve()
    )


def test_create_app_validates_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    app_ok = create_app(
        AUTH_TOKEN,
        web_public=True,
        cors_origins=("https://studio.example.com/", "http://localhost:5173"),
    )
    assert app_ok.state.web_public is True

    for bad_origin in ("*", "null", "http://evil.example.com", "https://good.com\r\nX-Bad: 1"):
        with pytest.raises(ValueError, match="invalid CORS origin"):
            create_app(AUTH_TOKEN, web_public=True, cors_origins=(bad_origin,))

    # Env var fallback path
    monkeypatch.setenv("HAWAVOCLEAN_CORS_ORIGINS", "https://app.example.org")
    app_env = create_app(AUTH_TOKEN, web_public=True)
    assert app_env.state.web_public is True


def test_web_public_server_lockdown_and_upload_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_wav = _write_wav(outside_dir / "secret.wav")
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(work_dir))

    app = create_app(
        AUTH_TOKEN,
        web_public=True,
        cors_origins=("https://audio.example.com",),
    )
    with TestClient(app, base_url="https://audio.example.com") as client:
        # 0. Untrusted Host header is still rejected (DNS rebinding protection)
        evil_host = client.get(
            "/api/health",
            headers={**ROOT_HEADERS, "Host": "evil.example.com"},
        )
        assert evil_host.status_code == 403

        # 1. Health reports web_public deployment mode with configured public host + Origin
        health = client.get(
            "/api/health",
            headers={**ROOT_HEADERS, "Origin": "https://audio.example.com"},
        )
        assert health.status_code == 200
        assert health.json()["deployment_mode"] == "web_public"
        assert health.headers.get("access-control-allow-origin") == "https://audio.example.com"

        # 2. Native source registration & model install are forbidden
        reg = client.post(
            NATIVE_SOURCE_PATH,
            json={"path": str(outside_wav)},
            headers=ROOT_HEADERS,
        )
        assert reg.status_code == 403
        assert "web-public" in reg.json()["message"]

        install = client.post(
            "/api/v1/models/nisqa_v2/install",
            headers=ROOT_HEADERS,
        )
        assert install.status_code == 403
        assert "web-public" in install.json()["message"]

        # 3. Direct read of host file outside work dir is forbidden even with root token
        read_outside = client.post(
            "/api/analyze",
            json={"path": str(outside_wav)},
            headers=ROOT_HEADERS,
        )
        assert read_outside.status_code == 403

        # 4. Upload inside work_dir succeeds and can be analyzed
        wav_bytes = outside_wav.read_bytes()
        up = client.post(
            "/api/upload",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            headers=ROOT_HEADERS,
        )
        assert up.status_code == 200
        uploaded_path = up.json()["path"]
        assert Path(uploaded_path).resolve().is_relative_to(work_dir.resolve())

        analyzed = client.post(
            "/api/analyze",
            json={"path": uploaded_path},
            headers=ROOT_HEADERS,
        )
        assert analyzed.status_code == 200
        assert analyzed.json()["sample_rate"] == 16000


def test_cli_serve_passes_web_public_and_cors_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_server(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(app_mod, "run_server", fake_run_server)
    monkeypatch.setattr(cli, "run_server", fake_run_server, raising=False)

    ns = argparse.Namespace(
        host="0.0.0.0",
        port=8765,
        token="secret-token-123",
        token_env=None,
        token_stdin=False,
        ui_dir=None,
        web_public=True,
        cors_origins="https://studio.example.com, https://app.example.com",
    )
    rc = cli.cmd_serve(ns)
    assert rc == 0
    assert captured["web_public"] is True
    assert captured["cors_origins"] == ("https://studio.example.com", "https://app.example.com")
    assert captured["host"] == "0.0.0.0"
