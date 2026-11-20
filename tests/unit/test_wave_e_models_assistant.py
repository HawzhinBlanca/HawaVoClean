"""Wave E unit tests: Hash-verified model manager, packaged assistant, and Windows path policy."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import hawavoclean.assistant as assistant
import hawavoclean.model_manager as model_manager
from hawavoclean.enhancement.dereverb_neural import DEREVERB_CHECKPOINT_SHA256
from hawavoclean.enhancement.studio_pro import STUDIO_PRO_CHECKPOINT_SHA256
from hawavoclean.restoration.neural_bwe import LAVASR_V2_CONFIG_SHA256, LAVASR_V2_WEIGHTS_SHA256
from hawavoclean.server.app import SESSION_PATH, create_app
from hawavoclean.server.policy import (
    PathPolicyError,
    allowed_roots,
    is_blocked_windows_system_path,
    resolve_client_output_path,
    resolve_client_path,
)

pytestmark = pytest.mark.unit

TOKEN = "wave-e-test-token"
AUTH_HEADERS = {"X-Hawa-Token": TOKEN}


# ---------------------------------------------------------------------------
# 1. Hash-Verified Model Manager
# ---------------------------------------------------------------------------


def test_model_registry_single_source_of_truth_and_https_urls(tmp_path: Path) -> None:
    registry = model_manager.model_registry(models_root=tmp_path)
    expected_ids = {
        "deepfilternet3",
        "mossformer2_se_48k",
        "tfgridnet_dereverb",
        "nisqa_v2",
        "lavasr_v2",
        "roformer",
    }
    assert set(registry) == expected_ids

    assert registry["deepfilternet3"].bundled is True
    assert registry["deepfilternet3"].download_urls == ()
    assert registry["mossformer2_se_48k"].expected_sha256 == STUDIO_PRO_CHECKPOINT_SHA256
    assert registry["tfgridnet_dereverb"].expected_sha256 == DEREVERB_CHECKPOINT_SHA256
    assert registry["lavasr_v2"].expected_sha256 == LAVASR_V2_WEIGHTS_SHA256
    assert {a.expected_sha256 for a in registry["lavasr_v2"].artifacts} == {
        LAVASR_V2_CONFIG_SHA256,
        LAVASR_V2_WEIGHTS_SHA256,
    }

    for spec in registry.values():
        assert len(spec.expected_sha256) == 64
        for url in spec.download_urls:
            assert url.startswith("https://")
            assert model_manager.validate_download_url(url) == url


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "http://huggingface.co/model.pt",
        "file:///etc/passwd",
        "ftp://example.com/model.pt",
        "https://user:pass@example.com/model.pt",
        "https://example.com/model.pt\nHost: evil.com",
        "https:///no-host",
    ],
)
def test_validate_download_url_refuses_non_https_and_malformed(bad_url: str) -> None:
    with pytest.raises(model_manager.ModelDownloadError):
        model_manager.validate_download_url(bad_url)


def test_list_models_does_not_crash_when_optional_weights_missing(tmp_path: Path) -> None:
    statuses = model_manager.list_models(models_root=tmp_path)
    by_id = {status.model_id: status for status in statuses}
    assert len(by_id) == 6

    dfn = by_id["deepfilternet3"]
    assert dfn.bundled is True
    assert dfn.downloadable is False
    assert dfn.installed is True
    assert dfn.verified is True
    assert dfn.size_bytes > 0
    assert dfn.path is not None

    for opt_id in ("mossformer2_se_48k", "tfgridnet_dereverb", "nisqa_v2", "lavasr_v2", "roformer"):
        opt = by_id[opt_id]
        assert opt.bundled is False
        assert opt.downloadable is True
        assert opt.installed is False
        assert opt.verified is False
        assert opt.size_bytes == 0
        assert opt.path is None


def test_install_model_success_with_progress_and_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified-nisqa-checkpoint-bytes-for-wave-e-unit-test"
    payload_sha = hashlib.sha256(payload).hexdigest()

    # Patch single-source NISQA digest helper for a fast deterministic offline test.
    monkeypatch.setattr(model_manager, "_load_nisqa_constants", lambda: ("nisqa.tar", payload_sha))

    seen_progress: list[tuple[int, int | None]] = []

    def fake_fetcher(url: str) -> tuple[list[bytes], int]:
        assert url.startswith("https://")
        return [payload[:16], payload[16:32], payload[32:]], len(payload)

    result = model_manager.install_model(
        "nisqa_v2",
        progress_cb=lambda done, total: seen_progress.append((done, total)),
        fetcher=fake_fetcher,
        models_root=tmp_path,
    )

    assert result.installed is True
    assert result.verified is True
    assert result.already_installed is False
    assert result.sha256 == payload_sha
    assert result.size_bytes == len(payload)
    target = Path(result.path)
    assert target.is_file()
    assert target.read_bytes() == payload
    # Progress reported incrementally up to full byte count.
    assert seen_progress == [
        (16, len(payload)),
        (32, len(payload)),
        (len(payload), len(payload)),
    ]
    # Zero leftover .part temporary files.
    assert list(target.parent.glob("*.part")) == []

    # list_models immediately reflects installed + verified status.
    by_id = {m.model_id: m for m in model_manager.list_models(models_root=tmp_path)}
    assert by_id["nisqa_v2"].installed is True
    assert by_id["nisqa_v2"].verified is True
    assert by_id["nisqa_v2"].size_bytes == len(payload)


def test_install_model_sha256_mismatch_fails_closed_and_deletes_part_file(
    tmp_path: Path,
) -> None:
    def tampered_fetcher(_url: str) -> list[bytes]:
        return [b"corrupted-or-tampered-bytes"]

    with pytest.raises(model_manager.ModelHashMismatchError, match="SHA-256 mismatch"):
        model_manager.install_model(
            "nisqa_v2",
            fetcher=tampered_fetcher,
            models_root=tmp_path,
        )

    nisqa_dir = tmp_path / "nisqa"
    assert not (nisqa_dir / "nisqa.tar").exists()
    assert list(nisqa_dir.glob("*.part")) == []


def test_install_model_rejects_unknown_model_id(tmp_path: Path) -> None:
    for bad_id in ("unknown_model", "../etc/passwd", "deepfilternet99", ""):
        with pytest.raises(model_manager.UnknownModelError):
            model_manager.install_model(bad_id, models_root=tmp_path)


def test_install_model_multi_artifact_cleans_up_all_parts_on_second_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_cfg = b"sample_rate: 48000\n"
    cfg_sha = hashlib.sha256(good_cfg).hexdigest()
    monkeypatch.setattr(model_manager, "LAVASR_V2_CONFIG_SHA256", cfg_sha)

    def fetcher(url: str) -> list[bytes]:
        if url.endswith("config.yaml"):
            return [good_cfg]
        return [b"bad-weights-digest"]

    with pytest.raises(model_manager.ModelHashMismatchError):
        model_manager.install_model("lavasr_v2", fetcher=fetcher, models_root=tmp_path)

    lavasr_dir = tmp_path / "lavasr" / "enhancer_v2"
    # Neither artifact may be promoted if the second fails digest verification.
    assert not (lavasr_dir / "config.yaml").exists()
    assert not (lavasr_dir / "pytorch_model.bin").exists()
    assert list(lavasr_dir.glob("*.part")) == []


# ---------------------------------------------------------------------------
# 2. FastAPI /api/v1/models endpoints
# ---------------------------------------------------------------------------


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(work))
    monkeypatch.setenv("HAWAVOCLEAN_STATE_DIR", str(state))
    app = create_app(TOKEN, ui_dir=None)
    return TestClient(app, base_url="http://127.0.0.1")


def test_api_v1_models_requires_auth_and_lists_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_client(tmp_path, monkeypatch) as client:
        unauth = client.get("/api/v1/models")
        assert unauth.status_code == 401

        # Session Bearer authentication works cleanly.
        sess_res = client.post(SESSION_PATH, headers=AUTH_HEADERS)
        assert sess_res.status_code == 200
        bearer = {"Authorization": f"Bearer {sess_res.json()['sessionToken']}"}

        res = client.get("/api/v1/models", headers=bearer)
        assert res.status_code == 200
        body = res.json()
        assert body["schema_version"] == 1
        ids = {item["model_id"] for item in body["models"]}
        assert ids == {
            "deepfilternet3",
            "mossformer2_se_48k",
            "tfgridnet_dereverb",
            "nisqa_v2",
            "lavasr_v2",
            "roformer",
        }
        # Never leak raw home directory path in JSON responses.
        assert str(Path.home()) not in res.text


def test_api_v1_models_install_unknown_404_mismatch_422_and_success_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _make_client(tmp_path, monkeypatch) as client:
        # Unknown model -> 404
        unknown = client.post("/api/v1/models/does-not-exist/install", headers=AUTH_HEADERS)
        assert unknown.status_code == 404
        assert unknown.json()["error"] == "not_found"

        # Tampered digest -> 422 fail closed
        monkeypatch.setattr(
            model_manager,
            "_https_stream_fetcher",
            lambda _url: (iter([b"tampered-bytes"]), 14),
        )
        bad = client.post("/api/v1/models/nisqa_v2/install", headers=AUTH_HEADERS)
        assert bad.status_code == 422
        assert bad.json()["error"] == "integrity_error"

        # Valid digest -> 200 + refreshed capabilities + updated models
        good_payload = b"valid-nisqa-tar-content-for-api-test"
        good_sha = hashlib.sha256(good_payload).hexdigest()
        monkeypatch.setattr(model_manager, "_load_nisqa_constants", lambda: ("nisqa.tar", good_sha))
        monkeypatch.setattr(
            model_manager,
            "_https_stream_fetcher",
            lambda _url: (iter([good_payload]), len(good_payload)),
        )

        ok = client.post("/api/v1/models/nisqa_v2/install", headers=AUTH_HEADERS)
        assert ok.status_code == 200
        body = ok.json()
        assert body["ok"] is True
        assert body["result"]["model_id"] == "nisqa_v2"
        assert body["result"]["installed"] is True
        assert body["result"]["verified"] is True
        assert body["result"]["sha256"] == good_sha
        assert isinstance(body["capabilities"], list) and len(body["capabilities"]) > 0
        nisqa_entry = next(m for m in body["models"] if m["model_id"] == "nisqa_v2")
        assert nisqa_entry["installed"] is True
        assert nisqa_entry["verified"] is True


# ---------------------------------------------------------------------------
# 3. Packaged Local-Chat Assistant
# ---------------------------------------------------------------------------


def test_assistant_resolves_writable_app_data_root_before_dev_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "app_data"
    dev_dir = tmp_path / "dev_local_ai"
    state_dir.mkdir()
    dev_dir.mkdir()

    monkeypatch.setenv("HAWAVOCLEAN_STATE_DIR", str(state_dir))
    monkeypatch.delenv("HAWAVOCLEAN_CHAT_MODEL", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_OLLAMA_URL", raising=False)
    monkeypatch.setattr(assistant, "_LOCAL_ROOT", dev_dir)

    assert assistant.chat_runtime_config_path() == state_dir.resolve() / "chat-runtime.json"

    # Unconfigured returns None (never invalid "default").
    assert assistant._chat_runtime_config() == (None, "http://127.0.0.1:11434")

    # Dev fallback works when packaged config does not exist.
    (dev_dir / "chat-runtime.json").write_text(
        json.dumps({"model": "qwen2.5:7b", "url": "http://127.0.0.1:11434"}),
        encoding="utf-8",
    )
    assert assistant._chat_runtime_config() == ("qwen2.5:7b", "http://127.0.0.1:11434")

    # Writable app_data_root() config takes precedence in packaged apps.
    (state_dir / "chat-runtime.json").write_text(
        json.dumps({"model": "llama3.2:3b", "url": "http://127.0.0.1:11435"}),
        encoding="utf-8",
    )
    assert assistant._chat_runtime_config() == ("llama3.2:3b", "http://127.0.0.1:11435")


def test_assistant_ollama_tags_probe_auto_picks_real_model_and_reports_truthful_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "auto")
    monkeypatch.delenv("HAWAVOCLEAN_CHAT_MODEL", raising=False)
    monkeypatch.delenv("HAWAVOCLEAN_OLLAMA_URL", raising=False)
    monkeypatch.setattr(assistant, "_LOCAL_ROOT", tmp_path / "dev")

    # Case A: Ollama reachable with embedding model + real chat model -> skips embed, picks chat model.
    monkeypatch.setattr(
        assistant,
        "_probe_ollama_tags",
        lambda _url, **_kw: (True, ["nomic-embed-text:latest", "qwen3:8b"]),
    )
    status = assistant.assistant_status()
    assert status["available"] is True
    assert status["ollama_reachable"] is True
    assert status["installed_models"] == ["nomic-embed-text:latest", "qwen3:8b"]
    assert status["local_model"] == "qwen3:8b"
    assert status["active_mode"] == "local_llm"

    observed: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        observed.update(url=url, payload=payload, headers=headers)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "operation": "clean",
                        "target": None,
                        "cleanup_profile": "studio",
                        "model_size": "large",
                        "quality": "high",
                        "strength": 1.0,
                        "time_range": None,
                        "explanation": "Clean dialogue with studio profile.",
                    }
                )
            }
        }

    monkeypatch.setattr(assistant, "_post_json", fake_post)
    plan = assistant.plan_audio(assistant.AudioPlanRequest(message="Clean this voice recording"))
    assert plan.provider == "ollama"
    assert observed["payload"]["model"] == "qwen3:8b"

    # Case B: Ollama unreachable -> deterministic rules mode cleanly, never sends "default".
    monkeypatch.setattr(assistant, "_probe_ollama_tags", lambda _url, **_kw: (False, []))
    monkeypatch.setattr(
        assistant,
        "_post_json",
        lambda *_args: pytest.fail("unreachable Ollama must not be called with 'default'"),
    )
    status_down = assistant.assistant_status()
    assert status_down["available"] is True  # auto still has deterministic rules
    assert status_down["ollama_reachable"] is False
    assert status_down["installed_models"] == []
    assert status_down["local_model"] is None
    assert status_down["active_mode"] == "deterministic_rules"

    fallback_plan = assistant.plan_audio(assistant.AudioPlanRequest(message="Clean my voice"))
    assert fallback_plan.provider == "rules"
    assert fallback_plan.fallback_reason is not None

    # Case C: provider="ollama" with Ollama reachable but zero models installed -> unavailable.
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "ollama")
    monkeypatch.setattr(assistant, "_probe_ollama_tags", lambda _url, **_kw: (True, []))
    status_empty = assistant.assistant_status()
    assert status_empty["available"] is False
    assert status_empty["ollama_reachable"] is True
    assert status_empty["installed_models"] == []
    assert status_empty["local_model"] is None
    assert status_empty["active_mode"] == "deterministic_rules"


def test_assistant_tags_probe_rejects_non_loopback_urls() -> None:
    with pytest.raises(assistant.AssistantError, match="loopback"):
        assistant._probe_ollama_tags("https://evil.example.com")
    with pytest.raises(assistant.AssistantError, match="loopback"):
        assistant._probe_ollama_tags("http://192.168.1.50:11434")


def test_api_v1_assistant_status_returns_truthful_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "auto")
    monkeypatch.delenv("HAWAVOCLEAN_CHAT_MODEL", raising=False)
    monkeypatch.setattr(assistant, "_LOCAL_ROOT", tmp_path / "dev")
    monkeypatch.setattr(
        assistant, "_probe_ollama_tags", lambda _url, **_kw: (True, ["llama3.2:3b"])
    )
    with _make_client(tmp_path, monkeypatch) as client:
        res = client.get("/api/v1/assistant/status", headers=AUTH_HEADERS)
        assert res.status_code == 200
        body = res.json()
        assert body["available"] is True
        assert body["ollama_reachable"] is True
        assert body["installed_models"] == ["llama3.2:3b"]
        assert body["local_model"] == "llama3.2:3b"
        assert body["active_mode"] == "local_llm"


# ---------------------------------------------------------------------------
# 4. Windows Secondary Drives Path Policy
# ---------------------------------------------------------------------------


def test_windows_policy_allows_secondary_drives_and_blocks_system_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "work").mkdir()
    (tmp_path / "home").mkdir()

    drive_d = tmp_path / "drive_D"
    drive_e = tmp_path / "drive_E"
    (drive_d / "Projects").mkdir(parents=True)
    (drive_d / "Windows" / "System32").mkdir(parents=True)
    (drive_d / "Program Files" / "Vendor").mkdir(parents=True)
    (drive_d / "ProgramData" / "Secrets").mkdir(parents=True)
    (drive_e / "Audio").mkdir(parents=True)

    clip_d = drive_d / "Projects" / "interview.wav"
    clip_d.write_bytes(b"RIFF")
    clip_e = drive_e / "Audio" / "field.wav"
    clip_e.write_bytes(b"RIFF")

    roots = allowed_roots(platform="win32", windows_drives=[drive_d, drive_e])
    assert drive_d.resolve() in roots
    assert drive_e.resolve() in roots

    # Secondary drive user files are allowed for read and write.
    assert (
        resolve_client_path(
            str(clip_d),
            must_exist=True,
            platform="win32",
            windows_drives=[drive_d, drive_e],
        )
        == clip_d.resolve()
    )
    assert (
        resolve_client_path(
            str(clip_e),
            must_exist=True,
            platform="win32",
            windows_drives=[drive_d, drive_e],
        )
        == clip_e.resolve()
    )
    out = resolve_client_output_path(
        str(drive_d / "Projects" / "interview_clean.wav"),
        platform="win32",
        windows_drives=[drive_d, drive_e],
    )
    assert out == drive_d.resolve() / "Projects" / "interview_clean.wav"

    # Windows system directories are strictly blocked even on secondary drives.
    for blocked in (
        drive_d / "Windows" / "System32" / "cmd.exe",
        drive_d / "windows" / "win.ini",
        drive_d / "Program Files" / "Vendor" / "app.dll",
        drive_d / "Program Files (x86)" / "Legacy" / "app.exe",
        drive_d / "ProgramData" / "Secrets" / "key.pem",
        drive_d / "Projects" / ".." / "Windows" / "win.ini",
    ):
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(
                str(blocked),
                platform="win32",
                windows_drives=[drive_d, drive_e],
            )
        assert exc.value.status == 403 and exc.value.code == "forbidden", str(blocked)

        with pytest.raises(PathPolicyError) as out_exc:
            resolve_client_output_path(
                str(blocked),
                platform="win32",
                windows_drives=[drive_d, drive_e],
            )
        assert out_exc.value.status == 403 and out_exc.value.code == "forbidden", str(blocked)


def test_windows_policy_preserves_symlink_escape_and_unc_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    (tmp_path / "work").mkdir()
    (tmp_path / "home").mkdir()

    drive_d = tmp_path / "drive_D"
    (drive_d / "Projects").mkdir(parents=True)
    (drive_d / "Windows").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"secret")

    # UNC / network device paths are refused on Windows.
    for unc in ("\\\\fileserver\\share\\clip.wav", "//fileserver/share/clip.wav"):
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(unc, platform="win32", windows_drives=[drive_d])
        assert exc.value.status == 403 and exc.value.code == "forbidden"

    # Symlink escaping into protected Windows dir or outside root is refused.
    link_sys = drive_d / "Projects" / "link_sys"
    link_out = drive_d / "Projects" / "link_out"
    try:
        os.symlink(str(drive_d / "Windows"), link_sys)
        os.symlink(str(outside), link_out)
    except OSError:
        pytest.skip("Symlink creation requires privilege on this filesystem")

    for escaped in (link_sys / "win.ini", link_out / "secret.wav"):
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(str(escaped), platform="win32", windows_drives=[drive_d])
        assert exc.value.status == 403 and exc.value.code == "forbidden", str(escaped)

    assert is_blocked_windows_system_path(
        (drive_d / "Windows" / "win.ini").resolve(),
        platform="win32",
        windows_drives=[drive_d],
    )
