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


def test_model_manager_stream_variants_ast_fallback_and_already_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ctypes
    import importlib.util
    import io
    import sys as sys_mod
    import types
    import urllib.error
    import urllib.request
    from typing import cast

    from hawavoclean import model_manager as mm
    from hawavoclean.model_manager import (
        ModelDownloadError,
        ModelNotDownloadableError,
        _https_stream_fetcher,
        _HttpsOnlyRedirectHandler,
        _load_nisqa_constants,
        _nisqa_discovered_path,
        _normalize_fetch_stream,
        install_model,
    )
    from hawavoclean.server import policy as policy_mod

    # 1. AST fallback when torch is absent
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    fname, sha = _load_nisqa_constants()
    assert fname.endswith(".tar") and len(sha) == 64

    # 2. _nisqa_discovered_path env overrides
    custom_weights = tmp_path / "custom_nisqa.tar"
    monkeypatch.setenv("HAWAVOCLEAN_NISQA_WEIGHTS", str(custom_weights))
    assert _nisqa_discovered_path(None) == custom_weights.resolve()
    monkeypatch.delenv("HAWAVOCLEAN_NISQA_WEIGHTS", raising=False)
    monkeypatch.setenv("HAWAVOCLEAN_MODEL_DIR", str(tmp_path / "mdir"))
    assert _nisqa_discovered_path(None).parent.name == "nisqa"
    monkeypatch.delenv("HAWAVOCLEAN_MODEL_DIR", raising=False)
    assert _nisqa_discovered_path(tmp_path / "mroot") == tmp_path / "mroot" / "nisqa" / fname

    # 3. _normalize_fetch_stream variants: bytes, (bytes, total), file-like .read()
    chunks_b, tot_b = _normalize_fetch_stream(b"hello")
    assert list(chunks_b) == [b"hello"] and tot_b == 5
    chunks_tb, tot_tb = _normalize_fetch_stream(cast(Any, (b"abc", 3)))
    assert list(chunks_tb) == [b"abc"] and tot_tb == 3
    chunks_io, tot_io = _normalize_fetch_stream(io.BytesIO(b"streamed-bytes"))
    assert b"".join(chunks_io) == b"streamed-bytes" and tot_io is None

    # 4. _HttpsOnlyRedirectHandler rejects HTTP downgrade redirects
    handler = _HttpsOnlyRedirectHandler()
    req = urllib.request.Request("https://example.com/model.bin")
    with pytest.raises(ModelDownloadError, match="Only HTTPS"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://example.com/downgrade")

    # 5. _https_stream_fetcher success, open error, and mid-stream read error
    class _FakeResp:
        def __init__(self, payload: bytes, *, fail_on_read: bool = False) -> None:
            self._buf = io.BytesIO(payload)
            self._fail = fail_on_read
            self.headers = {"Content-Length": str(len(payload))}
            self.closed = False

        def read(self, n: int = -1) -> bytes:
            if self._fail:
                raise urllib.error.URLError("connection reset")
            return self._buf.read(n)

        def close(self) -> None:
            self.closed = True

    class _FakeOpener:
        def __init__(self, resp: _FakeResp | None = None, *, fail_open: bool = False) -> None:
            self._resp = resp
            self._fail_open = fail_open

        def open(self, _req: Any, timeout: float = 0.0) -> _FakeResp:
            _ = timeout
            if self._fail_open or self._resp is None:
                raise urllib.error.URLError("dns failure")
            return self._resp

    good_resp = _FakeResp(b"chunk-payload")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _FakeOpener(good_resp))
    stream_it, content_len = _https_stream_fetcher("https://example.com/weights.bin")
    assert content_len == len(b"chunk-payload")
    assert b"".join(stream_it) == b"chunk-payload"
    assert good_resp.closed

    bad_read = _FakeResp(b"x", fail_on_read=True)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _FakeOpener(bad_read))
    bad_it, _ = _https_stream_fetcher("https://example.com/weights.bin")
    with pytest.raises(ModelDownloadError, match="Interrupted"):
        list(bad_it)
    assert bad_read.closed

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _FakeOpener(fail_open=True))
    with pytest.raises(ModelDownloadError, match="Could not download"):
        _https_stream_fetcher("https://example.com/weights.bin")

    # 6. Bundled deepfilternet3 returns already_installed=True; oversized download fails closed
    dfn_res = install_model("deepfilternet3")
    assert dfn_res.installed and dfn_res.verified and dfn_res.already_installed

    import dataclasses

    models_root = tmp_path / "models_install"
    payload = b"verified-model-bytes"
    payload_sha = hashlib.sha256(payload).hexdigest()
    spec = mm.get_model_spec("nisqa_v2", models_root=models_root)
    patched_spec = dataclasses.replace(
        spec,
        expected_sha256=payload_sha,
        bundled=False,
        artifacts=(
            mm.ModelArtifact(
                filename=spec.artifacts[0].filename,
                expected_sha256=payload_sha,
                download_urls=spec.artifacts[0].download_urls,
            ),
        ),
    )
    monkeypatch.setattr(mm, "get_model_spec", lambda _mid, **_kw: patched_spec)
    # Oversized payload rejected before hash check
    with pytest.raises(ModelDownloadError, match="exceeded maximum"):
        mm._stream_and_verify_artifact(
            patched_spec.artifacts[0],
            models_root / "nisqa" / patched_spec.artifacts[0].filename,
            fetcher=lambda _u: [b"", b"123456"],
            max_bytes=4,
            progress_cb=None,
            bytes_before=0,
            total_hint=None,
        )

    # Install once, then verify fetcher=None short-circuits as already_installed
    first = install_model("nisqa_v2", models_root=models_root, fetcher=lambda _u: [b"", payload])
    assert first.installed and first.verified and not first.already_installed
    second = install_model("nisqa_v2", models_root=models_root, fetcher=None)
    assert second.installed and second.verified and second.already_installed

    # Missing download URL raises ModelNotDownloadableError
    no_url_spec = dataclasses.replace(
        patched_spec,
        download_urls=(),
        artifacts=(mm.ModelArtifact(filename="a.bin", expected_sha256=payload_sha),),
    )
    monkeypatch.setattr(mm, "get_model_spec", lambda _mid, **_kw: no_url_spec)
    with pytest.raises(ModelNotDownloadableError):
        install_model("nisqa_v2", models_root=models_root, fetcher=lambda _u: [payload])

    # 7. Windows policy drive enumeration & env system-dir blocking exercised cross-platform
    monkeypatch.setenv("SYSTEMDRIVE", "c")
    assert policy_mod._windows_system_drive() == "C:"

    fake_kernel32 = types.SimpleNamespace(
        # Bits for C (2), D (3), E (4), Z (25)
        GetLogicalDrives=lambda: (1 << 2) | (1 << 3) | (1 << 4) | (1 << 25),
        GetDriveTypeW=lambda root: {
            "C:\\": 3,  # fixed (system)
            "D:\\": 3,  # fixed (secondary)
            "E:\\": 2,  # removable (secondary)
            "Z:\\": 4,  # remote/network (excluded)
        }.get(root, 0),
    )
    monkeypatch.setattr(sys_mod, "platform", "win32")
    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=fake_kernel32), raising=False
    )
    drives = policy_mod.windows_local_drives(include_system_drive=False)
    drive_strs = [str(d) for d in drives]
    assert "D:\\" in drive_strs and "E:\\" in drive_strs
    assert "Z:\\" not in drive_strs
    all_drives = policy_mod.windows_local_drives(include_system_drive=True)
    assert any("C:" in str(d) for d in all_drives)

    fake_sysroot = tmp_path / "WinSysRoot"
    fake_sysroot.mkdir()
    monkeypatch.setenv("SYSTEMROOT", str(fake_sysroot))
    assert policy_mod.is_blocked_windows_system_path(
        (fake_sysroot / "cmd.exe").resolve(), platform="win32"
    )


def test_assistant_transport_and_runtime_rss_cross_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import io
    import urllib.error
    import urllib.request

    from hawavoclean import assistant as ast_mod
    from hawavoclean import runtime as rt_mod

    class _Resp:
        def __init__(self, body: bytes, status: int = 200) -> None:
            self._body = body
            self.status = status

        def read(self, n: int = -1) -> bytes:
            return self._body if n < 0 else self._body[:n]

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    class _Opener:
        def __init__(self, resp: _Resp | None = None, *, exc: Exception | None = None) -> None:
            self._resp = resp
            self._exc = exc

        def open(self, _req: Any, timeout: float = 0.0) -> _Resp:
            _ = timeout
            if self._exc is not None:
                raise self._exc
            assert self._resp is not None
            return self._resp

    # 1. _probe_ollama_tags: 200 mixed entries, non-200, oversized, bad JSON, non-dict, non-list, error
    payload_tags = json.dumps(
        {
            "models": [
                "not-a-dict",
                {"name": "default"},
                {"name": "nomic-embed-text:latest"},
                {"name": "custom-chat:7b"},
                {"model": "custom-chat:7b"},
                {"model": "phi4:latest"},
            ]
        }
    ).encode("utf-8")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _Opener(_Resp(payload_tags)))
    ok, models = ast_mod._probe_ollama_tags("http://127.0.0.1:11434")
    assert ok and models == ["nomic-embed-text:latest", "custom-chat:7b", "phi4:latest"]
    assert (
        ast_mod._pick_installed_chat_model(["nomic-embed-text", "custom-chat:7b"])
        == "custom-chat:7b"
    )
    assert ast_mod._model_is_installed("phi4", models)
    assert ast_mod._model_is_installed("phi4:latest", models)

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b"{}", status=503))
    )
    assert ast_mod._probe_ollama_tags("http://127.0.0.1:11434") == (False, [])

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_a: _Opener(_Resp(b"x" * (ast_mod.MAX_RESPONSE_BYTES + 2))),
    )
    assert ast_mod._probe_ollama_tags("http://127.0.0.1:11434") == (False, [])

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b"not-json")))
    assert ast_mod._probe_ollama_tags("http://127.0.0.1:11434") == (False, [])

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b"[1,2]")))
    assert ast_mod._probe_ollama_tags("http://127.0.0.1:11434") == (False, [])

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b'{"models":null}'))
    )
    assert ast_mod._probe_ollama_tags("http://127.0.0.1:11434") == (True, [])

    # 2. _post_json: success, oversized, non-dict, bad JSON, URLError
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b'{"ok":true}')))
    assert ast_mod._post_json("http://127.0.0.1:11434/chat", {"a": 1}, {}) == {"ok": True}

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_a: _Opener(_Resp(b"x" * (ast_mod.MAX_RESPONSE_BYTES + 2))),
    )
    with pytest.raises(ast_mod.AssistantError, match="too large"):
        ast_mod._post_json("http://127.0.0.1:11434/chat", {}, {})

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b"[1,2]")))
    with pytest.raises(ast_mod.AssistantError, match="invalid response"):
        ast_mod._post_json("http://127.0.0.1:11434/chat", {}, {})

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_a: _Opener(_Resp(b"{bad")))
    with pytest.raises(ast_mod.AssistantError, match="invalid response"):
        ast_mod._post_json("http://127.0.0.1:11434/chat", {}, {})

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *_a: _Opener(exc=urllib.error.URLError("down"))
    )
    with pytest.raises(ast_mod.AssistantUnavailableError):
        ast_mod._post_json("http://127.0.0.1:11434/chat", {}, {})

    # 3. runtime.py: _linux_current_rss_bytes & process_current_rss_bytes branches
    real_open = builtins.open

    def _fake_statm(content: str) -> Any:
        def _opener(path: Any, *args: Any, **kwargs: Any) -> Any:
            if str(path) == "/proc/self/statm":
                return io.StringIO(content)
            return real_open(path, *args, **kwargs)

        return _opener

    monkeypatch.setattr(builtins, "open", _fake_statm("1000 250 100 0 0 0 0\n"))
    assert rt_mod._linux_current_rss_bytes() > 0

    monkeypatch.setattr(builtins, "open", _fake_statm("1000\n"))
    with pytest.raises(OSError, match="Malformed"):
        rt_mod._linux_current_rss_bytes()

    monkeypatch.setattr(builtins, "open", _fake_statm("1000 0\n"))
    with pytest.raises(OSError, match="Non-positive"):
        rt_mod._linux_current_rss_bytes()

    import sys as sys_mod

    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(sys_mod, "platform", "linux")
    monkeypatch.setattr(rt_mod, "_linux_current_rss_bytes", lambda: 12345678)
    assert rt_mod.process_current_rss_bytes() == 12345678

    monkeypatch.setattr(sys_mod, "platform", "win32")
    monkeypatch.setattr(
        rt_mod, "_windows_rss_bytes", lambda *, current=False: 87654321 if current else 99999999
    )
    assert rt_mod.process_current_rss_bytes() == 87654321

    def _boom(*, current: bool = False) -> int:
        _ = current
        raise OSError("probe failed")

    monkeypatch.setattr(rt_mod, "_windows_rss_bytes", _boom)
    monkeypatch.setattr(rt_mod, "process_peak_rss_bytes", lambda: 424242)
    assert rt_mod.process_current_rss_bytes() == 424242
