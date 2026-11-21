"""The source launcher owns only the native app and chat service it starts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_local_audio_app.py"
spec = importlib.util.spec_from_file_location("local_audio_launcher", SCRIPT)
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "LOCAL", tmp_path / ".local-ai")
    for relative in (
        ".venv/bin/python",
        "ui/dist/index.html",
        "desktop/dist/main.js",
        ".local-ai/chat-runtime.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / ".local-ai/ollama-models/manifests").mkdir(parents=True)
    monkeypatch.setattr(
        launcher, "desktop_command", lambda: ["/native/electron", str(tmp_path / "desktop")]
    )
    monkeypatch.setattr(
        launcher, "_chat_runtime_config", lambda: ("qwen3:8b", "http://127.0.0.1:11435")
    )
    monkeypatch.setattr(launcher, "executable", lambda name: "/local/bin/" + name)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    return tmp_path


def supervisor() -> SimpleNamespace:
    return SimpleNamespace(
        process=SimpleNamespace(wait=Mock(return_value=0), poll=Mock(return_value=None)),
        terminate_tree=Mock(),
        close=Mock(),
    )


def test_check_does_not_start_processes_or_build(
    installed: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spawn, build = Mock(), Mock()
    monkeypatch.setattr(launcher.ProcessSupervisor, "spawn", spawn)
    monkeypatch.setattr(launcher, "prepare_assets", build)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check"])
    assert launcher.main() == 0
    spawn.assert_not_called()
    build.assert_not_called()
    assert '"cloud_enabled": false' in capsys.readouterr().out
    assert (installed / "ui/dist/index.html").is_file()


def test_existing_chat_service_is_never_owned_or_stopped(
    installed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = supervisor()
    spawn = Mock(return_value=native)
    monkeypatch.setattr(launcher.ProcessSupervisor, "spawn", spawn)
    monkeypatch.setattr(launcher, "service_running", lambda _: True)
    assert launcher.main() == 0
    assert spawn.call_count == 1
    assert spawn.call_args.args[0][0] == "/native/electron"
    assert spawn.call_args.kwargs["cwd"] == installed / "desktop"
    assert spawn.call_args.kwargs["env"]["HAWAVOCLEAN_CHAT_PROVIDER"] == "auto"
    assert spawn.call_args.kwargs["env"]["OLLAMA_NO_CLOUD"] == "1"
    native.terminate_tree.assert_called_once_with(3)
    native.close.assert_called_once()


def test_private_chat_uses_local_cache_and_is_cleaned_with_its_app(
    installed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat, native = supervisor(), supervisor()
    spawn = Mock(side_effect=[chat, native])
    monkeypatch.setattr(launcher.ProcessSupervisor, "spawn", spawn)
    healthy = iter([False, True])
    monkeypatch.setattr(launcher, "service_running", lambda _: next(healthy))
    assert launcher.main() == 0
    call = spawn.call_args_list[0]
    assert call.args[0] == ["/local/bin/ollama", "serve"]
    assert call.kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:11435"
    assert call.kwargs["env"]["OLLAMA_MODELS"] == str(installed / ".local-ai/ollama-models")
    for process in (chat, native):
        process.close.assert_called_once()


def test_failed_private_chat_is_cleaned_without_launching_desktop(
    installed: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = supervisor()
    chat.process.poll.return_value = 1
    spawn = Mock(return_value=chat)
    monkeypatch.setattr(launcher.ProcessSupervisor, "spawn", spawn)
    monkeypatch.setattr(launcher, "service_running", lambda _: False)
    with pytest.raises(RuntimeError, match="Local chat could not start"):
        launcher.main()
    assert spawn.call_count == 1
    assert spawn.call_args.kwargs["env"]["OLLAMA_MODELS"] == str(
        installed / ".local-ai/ollama-models"
    )
    chat.close.assert_called_once()
