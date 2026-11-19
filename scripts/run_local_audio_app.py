#!/usr/bin/env python3
"""Launch the source checkout's native app and its optional private local chat."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hawavoclean.assistant import _chat_runtime_config
from hawavoclean.process_supervisor import ProcessSupervisor

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local-ai"


def executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for folder in ("/opt/homebrew/bin", "/usr/local/bin"):
        path = Path(folder) / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError(f"Install {name} before launching this source checkout.")


def desktop_command() -> list[str]:
    result = subprocess.run(
        [executable("node"), "-p", "require('electron')"],
        cwd=ROOT / "desktop",
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    electron = Path(result.stdout.strip())
    if not electron.is_absolute() or not electron.is_file():
        raise RuntimeError("The local Electron installation is incomplete.")
    return [str(electron), str(ROOT / "desktop")]


def launch_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        HAWAVOCLEAN_CHAT_PROVIDER="auto",
        OLLAMA_NO_CLOUD="1",
        HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND=json.dumps(
            [str(ROOT / ".venv/bin/python"), "-m", "hawavoclean.cli"]
        ),
    )
    environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + environment.get(
        "PATH", "/usr/bin:/bin"
    )
    environment.pop("ELECTRON_RUN_AS_NODE", None)
    environment.pop("HAWAVOCLEAN_DESKTOP_SELFTEST", None)
    return environment


def service_running(url: str) -> bool:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: object, **_kwargs: object) -> None:
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(url + "/api/version", timeout=1) as response:
            value = json.loads(response.read(4096))
        return isinstance(value, dict) and isinstance(value.get("version"), str)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def prepare_assets() -> None:
    for folder, output, commands in (
        ("ui", "dist/index.html", (("tsc", "-b"), ("vite", "build"))),
        ("desktop", "dist/main.js", (("tsc", "-p", "tsconfig.json"),)),
    ):
        if not (ROOT / folder / output).is_file():
            for command in commands:
                tool = ROOT / folder / "node_modules/.bin" / command[0]
                subprocess.run([str(tool), *command[1:]], cwd=ROOT / folder, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check installed paths only")
    arguments = parser.parse_args()
    command, environment = desktop_command(), launch_environment()
    chat = (LOCAL / "chat-runtime.json").is_file() and (LOCAL / "ollama-models/manifests").is_dir()
    model, url = _chat_runtime_config()  # Includes strict loopback URL validation.
    assets = (".venv/bin/python", "ui/dist/index.html", "desktop/dist/main.js")
    missing = [str(ROOT / path) for path in assets if not (ROOT / path).is_file()]
    if arguments.check:
        values = {"desktop": command[0], "engine": str(ROOT / assets[0]), "cloud_enabled": False}
        values.update(ready=not missing, missing=missing, local_chat_installed=chat)
        values.update(local_chat_model=model, local_chat_url=url)
        print(json.dumps(values, indent=2))
        return 0 if not missing else 1
    prepare_assets()
    owned_chat: ProcessSupervisor | None = None
    desktop: ProcessSupervisor | None = None
    try:
        if chat and not service_running(url):
            chat_environment = {
                **environment,
                "OLLAMA_MODELS": str(LOCAL / "ollama-models"),
                "OLLAMA_HOST": urlsplit(url).netloc,
            }
            streams: dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            args = [executable("ollama"), "serve"]
            owned_chat = ProcessSupervisor.spawn(args, env=chat_environment, **streams)
            deadline = time.monotonic() + 20
            while not service_running(url):
                if owned_chat.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Local chat could not start; close this window and try again."
                    )
                time.sleep(0.2)
        desktop = ProcessSupervisor.spawn(command, cwd=ROOT / "desktop", env=environment)
        return desktop.process.wait()
    finally:
        for process in (desktop, owned_chat):
            if process is not None:
                process.terminate_tree(3)
                process.close()


def interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"HawaVoClean could not start: {error}", file=__import__("sys").stderr)
        raise SystemExit(1) from None
