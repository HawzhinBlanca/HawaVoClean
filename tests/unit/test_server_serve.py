"""``hawavoclean serve``: ready line on stdout (and nothing else), loopback
binding, token gate, clean exit on ``POST /api/shutdown``."""

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import hawavoclean.cli as cli
from hawavoclean import __version__
from hawavoclean.errors import ExitCode, InvalidUserInputError
from hawavoclean.server.app import _validate_loopback, bind_loopback_socket

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[2]


def _get(url: str, token: str | None = "t") -> tuple[int, Any]:
    req = urllib.request.Request(url, headers={"X-Hawa-Token": token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read().decode())


def test_serve_prints_one_ready_line_then_exits_on_shutdown() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "hawavoclean.cli", "serve", "--port", "0", "--token-stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    proc.stdin.write("t\n")
    proc.stdin.close()
    try:
        line = proc.stdout.readline()
        ready = json.loads(line)
        assert ready["event"] == "ready"
        if sys.platform != "win32":
            assert ready["pid"] == proc.pid
        else:
            assert isinstance(ready["pid"], int) and ready["pid"] > 0
        assert ready["version"] == __version__
        port = ready["port"]
        assert isinstance(port, int) and 1024 <= port <= 65535
        base = f"http://127.0.0.1:{port}"

        status, body = _get(f"{base}/api/health", token=None)
        assert status == 401 and body["error"] == "unauthorized"
        status, body = _get(f"{base}/api/health")
        assert status == 200 and body["ok"] is True and body["engine_pid"] == proc.pid
        # Loopback only: the port is not reachable on a non-loopback interface
        # (socket bound to 127.0.0.1 specifically).
        hostname_ip = None
        with contextlib.suppress(OSError):
            hostname_ip = socket.gethostbyname(socket.gethostname())
        if hostname_ip and not hostname_ip.startswith("127."):
            with pytest.raises(OSError):
                socket.create_connection((hostname_ip, port), timeout=1).close()

        t0 = time.monotonic()
        req = urllib.request.Request(
            f"{base}/api/shutdown", method="POST", headers={"X-Hawa-Token": "t"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert json.loads(resp.read().decode()) == {"ok": True}
        rc = proc.wait(timeout=5)
        assert time.monotonic() - t0 < 3.0
        assert rc == 0
        rest = proc.stdout.read()
        assert rest == "", f"stdout had more than the ready line: {rest!r}"
        err = proc.stderr.read()
        assert "listening on" in err
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_serve_rejects_non_loopback_host() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hawavoclean.cli",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
            "--token",
            "t",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=60,
    )
    assert proc.returncode == int(ExitCode.INVALID_USER_INPUT)
    assert proc.stdout == ""
    assert "loopback" in proc.stderr


def test_validate_loopback() -> None:
    assert _validate_loopback("127.0.0.1") == "127.0.0.1"
    assert _validate_loopback("127.0.0.2") == "127.0.0.2"
    assert _validate_loopback("localhost") == "127.0.0.1"
    assert _validate_loopback("") == "127.0.0.1"
    for bad in ("0.0.0.0", "192.168.1.5", "::1", "example.com", "8.8.8.8"):
        with pytest.raises(InvalidUserInputError):
            _validate_loopback(bad)


def test_bind_loopback_socket_assigns_port() -> None:
    sock = bind_loopback_socket("localhost", 0)
    try:
        host, port = sock.getsockname()[:2]
        assert host == "127.0.0.1" and port > 0
    finally:
        sock.close()


def test_cmd_serve_requires_token_and_checks_ui_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", "serve", "--port", "0"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2  # argparse: --token is required

    monkeypatch.setattr(
        sys,
        "argv",
        ["hawavoclean", "serve", "--port", "0", "--token", "t", "--ui-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == int(ExitCode.INVALID_USER_INPUT)  # no index.html

    monkeypatch.setattr(sys, "argv", ["hawavoclean", "serve", "--port", "0", "--token", ""])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == int(ExitCode.INVALID_USER_INPUT)


def test_cmd_serve_reads_bounded_token_from_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hawavoclean.server.app as app_mod

    observed: dict[str, object] = {}

    def fake_run_server(host: str, port: int, token: str, ui_dir: Path | None) -> int:
        observed.update(host=host, port=port, token=token, ui_dir=ui_dir)
        return 0

    monkeypatch.setattr(app_mod, "run_server", fake_run_server)
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("pipe-secret\n"))
    monkeypatch.setattr(sys, "argv", ["hawavoclean", "serve", "--token-stdin"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert observed == {
        "host": "127.0.0.1",
        "port": 0,
        "token": "pipe-secret",
        "ui_dir": None,
    }

    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("x" * 258))
    monkeypatch.setattr(sys, "argv", ["hawavoclean", "serve", "--token-stdin"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == int(ExitCode.INVALID_USER_INPUT)


def test_cmd_serve_without_ui_extra_is_a_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "hawavoclean.server.app":
            raise ImportError("No module named 'fastapi'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sys, "argv", ["hawavoclean", "serve", "--token", "t"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == int(ExitCode.PREFLIGHT_FAILURE)


def test_run_server_in_process_ready_line_health_and_shutdown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drive ``run_server`` on a thread with the process-level side effects
    (fd redirection, hard-exit timer) stubbed out."""
    import threading

    import hawavoclean.server.app as app_mod

    hard_exits: list[float] = []
    monkeypatch.setattr(app_mod, "_schedule_hard_exit", hard_exits.append)
    monkeypatch.setattr(app_mod, "_redirect_stdout_to_stderr", lambda: None)
    result: dict[str, int] = {}

    def target() -> None:
        result["rc"] = app_mod.run_server("localhost", 0, "tok", None)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    ready: dict[str, Any] | None = None
    deadline = time.monotonic() + 20
    captured_out = ""
    while time.monotonic() < deadline and ready is None:
        captured_out += capsys.readouterr().out
        for line in captured_out.splitlines():
            if line.strip():
                ready = json.loads(line)
                break
        time.sleep(0.05)
    assert ready is not None, "no ready line"
    assert ready["event"] == "ready" and ready["version"] == __version__
    base = f"http://127.0.0.1:{ready['port']}"
    status, body = _get(f"{base}/api/health", token="tok")
    assert status == 200 and body["ok"] is True
    req = urllib.request.Request(
        f"{base}/api/shutdown", method="POST", headers={"X-Hawa-Token": "tok"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert json.loads(resp.read().decode()) == {"ok": True}
    thread.join(timeout=10)
    assert not thread.is_alive(), "run_server did not return after /api/shutdown"
    assert result["rc"] == 0
    assert hard_exits == [0.7]
    assert capsys.readouterr().out == ""  # nothing after the ready line


def test_run_server_rejects_empty_token_and_bad_ui_dir(tmp_path: Path) -> None:
    from hawavoclean.server.app import run_server

    with pytest.raises(InvalidUserInputError):
        run_server("127.0.0.1", 0, "", None)
    with pytest.raises(InvalidUserInputError):
        run_server("127.0.0.1", 0, "t", tmp_path)


def test_run_server_disables_access_logs_that_would_expose_query_tokens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import uvicorn

    import hawavoclean.server.app as app_mod

    captured: dict[str, Any] = {}

    class FakeConfig:
        def __init__(self, _app: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    class FakeServer:
        def __init__(self, _config: Any) -> None:
            self.should_exit = False

        def run(self, *, sockets: list[socket.socket]) -> None:
            for bound in sockets:
                bound.close()

    monkeypatch.setattr(uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(app_mod, "_configure_uvicorn_logging", lambda: None)
    monkeypatch.setattr(app_mod, "_redirect_stdout_to_stderr", lambda: None)
    assert app_mod.run_server("127.0.0.1", 0, "secret-token", None) == 0
    assert captured["access_log"] is False
    assert json.loads(capsys.readouterr().out)["event"] == "ready"


def test_schedule_hard_exit_and_redirect_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import hawavoclean.server.app as app_mod

    started: list[Any] = []

    class FakeTimer:
        def __init__(self, delay: float, func: Any, args: tuple[Any, ...] = ()) -> None:
            started.append((delay, func, args))
            self.daemon = False

        def start(self) -> None:
            started.append("started")

    monkeypatch.setattr(threading, "Timer", FakeTimer)
    app_mod._schedule_hard_exit(0.7)
    assert started[0][0] == 0.7 and started[0][1] is os._exit and started[0][2] == (0,)
    assert started[-1] == "started"

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "dup2", lambda a, b: calls.append((a, b)))
    app_mod._redirect_stdout_to_stderr()  # under capture fileno() may raise: suppressed
    assert calls == [] or calls[0][0] == sys.stderr.fileno()
