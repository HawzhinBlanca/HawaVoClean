"""``hawavoclean process --progress-json``: stdout is pure JSON lines, logs
stay on stderr, the stream ends in ``done`` or ``error`` (exit code unchanged)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.cli as cli
from hawavoclean.errors import ExitCode, HawaVoCleanError
from hawavoclean.publication import public_output_path

REPO = Path(__file__).resolve().parents[2]


def _tiny_wav(path: Path) -> Path:
    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    sig = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sig = sig + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    sf.write(str(path), sig.astype(np.float32), sr)
    return path


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hawavoclean.cli", *argv],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO,
    )


@pytest.mark.unit
def test_progress_json_stdout_is_pure_json_lines(tmp_path: Path) -> None:
    src = _tiny_wav(tmp_path / "tiny.wav")
    out = tmp_path / "out.wav"
    proc = _run_cli(
        "process", str(src), "-o", str(out), "--profile", "development", "--progress-json"
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, "no JSON lines on stdout"
    events = [json.loads(ln) for ln in lines]  # every line must parse
    kinds = [e["event"] for e in events]
    assert set(kinds) <= {"progress", "done"}
    assert kinds[-1] == "done" and kinds.count("done") == 1
    stages = [e["stage"] for e in events if e["event"] == "progress"]
    assert stages[:3] == ["preflight", "decode", "segment"]
    assert stages[-1] == "publish"
    assert "enhance" in stages and "guard" in stages and "finish" in stages
    done = events[-1]
    assert done["progress"] == 1.0
    assert Path(done["output_path"]) == public_output_path(out)
    assert Path(done["report_path"]) == public_output_path(out.parent / "out.hawavoclean.json")
    assert Path(done["report_path"]).exists()
    # Unit events carry the unit object with the contract keys.
    unit_events = [e for e in events if "unit" in e]
    assert unit_events and all(set(e["unit"]) == {"index", "total"} for e in unit_events)
    # Logs went to stderr, not stdout.
    assert "hawavoclean.pipeline" in proc.stderr
    assert "[INFO]" not in proc.stdout


@pytest.mark.unit
def test_progress_json_error_event_and_exit_code(tmp_path: Path) -> None:
    proc = _run_cli(
        "process", str(tmp_path / "missing.wav"), "-o", str(tmp_path / "o.wav"), "--progress-json"
    )
    assert proc.returncode == int(ExitCode.INVALID_USER_INPUT)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    err = json.loads(lines[0])
    assert err["event"] == "error"
    assert err["code"] == "INVALID_USER_INPUT"
    assert "missing.wav" in err["message"]


@pytest.mark.unit
def test_without_flag_stdout_has_no_json(tmp_path: Path) -> None:
    src = _tiny_wav(tmp_path / "tiny.wav")
    proc = _run_cli("process", str(src), "-o", str(tmp_path / "o.wav"), "--profile", "development")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert '"event"' not in proc.stdout


class _FakeSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def emit(self, obj: dict[str, Any]) -> None:
        self.events.append(obj)

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
def test_cmd_process_emits_done_through_sink(monkeypatch: Any, tmp_path: Path) -> None:
    sink = _FakeSink()
    monkeypatch.setattr(cli, "_JsonLineSink", lambda: sink)
    src = _tiny_wav(tmp_path / "tiny.wav")
    args = argparse.Namespace(
        input=str(src),
        output=str(tmp_path / "o.wav"),
        config=None,
        profile="development",
        overwrite=True,
        progress_json=True,
    )
    assert cli.cmd_process(args) == int(ExitCode.SUCCESS)
    assert sink.closed
    assert sink.events[0]["event"] == "progress" and sink.events[0]["stage"] == "preflight"
    assert sink.events[-1]["event"] == "done"
    assert sink.events[-1]["output_path"] == str(public_output_path(tmp_path / "o.wav"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (HawaVoCleanError("custom", exit_code=ExitCode.INVALID_USER_INPUT), "INVALID_USER_INPUT"),
        (RuntimeError("surprise"), "PUBLICATION_FAILURE"),
    ],
)
def test_cmd_process_emits_error_through_sink(
    monkeypatch: Any, tmp_path: Path, exc: Exception, code: str
) -> None:
    sink = _FakeSink()
    monkeypatch.setattr(cli, "_JsonLineSink", lambda: sink)

    def boom(**_kw: Any) -> Any:
        raise exc

    monkeypatch.setattr(cli, "run_pipeline", boom)
    args = argparse.Namespace(
        input="x.wav",
        output=str(tmp_path / "o.wav"),
        config=None,
        profile="development",
        overwrite=False,
        progress_json=True,
    )
    rc = cli.cmd_process(args)
    assert rc == int(ExitCode[code])
    assert sink.events == [{"event": "error", "code": code, "message": sink.events[0]["message"]}]
    assert sink.closed


@pytest.mark.unit
def test_cmd_process_without_progress_json_attribute_still_works(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Callers that build Namespaces by hand (older tests, scripts) omit the flag."""
    called: list[bool] = []

    def fake_run(**kw: Any) -> None:
        called.append(kw["on_progress"] is None)

    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    args = argparse.Namespace(
        input="x.wav",
        output=str(tmp_path / "o.wav"),
        config=None,
        profile="development",
        overwrite=False,
    )
    assert cli.cmd_process(args) == 0
    assert called == [True]


@pytest.mark.unit
def test_json_line_sink_restores_stdout(capfd: pytest.CaptureFixture[str]) -> None:
    sink = cli._JsonLineSink()
    sink.emit({"event": "progress", "stage": "preflight", "progress": 0.02, "message": "x"})
    print("this goes to stderr while the sink is open")
    sink.close()
    print("back on stdout")
    out, err = capfd.readouterr()
    assert out.splitlines() == [
        '{"event":"progress","stage":"preflight","progress":0.02,"message":"x"}',
        "back on stdout",
    ]
    assert "this goes to stderr" in err


@pytest.mark.unit
def test_json_line_sink_survives_a_closed_reader(
    capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """If the parent stopped reading (EPIPE), processing must go on; the sink
    logs once per failed write and never raises."""
    sink = cli._JsonLineSink()
    try:
        real_out = sink._out

        class _Broken:
            def write(self, _s: str) -> int:
                raise OSError(32, "Broken pipe")

            def flush(self) -> None:
                pass

            def fileno(self) -> int:
                return real_out.fileno()

            def close(self) -> None:
                real_out.close()

        sink._out = _Broken()  # type: ignore[assignment]
        with caplog.at_level("WARNING", logger="hawavoclean.cli"):
            sink.emit({"event": "progress"})
        assert any("progress stream closed" in r.message for r in caplog.records)
    finally:
        sink.close()
    capfd.readouterr()
