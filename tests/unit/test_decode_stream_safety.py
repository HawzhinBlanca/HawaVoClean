"""Adversarial safety boundaries for the streamed FFmpeg decoder.

These tests use a controlled binary pipe instead of sleeping child processes,
so a regression is deterministic on every CI platform.  ProcessSupervisor's
own suite separately proves that ``terminate_tree`` reaches the POSIX process
group and Windows Job Object.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import hawavoclean.audio.decode as decode_module
from hawavoclean.audio.decode import iter_decode_audio
from hawavoclean.audio.types import AudioProbeResult
from hawavoclean.errors import (
    InvalidUserInputError,
    MediaPreflightError,
    MediaPreflightReason,
)

pytestmark = pytest.mark.unit


class _ControlledStdout:
    """Return configured blocks, then emulate an OS pipe stuck in ``read``."""

    def __init__(self, blocks: Sequence[bytes] = ()) -> None:
        self._blocks = list(blocks)
        self.release = threading.Event()
        self.read_started = threading.Event()
        self.read_returned = threading.Event()
        self.closed = False

    def read1(self, _size: int) -> bytes:
        self.read_started.set()
        if self._blocks:
            return self._blocks.pop(0)
        self.release.wait(timeout=5.0)
        self.read_returned.set()
        return b""

    def read(self, size: int) -> bytes:
        return self.read1(size)

    def close(self) -> None:
        self.closed = True
        self.release.set()


class _FakeProcess:
    def __init__(self, stdout: _ControlledStdout) -> None:
        self.pid = 4242
        self.stdout = stdout
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("controlled-ffmpeg", timeout or 0.0)
        return self.returncode


class _FakeSupervisor:
    def __init__(self, stdout: _ControlledStdout) -> None:
        self.process = _FakeProcess(stdout)
        self.terminate_calls = 0
        self.close_calls = 0
        self.events: list[str] = []

    def terminate_tree(self, _grace_s: float) -> None:
        self.events.append("terminate")
        self.terminate_calls += 1
        self.process.returncode = -9
        self.process.stdout.release.set()

    def close(self) -> None:
        self.events.append("close")
        self.close_calls += 1
        self.process.stdout.release.set()


def _probe(path: Path, *, samples: int = 1, sample_rate: int = 48_000) -> AudioProbeResult:
    return AudioProbeResult(
        path=path,
        format_name="wav",
        codec_name="pcm_f32le",
        sample_rate=sample_rate,
        channels=1,
        duration_s=samples / sample_rate,
        samples=samples,
        bit_depth=32,
        sha256="0" * 64,
    )


def _install_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    supervisor: _FakeSupervisor,
    captured: dict[str, Any],
) -> None:
    monkeypatch.setattr("hawavoclean.audio.decode.shutil.which", lambda _name: "/signed/ffmpeg")

    class _Factory:
        @staticmethod
        def spawn(command: Sequence[str], **kwargs: Any) -> _FakeSupervisor:
            captured["command"] = list(command)
            captured["kwargs"] = kwargs
            return supervisor

    monkeypatch.setattr("hawavoclean.audio.decode.ProcessSupervisor", _Factory)


def test_stalled_ffmpeg_pipe_hits_no_progress_deadline_and_terminates_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stdout = _ControlledStdout()
    supervisor = _FakeSupervisor(stdout)
    captured: dict[str, Any] = {}
    _install_supervisor(monkeypatch, supervisor, captured)

    started = time.monotonic()
    with pytest.raises(InvalidUserInputError, match="made no progress"):
        list(
            iter_decode_audio(
                _probe(tmp_path / "stalled.wav"),
                chunk_samples=4,
                timeout_s=2.0,
                no_progress_timeout_s=0.05,
            )
        )
    elapsed = time.monotonic() - started

    assert stdout.read_started.is_set(), "the adversarial reader never entered blocking read"
    assert stdout.read_returned.wait(timeout=0.5), "tree termination did not unblock the reader"
    assert elapsed < 0.75, "the no-progress deadline behaved like the old 30-minute timeout"
    assert supervisor.events == ["terminate", "close"]
    assert supervisor.terminate_calls == supervisor.close_calls == 1
    assert captured["kwargs"]["stdout"] is subprocess.PIPE


def test_decoded_sample_ceiling_ignores_false_probe_length_and_terminates_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The metadata claims one frame. Lower the default six-hour constant to
    # four seconds at 1 Hz, then have the decoder emit five frames. This tests
    # the production default path without allocating six hours of samples.
    raw = np.arange(5, dtype=np.float32).tobytes()
    stdout = _ControlledStdout([raw])
    supervisor = _FakeSupervisor(stdout)
    captured: dict[str, Any] = {}
    _install_supervisor(monkeypatch, supervisor, captured)
    monkeypatch.setattr(decode_module, "MAX_STREAM_DECODE_DURATION_S", 4)

    with pytest.raises(MediaPreflightError) as raised:
        list(
            iter_decode_audio(
                _probe(tmp_path / "metadata-lie.wav", samples=1, sample_rate=1),
                chunk_samples=2,
                timeout_s=2.0,
                no_progress_timeout_s=0.5,
            )
        )

    assert raised.value.reason is MediaPreflightReason.RESOURCE_BOMB
    assert "ceiling of 4 samples" in str(raised.value)
    assert supervisor.events == ["terminate", "close"]
    command = captured["command"]
    assert command[command.index("-t") + 1] == "5.000000000"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"no_progress_timeout_s": float("inf")}, "no_progress_timeout_s"),
        ({"max_decoded_samples": 0}, "max_decoded_samples"),
        ({"max_decoded_samples": True}, "max_decoded_samples"),
    ],
)
def test_stream_safety_limits_cannot_be_disabled(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        next(iter_decode_audio(_probe(tmp_path / "bad-limit.wav"), **kwargs))
