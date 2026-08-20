"""A worker killed mid-unit must cost that unit its enhancement, and nothing
more.

This is the fail-closed invariant with a pool underneath it. The pool made a
new way to get this wrong — a dead worker could take the units queued behind
it, or take the job — so the contract is tested end to end, through the real
pipeline, with a real SIGKILL inside a real worker process:

* the job still finishes and publishes;
* the unit whose worker died is published as ORIGINAL audio and recorded as
  ``original_error``, never as silence and never as a claimed enhancement;
* the other units are unaffected, which is the part a pool could break; and
* the master is still sample-exact against the source.
"""

import dataclasses
import os
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.enhancement.worker import POOL_SIZE_ENV
from hawavoclean.pipeline import run_pipeline

SR = 48000
KILL_MARKER_ENV = "HAWAVOCLEAN_TEST_POOL_KILL_MARKER"


class _KillsOnceThenIdentity:
    """SIGKILLs its own worker process for exactly one unit of the run.

    The marker file is created with ``O_EXCL`` so that exactly one worker in
    the pool wins the race, whichever it is — the test does not need to know
    which unit dies, only that one does and that the rest survive it. The
    marker path travels by environment because the enhancer is constructed in
    a spawned child, and a fixed path would be shared with every other run on
    the machine.
    """

    def __init__(self, _core_id: str = "x", sample_rate: int = SR, **_: Any) -> None:
        self._meta = EnhancerMetadata("kills-once", "0", "chaos", sample_rate, True)

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def enhance(self, waveform: Any, sample_rate: int) -> EnhancementResult:
        marker = os.environ[KILL_MARKER_ENV]
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            os.close(fd)
            os.kill(os.getpid(), signal.SIGKILL)
        out = np.array(waveform, dtype=np.float32, copy=True)
        return EnhancementResult(out, sample_rate, 1.0, len(waveform), len(out))


def _multi_unit_fixture(path: Path, bursts: int = 6, speech_s: float = 8.0) -> Path:
    rng = np.random.default_rng(909)
    parts: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for k in range(bursts):
        n = int(SR * speech_s)
        t = np.arange(n) / SR
        f0 = 110.0 + 15.0 * k
        voiced = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.12 * np.sin(2 * np.pi * 2 * f0 * t)
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.1 * t)
        parts.append((voiced * env + 0.02 * rng.standard_normal(n)).astype(np.float32))
        parts.append(np.zeros(int(SR * 1.5), dtype=np.float32))
    sf.write(str(path), np.concatenate(parts), SR, subtype="PCM_16")
    return path


@pytest.mark.chaos
def test_a_worker_killed_mid_unit_reverts_that_unit_and_the_job_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hawavoclean.enhancement.factory import resolve_core

    monkeypatch.setenv(KILL_MARKER_ENV, str(tmp_path / "killed-once"))
    monkeypatch.setenv(POOL_SIZE_ENV, "3")

    src = _multi_unit_fixture(tmp_path / "multi.wav")
    out = tmp_path / "killed.wav"

    # Substituting the core is the only injection point: everything else —
    # the pool, the worker processes, the deadline, the guard — is the real
    # machinery under test.
    def _resolve(core_id: str) -> Any:
        return dataclasses.replace(resolve_core(core_id), enhancer_class=_KillsOnceThenIdentity)

    monkeypatch.setattr("hawavoclean.pipeline.resolve_core", _resolve)

    report = run_pipeline(input_path=src, output_path=out, profile="production", overwrite=True)

    assert out.exists(), "a killed worker took the whole job down"
    assert (tmp_path / "killed-once").exists(), "no worker was actually killed"

    speech = [u for u in report.units if u.is_speech]
    assert len(speech) >= 2, "fixture must produce more than one speech unit"

    errored = [u for u in speech if u.final_decision == "original_error"]
    assert len(errored) == 1, (
        "exactly one unit should have lost its worker, got "
        f"{[(u.unit_id, u.final_decision) for u in speech]}"
    )
    assert errored[0].candidate_sha256 is None
    assert "fail-closed" in (errored[0].decision_reason or "").lower()

    survivors = [u for u in speech if u is not errored[0]]
    assert survivors, "nothing survived to prove the blast radius was one unit"
    assert all(u.final_decision != "original_error" for u in survivors), (
        "the dead worker took units queued behind it: "
        f"{[(u.unit_id, u.final_decision) for u in survivors]}"
    )

    # Fail closed means ORIGINAL audio, never silence and never a gap.
    produced, sr_p = sf.read(str(out), dtype="float32", always_2d=True)
    original, sr_o = sf.read(str(src), dtype="float32", always_2d=True)
    assert (sr_p, produced.shape) == (sr_o, original.shape)
    start, end = errored[0].start_sample, errored[0].end_sample
    reverted = produced[start:end, 0]
    source = original[start:end, 0]
    assert float(np.max(np.abs(reverted))) > 1e-4, "the reverted unit was published as silence"
    # Mastering applies one static gain to the whole file, so the reverted
    # region must still be the source signal up to that scalar.
    scale = float(np.dot(reverted, source) / max(np.dot(source, source), 1e-20))
    residual = reverted - scale * source
    rel = float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(reverted**2)), 1e-20))
    assert rel < 0.05, f"the reverted unit is not the original audio (relative residual {rel:.3f})"
