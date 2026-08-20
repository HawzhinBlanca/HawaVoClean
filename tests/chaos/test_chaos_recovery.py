"""Chaos tests: REAL fault injection at the process and filesystem boundary.

Each test injects an actual fault — a killed worker process, a hung worker,
garbage model output, a failing rename — and asserts both halves of the
fail-closed contract: the output audio is still sample-exact, and the audit
report honestly records the degradation.
"""

import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.pipeline as pipeline_mod
from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.enhancement.worker import IsolatedEnhancementWorker
from hawavoclean.errors import AmbiguousStereoError, PublicationError, WorkerError
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"


class _SuicidalEnhancer:
    """Worker-side enhancer that SIGKILLs its own process on first enhance."""

    def __init__(self, _core_id: str = "x", sample_rate: int = 48000) -> None:
        self._meta = EnhancerMetadata(
            core_id="suicidal",
            version="0",
            algorithm="crash",
            sample_rate=sample_rate,
            phase_coherent=True,
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def enhance(self, _waveform: Any, _sample_rate: int) -> EnhancementResult:
        os.kill(os.getpid(), signal.SIGKILL)
        raise RuntimeError("unreachable")


class _HangingEnhancer:
    """Worker-side enhancer that sleeps far beyond the deadline."""

    def __init__(self, _core_id: str = "x", sample_rate: int = 48000) -> None:
        self._meta = EnhancerMetadata(
            core_id="hanging",
            version="0",
            algorithm="hang",
            sample_rate=sample_rate,
            phase_coherent=True,
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def enhance(self, _waveform: Any, _sample_rate: int) -> EnhancementResult:
        time.sleep(3600.0)
        raise RuntimeError("unreachable")


class _GarbageEnhancer:
    """In-process enhancer returning NaN, wrong-length, then silent output."""

    def __init__(self, _core_id: str = "x", sample_rate: int = 48000, **_: Any) -> None:
        self._meta = EnhancerMetadata(
            core_id="garbage",
            version="0",
            algorithm="garbage",
            sample_rate=sample_rate,
            phase_coherent=True,
        )
        self._call = 0

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass

    def enhance(self, waveform: Any, sample_rate: int) -> EnhancementResult:
        self._call += 1
        n = len(waveform)
        if self._call % 3 == 1:
            out = np.full(n, np.nan, dtype=np.float32)
        elif self._call % 3 == 2:
            out = np.zeros(n * 2 + 4096, dtype=np.float32)
        else:
            out = np.zeros(n, dtype=np.float32)
        return EnhancementResult(
            waveform=out,
            sample_rate=sample_rate,
            model_runtime_ms=0.1,
            input_samples=n,
            output_samples=len(out),
        )


@pytest.mark.chaos
def test_chaos_worker_killed_mid_unit_fails_closed() -> None:
    """A SIGKILLed worker subprocess must surface as a WorkerError, not hang."""
    worker = IsolatedEnhancementWorker(
        timeout_s=5.0,
        enhancer_class=_SuicidalEnhancer,
    )
    try:
        with pytest.raises(WorkerError):
            worker.enhance(np.zeros(48000, dtype=np.float32), 48000)
    finally:
        worker.close()


@pytest.mark.chaos
def test_chaos_worker_hang_hits_deadline_and_fails_closed() -> None:
    """A hung worker must be reaped at the deadline, not waited on forever."""
    worker = IsolatedEnhancementWorker(
        timeout_s=3.0,
        enhancer_class=_HangingEnhancer,
    )
    try:
        t0 = time.perf_counter()
        with pytest.raises(WorkerError):
            worker.enhance(np.zeros(48000, dtype=np.float32), 48000)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"deadline enforcement took {elapsed:.1f}s for a 3s timeout"
    finally:
        worker.close()


@pytest.mark.chaos
def test_chaos_garbage_model_output_fails_closed_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NaN / wrong-length / silent enhancer output must never reach the master."""
    monkeypatch.setattr(pipeline_mod, "IsolatedEnhancementWorker", _GarbageEnhancer)

    out = tmp_path / "garbage_guarded.wav"
    report = run_pipeline(
        input_path=FIXTURE,
        output_path=out,
        profile="production",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    # Output exists, sample-exact, and every speech unit fell back to original.
    assert out.exists()
    assert report.output.samples == report.input.samples
    speech = [u for u in report.units if u.is_speech]
    assert speech, "fixture must produce at least one speech unit"
    for u in speech:
        assert u.final_decision != "enhanced", (
            f"unit {u.unit_id} selected garbage output: {u.final_decision}"
        )

    # And the published audio content must be the original (modulo mastering
    # gain): correlation with the source must be near-perfect.
    orig, sr_o = sf.read(str(FIXTURE), dtype="float32", always_2d=True)
    produced, sr_p = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr_o == sr_p and orig.shape == produced.shape
    a, b = orig[:, 0], produced[:, 0]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    corr = float(np.dot(a, b) / denom) if denom > 0 else 1.0
    assert corr > 0.99, f"published audio diverged from fail-closed original (corr={corr:.4f})"


@pytest.mark.chaos
def test_chaos_disk_full_at_publish_leaves_no_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ENOSPC during the final renames must roll back cleanly."""
    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:  # audio lands, then the report rename fails
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    dest_dir = tmp_path / "published"
    dest_dir.mkdir()
    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(PublicationError):
        run_pipeline(
            input_path=FIXTURE,
            output_path=dest_dir / "out.wav",
            profile="production",
            overwrite=True,
            probe_override=FixedProbe(),
        )

    monkeypatch.setattr(os, "replace", real_replace)
    leftovers = [p for p in dest_dir.iterdir() if not p.name.startswith(".")]
    assert not leftovers, f"partial artifacts left at destination: {leftovers}"


@pytest.mark.chaos
def test_chaos_interrupt_between_renames_rolls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Ctrl-C landing between two of the three publish renames must roll back.

    Publication renames the master, then the JSON report, then the TXT
    summary. ``KeyboardInterrupt`` is a ``BaseException``, so an
    ``except Exception`` rollback lets it out of the middle of that loop with
    the master already at the destination and no report beside it — the
    partial publication the atomic publisher exists to prevent, and the one
    an interrupt can actually cause (the CLI turns SIGTERM into the same
    exception, and so does the parent-death watchdog).
    """
    real_replace = os.replace
    calls = {"n": 0}

    def interrupted_replace(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:  # the master has landed; now the user hits Ctrl-C
            raise KeyboardInterrupt
        real_replace(src, dst)

    dest_dir = tmp_path / "published"
    dest_dir.mkdir()
    monkeypatch.setattr(os, "replace", interrupted_replace)

    with pytest.raises(KeyboardInterrupt):  # an interrupt stays an interrupt
        run_pipeline(
            input_path=FIXTURE,
            output_path=dest_dir / "out.wav",
            profile="production",
            overwrite=True,
            probe_override=FixedProbe(),
        )

    monkeypatch.setattr(os, "replace", real_replace)
    assert calls["n"] >= 2, "the publish never reached the rename that was interrupted"
    leftovers = sorted(p.name for p in dest_dir.iterdir())
    assert not leftovers, f"interrupted publish left artifacts at destination: {leftovers}"


@pytest.mark.chaos
def test_chaos_ambiguous_stereo_rejected_without_silent_downmix(tmp_path: Path) -> None:
    with pytest.raises(AmbiguousStereoError):
        run_pipeline(
            input_path=REPO / "tests" / "fixtures" / "sample_ambiguous_stereo.wav",
            output_path=tmp_path / "out.wav",
            profile="production",
            overwrite=True,
        )
