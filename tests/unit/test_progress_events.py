"""ProgressEvent emission from run_pipeline: order, weights, unit bookkeeping,
and the guarantee that a broken callback never breaks a run."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.pipeline import run_pipeline
from hawavoclean.progress import (
    PROGRESS_DECODE,
    PROGRESS_FINISH_END,
    PROGRESS_FINISH_START,
    PROGRESS_PREFLIGHT,
    PROGRESS_PUBLISH,
    PROGRESS_SEGMENT,
    PROGRESS_UNITS_END,
    ProgressEvent,
    emit_progress,
    unit_progress,
)

REPO = Path(__file__).resolve().parents[2]
SPLIT_FIXTURE = REPO / "tests" / "fixtures" / "sample_split_speakers.wav"


def _tiny_wav(path: Path, seconds: float = 1.5, sr: int = 16000) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    sig = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sig = sig + 0.01 * rng.standard_normal(t.size)
    sf.write(str(path), sig.astype(np.float32), sr)
    return path


def _run(tmp_path: Path, src: Path, cb: object) -> list[ProgressEvent]:
    events: list[ProgressEvent] = []

    def on_progress(ev: ProgressEvent) -> None:
        events.append(ev)
        if callable(cb):
            cb(ev)

    run_pipeline(
        input_path=src,
        output_path=tmp_path / "out.wav",
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
        on_progress=on_progress,
    )
    return events


@pytest.mark.unit
def test_single_unit_run_emits_contract_sequence(tmp_path: Path) -> None:
    src = _tiny_wav(tmp_path / "tiny.wav")
    events = _run(tmp_path, src, None)

    stages = [e.stage for e in events]
    assert stages == [
        "preflight",
        "decode",
        "segment",
        "enhance",
        "guard",
        "finish",
        "finish",
        "publish",
    ]
    by_stage = {e.stage: e for e in events}
    assert by_stage["preflight"].progress == PROGRESS_PREFLIGHT == 0.02
    assert by_stage["decode"].progress == PROGRESS_DECODE == 0.05
    assert by_stage["segment"].progress == PROGRESS_SEGMENT == 0.08
    assert by_stage["guard"].progress == PROGRESS_UNITS_END == 0.80
    assert by_stage["publish"].progress == PROGRESS_PUBLISH == 0.98
    finish = [e for e in events if e.stage == "finish"]
    assert finish[0].progress == PROGRESS_FINISH_START == 0.80
    assert finish[-1].progress == PROGRESS_FINISH_END == 0.95

    progresses = [e.progress for e in events]
    assert progresses == sorted(progresses), "progress must never go backwards"
    assert all(0.0 <= p <= 1.0 for p in progresses)

    assert by_stage["decode"].message == "Decoded 1.5 s @ 16 kHz, 1 ch"
    assert by_stage["segment"].message == "1 unit"
    assert by_stage["enhance"].message == "Enhancing unit 1/1"
    assert by_stage["enhance"].unit_index == 1 and by_stage["enhance"].unit_total == 1
    assert by_stage["guard"].message.startswith("Unit 1/1: ")
    assert by_stage["guard"].unit_index == 1 and by_stage["guard"].unit_total == 1
    assert by_stage["preflight"].unit_index is None
    assert by_stage["publish"].message == "Publishing master"


@pytest.mark.unit
def test_multi_unit_run_spans_units_linearly(tmp_path: Path) -> None:
    events = _run(tmp_path, SPLIT_FIXTURE, None)
    guards = [e for e in events if e.stage == "guard"]
    enhances = [e for e in events if e.stage == "enhance"]
    total = guards[0].unit_total
    assert total is not None and total >= 2
    assert [g.unit_index for g in guards] == list(range(1, total + 1))
    # Every unit gets a guard event; non-speech units report NO_SPEECH and
    # never an enhance event; speech units get an enhance event first.
    for g in guards:
        assert g.progress == pytest.approx(unit_progress(g.unit_index or 0, total, done=True))
        assert g.message == f"Unit {g.unit_index}/{total}: " + g.message.split(": ", 1)[1]
    no_speech = [g for g in guards if g.message.endswith("NO_SPEECH")]
    assert no_speech, "fixture has non-speech units"
    enhanced_ids = {e.unit_index for e in enhances}
    assert enhanced_ids.isdisjoint({g.unit_index for g in no_speech})
    for e in enhances:
        assert e.progress == pytest.approx(unit_progress(e.unit_index or 0, total, done=False))
        assert e.message == f"Enhancing unit {e.unit_index}/{total}"
    # Enhance for unit i is emitted before guard for unit i
    order = [(e.stage, e.unit_index) for e in events if e.stage in ("enhance", "guard")]
    for e in enhances:
        assert order.index(("enhance", e.unit_index)) < order.index(("guard", e.unit_index))
    assert guards[-1].progress == pytest.approx(0.80)
    progresses = [e.progress for e in events]
    assert progresses == sorted(progresses)


@pytest.mark.unit
def test_callback_exceptions_never_break_the_run(tmp_path: Path) -> None:
    src = _tiny_wav(tmp_path / "tiny.wav")
    seen: list[str] = []

    def explode(ev: ProgressEvent) -> None:
        seen.append(ev.stage)
        raise RuntimeError("sink is broken")

    report = run_pipeline(
        input_path=src,
        output_path=tmp_path / "out.wav",
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
        on_progress=explode,
    )
    assert (tmp_path / "out.wav").exists()
    assert report.output.samples == report.input.samples
    assert seen[0] == "preflight" and seen[-1] == "publish"


@pytest.mark.unit
def test_no_callback_is_the_default(tmp_path: Path) -> None:
    src = _tiny_wav(tmp_path / "tiny.wav")
    report = run_pipeline(
        input_path=src,
        output_path=tmp_path / "out.wav",
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )
    assert report.summary.units_total >= 1


@pytest.mark.unit
def test_event_to_dict_matches_contract_shape() -> None:
    plain = ProgressEvent("decode", 0.05, "Decoded 61.2 s @ 48 kHz, 1 ch")
    assert plain.to_dict() == {
        "event": "progress",
        "stage": "decode",
        "progress": 0.05,
        "message": "Decoded 61.2 s @ 48 kHz, 1 ch",
    }
    unit = ProgressEvent("enhance", 0.2333333, "Enhancing unit 1/5", unit_index=1, unit_total=5)
    d = unit.to_dict()
    assert d["unit"] == {"index": 1, "total": 5}
    assert d["progress"] == 0.2333
    assert json.loads(json.dumps(d)) == d  # JSON serialisable


@pytest.mark.unit
def test_unit_progress_arithmetic() -> None:
    assert unit_progress(1, 1, done=False) == pytest.approx(0.08)
    assert unit_progress(1, 1, done=True) == pytest.approx(0.80)
    assert unit_progress(1, 4, done=True) == pytest.approx(0.08 + 0.72 / 4)
    assert unit_progress(4, 4, done=False) == pytest.approx(0.08 + 0.72 * 3 / 4)
    assert unit_progress(3, 0, done=True) == pytest.approx(0.80)  # degenerate: no units
    assert unit_progress(9, 4, done=True) == pytest.approx(0.80)  # clamped


@pytest.mark.unit
def test_emit_progress_swallows_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[ProgressEvent] = []
    ev = ProgressEvent("publish", 0.98, "Publishing master")
    emit_progress(None, ev)  # no callback: no-op
    emit_progress(calls.append, ev)
    assert calls == [ev]

    def boom(_ev: ProgressEvent) -> None:
        raise ValueError("nope")

    with caplog.at_level("WARNING", logger="hawavoclean.progress"):
        emit_progress(boom, ev)
    assert any("Progress callback failed" in r.message for r in caplog.records)
