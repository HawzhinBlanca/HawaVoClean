"""Multi-pass orchestration: the separation metric, the auto-mode verdict,
pass records in the report, progress rescaling, and the CLI surface.

Red-first for the multipass feature: every test here was written before
``hawavoclean/multipass.py`` existed. Real-file evidence (teat1vo, Flute 09)
lives in ``tests/integration/test_multipass_real.py``; these tests cover the
same logic on synthetic signals and stubbed pipelines so CI needs no
gitignored media.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.multipass as multipass
from hawavoclean.errors import HawaVoCleanError, InvalidUserInputError
from hawavoclean.hashing import hash_file
from hawavoclean.multipass import (
    MIN_SEPARATION_GAIN_DB,
    auto_pass_verdict,
    rescale_event,
    run_multipass,
    speech_floor_separation_db,
)
from hawavoclean.paths import work_root
from hawavoclean.progress import ProgressEvent
from hawavoclean.publication import public_output_path
from hawavoclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    HawaVoCleanReport,
    MediaStats,
    PassRecord,
    UnitDecisionRecord,
    UnitSummary,
)
from hawavoclean.report.summary import generate_human_summary
from hawavoclean.report.writer import load_json_report, write_json_report
from tests.support.wavbytes import masked_wav_bytes

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Separation metric: pure function, known synthetic signals
# ---------------------------------------------------------------------------


def _bursty_signal(
    loud_amp: float, floor_amp: float, sr: int = 16000, seconds: float = 4.0
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """20% loud sine bursts over a quiet noise floor; both levels known."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    x = floor_amp * rng.standard_normal(n)
    # One long burst covering 20% of the signal, so p90 sits inside the burst
    # frames and p10 inside the floor frames.
    burst = slice(0, int(0.2 * n))
    x[burst] = loud_amp * np.sin(2 * np.pi * 440 * t[: int(0.2 * n)])
    return x.astype(np.float32)


@pytest.mark.unit
def test_separation_known_burst_floor_ratio() -> None:
    # Sine RMS = A/sqrt(2); white noise RMS = sigma. A=0.5 vs sigma=0.005:
    # 20*log10((0.5/sqrt(2))/0.005) = 36.99 dB.
    x = _bursty_signal(loud_amp=0.5, floor_amp=0.005)
    sep = speech_floor_separation_db(x)
    assert sep == pytest.approx(36.99, abs=1.0)


@pytest.mark.unit
def test_separation_constant_signal_is_near_zero() -> None:
    t = np.arange(32000) / 16000
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert speech_floor_separation_db(x) == pytest.approx(0.0, abs=0.5)


@pytest.mark.unit
def test_separation_silence_is_zero() -> None:
    assert speech_floor_separation_db(np.zeros(48000, dtype=np.float32)) == 0.0


@pytest.mark.unit
def test_separation_short_signal_does_not_crash() -> None:
    # Shorter than one 2048-sample frame: one truncated frame, zero spread.
    x = (0.1 * np.ones(300)).astype(np.float32)
    assert speech_floor_separation_db(x) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_separation_empty_signal_is_zero() -> None:
    assert speech_floor_separation_db(np.zeros(0, dtype=np.float32)) == 0.0


@pytest.mark.unit
def test_separation_matches_reference_frame_computation() -> None:
    """The cumsum implementation must equal the naive framed RMS p90-p10."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50_000).astype(np.float32) * np.linspace(0.001, 0.5, 50_000).astype(
        np.float32
    )
    frame, hop = 2048, 1024
    xs = x.astype(np.float64)
    n = 1 + (xs.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(xs[idx] ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    expected = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))
    assert speech_floor_separation_db(x) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# PassRecord schema and report round-trip
# ---------------------------------------------------------------------------


def _media(path: str = "x.wav", sha: str = "a" * 64) -> MediaStats:
    return MediaStats(
        path=path,
        sha256=sha,
        sample_rate=48000,
        channels=1,
        samples=48000,
        duration_s=1.0,
        integrated_lufs=-19.0,
        true_peak_dbtp=-1.5,
    )


def _unit(unit_id: int, strength: float, decision: str) -> UnitDecisionRecord:
    return UnitDecisionRecord(
        unit_id=unit_id,
        channel=0,
        start_sample=0,
        end_sample=1000,
        start_time_s=0.0,
        end_time_s=1.0,
        is_speech=True,
        input_sha256="c" * 64,
        output_sha256="d" * 64,
        guard_a_verdict="PASS" if decision == "enhanced" else "REVERT",
        chosen_strength=strength,
        final_decision=decision,
    )


def _report(
    passes: list[PassRecord] | None = None,
    units: list[UnitDecisionRecord] | None = None,
    out_sha: str = "b" * 64,
) -> HawaVoCleanReport:
    return HawaVoCleanReport(
        job_id="job",
        config_hash="c" * 64,
        input=_media("in.wav", "a" * 64),
        output=_media("out.wav", out_sha),
        core=CoreMetadata(id="wiener-dd-48k-v1", algorithm="wiener-dd", params_hash="e" * 64),
        guard=GuardMetadata(id="g", probe_hash="f" * 64, calibration_id="cal"),
        environment=EnvironmentMetadata(
            platform="p",
            os_version="v",
            python_version="3",
            numpy_version="2",
            scipy_version="1",
            soundfile_version="0",
        ),
        summary=UnitSummary(units_total=len(units or []), enhanced=1),
        units=units or [],
        passes=passes or [],
    )


def _pass_record(index: int, sep: float, enhanced: int = 1, **kw: Any) -> PassRecord:
    defaults: dict[str, Any] = {
        "pass_index": index,
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "units_total": 2,
        "enhanced": enhanced,
        "reverted": 0,
        "chosen_strengths": [0.5],
        "separation_db": sep,
        "integrated_lufs": -19.0,
    }
    defaults.update(kw)
    return PassRecord(**defaults)


@pytest.mark.unit
def test_report_passes_default_is_empty_and_schema_version_1() -> None:
    rep = _report(passes=None)
    assert rep.passes == []
    assert rep.schema_version == 1
    # A pre-multipass report (no "passes" key at all) must still validate.
    raw = json.loads(rep.model_dump_json())
    del raw["passes"]
    old = HawaVoCleanReport.model_validate(raw)
    assert old.passes == []


@pytest.mark.unit
def test_report_with_passes_round_trips(tmp_path: Path) -> None:
    recs = [
        _pass_record(1, 19.7),
        _pass_record(2, 23.6, chosen_strengths=[1.0]),
        _pass_record(
            3,
            23.1,
            discarded=True,
            discard_reason="separation gain below floor",
        ),
    ]
    rep = _report(passes=recs)
    p = tmp_path / "r.json"
    write_json_report(rep, p)
    loaded = load_json_report(p)
    assert [pr.pass_index for pr in loaded.passes] == [1, 2, 3]
    assert loaded.passes[2].discarded is True
    assert loaded.passes[2].discard_reason == "separation gain below floor"
    assert loaded.passes[0].discarded is False and loaded.passes[0].discard_reason is None
    assert loaded.passes[1].separation_db == pytest.approx(23.6)


@pytest.mark.unit
def test_txt_summary_gains_passes_section_only_for_multipass() -> None:
    single = generate_human_summary(_report(passes=[]))
    assert "PASS" not in single.replace("PASSED", "").replace("PASS)", "")
    multi = generate_human_summary(
        _report(
            passes=[
                _pass_record(1, 19.7),
                _pass_record(2, 23.6, chosen_strengths=[1.0]),
                _pass_record(3, 23.1, discarded=True, discard_reason="regressed"),
            ]
        )
    )
    assert "MULTI-PASS" in multi
    assert "Pass 1" in multi and "Pass 2" in multi and "Pass 3" in multi
    assert "DISCARDED" in multi and "regressed" in multi


# ---------------------------------------------------------------------------
# Auto-mode verdict: pure decision function
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_auto_keeps_improving_pass() -> None:
    keep, reason = auto_pass_verdict(_pass_record(1, 19.7), _pass_record(2, 23.6))
    assert keep is True and reason is None


@pytest.mark.unit
def test_auto_discards_on_separation_below_floor() -> None:
    prev = _pass_record(2, 23.6)
    new = _pass_record(3, 23.1)
    keep, reason = auto_pass_verdict(prev, new)
    assert keep is False
    assert reason is not None and "separation" in reason
    # Exactly the floor is kept: >= 0.5 dB is an improvement.
    keep2, _ = auto_pass_verdict(prev, _pass_record(3, 23.6 + MIN_SEPARATION_GAIN_DB))
    assert keep2 is True


@pytest.mark.unit
def test_auto_discards_on_guard_regression() -> None:
    keep, reason = auto_pass_verdict(
        _pass_record(1, 19.7, enhanced=2), _pass_record(2, 30.0, enhanced=1)
    )
    assert keep is False
    assert reason is not None and "guard" in reason


@pytest.mark.unit
def test_auto_reports_both_failures_when_both_regress() -> None:
    keep, reason = auto_pass_verdict(
        _pass_record(1, 19.7, enhanced=2), _pass_record(2, 19.8, enhanced=1)
    )
    assert keep is False
    assert reason is not None and "guard" in reason and "separation" in reason


# ---------------------------------------------------------------------------
# Progress rescaling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rescale_event_explicit_total() -> None:
    ev = ProgressEvent("guard", 0.5, "Unit 1/2", unit_index=1, unit_total=2)
    out = rescale_event(ev, pass_index=2, pass_total=3)
    assert out.progress == pytest.approx(1 / 3 + 0.5 / 3)
    assert out.pass_index == 2 and out.pass_total == 3
    assert out.stage == "guard" and out.message == "Unit 1/2"
    assert out.unit_index == 1 and out.unit_total == 2
    d = out.to_dict()
    assert d["pass"] == {"index": 2, "total": 3}


@pytest.mark.unit
def test_rescale_event_auto_treats_current_pass_as_last() -> None:
    ev = ProgressEvent("enhance", 0.5, "m")
    out = rescale_event(ev, pass_index=2, pass_total=None)
    # Auto: pass 2 is treated as the last of 2 -> [0.5, 1.0].
    assert out.progress == pytest.approx(0.75)
    assert out.to_dict()["pass"] == {"index": 2, "total": None}


@pytest.mark.unit
def test_single_pass_event_dict_has_no_pass_key() -> None:
    ev = ProgressEvent("guard", 0.5, "m", unit_index=1, unit_total=2)
    d = ev.to_dict()
    assert "pass" not in d
    assert set(d) == {"event", "stage", "progress", "message", "unit"}


# ---------------------------------------------------------------------------
# Orchestration on a stubbed pipeline (auto logic, discard, cleanup)
# ---------------------------------------------------------------------------


class _StubPipeline:
    """Stands in for run_pipeline: writes a real tiny wav per pass and returns
    a fabricated report; separation per pass comes from a scripted sequence."""

    def __init__(self, separations: list[float], enhanced: list[int] | None = None) -> None:
        self.separations = separations
        self.enhanced = enhanced or [1] * len(separations)
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, **kwargs: Any) -> HawaVoCleanReport:
        k = len(self.calls) + 1
        in_path = Path(kwargs["input_path"])
        out_path = Path(kwargs["output_path"])
        self.calls.append((in_path, out_path))
        rng = np.random.default_rng(k)
        sf.write(str(out_path), (0.1 * rng.standard_normal(4800)).astype(np.float32), 48000)
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress(ProgressEvent("preflight", 0.02, "Preflight checks passed"))
            on_progress(ProgressEvent("publish", 0.98, "Publishing master"))
        units = [_unit(0, 1.0 if k > 1 else 0.5, "enhanced"), _unit(1, 0.0, "original_no_speech")]
        return HawaVoCleanReport(
            job_id=f"job{k}",
            config_hash="c" * 64,
            input=_media(str(in_path), hash_file(in_path) if in_path.exists() else "a" * 64),
            output=_media(str(out_path), hash_file(out_path)),
            core=CoreMetadata(id="wiener-dd-48k-v1", algorithm="wiener-dd", params_hash="e" * 64),
            guard=GuardMetadata(id="g", probe_hash="f" * 64, calibration_id="cal"),
            environment=EnvironmentMetadata(
                platform="p",
                os_version="v",
                python_version="3",
                numpy_version="2",
                scipy_version="1",
                soundfile_version="0",
            ),
            summary=UnitSummary(
                units_total=2, enhanced=self.enhanced[k - 1], no_speech=1, reverted=0
            ),
            units=units,
        )


def _install_stub(monkeypatch: pytest.MonkeyPatch, stub: _StubPipeline) -> None:
    monkeypatch.setattr(multipass, "run_pipeline", stub)

    def fake_measure(path: Path) -> float:
        # Find which pass wrote this file and script its separation.
        for i, (_, out) in enumerate(stub.calls):
            if out == path:
                return stub.separations[i]
        raise AssertionError(f"measured a file no pass wrote: {path}")

    monkeypatch.setattr(multipass, "measure_separation_db", fake_measure)


def _src_wav(tmp_path: Path) -> Path:
    p = tmp_path / "src.wav"
    rng = np.random.default_rng(0)
    sf.write(str(p), (0.1 * rng.standard_normal(4800)).astype(np.float32), 48000)
    return p


def _no_multipass_litter() -> bool:
    root = work_root()
    return not root.exists() or not list(root.glob("multipass-*"))


@pytest.mark.unit
def test_explicit_two_passes_chains_input_and_records_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubPipeline([19.7, 23.6])
    _install_stub(monkeypatch, stub)
    src = _src_wav(tmp_path)
    out = tmp_path / "out.wav"
    report = run_multipass(input_path=src, output_path=out, passes=2)

    assert len(stub.calls) == 2
    assert stub.calls[0][0] == src.resolve()
    assert stub.calls[1][0] == stub.calls[0][1], "pass 2 must consume pass 1's output"
    assert out.exists()
    assert hash_file(out) == report.output.sha256
    assert report.output.path == str(public_output_path(out))
    # Report input is the ORIGINAL source, passes[] chains the journey.
    assert report.input.sha256 == hash_file(src)
    assert [p.pass_index for p in report.passes] == [1, 2]
    assert report.passes[0].input_sha256 == hash_file(src)
    assert report.passes[1].input_sha256 == report.passes[0].output_sha256
    assert report.passes[1].output_sha256 == report.output.sha256
    assert report.passes[0].separation_db == pytest.approx(19.7)
    assert report.passes[1].separation_db == pytest.approx(23.6)
    assert not any(p.discarded for p in report.passes)
    assert report.passes[0].chosen_strengths == [0.5]
    assert report.passes[1].chosen_strengths == [1.0]
    # Sidecars at the destination, and the txt mentions the passes.
    report_json = out.parent / f"{out.stem}.hawavoclean.json"
    report_txt = out.parent / f"{out.stem}.hawavoclean.txt"
    assert load_json_report(report_json).passes == report.passes
    assert "MULTI-PASS" in report_txt.read_text(encoding="utf-8")
    assert _no_multipass_litter()


@pytest.mark.unit
def test_auto_discards_regressing_pass_and_ships_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pass 3 regresses (23.1 < 23.6 + 0.5): auto must ship pass 2's audio and
    # record pass 3 as discarded.
    stub = _StubPipeline([19.7, 23.6, 23.1])
    _install_stub(monkeypatch, stub)
    out = tmp_path / "out.wav"
    report = run_multipass(input_path=_src_wav(tmp_path), output_path=out, passes="auto")

    assert len(stub.calls) == 3
    assert len(report.passes) == 3
    assert [p.discarded for p in report.passes] == [False, False, True]
    assert report.passes[2].discard_reason is not None
    assert "separation" in report.passes[2].discard_reason
    # The shipped audio IS pass 2's output.
    assert report.output.sha256 == report.passes[1].output_sha256
    assert hash_file(out) == report.passes[1].output_sha256
    assert _no_multipass_litter()


@pytest.mark.unit
def test_auto_stops_at_max_passes_when_everything_improves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubPipeline([10.0, 15.0, 20.0, 25.0, 30.0])
    _install_stub(monkeypatch, stub)
    out = tmp_path / "out.wav"
    report = run_multipass(input_path=_src_wav(tmp_path), output_path=out, passes="auto")
    assert len(stub.calls) == 4, "auto is capped at 4 passes"
    assert len(report.passes) == 4
    assert not any(p.discarded for p in report.passes)
    assert report.output.sha256 == report.passes[3].output_sha256


@pytest.mark.unit
def test_auto_discarding_second_pass_ships_pass_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Flute 09 shape: an already-good source where pass 2 changes nothing.
    stub = _StubPipeline([25.0, 25.1])
    _install_stub(monkeypatch, stub)
    out = tmp_path / "out.wav"
    report = run_multipass(input_path=_src_wav(tmp_path), output_path=out, passes="auto")
    assert len(report.passes) == 2
    assert report.passes[1].discarded is True
    assert report.output.sha256 == report.passes[0].output_sha256
    assert hash_file(out) == report.passes[0].output_sha256


@pytest.mark.unit
def test_pass_failure_fails_whole_run_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def failing_pipeline(**kwargs: Any) -> HawaVoCleanReport:
        calls.append(1)
        if len(calls) == 2:
            raise HawaVoCleanError("pass 2 exploded")
        return _StubPipeline([10.0]).__call__(**kwargs)

    monkeypatch.setattr(multipass, "run_pipeline", failing_pipeline)
    monkeypatch.setattr(multipass, "measure_separation_db", lambda _p: 10.0)
    out = tmp_path / "out.wav"
    with pytest.raises(HawaVoCleanError, match="pass 2 exploded"):
        run_multipass(input_path=_src_wav(tmp_path), output_path=out, passes=2)
    assert not out.exists(), "a failed multipass run must ship nothing"
    assert not (out.parent / f"{out.stem}.hawavoclean.json").exists()
    assert _no_multipass_litter()


@pytest.mark.unit
def test_multipass_refuses_existing_destination_before_any_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubPipeline([10.0, 20.0])
    _install_stub(monkeypatch, stub)
    out = tmp_path / "out.wav"
    out.write_bytes(b"precious")
    with pytest.raises(HawaVoCleanError):
        run_multipass(input_path=_src_wav(tmp_path), output_path=out, passes=2)
    assert stub.calls == [], "the destination must be refused before pass 1 runs"
    assert out.read_bytes() == b"precious"


@pytest.mark.unit
def test_multipass_rejects_bad_pass_counts(tmp_path: Path) -> None:
    src = _src_wav(tmp_path)
    for bad in (0, 5, -1):
        with pytest.raises(InvalidUserInputError):
            run_multipass(input_path=src, output_path=tmp_path / "o.wav", passes=bad)
    with pytest.raises(InvalidUserInputError):
        run_multipass(input_path=src, output_path=tmp_path / "o.wav", passes="never")


@pytest.mark.unit
def test_multipass_progress_events_carry_pass_and_rescale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubPipeline([19.7, 23.6])
    _install_stub(monkeypatch, stub)
    events: list[ProgressEvent] = []
    run_multipass(
        input_path=_src_wav(tmp_path),
        output_path=tmp_path / "out.wav",
        passes=2,
        on_progress=events.append,
    )
    assert events, "progress must flow through multipass"
    assert all(e.pass_index in (1, 2) for e in events)
    assert all(e.pass_total == 2 for e in events)
    pass1 = [e for e in events if e.pass_index == 1]
    pass2 = [e for e in events if e.pass_index == 2]
    assert pass1 and pass2
    assert all(0.0 <= e.progress <= 0.5 for e in pass1)
    assert all(0.5 <= e.progress <= 1.0 for e in pass2)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hawavoclean.cli", *argv],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO,
    )


def _tiny_speechlike_wav(path: Path) -> Path:
    sr = 16000
    t = np.arange(int(1.5 * sr)) / sr
    sig = 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sig = sig + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    sf.write(str(path), sig.astype(np.float32), sr)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["0", "5", "-1", "banana", "1.5"])
def test_cli_rejects_bad_passes_values(bad: str, tmp_path: Path) -> None:
    proc = _run_cli(
        "process", str(tmp_path / "in.wav"), "-o", str(tmp_path / "o.wav"), "--passes", bad
    )
    assert proc.returncode == 2
    assert "--passes" in proc.stderr


@pytest.mark.unit
def test_cli_batch_refuses_passes_flag(tmp_path: Path) -> None:
    src = _tiny_speechlike_wav(tmp_path / "a.wav")
    proc = _run_cli("batch", str(src), "-o", str(tmp_path / "outdir"), "--passes", "2")
    assert proc.returncode == 2
    assert "--passes" in proc.stderr


@pytest.mark.unit
def test_cli_two_passes_end_to_end_report_and_progress(tmp_path: Path) -> None:
    src = _tiny_speechlike_wav(tmp_path / "tiny.wav")
    out = tmp_path / "out.wav"
    proc = _run_cli(
        "process",
        str(src),
        "-o",
        str(out),
        "--profile",
        "development",
        "--passes",
        "2",
        "--progress-json",
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    progress = [e for e in events if e["event"] == "progress"]
    assert progress and events[-1]["event"] == "done"
    assert all(e["pass"]["total"] == 2 for e in progress)
    assert {e["pass"]["index"] for e in progress} == {1, 2}
    report = load_json_report(out.parent / f"{out.stem}.hawavoclean.json")
    assert [p.pass_index for p in report.passes] == [1, 2]
    assert report.output.sha256 == hash_file(out)
    assert report.passes[1].output_sha256 == report.output.sha256
    # The published pair still verifies.
    v = _run_cli("verify", str(out), "-r", str(out.parent / f"{out.stem}.hawavoclean.json"))
    assert v.returncode == 0, v.stderr


@pytest.mark.unit
def test_cli_single_pass_stream_and_audio_unchanged(tmp_path: Path) -> None:
    """--passes 1 must be byte-identical to a plain run: same audio bytes,
    and the progress stream must carry no "pass" field."""
    src = _tiny_speechlike_wav(tmp_path / "tiny.wav")
    plain_out = tmp_path / "plain.wav"
    p1 = _run_cli(
        "process", str(src), "-o", str(plain_out), "--profile", "development", "--progress-json"
    )
    assert p1.returncode == 0, p1.stderr[-2000:]
    passes_out = tmp_path / "p1.wav"
    p2 = _run_cli(
        "process",
        str(src),
        "-o",
        str(passes_out),
        "--profile",
        "development",
        "--passes",
        "1",
        "--progress-json",
    )
    assert p2.returncode == 0, p2.stderr[-2000:]
    # Byte-identical modulo libsndfile's PEAK write-time second — see
    # tests/support/wavbytes.py: the samples and every pipeline-produced
    # header field must match exactly.
    assert masked_wav_bytes(plain_out.read_bytes()) == masked_wav_bytes(passes_out.read_bytes())
    for proc in (p1, p2):
        for ln in proc.stdout.splitlines():
            if ln.strip():
                assert "pass" not in json.loads(ln), "single-pass stream must not change"
    report = load_json_report(passes_out.parent / f"{passes_out.stem}.hawavoclean.json")
    assert report.passes == []


@pytest.mark.unit
def test_measure_separation_reads_written_file_mono_mix(tmp_path: Path) -> None:
    """File-level measurement equals the pure metric on the file's mono mix."""
    p = tmp_path / "m.wav"
    x = _bursty_signal(loud_amp=0.5, floor_amp=0.005)
    stereo = np.stack([x, x], axis=1)
    sf.write(str(p), stereo, 16000)
    direct = speech_floor_separation_db(x.astype(np.float64))
    assert multipass.measure_separation_db(p) == pytest.approx(direct, abs=0.2)


@pytest.mark.unit
def test_run_multipass_passes_1_delegates_to_single_pipeline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubPipeline([19.7])
    monkeypatch.setattr(multipass, "run_pipeline", stub)
    out = tmp_path / "out.wav"
    report = run_multipass(input_path=_src_wav(tmp_path), output_path=out, passes=1)
    # One ordinary run, straight at the destination, no pass records.
    assert len(stub.calls) == 1
    assert stub.calls[0][1] == out
    assert report.passes == []
    assert _no_multipass_litter()


@pytest.mark.unit
def test_masked_wav_bytes_masks_only_the_peak_timestamp(tmp_path: Path) -> None:
    """The mask must neutralise the PEAK write-time second and nothing else."""
    p = tmp_path / "w.wav"
    # subtype FLOAT: libsndfile adds its PEAK chunk only to float WAVs, which
    # is what the pipeline publishes.
    sf.write(str(p), (0.25 * np.ones(256)).astype(np.float32), 16000, subtype="FLOAT")
    raw = p.read_bytes()
    peak = raw.find(b"PEAK")
    assert peak > 0, "libsndfile stopped writing PEAK chunks: revisit the mask"
    # A different write-time second must compare equal...
    stamped = bytearray(raw)
    stamped[peak + 12 : peak + 16] = b"\xde\xad\xbe\xef"
    assert masked_wav_bytes(bytes(stamped)) == masked_wav_bytes(raw)
    # ...while a single tampered audio byte must still be caught.
    tampered = bytearray(raw)
    tampered[-1] ^= 0x01
    assert masked_wav_bytes(bytes(tampered)) != masked_wav_bytes(raw)
    # Non-RIFF payloads pass through untouched.
    assert masked_wav_bytes(b"not a wav") == b"not a wav"
