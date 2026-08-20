"""Parallel enhancement must be invisible in the product.

The pool is a scheduling change and nothing else. These tests hold it to
that: the same input, run through one worker and through four, must publish
the same bytes and record the same decisions — and ten consecutive runs must
publish the same bytes as each other, because "same input, same config, same
answer" is the promise the whole audit report rests on.

The fixture is generated rather than committed because the shipped fixtures
produce a single speech unit, and a single unit cannot tell a pool from a
sequence. This one is long enough, and gapped enough, to segment into
several units on both profiles.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.enhancement.worker import POOL_SIZE_ENV
from hawavoclean.pipeline import run_pipeline
from hawavoclean.report.schema import HawaVoCleanReport
from tests.support.wavbytes import masked_wav_bytes

SR = 48000


def _multi_unit_fixture(
    path: Path, bursts: int = 6, speech_s: float = 8.0, gap_s: float = 1.5
) -> Path:
    """Speech-like bursts separated by real silence, so segmentation has
    boundaries to cut on. The shipped defaults measure 4 units (3 of them
    speech) on the production profile — enough for scheduling to matter."""
    rng = np.random.default_rng(4242)
    parts: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for k in range(bursts):
        n = int(SR * speech_s)
        t = np.arange(n) / SR
        f0 = 110.0 + 15.0 * k
        voiced = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.12 * np.sin(2 * np.pi * 2 * f0 * t)
        # Syllable-rate envelope plus a little noise: enough structure for the
        # VAD to call it speech, and enough noise for the core to have work.
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.1 * t)
        burst = (voiced * env + 0.02 * rng.standard_normal(n)).astype(np.float32)
        parts.append(burst)
        parts.append(np.zeros(int(SR * gap_s), dtype=np.float32))
    sf.write(str(path), np.concatenate(parts), SR, subtype="PCM_16")
    return path


def _run(src: Path, dest: Path, workers: str, monkeypatch: Any) -> HawaVoCleanReport:
    monkeypatch.setenv(POOL_SIZE_ENV, workers)
    return run_pipeline(input_path=src, output_path=dest, profile="production", overwrite=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(masked_wav_bytes(path.read_bytes())).hexdigest()


def _decisions(report: HawaVoCleanReport) -> list[tuple[Any, ...]]:
    """Everything about a unit except how long it took."""
    return [
        (
            u.unit_id,
            u.channel,
            u.start_sample,
            u.end_sample,
            u.is_speech,
            u.input_sha256,
            u.candidate_sha256,
            u.output_sha256,
            u.guard_a_verdict,
            u.guard_a_scores,
            u.guard_b_verdict,
            u.guard_b_scores,
            u.chosen_strength,
            u.finish_preset_applied,
            tuple(u.finish_actions),
            u.final_decision,
            u.decision_reason,
        )
        for u in report.units
    ]


@pytest.mark.integration
def test_four_workers_publish_exactly_what_one_worker_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _multi_unit_fixture(tmp_path / "multi.wav")

    seq = _run(src, tmp_path / "seq.wav", "1", monkeypatch)
    par = _run(src, tmp_path / "par.wav", "4", monkeypatch)

    speech = sum(1 for u in seq.units if u.is_speech)
    assert speech >= 2, (
        f"fixture produced {speech} speech units — a pool cannot be told from a "
        "sequence with fewer than two"
    )

    assert _sha(tmp_path / "par.wav") == _sha(tmp_path / "seq.wav"), (
        "parallel enhancement changed the published master"
    )
    assert _decisions(par) == _decisions(seq), "parallel enhancement changed a per-unit decision"
    assert par.output.sha256 == seq.output.sha256
    assert par.summary == seq.summary


@pytest.mark.integration
def test_ten_consecutive_runs_publish_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Determinism under the pool, run for run. Scheduling varies between
    runs by nature; if any of it reached the output, ten runs would not agree."""
    src = _multi_unit_fixture(tmp_path / "multi.wav", bursts=4, speech_s=6.0)
    monkeypatch.setenv(POOL_SIZE_ENV, "4")

    hashes: list[str] = []
    decisions: list[list[tuple[Any, ...]]] = []
    for i in range(10):
        dest = tmp_path / f"run{i}.wav"
        report = run_pipeline(
            input_path=src, output_path=dest, profile="production", overwrite=True
        )
        hashes.append(_sha(dest))
        decisions.append(_decisions(report))

    assert len(set(hashes)) == 1, f"10 runs produced {len(set(hashes))} distinct masters: {hashes}"
    assert all(d == decisions[0] for d in decisions), "10 runs did not agree on the decisions"
