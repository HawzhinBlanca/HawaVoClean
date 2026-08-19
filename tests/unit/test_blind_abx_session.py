"""Blind ABX session: randomized trials and vote aggregation are real."""

import json
from pathlib import Path

from voiceclean.eval.blind_abx import (
    BlindListeningSession,
    ListenerVote,
    generate_blind_trial_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def test_session_randomizes_and_aggregates(tmp_path: Path) -> None:
    session = BlindListeningSession(seed=7)
    for i in range(10):
        session.create_trial(f"item{i}", "sysX", f"x{i}.wav", "sysY", f"y{i}.wav")

    # Blinding is real: across 10 trials both orderings must occur.
    a_ids = {t.system_a_real_id for t in session.trials}
    assert a_ids == {"sysX", "sysY"}

    session.record_vote(
        ListenerVote(
            trial_id=session.trials[0].trial_id,
            listener_id="l1",
            fidelity_score=4,
            naturalness_score=4,
            clarity_score=5,
            artifact_score=5,
            preferred_sample="A",
        )
    )
    session.record_vote(
        ListenerVote(
            trial_id=session.trials[1].trial_id,
            listener_id="l1",
            fidelity_score=3,
            naturalness_score=3,
            clarity_score=3,
            artifact_score=4,
            preferred_sample="B",
        )
    )
    stats = session.compute_summary_statistics()
    assert stats["total_votes"] == 2
    assert sum(stats["system_preferences"].values()) == 2

    session.save_session(tmp_path / "session.json")
    data = json.loads((tmp_path / "session.json").read_text())
    assert data["trials_count"] == 10


def test_generate_manifest_from_corpus(tmp_path: Path) -> None:
    out = generate_blind_trial_manifest(
        system_a_manifest=REPO / "data" / "acceptance" / "manifest.json",
        system_b_manifest=REPO / "data" / "acceptance" / "manifest.json",
        output_sheet_path=tmp_path / "sheet.json",
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["trials_count"] > 0
