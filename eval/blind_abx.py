"""Double-blind randomized ABX and pairwise listening test engine."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.corpus import load_corpus_manifest
from voiceclean.hashing import hash_bytes


@dataclass
class BlindTrial:
    """Individual randomized trial presenting Sample A vs Sample B."""

    trial_id: str
    item_id: str
    sample_a_path: str
    sample_b_path: str
    system_a_real_id: str
    system_b_real_id: str
    randomization_seed: int


@dataclass
class ListenerVote:
    """Anonymized vote recorded from a native listener."""

    trial_id: str
    listener_id: str
    fidelity_score: int  # 1..5
    naturalness_score: int  # 1..5
    clarity_score: int  # 1..5
    artifact_score: int  # 1..5 (5 = no artifacts)
    preferred_sample: str  # "A", "B", or "equal"


class BlindListeningSession:
    """Manages randomized trials, blinded evaluation sessions, and result aggregation."""

    def __init__(self, session_id: str | None = None, seed: int = 42) -> None:
        self.session_id = session_id or hash_bytes(str(random.random()).encode("utf-8"))[:12]
        self.rng = random.Random(seed)
        self.trials: list[BlindTrial] = []
        self.votes: list[ListenerVote] = []

    def create_trial(
        self,
        item_id: str,
        system_1_name: str,
        system_1_path: str,
        system_2_name: str,
        system_2_path: str,
    ) -> BlindTrial:
        """Create a randomized A/B trial hiding real system names."""
        flip = self.rng.choice([True, False])
        if flip:
            sys_a, path_a = system_1_name, system_1_path
            sys_b, path_b = system_2_name, system_2_path
        else:
            sys_a, path_a = system_2_name, system_2_path
            sys_b, path_b = system_1_name, system_1_path

        trial_id = f"trial_{len(self.trials) + 1:04d}_{item_id}"
        trial = BlindTrial(
            trial_id=trial_id,
            item_id=item_id,
            sample_a_path=path_a,
            sample_b_path=path_b,
            system_a_real_id=sys_a,
            system_b_real_id=sys_b,
            randomization_seed=self.rng.randint(1000, 999999),
        )
        self.trials.append(trial)
        return trial

    def record_vote(self, vote: ListenerVote) -> None:
        """Record listener evaluation."""
        self.votes.append(vote)

    def compute_summary_statistics(self) -> dict[str, Any]:
        """Aggregate ratings and compute win rates."""
        system_pref_counts: dict[str, int] = {}
        for v in self.votes:
            # Find trial
            trial = next((t for t in self.trials if t.trial_id == v.trial_id), None)
            if not trial:
                continue

            if v.preferred_sample == "A":
                winner = trial.system_a_real_id
                system_pref_counts[winner] = system_pref_counts.get(winner, 0) + 1
            elif v.preferred_sample == "B":
                winner = trial.system_b_real_id
                system_pref_counts[winner] = system_pref_counts.get(winner, 0) + 1

        total_votes = len(self.votes)
        return {
            "total_trials": len(self.trials),
            "total_votes": total_votes,
            "system_preferences": system_pref_counts,
        }

    def save_session(self, output_path: Path | str) -> None:
        """Persist session and votes to JSON."""
        data = {
            "session_id": self.session_id,
            "trials_count": len(self.trials),
            "votes_count": len(self.votes),
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "item_id": t.item_id,
                    "sample_a_path": t.sample_a_path,
                    "sample_b_path": t.sample_b_path,
                    "system_a_real_id": t.system_a_real_id,
                    "system_b_real_id": t.system_b_real_id,
                }
                for t in self.trials
            ],
            "summary": self.compute_summary_statistics(),
        }
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def generate_blind_trial_manifest(
    system_a_manifest: Path | str,
    system_b_manifest: Path | str,
    output_sheet_path: Path | str = "eval/blind_abx_session.json",
    seed: int = 42,
) -> Path:
    """Generate a randomized blind listening trial session from two corpus manifests."""
    m_a = load_corpus_manifest(system_a_manifest)
    m_b = load_corpus_manifest(system_b_manifest)

    session = BlindListeningSession(seed=seed)
    items_b_map = {item.id: item for item in m_b.items}

    for it_a in m_a.items:
        it_b = items_b_map.get(it_a.id, it_a)
        session.create_trial(
            item_id=it_a.id,
            system_1_name="System_A",
            system_1_path=it_a.audio_path,
            system_2_name="System_B",
            system_2_path=it_b.audio_path,
        )

    out_p = Path(output_sheet_path).resolve()
    session.save_session(out_p)
    return out_p
