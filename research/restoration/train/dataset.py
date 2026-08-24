"""Leakage-proof dataset partitioning and manifest management for restoration training."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UtteranceEntry:
    """A clean speech utterance entry in a dataset manifest."""

    utterance_id: str
    speaker_id: str
    audio_path: str
    duration_s: float
    sha256: str
    session_id: str


class SplitManager:
    """Manages train, dev, calibration, and locked acceptance splits with zero cross-split leakage."""

    VALID_SPLITS = ("train", "development", "calibration", "locked_acceptance", "locked_corruption")

    def __init__(self, manifests_dir: Path | str) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.splits: dict[str, list[UtteranceEntry]] = {s: [] for s in self.VALID_SPLITS}

    def add_utterance(self, split: str, entry: UtteranceEntry) -> None:
        """Add utterance to split ensuring no utterance crosses splits."""
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        # Check for leakage across existing splits
        for other_split, entries in self.splits.items():
            for e in entries:
                if e.utterance_id == entry.utterance_id or e.sha256 == entry.sha256:
                    raise ValueError(
                        f"Data leakage detected! Utterance {entry.utterance_id} (hash {entry.sha256[:8]}) already in split '{other_split}'"
                    )

        self.splits[split].append(entry)

    def save_manifests(self) -> dict[str, str]:
        """Save JSONL manifests and return SHA-256 hash of each manifest."""
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        manifest_hashes: dict[str, str] = {}

        for split, entries in self.splits.items():
            manifest_file = self.manifests_dir / f"{split}.jsonl"
            lines = [json.dumps(entry.__dict__) for entry in entries]
            content = "\n".join(lines) + ("\n" if lines else "")
            with open(manifest_file, "w", encoding="utf-8") as f:
                f.write(content)

            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            manifest_hashes[split] = h

        return manifest_hashes
