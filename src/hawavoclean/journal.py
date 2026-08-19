"""Append-only job journal with fsync and recovery support."""

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any


class JournalEvent(StrEnum):
    """Authoritative event types as defined in BLUEPRINT.md section 17.2."""

    JOB_STARTED = "JOB_STARTED"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    AUDIO_PROBED = "AUDIO_PROBED"
    AUDIO_DECODED = "AUDIO_DECODED"
    SEGMENTATION_COMPLETE = "SEGMENTATION_COMPLETE"
    UNIT_ENHANCED = "UNIT_ENHANCED"
    GUARD_A_COMPLETE = "GUARD_A_COMPLETE"
    UNIT_SELECTED = "UNIT_SELECTED"
    FINISH_COMPLETE = "FINISH_COMPLETE"
    GUARD_B_COMPLETE = "GUARD_B_COMPLETE"
    UNIT_COMMITTED = "UNIT_COMMITTED"
    ASSEMBLY_COMPLETE = "ASSEMBLY_COMPLETE"
    FINAL_VALIDATION_PASSED = "FINAL_VALIDATION_PASSED"
    OUTPUT_PUBLISHED = "OUTPUT_PUBLISHED"
    JOB_COMPLETE = "JOB_COMPLETE"


class JobJournal:
    """Manages an append-only JSONL journal with immediate fsync for crash safety."""

    def __init__(self, journal_path: Path | str) -> None:
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch(mode=0o600)

    def append(self, event: JournalEvent, payload: dict[str, Any] | None = None) -> None:
        """Append an event line and fsync to disk."""
        record = {
            "event": str(event),
            "payload": payload or {},
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read_events(self) -> list[dict[str, Any]]:
        """Read all validated journal records from disk."""
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as f:
            for _line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    events.append(record)
                except Exception:
                    # Ignore incomplete trailing lines from hard crash
                    break
        return events

    def get_committed_units(self) -> set[int]:
        """Return set of unit IDs that have reached UNIT_COMMITTED."""
        committed: set[int] = set()
        for rec in self.read_events():
            if rec.get("event") == JournalEvent.UNIT_COMMITTED:
                uid = rec.get("payload", {}).get("unit_id")
                if uid is not None:
                    committed.add(int(uid))
        return committed
