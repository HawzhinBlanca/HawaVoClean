"""The release ledger is attributable, ordered, and tamper-evident."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release_evidence import (
    EvidenceError,
    build_entry,
    read_and_verify_ledger,
    verify_baseline,
    verify_schema_contract,
)

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "evidence" / "release"
COMMIT = "e00bd1b851a26a52e5d2fa30bb18bc613ae37668"


def _entry(sequence: int, previous: str | None, *, summary: str = "proved") -> dict[str, object]:
    return build_entry(
        sequence=sequence,
        previous=previous,
        recorded_at=f"2026-08-21T08:2{sequence}:00Z",
        task_id="T0.1",
        source_commit=COMMIT,
        command="proof command",
        tools={"python": "3.14.6"},
        inputs=[],
        status="passed",
        summary=summary,
        outputs=[],
        known_limits=["unit fixture"],
    )


def _write(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n" for entry in entries
        ),
        encoding="utf-8",
    )


def test_committed_release_evidence_verifies() -> None:
    verify_schema_contract(EVIDENCE / "evidence-entry.schema.json")
    summary = verify_baseline(EVIDENCE / "baseline.json", repo_root=REPO)
    entries = read_and_verify_ledger(EVIDENCE / "ledger.jsonl")
    assert summary == {"commits": 5, "tracked_artifacts": 7, "external_checked": 0}
    assert entries[0]["task_id"] == "T0.1"


def test_editing_an_entry_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _entry(1, None)
    _write(path, [first])
    raw = path.read_text(encoding="utf-8").replace("proved", "fabricated")
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(EvidenceError, match="self digest mismatch"):
        read_and_verify_ledger(path)


def test_reordering_or_deleting_entries_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _entry(1, None)
    second = _entry(2, str(first["entry_sha256"]))
    _write(path, [second, first])
    with pytest.raises(EvidenceError, match="sequence"):
        read_and_verify_ledger(path)

    _write(path, [second])
    with pytest.raises(EvidenceError, match="sequence"):
        read_and_verify_ledger(path)


def test_truncated_tail_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _entry(1, None)
    path.write_text(json.dumps(first) + '\n{"schema_version":', encoding="utf-8")
    with pytest.raises(EvidenceError, match="not complete JSON"):
        read_and_verify_ledger(path)


def test_chain_link_is_checked_separately_from_self_hash(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _entry(1, None)
    second = _entry(2, "0" * 64)
    _write(path, [first, second])
    with pytest.raises(EvidenceError, match="does not link"):
        read_and_verify_ledger(path)
