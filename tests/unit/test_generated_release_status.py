from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import generate_release_status as status

ROOT = Path(__file__).resolve().parents[2]


def test_committed_release_status_is_exactly_generated() -> None:
    assert (ROOT / "docs" / "generated-release-status.md").read_text(
        encoding="utf-8"
    ) == status.render_status()


def test_snapshot_names_the_exact_proofs_and_open_human_gates() -> None:
    rendered = status.render_status()
    gate = json.loads(
        (ROOT / "evidence" / "release" / "t3.1-release-gate-proof.json").read_text(encoding="utf-8")
    )
    assert gate["source_commit"] in rendered
    assert "unapproved" in rendered
    assert "final candidate must rerun" in rendered


def test_generator_rejects_a_broken_ledger_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entries = [
        json.loads(line)
        for line in (ROOT / "evidence" / "release" / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    entries[-1]["previous_entry_sha256"] = "0" * 64
    broken = tmp_path / "ledger.jsonl"
    broken.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    monkeypatch.setattr(status, "LEDGER", broken)
    with pytest.raises(status.ReleaseStatusError, match="does not link"):
        status.render_status()
