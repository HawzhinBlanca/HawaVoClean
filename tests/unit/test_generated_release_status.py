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
    gate = json.loads(status.FULL_GATE_PROOF.read_text(encoding="utf-8"))
    completed, total = status._task_counts(status._validate_ledger())
    assert gate["source_commit"] in rendered
    assert "unapproved" in rendered
    assert "final candidate must rerun" in rendered
    assert f"{completed}/{total} tasks" in rendered


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


def test_generator_rejects_ledger_content_with_a_stale_self_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entries = [
        json.loads(line)
        for line in (ROOT / "evidence" / "release" / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    entries[-1]["result"]["summary"] += " (tampered)"
    broken = tmp_path / "ledger.jsonl"
    broken.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    monkeypatch.setattr(status, "LEDGER", broken)
    with pytest.raises(status.ReleaseStatusError, match="digest does not recompute"):
        status.render_status()


def test_latest_ledger_result_is_the_only_completion_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "true-10-plan.md"
    plan.write_text("- [ ] **T9.1 — Corrected task**\n", encoding="utf-8")
    monkeypatch.setattr(status, "PLAN", plan)
    entries = [
        {"task_id": "T9.1", "result": {"status": "passed"}},
        {"task_id": "T9.1", "result": {"status": "blocked"}},
    ]
    assert status._task_counts(entries) == (0, 1)


def test_changed_current_bound_input_reopens_an_old_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "true-10-plan.md"
    plan.write_text("- [ ] **T9.1 — Current-bound task**\n", encoding="utf-8")
    contract = tmp_path / "governance.json"
    contract.write_text('{"revision":2}\n', encoding="utf-8")
    monkeypatch.setattr(status, "PLAN", plan)
    monkeypatch.setattr(status, "CURRENT_EVIDENCE_BINDINGS", {"T9.1": contract})
    stale = [
        {
            "task_id": "T9.1",
            "inputs": [{"path": contract.as_posix(), "sha256": "0" * 64}],
            "result": {"status": "passed"},
        }
    ]
    assert status._task_counts(stale) == (0, 1)


def test_current_bound_pass_requires_exact_input_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "true-10-plan.md"
    plan.write_text("- [x] **T9.1 — Current-bound task**\n", encoding="utf-8")
    contract = tmp_path / "governance.json"
    contract.write_text('{"revision":2}\n', encoding="utf-8")
    monkeypatch.setattr(status, "PLAN", plan)
    monkeypatch.setattr(status, "CURRENT_EVIDENCE_BINDINGS", {"T9.1": contract})
    current = [
        {
            "task_id": "T9.1",
            "inputs": [
                {
                    "path": contract.as_posix(),
                    "sha256": status._sha256(contract),
                }
            ],
            "result": {"status": "passed"},
        }
    ]
    assert status._task_counts(current) == (1, 1)


def test_generator_rejects_evidence_for_a_task_outside_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = tmp_path / "true-10-plan.md"
    plan.write_text("- [ ] **T9.1 — Known task**\n", encoding="utf-8")
    monkeypatch.setattr(status, "PLAN", plan)
    with pytest.raises(status.ReleaseStatusError, match="unknown task T9.2"):
        status._task_counts([{"task_id": "T9.2", "result": {"status": "passed"}}])


def test_generator_rejects_checkbox_not_backed_by_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = status.PLAN.read_text(encoding="utf-8").replace("- [ ] **T4.6", "- [x] **T4.6", 1)
    corrupt = tmp_path / "true-10-plan.md"
    corrupt.write_text(plan, encoding="utf-8")
    monkeypatch.setattr(status, "PLAN", corrupt)
    with pytest.raises(status.ReleaseStatusError, match="checkboxes disagree"):
        status.render_status()


def test_generator_rejects_a_fabricated_release_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proof = json.loads(status.FULL_GATE_PROOF.read_text(encoding="utf-8"))
    proof["reproducibility"]["artifact_sha256"]["wheel"] = "fabricated"
    corrupt = tmp_path / "release-proof.json"
    corrupt.write_text(json.dumps(proof), encoding="utf-8")
    monkeypatch.setattr(status, "FULL_GATE_PROOF", corrupt)
    with pytest.raises(status.ReleaseStatusError, match="artifact wheel is not a valid digest"):
        status.render_status()
