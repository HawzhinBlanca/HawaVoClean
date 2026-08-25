"""The release ledger is attributable, ordered, and tamper-evident."""

from __future__ import annotations

import json
import re
import subprocess
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


def _pinned_commits() -> dict[str, str]:
    """Every 40-hex commit the baseline names, keyed by its JSON path."""
    baseline = json.loads((REPO / "evidence" / "release" / "baseline.json").read_text())
    found: dict[str, str] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str) and re.fullmatch(r"[0-9a-f]{40}", node):
            found[path] = node

    walk(baseline, "baseline")
    return found


def _trunk_ref() -> str:
    """
    A ref naming the trunk's history, in whatever checkout we are standing in.

    `main` is not it. A pull-request build checks out a detached merge commit
    and creates no local `main`, so hard-coding that name made the first version
    of this guard fail in CI on three anchors that are perfectly reachable —
    the same class of environment assumption the guard exists to catch, made by
    the guard itself.

    Raises rather than returning a fallback: a reachability test that cannot
    find the trunk must fail loudly, not pass vacuously.
    """
    for ref in ("origin/main", "main", "HEAD"):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            cwd=REPO,
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return ref
    raise AssertionError("no trunk ref found: tried origin/main, main, HEAD")


def _reachable_from_a_durable_ref(sha: str) -> bool:
    """
    Is this commit reachable from something a fresh clone will fetch?

    Deliberately NOT `git cat-file -e`. An object can exist in a local
    repository — in the reflog, or as a dangling object — long after the last
    ref pointing at it is gone, so an existence check passes on the machine
    that deleted the ref and fails in CI, which is exactly how this went wrong:
    `release_evidence.py verify` reported `missing commit object` on every
    pull request while the same command passed locally.

    A branch tip is not durable either; branches get deleted once merged. The
    two things a clone reliably carries are the trunk's history and tags.
    """
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, _trunk_ref()],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if ancestor.returncode == 0:
        return True
    tagged = subprocess.run(
        ["git", "tag", "--points-at", sha],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(tagged.stdout.strip())


@pytest.mark.parametrize("where,sha", sorted(_pinned_commits().items()))
def test_every_baseline_pinned_commit_survives_a_fresh_clone(where: str, sha: str) -> None:
    """
    The baseline pins commits as provenance anchors. If one becomes
    unreachable the whole ledger stops verifying — and it stops verifying for
    everyone else first, because the person who deleted the ref still has the
    object locally.

    Five commits are pinned. Four are ancestors of `main`, so merging kept them
    safe. The fifth was only ever a branch tip on work that never merged, and
    deleting that branch — correctly judged to hold no content `main` lacked —
    orphaned it. Content equivalence is not the test; reachability is.
    """
    assert _reachable_from_a_durable_ref(sha), (
        f"{where} = {sha} is not an ancestor of the trunk and carries no tag, so a fresh "
        f"clone cannot see it and `release_evidence.py verify` will fail in CI. "
        f"Tag it rather than relying on a branch nobody has a reason to keep."
    )
