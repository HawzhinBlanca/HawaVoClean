"""Tests for the explicit server job and batch finite state machines."""

from __future__ import annotations

from pathlib import Path

import pytest

from hawavoclean.server.jobs import (
    ALLOWED_JOB_TRANSITIONS,
    JobRecord,
    JobState,
    derive_batch_state,
    transition_job_state,
)


def _record(state: JobState = "queued") -> JobRecord:
    return JobRecord(
        job_id="job-1",
        input_path=Path("/tmp/in.wav"),
        output_path=Path("/tmp/out.wav"),
        report_path=Path("/tmp/out.json"),
        profile="studio",
        overwrite=True,
        state=state,
    )


def test_job_state_machine_allows_every_declared_transition() -> None:
    for source, targets in ALLOWED_JOB_TRANSITIONS.items():
        for target in targets:
            rec = _record(source)
            assert transition_job_state(rec, target) == target
            assert rec.state == target


def test_job_state_machine_rejects_every_undeclared_transition() -> None:
    all_states: tuple[JobState, ...] = (
        "queued",
        "running",
        "done",
        "failed",
        "cancelled",
        "interrupted",
    )
    for source in all_states:
        allowed = ALLOWED_JOB_TRANSITIONS[source]
        for target in all_states:
            if target in allowed:
                continue
            rec = _record(source)
            with pytest.raises(ValueError, match="Illegal job state transition"):
                transition_job_state(rec, target)
            assert rec.state == source


def test_derive_batch_state_covers_all_phases() -> None:
    assert (
        derive_batch_state(
            total=2, completed=0, failed=0, cancelled=0, running=1, queued=1, is_paused=True
        )
        == "paused"
    )
    assert (
        derive_batch_state(
            total=2, completed=0, failed=0, cancelled=0, running=1, queued=1, is_paused=False
        )
        == "running"
    )
    assert (
        derive_batch_state(
            total=2, completed=0, failed=0, cancelled=0, running=0, queued=2, is_paused=False
        )
        == "queued"
    )
    assert (
        derive_batch_state(
            total=2, completed=0, failed=0, cancelled=2, running=0, queued=0, is_paused=False
        )
        == "cancelled"
    )
    assert (
        derive_batch_state(
            total=2, completed=0, failed=2, cancelled=0, running=0, queued=0, is_paused=False
        )
        == "failed"
    )
    assert (
        derive_batch_state(
            total=2, completed=1, failed=1, cancelled=0, running=0, queued=0, is_paused=False
        )
        == "done"
    )
    assert (
        derive_batch_state(
            total=0,
            completed=0,
            failed=0,
            cancelled=0,
            running=0,
            queued=0,
            is_paused=False,
            stored_state="done",
        )
        == "done"
    )
    assert (
        derive_batch_state(
            total=0,
            completed=0,
            failed=0,
            cancelled=0,
            running=0,
            queued=0,
            is_paused=False,
            stored_state="unknown",
        )
        == "queued"
    )
