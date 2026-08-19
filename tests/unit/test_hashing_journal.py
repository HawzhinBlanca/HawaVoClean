"""Unit tests for SHA-256 hashing, cache key derivation, and append-only journal."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from voiceclean.hashing import compute_cache_key, hash_numpy
from voiceclean.journal import JobJournal, JournalEvent


@pytest.mark.unit
def test_hash_numpy_deterministic() -> None:
    arr1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    arr2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert hash_numpy(arr1) == hash_numpy(arr2)


@pytest.mark.unit
def test_cache_key_changes_on_config_change() -> None:
    pcm = b"\x00\x00\x80?" * 100
    key1 = compute_cache_key(pcm, 48000, {"model": "v1"}, "guard1", "config_hash_A", "1.0.0")
    key2 = compute_cache_key(pcm, 48000, {"model": "v1"}, "guard1", "config_hash_B", "1.0.0")
    assert key1 != key2


@pytest.mark.unit
def test_job_journal_fsync_and_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        jpath = Path(tmpdir) / "journal.jsonl"
        journal = JobJournal(jpath)

        journal.append(JournalEvent.JOB_STARTED, {"job": "test"})
        journal.append(JournalEvent.UNIT_COMMITTED, {"unit_id": 0})
        journal.append(JournalEvent.UNIT_COMMITTED, {"unit_id": 1})

        events = journal.read_events()
        assert len(events) == 3
        committed = journal.get_committed_units()
        assert committed == {0, 1}
