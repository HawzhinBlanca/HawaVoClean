"""Unit tests for SHA-256 hashing, cache key derivation, and append-only journal."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hawavoclean.hashing import compute_cache_key, compute_job_id, hash_numpy
from hawavoclean.journal import JobJournal, JournalEvent


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


@pytest.mark.unit
def test_job_id_separates_a_restoration_from_the_master_it_replaces() -> None:
    """Restore-only inputs are part of the job's identity.

    Without them a natural master and a generative reconstruction of the same
    file carried the same job_id, and so did reconstructions built from two
    different speaker profiles -- measured on one fixture, four runs (natural,
    character_01, character_07, natural again) all reported
    ``19ddba6060ac85c9``. That id keys the report, the provenance record and
    the dither seed, so an auditor holding two of those reports had no field
    that told a recording from a reconstruction.
    """
    base = {
        "input_hash": "a" * 64,
        "config_hash": "b" * 64,
        "core_hash": "core-x",
        "guard_hash": "guard-y",
        "tool_version": "3.3.0",
    }
    natural = compute_job_id(**base)
    first = compute_job_id(**base, restore_context="restore:character_01:auto:None")
    second = compute_job_id(**base, restore_context="restore:character_07:auto:None")

    assert len({natural, first, second}) == 3, "each run must own its identity"
    # Deterministic: the same inputs still name the same job.
    assert first == compute_job_id(**base, restore_context="restore:character_01:auto:None")


@pytest.mark.unit
def test_a_natural_job_id_is_exactly_what_it_always_was() -> None:
    """Adding restore inputs must not rename every natural job.

    The id seeds the dither, the dither reaches the published bytes, and the
    release evidence pins those bytes per case. A natural run's identity is
    therefore still the five-part composite, byte for byte.
    """
    base = {
        "input_hash": "a" * 64,
        "config_hash": "b" * 64,
        "core_hash": "core-x",
        "guard_hash": "guard-y",
        "tool_version": "3.3.0",
    }
    legacy = hashlib.sha256(
        f"{base['input_hash']}:{base['config_hash']}:{base['core_hash']}:"
        f"{base['guard_hash']}:{base['tool_version']}".encode()
    ).hexdigest()[:16]

    assert compute_job_id(**base) == legacy
    assert compute_job_id(**base, restore_context=None) == legacy
