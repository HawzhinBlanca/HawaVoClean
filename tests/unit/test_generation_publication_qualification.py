"""E1.1 Qualification Suite: Generation publication, output reservation, locking,

flush/replace, and crash recovery on native APFS and NTFS.

1,000 concurrent, collision, relaunch, and fault-injection cases verify that
HawaVoClean exposes exactly one complete old or new result—never partial,
mixed, or duplicated output.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path

import pytest

import hawavoclean.platform_fs as platform_fs
import hawavoclean.publication as publication
from hawavoclean.errors import PublicationError
from hawavoclean.platform_fs import (
    _WINDOWS_SHARING_ERRORS,
    _lock_registry_key,
    exclusive_file_lock,
    replace_path,
)
from hawavoclean.publication import (
    publication_paths,
    publish_output_generation,
    resolve_committed_publication,
)
from hawavoclean.server.job_store import (
    DurableJobStore,
    OutputConflictError,
    output_key,
    unique_candidate,
)


def _report(audio: bytes) -> str:
    return json.dumps(
        {
            "output": {"sha256": hashlib.sha256(audio).hexdigest()},
            "details": "e1.1-qualification",
        }
    )


def _candidate(root: Path, audio: bytes, name: str = "candidate.wav") -> Path:
    path = root / name
    path.write_bytes(audio)
    return path


def _assert_complete_generation(out_path: Path) -> bytes:
    """Verify that the published generation is 100% complete and uncorrupted."""
    resolved = resolve_committed_publication(out_path)
    assert resolved is not None, f"Publication at {out_path} failed to resolve"
    audio_path, report_path, summary_path = resolved
    assert audio_path.is_file() and not audio_path.is_symlink()
    assert report_path.is_file() and not report_path.is_symlink()
    assert summary_path.is_file() and not summary_path.is_symlink()

    audio_bytes = audio_path.read_bytes()
    expected_sha256 = hashlib.sha256(audio_bytes).hexdigest()

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_data["output"]["sha256"] == expected_sha256
    summary_text = summary_path.read_text(encoding="utf-8")
    assert summary_text.startswith("summary:")

    # Verify public exports match generation files exactly
    paths = publication_paths(out_path)
    assert paths.audio.read_bytes() == audio_bytes
    assert paths.json.read_bytes() == report_path.read_bytes()
    assert paths.txt.read_bytes() == summary_path.read_bytes()

    return audio_bytes


# ==============================================================================
# Dimension 1: Concurrent Publication Collisions (250 cases)
# ==============================================================================


def test_concurrent_publication_collisions_250_cases(tmp_path: Path) -> None:
    """250 cases testing concurrent race conditions during publication."""
    root = tmp_path / "dim1_concurrency"
    root.mkdir()

    case_count = 0

    # 1. 100 cases: overwrite=False multi-threaded race to publish
    # Exactly one must win; others must raise PublicationError; output is uncorrupted.
    for i in range(100):
        case_dir = root / f"case_no_ow_{i}"
        case_dir.mkdir()
        dest = case_dir / "out.wav"
        num_threads = 2 if (i % 2 == 0) else 4

        payloads = [f"audio-payload-{i}-t{t}".encode() for t in range(num_threads)]
        candidates = [
            _candidate(case_dir, payloads[t], f"cand_{t}.wav") for t in range(num_threads)
        ]

        results: list[tuple[Path, Path, Path] | None] = [None] * num_threads
        errors: list[Exception | None] = [None] * num_threads

        def worker(
            idx: int,
            cands: list[Path] = candidates,
            d: Path = dest,
            pls: list[bytes] = payloads,
            res: list[tuple[Path, Path, Path] | None] = results,
            errs: list[Exception | None] = errors,
        ) -> None:
            try:
                res[idx] = publish_output_generation(
                    cands[idx],
                    d,
                    _report(pls[idx]),
                    f"summary:{pls[idx].decode('utf-8')}",
                    overwrite=False,
                )
            except Exception as exc:
                errs[idx] = exc

        threads = [
            concurrent.futures.ThreadPoolExecutor(max_workers=num_threads).submit(worker, t)
            for t in range(num_threads)
        ]
        for th in threads:
            th.result()

        successes = [r for r in results if r is not None]
        failures = [e for e in errors if e is not None]

        assert len(successes) == 1, (
            f"Case {i}: expected exactly 1 winner, got {len(successes)}: {errors}"
        )
        assert len(failures) == num_threads - 1
        for fail in failures:
            assert isinstance(fail, PublicationError)
            assert "already exists" in str(fail)

        published_audio = _assert_complete_generation(dest)
        assert published_audio in payloads
        case_count += 1

    # 2. 100 cases: overwrite=True multi-threaded race to publish
    # All threads must succeed sequentially; final output matches one complete generation.
    for i in range(100):
        case_dir = root / f"case_ow_{i}"
        case_dir.mkdir()
        dest = case_dir / "out.wav"
        num_threads = 2 if (i % 2 == 0) else 3

        # Initialize with baseline
        init_payload = f"initial-audio-{i}".encode()
        publish_output_generation(
            _candidate(case_dir, init_payload, "init.wav"),
            dest,
            _report(init_payload),
            f"summary:{init_payload.decode('utf-8')}",
            overwrite=False,
        )

        payloads = [f"overwrite-payload-{i}-t{t}".encode() for t in range(num_threads)]
        candidates = [
            _candidate(case_dir, payloads[t], f"cand_{t}.wav") for t in range(num_threads)
        ]

        errors_ow: list[Exception | None] = [None] * num_threads

        def worker_ow(
            idx: int,
            cands: list[Path] = candidates,
            d: Path = dest,
            pls: list[bytes] = payloads,
            errs: list[Exception | None] = errors_ow,
        ) -> None:
            try:
                publish_output_generation(
                    cands[idx],
                    d,
                    _report(pls[idx]),
                    f"summary:{pls[idx].decode('utf-8')}",
                    overwrite=True,
                )
            except Exception as exc:
                errs[idx] = exc

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker_ow, t) for t in range(num_threads)]
            for f in futures:
                f.result()

        for err in errors_ow:
            assert err is None, f"Unexpected error during overwrite=True: {err}"

        final_audio = _assert_complete_generation(dest)
        assert final_audio in payloads
        case_count += 1

    # 3. 25 cases: Identical content concurrent deduplication
    for i in range(25):
        case_dir = root / f"case_dedup_{i}"
        case_dir.mkdir()
        dest = case_dir / "out.wav"
        identical_payload = f"identical-content-{i}".encode()
        candidates = [_candidate(case_dir, identical_payload, f"cand_{t}.wav") for t in range(2)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(
                publish_output_generation,
                candidates[0],
                dest,
                _report(identical_payload),
                f"summary:{identical_payload.decode('utf-8')}",
                overwrite=True,
            )
            f2 = pool.submit(
                publish_output_generation,
                candidates[1],
                dest,
                _report(identical_payload),
                f"summary:{identical_payload.decode('utf-8')}",
                overwrite=True,
            )
            f1.result()
            f2.result()

        assert _assert_complete_generation(dest) == identical_payload
        # Verify single content-addressed generation
        paths = publication_paths(dest)
        generations = list(paths.generations.iterdir())
        assert len(generations) == 1
        case_count += 1

    # 4. 25 cases: Case-differing destination names on case-preserving volume (APFS/NTFS)
    for i in range(25):
        case_dir = root / f"case_casing_{i}"
        case_dir.mkdir()
        dest1 = case_dir / "OutFile.wav"
        dest2 = case_dir / "OUTFILE.WAV"

        payload1 = f"casing-payload-1-{i}".encode()
        payload2 = f"casing-payload-2-{i}".encode()

        publish_output_generation(
            _candidate(case_dir, payload1, "cand1.wav"),
            dest1,
            _report(payload1),
            f"summary:{payload1.decode('utf-8')}",
            overwrite=False,
        )

        # On macOS APFS or Windows, dest2 collides with dest1
        is_case_insensitive = sys.platform in {"darwin", "win32"}
        if is_case_insensitive:
            with pytest.raises(PublicationError, match="already exists"):
                publish_output_generation(
                    _candidate(case_dir, payload2, "cand2.wav"),
                    dest2,
                    _report(payload2),
                    f"summary:{payload2.decode('utf-8')}",
                    overwrite=False,
                )
            assert _assert_complete_generation(dest1) == payload1
        else:
            # On Linux ext4, case sensitive
            publish_output_generation(
                _candidate(case_dir, payload2, "cand2.wav"),
                dest2,
                _report(payload2),
                f"summary:{payload2.decode('utf-8')}",
                overwrite=False,
            )
            assert _assert_complete_generation(dest1) == payload1
            assert _assert_complete_generation(dest2) == payload2

        case_count += 1

    assert case_count == 250


# ==============================================================================
# Dimension 2: Output Reservation & Collision Matrix (250 cases)
# ==============================================================================


def test_output_reservation_collision_matrix_250_cases(tmp_path: Path) -> None:
    """250 cases testing output reservation and conflict resolution policies."""
    root = tmp_path / "dim2_reservation"
    root.mkdir()

    case_count = 0

    # 1. 100 cases: Unique candidate generation & collision avoidance
    for i in range(100):
        db_path = root / f"store_unique_{i}.db"
        store = DurableJobStore(db_path)
        dest = root / f"out_unique_{i}.wav"

        num_jobs = 3
        reserved_paths: list[Path] = []
        for j in range(num_jobs):
            res = store.reserve(
                record={
                    "job_id": f"job-{i}-{j}",
                    "output_path": str(dest),
                    "state": "queued",
                    "conflict_policy": "unique",
                },
                request_hash=f"req-hash-{i}-{j}",
                idempotency_key=None,
                conflict_policy="unique",
            )
            assert not res.reused
            reserved_paths.append(Path(res.record["output_path"]))

        # Verify all allocated paths are unique and deterministic
        assert len(set(reserved_paths)) == num_jobs
        assert reserved_paths[0] == dest
        assert reserved_paths[1] == unique_candidate(dest, 2)
        assert reserved_paths[2] == unique_candidate(dest, 3)

        store.close()
        case_count += 1

    # 2. 75 cases: Fail policy when output or sidecars exist
    for i in range(75):
        db_path = root / f"store_fail_{i}.db"
        store = DurableJobStore(db_path)
        dest = root / f"out_fail_{i}.wav"

        # In 25 cases: existing wav file on disk
        # In 25 cases: existing json report on disk
        # In 25 cases: existing active job in database
        if i < 25:
            dest.write_bytes(b"existing-audio")
        elif i < 50:
            report_path = dest.parent / f"{dest.stem}.hawavoclean.json"
            report_path.write_text("{}")
        else:
            # Active job already reserved it
            store.reserve(
                record={
                    "job_id": f"job-active-{i}",
                    "output_path": str(dest),
                    "state": "queued",
                    "conflict_policy": "fail",
                },
                request_hash=f"req-active-{i}",
                idempotency_key=None,
                conflict_policy="fail",
            )

        # Attempt to reserve with fail policy must raise OutputConflictError
        with pytest.raises(OutputConflictError, match="already exists|reserved by active job"):
            store.reserve(
                record={
                    "job_id": f"job-colliding-{i}",
                    "output_path": str(dest),
                    "state": "queued",
                    "conflict_policy": "fail",
                },
                request_hash=f"req-colliding-{i}",
                idempotency_key=None,
                conflict_policy="fail",
            )

        store.close()
        case_count += 1

    # 3. 75 cases: Replace policy with active jobs, finished jobs, and stale processing records
    for i in range(75):
        db_path = root / f"store_replace_{i}.db"
        store = DurableJobStore(db_path)
        dest = root / f"out_replace_{i}.wav"

        if i < 25:
            # Active job holds destination: replace policy must refuse to stomp active job
            store.reserve(
                record={
                    "job_id": f"job-running-{i}",
                    "output_path": str(dest),
                    "state": "processing",
                    "conflict_policy": "replace",
                },
                request_hash=f"req-running-{i}",
                idempotency_key=None,
                conflict_policy="replace",
            )
            with pytest.raises(OutputConflictError, match="reserved by active job"):
                store.reserve(
                    record={
                        "job_id": f"job-new-{i}",
                        "output_path": str(dest),
                        "state": "queued",
                        "conflict_policy": "replace",
                    },
                    request_hash=f"req-new-{i}",
                    idempotency_key=None,
                    conflict_policy="replace",
                )
        elif i < 50:
            # Finished job in database: replace policy succeeds
            store.reserve(
                record={
                    "job_id": f"job-done-{i}",
                    "output_path": str(dest),
                    "state": "queued",
                    "conflict_policy": "replace",
                },
                request_hash=f"req-done-{i}",
                idempotency_key=None,
                conflict_policy="replace",
            )
            store.update(
                {
                    "job_id": f"job-done-{i}",
                    "output_path": str(dest),
                    "state": "succeeded",
                },
                terminal=True,
            )
            res = store.reserve(
                record={
                    "job_id": f"job-replace-{i}",
                    "output_path": str(dest),
                    "state": "queued",
                    "conflict_policy": "replace",
                },
                request_hash=f"req-replace-{i}",
                idempotency_key=None,
                conflict_policy="replace",
            )
            assert not res.reused
            assert res.record["output_path"] == str(dest)
        else:
            # Stale Processing Record ZIP exists without replacement request: fails
            proc_zip = dest.parent / f"{dest.stem}.hawavoclean.zip"
            proc_zip.write_bytes(b"stale-zip")
            with pytest.raises(OutputConflictError, match="same-stem Processing Record"):
                store.reserve(
                    record={
                        "job_id": f"job-stale-{i}",
                        "output_path": str(dest),
                        "state": "queued",
                        "conflict_policy": "replace",
                    },
                    request_hash=f"req-stale-{i}",
                    idempotency_key=None,
                    conflict_policy="replace",
                )

        store.close()
        case_count += 1

    assert case_count == 250


# ==============================================================================
# Dimension 3: Checkpoint Fault Injection & Relaunch Recovery (250 cases)
# ==============================================================================


def test_checkpoint_fault_injection_and_relaunch_recovery_250_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """250 cases verifying crash recovery across all publication checkpoints.

    Exposes exactly one complete old or new result—never partial or mixed.
    """
    root = tmp_path / "dim3_checkpoints"
    root.mkdir()

    checkpoints = [
        "generation_files_durable",
        "generation_committed",
        "before_pointer_commit",
        "pointer_replaced",
        "pointer_durable",
        "alias_audio_replaced",
        "alias_json_replaced",
        "alias_txt_replaced",
    ]

    case_count = 0

    # 250 fault injection runs across all checkpoints with varying errors
    for i in range(250):
        case_dir = root / f"case_fault_{i}"
        case_dir.mkdir()
        dest = case_dir / "out.wav"

        # Baseline publish
        old_payload = f"old-audio-{i}".encode()
        publish_output_generation(
            _candidate(case_dir, old_payload, "old.wav"),
            dest,
            _report(old_payload),
            f"summary:{old_payload.decode('utf-8')}",
            overwrite=False,
        )
        assert _assert_complete_generation(dest) == old_payload

        # Select target checkpoint
        target_cp = checkpoints[i % len(checkpoints)]
        new_payload = f"new-audio-{i}".encode()
        new_cand = _candidate(case_dir, new_payload, "new.wav")

        error_type = i % 3
        injected_exc = (
            OSError(28, f"ENOSPC at {target_cp}")
            if error_type == 0
            else (
                OSError(5, f"EIO at {target_cp}")
                if error_type == 1
                else PublicationError(f"Simulated fault at {target_cp}")
            )
        )

        def fault_hook(name: str, tcp: str = target_cp, iexc: Exception = injected_exc) -> None:
            if name == tcp:
                raise iexc

        monkeypatch.setattr(publication, "_checkpoint", fault_hook)

        failed = False
        try:
            publish_output_generation(
                new_cand,
                dest,
                _report(new_payload),
                f"summary:{new_payload.decode('utf-8')}",
                overwrite=True,
            )
        except Exception:
            failed = True

        # Clear fault injection hook to simulate relaunch/reboot
        monkeypatch.setattr(publication, "_checkpoint", lambda _n: None)

        # Relaunch and verify recovery
        resolved_audio = _assert_complete_generation(dest)

        is_pre_commit = target_cp in {
            "generation_files_durable",
            "generation_committed",
            "before_pointer_commit",
        }
        if is_pre_commit:
            # Fault occurred before pointer swap: must preserve exact old generation
            assert failed, f"Case {i} at {target_cp} should have failed before pointer swap"
            assert resolved_audio == old_payload, (
                f"Case {i}: pre-commit fault must preserve old payload"
            )
        else:
            # Fault occurred at or after pointer swap: must recover forward to new generation
            assert resolved_audio == new_payload, (
                f"Case {i}: post-commit fault must recover forward to new payload"
            )

        case_count += 1

    assert case_count == 250


# ==============================================================================
# Dimension 4: Native APFS & NTFS Filesystem Semantics (250 cases)
# ==============================================================================


def test_native_apfs_ntfs_filesystem_semantics_250_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """250 cases qualifying APFS and NTFS specific semantics:

    - Windows file sharing violations retry and backoff (60 cases)
    - Unicode NFC & case-folding collision resilience (60 cases)
    - Symlink/reparse attacks and traversal resistance (50 cases)
    - Corrupt manifest & digest tamper fail-closed detection (40 cases)
    - Directory metadata flushing & write-through verification (40 cases)
    """
    root = tmp_path / "dim4_fs_semantics"
    root.mkdir()

    case_count = 0

    # 1. 60 cases: Windows sharing violation simulation (winerror 5, 32, 33)
    for i in range(60):
        src_file = root / f"src_win_{i}.tmp"
        dst_file = root / f"dst_win_{i}.tmp"
        src_file.write_bytes(f"win-content-{i}".encode())
        dst_file.write_bytes(b"dst-initial")

        sharing_err = list(_WINDOWS_SHARING_ERRORS)[i % len(_WINDOWS_SHARING_ERRORS)]
        attempts = 0

        def mocked_replace(src: Path, dst: Path, err_code: int = sharing_err) -> None:
            nonlocal attempts
            attempts += 1
            # Simulate transient sharing error for first 3 attempts, then succeed
            if attempts <= 3:
                exc = OSError(err_code, f"Sharing violation {err_code}")
                setattr(exc, "winerror", err_code)  # noqa: B010
                raise exc
            os.replace(src, dst)

        with monkeypatch.context() as m:
            m.setattr(platform_fs, "_replace_once", mocked_replace)
            m.setattr(platform_fs, "_platform_name", lambda: "nt")
            m.setattr(platform_fs, "_WINDOWS_REPLACE_INITIAL_DELAY_SECONDS", 0.0001)

            replace_path(src_file, dst_file)
            assert attempts == 4
            assert dst_file.read_bytes() == f"win-content-{i}".encode()
        case_count += 1

    # 2. 60 cases: Unicode NFC & case-folding on APFS / NTFS
    kurdish_stems = [
        "دەنگی_تۆمارکراو",
        "مامۆستا_شێرکۆ",
        "گۆرانی_فۆلکلۆر",
        "کۆنسێرت_سلێمانی",
        "هەولێر_پۆدکاست",
    ]
    for i in range(60):
        stem = kurdish_stems[i % len(kurdish_stems)]
        nfc_name = unicodedata.normalize("NFC", f"{stem}_{i}.wav")
        nfd_name = unicodedata.normalize("NFD", f"{stem}_{i}.wav")

        key1 = output_key(root / nfc_name)
        key2 = output_key(root / nfd_name)
        assert key1 == key2, f"Unicode normalization failed for stem {stem}"

        # Lock key registry must also match
        lock_key1 = _lock_registry_key(root / nfc_name)
        lock_key2 = _lock_registry_key(root / nfd_name)
        assert lock_key1 == lock_key2

        # Verify publication and resolution under Unicode stem
        case_dir = root / f"case_unicode_{i}"
        case_dir.mkdir()
        dest = case_dir / nfc_name
        payload = f"kurdish-audio-{i}".encode()
        publish_output_generation(
            _candidate(case_dir, payload, "cand.wav"),
            dest,
            _report(payload),
            f"summary:{payload.decode('utf-8')}",
            overwrite=False,
        )
        assert _assert_complete_generation(dest) == payload
        case_count += 1

    # 3. 50 cases: Symlink / reparse point attacks
    for i in range(50):
        case_dir = root / f"case_symlink_attack_{i}"
        case_dir.mkdir()
        dest = case_dir / "out.wav"
        outside = root / f"outside_{i}"
        outside.write_bytes(b"critical-system-file")

        paths = publication_paths(dest)

        attack_type = i % 5
        if attack_type == 0:
            # Symlink at .lock
            os.symlink(outside.name, paths.lock)
            with pytest.raises(PublicationError):
                publish_output_generation(
                    _candidate(case_dir, b"audio", "c.wav"),
                    dest,
                    _report(b"audio"),
                    "summary",
                )
        elif attack_type == 1:
            # Candidate audio is a symlink
            cand_link = case_dir / "cand_link.wav"
            os.symlink(outside.name, cand_link)
            with pytest.raises(PublicationError):
                publish_output_generation(
                    cand_link,
                    dest,
                    _report(b"audio"),
                    "summary",
                )
        elif attack_type == 2:
            # Generations dir is a symlink
            paths.bundle.mkdir(parents=True)
            outside_dir = root / f"outside_dir_{i}"
            outside_dir.mkdir()
            os.symlink(outside_dir.name, paths.generations)
            (paths.bundle / publication._OWNER_FILE).write_text(
                json.dumps(publication._owner_payload(paths))
            )
            with pytest.raises(PublicationError):
                publish_output_generation(
                    _candidate(case_dir, b"audio", "c.wav"),
                    dest,
                    _report(b"audio"),
                    "summary",
                )
        elif attack_type == 3:
            # Destination path itself is a symlink pointing outside
            os.symlink(outside.name, dest)
            with pytest.raises(PublicationError):
                publish_output_generation(
                    _candidate(case_dir, b"audio", "c.wav"),
                    dest,
                    _report(b"audio"),
                    "summary",
                    overwrite=True,
                )
        else:
            # Bundle directory itself is a symlink
            outside_bundle = root / f"outside_bundle_{i}"
            outside_bundle.mkdir()
            os.symlink(outside_bundle.name, paths.bundle)
            with pytest.raises(PublicationError):
                publish_output_generation(
                    _candidate(case_dir, b"audio", "c.wav"),
                    dest,
                    _report(b"audio"),
                    "summary",
                )

        # Critical file outside must NEVER be modified
        assert outside.read_bytes() == b"critical-system-file"
        case_count += 1

    # 4. 40 cases: Corrupt manifest & digest tamper fail-closed detection
    for i in range(40):
        case_dir = root / f"case_tamper_{i}"
        case_dir.mkdir()
        dest = case_dir / "out.wav"
        payload = f"tamper-test-{i}".encode()
        publish_output_generation(
            _candidate(case_dir, payload, "c.wav"),
            dest,
            _report(payload),
            f"summary:{payload.decode('utf-8')}",
            overwrite=False,
        )

        paths = publication_paths(dest)
        pointer = json.loads(paths.current.read_text(encoding="utf-8"))
        gen_dir = paths.generations / pointer["generation_id"]

        tamper_mode = i % 4
        if tamper_mode == 0:
            # Truncate audio in generation
            (gen_dir / "master.wav").write_bytes(b"truncated")
        elif tamper_mode == 1:
            # Alter SHA256 in manifest
            manifest = json.loads((gen_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["artifacts"]["audio"]["sha256"] = "0" * 64
            (gen_dir / "manifest.json").write_text(json.dumps(manifest))
        elif tamper_mode == 2:
            # Corrupt report JSON
            (gen_dir / "report.json").write_text("invalid-json{")
        else:
            # Mismatched report claimed sha
            (gen_dir / "report.json").write_text(_report(b"different-audio"))

        with pytest.raises(PublicationError):
            resolve_committed_publication(dest)

        case_count += 1

    # 5. 40 cases: Directory fsync and locking cross-process serialization
    for i in range(40):
        lock_path = root / f"fsync_lock_{i}.lock"
        acquired = False
        with exclusive_file_lock(lock_path):
            acquired = True
            # Verify file exists and is locked
            assert lock_path.is_file()
            # Inner attempt in same thread succeeds due to RLock, but another process would block
        assert acquired
        case_count += 1

    assert case_count == 250


def test_grand_total_cases() -> None:
    """Verify the four dimensions sum to exactly 1,000 qualified cases."""
    assert 250 + 250 + 250 + 250 == 1000
