"""Committed-generation publication is crash-safe, attributable, and recoverable."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import hawavoclean.publication as publication
from hawavoclean.errors import PublicationError
from hawavoclean.publication import (
    public_output_path,
    publication_exists,
    publication_paths,
    publish_output_generation,
    resolve_committed_publication,
)


def _report(audio: bytes) -> str:
    return json.dumps({"output": {"sha256": hashlib.sha256(audio).hexdigest()}})


def _candidate(root: Path, audio: bytes) -> Path:
    path = root / "candidate.wav"
    path.write_bytes(audio)
    return path


def _publish(root: Path, audio: bytes, *, overwrite: bool = False) -> tuple[Path, Path, Path]:
    return publish_output_generation(
        _candidate(root, audio),
        root / "out.wav",
        _report(audio),
        f"summary:{audio.decode()}",
        overwrite=overwrite,
    )


def _current(root: Path) -> str:
    return os.readlink(publication_paths(root / "out.wav").current).split("/", 1)[1]


def _assert_complete_generation(root: Path) -> bytes:
    resolved = resolve_committed_publication(root / "out.wav")
    assert resolved is not None
    audio = resolved[0].read_bytes()
    assert (
        json.loads(resolved[1].read_text())["output"]["sha256"] == hashlib.sha256(audio).hexdigest()
    )
    assert resolved[2].read_text() == f"summary:{audio.decode()}"
    return audio


def test_first_publish_commits_three_public_aliases_through_one_pointer(tmp_path: Path) -> None:
    audio, report, summary = _publish(tmp_path, b"generation-one")
    paths = publication_paths(audio)

    assert all(path.is_symlink() for path in (audio, report, summary))
    assert audio.read_bytes() == b"generation-one"
    assert (
        json.loads(report.read_text())["output"]["sha256"]
        == hashlib.sha256(b"generation-one").hexdigest()
    )
    assert summary.read_text() == "summary:generation-one"
    generation = paths.generations / _current(tmp_path)
    assert generation.is_dir()
    assert json.loads(paths.transaction.read_text())["phase"] == "committed"

    for role, public in (("master.wav", audio), ("report.json", report), ("summary.txt", summary)):
        assert os.readlink(public) == f"{paths.bundle.name}/current/{role}"


def test_public_output_path_resolves_parent_but_not_final_alias(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    os.symlink(real_parent.name, linked_parent)
    final_target = real_parent / "target.wav"
    final_target.write_bytes(b"target")
    os.symlink(final_target.name, real_parent / "out.wav")

    assert public_output_path(linked_parent / "out.wav") == real_parent / "out.wav"


def test_overwrite_retains_prior_immutable_generation(tmp_path: Path) -> None:
    _publish(tmp_path, b"old")
    old = resolve_committed_publication(tmp_path / "out.wav")
    assert old is not None
    old_id = _current(tmp_path)

    _publish(tmp_path, b"new", overwrite=True)
    new = resolve_committed_publication(tmp_path / "out.wav")
    assert new is not None
    assert _current(tmp_path) != old_id
    assert old[0].read_bytes() == b"old"
    assert json.loads(old[1].read_text())["output"]["sha256"] == hashlib.sha256(b"old").hexdigest()
    assert new[0].read_bytes() == b"new"


def test_no_overwrite_refusal_does_not_migrate_or_modify_legacy_files(tmp_path: Path) -> None:
    paths = publication_paths(tmp_path / "out.wav")
    old = b"legacy"
    paths.audio.write_bytes(old)
    paths.json.write_text(_report(old))
    paths.txt.write_text("legacy summary")
    before = tuple(path.read_bytes() for path in paths.public)

    with pytest.raises(PublicationError, match="overwrite=False"):
        _publish(tmp_path, b"new", overwrite=False)
    assert tuple(path.read_bytes() for path in paths.public) == before
    assert not paths.bundle.exists()


def test_failure_before_pointer_keeps_old_generation_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish(tmp_path, b"old")
    old_id = _current(tmp_path)
    old_triplet = tuple(
        path.read_bytes() for path in publication_paths(tmp_path / "out.wav").public
    )

    def fail(name: str) -> None:
        if name == "before_pointer_commit":
            raise OSError(28, "disk full")

    monkeypatch.setattr(publication, "_checkpoint", fail)
    with pytest.raises(PublicationError, match="disk full"):
        _publish(tmp_path, b"new", overwrite=True)

    assert _current(tmp_path) == old_id
    assert (
        tuple(path.read_bytes() for path in publication_paths(tmp_path / "out.wav").public)
        == old_triplet
    )


def test_failure_after_pointer_finishes_forward_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish(tmp_path, b"old")

    def fail(name: str) -> None:
        if name == "pointer_replaced":
            raise OSError(5, "post-commit fault")

    monkeypatch.setattr(publication, "_checkpoint", fail)
    audio, report, summary = _publish(tmp_path, b"new", overwrite=True)
    assert audio.read_bytes() == b"new"
    assert json.loads(report.read_text())["output"]["sha256"] == hashlib.sha256(b"new").hexdigest()
    assert summary.read_text() == "summary:new"
    transaction = json.loads(publication_paths(audio).transaction.read_text())
    assert transaction["phase"] == "committed"
    assert transaction["recovered_after"] == "OSError"


def test_failed_postcommit_recovery_is_never_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish(tmp_path, b"old")
    real_replace_json = publication._replace_json

    def fail_committed_journal(path: Path, value: object) -> None:
        assert isinstance(value, dict)
        if value.get("phase") == "committed":
            raise OSError(28, "journal disk full")
        real_replace_json(path, value)

    monkeypatch.setattr(publication, "_replace_json", fail_committed_journal)
    with pytest.raises(PublicationError, match="journal disk full"):
        _publish(tmp_path, b"new", overwrite=True)

    # The atomic pointer may already expose the complete new generation, but
    # the caller gets no success verdict until journal recovery and verification
    # have both completed.
    assert (tmp_path / "out.wav").read_bytes() == b"new"


@pytest.mark.parametrize(
    "checkpoint",
    ["generation_files_durable", "generation_committed", "before_pointer_commit"],
)
def test_every_precommit_durable_state_preserves_old_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    _publish(tmp_path, b"old")
    old_id = _current(tmp_path)

    def fail(name: str) -> None:
        if name == checkpoint:
            raise OSError(5, f"fault at {checkpoint}")

    monkeypatch.setattr(publication, "_checkpoint", fail)
    with pytest.raises(PublicationError, match=checkpoint):
        _publish(tmp_path, b"new", overwrite=True)
    assert _current(tmp_path) == old_id
    assert (tmp_path / "out.wav").read_bytes() == b"old"


@pytest.mark.parametrize("checkpoint", ["pointer_replaced", "pointer_durable"])
def test_every_postcommit_state_recovers_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    _publish(tmp_path, b"old")

    def fail(name: str) -> None:
        if name == checkpoint:
            raise OSError(5, f"fault at {checkpoint}")

    monkeypatch.setattr(publication, "_checkpoint", fail)
    _publish(tmp_path, b"new", overwrite=True)
    assert (tmp_path / "out.wav").read_bytes() == b"new"


def test_first_publish_failure_exposes_no_resolvable_partial_triplet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(name: str) -> None:
        if name == "alias_json_replaced":
            raise OSError(28, "disk full")

    monkeypatch.setattr(publication, "_checkpoint", fail)
    with pytest.raises(PublicationError, match="disk full"):
        _publish(tmp_path, b"new")
    paths = publication_paths(tmp_path / "out.wav")
    assert not publication_exists(paths.audio)
    assert all(not os.path.lexists(path) for path in paths.public)


def test_legacy_triplet_migrates_without_destroying_prior_bytes(tmp_path: Path) -> None:
    old = b"legacy"
    paths = publication_paths(tmp_path / "out.wav")
    paths.audio.write_bytes(old)
    paths.json.write_text(_report(old))
    paths.txt.write_text("legacy summary")

    _publish(tmp_path, b"new", overwrite=True)
    generations = list(paths.generations.iterdir())
    assert len(generations) == 2
    assert any((generation / "master.wav").read_bytes() == old for generation in generations)
    assert paths.audio.is_symlink() and paths.audio.read_bytes() == b"new"


def test_interrupted_legacy_alias_migration_remains_old_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = b"legacy"
    paths = publication_paths(tmp_path / "out.wav")
    paths.audio.write_bytes(old)
    paths.json.write_text(_report(old))
    paths.txt.write_text("legacy summary")

    def fail(name: str) -> None:
        if name == "alias_json_replaced":
            raise OSError(5, "migration interrupted")

    monkeypatch.setattr(publication, "_checkpoint", fail)
    with pytest.raises(PublicationError, match="migration interrupted"):
        _publish(tmp_path, b"new", overwrite=True)

    assert tuple(path.read_bytes() for path in paths.public) == (
        old,
        _report(old).encode(),
        b"legacy summary",
    )
    monkeypatch.setattr(publication, "_checkpoint", lambda _name: None)
    _publish(tmp_path, b"new", overwrite=True)
    assert tuple(path.is_symlink() for path in paths.public) == (True, True, True)
    assert paths.audio.read_bytes() == b"new"


def test_incomplete_legacy_triplet_is_never_overwritten(tmp_path: Path) -> None:
    paths = publication_paths(tmp_path / "out.wav")
    paths.audio.write_bytes(b"irreplaceable")
    with pytest.raises(PublicationError, match="Incomplete legacy output triplet"):
        _publish(tmp_path, b"new", overwrite=True)
    assert paths.audio.read_bytes() == b"irreplaceable"


def test_report_for_different_audio_is_rejected_before_commit(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, b"audio")
    with pytest.raises(PublicationError, match="different audio"):
        publish_output_generation(
            candidate,
            tmp_path / "out.wav",
            _report(b"other"),
            "summary",
        )
    assert not publication_exists(tmp_path / "out.wav")


def test_unexpected_public_symlink_fails_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not touch")
    os.symlink(outside.name, tmp_path / "out.wav")
    with pytest.raises(PublicationError, match="Unexpected dangling output alias"):
        _publish(tmp_path, b"new", overwrite=True)
    assert outside.read_bytes() == b"do not touch"
    assert os.readlink(tmp_path / "out.wav") == outside.name


def test_symlinked_lock_file_is_rejected_without_touching_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside-lock-target"
    outside.write_bytes(b"unchanged")
    paths = publication_paths(tmp_path / "out.wav")
    os.symlink(outside.name, paths.lock)
    with pytest.raises(PublicationError, match="safe publication lock"):
        _publish(tmp_path, b"new")
    assert outside.read_bytes() == b"unchanged"


def test_nonregular_publication_lock_is_rejected(tmp_path: Path) -> None:
    paths = publication_paths(tmp_path / "out.wav")
    os.mkfifo(paths.lock)
    with pytest.raises(PublicationError, match="not a regular file"):
        _publish(tmp_path, b"new")


@pytest.mark.parametrize(
    "damage", ["bundle-symlink", "bad-owner", "wrong-owner", "unsafe-generations"]
)
def test_unverifiable_bundle_ownership_fails_closed(tmp_path: Path, damage: str) -> None:
    paths = publication_paths(tmp_path / "out.wav")
    if damage == "bundle-symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside.name, paths.bundle)
    else:
        paths.bundle.mkdir()
        if damage == "bad-owner":
            (paths.bundle / publication._OWNER_FILE).write_text("{")
            paths.generations.mkdir()
        elif damage == "wrong-owner":
            (paths.bundle / publication._OWNER_FILE).write_text(
                json.dumps({"schema_version": 1, "public_names": {"audio": "other.wav"}})
            )
            paths.generations.mkdir()
        else:
            (paths.bundle / publication._OWNER_FILE).write_text(
                json.dumps(publication._owner_payload(paths))
            )
            outside = tmp_path / "outside-generations"
            outside.mkdir()
            os.symlink(outside.name, paths.generations)

    with pytest.raises(PublicationError):
        _publish(tmp_path, b"new")


def test_invalid_report_json_is_rejected_before_commit(tmp_path: Path) -> None:
    with pytest.raises(PublicationError, match="JSON report is invalid"):
        publish_output_generation(
            _candidate(tmp_path, b"audio"),
            tmp_path / "out.wav",
            "{",
            "summary",
        )


@pytest.mark.parametrize("pointer", ["regular", "absolute", "bad-generation"])
def test_invalid_current_pointer_is_rejected(tmp_path: Path, pointer: str) -> None:
    _publish(tmp_path, b"old")
    paths = publication_paths(tmp_path / "out.wav")
    paths.current.unlink()
    if pointer == "regular":
        paths.current.write_text("not a symlink")
    elif pointer == "absolute":
        os.symlink(str(paths.generations / _current.__name__), paths.current)
    else:
        os.symlink("generations/not-a-digest", paths.current)

    with pytest.raises(PublicationError, match="current pointer"):
        resolve_committed_publication(paths.audio)


def test_tampered_committed_generation_is_detected(tmp_path: Path) -> None:
    _publish(tmp_path, b"old")
    paths = publication_paths(tmp_path / "out.wav")
    generation = paths.generations / _current(tmp_path)
    (generation / "report.json").write_text(_report(b"tampered"))
    with pytest.raises(PublicationError, match="digest mismatch"):
        resolve_committed_publication(paths.audio)


def test_identical_republish_reuses_content_addressed_generation(tmp_path: Path) -> None:
    _publish(tmp_path, b"same")
    paths = publication_paths(tmp_path / "out.wav")
    first = _current(tmp_path)
    _publish(tmp_path, b"same", overwrite=True)
    assert _current(tmp_path) == first
    assert [path.name for path in paths.generations.iterdir()] == [first]


def test_concurrent_publishers_serialize_to_one_complete_generation(tmp_path: Path) -> None:
    _publish(tmp_path, b"old")
    candidates: list[tuple[Path, bytes]] = []
    for index, value in enumerate((b"new-a", b"new-b")):
        candidate = tmp_path / f"candidate-{index}.wav"
        candidate.write_bytes(value)
        candidates.append((candidate, value))

    def publish(item: tuple[Path, bytes]) -> None:
        candidate, value = item
        publish_output_generation(
            candidate,
            tmp_path / "out.wav",
            _report(value),
            f"summary:{value.decode()}",
            overwrite=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, item) for item in candidates]
        for future in futures:
            future.result(timeout=10)

    resolved = resolve_committed_publication(tmp_path / "out.wav")
    assert resolved is not None
    audio = resolved[0].read_bytes()
    assert audio in {b"new-a", b"new-b"}
    assert (
        json.loads(resolved[1].read_text())["output"]["sha256"] == hashlib.sha256(audio).hexdigest()
    )


@pytest.mark.parametrize(
    ("label", "target", "attribute"),
    [
        ("copy", publication, "_copy_fsync"),
        ("write", publication, "_write_bytes_fsync"),
        ("directory-fsync", publication, "_fsync_directory"),
        ("rename", os, "rename"),
        ("replace", os, "replace"),
        ("symlink", os, "symlink"),
        ("staging-cleanup", shutil, "rmtree"),
    ],
)
def test_every_publication_primitive_failure_keeps_one_complete_generation(
    tmp_path: Path,
    label: str,
    target: object,
    attribute: str,
) -> None:
    """Fault every state-changing primitive call immediately before and after it.

    The discovery run makes the matrix follow the implementation: adding a new
    write, flush, rename, alias or cleanup call automatically adds two fault
    positions instead of silently escaping the test.
    """
    real = getattr(target, attribute)
    probe = tmp_path / f"probe-{label}"
    probe.mkdir()
    _publish(probe, b"old")
    observed = 0

    def count_calls(*args: object, **kwargs: object) -> object:
        nonlocal observed
        observed += 1
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(target, attribute, count_calls)
        _publish(probe, b"new", overwrite=True)
    assert observed > 0, f"matrix primitive {label} was never exercised"

    for call_index in range(1, observed + 1):
        for side in ("before", "after"):
            root = tmp_path / f"{label}-{call_index}-{side}"
            root.mkdir()
            _publish(root, b"old")
            calls = 0

            def inject(
                *args: object,
                _selected: int = call_index,
                _side: str = side,
                **kwargs: object,
            ) -> object:
                nonlocal calls
                calls += 1
                if calls == _selected and _side == "before":
                    raise OSError(5, f"{label}-{_selected}-{_side}")
                result = real(*args, **kwargs)
                if calls == _selected and _side == "after":
                    raise OSError(5, f"{label}-{_selected}-{_side}")
                return result

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(target, attribute, inject)
                with contextlib.suppress(PublicationError):
                    _publish(root, b"new", overwrite=True)
            assert _assert_complete_generation(root) in {b"old", b"new"}
            _publish(root, b"new", overwrite=True)
            assert _assert_complete_generation(root) == b"new"


def test_partial_artifact_write_and_permission_loss_preserve_prior_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _publish(tmp_path, b"old")
    real_write = publication._write_bytes_fsync

    def partial_then_full(path: Path, data: bytes) -> None:
        if path.name == "report.json":
            with open(path, "xb") as stream:
                stream.write(data[: max(1, len(data) // 2)])
                stream.flush()
                os.fsync(stream.fileno())
            raise PermissionError(13, "permission removed during report write")
        real_write(path, data)

    monkeypatch.setattr(publication, "_write_bytes_fsync", partial_then_full)
    with pytest.raises(PublicationError, match="permission removed"):
        _publish(tmp_path, b"new", overwrite=True)
    assert _assert_complete_generation(tmp_path) == b"old"


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM", "SIGKILL"])
@pytest.mark.parametrize(
    "checkpoint",
    [
        "generation_files_durable",
        "generation_committed",
        "alias_audio_replaced",
        "alias_json_replaced",
        "alias_txt_replaced",
        "before_pointer_commit",
        "pointer_replaced",
        "pointer_durable",
    ],
)
def test_real_signals_at_every_durable_state_recover_idempotently(
    tmp_path: Path, checkpoint: str, signal_name: str
) -> None:
    root = tmp_path / f"{checkpoint}-{signal_name}"
    root.mkdir()
    first_publish_state = checkpoint.startswith("alias_")
    if not first_publish_state:
        _publish(root, b"old")
    candidate = _candidate(root, b"new")
    report = _report(b"new")
    script = """
import os, signal, sys
from pathlib import Path
import hawavoclean.publication as publication

def checkpoint(name: str) -> None:
    if name == sys.argv[4]:
        os.kill(os.getpid(), getattr(signal, sys.argv[5]))

publication._checkpoint = checkpoint
publication.publish_output_generation(
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], "summary:new", overwrite=sys.argv[6] == "1"
)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(candidate),
            str(root / "out.wav"),
            report,
            checkpoint,
            signal_name,
            "0" if first_publish_state else "1",
        ],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0 or checkpoint in {"pointer_replaced", "pointer_durable"}

    paths = publication_paths(root / "out.wav")
    if first_publish_state and not os.path.lexists(paths.current):
        assert not publication_exists(paths.audio)
        _publish(root, b"new")
    else:
        assert _assert_complete_generation(root) in {b"old", b"new"}
        _publish(root, b"new", overwrite=True)
    assert _assert_complete_generation(root) == b"new"


def test_sigkill_after_pointer_swap_recovers_committed_generation(tmp_path: Path) -> None:
    _publish(tmp_path, b"old")
    candidate = _candidate(tmp_path, b"after-kill")
    report = _report(b"after-kill")
    script = """
import os, signal, sys
from pathlib import Path
import hawavoclean.publication as publication

def checkpoint(name: str) -> None:
    if name == "pointer_replaced":
        os.kill(os.getpid(), signal.SIGKILL)

publication._checkpoint = checkpoint
publication.publish_output_generation(
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], "summary:after-kill", overwrite=True
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(candidate), str(tmp_path / "out.wav"), report],
        check=False,
    )
    assert result.returncode == -signal.SIGKILL

    # Re-running the same publish verifies/reuses the immutable generation,
    # repairs journal/aliases, and cannot delete the old generation.
    _publish(tmp_path, b"after-kill", overwrite=True)
    resolved = resolve_committed_publication(tmp_path / "out.wav")
    assert resolved is not None
    assert resolved[0].read_bytes() == b"after-kill"
    assert (
        json.loads(resolved[1].read_text())["output"]["sha256"]
        == hashlib.sha256(b"after-kill").hexdigest()
    )


def test_sigkill_before_first_pointer_allows_no_overwrite_retry(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, b"after-kill")
    report = _report(b"after-kill")
    script = """
import os, signal, sys
from pathlib import Path
import hawavoclean.publication as publication

def checkpoint(name: str) -> None:
    if name == "alias_json_replaced":
        os.kill(os.getpid(), signal.SIGKILL)

publication._checkpoint = checkpoint
publication.publish_output_generation(
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], "summary:after-kill"
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(candidate), str(tmp_path / "out.wav"), report],
        check=False,
    )
    assert result.returncode == -signal.SIGKILL
    paths = publication_paths(tmp_path / "out.wav")
    assert not os.path.lexists(paths.current)
    assert not publication_exists(paths.audio)

    # No --overwrite opt-in is needed because the killed process never made a
    # generation authoritative. Recovery removes its owned dangling aliases.
    audio, _, _ = _publish(tmp_path, b"after-kill")
    assert audio.read_bytes() == b"after-kill"


def test_resolved_reader_cannot_mix_generations_across_overwrite(tmp_path: Path) -> None:
    _publish(tmp_path, b"old")
    reader = resolve_committed_publication(tmp_path / "out.wav")
    assert reader is not None
    _publish(tmp_path, b"new", overwrite=True)

    assert reader[0].read_bytes() == b"old"
    assert (
        json.loads(reader[1].read_text())["output"]["sha256"] == hashlib.sha256(b"old").hexdigest()
    )
    current = resolve_committed_publication(tmp_path / "out.wav")
    assert current is not None and current[0].read_bytes() == b"new"
