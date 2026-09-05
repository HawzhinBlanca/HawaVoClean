"""Batch command: isolation of failures, summary, exit codes, stem cleaning."""

import sys
from pathlib import Path
from typing import Any

import pytest

import hawavoclean.cli as cli
from hawavoclean.cli import _clean_stem
from hawavoclean.errors import ExitCode

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"


def _run_cli(monkeypatch: Any, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def test_clean_stem_strips_stacked_audio_suffixes() -> None:
    assert _clean_stem(Path("Flute 09.m4a.mp4")) == "Flute 09"
    assert _clean_stem(Path("take.wav")) == "take"
    assert _clean_stem(Path("notes.txt")) == "notes.txt"
    assert _clean_stem(Path(".wav")) == ".wav"  # no empty stems


def test_batch_processes_all_and_exits_zero(monkeypatch: Any, tmp_path: Path) -> None:
    rc = _run_cli(
        monkeypatch,
        "batch",
        str(FIX / "sample_sorani_podcast.wav"),
        str(FIX / "sample_noisy_hum.wav"),
        "-o",
        str(tmp_path),
        "--overwrite",
    )
    assert rc == 0
    assert (tmp_path / "sample_sorani_podcast_clean.wav").exists()
    assert (tmp_path / "sample_noisy_hum_clean.wav").exists()
    assert (tmp_path / "sample_noisy_hum_clean.hawavoclean.json").exists()


def test_batch_isolates_failures_and_exits_nonzero(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """One bad input must not stop the good one, and must fail the exit code."""
    rc = _run_cli(
        monkeypatch,
        "batch",
        str(FIX / "sample_ambiguous_stereo.wav"),  # raises AmbiguousStereoError
        str(FIX / "sample_sorani_podcast.wav"),
        "-o",
        str(tmp_path),
        "--overwrite",
    )
    assert rc == int(ExitCode.PUBLICATION_FAILURE)
    out = capsys.readouterr().out
    assert "1/2 succeeded" in out
    assert "FAILED" in out and "ambiguous" in out.lower()
    assert (tmp_path / "sample_sorani_podcast_clean.wav").exists()


def test_batch_skip_existing(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    src = FIX / "sample_sorani_podcast.wav"
    assert _run_cli(monkeypatch, "batch", str(src), "-o", str(tmp_path), "--overwrite") == 0
    rc = _run_cli(monkeypatch, "batch", str(src), "-o", str(tmp_path), "--skip-existing")
    assert rc == 0
    assert "SKIP" in capsys.readouterr().out


def test_batch_no_valid_inputs(monkeypatch: Any, tmp_path: Path) -> None:
    rc = _run_cli(monkeypatch, "batch", str(tmp_path / "ghost.wav"), "-o", str(tmp_path))
    assert rc == int(ExitCode.INVALID_USER_INPUT)


def test_batch_hung_file_is_killed_and_batch_continues(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    """A file that hangs must be killed at the deadline; the NEXT file must
    still be processed and the exit code must be non-zero."""
    import hawavoclean.cli as c

    real = c._run_one_isolated
    calls = {"n": 0}

    # Mirrors _run_one_isolated's real signature, so a future parameter added
    # there fails this test loudly instead of being silently swallowed.
    def fake(
        src: Path,
        dest: Path,
        profile: str,
        overwrite: bool,
        timeout_s: float,
        **kwargs: Any,
    ) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate the child being killed at its deadline
            return f"FAILED: timed out after {timeout_s:.0f}s (killed; batch continued)"
        return real(src, dest, profile, overwrite, timeout_s, **kwargs)

    monkeypatch.setattr(c, "_run_one_isolated", fake)
    rc = _run_cli(
        monkeypatch,
        "batch",
        str(FIX / "sample_noisy_hum.wav"),
        str(FIX / "sample_sorani_podcast.wav"),
        "-o",
        str(tmp_path),
        "--overwrite",
        # The first file's hang is simulated above, so this deadline never gates
        # it. It only bounds the SECOND file, which really is decoded, enhanced
        # and published by a cold child process — five seconds is a warm
        # workstation's timing, and a CI runner misses it, turning "the batch
        # continued" into a spurious failure. The genuine deadline behaviour is
        # covered by test_batch_real_deadline_kills_a_genuinely_hung_child.
        "--per-file-timeout-s",
        "600",
    )
    out = capsys.readouterr().out
    assert rc == int(ExitCode.PUBLICATION_FAILURE)
    assert "timed out" in out
    assert "1/2 succeeded" in out
    assert (tmp_path / "sample_sorani_podcast_clean.wav").exists(), (
        "batch did not continue past the hang"
    )


def test_batch_real_deadline_kills_a_genuinely_hung_child(tmp_path: Path) -> None:
    """Real subprocess, real deadline: point the child at a FIFO so decode
    blocks forever, with a 5 s deadline. Must return within ~5 s."""
    import os
    import time

    import hawavoclean.cli as c

    if not hasattr(os, "mkfifo"):
        pytest.skip("Named pipes (mkfifo) unsupported on Windows")
    fifo = tmp_path / "never.wav"
    os.mkfifo(fifo)  # reading it blocks until a writer appears (never)
    t0 = time.perf_counter()
    status = c._run_one_isolated(fifo, tmp_path / "o.wav", "production", True, 5.0)
    elapsed = time.perf_counter() - t0
    assert "timed out" in status or "FAILED" in status, status
    assert elapsed < 30.0, f"deadline not enforced: {elapsed:.1f}s"


def test_batch_keeps_one_warm_child_across_files(monkeypatch: Any, tmp_path: Path) -> None:
    """The per-file child existed for isolation, not to reload the model.

    Three files, one child: the interpreter and the enhancement core are
    started once for the batch instead of once per file. That is the whole
    saving, so it is asserted directly rather than inferred from a clock.
    """
    import hawavoclean.cli as c

    c._batch_child.close()
    before = c._batch_child.spawns
    srcs = []
    for i in range(3):
        p = tmp_path / f"take{i}.wav"
        p.write_bytes((FIX / "sample_sorani_podcast.wav").read_bytes())
        srcs.append(str(p))
    try:
        rc = _run_cli(monkeypatch, "batch", *srcs, "-o", str(tmp_path / "out"), "--overwrite")
    finally:
        c._batch_child.close()
    assert rc == 0
    assert c._batch_child.spawns - before == 1, (
        f"batch spawned {c._batch_child.spawns - before} children for 3 files; "
        "the warm child is not being reused"
    )
    for i in range(3):
        assert (tmp_path / "out" / f"take{i}_clean.wav").exists()


def test_a_deadline_breach_discards_the_child_and_the_next_file_gets_a_new_one(
    tmp_path: Path,
) -> None:
    """Isolation, with the warm child: a file that misses its deadline is
    killed with its whole process group, and the file after it publishes on a
    replacement child rather than inheriting a process nobody can reason about.

    The deadline is made unmeetable rather than the work made slow, so the
    breach is a property of the test and not of the machine's load.
    """
    import hawavoclean.cli as c

    c._batch_child.close()
    before = c._batch_child.spawns
    try:
        missed = c._run_one_isolated(
            FIX / "sample_sorani_podcast.wav", tmp_path / "late.wav", "production", True, 0.001
        )
        assert "timed out" in missed, missed
        assert not (tmp_path / "late.wav").exists(), "a killed child still published"

        good = c._run_one_isolated(
            FIX / "sample_sorani_podcast.wav", tmp_path / "good.wav", "production", True, 300.0
        )
        assert good.startswith("ok"), good
        assert (tmp_path / "good.wav").exists()
        assert c._batch_child.spawns - before == 2, (
            "the child that missed its deadline was reused instead of replaced"
        )
    finally:
        c._batch_child.close()


def test_a_batched_file_publishes_exactly_what_a_standalone_run_publishes(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Reusing a warm pool across files must not change one byte of any of
    them — the same claim the pool makes across units, made across files."""
    import hashlib

    import hawavoclean.cli as c
    from tests.support.wavbytes import masked_wav_bytes

    src = tmp_path / "take.wav"
    src.write_bytes((FIX / "sample_noisy_hum.wav").read_bytes())

    alone = tmp_path / "alone.wav"
    assert _run_cli(monkeypatch, "process", str(src), "-o", str(alone), "--overwrite") == 0

    batched_dir = tmp_path / "batched"
    c._batch_child.close()
    try:
        assert _run_cli(monkeypatch, "batch", str(src), "-o", str(batched_dir), "--overwrite") == 0
    finally:
        c._batch_child.close()

    def sha(p: Path) -> str:
        return hashlib.sha256(masked_wav_bytes(p.read_bytes())).hexdigest()

    assert sha(batched_dir / "take_clean.wav") == sha(alone), (
        "the batch's warm pool changed the published master"
    )


def test_noise_on_the_protocol_pipe_does_not_fail_a_file() -> None:
    """fd 1 is inherited all the way down to the enhancement worker processes.

    A library banner or a stray ``print`` three processes below would arrive on
    the same pipe the batch protocol uses. The child pushes all of that onto
    stderr, and the parent additionally refuses to mistake a non-JSON line for
    an answer — belt and braces, because the failure mode is "a healthy file is
    reported as failed".
    """
    import hawavoclean.cli as c

    child = c._BatchChild()
    child._lines.put("Intel MKL WARNING: something entirely unrelated\n")
    child._lines.put("\n")
    child._lines.put('[{"not": "an object"}]\n')
    child._lines.put('{"ok": true}\n')
    reply = child._await_reply(__import__("time").monotonic() + 5.0)
    assert reply == {"ok": True}
