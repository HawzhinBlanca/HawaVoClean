"""Path policy: absolute, under home / /Volumes / work dir, symlinks resolved."""

import os
from pathlib import Path

import pytest

from hawavoclean.server.policy import (
    PathPolicyError,
    allowed_roots,
    resolve_client_output_path,
    resolve_client_path,
)

pytestmark = pytest.mark.unit


def test_allowed_roots_are_home_volumes_and_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "w"))
    roots = allowed_roots()
    assert Path.home().resolve() in roots
    assert Path("/Volumes").resolve() in roots
    assert (tmp_path / "w").resolve() in roots
    assert len(roots) == len(set(roots))


def test_relative_and_empty_paths_are_400() -> None:
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path("relative/file.wav")
    assert exc.value.status == 400 and exc.value.code == "bad_request"
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path("   ")
    assert exc.value.status == 400


def test_outside_roots_is_403(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "w"))
    anchor = Path.cwd().anchor
    for bad in (
        str(Path(anchor) / "etc" / "passwd"),
        anchor,
        str(Path(anchor) / "usr" / "bin" / "python3"),
        str(tmp_path / "outside.wav"),
    ):
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(bad)
        assert exc.value.status == 403 and exc.value.code == "forbidden", bad


def test_inside_work_dir_and_home_are_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert resolve_client_path(str(f)) == f.resolve()
    assert resolve_client_path(str(f), must_exist=True) == f.resolve()
    assert resolve_client_path(str(tmp_path)) == tmp_path.resolve()  # the root itself
    # A (non-existent) file under home passes the policy; existence is separate.
    assert resolve_client_path(str(Path.home() / "definitely-not-there.wav")).is_absolute()
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path(str(tmp_path / "missing.wav"), must_exist=True)
    assert exc.value.status == 404 and exc.value.code == "not_found"
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path(str(tmp_path), must_exist=True)  # a directory is not a file
    assert exc.value.status == 404


def test_output_policy_preserves_final_symlink_name_but_resolves_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    hidden = tmp_path / ".out.wav.hawavoclean" / "current"
    hidden.mkdir(parents=True)
    target = hidden / "master.wav"
    target.write_bytes(b"old")
    public = tmp_path / "out.wav"
    public.symlink_to(target.relative_to(tmp_path))

    assert resolve_client_path(str(public), must_exist=True) == target
    assert resolve_client_output_path(str(public)) == public


def test_text_that_cannot_be_a_filename_is_400_not_a_raised_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NUL and unpaired surrogates make ``lstat`` raise ``ValueError`` from
    inside ``Path.resolve()``. The policy documents 400/403/404 as its only
    answers, so both must be refusals — not a 500 with a Python exception."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    for bad, needle in (
        (f"{tmp_path}/a\x00b.wav", "NUL byte"),
        (f"{tmp_path}/trailing.wav\x00", "NUL byte"),
        ("\x00", "NUL byte"),
        (f"{tmp_path}/a\ud800b.wav", "usable filename"),  # JSON "\ud800" decodes to this
        (f"{tmp_path}/a\udfffb.wav", "usable filename"),
    ):
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(bad)
        assert exc.value.status == 400 and exc.value.code == "bad_request", bad
        assert needle in exc.value.message, exc.value.message
        # must_exist takes the same route, and never reaches the stat either
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(bad, must_exist=True)
        assert exc.value.status == 400, bad


def test_other_control_characters_are_legal_in_a_posix_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only NUL is impossible. Newline, CR, tab, ESC, DEL and non-UTF-8 bytes
    are all things a real file can be called, so they get the ordinary
    answers — a resolved path, or 404 — never a refusal of their own."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    for name in ("a\nb.wav", "a\rb.wav", "a\tb.wav", "a\x1bb.wav", "a\x7fb.wav", "a\udcffb.wav"):
        target = tmp_path / name
        resolved = resolve_client_path(str(target))
        assert resolved == target.resolve()
        with pytest.raises(PathPolicyError) as exc:
            resolve_client_path(str(target), must_exist=True)
        assert exc.value.status == 404, name
    if os.name != "nt":
        real = tmp_path / 'it\'s a "take"\n01.wav'
        real.write_bytes(b"RIFF")
        assert resolve_client_path(str(real), must_exist=True) == real.resolve()


def test_dotdot_and_symlink_escapes_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "isolated_home")
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "w"))
    (tmp_path / "w").mkdir()
    (tmp_path / "isolated_home").mkdir()
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path(str(tmp_path / "w" / ".." / "outside.wav"))
    assert exc.value.status == 403
    link = tmp_path / "w" / "escape"
    try:
        os.symlink(str(Path(Path.cwd().anchor) / "etc"), link)
    except OSError:
        pytest.skip("Symlink creation requires privilege on this filesystem")
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path(str(link / "passwd"))
    assert exc.value.status == 403
