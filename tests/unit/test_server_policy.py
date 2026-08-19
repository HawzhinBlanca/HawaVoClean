"""Path policy: absolute, under home / /Volumes / work dir, symlinks resolved."""

import os
from pathlib import Path

import pytest

from hawavoclean.server.policy import PathPolicyError, allowed_roots, resolve_client_path

pytestmark = pytest.mark.unit


def test_allowed_roots_are_home_volumes_and_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "w"))
    roots = allowed_roots()
    assert Path.home().resolve() in roots
    assert Path("/Volumes") in roots
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
    for bad in ("/etc/passwd", "/", "/usr/bin/python3", str(tmp_path / "outside.wav")):
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


def test_dotdot_and_symlink_escapes_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "w"))
    (tmp_path / "w").mkdir()
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path(str(tmp_path / "w" / ".." / ".." / ".." / "etc" / "passwd"))
    assert exc.value.status == 403
    link = tmp_path / "w" / "escape"
    os.symlink("/etc", link)
    with pytest.raises(PathPolicyError) as exc:
        resolve_client_path(str(link / "passwd"))
    assert exc.value.status == 403
