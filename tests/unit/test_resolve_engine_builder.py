"""Regression tests for the self-contained Resolve engine launcher."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from scripts import build_resolve_engine


def test_launcher_does_not_reenter_cli_in_spawned_child(tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "hawavoclean"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        """import multiprocessing
import sys
from pathlib import Path


def _child(marker: str) -> None:
    Path(marker).write_text("spawned once\\n")


def main() -> int:
    marker = sys.argv[1]
    process = multiprocessing.get_context("spawn").Process(target=_child, args=(marker,))
    process.start()
    process.join(20)
    if process.exitcode != 0:
        raise RuntimeError(f"spawned child failed with {process.exitcode}")
    return 0
"""
    )
    build_resolve_engine._write_launchers(tmp_path)
    marker = tmp_path / "child-ran.txt"

    result = subprocess.run(
        [sys.executable, "-I", "-B", str(tmp_path / "launcher.py"), str(marker)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "spawned once\n"
    assert "bootstrapping phase" not in result.stderr
    assert not list((tmp_path / "site-packages").rglob("*.py[co]"))
    assert not list((tmp_path / "site-packages").rglob("__pycache__"))


def test_shell_launcher_is_relocatable_and_isolated() -> None:
    source = build_resolve_engine.SHELL_LAUNCHER_SOURCE
    assert 'dirname -- "$0"' in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert 'python3.11" -I -B' in source


def test_target_install_pruning_removes_build_paths_and_rewrites_record(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    package = site / "hawavoclean"
    dist_info = site / "hawavoclean-3.3.0.dist-info"
    scripts = site / "bin"
    base_site = tmp_path / "python" / "lib" / "python3.11" / "site-packages"
    package.mkdir(parents=True)
    dist_info.mkdir()
    scripts.mkdir()
    base_site.mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = '3.3.0'\n")
    (scripts / "hawavoclean").write_text("#!/private/tmp/build/python\n")
    (base_site / "pip.py").write_text("# build tool\n")
    (dist_info / "direct_url.json").write_text('{"url":"file:///private/tmp/wheel.whl"}')
    (dist_info / "uv_cache.json").write_text('{"timestamp":"nondeterministic"}')
    record = dist_info / "RECORD"
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(
            [
                ("bin/hawavoclean", "old", "1"),
                ("hawavoclean/__init__.py", "old", "1"),
                ("hawavoclean-3.3.0.dist-info/direct_url.json", "old", "1"),
                ("hawavoclean-3.3.0.dist-info/uv_cache.json", "old", "1"),
                ("hawavoclean-3.3.0.dist-info/RECORD", "", ""),
            ]
        )

    build_resolve_engine._prune_and_normalize_install(tmp_path, site)

    assert not scripts.exists()
    assert not base_site.exists()
    assert not (dist_info / "direct_url.json").exists()
    assert not (dist_info / "uv_cache.json").exists()
    rows = [tuple(row) for row in csv.reader(record.read_text().splitlines())]
    expected_hash = build_resolve_engine._record_digest(package / "__init__.py")
    assert rows == [
        ("hawavoclean-3.3.0.dist-info/RECORD", "", ""),
        (
            "hawavoclean/__init__.py",
            expected_hash,
            str((package / "__init__.py").stat().st_size),
        ),
    ]
    assert len(hashlib.sha256(record.read_bytes()).hexdigest()) == 64
