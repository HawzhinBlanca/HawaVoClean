"""Regression tests for the self-contained Resolve engine launcher."""

from __future__ import annotations

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
        [sys.executable, "-I", str(tmp_path / "launcher.py"), str(marker)],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "spawned once\n"
    assert "bootstrapping phase" not in result.stderr


def test_shell_launcher_is_relocatable_and_isolated() -> None:
    source = build_resolve_engine.SHELL_LAUNCHER_SOURCE
    assert 'dirname -- "$0"' in source
    assert "PYTHONNOUSERSITE=1" in source
    assert 'python3.11" -I' in source
