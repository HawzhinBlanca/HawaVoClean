"""Test isolation: no test may read or leave state outside its own sandbox.

The suite once passed with the entire enhance/guard/finish loop skipped,
because a stale repo-level workspace cache was silently serving results.
This conftest makes that class of failure loud and impossible:

- A pre-existing repo-level ``.voiceclean-work`` FAILS the session outright
  (silent repair would hide exactly the bug class this guards against).
- Every test gets a fresh ``VOICECLEAN_WORK_DIR`` under its own tmp dir.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart() -> None:
    legacy = REPO_ROOT / ".voiceclean-work"
    if legacy.exists():
        raise pytest.UsageError(
            f"Pre-existing workspace state found at {legacy}. Delete it before "
            "running the suite: stale workspaces have previously caused the "
            "suite to pass without executing the code under test."
        )


@pytest.fixture(autouse=True)
def _isolated_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICECLEAN_WORK_DIR", str(tmp_path / "vc-work"))


@pytest.fixture(autouse=True)
def _stable_cwd() -> object:
    """Restore the CWD after each test so chdir-based tests cannot leak."""
    prev = os.getcwd()
    yield
    os.chdir(prev)
