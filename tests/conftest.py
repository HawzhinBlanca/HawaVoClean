"""Test isolation: no test may read or leave state outside its own sandbox.

The suite once passed with the entire enhance/guard/finish loop skipped,
because a stale repo-level workspace cache was silently serving results.
This conftest makes that class of failure loud and impossible:

- A pre-existing repo-level ``.hawavoclean-work`` FAILS the session outright
  (silent repair would hide exactly the bug class this guards against).
- Every test gets a fresh ``HAWAVOCLEAN_WORK_DIR`` under its own tmp dir.
"""

import contextlib
import os
from pathlib import Path

# Pre-import torch before coverage tracing to prevent Python 3.14 multi-load errors
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart() -> None:
    legacy = REPO_ROOT / ".hawavoclean-work"
    if legacy.exists():
        raise pytest.UsageError(
            f"Pre-existing workspace state found at {legacy}. Delete it before "
            "running the suite: stale workspaces have previously caused the "
            "suite to pass without executing the code under test."
        )


@pytest.fixture(autouse=True)
def _isolated_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path / "vc-work"))


@pytest.fixture(autouse=True)
def _stable_cwd() -> object:
    """Restore the CWD after each test so chdir-based tests cannot leak."""
    prev = os.getcwd()
    yield
    os.chdir(prev)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    """Cleanly teardown internal weakref handles and run garbage collection before module dict teardown."""
    import contextlib
    import gc
    import sys

    gc.collect()

    # Prevent torch StorageWeakRef noisy deallocator exceptions on Python 3.14 exit
    if "torch" in sys.modules:
        try:
            import torch.multiprocessing.reductions

            swr = getattr(torch.multiprocessing.reductions, "StorageWeakRef", None)
            if swr is not None and hasattr(swr, "__del__"):
                orig_del = swr.__del__

                def _safe_del(self: object) -> None:
                    with contextlib.suppress(Exception):
                        orig_del(self)

                swr.__del__ = _safe_del
        except Exception:
            pass

    gc.collect()
