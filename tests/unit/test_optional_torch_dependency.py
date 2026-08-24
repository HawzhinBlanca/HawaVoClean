"""The CLI must come up on a natural-mode-only install.

Torch is an optional extra. It was reachable from ``import hawavoclean.cli``
through multipass -> pipeline -> the restoration package, which eagerly
re-exported two ``torch.nn.Module`` subclasses -- so the published wheel could
not print its own version without the restore extra installed. CI caught it
only in the wheel smoke step, which runs after the suite and had been masked
for several commits by earlier failures. These tests move the guarantee into
the suite, where it fails in half a second instead of after a full CI matrix.
"""

import importlib
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit


class _NoTorch(MetaPathFinder):
    """Make torch unimportable, the way a base wheel install has it."""

    def find_spec(
        self,
        name: str,
        path: Any = None,  # noqa: ARG002 - signature fixed by the finder protocol
        target: Any = None,  # noqa: ARG002 - signature fixed by the finder protocol
    ) -> ModuleSpec | None:
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


#: Modules that pull torch in at class-definition time. They have to leave
#: ``sys.modules`` as well: with them cached, ``import_module`` hands the
#: cached object back and the lazy guard never runs -- which passed in
#: isolation and failed only in the full suite, where an earlier test had
#: already imported them.
_TORCH_BACKED_MODULES = (
    "hawavoclean.restoration.hawarestore_kd",
    "hawavoclean.restoration.universr_upstream",
)


@pytest.fixture
def without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    doomed = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    doomed += [m for m in _TORCH_BACKED_MODULES if m in sys.modules]
    for module in doomed:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(sys, "meta_path", [_NoTorch(), *sys.meta_path])


def _reload(name: str) -> ModuleType:
    return importlib.reload(importlib.import_module(name))


@pytest.mark.usefixtures("without_torch")
def test_the_cli_imports_without_torch() -> None:
    """The entry point the wheel installs must not need the restore extra."""
    for name in ("hawavoclean.cli", "hawavoclean.pipeline", "hawavoclean.multipass"):
        assert _reload(name) is not None


@pytest.mark.usefixtures("without_torch")
def test_natural_mode_restoration_helpers_import_without_torch() -> None:
    """Bandwidth detection, the guard and profiles are numpy/scipy: still reachable."""
    restoration = _reload("hawavoclean.restoration")
    for name in ("BandwidthDetector", "RestorationGuard", "load_speaker_profile"):
        assert getattr(restoration, name) is not None


@pytest.mark.usefixtures("without_torch")
def test_the_torch_backed_restorer_says_what_is_missing() -> None:
    """Asking for the model without the extra must name the extra, not the traceback."""
    restoration = _reload("hawavoclean.restoration")
    with pytest.raises(ModuleNotFoundError, match=r"hawavoclean\[restore\]"):
        _ = restoration.HawaRestoreKD
