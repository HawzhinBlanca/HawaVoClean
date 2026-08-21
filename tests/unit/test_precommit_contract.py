from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_precommit_uses_the_frozen_release_environment_and_nonmutating_commands() -> None:
    contract = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "repo: local" in contract
    assert contract.count("uv run --frozen") == 3
    assert "ruff check ." in contract and "ruff format --check ." in contract
    assert "mypy --strict src tests scripts data" in contract
    assert "--fix" not in contract
    assert "additional_dependencies" not in contract
