"""T3.3 proof: a deliberately failing test on a disposable branch.

This file exists to be rejected. T3.3 requires showing that a pull request
carrying a failing test is *mechanically* prevented from merging into a
protected `main` — not that a reviewer would decline it, and not that the check
is configured as required, but that the merge is actually closed by it.

It fails inside the default suite, which is step 10 of the release gate's first
pass, so `exact-release-gate` fails early rather than after the full
two-checkout contract, and `required` then fails on
`test "$EXACT_RELEASE_GATE" = success`.

`pytest.fail` rather than `assert False`: this repository selects flake8-bugbear,
whose B011 rejects the latter, and a proof that stops at the linter is a proof
about the linter rather than about a failing test.

This branch is never merged. It is opened, observed, recorded, and closed.
"""

from __future__ import annotations

import pytest


def test_t33_deliberate_failure_must_block_the_merge() -> None:
    """Fail on purpose. See the module docstring."""
    pytest.fail("T3.3 proof: this failure must make the pull request unmergeable")
