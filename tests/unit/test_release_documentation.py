"""Executable consistency checks for the current release documentation."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

from hawavoclean.multipass import MAX_PASSES
from hawavoclean.server.app import (
    DEFAULT_MAX_CONCURRENT_UPLOADS,
    DEFAULT_MAX_UPLOAD_BYTES,
)
from hawavoclean.server.jobs import (
    DEFAULT_MAX_ACTIVE_JOBS,
    DEFAULT_MAX_TERMINAL_JOBS,
    DEFAULT_TERMINAL_JOB_TTL_S,
)
from hawavoclean.server.retention import (
    DEFAULT_MAX_UPLOAD_TOTAL_BYTES,
    DEFAULT_MIN_FREE_BYTES,
    DEFAULT_UPLOAD_TTL_S,
)

ROOT = Path(__file__).resolve().parents[2]
CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "STATUS.md",
    ROOT / "RISKS.md",
    ROOT / "SECURITY.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "model-provenance.md",
    ROOT / "docs" / "operations.md",
)
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_support_runbook_matches_package_and_release_host_contracts() -> None:
    project = tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]
    toolchain = json.loads(_text(ROOT / "evidence" / "release" / "toolchain-lock.json"))
    operations = _text(ROOT / "docs" / "operations.md")

    assert project["requires-python"] == ">=3.11,<3.15"
    assert "CPython 3.11–3.14 on macOS and Linux" in operations
    assert "Windows" in operations and "Unsupported" in operations
    assert "CPU container" in operations and "No studio, CUDA, or" in operations
    assert toolchain["release_host"] == "macos-arm64"
    assert f"Resolve Studio {toolchain['resolve_host']['version']}" in operations
    assert f"`1`–`{MAX_PASSES}` or `auto`" in operations


def test_operational_limits_are_the_live_server_defaults() -> None:
    operations = _text(ROOT / "docs" / "operations.md")
    assert f"| Active jobs | {DEFAULT_MAX_ACTIVE_JOBS} |" in operations
    assert (
        f"| Retained terminal jobs | {DEFAULT_MAX_TERMINAL_JOBS} for at most 24 hours |"
        in operations
    )
    assert DEFAULT_TERMINAL_JOB_TTL_S == 24 * 60 * 60
    assert f"| One upload | {DEFAULT_MAX_UPLOAD_BYTES // 1024**3} GiB |" in operations
    assert f"| Concurrent uploads | {DEFAULT_MAX_CONCURRENT_UPLOADS} |" in operations
    assert (
        f"| Total managed uploads | {DEFAULT_MAX_UPLOAD_TOTAL_BYTES // 1024**3} GiB |" in operations
    )
    assert f"| Upload retention | {int(DEFAULT_UPLOAD_TTL_S // 3600)} hours |" in operations
    assert f"| Free-space reserve | {DEFAULT_MIN_FREE_BYTES // 1024**2} MiB |" in operations


def test_model_documentation_names_every_locked_core_and_parameter_identity() -> None:
    provenance = _text(ROOT / "docs" / "model-provenance.md")
    locks = sorted((ROOT / "src" / "hawavoclean" / "resources" / "models").glob("*-core.lock.toml"))
    assert len(locks) == 3
    for path in locks:
        lock = tomllib.loads(_text(path))
        assert lock["core_id"] in provenance
        assert lock["params_hash"] in provenance


def test_current_docs_have_no_superseded_release_claims() -> None:
    joined = "\n".join(_text(path) for path in CURRENT_DOCS)
    for stale in (
        "HawaVoClean v1",
        "v1 ships exactly one",
        "Atomic WAV",
        "destination-filesystem staging with rollback",
        "loaded strictly with `weights_only=True`",
        "security@hawzhin.ai",
    ):
        assert stale not in joined
    assert "release candidate in progress" in _text(ROOT / "STATUS.md")
    assert "not a published 10/10 release" in _text(ROOT / "README.md")


def test_current_documentation_local_links_resolve() -> None:
    failures: list[str] = []
    for document in CURRENT_DOCS:
        for raw_target in LINK.findall(_text(document)):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            resolved = (document.parent / unquote(target)).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not failures, "broken local documentation links:\n" + "\n".join(failures)


def test_documentation_consistency_gate_no_open_item_described_as_shipped() -> None:
    """Documentation consistency gate (G0.8): ensure open items are not described as shipped."""
    open_claims = (
        "Windows 11 installer shipped",
        "Resolve PKG signed and notarized",
        "Sorani 300-hour corpus completed",
        "UAE cloud deployed to production",
        "Restore is fully production qualified",
        "HawaVoClean is published 10/10",
    )
    all_docs = CURRENT_DOCS + (
        ROOT / "docs" / "high-end-production-implementation.md",
        ROOT / "docs" / "true-10-readiness-task-sheet.md",
        ROOT / "docs" / "media-preflight.md",
        ROOT / "docs" / "natural-streaming-render.md",
    )
    joined = "\n".join(_text(p) for p in all_docs).lower()
    for claim in open_claims:
        assert claim.lower() not in joined, f"Disallowed shipped claim found: {claim}"

    task_sheet = _text(ROOT / "docs" / "true-10-readiness-task-sheet.md")
    assert "Honest readiness rating" in task_sheet
    assert "[x] | G0.1" in task_sheet
    assert "[x] | G0.2" in task_sheet
    assert "[x] | G0.3" in task_sheet
    assert "[x] | G0.4" in task_sheet
    assert "[x] | G0.8" in task_sheet
