"""Transactional Resolve activation: install, upgrade, rollback and repeat."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ACTIVATE = ROOT / "resolve-plugin" / "activate.sh"
PLUGIN_ID = "com.hawavoclean.resolve"


def _write_stage(root: Path, version: str, payload: str) -> Path:
    stage = root / f"stage-{version}-{payload}"
    engine = stage / "engine"
    engine.mkdir(parents=True)
    files = {
        "PLUGIN_ID": f"{PLUGIN_ID}\n",
        "VERSION": f"{version}\n",
        "manifest.xml": f"<Plugin><Id>{PLUGIN_ID}</Id></Plugin>\n",
        "package.json": f'{{"name":"hawavoclean-resolve","version":"{version}"}}\n',
        "main.js": f"// {payload}\n",
        "preload.js": "// preload\n",
        "engine.json": '{"command":["./engine/hawavoclean-engine","serve"],"cwd":".","env":{}}\n',
        "index.html": f"<html>{payload}</html>\n",
        "engine/hawavoclean-engine": "#!/bin/sh\nexit 0\n",
    }
    for relative, content in files.items():
        path = stage / relative
        path.write_text(content)
    (engine / "hawavoclean-engine").chmod(0o755)
    (stage / "SYMLINKS").write_text("")
    checksums: list[str] = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  ./{path.relative_to(stage).as_posix()}\n")
    (stage / "SHA256SUMS").write_text("".join(checksums))
    return stage


def _activate(stage: Path, dest: Path, failpoint: str = "") -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HAWA_ACTIVATE_ALLOW_RUNNING": "1"}
    if failpoint:
        env["HAWA_INSTALL_FAILPOINT"] = failpoint
    return subprocess.run(
        ["bash", str(ACTIVATE), "--stage", str(stage), "--dest", str(dest)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _verify_only(stage: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ACTIVATE), "--stage", str(stage), "--verify-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _installed(dest: Path) -> Path:
    return dest / PLUGIN_ID


def test_install_upgrade_and_exact_repeat_are_transactional(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    first = _write_stage(tmp_path, "3.3.0", "first")
    result = _activate(first, dest)
    assert result.returncode == 0, result.stderr
    assert (_installed(dest) / "main.js").read_text() == "// first\n"

    second = _write_stage(tmp_path, "3.3.1", "second")
    result = _activate(second, dest)
    assert result.returncode == 0, result.stderr
    target = _installed(dest)
    assert (target / "main.js").read_text() == "// second\n"
    inode = target.stat().st_ino

    result = _activate(second, dest)
    assert result.returncode == 0, result.stderr
    assert "already installed" in result.stdout
    assert target.stat().st_ino == inode
    assert not list(dest.glob(f".{PLUGIN_ID}.transaction.*"))


@pytest.mark.parametrize(
    "failpoint", ["after_backup", "after_activate", "corrupt_after_activate", "after_verify"]
)
def test_every_injected_upgrade_failure_restores_the_prior_plugin(
    tmp_path: Path, failpoint: str
) -> None:
    dest = tmp_path / "plugins"
    old = _write_stage(tmp_path, "3.3.0", "known-good")
    new = _write_stage(tmp_path, "3.3.1", "candidate")
    assert _activate(old, dest).returncode == 0

    result = _activate(new, dest, failpoint)
    assert result.returncode != 0
    assert "restoring the prior plugin" in result.stderr
    assert (_installed(dest) / "main.js").read_text() == "// known-good\n"
    assert not list(dest.glob(f".{PLUGIN_ID}.transaction.*"))


def test_failed_first_install_leaves_no_target(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    stage = _write_stage(tmp_path, "3.3.0", "candidate")
    result = _activate(stage, dest, "after_activate")
    assert result.returncode != 0
    assert not _installed(dest).exists()
    assert not list(dest.glob(f".{PLUGIN_ID}.transaction.*"))


def test_tampered_stage_and_unknown_existing_target_are_refused(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    stage = _write_stage(tmp_path, "3.3.0", "candidate")
    (stage / "main.js").write_text("tampered\n")
    result = _activate(stage, dest)
    assert result.returncode != 0
    assert "failed content verification" in result.stderr
    assert not _installed(dest).exists()

    dest.mkdir(exist_ok=True)
    unknown = _installed(dest)
    unknown.mkdir()
    (unknown / "unrelated.txt").write_text("owner unknown")
    clean = _write_stage(tmp_path, "3.3.1", "clean")
    result = _activate(clean, dest)
    assert result.returncode != 0
    assert "not recognizably owned" in result.stderr
    assert (unknown / "unrelated.txt").read_text() == "owner unknown"


def test_unlisted_extra_file_is_outside_manifest_and_refused(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    stage = _write_stage(tmp_path, "3.3.0", "candidate")
    (stage / "unlisted.js").write_text("not covered by SHA256SUMS\n")

    result = _activate(stage, dest)

    assert result.returncode != 0
    assert "regular-file inventory" in result.stderr
    assert not _installed(dest).exists()


def test_verify_only_has_no_install_side_effect(tmp_path: Path) -> None:
    stage = _write_stage(tmp_path, "3.3.0", "candidate")

    result = _verify_only(stage)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(stage)
    assert not (tmp_path / "plugins").exists()


def test_interrupted_backup_state_is_recovered_before_next_install(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    old = _write_stage(tmp_path, "3.3.0", "old")
    new = _write_stage(tmp_path, "3.3.1", "new")
    assert _activate(old, dest).returncode == 0

    transaction = dest / f".{PLUGIN_ID}.transaction.interrupted"
    transaction.mkdir()
    _installed(dest).rename(transaction / "previous")
    assert not _installed(dest).exists()

    result = _activate(new, dest)
    assert result.returncode == 0, result.stderr
    assert "Recovering the prior plugin" in result.stdout
    assert (_installed(dest) / "main.js").read_text() == "// new\n"
    assert not transaction.exists()


def test_interrupted_corrupt_activation_restores_verified_previous(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    old = _write_stage(tmp_path, "3.3.0", "old")
    candidate = _write_stage(tmp_path, "3.3.1", "candidate")
    next_stage = _write_stage(tmp_path, "3.3.2", "next")
    assert _activate(old, dest).returncode == 0

    transaction = dest / f".{PLUGIN_ID}.transaction.interrupted"
    transaction.mkdir()
    _installed(dest).rename(transaction / "previous")
    candidate.rename(_installed(dest))
    (_installed(dest) / "main.js").write_text("corrupt\n")

    result = _activate(next_stage, dest)

    assert result.returncode == 0, result.stderr
    assert "Restoring the prior plugin" in result.stdout
    assert (_installed(dest) / "main.js").read_text() == "// next\n"
    assert not transaction.exists()


def test_ambiguous_stale_transaction_preserves_everything_for_inspection(tmp_path: Path) -> None:
    dest = tmp_path / "plugins"
    dest.mkdir()
    target = _installed(dest)
    target.mkdir()
    (target / "unrelated.txt").write_text("unknown owner\n")
    transaction = dest / f".{PLUGIN_ID}.transaction.interrupted"
    transaction.mkdir()
    old = _write_stage(tmp_path, "3.3.0", "old")
    old.rename(transaction / "previous")
    next_stage = _write_stage(tmp_path, "3.3.1", "next")

    result = _activate(next_stage, dest)

    assert result.returncode != 0
    assert "manual inspection" in result.stderr
    assert (target / "unrelated.txt").read_text() == "unknown owner\n"
    assert (transaction / "previous" / "main.js").read_text() == "// old\n"


def test_assembler_contract_has_no_mutable_or_repo_local_fallback() -> None:
    install = (ROOT / "resolve-plugin" / "install.sh").read_text()
    activate = ACTIVATE.read_text()
    main = (ROOT / "resolve-plugin" / PLUGIN_ID / "main.js").read_text()

    assert "--frozen-lockfile" in install
    assert 'cd "$UI_DIR" && node "$PNPM_CLI" install --frozen-lockfile' in install
    assert 'cd "$SRC_DIR" && node "$PNPM_CLI" install --frozen-lockfile' in install
    assert 'npm --prefix "$TOOLCHAIN_DIR" ci --ignore-scripts' in install
    assert "retrying without" not in install
    assert ".venv/bin/hawavoclean" not in install
    assert '"./engine/hawavoclean-engine", "serve"' in install
    assert '"PYTHONDONTWRITEBYTECODE": "1"' in install
    assert "stage-selftest.sh" in install
    assert 'activate.sh" --stage "$FINAL_STAGE" --verify-only' in install
    assert "engine regular-file inventory" in install
    assert "engine symlink inventory" in install
    assert "write_checksum_manifest" in install
    assert 'batch+=("$rel")' in install
    assert "rm -rf" not in install + activate
    assert "after_backup" in activate and "after_activate" in activate
    assert "restoring the prior plugin" in activate
    assert "relative executable may not escape" in main

    toolchain = json.loads((ROOT / "resolve-plugin" / "toolchain" / "package.json").read_text())
    plugin = json.loads((ROOT / "resolve-plugin" / PLUGIN_ID / "package.json").read_text())
    ui = json.loads((ROOT / "ui" / "package.json").read_text())
    assert toolchain["dependencies"] == {"pnpm": "11.22.0"}
    assert plugin["packageManager"] == "pnpm@11.22.0"
    assert ui["packageManager"] == "pnpm@11.22.0"
    assert (ROOT / "resolve-plugin" / PLUGIN_ID / "pnpm-lock.yaml").is_file()
