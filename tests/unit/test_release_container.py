"""Static contract tests for the supported CPU reference container."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
WOLFI_LOCK = ROOT / "docker" / "wolfi-packages.lock"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_container_bases_and_installer_are_immutable() -> None:
    text = _dockerfile()
    from_lines = [line for line in text.splitlines() if line.startswith("FROM ")]

    assert len(from_lines) == 3
    assert all(re.search(r"@sha256:[0-9a-f]{64}(?:\s|$)", line) for line in from_lines)
    assert "uv:0.11.14@sha256:" in from_lines[0]
    assert all("cgr.dev/chainguard/wolfi-base@sha256:" in line for line in from_lines[1:])
    assert text.count("uv sync --frozen") == 1
    assert "uv pip install --python /app/.venv/bin/python --no-deps" in text
    assert text.count("xargs apk add --no-cache < /tmp/wolfi-packages.lock") == 2
    assert "apt-get" not in text


def test_container_copies_the_declared_custom_build_hook() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    hook = project["tool"]["hatch"]["build"]["hooks"]["custom"]["path"]

    assert hook == "hatch_build.py"
    assert re.search(rf"^COPY [^\n]*\b{re.escape(hook)}\b[^\n]* \./$", _dockerfile(), re.MULTILINE)
    assert 'HAWAVOCLEAN_SOURCE_REVISION="${SOURCE_REVISION}"' in _dockerfile()
    assert 'SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}"' in _dockerfile()


def test_container_is_cpu_only_non_root_and_read_only_friendly() -> None:
    text = _dockerfile()

    assert "nvidia" not in text.lower()
    assert 'HAWAVOCLEAN_DEVICE="cpu"' in text
    assert 'HAWAVOCLEAN_WORK_DIR="/cache/work"' in text
    assert "USER 10001:10001" in text
    assert "HEALTHCHECK" in text and '["hawavoclean", "doctor"]' in text
    assert "VOLUME " not in text
    assert 'ENTRYPOINT ["hawavoclean"]' in text


def test_container_metadata_requires_source_identity() -> None:
    text = _dockerfile()

    assert "ARG SOURCE_REVISION" in text
    assert "ARG SOURCE_DATE_EPOCH" in text
    assert "ARG SOURCE_DATE" in text
    assert 'org.opencontainers.image.version="3.3.0"' in text
    assert 'org.opencontainers.image.revision="${SOURCE_REVISION}"' in text
    assert 'org.opencontainers.image.created="${SOURCE_DATE}"' in text


def test_wolfi_runtime_lock_pins_every_package_and_core_runtime() -> None:
    packages = WOLFI_LOCK.read_text(encoding="utf-8").splitlines()
    names = [line.split("=", 1)[0] for line in packages]

    assert len(packages) >= 75
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-zA-Z0-9+_.-]+=[a-zA-Z0-9+_.:-]+", line) for line in packages)
    assert {"python-3.12", "ffmpeg-7", "libsndfile"}.issubset(names)


def test_docker_context_excludes_private_and_generated_state() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for required in {".git", ".claude", ".venv", "build", "evidence", "test_output"}:
        assert required in ignored
