#!/usr/bin/env python3
"""Run the complete v3.3 release gate twice from isolated clean checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scripts import generate_sbom

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN_LOCK = ROOT / "evidence" / "release" / "toolchain-lock.json"
REGRESSION_MANIFEST = ROOT / "evidence" / "release" / "audio-regressions.json"
SDK_NODE = Path(
    "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/"
    "Workflow Integrations/Examples/SamplePlugin/WorkflowIntegration.node"
)
RESOLVE_INFO = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Info.plist")
PROMISED_ARTIFACTS = (
    "audio-regression",
    "container-audio",
    "container-image",
    "resolve-engine",
    "resolve-plugin",
    "sbom",
    "sdist",
    "ui",
    "wheel",
    "wheel-smoke-audio",
)
PRIVATE_FIELDS = (
    ("input", "input_sha256"),
    ("reference_audio", "audio_sha256"),
    ("reference_report", "report_sha256"),
)
SMOKE_PROFILE = "production"


class GateError(RuntimeError):
    """The source or one of its required release proofs is invalid."""


@dataclass(frozen=True)
class StepResult:
    name: str
    command: list[str]
    duration_seconds: float
    exit_code: int
    log: str
    log_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-3000:]
        raise GateError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed.stdout.strip()


def _clean_status(root: Path) -> str:
    return _capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"], cwd=root
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON contract is not an object: {path}")
    return cast(dict[str, Any], value)


def _toolchain_lock(path: Path = TOOLCHAIN_LOCK) -> dict[str, Any]:
    value = _load_json(path)
    tools = value.get("tools")
    resolve = value.get("resolve_host")
    if (
        value.get("schema_version") != 1
        or value.get("release_host") != "macos-arm64"
        or not isinstance(tools, dict)
        or not isinstance(resolve, dict)
    ):
        raise GateError("release toolchain lock has an unexpected shape")
    required_tools = {
        "check-jsonschema",
        "docker",
        "ffmpeg",
        "mypy",
        "node",
        "npm",
        "pip-audit",
        "pnpm",
        "pytest",
        "python",
        "resolve-engine-python",
        "ruff",
        "trivy",
        "uv",
    }
    if set(tools) != required_tools or not all(isinstance(item, str) for item in tools.values()):
        raise GateError("release toolchain lock does not name every exact tool version")
    if set(resolve) != {"build", "sdk_node_sha256", "version"} or not all(
        isinstance(item, str) for item in resolve.values()
    ):
        raise GateError("release host lock is incomplete")
    return value


def _expect(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise GateError(f"{label} version drift: expected {expected}, got {actual}")


def _preflight_toolchain(lock: dict[str, Any]) -> dict[str, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise GateError("the full v3.3 release gate requires the declared macOS arm64 release host")
    tools = cast(dict[str, str], lock["tools"])
    trivy_details = _capture(["trivy", "--version"])
    actual = {
        "uv": _capture(["uv", "--version"]).split()[1],
        "python": platform.python_version(),
        "node": _capture(["node", "--version"]).removeprefix("v"),
        "npm": _capture(["npm", "--version"]),
        "docker": re.search(r"Docker version ([^,]+)", _capture(["docker", "--version"])),
        "trivy": trivy_details.splitlines()[0].removeprefix("Version: "),
        "ffmpeg": _capture(["ffmpeg", "-version"]).splitlines()[0].split()[2],
        "pip-audit": _capture(
            ["uvx", "--from", f"pip-audit=={tools['pip-audit']}", "pip-audit", "--version"]
        ).split()[-1],
        "check-jsonschema": _capture(
            [
                "uvx",
                "--from",
                f"check-jsonschema=={tools['check-jsonschema']}",
                "check-jsonschema",
                "--version",
            ]
        ).split()[-1],
    }
    docker_match = actual["docker"]
    if not isinstance(docker_match, re.Match):
        raise GateError("could not parse Docker version")
    actual["docker"] = docker_match.group(1)

    managed_python = Path(_capture(["uv", "python", "find", "--managed-python", "3.11"]))
    actual["resolve-engine-python"] = _capture(
        [str(managed_python), "-c", "import platform;print(platform.python_version())"]
    )
    for label, expected in tools.items():
        if label in {"mypy", "pnpm", "pytest", "ruff"}:
            continue
        _expect(label, str(actual[label]), expected)

    actual["host-architecture"] = platform.machine()
    actual["host-macos"] = platform.mac_ver()[0]
    actual["trivy-state-sha256"] = hashlib.sha256(trivy_details.encode("utf-8")).hexdigest()
    database_updated = re.search(r"^  UpdatedAt: (.+)$", trivy_details, re.MULTILINE)
    check_bundle = re.search(r"^  Digest: (sha256:[0-9a-f]{64})$", trivy_details, re.MULTILINE)
    if database_updated is None or check_bundle is None:
        raise GateError("Trivy did not report its vulnerability database and check-bundle identity")
    actual["trivy-database-updated-at"] = database_updated.group(1)
    actual["trivy-check-bundle-digest"] = check_bundle.group(1)

    _capture(["docker", "info", "--format", "{{.ServerVersion}}"])
    if not SDK_NODE.is_file() or SDK_NODE.is_symlink():
        raise GateError(f"the pinned Resolve SDK bridge is unavailable: {SDK_NODE}")
    resolve_lock = cast(dict[str, str], lock["resolve_host"])
    _expect("Resolve SDK bridge hash", _sha256(SDK_NODE), resolve_lock["sdk_node_sha256"])
    try:
        with RESOLVE_INFO.open("rb") as stream:
            resolve_info: Any = plistlib.load(stream)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise GateError(f"cannot read the installed Resolve identity: {exc}") from exc
    if not isinstance(resolve_info, dict):
        raise GateError("installed Resolve Info.plist is not a dictionary")
    _expect(
        "Resolve version",
        str(resolve_info.get("CFBundleShortVersionString")),
        resolve_lock["version"],
    )
    _expect("Resolve build", str(resolve_info.get("CFBundleVersion")), resolve_lock["build"])
    return {key: str(value) for key, value in actual.items()}


def _artifact_identity(name: str, path: Path) -> dict[str, Any]:
    component = generate_sbom._artifact_component(name, path.resolve())
    hashes = component.get("hashes")
    if not isinstance(hashes, list):
        raise GateError(f"artifact has no hash inventory: {path}")
    digest = next(
        (
            item.get("content")
            for item in hashes
            if isinstance(item, dict) and item.get("alg") == "SHA-256"
        ),
        None,
    )
    if not isinstance(digest, str):
        raise GateError(f"artifact has no SHA-256 identity: {path}")
    properties = component.get("properties", [])
    details = {
        str(item["name"]): str(item["value"])
        for item in properties
        if isinstance(item, dict) and "name" in item and "value" in item
    }
    return {
        "sha256": digest,
        "kind": details.get("hawavoclean:artifact-kind", "unknown"),
        "file_count": int(details.get("hawavoclean:artifact-tree-file-count", "0")),
        "symlink_count": int(details.get("hawavoclean:artifact-tree-symlink-count", "0")),
        "total_size": int(
            details.get(
                "hawavoclean:artifact-tree-total-size",
                details.get("hawavoclean:artifact-file-size", "0"),
            )
        ),
    }


def _copy_private_inputs(source_root: Path, checkout: Path, manifest_path: Path) -> dict[str, str]:
    manifest = _load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise GateError("audio regression manifest has no cases")
    copied: dict[str, str] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise GateError("audio regression manifest contains a non-object case")
        for path_field, hash_field in PRIVATE_FIELDS:
            relative = case.get(path_field)
            expected = case.get(hash_field)
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise GateError(f"audio regression case has invalid {path_field}/{hash_field}")
            unresolved = source_root / relative
            source = unresolved.resolve()
            try:
                source.relative_to(source_root.resolve())
            except ValueError as exc:
                raise GateError(
                    f"private regression path escapes the source root: {relative}"
                ) from exc
            if unresolved.is_symlink() or not source.is_file():
                raise GateError(f"private regression artifact is unavailable: {relative}")
            actual = _sha256(source)
            if actual != expected:
                raise GateError(
                    f"private regression artifact hash drift for {relative}: {actual} != {expected}"
                )
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            copied[relative] = actual
    return dict(sorted(copied.items()))


class Runner:
    def __init__(self, checkout: Path, log_dir: Path, environment: dict[str, str]) -> None:
        self.checkout = checkout
        self.log_dir = log_dir
        self.environment = environment
        self.steps: list[StepResult] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        name: str,
        command: list[str],
        *,
        cwd: Path | None = None,
        extra_environment: dict[str, str] | None = None,
        timeout: int = 3600,
    ) -> None:
        if any(step.name == name for step in self.steps):
            raise GateError(f"duplicate release-gate step name: {name}")
        log_path = self.log_dir / f"{len(self.steps) + 1:02d}-{name}.log"
        print(f"[{len(self.steps) + 1:02d}] {name} ...", flush=True)
        started = time.monotonic()
        exit_code = 1
        environment = dict(self.environment)
        if extra_environment is not None:
            environment.update(extra_environment)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=cwd or self.checkout,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nrelease gate timeout after {timeout} seconds\n")
            exit_code = 124
        duration = time.monotonic() - started
        result = StepResult(
            name=name,
            command=command,
            duration_seconds=round(duration, 3),
            exit_code=exit_code,
            log=log_path.name,
            log_sha256=_sha256(log_path),
        )
        self.steps.append(result)
        if exit_code != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise GateError(f"release step failed: {name} (exit {exit_code})\n{tail}")
        print(f"     passed in {duration:.1f}s", flush=True)


def _single_glob(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise GateError(f"expected one {label}, found {len(matches)} under {root}")
    return matches[0]


def _assert_python_tool_versions(checkout: Path, expected: dict[str, str]) -> dict[str, str]:
    commands = {
        "python": [
            "uv",
            "run",
            "--frozen",
            "python",
            "-c",
            "import platform;print(platform.python_version())",
        ],
        "ruff": ["uv", "run", "--frozen", "ruff", "--version"],
        "mypy": ["uv", "run", "--frozen", "mypy", "--version"],
        "pytest": ["uv", "run", "--frozen", "pytest", "--version"],
    }
    actual: dict[str, str] = {}
    for name, command in commands.items():
        output = _capture(command, cwd=checkout)
        if name == "python":
            actual[name] = output
        else:
            match = re.search(r"([0-9]+(?:\.[0-9]+)+)", output)
            if match is None:
                raise GateError(f"cannot parse {name} version: {output}")
            actual[name] = match.group(1)
        _expect(name, actual[name], expected[name])
    return actual


def _assert_pnpm_version(checkout: Path, expected: str) -> Path:
    cli = checkout / "resolve-plugin" / "toolchain" / "node_modules" / "pnpm" / "bin" / "pnpm.mjs"
    if not cli.is_file():
        raise GateError("the exact pnpm bootstrap did not produce its declared CLI")
    _expect("pnpm", _capture(["node", str(cli), "--version"], cwd=checkout), expected)
    return cli


def _container_packages(checkout: Path, image: str) -> dict[str, Any]:
    installed = set(
        _capture(["docker", "run", "--rm", "--entrypoint", "apk", image, "info", "-v"]).splitlines()
    )
    expected = {
        line.strip().replace("=", "-")
        for line in (checkout / "docker" / "wolfi-packages.lock").read_text().splitlines()
        if line.strip()
    }
    missing = sorted(expected - installed)
    if missing:
        raise GateError(f"container is missing exact locked packages: {missing[:10]}")
    return {
        "locked": len(expected),
        "verified": len(expected),
        "unexpected_count": len(installed - expected),
    }


def _verify_command(prefix: list[str], output: str, report: str) -> list[str]:
    """Build the explicit verification contract; no adjacent-file guessing."""
    return [*prefix, "verify", output, "--report", report]


def _run_pass(
    index: int,
    checkout: Path,
    log_dir: Path,
    commit: str,
    epoch: int,
    source_date: str,
    lock: dict[str, Any],
) -> dict[str, Any]:
    started_at = _utc_now()
    started = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "CI": "1",
            "HAWAVOCLEAN_DEVICE": "cpu",
            "LC_ALL": "C",
            "NO_COLOR": "1",
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "UV_FROZEN": "1",
        }
    )
    runner = Runner(checkout, log_dir, env)
    build_root = checkout / "build" / "full-release-gate"
    artifact_root = build_root / "artifacts"
    artifact_root.mkdir(parents=True)
    tools = cast(dict[str, str], lock["tools"])

    runner.run("python-lock-sync", ["uv", "sync", "--frozen", "--all-extras"], timeout=7200)
    python_tools = _assert_python_tool_versions(checkout, tools)
    runner.run("format", ["uv", "run", "--frozen", "ruff", "format", "--check", "."])
    runner.run("lint", ["uv", "run", "--frozen", "ruff", "check", "."])
    runner.run(
        "strict-types",
        ["uv", "run", "--frozen", "mypy", "--strict", "src", "tests", "scripts", "data"],
    )
    runner.run(
        "release-identity", ["uv", "run", "--frozen", "python", "scripts/sync_release_identity.py"]
    )
    runner.run(
        "generated-schemas",
        ["uv", "run", "--frozen", "python", "scripts/generate_schemas.py", "--check"],
    )
    runner.run(
        "sorani-protocol-design",
        ["uv", "run", "--frozen", "python", "-m", "scripts.validate_sorani_protocol"],
    )
    runner.run(
        "sorani-source-design",
        ["uv", "run", "--frozen", "python", "-m", "scripts.validate_sorani_sources"],
    )
    runner.run(
        "default-tests-branch-coverage",
        [
            "uv",
            "run",
            "--frozen",
            "pytest",
            "-q",
            "--cov=hawavoclean",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-fail-under=92.49",
        ],
        timeout=7200,
    )
    runner.run(
        "fuzz-tests",
        ["uv", "run", "--frozen", "pytest", "-q", "-m", "fuzz", "tests/fuzz"],
        timeout=7200,
    )
    runner.run(
        "mutation-gate",
        ["uv", "run", "--frozen", "python", "scripts/mutation_gate.py"],
        timeout=7200,
    )

    regression_output = build_root / "audio-regression.json"
    runner.run(
        "real-audio-regressions",
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/audio_regression_gate.py",
            "--runs",
            "2",
            "--output-json",
            str(regression_output),
        ],
        timeout=14400,
    )

    toolchain_dir = checkout / "resolve-plugin" / "toolchain"
    runner.run(
        "pnpm-bootstrap",
        [
            "npm",
            "--prefix",
            str(toolchain_dir),
            "ci",
            "--ignore-scripts",
            "--audit=false",
            "--fund=false",
        ],
    )
    pnpm = _assert_pnpm_version(checkout, tools["pnpm"])
    runner.run(
        "ui-lock-install",
        ["node", str(pnpm), "--dir", "ui", "install", "--frozen-lockfile"],
        timeout=3600,
    )
    runner.run("ui-typecheck", ["node", str(pnpm), "--dir", "ui", "typecheck"])
    runner.run("ui-tests", ["node", str(pnpm), "--dir", "ui", "test:run"], timeout=3600)
    runner.run("ui-build", ["node", str(pnpm), "--dir", "ui", "build"])
    runner.run(
        "plugin-lock-install",
        [
            "node",
            str(pnpm),
            "--dir",
            "resolve-plugin/com.hawavoclean.resolve",
            "install",
            "--frozen-lockfile",
        ],
    )
    runner.run(
        "ui-audit", ["node", str(pnpm), "--dir", "ui", "audit", "--audit-level", "low", "--json"]
    )
    runner.run(
        "plugin-audit",
        [
            "node",
            str(pnpm),
            "--dir",
            "resolve-plugin/com.hawavoclean.resolve",
            "audit",
            "--audit-level",
            "low",
            "--json",
        ],
    )
    runner.run(
        "toolchain-audit",
        ["npm", "--prefix", str(toolchain_dir), "audit", "--audit-level=low", "--json"],
    )

    audit_requirements = build_root / "audit-requirements.txt"
    runner.run(
        "python-audit-export",
        [
            "uv",
            "export",
            "--quiet",
            "--frozen",
            "--all-extras",
            "--all-groups",
            "--no-emit-project",
            "--output-file",
            str(audit_requirements),
        ],
    )
    runner.run(
        "python-audit",
        [
            "uvx",
            "--from",
            f"pip-audit=={tools['pip-audit']}",
            "pip-audit",
            "--requirement",
            str(audit_requirements),
            "--no-deps",
            "--disable-pip",
            "--require-hashes",
            "--format",
            "json",
        ],
        timeout=3600,
    )

    python_artifacts = artifact_root / "python"
    python_artifacts.mkdir()
    runner.run(
        "build-wheel-sdist",
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(python_artifacts)],
        extra_environment={
            "HAWAVOCLEAN_SOURCE_REVISION": commit,
            "SOURCE_DATE_EPOCH": str(epoch),
        },
        timeout=3600,
    )
    wheel = _single_glob(python_artifacts, "*.whl", "wheel")
    sdist = _single_glob(python_artifacts, "*.tar.gz", "source archive")

    smoke_requirements = build_root / "smoke-requirements.txt"
    smoke_venv = build_root / "wheel-smoke-venv"
    runner.run(
        "wheel-smoke-export",
        [
            "uv",
            "export",
            "--quiet",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(smoke_requirements),
        ],
    )
    runner.run(
        "wheel-smoke-venv",
        ["uv", "venv", "--python", "3.11", "--managed-python", str(smoke_venv)],
    )
    smoke_python = smoke_venv / "bin" / "python"
    smoke_cli = smoke_venv / "bin" / "hawavoclean"
    runner.run(
        "wheel-smoke-dependencies",
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(smoke_python),
            "--require-hashes",
            str(smoke_requirements),
        ],
        timeout=3600,
    )
    runner.run(
        "wheel-smoke-install",
        ["uv", "pip", "install", "--python", str(smoke_python), "--no-deps", str(wheel)],
    )
    runner.run("wheel-smoke-doctor", [str(smoke_cli), "doctor"])
    smoke_dir = build_root / "wheel-smoke"
    smoke_dir.mkdir()
    smoke_output = smoke_dir / "clean.wav"
    runner.run(
        "wheel-cli-process",
        [
            str(smoke_cli),
            "process",
            "tests/fixtures/sample_sorani_podcast.wav",
            "--output",
            str(smoke_output),
            "--profile",
            SMOKE_PROFILE,
            "--overwrite",
        ],
    )
    runner.run(
        "wheel-cli-verify",
        _verify_command(
            [str(smoke_cli)],
            str(smoke_output),
            str(smoke_output.with_suffix(".hawavoclean.json")),
        ),
    )

    engine = artifact_root / "resolve-engine"
    runner.run(
        "resolve-engine-build",
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/build_resolve_engine.py",
            "--wheel",
            str(wheel),
            "--output",
            str(engine),
        ],
        timeout=7200,
    )
    runner.run(
        "resolve-plugin-self-test",
        [
            "bash",
            "resolve-plugin/install.sh",
            "--engine-bundle",
            str(engine),
            "--skip-ui-build",
            "--no-install",
        ],
        timeout=7200,
    )
    plugin = _single_glob(
        checkout / "build" / "resolve-plugin" / "stages",
        "*/com.hawavoclean.resolve",
        "content-addressed Resolve plugin",
    )

    image_tag = f"hawavoclean:release-gate-{commit[:12]}-run{index}"
    runner.run(
        "container-build",
        [
            "docker",
            "build",
            "--platform",
            "linux/arm64",
            "--provenance=false",
            "--build-arg",
            f"SOURCE_REVISION={commit}",
            "--build-arg",
            f"SOURCE_DATE_EPOCH={epoch}",
            "--build-arg",
            f"SOURCE_DATE={source_date}",
            "--tag",
            image_tag,
            ".",
        ],
        timeout=7200,
    )
    inspection = _load_json_from_text(_capture(["docker", "image", "inspect", image_tag]))
    if (
        not isinstance(inspection, list)
        or len(inspection) != 1
        or not isinstance(inspection[0], dict)
    ):
        raise GateError("Docker did not return exactly one image inspection record")
    image_id = inspection[0].get("Id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise GateError("Docker image has no immutable sha256 identity")
    package_proof = _container_packages(checkout, image_id)

    container_dir = build_root / "container-smoke"
    container_dir.mkdir()
    shutil.copy2(
        checkout / "tests" / "fixtures" / "sample_sorani_podcast.wav", container_dir / "input.wav"
    )
    container_dir.chmod(0o777)
    docker_runtime = [
        "docker",
        "run",
        "--rm",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m",
        "--tmpfs",
        "/cache:rw,uid=10001,gid=10001,mode=0750,size=2g",
        "--mount",
        f"type=bind,source={container_dir},target=/work",
        image_id,
    ]
    runner.run("container-doctor", [*docker_runtime, "doctor"])
    runner.run(
        "container-process",
        [
            *docker_runtime,
            "process",
            "/work/input.wav",
            "--output",
            "/work/output.wav",
            "--profile",
            SMOKE_PROFILE,
            "--overwrite",
        ],
    )
    runner.run(
        "container-verify",
        _verify_command(
            docker_runtime,
            "/work/output.wav",
            "/work/output.hawavoclean.json",
        ),
    )

    image_scan = build_root / "trivy-image.json"
    config_scan = build_root / "trivy-config.json"
    runner.run(
        "container-vulnerability-scan",
        [
            "trivy",
            "image",
            "--quiet",
            "--scanners",
            "vuln",
            "--severity",
            "HIGH,CRITICAL",
            "--exit-code",
            "1",
            "--format",
            "json",
            "--output",
            str(image_scan),
            image_id,
        ],
        timeout=3600,
    )
    runner.run(
        "container-configuration-scan",
        [
            "trivy",
            "config",
            "--quiet",
            "--severity",
            "HIGH,CRITICAL",
            "--exit-code",
            "1",
            "--format",
            "json",
            "--output",
            str(config_scan),
            "Dockerfile",
        ],
        timeout=3600,
    )

    sbom = artifact_root / "hawavoclean-3.3.0.cdx.json"
    runner.run(
        "artifact-bound-sbom",
        [
            "bash",
            "scripts/generate_sbom.sh",
            "--image",
            image_id,
            "--artifact",
            f"wheel={wheel}",
            "--artifact",
            f"sdist={sdist}",
            "--artifact",
            f"ui={checkout / 'ui' / 'dist'}",
            "--artifact",
            f"resolve-plugin={plugin}",
            "--output",
            str(sbom),
        ],
        timeout=7200,
    )

    status = _clean_status(checkout)
    if status:
        raise GateError(f"release gate mutated tracked source:\n{status}")
    artifacts = {
        "audio-regression": _artifact_identity("audio-regression", regression_output),
        "container-audio": _artifact_identity("container-audio", container_dir / "output.wav"),
        "container-image": {"sha256": image_id.removeprefix("sha256:"), "kind": "oci-image"},
        "resolve-engine": _artifact_identity("resolve-engine", engine),
        "resolve-plugin": _artifact_identity("resolve-plugin", plugin),
        "sbom": _artifact_identity("sbom", sbom),
        "sdist": _artifact_identity("sdist", sdist),
        "ui": _artifact_identity("ui", checkout / "ui" / "dist"),
        "wheel": _artifact_identity("wheel", wheel),
        "wheel-smoke-audio": _artifact_identity("wheel-smoke-audio", smoke_output),
    }
    return {
        "index": index,
        "status": "passed",
        "started_at": started_at,
        "duration_seconds": round(time.monotonic() - started, 3),
        "steps": [asdict(step) for step in runner.steps],
        "python_tools": python_tools,
        "container_packages": package_proof,
        "artifacts": artifacts,
    }


def _load_json_from_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GateError(f"command emitted invalid JSON: {exc}") from exc


def _compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise GateError("at least two full clean-checkout passes are required")
    baseline = runs[0].get("artifacts")
    if not isinstance(baseline, dict) or set(baseline) != set(PROMISED_ARTIFACTS):
        raise GateError("release pass did not record the complete promised artifact set")
    hashes: dict[str, str] = {}
    for name in PROMISED_ARTIFACTS:
        identity = baseline.get(name)
        if not isinstance(identity, dict) or not isinstance(identity.get("sha256"), str):
            raise GateError(f"release pass artifact is missing its identity: {name}")
        hashes[name] = str(identity["sha256"])
    for run in runs[1:]:
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, dict):
            raise GateError("release pass omitted its artifact inventory")
        for name, expected in hashes.items():
            identity = artifacts.get(name)
            actual = identity.get("sha256") if isinstance(identity, dict) else None
            if actual != expected:
                raise GateError(
                    f"non-reproducible artifact {name}: pass 1={expected}, "
                    f"pass {run.get('index')}={actual}"
                )
    return {"status": "passed", "passes": len(runs), "artifact_sha256": hashes}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = dict(report)
    payload["proof_sha256"] = _canonical_sha256(report)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _worktree(commit: str) -> tuple[Path, Path]:
    parent = Path(tempfile.mkdtemp(prefix="hawavoclean-release-gate-"))
    checkout = parent / "checkout"
    _capture(["git", "worktree", "add", "--detach", str(checkout), commit])
    return parent, checkout


def _remove_worktree(parent: Path, checkout: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(checkout)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    shutil.rmtree(parent, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="full isolated passes; values below two are rejected",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "release-gate",
        help="ignored directory in which to retain logs and the proof report",
    )
    args = parser.parse_args()
    if args.runs < 2:
        print("release gate failed: --runs must be at least 2", file=sys.stderr)
        return 2

    started_at = _utc_now()
    session_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    session = args.output_dir.resolve() / session_name
    report_path = session / "release-gate-proof.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "source_commit": None,
        "source_date_epoch": None,
        "started_at": started_at,
        "finished_at": None,
        "status": "failed",
        "toolchain_lock_sha256": None,
        "toolchain": {},
        "external_inputs": {},
        "runs": [],
        "reproducibility": None,
        "known_limits": [
            "Real Sorani human acceptance remains Phase 5 and is not implied by this automated gate.",
            "Actual in-Resolve workflow and accessibility acceptance remain Phase 6.",
            "Vendor-owned Resolve Electron risk still requires explicit acceptance or a qualifying update.",
            "GitHub required-check and branch-protection proof remains T3.2/T3.3 after user checkpoint U1.",
        ],
    }
    try:
        if _clean_status(ROOT):
            raise GateError("the invoking checkout is not clean")
        commit = _capture(["git", "rev-parse", "HEAD"])
        epoch = int(_capture(["git", "show", "-s", "--format=%ct", commit]))
        source_date = (
            datetime.fromtimestamp(epoch, tz=UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        lock = _toolchain_lock()
        toolchain = _preflight_toolchain(lock)
        report.update(
            {
                "source_commit": commit,
                "source_date_epoch": epoch,
                "toolchain_lock_sha256": _sha256(TOOLCHAIN_LOCK),
                "toolchain": toolchain,
            }
        )
        runs: list[dict[str, Any]] = []
        for index in range(1, args.runs + 1):
            print(f"\n=== full isolated release pass {index}/{args.runs} ===", flush=True)
            parent, checkout = _worktree(commit)
            try:
                private = _copy_private_inputs(ROOT, checkout, REGRESSION_MANIFEST)
                if index == 1:
                    report["external_inputs"] = {
                        "private_regression_artifacts": private,
                        "resolve_sdk_node_sha256": _sha256(SDK_NODE),
                    }
                run = _run_pass(
                    index,
                    checkout,
                    session / f"pass-{index}" / "logs",
                    commit,
                    epoch,
                    source_date,
                    lock,
                )
                runs.append(run)
                report["runs"] = runs
            finally:
                _remove_worktree(parent, checkout)
        reproducibility = _compare_runs(runs)
        if _clean_status(ROOT):
            raise GateError("the release gate changed the invoking checkout")
        report["reproducibility"] = reproducibility
        report["status"] = "passed"
    except (GateError, OSError, subprocess.SubprocessError, ValueError) as exc:
        report["error"] = str(exc)
        print(f"\nrelease gate failed: {exc}", file=sys.stderr, flush=True)
    finally:
        report["finished_at"] = _utc_now()
        _write_report(report_path, report)
        print(f"release gate proof: {report_path}", flush=True)

    if report["status"] != "passed":
        return 1
    print("\nFULL RELEASE GATE PASSED: two clean checkouts produced identical artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
