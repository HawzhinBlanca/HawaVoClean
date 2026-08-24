"""Build and runtime provenance used by schema-v2 audit reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from hawavoclean.release import RELEASE_IDENTITY

BUILD_FIELDS = {
    "provenance_schema_version",
    "artifact_type",
    "source_revision",
    "source_date_epoch",
    "source_dirty",
    "dependency_lock_sha256",
    "release_identity_sha256",
    "build_id",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(RuntimeError):
    """Build provenance is absent, malformed, or inconsistent."""


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProvenanceError(f"cannot inspect source-tree provenance: {exc}") from exc
    return result.stdout.strip()


def _validate_build_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != BUILD_FIELDS:
        raise ProvenanceError("build provenance has an invalid field set")
    if value["provenance_schema_version"] != 1:
        raise ProvenanceError("unsupported build provenance schema")
    if value["artifact_type"] not in {"wheel", "sdist", "source-tree"}:
        raise ProvenanceError("build provenance has an invalid artifact type")
    if not isinstance(value["source_revision"], str) or not HEX40.fullmatch(
        value["source_revision"]
    ):
        raise ProvenanceError("build provenance source revision is not a full Git SHA")
    if not isinstance(value["source_date_epoch"], int) or value["source_date_epoch"] <= 0:
        raise ProvenanceError("build provenance has an invalid source date epoch")
    if not isinstance(value["source_dirty"], bool):
        raise ProvenanceError("build provenance has an invalid dirty-source flag")
    for field in ("dependency_lock_sha256", "release_identity_sha256", "build_id"):
        if not isinstance(value[field], str) or not HEX64.fullmatch(value[field]):
            raise ProvenanceError(f"build provenance {field} is not a SHA-256")
    base = {key: item for key, item in value.items() if key != "build_id"}
    expected = hashlib.sha256(_canonical(base)).hexdigest()
    if value["build_id"] != expected:
        raise ProvenanceError("build provenance ID does not recompute from its fields")
    if value["release_identity_sha256"] != RELEASE_IDENTITY.identity_sha256:
        raise ProvenanceError("build provenance names a different packaged release identity")
    if value["artifact_type"] != "source-tree" and value["source_dirty"]:
        raise ProvenanceError("release artifacts cannot claim dirty source")
    return dict(value)


def _source_tree_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    lock_path = root / "uv.lock"
    release_path = root / "src" / "hawavoclean" / "release.json"
    if not lock_path.is_file() or not release_path.is_file() or not (root / ".git").exists():
        raise ProvenanceError("package contains no build provenance and is not a source checkout")
    base: dict[str, Any] = {
        "provenance_schema_version": 1,
        "artifact_type": "source-tree",
        "source_revision": _git(root, "rev-parse", "HEAD"),
        "source_date_epoch": int(_git(root, "show", "-s", "--format=%ct", "HEAD")),
        "source_dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=normal")),
        "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "release_identity_sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
    }
    base["build_id"] = hashlib.sha256(_canonical(base)).hexdigest()
    return _validate_build_identity(base)


def packaged_build_identity() -> dict[str, Any]:
    """Read and validate the generated identity, or describe an editable tree."""
    try:
        item = resources.files("hawavoclean").joinpath("build-provenance.json")
        with item.open("rb") as stream:
            raw = stream.read()
    except (FileNotFoundError, OSError):
        return _source_tree_identity()
    try:
        value: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"build provenance is not valid UTF-8 JSON: {exc}") from exc
    return _validate_build_identity(value)


def distribution_record_sha256() -> str | None:
    """Hash the installed distribution's PEP 376 file inventory."""
    try:
        distribution = metadata.distribution("hawavoclean")
    except metadata.PackageNotFoundError:
        return None
    for entry in distribution.files or ():
        if str(entry).endswith(".dist-info/RECORD"):
            try:
                record_path = Path(str(distribution.locate_file(entry)))
                return hashlib.sha256(record_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ProvenanceError(f"cannot read installed distribution RECORD: {exc}") from exc
    return None


def current_build_report_fields() -> dict[str, Any]:
    """Build fields copied into a newly emitted audit report."""
    value = packaged_build_identity()
    value["distribution_record_sha256"] = distribution_record_sha256()
    return value


def verify_report_build(value: dict[str, Any]) -> None:
    """Reject internally fabricated provenance and wheel/report disagreement."""
    build = {key: value.get(key) for key in BUILD_FIELDS}
    _validate_build_identity(build)
    record = value.get("distribution_record_sha256")
    if record is not None and (not isinstance(record, str) or not HEX64.fullmatch(record)):
        raise ProvenanceError("distribution RECORD digest is not a SHA-256")
    if build["artifact_type"] == "wheel":
        packaged = packaged_build_identity()
        if build != packaged:
            raise ProvenanceError("report build identity does not match the installed wheel")
        current_record = distribution_record_sha256()
        if current_record is None:
            raise ProvenanceError("installed wheel has no readable RECORD inventory")
        if record != current_record:
            raise ProvenanceError("report RECORD digest does not match the installed wheel")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _command_version(command: str) -> str:
    try:
        result = subprocess.run(
            [command, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({type(exc).__name__})"
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0].strip() if first_line else "unavailable (empty version output)"


def runtime_versions() -> dict[str, str]:
    """Exact installed versions for processing-relevant runtimes."""
    result: dict[str, str] = {}
    for name in (
        "hawavoclean",
        "numpy",
        "scipy",
        "soundfile",
        "pyloudnorm",
        "pydantic",
        "cffi",
        "deepfilternet",
        "torch",
        "torchaudio",
        "nara-wpe",
    ):
        version = _package_version(name)
        if version is not None:
            result[name] = version
    try:
        import soundfile

        result["libsndfile"] = str(soundfile.__libsndfile_version__)
    except (ImportError, AttributeError):
        result["libsndfile"] = "unavailable"
    result["ffmpeg"] = _command_version("ffmpeg")
    result["ffprobe"] = _command_version("ffprobe")
    return dict(sorted(result.items()))


def deterministic_settings(config: Any) -> dict[str, str | int | bool]:
    """Describe the numeric/device choices that can affect emitted samples."""
    from hawavoclean.runtime import active_device

    return {
        "compute_device": active_device(),
        "requested_device": str(config.runtime.device),
        "worker_pool_size": int(config.runtime.pool_size()),
        "threads_per_worker": int(config.runtime.threads_per_worker()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "library-default"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "library-default"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "interpreter-randomized"),
        "output_bit_depth": str(config.input.output_bit_depth),
        "tpdf_dither": bool(config.loudness.dither),
        "dither_seed_derivation": "sha256(job_id:channel_index)",
        "result_order": "unit_index",
        "torch_deterministic_algorithms": "not-enforced; device is provenance boundary",
    }
