"""Hash-verified registry and installer for HawaVoClean's neural models.

Every neural model supported by HawaVoClean is declared once in this module,
reading its expected SHA-256 digest(s) from the package's existing lockfiles
and core constants so there is a single source of truth.

Security & Integrity Guarantees:
- Model IDs are strictly allowlisted; unknown IDs fail closed.
- Download URLs are strictly validated ``https://`` URLs only (including across
  HTTP redirects; downgrades to ``http://`` or ``file://`` are refused).
- Downloads stream into a private ``.part`` temporary file inside
  ``app_data_root() / "models"`` while hashing SHA-256 incrementally.
- Any SHA-256 mismatch or stream failure deletes the ``.part`` file immediately
  and raises :class:`ModelHashMismatchError` / :class:`ModelDownloadError`
  without touching the final destination.
- Exact SHA-256 matches are promoted atomically via ``os.replace``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import hmac
import importlib
import importlib.util
import os
import secrets
import threading
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hawavoclean.enhancement.dereverb_neural import (
    DEREVERB_ARCH_LICENSE,
    DEREVERB_CHECKPOINT_FILE,
    DEREVERB_CHECKPOINT_SHA256,
    DEREVERB_MODEL_LICENSE,
    dereverb_checkpoint_path,
)
from hawavoclean.enhancement.studio_pro import (
    STUDIO_PRO_CHECKPOINT_FILE,
    STUDIO_PRO_CHECKPOINT_SHA256,
    studio_pro_checkpoint_path,
)
from hawavoclean.hashing import hash_file
from hawavoclean.paths import app_data_root, models_dir
from hawavoclean.restoration.neural_bwe import (
    LAVASR_MODEL_LICENSE,
    LAVASR_V2_CONFIG_FILE,
    LAVASR_V2_CONFIG_SHA256,
    LAVASR_V2_WEIGHTS_FILE,
    LAVASR_V2_WEIGHTS_SHA256,
    lavasr_model_dir,
)
from hawavoclean.separation.roformer import (
    ROFORMER_CONFIG_FILENAME,
    ROFORMER_LICENSE,
    ROFORMER_MODEL_FILENAME,
    _roformer_model_dir,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[1]
_DOWNLOAD_CHUNK_BYTES = 256 * 1024
_DOWNLOAD_TIMEOUT_S = 60.0

# Pinned SHA-256 digests for MIT Mel-Band RoFormer (ViperX 1143, ep 3005).
ROFORMER_MODEL_SHA256 = "21b9d0958e35b8ebfbe2afe69bbd5444e5ffe2f5d80ae0d583b833d2f3c0d139"
ROFORMER_CONFIG_SHA256 = "d9b083b48dfdd0bd10f8a29a9c18777b0419496d938827f48a1db31bf0193aa3"

ProgressCallback = Callable[[int, int | None], None]
FetchStream = bytes | bytearray | Iterable[bytes] | tuple[Iterable[bytes], int | None]
Fetcher = Callable[[str], FetchStream]

_INSTALL_LOCKS_GUARD = threading.Lock()
_INSTALL_LOCKS: dict[str, threading.Lock] = {}


class ModelManagerError(RuntimeError):
    """Base error for model registry and hash-verified installation failures."""


class UnknownModelError(ModelManagerError):
    """The requested ``model_id`` is not in the HawaVoClean allowlist."""


class ModelNotDownloadableError(ModelManagerError):
    """The requested model is bundled or has no downloadable artifact URLs."""


class ModelHashMismatchError(ModelManagerError):
    """A downloaded artifact's SHA-256 digest did not match the pinned lock."""


class ModelDownloadError(ModelManagerError):
    """Downloading a model artifact failed before digest verification."""


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One hash-pinned file belonging to a neural model."""

    filename: str
    expected_sha256: str
    download_urls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Declarative specification for one allowlisted HawaVoClean neural model."""

    model_id: str
    display_name: str
    description: str
    used_by: tuple[str, ...]
    license: str
    expected_sha256: str
    approx_size_mb: float
    bundled: bool
    target_path: Path
    download_urls: tuple[str, ...]
    artifacts: tuple[ModelArtifact, ...]


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Current local installation and verification state of one neural model."""

    model_id: str
    display_name: str
    description: str
    used_by: tuple[str, ...]
    license: str
    expected_sha256: str
    approx_size_mb: float
    bundled: bool
    downloadable: bool
    installed: bool
    verified: bool
    size_bytes: int
    path: str | None

    def to_dict(self, *, redact_path: Callable[[str], str] | None = None) -> dict[str, Any]:
        """Serialize status into a JSON-friendly dictionary."""
        reported_path = self.path
        if reported_path is not None and redact_path is not None:
            reported_path = redact_path(reported_path)
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "description": self.description,
            "used_by": list(self.used_by),
            "license": self.license,
            "expected_sha256": self.expected_sha256,
            "approx_size_mb": self.approx_size_mb,
            "bundled": self.bundled,
            "downloadable": self.downloadable,
            "installed": self.installed,
            "verified": self.verified,
            "size_bytes": self.size_bytes,
            "path": reported_path,
        }


@dataclass(frozen=True, slots=True)
class ModelInstallResult:
    """Result of a hash-verified :func:`install_model` operation."""

    model_id: str
    installed: bool
    verified: bool
    sha256: str
    size_bytes: int
    path: str
    already_installed: bool = False
    message: str = ""

    def to_dict(self, *, redact_path: Callable[[str], str] | None = None) -> dict[str, Any]:
        """Serialize installation outcome into a JSON-friendly dictionary."""
        reported_path = redact_path(self.path) if redact_path is not None else self.path
        return {
            "model_id": self.model_id,
            "installed": self.installed,
            "verified": self.verified,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "path": reported_path,
            "already_installed": self.already_installed,
            "message": self.message,
        }


def validate_download_url(url: str) -> str:
    """Validate that a model download URL uses strict HTTPS with no credentials."""
    if not isinstance(url, str) or not url or url != url.strip():
        raise ModelDownloadError("Model download URL must be a non-empty HTTPS URL.")
    if any(ord(ch) < 32 or ch.isspace() for ch in url):
        raise ModelDownloadError("Model download URL must not contain whitespace or control chars.")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ModelDownloadError("Model download URL is malformed.") from exc
    if parsed.scheme != "https":
        raise ModelDownloadError(
            f"Only HTTPS model downloads are permitted (got scheme {parsed.scheme!r})."
        )
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ModelDownloadError("Model download URL must have a hostname and no credentials.")
    if parsed.port is not None and not (1 <= parsed.port <= 65535):
        raise ModelDownloadError("Model download URL has an invalid port.")
    return url


def _load_deepfilternet3_lock() -> tuple[str, str, tuple[ModelArtifact, ...]]:
    """Read DeepFilterNet3 license and weight SHA-256 digests from ``studio-core.lock.toml``."""
    lock_candidates = (
        models_dir() / "studio-core.lock.toml",
        _PACKAGE_ROOT / "resources" / "models" / "studio-core.lock.toml",
    )
    lock_path = next((p for p in lock_candidates if p.is_file()), lock_candidates[-1])
    with lock_path.open("rb") as handle:
        data = tomllib.load(handle)
    license_name = str(data.get("model_license", "MIT"))
    weights = data.get("weight_sha256", {})
    if not isinstance(weights, dict) or not weights:
        raise ModelManagerError("studio-core.lock.toml is missing [weight_sha256] entries")
    artifacts: list[ModelArtifact] = []
    primary_sha = ""
    for rel_key, digest in weights.items():
        rel_str = str(rel_key)
        sha_str = str(digest)
        # Strip leading "deepfilternet3/" prefix so filename is relative to target_dir.
        rel_filename = rel_str.removeprefix("deepfilternet3/")
        artifacts.append(ModelArtifact(filename=rel_filename, expected_sha256=sha_str))
        if rel_filename.endswith(".ckpt.best") or not primary_sha:
            primary_sha = sha_str
    return license_name, primary_sha, tuple(artifacts)


def _load_nisqa_constants() -> tuple[str, str]:
    """Read ``(NISQA_WEIGHTS_FILENAME, NISQA_V2_WEIGHTS_SHA256)`` without requiring torch."""
    if importlib.util.find_spec("torch") is not None:
        nisqa_arch = importlib.import_module("hawavoclean.guard.nisqa_arch")
        return str(nisqa_arch.NISQA_WEIGHTS_FILENAME), str(nisqa_arch.NISQA_V2_WEIGHTS_SHA256)

    source_path = _PACKAGE_ROOT / "guard" / "nisqa_arch.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id in {"NISQA_WEIGHTS_FILENAME", "NISQA_V2_WEIGHTS_SHA256"}
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                found[target.id] = node.value.value
    if "NISQA_WEIGHTS_FILENAME" not in found or "NISQA_V2_WEIGHTS_SHA256" not in found:
        raise ModelManagerError("Could not read NISQA v2 pinned digest from nisqa_arch.py")
    return found["NISQA_WEIGHTS_FILENAME"], found["NISQA_V2_WEIGHTS_SHA256"]


def _nisqa_discovered_path(models_root: Path | None) -> Path:
    filename, _ = _load_nisqa_constants()
    if models_root is not None:
        return models_root / "nisqa" / filename
    override = os.environ.get("HAWAVOCLEAN_NISQA_WEIGHTS")
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("HAWAVOCLEAN_MODEL_DIR"):
        return models_dir() / "nisqa" / filename
    home = Path.home()
    candidates = (
        app_data_root() / "models" / "nisqa" / filename,
        models_dir() / "nisqa" / filename,
        _REPO_ROOT / ".local-ai" / "models" / "nisqa" / filename,
        home / ".torchmetrics" / "NISQA" / filename,
        home / ".cache" / "torchmetrics" / "NISQA" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _default_models_root(models_root: Path | None) -> Path:
    return models_root.resolve() if models_root is not None else app_data_root() / "models"


def model_registry(*, models_root: Path | None = None) -> dict[str, ModelSpec]:
    """Build the allowlisted HawaVoClean neural model registry."""
    root = _default_models_root(models_root)
    dfn_license, dfn_primary_sha, dfn_artifacts = _load_deepfilternet3_lock()
    nisqa_filename, nisqa_sha = _load_nisqa_constants()

    specs = (
        ModelSpec(
            model_id="deepfilternet3",
            display_name="DeepFilterNet3",
            description="Bundled real-time 48 kHz neural speech denoiser for Studio and Lowband profiles.",
            used_by=("studio", "lowband"),
            license=dfn_license,
            expected_sha256=dfn_primary_sha,
            approx_size_mb=8.3,
            bundled=True,
            target_path=_PACKAGE_ROOT / "resources" / "models" / "deepfilternet3",
            download_urls=(),
            artifacts=dfn_artifacts,
        ),
        ModelSpec(
            model_id="mossformer2_se_48k",
            display_name="MossFormer2 SE 48k",
            description="Flagship 48 kHz studio dialogue denoiser for heavy background noise (Studio Pro).",
            used_by=("studio_pro",),
            license="Apache-2.0",
            expected_sha256=STUDIO_PRO_CHECKPOINT_SHA256,
            approx_size_mb=211.0,
            bundled=False,
            target_path=root / "mossformer2_se_48k" / STUDIO_PRO_CHECKPOINT_FILE,
            download_urls=(
                validate_download_url(
                    "https://huggingface.co/alibabasglab/MossFormer2_SE_48K/resolve/main/"
                    f"{STUDIO_PRO_CHECKPOINT_FILE}"
                ),
            ),
            artifacts=(
                ModelArtifact(
                    filename=STUDIO_PRO_CHECKPOINT_FILE,
                    expected_sha256=STUDIO_PRO_CHECKPOINT_SHA256,
                    download_urls=(
                        validate_download_url(
                            "https://huggingface.co/alibabasglab/MossFormer2_SE_48K/resolve/main/"
                            f"{STUDIO_PRO_CHECKPOINT_FILE}"
                        ),
                    ),
                ),
            ),
        ),
        ModelSpec(
            model_id="tfgridnet_dereverb",
            display_name="TF-GridNetV3 Dereverb",
            description="Pure discriminative room-echo and reverberation remover (Dereverb profile).",
            used_by=("dereverb",),
            license=f"{DEREVERB_MODEL_LICENSE} / {DEREVERB_ARCH_LICENSE}",
            expected_sha256=DEREVERB_CHECKPOINT_SHA256,
            approx_size_mb=88.5,
            bundled=False,
            target_path=root / "gap_urgenet_predictor" / DEREVERB_CHECKPOINT_FILE,
            download_urls=(
                validate_download_url(
                    "https://huggingface.co/Xiaobin-Rong/gap-urgenet/resolve/main/"
                    f"{DEREVERB_CHECKPOINT_FILE}"
                ),
            ),
            artifacts=(
                ModelArtifact(
                    filename=DEREVERB_CHECKPOINT_FILE,
                    expected_sha256=DEREVERB_CHECKPOINT_SHA256,
                    download_urls=(
                        validate_download_url(
                            "https://huggingface.co/Xiaobin-Rong/gap-urgenet/resolve/main/"
                            f"{DEREVERB_CHECKPOINT_FILE}"
                        ),
                    ),
                ),
            ),
        ),
        ModelSpec(
            model_id="nisqa_v2",
            display_name="NISQA v2 Perceptual MOS",
            description="Non-intrusive 5-axis perceptual speech quality scorer for Smart Safe previews.",
            used_by=("smart_safe",),
            license="MIT",
            expected_sha256=nisqa_sha,
            approx_size_mb=1.0,
            bundled=False,
            target_path=root / "nisqa" / nisqa_filename,
            download_urls=(
                validate_download_url(
                    f"https://github.com/gabrielmittag/NISQA/raw/master/weights/{nisqa_filename}"
                ),
            ),
            artifacts=(
                ModelArtifact(
                    filename=nisqa_filename,
                    expected_sha256=nisqa_sha,
                    download_urls=(
                        validate_download_url(
                            f"https://github.com/gabrielmittag/NISQA/raw/master/weights/{nisqa_filename}"
                        ),
                    ),
                ),
            ),
        ),
        ModelSpec(
            model_id="lavasr_v2",
            display_name="LavaSR v2 Bandwidth Restoration",
            description="Guarded 16 kHz to 48 kHz high-frequency restoration for narrowband voice.",
            used_by=("restore_source",),
            license=LAVASR_MODEL_LICENSE,
            expected_sha256=LAVASR_V2_WEIGHTS_SHA256,
            approx_size_mb=54.0,
            bundled=False,
            target_path=root / "lavasr" / "enhancer_v2",
            download_urls=(
                validate_download_url(
                    "https://huggingface.co/YatharthS/LavaSR/resolve/main/enhancer_v2/"
                    f"{LAVASR_V2_CONFIG_FILE}"
                ),
                validate_download_url(
                    "https://huggingface.co/YatharthS/LavaSR/resolve/main/enhancer_v2/"
                    f"{LAVASR_V2_WEIGHTS_FILE}"
                ),
            ),
            artifacts=(
                ModelArtifact(
                    filename=LAVASR_V2_CONFIG_FILE,
                    expected_sha256=LAVASR_V2_CONFIG_SHA256,
                    download_urls=(
                        validate_download_url(
                            "https://huggingface.co/YatharthS/LavaSR/resolve/main/enhancer_v2/"
                            f"{LAVASR_V2_CONFIG_FILE}"
                        ),
                    ),
                ),
                ModelArtifact(
                    filename=LAVASR_V2_WEIGHTS_FILE,
                    expected_sha256=LAVASR_V2_WEIGHTS_SHA256,
                    download_urls=(
                        validate_download_url(
                            "https://huggingface.co/YatharthS/LavaSR/resolve/main/enhancer_v2/"
                            f"{LAVASR_V2_WEIGHTS_FILE}"
                        ),
                    ),
                ),
            ),
        ),
        ModelSpec(
            model_id="roformer",
            display_name="Mel-Band RoFormer Voice Separation",
            description="High-precision vocal and background music separation for speech isolation.",
            used_by=("separate",),
            license=ROFORMER_LICENSE,
            expected_sha256=ROFORMER_MODEL_SHA256,
            approx_size_mb=961.0,
            bundled=False,
            target_path=root / "roformer",
            download_urls=(
                validate_download_url(
                    "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
                    f"mdx_model_data/mdx_c_configs/{ROFORMER_CONFIG_FILENAME}"
                ),
                validate_download_url(
                    "https://github.com/TRvlvr/model_repo/releases/download/"
                    f"all_public_uvr_models/{ROFORMER_MODEL_FILENAME}"
                ),
            ),
            artifacts=(
                ModelArtifact(
                    filename=ROFORMER_CONFIG_FILENAME,
                    expected_sha256=ROFORMER_CONFIG_SHA256,
                    download_urls=(
                        validate_download_url(
                            "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
                            f"mdx_model_data/mdx_c_configs/{ROFORMER_CONFIG_FILENAME}"
                        ),
                    ),
                ),
                ModelArtifact(
                    filename=ROFORMER_MODEL_FILENAME,
                    expected_sha256=ROFORMER_MODEL_SHA256,
                    download_urls=(
                        validate_download_url(
                            "https://github.com/TRvlvr/model_repo/releases/download/"
                            f"all_public_uvr_models/{ROFORMER_MODEL_FILENAME}"
                        ),
                    ),
                ),
            ),
        ),
    )
    return {spec.model_id: spec for spec in specs}


def get_model_spec(model_id: str, *, models_root: Path | None = None) -> ModelSpec:
    """Return the specification for an allowlisted ``model_id`` or raise :class:`UnknownModelError`."""
    registry = model_registry(models_root=models_root)
    if not isinstance(model_id, str) or model_id not in registry:
        raise UnknownModelError(f"Unknown model_id: {model_id!r}")
    return registry[model_id]


def _artifact_destination(spec: ModelSpec, artifact: ModelArtifact) -> Path:
    if len(spec.artifacts) == 1 and spec.target_path.name == artifact.filename:
        return spec.target_path
    return spec.target_path / artifact.filename


def _discover_base_dir(spec: ModelSpec, *, models_root: Path | None) -> Path:
    """Find the active local base directory for a model without network I/O."""
    if spec.model_id == "deepfilternet3":
        pkg_dir = models_dir() / "deepfilternet3"
        return pkg_dir if pkg_dir.is_dir() else spec.target_path

    if models_root is not None:
        return spec.target_path.parent if len(spec.artifacts) == 1 else spec.target_path

    # Check app_data_root()/models first, then fall back to each core's resolver.
    target_base = spec.target_path.parent if len(spec.artifacts) == 1 else spec.target_path
    if all((target_base / art.filename).is_file() for art in spec.artifacts):
        return target_base

    if spec.model_id == "mossformer2_se_48k":
        return studio_pro_checkpoint_path().parent
    if spec.model_id == "tfgridnet_dereverb":
        return dereverb_checkpoint_path().parent
    if spec.model_id == "nisqa_v2":
        return _nisqa_discovered_path(None).parent
    if spec.model_id == "lavasr_v2":
        return lavasr_model_dir()
    if spec.model_id == "roformer":
        with contextlib.suppress(Exception):
            roformer_dir = _roformer_model_dir()
            if (roformer_dir / ROFORMER_MODEL_FILENAME).is_file():
                return roformer_dir
    return target_base


def _inspect_spec(spec: ModelSpec, *, models_root: Path | None) -> ModelStatus:
    base_dir = _discover_base_dir(spec, models_root=models_root)
    present_count = 0
    verified_count = 0
    total_size = 0

    for artifact in spec.artifacts:
        file_path = base_dir / artifact.filename
        try:
            if not file_path.is_file():
                continue
            total_size += int(file_path.stat().st_size)
            present_count += 1
            actual_sha = hash_file(file_path)
            if hmac.compare_digest(actual_sha, artifact.expected_sha256):
                verified_count += 1
        except OSError:
            continue

    installed = present_count == len(spec.artifacts) and len(spec.artifacts) > 0
    verified = installed and verified_count == len(spec.artifacts)
    downloadable = not spec.bundled and len(spec.download_urls) > 0
    if installed:
        if len(spec.artifacts) == 1:
            resolved_path: str | None = str((base_dir / spec.artifacts[0].filename).resolve())
        else:
            resolved_path = str(base_dir.resolve())
    else:
        resolved_path = None

    return ModelStatus(
        model_id=spec.model_id,
        display_name=spec.display_name,
        description=spec.description,
        used_by=spec.used_by,
        license=spec.license,
        expected_sha256=spec.expected_sha256,
        approx_size_mb=spec.approx_size_mb,
        bundled=spec.bundled,
        downloadable=downloadable,
        installed=installed,
        verified=verified,
        size_bytes=total_size,
        path=resolved_path,
    )


def list_models(*, models_root: Path | None = None) -> list[ModelStatus]:
    """Inspect local install and verification status for all allowlisted models."""
    registry = model_registry(models_root=models_root)
    return [_inspect_spec(spec, models_root=models_root) for spec in registry.values()]


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only to strictly validated HTTPS URLs."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _https_stream_fetcher(url: str) -> tuple[Iterator[bytes], int | None]:
    """Stream bytes from a strictly validated HTTPS URL."""
    safe_url = validate_download_url(url)
    request = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "HawaVoClean-ModelManager/1.0"},
        method="GET",
    )
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler())
    try:
        response = opener.open(request, timeout=_DOWNLOAD_TIMEOUT_S)
    except (OSError, urllib.error.URLError) as exc:
        raise ModelDownloadError(f"Could not download model weights: {exc}") from exc

    raw_length = response.headers.get("Content-Length") if response.headers is not None else None
    total_bytes: int | None = None
    if raw_length is not None and raw_length.strip().isdigit():
        parsed_len = int(raw_length.strip())
        if parsed_len > 0:
            total_bytes = parsed_len

    def _chunks() -> Iterator[bytes]:
        try:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
        except (OSError, urllib.error.URLError) as exc:
            raise ModelDownloadError(f"Interrupted while downloading model weights: {exc}") from exc
        finally:
            with contextlib.suppress(Exception):
                response.close()

    return _chunks(), total_bytes


def _normalize_fetch_stream(result: FetchStream) -> tuple[Iterable[bytes], int | None]:
    if isinstance(result, (bytes, bytearray)):
        data = bytes(result)
        return (data,), len(data)
    if isinstance(result, tuple) and len(result) == 2:
        stream_part, total_part = result
        total_int = int(total_part) if isinstance(total_part, int) and total_part > 0 else None
        if isinstance(stream_part, (bytes, bytearray)):
            data = bytes(stream_part)
            return (data,), total_int or len(data)
        return stream_part, total_int
    if hasattr(result, "read"):
        reader: Any = result

        def _read_iter() -> Iterator[bytes]:
            while True:
                chunk = reader.read(_DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                yield bytes(chunk)

        return _read_iter(), None
    return result, None


def _model_lock(model_id: str) -> threading.Lock:
    with _INSTALL_LOCKS_GUARD:
        lock = _INSTALL_LOCKS.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _INSTALL_LOCKS[model_id] = lock
        return lock


def _stream_and_verify_artifact(
    artifact: ModelArtifact,
    dest_path: Path,
    *,
    fetcher: Fetcher,
    max_bytes: int,
    progress_cb: ProgressCallback | None,
    bytes_before: int,
    total_hint: int | None,
) -> tuple[Path, str, int]:
    """Download one artifact to a ``.part`` file and verify its SHA-256 digest."""
    if not artifact.download_urls:
        raise ModelNotDownloadableError(f"Artifact {artifact.filename} has no download URL.")
    url = validate_download_url(artifact.download_urls[0])
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.parent / f".{dest_path.name}.{secrets.token_hex(6)}.part"

    digest = hashlib.sha256()
    written = 0
    try:
        chunks, stream_total = _normalize_fetch_stream(fetcher(url))
        effective_total = total_hint
        if effective_total is None and stream_total is not None:
            effective_total = bytes_before + stream_total

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
        fd = os.open(part_path, flags, 0o600)
        with os.fdopen(fd, "wb") as out:
            for raw_chunk in chunks:
                chunk = bytes(raw_chunk)
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise ModelDownloadError(
                        f"Downloaded artifact {artifact.filename} exceeded maximum allowed size."
                    )
                out.write(chunk)
                digest.update(chunk)
                if progress_cb is not None:
                    progress_cb(bytes_before + written, effective_total)
            out.flush()
            with contextlib.suppress(OSError):
                os.fsync(out.fileno())

        actual_sha = digest.hexdigest()
        if not hmac.compare_digest(actual_sha, artifact.expected_sha256):
            raise ModelHashMismatchError(
                f"SHA-256 mismatch for {artifact.filename}: "
                f"expected {artifact.expected_sha256[:12]}..., got {actual_sha[:12]}..."
            )
        return part_path, actual_sha, written
    except BaseException:
        with contextlib.suppress(OSError):
            part_path.unlink(missing_ok=True)
        raise


def install_model(
    model_id: str,
    *,
    progress_cb: ProgressCallback | None = None,
    fetcher: Fetcher | None = None,
    models_root: Path | None = None,
    force: bool = False,
) -> ModelInstallResult:
    """Download, SHA-256 verify, and atomically install one allowlisted neural model.

    Args:
        model_id: Allowlisted model identifier from :func:`model_registry`.
        progress_cb: Optional callback ``(bytes_downloaded, total_bytes_or_none)``.
        fetcher: Optional injectable transport ``fetcher(url) -> Iterable[bytes]``
            for deterministic offline testing. Defaults to strict HTTPS streaming.
        models_root: Optional override for the writable models directory
            (defaults to ``app_data_root() / "models"``).
        force: Re-download and re-verify even if the target files are already present.
    """
    spec = get_model_spec(model_id, models_root=models_root)

    with _model_lock(spec.model_id):
        if spec.bundled:
            status = _inspect_spec(spec, models_root=models_root)
            if status.installed and status.verified and status.path is not None:
                return ModelInstallResult(
                    model_id=spec.model_id,
                    installed=True,
                    verified=True,
                    sha256=spec.expected_sha256,
                    size_bytes=status.size_bytes,
                    path=status.path,
                    already_installed=True,
                    message=f"{spec.display_name} is bundled and verified.",
                )
            raise ModelNotDownloadableError(
                f"{spec.display_name} is bundled with HawaVoClean and cannot be downloaded separately."
            )

        if not spec.download_urls:
            raise ModelNotDownloadableError(
                f"{spec.display_name} does not have a configured HTTPS download URL."
            )

        # Skip download only if target files under models_root are already verified
        # and caller did not pass a custom fetcher or force=True.
        if fetcher is None and not force:
            target_destinations = [_artifact_destination(spec, art) for art in spec.artifacts]
            if all(dest.is_file() for dest in target_destinations) and all(
                hmac.compare_digest(hash_file(dest), art.expected_sha256)
                for dest, art in zip(target_destinations, spec.artifacts, strict=True)
            ):
                total_size = sum(dest.stat().st_size for dest in target_destinations)
                return ModelInstallResult(
                    model_id=spec.model_id,
                    installed=True,
                    verified=True,
                    sha256=spec.expected_sha256,
                    size_bytes=total_size,
                    path=str(spec.target_path.resolve()),
                    already_installed=True,
                    message=f"{spec.display_name} is already installed and verified.",
                )

        active_fetcher = _https_stream_fetcher if fetcher is None else fetcher
        # Generous upper bound (3x approx size, minimum 64 MiB) to stop runaway streams.
        max_bytes = max(int(spec.approx_size_mb * 1024 * 1024 * 3), 64 * 1024 * 1024)
        approx_total_hint = (
            int(spec.approx_size_mb * 1024 * 1024) if len(spec.artifacts) > 1 else None
        )

        staged: list[tuple[Path, Path, str, int]] = []
        bytes_so_far = 0
        try:
            for artifact in spec.artifacts:
                dest_path = _artifact_destination(spec, artifact)
                part_path, actual_sha, written = _stream_and_verify_artifact(
                    artifact,
                    dest_path,
                    fetcher=active_fetcher,
                    max_bytes=max_bytes,
                    progress_cb=progress_cb,
                    bytes_before=bytes_so_far,
                    total_hint=approx_total_hint,
                )
                staged.append((part_path, dest_path, actual_sha, written))
                bytes_so_far += written

            for part_path, dest_path, _sha, _written in staged:
                os.replace(part_path, dest_path)
        except BaseException:
            for part_path, _dest, _sha, _written in staged:
                with contextlib.suppress(OSError):
                    part_path.unlink(missing_ok=True)
            raise

        return ModelInstallResult(
            model_id=spec.model_id,
            installed=True,
            verified=True,
            sha256=spec.expected_sha256,
            size_bytes=bytes_so_far,
            path=str(spec.target_path.resolve()),
            already_installed=False,
            message=f"{spec.display_name} downloaded and SHA-256 verified.",
        )


__all__ = [
    "Fetcher",
    "ModelArtifact",
    "ModelDownloadError",
    "ModelHashMismatchError",
    "ModelInstallResult",
    "ModelManagerError",
    "ModelNotDownloadableError",
    "ModelSpec",
    "ModelStatus",
    "ProgressCallback",
    "ROFORMER_CONFIG_SHA256",
    "ROFORMER_MODEL_SHA256",
    "UnknownModelError",
    "get_model_spec",
    "install_model",
    "list_models",
    "model_registry",
    "validate_download_url",
]
