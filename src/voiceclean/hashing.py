"""Cryptographic hashing utilities for files, audio arrays, configurations, and cache keys."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file using stdlib file_digest."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hash_numpy(arr: np.ndarray[Any, Any]) -> str:
    """Compute SHA-256 hex digest of a contiguous numpy array."""
    contiguous = np.ascontiguousarray(arr)
    return hash_bytes(contiguous.tobytes())


def hash_json_canonical(obj: Any) -> str:
    """Compute SHA-256 hex digest of a JSON-serializable object with sorted keys."""
    canonical_json = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hash_bytes(canonical_json.encode("utf-8"))


def compute_job_id(
    input_hash: str,
    config_hash: str,
    core_hash: str,
    guard_hash: str,
    tool_version: str,
) -> str:
    """Derive unique, deterministic job ID as specified in BLUEPRINT.md section 17.1."""
    composite = f"{input_hash}:{config_hash}:{core_hash}:{guard_hash}:{tool_version}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]


def compute_cache_key(
    unit_pcm_bytes: bytes,
    sample_rate: int,
    model_hashes: dict[str, str],
    guard_hash: str,
    config_hash: str,
    tool_version: str,
) -> str:
    """Derive unit enhancement cache key as specified in BLUEPRINT.md section 17.3."""
    h = hashlib.sha256()
    h.update(unit_pcm_bytes)
    h.update(str(sample_rate).encode("utf-8"))
    canonical_models = json.dumps(model_hashes, sort_keys=True, separators=(",", ":"))
    h.update(canonical_models.encode("utf-8"))
    h.update(guard_hash.encode("utf-8"))
    h.update(config_hash.encode("utf-8"))
    h.update(tool_version.encode("utf-8"))
    return h.hexdigest()
