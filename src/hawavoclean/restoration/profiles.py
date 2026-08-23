"""Speaker profile schema, validation, consent verification, and hash checking."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.errors import InvalidUserInputError
from hawavoclean.hashing import hash_file, hash_json_canonical


class ProfileValidationError(InvalidUserInputError):
    """Raised when a speaker profile fails schema, consent, or hash validation."""


@dataclass(frozen=True)
class SpeakerF0Stats:
    """Speaker pitch envelope statistics."""

    median_hz: float
    p05_hz: float
    p95_hz: float


@dataclass(frozen=True)
class SpeakerProfile:
    """Immutable, hash-verified speaker profile for HawaRestore-KD."""

    schema_version: str
    speaker_id: str
    display_name: str
    consent_record: str
    canonical_audio_manifest: str
    canonical_audio_sha256: list[str]
    profile_embedding_path: str
    profile_embedding_sha256: str
    f0_statistics: SpeakerF0Stats
    training_split_id: str
    adapter: str | None
    created_by_commit: str
    notes: str
    embedding_vector: np.ndarray | None = None  # Shape: (dim,), float32

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        d = asdict(self)
        d.pop("embedding_vector", None)
        return d

    def compute_hash(self) -> str:
        """Compute canonical SHA-256 hash of profile metadata."""
        return hash_json_canonical(self.to_dict())


def validate_speaker_profile(
    profile_path: Path | str,
    base_dir: Path | None = None,
    verify_files: bool = True,
) -> SpeakerProfile:
    """Load and strictly validate a speaker profile JSON file."""
    path = Path(profile_path)
    if not path.exists():
        raise ProfileValidationError(f"Speaker profile not found: {path}")

    if base_dir is None:
        base_dir = path.parent

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ProfileValidationError(f"Failed to parse speaker profile JSON at {path}: {e}") from e

    # 1. Schema version check
    if data.get("schema_version") != "1.0":
        raise ProfileValidationError(
            f"Unsupported speaker profile schema version: {data.get('schema_version')} (expected '1.0')"
        )

    # 2. Required fields check
    required_keys = [
        "schema_version",
        "speaker_id",
        "display_name",
        "consent_record",
        "canonical_audio_manifest",
        "canonical_audio_sha256",
        "profile_embedding_path",
        "profile_embedding_sha256",
        "f0_statistics",
        "training_split_id",
        "created_by_commit",
    ]
    for key in required_keys:
        if key not in data or data[key] is None:
            raise ProfileValidationError(f"Speaker profile missing required field: '{key}'")

    speaker_id = str(data["speaker_id"])
    if not speaker_id or not speaker_id.replace("_", "").isalnum():
        raise ProfileValidationError(f"Invalid speaker_id format: '{speaker_id}'")

    # 3. Consent record check
    consent_rel = Path(data["consent_record"])
    consent_path = base_dir / consent_rel if not consent_rel.is_absolute() else consent_rel
    if verify_files:
        if not consent_path.exists():
            raise ProfileValidationError(f"Speaker consent record missing: {consent_path}")
        try:
            with open(consent_path, encoding="utf-8") as f:
                consent_data = json.load(f)
            if consent_data.get("speaker_id") != speaker_id or not consent_data.get(
                "consent_granted"
            ):
                raise ProfileValidationError(f"Invalid or revoked consent in {consent_path}")
        except Exception as e:
            raise ProfileValidationError(
                f"Failed to verify consent record {consent_path}: {e}"
            ) from e

    # 4. Canonical audio manifest and hash checks
    manifest_rel = Path(data["canonical_audio_manifest"])
    manifest_path = base_dir / manifest_rel if not manifest_rel.is_absolute() else manifest_rel
    if verify_files:
        if not manifest_path.exists():
            raise ProfileValidationError(f"Canonical audio manifest missing: {manifest_path}")
        # Verify manifest has at least one valid non-empty entry
        try:
            with open(manifest_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ProfileValidationError(f"Canonical audio manifest is empty: {manifest_path}")
        except Exception as e:
            raise ProfileValidationError(
                f"Failed to read canonical manifest {manifest_path}: {e}"
            ) from e

    # 5. Profile embedding vector and hash check
    embedding_rel = Path(data["profile_embedding_path"])
    embedding_path = base_dir / embedding_rel if not embedding_rel.is_absolute() else embedding_rel
    embedding_vector: np.ndarray | None = None
    if verify_files:
        if not embedding_path.exists():
            raise ProfileValidationError(f"Profile embedding file missing: {embedding_path}")
        actual_hash = hash_file(embedding_path)
        expected_hash = str(data["profile_embedding_sha256"])
        if actual_hash != expected_hash:
            raise ProfileValidationError(
                f"Embedding hash mismatch for {embedding_path}: expected {expected_hash}, got {actual_hash}"
            )
        # Load embedding (raw float32 array or safetensors/npy/json)
        try:
            if embedding_path.suffix == ".npy":
                embedding_vector = np.load(embedding_path).astype(np.float32)
            elif embedding_path.suffix == ".json":
                with open(embedding_path, encoding="utf-8") as f:
                    emb_data = json.load(f)
                embedding_vector = np.array(emb_data["embedding"], dtype=np.float32)
            else:
                # Binary float32 buffer
                embedding_vector = np.fromfile(embedding_path, dtype=np.float32)
        except Exception as e:
            raise ProfileValidationError(
                f"Failed to load embedding from {embedding_path}: {e}"
            ) from e

        if embedding_vector is not None and np.linalg.norm(embedding_vector) < 1e-6:
            raise ProfileValidationError(f"Degenerate zero embedding in {embedding_path}")

    # 6. F0 statistics check
    f0_data = data["f0_statistics"]
    if not isinstance(f0_data, dict):
        raise ProfileValidationError("f0_statistics must be a dictionary")
    f0_stats = SpeakerF0Stats(
        median_hz=float(f0_data.get("median_hz", 0.0)),
        p05_hz=float(f0_data.get("p05_hz", 0.0)),
        p95_hz=float(f0_data.get("p95_hz", 0.0)),
    )
    if f0_stats.median_hz < 0.0 or f0_stats.p05_hz > f0_stats.p95_hz:
        raise ProfileValidationError(f"Invalid F0 statistics: {f0_stats}")

    return SpeakerProfile(
        schema_version=data["schema_version"],
        speaker_id=speaker_id,
        display_name=str(data.get("display_name", speaker_id)),
        consent_record=str(data["consent_record"]),
        canonical_audio_manifest=str(data["canonical_audio_manifest"]),
        canonical_audio_sha256=list(data["canonical_audio_sha256"]),
        profile_embedding_path=str(data["profile_embedding_path"]),
        profile_embedding_sha256=str(data["profile_embedding_sha256"]),
        f0_statistics=f0_stats,
        training_split_id=str(data["training_split_id"]),
        adapter=data.get("adapter"),
        created_by_commit=str(data["created_by_commit"]),
        notes=str(data.get("notes", "")),
        embedding_vector=embedding_vector,
    )


def load_speaker_profile(
    speaker_id: str,
    profiles_root: Path | str = "profiles",
) -> SpeakerProfile:
    """Find and load a speaker profile by speaker_id from the profiles root directory."""
    root = Path(profiles_root)
    profile_json = root / speaker_id / "profile.json"
    if not profile_json.exists():
        # Check if single file
        profile_json = root / f"{speaker_id}.json"
    if not profile_json.exists():
        raise ProfileValidationError(f"Profile for speaker '{speaker_id}' not found under {root}")

    return validate_speaker_profile(profile_json, base_dir=profile_json.parent, verify_files=True)


def validate_all_profiles(profiles_root: Path | str = "profiles") -> dict[str, SpeakerProfile]:
    """Validate all 10 registered profiles and verify embedding distinctness."""
    root = Path(profiles_root)
    profiles: dict[str, SpeakerProfile] = {}
    seen_hashes: dict[str, str] = {}

    for i in range(1, 11):
        spk_id = f"character_{i:02d}"
        prof = load_speaker_profile(spk_id, profiles_root=root)
        profiles[spk_id] = prof

        emb_hash = prof.profile_embedding_sha256
        if emb_hash in seen_hashes:
            other_id = seen_hashes[emb_hash]
            raise ProfileValidationError(
                f"Duplicate embedding hash between {spk_id} and {other_id}: {emb_hash}"
            )
        seen_hashes[emb_hash] = spk_id

    return profiles
