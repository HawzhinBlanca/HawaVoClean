"""Unit tests for speaker profile loading, schema validation, and consent verification."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hawavoclean.errors import InvalidUserInputError
from hawavoclean.restoration.profiles import (
    ProfileValidationError,
    SpeakerProfile,
    load_speaker_profile,
    validate_speaker_profile,
)


def test_load_all_ten_speaker_profiles() -> None:
    """Verify that all 10 canonical speaker profiles load cleanly and pass schema checks."""
    profiles_root = Path("profiles")
    assert profiles_root.exists(), "profiles directory must exist"

    for i in range(1, 11):
        spk_id = f"character_{i:02d}"
        profile = load_speaker_profile(spk_id, profiles_root=profiles_root)
        assert isinstance(profile, SpeakerProfile)
        assert profile.speaker_id == spk_id
        assert profile.consent_record == "consent/consent.json"
        assert profile.embedding_vector is not None
        assert len(profile.embedding_vector) == 192
        assert profile.f0_statistics.median_hz > 50.0
        assert len(profile.canonical_audio_sha256) >= 1


def test_validate_speaker_profile_missing_file(tmp_path: Path) -> None:
    """Test validation failure when profile.json does not exist."""
    with pytest.raises(InvalidUserInputError, match="not found"):
        validate_speaker_profile(tmp_path / "missing.json")


def test_validate_speaker_profile_corrupted_consent(tmp_path: Path) -> None:
    """Test validation failure when consent verification flag is false or missing."""
    prof_data = {
        "schema_version": "1.0",
        "speaker_id": "test_bad_spk",
        "display_name": "Test Speaker",
        "gender": "male",
        "language_dialect": "ckb",
        "profile_embedding_path": "embedding.npy",
        "canonical_audio_manifest": "manifest.jsonl",
        "canonical_audio_sha256": [
            "0000000000000000000000000000000000000000000000000000000000000000"
        ],
        "consent_record": "consent.json",
        "f0_statistics": {"median_hz": 120.0, "p05_hz": 80.0, "p95_hz": 180.0},
        "training_split_id": "split_01",
        "adapter": None,
        "created_by_commit": "test",
        "notes": "",
    }

    np.save(tmp_path / "embedding.npy", np.zeros(256, dtype=np.float32))
    prof_data["profile_embedding_sha256"] = "dummy"

    with open(tmp_path / "manifest.jsonl", "w") as f:
        f.write('{"path": "dummy.wav"}\n')

    # Consent explicitly not authorized
    with open(tmp_path / "consent.json", "w") as f:
        json.dump({"speaker_id": "test_bad_spk", "consent_verified": False}, f)

    with open(tmp_path / "profile.json", "w") as f:
        json.dump(prof_data, f)

    with pytest.raises(InvalidUserInputError, match="Invalid or revoked consent"):
        validate_speaker_profile(tmp_path / "profile.json", base_dir=tmp_path)


def test_a_profile_may_not_be_looked_up_under_another_speakers_name(tmp_path: Path) -> None:
    """The lookup key and the profile's declared identity must be one person.

    They were not required to agree, so a directory named ``character_09``
    holding ``character_01``'s profile loaded without complaint. The report
    then attributed the reconstruction to character_09, while the consent
    record -- which is validated against the DECLARED id -- belonged to
    character_01. A reconstruction credited to someone who never agreed to it
    is the exact failure this project's restore mode is gated on avoiding.
    """
    genuine = Path("profiles") / "character_01"
    shutil.copytree(genuine, tmp_path / "character_09")

    with pytest.raises(ProfileValidationError, match="must agree"):
        load_speaker_profile("character_09", profiles_root=tmp_path)

    # The honest lookup is untouched.
    assert load_speaker_profile("character_01", profiles_root=Path("profiles")).speaker_id == (
        "character_01"
    )
