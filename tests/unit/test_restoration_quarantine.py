"""Unit tests for restoration research quarantine and profile contract versioning (R2.1, R2.6)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.cli import main
from hawavoclean.errors import ExitCode, InvalidUserInputError
from hawavoclean.pipeline import run_pipeline
from hawavoclean.restoration.profiles import (
    ProfileValidationError,
    load_speaker_profile,
    revoke_speaker_profile,
    validate_speaker_profile,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_PROFILES = _REPO_ROOT / "profiles"


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    import sys

    monkeypatch.setattr(sys, "argv", ["hawavoclean", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return int(exc.value.code or 0)


@pytest.fixture
def synthetic_wav_48k(tmp_path: Path) -> Path:
    sr = 48000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False, dtype=np.float32)
    sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test_input.wav"
    sf.write(str(wav_path), sig, sr, format="WAV", subtype="PCM_16")
    return wav_path


def test_production_cli_rejects_restore_mode_without_research_flag(
    synthetic_wav_48k: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Production CLI process command must fail closed if mode=restore is used with production profile."""
    out_wav = tmp_path / "out_prod.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(synthetic_wav_48k),
        "-o",
        str(out_wav),
        "--profile",
        "production",
        "--mode",
        "restore",
        "--speaker-id",
        "kurdish_fatih",
    )
    assert code == int(ExitCode.INVALID_USER_INPUT)
    err = capsys.readouterr().err
    assert "Production restoration capability is BLOCKED for profile 'production'" in err
    assert not out_wav.exists()


def test_production_cli_rejects_studio_restore_mode_without_research_flag(
    synthetic_wav_48k: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Production CLI process command must fail closed if mode=restore is used with studio profile."""
    out_wav = tmp_path / "out_studio.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(synthetic_wav_48k),
        "-o",
        str(out_wav),
        "--profile",
        "studio",
        "--mode",
        "restore",
        "--speaker-id",
        "kurdish_fatih",
    )
    assert code == int(ExitCode.INVALID_USER_INPUT)
    err = capsys.readouterr().err
    assert "Production restoration capability is BLOCKED for profile 'studio'" in err
    assert not out_wav.exists()


def test_production_batch_rejects_restore_mode_without_research_flag(
    synthetic_wav_48k: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Production CLI batch command must fail closed if mode=restore is used with production profile."""
    out_dir = tmp_path / "batch_out"
    code = _run_cli(
        monkeypatch,
        "batch",
        str(synthetic_wav_48k),
        "-o",
        str(out_dir),
        "--profile",
        "production",
        "--mode",
        "restore",
        "--speaker-id",
        "kurdish_fatih",
    )
    assert code == int(ExitCode.INVALID_USER_INPUT)
    err = capsys.readouterr().err
    assert "Production restoration capability is BLOCKED for profile 'production'" in err
    assert not list(out_dir.glob("*.wav"))


def test_pipeline_rejects_production_profile_restore_without_allow_flag(
    synthetic_wav_48k: Path, tmp_path: Path
) -> None:
    """Library run_pipeline rejects restore mode in production profile."""
    out_wav = tmp_path / "out.wav"
    with pytest.raises(InvalidUserInputError, match="Production restoration capability is BLOCKED"):
        run_pipeline(
            input_path=synthetic_wav_48k,
            output_path=out_wav,
            profile="production",
            mode="restore",
            speaker_id="character_01",
            profiles_dir=_REPO_PROFILES,
        )
    assert not out_wav.exists()


def test_pipeline_accepts_development_profile_and_records_quarantine_metadata(
    synthetic_wav_48k: Path, tmp_path: Path
) -> None:
    """Development profile allows research restoration and report records quarantine flags."""
    out_wav = tmp_path / "out_dev.wav"
    report = run_pipeline(
        input_path=synthetic_wav_48k,
        output_path=out_wav,
        profile="development",
        mode="restore",
        speaker_id="character_01",
        profiles_dir=_REPO_PROFILES,
        overwrite=True,
    )
    assert out_wav.is_file()
    assert report.restoration is not None
    restorer_info = report.restoration.get("restorer", {})
    assert restorer_info.get("research_quarantine") is True
    assert restorer_info.get("production_qualified") is False


def test_pipeline_accepts_allow_research_restore_flag_in_production(
    synthetic_wav_48k: Path, tmp_path: Path
) -> None:
    """Setting allow_research_restore=True permits research execution even in production profile."""
    out_wav = tmp_path / "out_prod_allowed.wav"
    report = run_pipeline(
        input_path=synthetic_wav_48k,
        output_path=out_wav,
        profile="production",
        mode="restore",
        speaker_id="character_01",
        profiles_dir=_REPO_PROFILES,
        overwrite=True,
        allow_research_restore=True,
    )
    assert out_wav.is_file()
    assert report.restoration is not None
    restorer_info = report.restoration.get("restorer", {})
    assert restorer_info.get("research_quarantine") is True
    assert restorer_info.get("production_qualified") is False


def test_restore_doctor_declares_research_quarantine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """restore-doctor must explicitly state that Restore is NOT production qualified."""
    code = _run_cli(monkeypatch, "restore-doctor")
    assert code == int(ExitCode.SUCCESS)
    out = capsys.readouterr().out
    assert "Production capability: Restore is NOT production qualified (blocked)" in out
    assert "RESEARCH-ONLY: quarantined prototype, not production qualified" in out


def test_doctor_declares_production_capabilities(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """System doctor must report production capability status and quarantine boundaries."""
    code = _run_cli(monkeypatch, "doctor")
    assert code == int(ExitCode.SUCCESS)
    out = capsys.readouterr().out
    assert "Production capabilities: Natural routes active" in out
    assert "Restore and Smart Safe quarantined (BLOCKED)" in out


def test_profile_contract_dimension_mismatch_fails_preflight(tmp_path: Path) -> None:
    """Profile with non-192 embedding dimension fails preflight with ProfileValidationError."""
    spk_dir = tmp_path / "bad_dim_speaker"
    spk_dir.mkdir(parents=True)

    # Write consent
    consent_dir = spk_dir / "consent"
    consent_dir.mkdir()
    consent_file = consent_dir / "consent.json"
    consent_file.write_text(
        json.dumps(
            {
                "speaker_id": "bad_dim_speaker",
                "consent_granted": True,
                "consent_date": "2026-09-04",
            }
        ),
        encoding="utf-8",
    )

    # Write manifest
    canon_dir = spk_dir / "canonical"
    canon_dir.mkdir()
    canon_file = canon_dir / "canonical.jsonl"
    canon_file.write_text(
        json.dumps({"file": "a.wav", "duration_s": 10.0}) + "\n", encoding="utf-8"
    )

    # Write bad 64-D embedding
    emb_dir = spk_dir / "embedding"
    emb_dir.mkdir()
    emb_file = emb_dir / "profile.npy"
    bad_emb = np.ones(64, dtype=np.float32)
    np.save(emb_file, bad_emb)

    from hawavoclean.hashing import hash_file

    emb_hash = hash_file(emb_file)

    profile_data: dict[str, Any] = {
        "schema_version": "1.0",
        "speaker_id": "bad_dim_speaker",
        "display_name": "Bad Dim Speaker",
        "consent_record": "consent/consent.json",
        "canonical_audio_manifest": "canonical/canonical.jsonl",
        "canonical_audio_sha256": ["0" * 64],
        "profile_embedding_path": "embedding/profile.npy",
        "profile_embedding_sha256": emb_hash,
        "f0_statistics": {"median_hz": 150.0, "p05_hz": 90.0, "p95_hz": 300.0},
        "training_split_id": "test",
        "adapter": None,
        "created_by_commit": "abcdef0",
        "notes": "Dimension mismatch test",
        "embedding_dim": 192,
    }
    profile_json = spk_dir / "profile.json"
    profile_json.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

    with pytest.raises(ProfileValidationError, match="Incompatible embedding dimension"):
        validate_speaker_profile(profile_json, base_dir=spk_dir)


def test_profile_revocation_lifecycle(tmp_path: Path) -> None:
    """Revoking a profile updates status and prevents subsequent profile loading."""
    spk_dir = tmp_path / "revokable_speaker"
    spk_dir.mkdir(parents=True)

    consent_dir = spk_dir / "consent"
    consent_dir.mkdir()
    consent_file = consent_dir / "consent.json"
    consent_file.write_text(
        json.dumps(
            {
                "speaker_id": "revokable_speaker",
                "consent_granted": True,
                "consent_date": "2026-09-04",
            }
        ),
        encoding="utf-8",
    )

    canon_dir = spk_dir / "canonical"
    canon_dir.mkdir()
    canon_file = canon_dir / "canonical.jsonl"
    canon_file.write_text(
        json.dumps({"file": "a.wav", "duration_s": 10.0}) + "\n", encoding="utf-8"
    )

    emb_dir = spk_dir / "embedding"
    emb_dir.mkdir()
    emb_file = emb_dir / "profile.npy"
    valid_emb = np.ones(192, dtype=np.float32)
    valid_emb /= np.linalg.norm(valid_emb)
    np.save(emb_file, valid_emb)

    from hawavoclean.hashing import hash_file

    emb_hash = hash_file(emb_file)

    profile_data: dict[str, Any] = {
        "schema_version": "1.0",
        "speaker_id": "revokable_speaker",
        "display_name": "Revokable Speaker",
        "consent_record": "consent/consent.json",
        "canonical_audio_manifest": "canonical/canonical.jsonl",
        "canonical_audio_sha256": ["0" * 64],
        "profile_embedding_path": "embedding/profile.npy",
        "profile_embedding_sha256": emb_hash,
        "f0_statistics": {"median_hz": 150.0, "p05_hz": 90.0, "p95_hz": 300.0},
        "training_split_id": "test",
        "adapter": None,
        "created_by_commit": "abcdef0",
        "notes": "Revocation test",
    }
    profile_json = spk_dir / "profile.json"
    profile_json.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

    # Initially loads fine
    prof = load_speaker_profile("revokable_speaker", profiles_root=tmp_path)
    assert prof.speaker_id == "revokable_speaker"
    assert prof.status == "active"

    # Revoke profile
    revoke_speaker_profile("revokable_speaker", profiles_root=tmp_path, reason="Contract expired")

    # Now load must fail closed
    with pytest.raises(ProfileValidationError, match="has been revoked"):
        load_speaker_profile("revokable_speaker", profiles_root=tmp_path)

    # Consent record must also reflect revocation
    cdata = json.loads(consent_file.read_text(encoding="utf-8"))
    assert cdata["consent_granted"] is False
    assert cdata["revoked"] is True


def test_revoke_nonexistent_profile_fails(tmp_path: Path) -> None:
    """Attempting to revoke a missing profile raises ProfileValidationError."""
    with pytest.raises(ProfileValidationError, match="Cannot revoke: profile 'ghost' not found"):
        revoke_speaker_profile("ghost", profiles_root=tmp_path)


def test_revoke_profile_without_consent_file(tmp_path: Path) -> None:
    """Revocation succeeds even if consent file is missing or removed."""
    spk_dir = tmp_path / "noconsent_speaker"
    spk_dir.mkdir(parents=True)
    profile_data = {
        "schema_version": "1.0",
        "speaker_id": "noconsent_speaker",
        "display_name": "No Consent",
        "consent_record": "consent/missing.json",
        "canonical_audio_manifest": "canonical/canonical.jsonl",
        "canonical_audio_sha256": ["0" * 64],
        "profile_embedding_path": "embedding/profile.npy",
        "profile_embedding_sha256": "0" * 64,
        "f0_statistics": {"median_hz": 150.0, "p05_hz": 90.0, "p95_hz": 300.0},
        "training_split_id": "test",
        "created_by_commit": "abcdef0",
    }
    profile_json = spk_dir / "profile.json"
    profile_json.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
    revoked_path = revoke_speaker_profile("noconsent_speaker", profiles_root=tmp_path)
    assert revoked_path.exists()
    data = json.loads(revoked_path.read_text(encoding="utf-8"))
    assert data["status"] == "revoked"


def test_speaker_profile_to_dict_preserves_custom_contract_metadata() -> None:
    """SpeakerProfile.to_dict retains custom contract metadata when non-default."""
    from hawavoclean.restoration.profiles import SpeakerF0Stats, SpeakerProfile

    prof = SpeakerProfile(
        schema_version="1.0",
        speaker_id="custom_spk",
        display_name="Custom",
        consent_record="consent.json",
        canonical_audio_manifest="canonical.jsonl",
        canonical_audio_sha256=["0" * 64],
        profile_embedding_path="profile.npy",
        profile_embedding_sha256="0" * 64,
        f0_statistics=SpeakerF0Stats(median_hz=120.0, p05_hz=80.0, p95_hz=200.0),
        training_split_id="test",
        adapter=None,
        created_by_commit="1234567",
        notes="",
        status="revoked",
        embedding_dim=256,
        extractor_name="ecapa2",
        extractor_version="1.0.0",
    )
    d = prof.to_dict()
    assert d["status"] == "revoked"
    assert d["embedding_dim"] == 256
    assert d["extractor_name"] == "ecapa2"
    assert d["extractor_version"] == "1.0.0"


def test_profile_contract_variance_dimension_mismatch(tmp_path: Path) -> None:
    """Profile with variance vector dimension mismatch raises ProfileValidationError."""
    spk_dir = tmp_path / "var_mismatch_spk"
    spk_dir.mkdir(parents=True)

    consent_dir = spk_dir / "consent"
    consent_dir.mkdir()
    consent_file = consent_dir / "consent.json"
    consent_file.write_text(
        json.dumps({"speaker_id": "var_mismatch_spk", "consent_granted": True}),
        encoding="utf-8",
    )

    canon_dir = spk_dir / "canonical"
    canon_dir.mkdir()
    (canon_dir / "canonical.jsonl").write_text(
        json.dumps({"file": "a.wav", "duration_s": 10.0}) + "\n", encoding="utf-8"
    )

    emb_dir = spk_dir / "embedding"
    emb_dir.mkdir()
    emb_file = emb_dir / "profile.npy"
    valid_emb = np.ones(192, dtype=np.float32)
    valid_emb /= np.linalg.norm(valid_emb)
    np.save(emb_file, valid_emb)

    var_file = emb_dir / "variance.npy"
    bad_var = np.ones(64, dtype=np.float32)
    np.save(var_file, bad_var)

    from hawavoclean.hashing import hash_file

    profile_data: dict[str, Any] = {
        "schema_version": "1.0",
        "speaker_id": "var_mismatch_spk",
        "display_name": "Var Mismatch",
        "consent_record": "consent/consent.json",
        "canonical_audio_manifest": "canonical/canonical.jsonl",
        "canonical_audio_sha256": ["0" * 64],
        "profile_embedding_path": "embedding/profile.npy",
        "profile_embedding_sha256": hash_file(emb_file),
        "profile_variance_path": "embedding/variance.npy",
        "profile_variance_sha256": hash_file(var_file),
        "f0_statistics": {"median_hz": 150.0, "p05_hz": 90.0, "p95_hz": 300.0},
        "training_split_id": "test",
        "created_by_commit": "abcdef0",
        "embedding_dim": 192,
    }
    profile_json = spk_dir / "profile.json"
    profile_json.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

    with pytest.raises(ProfileValidationError, match="Incompatible variance dimension"):
        validate_speaker_profile(profile_json, base_dir=spk_dir)


def test_profile_contract_variance_corrupt_fails(tmp_path: Path) -> None:
    """Corrupted variance vector file raises ProfileValidationError."""
    spk_dir = tmp_path / "var_corrupt_spk"
    spk_dir.mkdir(parents=True)

    consent_dir = spk_dir / "consent"
    consent_dir.mkdir()
    (consent_dir / "consent.json").write_text(
        json.dumps({"speaker_id": "var_corrupt_spk", "consent_granted": True}),
        encoding="utf-8",
    )

    canon_dir = spk_dir / "canonical"
    canon_dir.mkdir()
    (canon_dir / "canonical.jsonl").write_text(
        json.dumps({"file": "a.wav", "duration_s": 10.0}) + "\n", encoding="utf-8"
    )

    emb_dir = spk_dir / "embedding"
    emb_dir.mkdir()
    emb_file = emb_dir / "profile.npy"
    valid_emb = np.ones(192, dtype=np.float32)
    valid_emb /= np.linalg.norm(valid_emb)
    np.save(emb_file, valid_emb)

    var_file = emb_dir / "variance.npy"
    var_file.write_bytes(b"not a valid npy file")

    from hawavoclean.hashing import hash_file

    profile_data: dict[str, Any] = {
        "schema_version": "1.0",
        "speaker_id": "var_corrupt_spk",
        "display_name": "Var Corrupt",
        "consent_record": "consent/consent.json",
        "canonical_audio_manifest": "canonical/canonical.jsonl",
        "canonical_audio_sha256": ["0" * 64],
        "profile_embedding_path": "embedding/profile.npy",
        "profile_embedding_sha256": hash_file(emb_file),
        "profile_variance_path": "embedding/variance.npy",
        "profile_variance_sha256": hash_file(var_file),
        "f0_statistics": {"median_hz": 150.0, "p05_hz": 90.0, "p95_hz": 300.0},
        "training_split_id": "test",
        "created_by_commit": "abcdef0",
        "embedding_dim": 192,
    }
    profile_json = spk_dir / "profile.json"
    profile_json.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

    with pytest.raises(ProfileValidationError, match="Failed to load variance vector"):
        validate_speaker_profile(profile_json, base_dir=spk_dir)


def test_production_cli_accepts_restore_with_cli_flag(
    synthetic_wav_48k: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production CLI process accepts restore mode when --allow-research-restore is passed."""
    out_wav = tmp_path / "out_cli_allowed.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(synthetic_wav_48k),
        "-o",
        str(out_wav),
        "--profile",
        "production",
        "--mode",
        "restore",
        "--speaker-id",
        "character_01",
        "--overwrite",
        "--allow-research-restore",
    )
    assert code == int(ExitCode.SUCCESS)
    assert out_wav.is_file()


def test_production_cli_accepts_restore_with_env_var(
    synthetic_wav_48k: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production CLI process accepts restore mode when HAWAVOCLEAN_ALLOW_RESEARCH_RESTORE=1 is set."""
    monkeypatch.setenv("HAWAVOCLEAN_ALLOW_RESEARCH_RESTORE", "1")
    out_wav = tmp_path / "out_env_allowed.wav"
    code = _run_cli(
        monkeypatch,
        "process",
        str(synthetic_wav_48k),
        "-o",
        str(out_wav),
        "--profile",
        "production",
        "--mode",
        "restore",
        "--speaker-id",
        "character_01",
        "--overwrite",
    )
    assert code == int(ExitCode.SUCCESS)
    assert out_wav.is_file()
