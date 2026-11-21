"""Unit tests for restoration validation branches: profiles, config, and policy fail-closed."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.signal as sp_signal
from pydantic import ValidationError

from hawavoclean.hashing import hash_file
from hawavoclean.restoration.bandwidth import BandwidthEstimate, BandwidthEvidence
from hawavoclean.restoration.base import (
    RestorationCandidate,
    RestorationRenderResult,
)
from hawavoclean.restoration.config import RestorationConfig, RestorationGuardConfig
from hawavoclean.restoration.guard import GuardRResult, RestorationGuard
from hawavoclean.restoration.policy import RestorationPolicyManager
from hawavoclean.restoration.profiles import (
    ProfileValidationError,
    SpeakerProfile,
    load_speaker_profile,
    validate_all_profiles,
    validate_speaker_profile,
)

FloatArray = np.ndarray[Any, Any]

# ---------------------------------------------------------------------------
# Profile fixtures
# ---------------------------------------------------------------------------


def _build_profile_dir(
    tmp: Path,
    speaker_id: str = "spk_test01",
    embedding: FloatArray | None = None,
    emb_suffix: str = ".npy",
) -> tuple[Path, dict[str, Any]]:
    """Write a fully valid speaker profile into tmp; return (profile_path, data)."""
    tmp.mkdir(parents=True, exist_ok=True)
    if embedding is None:
        rng = np.random.default_rng(7)
        embedding = rng.standard_normal(16).astype(np.float32)
    if emb_suffix == ".npy":
        emb_path = tmp / "embedding.npy"
        np.save(emb_path, embedding)
    elif emb_suffix == ".json":
        emb_path = tmp / "embedding.json"
        emb_path.write_text(json.dumps({"embedding": [float(x) for x in embedding]}))
    else:
        emb_path = tmp / f"embedding{emb_suffix}"
        embedding.astype(np.float32).tofile(emb_path)
    (tmp / "manifest.jsonl").write_text('{"path": "clip_001.wav"}\n')
    (tmp / "consent.json").write_text(
        json.dumps({"speaker_id": speaker_id, "consent_granted": True})
    )
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "speaker_id": speaker_id,
        "display_name": "Test Speaker",
        "consent_record": "consent.json",
        "canonical_audio_manifest": "manifest.jsonl",
        "canonical_audio_sha256": ["ab" * 32],
        "profile_embedding_path": emb_path.name,
        "profile_embedding_sha256": hash_file(emb_path),
        "f0_statistics": {"median_hz": 120.0, "p05_hz": 80.0, "p95_hz": 180.0},
        "training_split_id": "split_01",
        "adapter": None,
        "created_by_commit": "deadbeef",
        "notes": "",
    }
    profile_path = tmp / "profile.json"
    profile_path.write_text(json.dumps(data))
    return profile_path, data


def _rewrite(profile_path: Path, data: dict[str, Any]) -> None:
    profile_path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# profiles.py: validation branches
# ---------------------------------------------------------------------------


def test_valid_profile_loads_npy_embedding(tmp_path: Path) -> None:
    """A fully valid profile loads, exposing the embedding and hash-stable metadata."""
    emb = np.arange(1, 17, dtype=np.float32)
    profile_path, _ = _build_profile_dir(tmp_path, embedding=emb)
    prof = validate_speaker_profile(profile_path)
    assert isinstance(prof, SpeakerProfile)
    assert prof.speaker_id == "spk_test01"
    assert prof.embedding_vector is not None
    assert prof.embedding_vector.dtype == np.float32
    np.testing.assert_array_equal(prof.embedding_vector, emb)
    assert prof.f0_statistics.median_hz == 120.0
    d = prof.to_dict()
    assert "embedding_vector" not in d
    h1 = prof.compute_hash()
    assert len(h1) == 64 and h1 == prof.compute_hash()


def test_missing_profile_file(tmp_path: Path) -> None:
    with pytest.raises(ProfileValidationError, match="not found"):
        validate_speaker_profile(tmp_path / "nope.json")


def test_malformed_profile_json(tmp_path: Path) -> None:
    bad = tmp_path / "profile.json"
    bad.write_text("{not valid json")
    with pytest.raises(ProfileValidationError, match="Failed to parse"):
        validate_speaker_profile(bad)


def test_wrong_schema_version(tmp_path: Path) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    data["schema_version"] = "2.0"
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="Unsupported speaker profile schema"):
        validate_speaker_profile(profile_path)


def test_absent_schema_version_reports_unsupported(tmp_path: Path) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    del data["schema_version"]
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="Unsupported speaker profile schema"):
        validate_speaker_profile(profile_path)


@pytest.mark.parametrize(
    "key",
    [
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
    ],
)
def test_each_missing_required_key_rejected(tmp_path: Path, key: str) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    del data[key]
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match=f"missing required field: '{key}'"):
        validate_speaker_profile(profile_path)


def test_null_required_key_counts_as_missing(tmp_path: Path) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    data["training_split_id"] = None
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="missing required field: 'training_split_id'"):
        validate_speaker_profile(profile_path)


@pytest.mark.parametrize("bad_id", ["bad id", "spk-01", ""])
def test_invalid_speaker_id_format(tmp_path: Path, bad_id: str) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    data["speaker_id"] = bad_id
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="Invalid speaker_id format"):
        validate_speaker_profile(profile_path)


def test_missing_consent_record_file(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "consent.json").unlink()
    with pytest.raises(ProfileValidationError, match="consent record missing"):
        validate_speaker_profile(profile_path)


def test_consent_wrong_speaker_id(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "consent.json").write_text(
        json.dumps({"speaker_id": "someone_else", "consent_granted": True})
    )
    with pytest.raises(ProfileValidationError, match="Invalid or revoked consent"):
        validate_speaker_profile(profile_path)


def test_consent_not_granted(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "consent.json").write_text(
        json.dumps({"speaker_id": "spk_test01", "consent_granted": False})
    )
    with pytest.raises(ProfileValidationError, match="Invalid or revoked consent"):
        validate_speaker_profile(profile_path)


def test_corrupted_consent_json(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "consent.json").write_text("{broken")
    with pytest.raises(ProfileValidationError, match="Failed to verify consent record"):
        validate_speaker_profile(profile_path)


def test_missing_canonical_manifest(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "manifest.jsonl").unlink()
    with pytest.raises(ProfileValidationError, match="manifest missing"):
        validate_speaker_profile(profile_path)


def test_empty_canonical_manifest(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "manifest.jsonl").write_text("\n   \n")
    with pytest.raises(ProfileValidationError, match="manifest is empty"):
        validate_speaker_profile(profile_path)


def test_missing_embedding_file(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "embedding.npy").unlink()
    with pytest.raises(ProfileValidationError, match="embedding file missing"):
        validate_speaker_profile(profile_path)


def test_embedding_hash_mismatch(tmp_path: Path) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    data["profile_embedding_sha256"] = "0" * 64
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="hash mismatch"):
        validate_speaker_profile(profile_path)


def test_json_embedding_loading_branch(tmp_path: Path) -> None:
    emb = np.array([0.5, -1.5, 2.0, 3.0], dtype=np.float32)
    profile_path, _ = _build_profile_dir(tmp_path, embedding=emb, emb_suffix=".json")
    prof = validate_speaker_profile(profile_path)
    assert prof.embedding_vector is not None
    np.testing.assert_allclose(prof.embedding_vector, emb)


def test_raw_binary_embedding_loading_branch(tmp_path: Path) -> None:
    emb = np.array([1.0, 2.0, 3.0, -4.0], dtype=np.float32)
    profile_path, _ = _build_profile_dir(tmp_path, embedding=emb, emb_suffix=".bin")
    prof = validate_speaker_profile(profile_path)
    assert prof.embedding_vector is not None
    np.testing.assert_array_equal(prof.embedding_vector, emb)


def test_malformed_json_embedding_rejected(tmp_path: Path) -> None:
    profile_path, data = _build_profile_dir(tmp_path, emb_suffix=".json")
    emb_path = tmp_path / "embedding.json"
    emb_path.write_text(json.dumps({"wrong_key": [1.0]}))
    data["profile_embedding_sha256"] = hash_file(emb_path)
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="Failed to load embedding"):
        validate_speaker_profile(profile_path)


def test_degenerate_zero_embedding_rejected(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path, embedding=np.zeros(16, dtype=np.float32))
    with pytest.raises(ProfileValidationError, match="Degenerate zero embedding"):
        validate_speaker_profile(profile_path)


@pytest.mark.parametrize(
    "f0",
    [
        {"median_hz": 120.0, "p05_hz": 200.0, "p95_hz": 100.0},  # p05 > p95
        {"median_hz": -5.0, "p05_hz": 80.0, "p95_hz": 180.0},  # negative median
    ],
)
def test_invalid_f0_statistics_rejected(tmp_path: Path, f0: dict[str, float]) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    data["f0_statistics"] = f0
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="Invalid F0 statistics"):
        validate_speaker_profile(profile_path)


def test_f0_statistics_must_be_dict(tmp_path: Path) -> None:
    profile_path, data = _build_profile_dir(tmp_path)
    data["f0_statistics"] = [120.0, 80.0, 180.0]
    _rewrite(profile_path, data)
    with pytest.raises(ProfileValidationError, match="must be a dictionary"):
        validate_speaker_profile(profile_path)


def test_verify_files_false_skips_file_checks(tmp_path: Path) -> None:
    """With verify_files=False the schema is still enforced but no files are touched."""
    profile_path, _ = _build_profile_dir(tmp_path)
    (tmp_path / "consent.json").unlink()
    (tmp_path / "manifest.jsonl").unlink()
    (tmp_path / "embedding.npy").unlink()
    prof = validate_speaker_profile(profile_path, verify_files=False)
    assert prof.embedding_vector is None
    assert prof.speaker_id == "spk_test01"


def test_absolute_referenced_paths_resolve(tmp_path: Path) -> None:
    """Absolute consent/manifest/embedding paths are used verbatim, not joined to base_dir."""
    profile_path, data = _build_profile_dir(tmp_path)
    data["consent_record"] = str(tmp_path / "consent.json")
    data["canonical_audio_manifest"] = str(tmp_path / "manifest.jsonl")
    data["profile_embedding_path"] = str(tmp_path / "embedding.npy")
    _rewrite(profile_path, data)
    other_base = tmp_path / "elsewhere"
    other_base.mkdir()
    prof = validate_speaker_profile(profile_path, base_dir=other_base)
    assert prof.embedding_vector is not None


def test_load_speaker_profile_directory_layout(tmp_path: Path) -> None:
    _build_profile_dir(tmp_path / "spk_dir01", speaker_id="spk_dir01")
    prof = load_speaker_profile("spk_dir01", profiles_root=tmp_path)
    assert prof.speaker_id == "spk_dir01"


def test_load_speaker_profile_flat_file_layout(tmp_path: Path) -> None:
    profile_path, _ = _build_profile_dir(tmp_path, speaker_id="spk_flat01")
    profile_path.rename(tmp_path / "spk_flat01.json")
    prof = load_speaker_profile("spk_flat01", profiles_root=tmp_path)
    assert prof.speaker_id == "spk_flat01"


def test_load_speaker_profile_not_found(tmp_path: Path) -> None:
    with pytest.raises(ProfileValidationError, match="not found under"):
        load_speaker_profile("spk_ghost", profiles_root=tmp_path)


def _build_registry(root: Path, duplicate_pair: tuple[int, int] | None = None) -> None:
    """Build the 10-profile character registry with per-speaker embeddings."""
    for i in range(1, 11):
        spk = f"character_{i:02d}"
        seed = duplicate_pair[0] if duplicate_pair and i in duplicate_pair else i
        rng = np.random.default_rng(100 + seed)
        emb = rng.standard_normal(16).astype(np.float32)
        _build_profile_dir(root / spk, speaker_id=spk, embedding=emb)


def test_validate_all_profiles_success(tmp_path: Path) -> None:
    _build_registry(tmp_path)
    profiles = validate_all_profiles(profiles_root=tmp_path)
    assert set(profiles) == {f"character_{i:02d}" for i in range(1, 11)}
    hashes = {p.profile_embedding_sha256 for p in profiles.values()}
    assert len(hashes) == 10


def test_validate_all_profiles_duplicate_hash_rejected(tmp_path: Path) -> None:
    """Two registered speakers sharing an embedding hash is a registry integrity failure."""
    _build_registry(tmp_path, duplicate_pair=(1, 2))
    with pytest.raises(ProfileValidationError, match="Duplicate embedding hash"):
        validate_all_profiles(profiles_root=tmp_path)


# ---------------------------------------------------------------------------
# config.py: strengths validator and immutability
# ---------------------------------------------------------------------------


def test_strengths_empty_ladder_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        RestorationConfig(strengths=[])


@pytest.mark.parametrize("bad", [[1.5, 0.0], [-0.1, 0.0]])
def test_strengths_out_of_range_rejected(bad: list[float]) -> None:
    with pytest.raises(ValidationError, match="must be in"):
        RestorationConfig(strengths=bad)


def test_strengths_without_zero_fallback_rejected() -> None:
    with pytest.raises(ValidationError, match="must include 0.0"):
        RestorationConfig(strengths=[1.0, 0.5])


def test_strengths_sorted_descending() -> None:
    cfg = RestorationConfig(strengths=[0.0, 0.25, 1.0, 0.5])
    assert cfg.strengths == [1.0, 0.5, 0.25, 0.0]


def test_restoration_config_frozen() -> None:
    cfg = RestorationConfig()
    with pytest.raises(ValidationError):
        cfg.enabled = True


def test_guard_config_frozen() -> None:
    guard_cfg = RestorationGuardConfig()
    with pytest.raises(ValidationError):
        guard_cfg.speaker_threshold = 0.1


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RestorationConfig.model_validate({"unexpected_field": 1})


def test_target_sample_rate_pinned_to_48k() -> None:
    with pytest.raises(ValidationError):
        RestorationConfig(target_sample_rate=44100)


def test_guard_thresholds_bounded() -> None:
    with pytest.raises(ValidationError):
        RestorationGuardConfig(speaker_threshold=1.5)


# ---------------------------------------------------------------------------
# policy.py: process_segment fail-closed branches
# ---------------------------------------------------------------------------


class LadderRestorer:
    """Stub Restorer returning pre-crafted candidates and recording its inputs."""

    def __init__(self, candidates: list[RestorationCandidate]) -> None:
        self.candidates = candidates
        self.calls = 0
        self.last_speaker_id: str | None = None
        self.last_speaker_embedding: FloatArray | None = None

    def restore(
        self,
        audio_48k: FloatArray,  # noqa: ARG002
        sample_rate: int,  # noqa: ARG002
        effective_cutoff_hz: float,  # noqa: ARG002
        speaker_id: str | None = None,
        speaker_embedding: FloatArray | None = None,
        f0_trajectory: FloatArray | None = None,  # noqa: ARG002
        vuv_mask: FloatArray | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,  # noqa: ARG002
        seed: int = 42,  # noqa: ARG002
    ) -> list[RestorationCandidate]:
        self.calls += 1
        self.last_speaker_id = speaker_id
        self.last_speaker_embedding = speaker_embedding
        return self.candidates

    def render(
        self,
        audio_48k: FloatArray,
        sample_rate: int,
        effective_cutoff_hz: float,
        speaker_id: str | None = None,
        speaker_embedding: FloatArray | None = None,
        f0_trajectory: FloatArray | None = None,
        vuv_mask: FloatArray | None = None,
        strengths: list[float] | None = None,
        seed: int = 42,
    ) -> RestorationRenderResult:
        cands = self.restore(
            audio_48k=audio_48k,
            sample_rate=sample_rate,
            effective_cutoff_hz=effective_cutoff_hz,
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
            f0_trajectory=f0_trajectory,
            vuv_mask=vuv_mask,
            strengths=strengths,
            seed=seed,
        )
        has_active = any(c.strength > 0.0 for c in cands)
        return RestorationRenderResult(
            success=has_active,
            fallback_status="none" if has_active else "no_active_candidates",
            model_name="ladder-restorer",
            provider="cpu",
            solver="mock",
            candidates=cands,
        )


class RaisingRestorer:
    """Stub Restorer that always fails at runtime."""

    def restore(
        self,
        audio_48k: FloatArray,  # noqa: ARG002
        sample_rate: int,  # noqa: ARG002
        effective_cutoff_hz: float,  # noqa: ARG002
        speaker_id: str | None = None,  # noqa: ARG002
        speaker_embedding: FloatArray | None = None,  # noqa: ARG002
        f0_trajectory: FloatArray | None = None,  # noqa: ARG002
        vuv_mask: FloatArray | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,  # noqa: ARG002
        seed: int = 42,  # noqa: ARG002
    ) -> list[RestorationCandidate]:
        raise RuntimeError("model exploded")

    def render(
        self,
        audio_48k: FloatArray,  # noqa: ARG002
        sample_rate: int,  # noqa: ARG002
        effective_cutoff_hz: float,  # noqa: ARG002
        speaker_id: str | None = None,  # noqa: ARG002
        speaker_embedding: FloatArray | None = None,  # noqa: ARG002
        f0_trajectory: FloatArray | None = None,  # noqa: ARG002
        vuv_mask: FloatArray | None = None,  # noqa: ARG002
        strengths: list[float] | None = None,  # noqa: ARG002
        seed: int = 42,  # noqa: ARG002
    ) -> RestorationRenderResult:
        raise RuntimeError("model exploded")


def _estimate(
    cutoff_hz: float = 4000.0,
    confidence: float = 0.95,
    recommended: bool = True,
) -> BandwidthEstimate:
    return BandwidthEstimate(
        effective_cutoff_hz=cutoff_hz,
        confidence=confidence,
        shape="codec_lowpass",
        restore_recommended=recommended,
        evidence=BandwidthEvidence(
            spectral_rolloff=0.0,
            above_cutoff_snr_db=0.0,
            stationarity=1.0,
            high_band_energy_ratio_db=0.0,
        ),
    )


def _manager(restorer: LadderRestorer | RaisingRestorer) -> RestorationPolicyManager:
    return RestorationPolicyManager(
        config=RestorationConfig(),
        restorer=restorer,
        guard=RestorationGuard(sample_rate=48000),
    )


def _lowpassed_speechlike(duration_s: float = 0.5) -> FloatArray:
    sr = 48000
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    clean = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
    sos = sp_signal.butter(6, 4000 / 24000, btype="lowpass", output="sos")
    filtered: FloatArray = sp_signal.sosfiltfilt(sos, clean).astype(np.float32)
    return filtered


def test_empty_audio_fails_closed() -> None:
    mgr = _manager(LadderRestorer([]))
    empty = np.zeros(0, dtype=np.float32)
    out, decision = mgr.process_segment(empty, 48000, _estimate())
    assert decision.action == "error"
    assert decision.applied_strength == 0.0
    assert decision.guard_result is not None and decision.guard_result.verdict == "ERROR"
    assert decision.error_message == "Empty audio"
    assert out.size == 0


def test_nonfinite_audio_fails_closed() -> None:
    mgr = _manager(LadderRestorer([]))
    bad = np.full(2400, np.nan, dtype=np.float32)
    out, decision = mgr.process_segment(bad, 48000, _estimate())
    assert decision.action == "error"
    assert decision.error_message is not None and "Non-finite" in decision.error_message
    assert out is bad  # Natural candidate returned untouched


def test_bypass_when_bandwidth_healthy() -> None:
    restorer = LadderRestorer([])
    mgr = _manager(restorer)
    audio = np.ones(2400, dtype=np.float32) * 0.1
    out, decision = mgr.process_segment(audio, 48000, _estimate(recommended=False))
    assert decision.action == "bypassed"
    assert decision.applied_strength == 0.0
    assert decision.guard_result is not None
    assert decision.guard_result.verdict == "NO_RESTORE"
    assert "healthy" in decision.guard_result.reason
    assert restorer.calls == 0  # the model must never run on healthy audio
    assert out is audio


def test_bypass_on_low_confidence() -> None:
    restorer = LadderRestorer([])
    mgr = _manager(restorer)
    audio = np.ones(2400, dtype=np.float32) * 0.1
    out, decision = mgr.process_segment(audio, 48000, _estimate(confidence=0.5))
    assert decision.action == "bypassed"
    assert decision.guard_result is not None
    assert "Low confidence" in decision.guard_result.reason
    assert restorer.calls == 0
    assert out is audio


def test_restorer_exception_fails_closed_to_natural() -> None:
    mgr = _manager(RaisingRestorer())
    audio = _lowpassed_speechlike(0.2)
    out, decision = mgr.process_segment(audio, 48000, _estimate())
    assert decision.action == "error"
    assert decision.applied_strength == 0.0
    assert decision.error_message is not None and "model exploded" in decision.error_message
    assert decision.guard_result is not None and decision.guard_result.verdict == "ERROR"
    assert out is audio


def test_guard_exception_fails_closed_to_natural(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _lowpassed_speechlike(0.2)
    cand = RestorationCandidate(strength=1.0, audio=audio, cutoff_hz=4000.0)
    mgr = _manager(LadderRestorer([cand]))

    def _boom(**_kwargs: object) -> tuple[FloatArray, GuardRResult]:
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(mgr.guard, "select_best_candidate", _boom)
    out, decision = mgr.process_segment(audio, 48000, _estimate())
    assert decision.action == "error"
    assert decision.error_message is not None and "guard exploded" in decision.error_message
    assert decision.guard_result is not None
    assert "Guard R evaluation error" in decision.guard_result.reason
    assert out is audio


def test_empty_candidate_ladder_is_bypass_not_rejection() -> None:
    mgr = _manager(LadderRestorer([]))
    audio = _lowpassed_speechlike(0.2)
    out, decision = mgr.process_segment(audio, 48000, _estimate())
    assert decision.action == "bypassed"
    assert decision.applied_strength == 0.0
    assert decision.guard_result is not None
    assert decision.guard_result.verdict == "NO_RESTORE"
    assert out is audio


def test_full_strength_acceptance_maps_to_restored() -> None:
    audio = _lowpassed_speechlike()
    cand = RestorationCandidate(strength=1.0, audio=audio, cutoff_hz=4000.0)
    mgr = _manager(LadderRestorer([cand]))
    out, decision = mgr.process_segment(audio, 48000, _estimate())
    assert decision.action == "restored"
    assert decision.applied_strength == 1.0
    np.testing.assert_array_equal(out, audio)


def test_mid_strength_acceptance_maps_to_reduced() -> None:
    audio = _lowpassed_speechlike()
    clipped = audio.copy()
    clipped[-10:] += 5.0  # peak > 1.05 -> structural rejection of the 1.0 candidate
    candidates = [
        RestorationCandidate(strength=1.0, audio=clipped, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.5, audio=audio, cutoff_hz=4000.0),
    ]
    mgr = _manager(LadderRestorer(candidates))
    out, decision = mgr.process_segment(audio, 48000, _estimate())
    assert decision.action == "reduced"
    assert decision.applied_strength == 0.5
    np.testing.assert_array_equal(out, audio)


def test_all_candidates_rejected_maps_to_reverted() -> None:
    audio = _lowpassed_speechlike(0.2)
    clipped = audio.copy()
    clipped[-10:] += 5.0
    candidates = [
        RestorationCandidate(strength=1.0, audio=clipped, cutoff_hz=4000.0),
        RestorationCandidate(strength=0.5, audio=clipped, cutoff_hz=4000.0),
    ]
    mgr = _manager(LadderRestorer(candidates))
    out, decision = mgr.process_segment(audio, 48000, _estimate())
    assert decision.action == "reverted"
    assert decision.applied_strength == 0.0
    np.testing.assert_array_equal(out, audio)


def test_speaker_profile_identity_forwarded_to_restorer(tmp_path: Path) -> None:
    """The profile's speaker_id and embedding vector must reach the restorer intact."""
    emb = np.arange(1, 17, dtype=np.float32)
    profile_path, _ = _build_profile_dir(tmp_path, embedding=emb)
    prof = validate_speaker_profile(profile_path)
    restorer = LadderRestorer([])
    mgr = _manager(restorer)
    audio = _lowpassed_speechlike(0.2)
    mgr.process_segment(audio, 48000, _estimate(), speaker_profile=prof)
    assert restorer.calls == 1
    assert restorer.last_speaker_id == "spk_test01"
    assert restorer.last_speaker_embedding is not None
    np.testing.assert_array_equal(restorer.last_speaker_embedding, emb)
