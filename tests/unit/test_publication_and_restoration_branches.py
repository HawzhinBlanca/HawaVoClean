from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hawavoclean.errors import PublicationError
from hawavoclean.publication import (
    _read_current_id,
    _repair_public_exports,
    _verify_generation,
    publication_paths,
)
from hawavoclean.restoration.guard import RestorationGuard
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.highband_events import HighBandEventResult
from hawavoclean.restoration.linguistic_guard import LinguisticGuardResult
from hawavoclean.restoration.protected_band import verify_protected_band_invariance

# --- 1. Publication Error and Security Branches ---


def test_read_generation_manifest_errors(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen_dir"
    gen_dir.mkdir()
    manifest_file = gen_dir / "manifest.json"

    # 1. Invalid role record (not a dict)
    bad_record_manifest = {
        "schema_version": 1,
        "generation_id": gen_dir.name,
        "artifacts": {
            "audio": "not_a_dict",
            "json": "not_a_dict",
            "txt": "not_a_dict",
        },
    }
    manifest_file.write_text(json.dumps(bad_record_manifest), encoding="utf-8")
    with pytest.raises(PublicationError, match="record is invalid"):
        _verify_generation(gen_dir)

    # 2. Missing artifact file
    valid_record_manifest = {
        "schema_version": 1,
        "generation_id": gen_dir.name,
        "artifacts": {
            "audio": {"filename": "master.wav", "size_bytes": 10, "sha256": "a" * 64},
            "json": {"filename": "report.json", "size_bytes": 10, "sha256": "b" * 64},
            "txt": {"filename": "summary.txt", "size_bytes": 10, "sha256": "c" * 64},
        },
    }
    manifest_file.write_text(json.dumps(valid_record_manifest), encoding="utf-8")
    with pytest.raises(PublicationError, match="missing or unsafe"):
        _verify_generation(gen_dir)

    # 3. Artifact exists and matches digest, but payload hash does not derive generation ID
    (gen_dir / "master.wav").write_bytes(b"1234567890")
    (gen_dir / "report.json").write_text(
        json.dumps({"output": {"sha256": "a" * 64}}), encoding="utf-8"
    )
    (gen_dir / "summary.txt").write_bytes(b"1234567890")

    def mock_sha(p: Path) -> str:
        if "master.wav" in p.name:
            return "a" * 64
        elif "report.json" in p.name:
            return "b" * 64
        return "c" * 64

    with (
        patch("hawavoclean.publication._sha256_file", side_effect=mock_sha),
        patch.object(Path, "stat", return_value=MagicMock(st_size=10)),
        pytest.raises(PublicationError, match="does not derive its ID"),
    ):
        _verify_generation(gen_dir)


def test_resolve_current_pointer_errors(tmp_path: Path) -> None:
    paths = publication_paths(tmp_path / "out.wav")
    paths.bundle.mkdir(parents=True)

    # 1. current is a directory instead of a regular file
    paths.current.mkdir()
    with pytest.raises(PublicationError, match="pointer is unsafe"):
        _read_current_id(paths)
    paths.current.rmdir()

    # 2. current has invalid JSON schema
    paths.current.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(PublicationError, match="pointer is invalid"):
        _read_current_id(paths)


def test_replace_public_aliases_refuses_unsafe_dest(tmp_path: Path) -> None:
    pub_audio = tmp_path / "out.wav"
    # Make pub_audio a directory
    pub_audio.mkdir()
    paths = publication_paths(pub_audio)

    gen_dir = paths.generations / "gen_dir"
    gen_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "generation_id": "gen_dir",
        "artifacts": {
            "audio": {"filename": "master.wav", "size_bytes": 10, "sha256": "a" * 64},
            "json": {"filename": "report.json", "size_bytes": 10, "sha256": "b" * 64},
            "txt": {"filename": "summary.txt", "size_bytes": 10, "sha256": "c" * 64},
        },
    }
    (gen_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (
        patch("hawavoclean.publication._verify_generation", return_value=manifest),
        pytest.raises(PublicationError, match="not a regular file"),
    ):
        _repair_public_exports(paths, "gen_dir")


# --- 2. Restoration Guard and Protected Band Branches ---


def test_verify_protected_band_invariance_empty_bins() -> None:
    sr = 48000
    audio = np.zeros(2048, dtype=np.float32)
    with patch("numpy.fft.rfftfreq", return_value=np.array([1000.0, 2000.0])):
        verif = verify_protected_band_invariance(audio, audio, sample_rate=sr, cutoff_hz=500.0)
        assert verif.passes_invariance is True
        assert verif.complex_stft_relative_error == 0.0


def test_guard_r_hf_events_failure() -> None:
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    nat = np.zeros(4800, dtype=np.float32)
    cand = np.zeros(4800, dtype=np.float32)

    bad_hf_res = HighBandEventResult(
        speech_window_leakage=0.5,
        spurious_burst_count=5,
        hf_envelope_divergence=0.8,
        impulse_discontinuity_ratio=4.0,
        passes_event_check=False,
    )
    with patch.object(guard.hf_event_detector, "evaluate", return_value=bad_hf_res):
        verdict, reason, metrics = guard.evaluate_candidate(nat, cand, cutoff_hz=4000.0)
        assert verdict is False
        assert "High-band event inconsistency" in reason
        assert "spurious burst(s)" in reason


def test_guard_r_harmonic_pitch_failure() -> None:
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    nat = np.zeros(4800, dtype=np.float32)
    cand = np.zeros(4800, dtype=np.float32)

    good_hf_res = HighBandEventResult(
        speech_window_leakage=0.0,
        spurious_burst_count=0,
        hf_envelope_divergence=0.0,
        impulse_discontinuity_ratio=1.0,
        passes_event_check=True,
    )

    fake_f0_nat = MagicMock()
    fake_f0_nat.f0_hz = np.array([200.0, 200.0])
    fake_f0_nat.vuv_mask = np.array([1.0, 1.0])
    fake_f0_nat.statistics.median_hz = 200.0

    fake_f0_cand = MagicMock()
    fake_f0_cand.f0_hz = np.array([300.0, 300.0])  # 50% pitch divergence!
    fake_f0_cand.vuv_mask = np.array([1.0, 1.0])
    fake_f0_cand.statistics.median_hz = 300.0

    with (
        patch.object(guard.hf_event_detector, "evaluate", return_value=good_hf_res),
        patch.object(guard.f0_extractor, "extract", side_effect=[fake_f0_nat, fake_f0_cand]),
    ):
        verdict, reason, metrics = guard.evaluate_candidate(nat, cand, cutoff_hz=4000.0)
        assert verdict is False
        assert "Harmonic pitch divergence" in reason


def test_guard_r_linguistic_check_failure() -> None:
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    nat = np.zeros(4800, dtype=np.float32)
    cand = np.zeros(4800, dtype=np.float32)

    good_hf_res = HighBandEventResult(
        speech_window_leakage=0.0,
        spurious_burst_count=0,
        hf_envelope_divergence=0.0,
        impulse_discontinuity_ratio=1.0,
        passes_event_check=True,
    )

    fake_f0 = MagicMock()
    fake_f0.f0_hz = np.array([200.0, 200.0])
    fake_f0.vuv_mask = np.array([1.0, 1.0])
    fake_f0.statistics.median_hz = 200.0

    bad_ling_res = LinguisticGuardResult(
        divergence=0.35,
        anchor_preserved=False,
        status="failed",
        max_frame_divergence=0.5,
        passes_check=False,
    )

    with (
        patch.object(guard.hf_event_detector, "evaluate", return_value=good_hf_res),
        patch.object(guard.f0_extractor, "extract", return_value=fake_f0),
        patch.object(guard.linguistic_guard, "evaluate", return_value=bad_ling_res),
    ):
        verdict, reason, metrics = guard.evaluate_candidate(nat, cand, cutoff_hz=4000.0)
        assert verdict is False
        assert "Linguistic posterior divergence" in reason


# --- 3. HawaRestore-KD Solver and Block Branches ---


def test_hawarestore_kd_heun_solver_and_short_tails() -> None:
    # Initialize HawaRestoreKD with solver="heun"
    restorer = HawaRestoreKD(device="cpu", solver="heun")

    # 1. Short tail block splitting (test lines 389-396)
    positions = restorer._block_positions(n_samples=50000)
    assert len(positions) >= 2

    # 2. Block restoration with solver="heun"
    test_audio = np.sin(np.linspace(0, 100, 48000, dtype=np.float32))
    cands = restorer.restore(
        test_audio,
        sample_rate=48000,
        effective_cutoff_hz=4000.0,
        strengths=[1.0, 0.0],
    )
    assert len(cands) == 2
    assert cands[0].strength == 1.0
    assert cands[1].strength == 0.0

    # 3. Non-finite restored audio fallback to zeros (lines 597-598)
    with patch.object(
        restorer, "_restore_block", return_value={1.0: np.full(48000, np.nan, dtype=np.float32)}
    ):
        cands_nan = restorer.restore(
            test_audio,
            sample_rate=48000,
            effective_cutoff_hz=4000.0,
            strengths=[1.0],
        )
        assert len(cands_nan) == 2
        assert np.all(cands_nan[0].audio == 0.0)


# --- 4. Finishing Detect and Analysis Accumulator Branches ---


def test_detect_defects_and_tonal_branches() -> None:
    from hawavoclean.finishing.detect import _band_stats, _ramp, _size_correction, detect_defects

    # 1. Short audio (< 512 samples)
    rep = detect_defects(np.zeros(100, dtype=np.float32), sample_rate=48000)
    assert rep.has_dc_offset is False
    assert rep.click_count == 0

    # 2. _ramp identical thresholds
    assert _ramp(5.0, 10.0, 10.0) == 0.0
    assert _ramp(15.0, 10.0, 10.0) == 1.0

    # 3. _band_stats with frequencies outside range (e.g. above Nyquist)
    empty_stats = _band_stats(
        stft_power=np.ones((10, 10), dtype=np.float64),
        freqs=np.linspace(0, 1000, 10, dtype=np.float64),
        active=np.ones(10, dtype=bool),
        low_hz=5000.0,
        high_hz=6000.0,
    )
    assert empty_stats.present is False

    # 4. _size_correction with not present band
    c_low, r_low, c_pres, r_pres, c_brill, r_brill = _size_correction(
        low=empty_stats,
        presence=empty_stats,
        brilliance=empty_stats,
        body_db=-20.0,
    )
    assert r_low == ":above-nyquist"
    assert c_low == 0.0


def test_analysis_accumulator_branches() -> None:
    from hawavoclean.server.analysis import _BucketReducer, _LoudnessAccumulator

    # 1. _BucketReducer push empty
    peaks = _BucketReducer(n_expected=1000, buckets=50)
    peaks.push(0, np.zeros(0, dtype=np.float32))
    assert peaks.seen == 0

    # 2. _BucketReducer finish trim=False with sparse buckets
    peaks.push(0, np.ones(50, dtype=np.float32))
    mins, maxs, rms_db = peaks.finish(trim=False)
    assert len(mins) == 50

    # 3. _LoudnessAccumulator silence/gate branches
    loudness = _LoudnessAccumulator(channels=1, sample_rate=48000)
    # n_blocks == 0
    assert loudness._integrated_lufs() == -70.0
    # blocks below absolute gate
    loudness._blocks = [np.zeros(1) for _ in range(5)]
    assert loudness._integrated_lufs() == -70.0
