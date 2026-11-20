"""Unit tests for UnitSliceStore and sub-100ms master re-stitching with overrides."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.hashing import hash_file
from hawavoclean.server.unit_restitch import (
    UnitOverrideAction,
    UnitSliceStore,
    restitch_master_with_overrides,
)


def _tone(
    length: int, freq_hz: float, amp: float, sample_rate: int = 48000
) -> np.ndarray[Any, np.dtype[np.float32]]:
    t = np.arange(length, dtype=np.float64) / float(sample_rate)
    return np.ascontiguousarray(amp * np.sin(2.0 * math.pi * freq_hz * t), dtype=np.float32)


def test_unit_slice_store_roundtrip_and_retention(tmp_path: Path) -> None:
    store = UnitSliceStore(work_dir=tmp_path, max_jobs=2, max_age_s=3600.0)
    orig = _tone(4800, 300.0, 0.3)
    cand = _tone(4800, 300.0, 0.2)

    saved = store.save_unit(
        "job_001",
        unit_id=0,
        original=orig,
        candidate=cand,
        unit_meta={"is_speech": True, "forced_boundary": False},
        channel=0,
    )
    assert saved is not None
    assert saved.is_file()
    assert saved.parent == (tmp_path / "unit_slices" / "job_001").resolve()

    store.save_unit("job_001", unit_id=1, original=orig, candidate=None, channel=0)
    assert store.list_units("job_001") == [(0, 0), (0, 1)]

    loaded0 = store.load_unit("job_001", 0)
    np.testing.assert_array_equal(loaded0.original, orig)
    assert loaded0.candidate is not None
    np.testing.assert_array_equal(loaded0.candidate, cand)
    assert loaded0.unit_meta["is_speech"] is True

    loaded1 = store.load_unit("job_001", 1)
    assert loaded1.candidate is None

    # Create 3 jobs and verify bounded cleanup keeps 2 newest.
    for idx, jid in enumerate(["job_old", "job_mid", "job_new"]):
        store.save_unit(jid, unit_id=0, original=orig, candidate=cand)
        jdir = store.job_dir(jid)
        ts = time.time() - 100.0 + idx * 10.0
        os.utime(jdir, (ts, ts))

    removed = store.cleanup()
    assert "job_001" in removed or "job_old" in removed
    remaining_dirs = [p.name for p in store.base_dir.iterdir() if p.is_dir()]
    assert len(remaining_dirs) == 2
    assert "job_new" in remaining_dirs

    assert store.delete_job("job_new") is True
    assert store.delete_job("job_new") is False


def test_unit_slice_store_rejects_bad_ids_and_nan(tmp_path: Path) -> None:
    store = UnitSliceStore(work_dir=tmp_path)
    with pytest.raises(ValueError, match="Invalid job_id"):
        store.job_dir("../escape")
    with pytest.raises(ValueError, match="Invalid job_id"):
        store.job_dir("a/b")
    with pytest.raises(ValueError, match="NaN or Inf"):
        store.save_unit("job_ok", 0, np.asarray([0.1, float("nan")], dtype=np.float32), None)


def test_restitch_master_with_overrides_actions_and_continuity(tmp_path: Path) -> None:
    sr = 48000
    unit_len = 24000  # 0.5 s per unit -> 4 units = 2.0 s total
    store = UnitSliceStore(work_dir=tmp_path)
    job_id = "restitch_demo"

    # Build 4 contiguous units where original has high DC-offset/amplitude and
    # candidate is noticeably attenuated so overrides are unambiguously measurable.
    unit_records: list[dict[str, Any]] = []
    for u in range(4):
        orig = _tone(unit_len, 440.0, 0.6, sr)
        cand = _tone(unit_len, 440.0, 0.2, sr)
        store.save_unit(
            job_id,
            unit_id=u,
            original=orig,
            candidate=cand,
            unit_meta={"forced_boundary": u == 1},
            channel=0,
        )
        unit_records.append(
            {
                "unit_id": u,
                "channel": 0,
                "start_sample": u * unit_len,
                "end_sample": (u + 1) * unit_len,
                "is_speech": True,
                "forced_boundary": u == 1,
                "chosen_strength": 1.0 if u != 2 else 0.0,
                "final_decision": "ENHANCED" if u != 2 else "REVERTED",
            }
        )

    overrides: dict[int, tuple[UnitOverrideAction, float | None]] = {
        0: ("auto", None),  # stays ENHANCED (amp 0.2)
        1: ("force_original", None),  # forced to original (amp 0.6)
        2: ("force_enhanced", None),  # forced from REVERTED to ENHANCED (amp 0.2)
        3: ("custom_strength", 0.5),  # 50% blend: 0.6 + 0.5 * (0.2 - 0.6) = 0.4
    }
    master_out = tmp_path / "out" / "master.wav"
    result = restitch_master_with_overrides(
        job_dir=store.job_dir(job_id),
        unit_records=unit_records,
        overrides=overrides,
        master_output_path=master_out,
        sample_rate=sr,
        channels=1,
        dither=False,
    )

    assert master_out.is_file()
    assert result["master_sha256"] == hash_file(master_out)
    assert result["total_samples"] == 4 * unit_len
    assert result["restitch_ms"] >= 0.0
    summary = result["summary"]
    assert summary["units_total"] == 4
    assert summary["enhanced"] == 3
    assert summary["reverted"] == 1
    assert summary["manual_overrides"] == 3
    assert summary["continuity_crossfaded"] >= 2  # units 0 and 2 border original unit 1

    decoded, dec_sr = sf.read(str(master_out), dtype="float32")
    assert dec_sr == sr
    assert decoded.shape == (4 * unit_len,)

    # Check interior peak amplitude of each unit (away from boundary tapers).
    margin = 4000
    peaks = [
        float(np.max(np.abs(decoded[u * unit_len + margin : (u + 1) * unit_len - margin])))
        for u in range(4)
    ]
    assert abs(peaks[0] - 0.2) < 0.01
    assert abs(peaks[1] - 0.6) < 0.01
    assert abs(peaks[2] - 0.2) < 0.01
    assert abs(peaks[3] - 0.4) < 0.01

    # Verify continuity taper eliminated hard step clicks at unit 0|1 and 1|2 seams.
    for seam in (unit_len, 2 * unit_len):
        jump = abs(float(decoded[seam]) - float(decoded[seam - 1]))
        # Max single-sample step of a 440 Hz, 0.6-amp sine at 48 kHz is ~0.035.
        assert jump < 0.05

    # Re-stitching with different override atomically replaces file and changes sha256.
    result2 = restitch_master_with_overrides(
        job_dir=store.job_dir(job_id),
        unit_records=unit_records,
        overrides={1: ("force_enhanced", None)},
        master_output_path=master_out,
        sample_rate=sr,
        channels=1,
        dither=False,
    )
    assert result2["master_sha256"] != result["master_sha256"]
    assert result2["master_sha256"] == hash_file(master_out)


def test_restitch_stereo_and_performance(tmp_path: Path) -> None:
    sr = 48000
    unit_len = 24000
    store = UnitSliceStore(work_dir=tmp_path)
    job_id = "stereo_fast"
    records: list[dict[str, Any]] = []

    for ch in range(2):
        for u in range(4):
            uid = ch * 10 + u
            orig = _tone(unit_len, 300.0 + ch * 150.0, 0.4, sr)
            cand = _tone(unit_len, 300.0 + ch * 150.0, 0.2, sr)
            store.save_unit(job_id, uid, orig, cand, channel=ch)
            records.append(
                {
                    "unit_id": uid,
                    "channel": ch,
                    "start_sample": u * unit_len,
                    "end_sample": (u + 1) * unit_len,
                    "is_speech": True,
                    "chosen_strength": 1.0,
                    "final_decision": "ENHANCED",
                }
            )

    out_path = tmp_path / "stereo_master.wav"
    res = restitch_master_with_overrides(
        job_dir=store.job_dir(job_id),
        unit_records=records,
        overrides={0: ("force_original", None), 10: ("custom_strength", 0.5)},
        master_output_path=out_path,
        sample_rate=sr,
        channels=2,
    )
    decoded, dec_sr = sf.read(str(out_path), dtype="float32", always_2d=True)
    assert dec_sr == sr
    assert decoded.shape == (4 * unit_len, 2)
    assert res["channels"] == 2
    assert res["summary"]["manual_overrides"] == 2
    # Must be well under interactive latency budget (generous CI margin < 1000 ms).
    assert res["restitch_ms"] < 1000.0


def test_restitch_validation_errors(tmp_path: Path) -> None:
    store = UnitSliceStore(work_dir=tmp_path)
    orig = _tone(4800, 440.0, 0.2)
    store.save_unit("job_val", 0, orig, None)
    recs = [
        {
            "unit_id": 0,
            "channel": 0,
            "start_sample": 0,
            "end_sample": 4800,
            "is_speech": True,
            "final_decision": "REVERTED",
        }
    ]

    with pytest.raises(ValueError, match="must not be empty"):
        restitch_master_with_overrides(store.job_dir("job_val"), [], {}, tmp_path / "m.wav")
    with pytest.raises(ValueError, match="no cached candidate"):
        restitch_master_with_overrides(
            store.job_dir("job_val"),
            recs,
            {0: ("force_enhanced", None)},
            tmp_path / "m.wav",
        )
    with pytest.raises(ValueError, match="custom_strength"):
        restitch_master_with_overrides(
            store.job_dir("job_val"),
            recs,
            {0: ("custom_strength", 1.5)},
            tmp_path / "m.wav",
        )
    with pytest.raises(ValueError, match="no cached candidate"):
        restitch_master_with_overrides(
            store.job_dir("job_val"),
            recs,
            {0: ("custom_strength", 0.5)},
            tmp_path / "m.wav",
        )
    with pytest.raises(ValueError, match="sample_rate"):
        restitch_master_with_overrides(
            store.job_dir("job_val"), recs, {}, tmp_path / "m.wav", sample_rate=0
        )
    with pytest.raises(ValueError, match="channels"):
        restitch_master_with_overrides(
            store.job_dir("job_val"), recs, {}, tmp_path / "m.wav", channels=0
        )
    with pytest.raises(ValueError, match="out of range"):
        restitch_master_with_overrides(
            store.job_dir("job_val"),
            [{**recs[0], "channel": 2}],
            {},
            tmp_path / "m.wav",
            channels=1,
        )
    with pytest.raises(ValueError, match="invalid sample bounds"):
        restitch_master_with_overrides(
            store.job_dir("job_val"),
            [{**recs[0], "start_sample": 100, "end_sample": 100}],
            {},
            tmp_path / "m.wav",
        )


def test_unit_slice_store_and_restitch_edge_cases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_jobs"):
        UnitSliceStore(work_dir=tmp_path, max_jobs=0)
    with pytest.raises(ValueError, match="max_age_s"):
        UnitSliceStore(work_dir=tmp_path, max_age_s=0.0)

    store = UnitSliceStore(work_dir=tmp_path, max_jobs=4, max_age_s=60.0)
    assert store.cleanup() == []
    assert store.list_units("nonexistent_job") == []

    with pytest.raises(ValueError, match="unit_id"):
        store.save_unit("job_e", -1, _tone(100, 440.0, 0.1), None)
    with pytest.raises(ValueError, match="channel"):
        store.save_unit("job_e", 0, _tone(100, 440.0, 0.1), None, channel=-1)
    with pytest.raises(ValueError, match="1D"):
        store.save_unit("job_e", 0, np.zeros((2, 10), dtype=np.float32), None)
    with pytest.raises(ValueError, match="1D"):
        store.save_unit("job_e", 0, _tone(10, 440.0, 0.1), np.zeros((2, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        store.save_unit(
            "job_e", 0, _tone(10, 440.0, 0.1), np.asarray([float("inf")], dtype=np.float32)
        )
    with pytest.raises(FileNotFoundError):
        store.load_unit("job_e", 99)

    # Save slightly shorter and longer slices + no-speech unit + custom_strength=0.0 + empty ch1.
    store.save_unit("job_edge", 0, _tone(2000, 440.0, 0.3), _tone(3000, 440.0, 0.1), channel=0)
    store.save_unit("job_edge", 1, _tone(2400, 440.0, 0.3), None, channel=0)
    recs = [
        {
            "unit_id": 0,
            "channel": 0,
            "start_sample": 0,
            "end_sample": 2400,
            "is_speech": True,
            "chosen_strength": 0.5,
            "final_decision": "ENHANCED",
        },
        {
            "unit_id": 1,
            "channel": 0,
            "start_sample": 2400,
            "end_sample": 4800,
            "is_speech": False,
            "chosen_strength": 0.0,
            "final_decision": "NO_SPEECH",
        },
    ]
    res = restitch_master_with_overrides(
        store.job_dir("job_edge"),
        recs,
        {0: ("custom_strength", 0.0)},
        tmp_path / "edge_master.wav",
        sample_rate=48000,
        channels=2,  # channel 1 intentionally empty
    )
    assert res["summary"]["reverted"] == 1
    assert res["summary"]["no_speech"] == 1
    assert res["summary"]["manual_overrides"] == 1

    # Expired job pruned by max_age_s.
    jdir = store.job_dir("job_edge")
    os.utime(jdir, (10.0, 10.0))
    assert "job_edge" in store.cleanup()


def test_unit_restitch_corruption_error_cleanup_and_branch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import cast

    from hawavoclean.server import unit_restitch
    from hawavoclean.server.unit_restitch import _blend_strength, load_unit_slice

    # _blend_strength <= 0.0 returns original copy.
    orig = _tone(1600, 440.0, 0.5)
    cand = _tone(1600, 440.0, 0.2)
    np.testing.assert_array_equal(_blend_strength(orig, cand, 0.0), orig)

    store = UnitSliceStore(work_dir=tmp_path, max_jobs=4, max_age_s=3600.0)
    jdir = store.job_dir("job_branches")
    store.save_unit("job_branches", 0, orig, cand)

    # Non-file subdir and unrelated file inside job_dir are ignored by list_units.
    (jdir / "subdir").mkdir()
    (jdir / "notes.txt").write_text("ignore", encoding="utf-8")
    assert store.list_units("job_branches") == [(0, 0)]

    # Regular file inside unit_slices base_dir is ignored by cleanup().
    (store.base_dir / "stray.txt").write_text("ignore", encoding="utf-8")
    assert store.cleanup() == []

    # Non-dict meta_json, truncated .npz, non-1D array, NaN array, non-scalar has_candidate fail closed.
    bad_meta_path = jdir / "unit_000001_ch0.npz"
    np.savez(
        bad_meta_path,
        original=orig,
        candidate=cand,
        has_candidate=np.asarray(True, dtype=np.bool_),
        meta_json=np.frombuffer(b"[1,2,3]", dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Corrupt unit metadata"):
        load_unit_slice(jdir, 1)

    bad_flag_path = jdir / "unit_000002_ch0.npz"
    np.savez(
        bad_flag_path,
        original=orig,
        candidate=cand,
        has_candidate=np.asarray([True, False], dtype=np.bool_),
        meta_json=np.frombuffer(b"{}", dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Corrupt cached slice"):
        load_unit_slice(jdir, 2)

    bad_orig_path = jdir / "unit_000003_ch0.npz"
    np.savez(
        bad_orig_path,
        original=np.asarray([0.1, float("nan")], dtype=np.float32),
        candidate=cand,
        has_candidate=np.asarray(True, dtype=np.bool_),
        meta_json=np.frombuffer(b"{}", dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Corrupt original waveform"):
        load_unit_slice(jdir, 3)

    bad_cand_path = jdir / "unit_000004_ch0.npz"
    np.savez(
        bad_cand_path,
        original=orig,
        candidate=np.asarray([0.1, float("inf")], dtype=np.float32),
        has_candidate=np.asarray(True, dtype=np.bool_),
        meta_json=np.frombuffer(b"{}", dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Corrupt candidate waveform"):
        load_unit_slice(jdir, 4)

    # Symlinked slice file refused.
    sym_slice = jdir / "unit_000005_ch0.npz"
    try:
        sym_slice.symlink_to(jdir / "unit_000000_ch0.npz")
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match="symlinked cached slice"):
            load_unit_slice(jdir, 5)

    # Unsupported override action rejected cleanly.
    recs = [
        {
            "unit_id": 0,
            "channel": 0,
            "start_sample": 0,
            "end_sample": 1600,
            "is_speech": True,
            "final_decision": "ENHANCED",
            "chosen_strength": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="Unsupported override action"):
        restitch_master_with_overrides(
            jdir,
            recs,
            {0: (cast(UnitOverrideAction, "bogus_action"), None)},
            tmp_path / "m_bad.wav",
        )

    # Failed os.replace in save_unit_slice_to_dir cleans up temp file.
    orig_replace = os.replace

    def _fail_replace(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("replace blocked")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError, match="replace blocked"):
        store.save_unit("job_fail_save", 0, orig, cand)
    assert not list(store.job_dir("job_fail_save").glob("*.tmp"))

    # Failed encode_audio during restitch leaves existing master WAV untouched and removes temp file.
    monkeypatch.setattr(os, "replace", orig_replace)
    master = tmp_path / "existing_master.wav"
    sentinel = b"EXISTING_MASTER_SENTINEL"
    master.write_bytes(sentinel)

    def _fail_encode(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("encode boom")

    monkeypatch.setattr(unit_restitch, "encode_audio", _fail_encode)
    with pytest.raises(OSError, match="encode boom"):
        restitch_master_with_overrides(jdir, recs, {}, master)
    assert master.read_bytes() == sentinel
    assert not list(tmp_path.glob("*.tmp.wav"))
