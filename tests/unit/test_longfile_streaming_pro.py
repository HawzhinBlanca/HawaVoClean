"""Pro long-file streaming hardening tests and 30-minute synthetic soak proof.

Verifies all three long-file OOM / disk-exhaustion fixes:
1. Streaming windowed splice for ``POST /api/v1/spectral-repair`` (``stream_spectral_box_repair``).
2. Bounded per-job + global LRU disk budget on ``UnitSliceStore`` with ``truncated.json`` fail-closed
   handling in ``restitch_master_with_overrides``, ``pipeline.py``, and ``POST /api/v1/jobs/{id}/override``.
3. Two-pass streaming delivery loudness (``apply_delivery_loudness_streaming`` & ``_run_delivery_export``)
   with parity against in-memory ``apply_delivery_loudness`` within <= 0.05 LU.
4. 30-minute (1800 s, 48 kHz) synthetic soak test proving bounded ``tracemalloc`` peak memory (< 80 MB).
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import scipy.signal
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean.audio.delivery import (
    DeliveryPreset,
    apply_delivery_loudness,
    apply_delivery_loudness_streaming,
    measure_delivery_loudness_streaming,
)
from hawavoclean.finishing.pro_repair import apply_spectral_box_repair
from hawavoclean.server.app import (
    _run_delivery_export,
    create_app,
    stream_spectral_box_repair,
)
from hawavoclean.server.contracts import MAX_SPECTROGRAM_WINDOW_S
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager
from hawavoclean.server.unit_restitch import (
    MAX_UNIT_SLICE_JOB_BYTES,
    MAX_UNIT_SLICE_MB_ENV,
    MAX_UNIT_SLICE_TOTAL_BYTES,
    MAX_UNIT_SLICE_TOTAL_MB_ENV,
    SLICE_TRUNCATED_MARKER,
    SliceCacheBudgetError,
    UnitSliceStore,
    dir_slice_bytes,
    read_slice_truncated_marker,
    restitch_master_with_overrides,
    save_unit_slice_to_dir,
    write_slice_truncated_marker,
)

pytestmark = pytest.mark.unit
SR = 48000
TOKEN = "longfile-pro-token"
HEADERS = {"X-Hawa-Token": TOKEN}


def _make_dynamic_signal(
    duration_s: float,
    *,
    channels: int = 2,
    sr: int = SR,
    add_spikes: bool = True,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Build deterministic multi-level speech-like signal with optional peak spikes."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    # Slow amplitude modulation so LRA is non-trivial (> 5 LU on >= 8 s clips).
    env = 0.06 + 0.14 * (0.5 + 0.5 * np.sin(2.0 * math.pi * 0.25 * t))
    base = env * (
        0.65 * np.sin(2.0 * math.pi * 240.0 * t)
        + 0.25 * np.sin(2.0 * math.pi * 480.0 * t)
        + 0.10 * np.sin(2.0 * math.pi * 960.0 * t)
    )
    if channels == 1:
        arr = base.astype(np.float32)[np.newaxis, :]
    else:
        right = env * (
            0.60 * np.sin(2.0 * math.pi * 240.0 * t + 0.3)
            + 0.30 * np.sin(2.0 * math.pi * 510.0 * t)
            + 0.10 * np.sin(2.0 * math.pi * 1020.0 * t)
        )
        arr = np.vstack([base.astype(np.float32), right.astype(np.float32)])
    if add_spikes and n > 20000:
        # Place sharp inter-sample transients near typical streaming chunk seams.
        for pos in (16380, 16384, 32768, n // 2, n - 4096):
            if 2 <= pos < n - 2:
                arr[:, pos - 1 : pos + 2] = np.float32(0.92)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _tone_1d(
    length: int, freq_hz: float, amp: float, sr: int = SR
) -> np.ndarray[Any, np.dtype[np.float32]]:
    t = np.arange(length, dtype=np.float64) / float(sr)
    return np.ascontiguousarray(amp * np.sin(2.0 * math.pi * freq_hz * t), dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Streaming Windowed Spectral Repair
# ---------------------------------------------------------------------------


def test_stream_spectral_box_repair_exterior_bit_identical_and_band_attenuated(
    tmp_path: Path,
) -> None:
    """Streaming spectral repair keeps exterior PCM samples bit-identical and attenuates target box."""
    duration_s = 3.0
    n = int(round(duration_s * SR))
    t = np.arange(n, dtype=np.float64) / float(SR)

    speech_l = 0.22 * np.sin(2.0 * math.pi * 280.0 * t)
    speech_r = 0.20 * np.sin(2.0 * math.pi * 340.0 * t)
    chime = np.zeros(n, dtype=np.float64)
    s0, s1 = int(1.0 * SR), int(1.8 * SR)
    chime[s0:s1] = 0.28 * np.sin(2.0 * math.pi * 1000.0 * t[s0:s1])
    stereo = np.column_stack(
        [(speech_l + chime).astype(np.float32), (speech_r + chime).astype(np.float32)]
    )

    src = tmp_path / "stereo_src.wav"
    out = tmp_path / "stereo_repaired.wav"
    sf.write(str(src), stereo, SR, format="WAV", subtype="PCM_24")

    # Use tiny chunk_samples so splice crosses multiple stream chunks.
    res_start, res_end = stream_spectral_box_repair(
        src,
        out,
        start_s=1.0,
        end_s=1.8,
        f_low_hz=850.0,
        f_high_hz=1150.0,
        attenuation_db=24.0,
        chunk_samples=16384,
    )
    assert res_start == pytest.approx(1.0)
    assert res_end == pytest.approx(1.8)
    assert out.is_file()

    orig_i32, orig_sr = sf.read(str(src), dtype="int32", always_2d=True)
    rep_i32, rep_sr = sf.read(str(out), dtype="int32", always_2d=True)
    assert rep_sr == orig_sr == SR
    assert rep_i32.shape == orig_i32.shape

    # Exterior outside [start_s - 0.02, end_s + 0.02] must be 100% bit-identical.
    guard = int(0.02 * SR)
    assert np.array_equal(orig_i32[: s0 - guard], rep_i32[: s0 - guard])
    assert np.array_equal(orig_i32[s1 + guard :], rep_i32[s1 + guard :])

    # Target chime inside box attenuated > 15 dB; speech band preserved (< 0.5 dB).
    orig_f32, _ = sf.read(str(src), dtype="float32", always_2d=True)
    rep_f32, _ = sf.read(str(out), dtype="float32", always_2d=True)
    sos_chime = scipy.signal.butter(4, [900.0, 1100.0], btype="bandpass", fs=SR, output="sos")
    sos_speech = scipy.signal.butter(4, [200.0, 420.0], btype="bandpass", fs=SR, output="sos")
    for ch in range(2):
        c_before = float(
            np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_chime, orig_f32[s0:s1, ch]) ** 2))
        )
        c_after = float(
            np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_chime, rep_f32[s0:s1, ch]) ** 2))
        )
        assert 20.0 * math.log10(c_before / max(c_after, 1e-9)) > 15.0

        v_before = float(
            np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_speech, orig_f32[s0:s1, ch]) ** 2))
        )
        v_after = float(
            np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_speech, rep_f32[s0:s1, ch]) ** 2))
        )
        assert abs(20.0 * math.log10(v_after / max(v_before, 1e-9))) < 0.5

    # Parity against full-file in-memory apply_spectral_box_repair.
    ref_full = apply_spectral_box_repair(
        orig_f32.T,
        SR,
        start_s=1.0,
        end_s=1.8,
        f_low_hz=850.0,
        f_high_hz=1150.0,
        attenuation_db=24.0,
    )
    assert float(np.max(np.abs(rep_f32.T - ref_full))) < 1e-4


def test_spectral_repair_window_cap_300s_rejected_as_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repair window exceeding MAX_SPECTROGRAM_WINDOW_S (300 s) fails closed with ValueError / 400."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))

    src = tmp_path / "short.wav"
    sf.write(str(src), _tone_1d(SR, 440.0, 0.2), SR, format="WAV", subtype="PCM_24")

    with pytest.raises(ValueError, match=r"exceeds maximum"):
        stream_spectral_box_repair(
            src,
            tmp_path / "out.wav",
            start_s=0.0,
            end_s=MAX_SPECTROGRAM_WINDOW_S + 1.0,
            f_low_hz=500.0,
            f_high_hz=1500.0,
        )

    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            resp = client.post(
                "/api/v1/spectral-repair",
                headers=HEADERS,
                json={
                    "path": str(src),
                    "startS": 0.0,
                    "endS": MAX_SPECTROGRAM_WINDOW_S + 5.0,
                    "fLowHz": 500.0,
                    "fHighHz": 1500.0,
                },
            )
            assert resp.status_code == 400
            assert resp.json()["error"] == "bad_request"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# 2. Bounded Disk Budget on UnitSliceStore & Fail-Closed Re-Stitch
# ---------------------------------------------------------------------------


def test_unit_slice_store_per_job_budget_truncates_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-job byte cap stops writing cleanly, writes truncated.json, and blocks restitch."""
    assert MAX_UNIT_SLICE_JOB_BYTES == 256 * 1024 * 1024
    assert MAX_UNIT_SLICE_TOTAL_BYTES == 1024 * 1024 * 1024

    # ~40 KB per unit (5000 float32 orig + 5000 float32 cand = 40 KB + npz headers).
    # Set env budget to 0.1 MiB (~104,858 bytes) so only 2 units fit cleanly.
    monkeypatch.setenv(MAX_UNIT_SLICE_MB_ENV, "0.1")
    store = UnitSliceStore(work_dir=tmp_path)
    assert store.max_job_bytes == int(round(0.1 * 1024 * 1024))

    orig = _tone_1d(5000, 300.0, 0.3)
    cand = _tone_1d(5000, 300.0, 0.2)

    results: list[Path | None] = []
    unit_records: list[dict[str, Any]] = []
    for uid in range(6):
        saved = store.save_unit(
            "job_capped",
            unit_id=uid,
            original=orig,
            candidate=cand,
            unit_meta={"start_sample": uid * 5000, "end_sample": (uid + 1) * 5000},
            channel=0,
        )
        results.append(saved)
        unit_records.append(
            {
                "unit_id": uid,
                "channel": 0,
                "start_sample": uid * 5000,
                "end_sample": (uid + 1) * 5000,
                "is_speech": True,
                "final_decision": "ENHANCED",
                "chosen_strength": 1.0,
            }
        )

    assert results[0] is not None and results[1] is not None
    assert all(r is None for r in results[2:])
    assert store.is_truncated("job_capped") is True
    assert store.list_units("job_capped") == [(0, 0), (0, 1)]
    assert dir_slice_bytes(store.job_dir("job_capped")) <= store.max_job_bytes + 512

    marker_path = store.job_dir("job_capped") / SLICE_TRUNCATED_MARKER
    assert marker_path.is_file()
    marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_data["truncated"] is True
    assert marker_data["reason"] == "slice_cache_budget_exceeded"
    assert marker_data["limit_bytes"] == store.max_job_bytes

    # restitch_master_with_overrides must fail closed with SliceCacheBudgetError (subclass of ValueError).
    master_out = tmp_path / "should_not_exist.wav"
    with pytest.raises(SliceCacheBudgetError, match="slice_cache_budget_exceeded") as exc_info:
        restitch_master_with_overrides(
            store.job_dir("job_capped"),
            unit_records,
            {0: ("force_original", None)},
            master_out,
            sample_rate=SR,
            channels=1,
        )
    assert isinstance(exc_info.value, ValueError)
    assert not master_out.exists()

    # Corrupt truncated.json also fails closed.
    marker_path.write_text("{not valid json", encoding="utf-8")
    assert read_slice_truncated_marker(store.job_dir("job_capped")) is not None
    with pytest.raises(SliceCacheBudgetError):
        restitch_master_with_overrides(
            store.job_dir("job_capped"),
            unit_records[:2],
            {},
            master_out,
        )


def test_unit_slice_store_global_lru_pruning_evicts_oldest_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Global byte cap evicts oldest sibling job directories under unit_slices/ via LRU."""
    # ~41 KB per job (1 unit of 5000 samples orig + cand). Cap total at 0.1 MiB (~105 KB) -> max 2 jobs.
    monkeypatch.setenv(MAX_UNIT_SLICE_MB_ENV, "1.0")
    monkeypatch.setenv(MAX_UNIT_SLICE_TOTAL_MB_ENV, "0.1")

    store = UnitSliceStore(work_dir=tmp_path)
    orig = _tone_1d(5000, 440.0, 0.3)
    cand = _tone_1d(5000, 440.0, 0.15)

    base_ts = time.time() - 500.0
    for idx, jid in enumerate(["job_1_old", "job_2_mid", "job_3_newer", "job_4_newest"]):
        saved = store.save_unit(jid, unit_id=0, original=orig, candidate=cand)
        assert saved is not None
        jdir = store.job_dir(jid)
        ts = base_ts + idx * 60.0
        os.utime(jdir, (ts, ts))

    remaining = sorted(p.name for p in store.base_dir.iterdir() if p.is_dir())
    assert "job_1_old" not in remaining
    assert "job_2_mid" not in remaining
    assert "job_4_newest" in remaining
    total_disk_bytes = sum(dir_slice_bytes(store.base_dir / name) for name in remaining)
    assert total_disk_bytes <= store.max_total_bytes


def test_save_unit_slice_direct_and_http_override_returns_409_on_truncated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP POST /api/v1/jobs/{id}/override returns 409 slice_cache_budget_exceeded when truncated."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("HAWAVOCLEAN_LAVASR_MODEL_DIR", str(tmp_path / "no-lavasr"))
    # Force slice cache truncation during real job execution (tiny 0.001 MiB ~ 1 KB cap).
    monkeypatch.setenv(MAX_UNIT_SLICE_MB_ENV, "0.001")

    # Build a 1.5 s speech-like WAV and run a real job through JobManager.
    n = int(1.5 * SR)
    t = np.arange(n, dtype=np.float64) / float(SR)
    rng = np.random.default_rng(7)
    env = np.zeros(n, dtype=np.float64)
    env[int(0.2 * SR) : int(1.2 * SR)] = 1.0
    sig = (
        env * (0.20 * np.sin(2.0 * math.pi * 180.0 * t) + 0.10 * np.sin(2.0 * math.pi * 360.0 * t))
        + 0.02 * rng.standard_normal(n)
    ).astype(np.float32)
    src = tmp_path / "budget_job_src.wav"
    sf.write(str(src), sig, SR, format="WAV", subtype="PCM_24")

    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            up = client.post(
                "/api/upload",
                headers=HEADERS,
                files={"file": ("budget_job.wav", src.read_bytes(), "audio/wav")},
            )
            assert up.status_code == 200, up.text
            src_id = up.json()["source_id"]

            sub = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [src_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "production",
                        "allowGenerativeReconstruction": False,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "slice-budget-409-test",
                },
            )
            assert sub.status_code == 202, sub.text
            job_id = sub.json()["jobs"][0]["jobId"]

            deadline = time.monotonic() + 60.0
            snap: dict[str, Any] = {}
            while time.monotonic() < deadline:
                snap = client.get(f"/api/jobs/{job_id}", headers=HEADERS).json()
                if snap["state"] in TERMINAL_STATES:
                    break
                time.sleep(0.05)
            assert snap["state"] == "done", snap

            # Pipeline completed cleanly, wrote truncated.json, and kept slice dir tiny.
            jdir = UnitSliceStore(tmp_path).job_dir(job_id)
            assert (jdir / SLICE_TRUNCATED_MARKER).is_file()
            assert read_slice_truncated_marker(jdir) is not None

            # Calling override MUST fail closed with 409 slice_cache_budget_exceeded.
            ov = client.post(
                f"/api/v1/jobs/{job_id}/override",
                headers=HEADERS,
                json={"unitIndex": 0, "decision": "force_original"},
            )
            assert ov.status_code == 409, ov.text
            assert ov.json()["error"] == "slice_cache_budget_exceeded"
    finally:
        manager.shutdown()

    # Also verify direct save_unit_slice_to_dir respects already-present truncated.json.
    direct_dir = tmp_path / "direct_job"
    write_slice_truncated_marker(direct_dir, limit_bytes=1024, used_bytes=1024, unit_id=0)
    assert save_unit_slice_to_dir(direct_dir, 1, _tone_1d(100, 440.0, 0.1), None) is None


# ---------------------------------------------------------------------------
# 3. Streaming Delivery Loudness Parity & Export Wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["match_input", "ebu_r128", "atsc_a85", "podcast", "streaming"])
@pytest.mark.parametrize("channels", [1, 2])
def test_apply_delivery_loudness_streaming_parity_all_presets(
    tmp_path: Path,
    preset: DeliveryPreset,
    channels: int,
) -> None:
    """Streaming delivery loudness matches in-memory apply_delivery_loudness within <= 0.05 LU."""
    wave = _make_dynamic_signal(8.0, channels=channels, add_spikes=True)
    master = tmp_path / f"master_{preset}_ch{channels}.wav"
    out = tmp_path / f"delivered_{preset}_ch{channels}.wav"
    sf.write(str(master), wave.T, SR, format="WAV", subtype="FLOAT")

    ref_wave, ref_metrics = apply_delivery_loudness(wave, SR, preset)
    # Use a small chunk_samples so peaks, lookahead ramps, and release tails cross chunk seams!
    stream_metrics = apply_delivery_loudness_streaming(
        master,
        out,
        preset,
        chunk_samples=16384,
    )

    assert out.is_file()
    assert stream_metrics["preset"] == ref_metrics["preset"]
    assert stream_metrics["limiter_mode"] == ref_metrics["limiter_mode"]
    assert stream_metrics["ceiling_dbtp"] == ref_metrics["ceiling_dbtp"]
    assert float(stream_metrics["input_lufs"]) == pytest.approx(
        float(ref_metrics["input_lufs"]), abs=0.05
    )
    assert float(stream_metrics["integrated_lufs"]) == pytest.approx(
        float(ref_metrics["integrated_lufs"]), abs=0.05
    )
    assert float(stream_metrics["loudness_range_lu"]) == pytest.approx(
        float(ref_metrics["loudness_range_lu"]), abs=0.05
    )
    assert float(stream_metrics["true_peak_dbtp"]) == pytest.approx(
        float(ref_metrics["true_peak_dbtp"]), abs=0.05
    )
    assert float(stream_metrics["limiter_max_gr_db"]) == pytest.approx(
        float(ref_metrics["limiter_max_gr_db"]), abs=0.05
    )

    stream_audio, out_sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert out_sr == SR
    stream_2d = stream_audio.T
    ref_2d = ref_wave[np.newaxis, :] if ref_wave.ndim == 1 else ref_wave
    assert stream_2d.shape == ref_2d.shape
    if preset == "match_input":
        assert np.array_equal(stream_2d, ref_2d)
    else:
        assert float(np.max(np.abs(stream_2d - ref_2d))) < 1e-4


def test_apply_delivery_loudness_streaming_match_input_gain_short_and_errors(
    tmp_path: Path,
) -> None:
    """Verify match_input with explicit input_lufs, short clips, silence, and fail-closed errors."""
    # Match input with explicit target input_lufs
    wave = _make_dynamic_signal(4.0, channels=1, add_spikes=False)
    master = tmp_path / "match_gain_src.wav"
    out = tmp_path / "match_gain_out.wav"
    sf.write(str(master), wave.T, SR, format="WAV", subtype="FLOAT")

    _, ref_m = apply_delivery_loudness(wave[0], SR, "match_input", input_lufs=-18.0)
    st_m = apply_delivery_loudness_streaming(
        master, out, "match_input", input_lufs=-18.0, chunk_samples=12000
    )
    assert st_m["limiter_mode"] == "bypass"
    assert float(st_m["integrated_lufs"]) == pytest.approx(
        float(ref_m["integrated_lufs"]), abs=0.05
    )
    assert float(st_m["applied_gain_db"]) == pytest.approx(
        float(ref_m["applied_gain_db"]), abs=0.05
    )

    # Short clip (< 0.4 s) + streaming measure parity
    short = _tone_1d(4000, 440.0, 0.25)
    short_wav = tmp_path / "short_clip.wav"
    short_out = tmp_path / "short_clip_out.wav"
    sf.write(str(short_wav), short, SR, format="WAV", subtype="FLOAT")
    _, ref_short = apply_delivery_loudness(short, SR, "ebu_r128")
    st_short = apply_delivery_loudness_streaming(
        short_wav, short_out, "ebu_r128", chunk_samples=1024
    )
    assert float(st_short["integrated_lufs"]) == pytest.approx(
        float(ref_short["integrated_lufs"]), abs=0.05
    )

    # Silence does not explode or trigger RuntimeWarnings
    sil_wav = tmp_path / "silence.wav"
    sil_out = tmp_path / "silence_out.wav"
    sf.write(str(sil_wav), np.zeros((SR, 2), dtype=np.float32), SR, format="WAV", subtype="FLOAT")
    st_sil = apply_delivery_loudness_streaming(sil_wav, sil_out, "streaming", chunk_samples=8192)
    assert float(st_sil["applied_gain_db"]) == 0.0
    assert float(st_sil["loudness_range_lu"]) == 0.0

    # measure_delivery_loudness_streaming + _run_delivery_export match_input & ebu_r128
    meas, lra = measure_delivery_loudness_streaming(master, chunk_samples=8192)
    assert math.isfinite(meas.integrated_lufs)
    assert lra >= 0.0

    exp_out = tmp_path / "exported_streaming.wav"
    exp_res = _run_delivery_export(
        master_resolved=master,
        dest_resolved=exp_out,
        fmt="wav24",
        delivery_preset="ebu_r128",
        source_resolved=None,
        preserve_metadata=False,
    )
    assert Path(exp_res["outputPath"]).is_file()
    assert exp_res["integratedLufs"] == pytest.approx(-23.0, abs=0.5)

    # Validation / security fail-closed checks
    with pytest.raises(ValueError, match="Unsupported delivery preset"):
        apply_delivery_loudness_streaming(master, tmp_path / "x.wav", cast(DeliveryPreset, "bogus"))
    with pytest.raises(ValueError, match="overwrite master_wav in place"):
        apply_delivery_loudness_streaming(master, master, "ebu_r128")
    sym_out = tmp_path / "sym_out.wav"
    sym_out.symlink_to(tmp_path / "target.wav")
    with pytest.raises(ValueError, match="symlink"):
        apply_delivery_loudness_streaming(master, sym_out, "ebu_r128")


# ---------------------------------------------------------------------------
# 4. 30-Minute (1800 s, 48 kHz) Synthetic Soak Proof
# ---------------------------------------------------------------------------


def test_30_minute_synthetic_soak_streaming_spectral_repair_and_delivery_loudness(
    tmp_path: Path,
) -> None:
    """Prove < 80 MB tracemalloc peak RAM on a 30-minute 48 kHz WAV for spectral repair & delivery."""
    duration_s = 1800.0  # 30 minutes = 86,400,000 samples (~259 MB PCM_24 on disk)
    total_samples = int(duration_s * SR)
    gen_chunk = 1 << 18  # 262,144 samples (~5.46 s) per write chunk
    chime_start_s = 900.0
    chime_end_s = 901.0
    chime_s0 = int(chime_start_s * SR)
    chime_s1 = int(chime_end_s * SR)

    long_wav = tmp_path / "soak_30min_48k.wav"
    with sf.SoundFile(
        str(long_wav),
        mode="w",
        samplerate=SR,
        channels=1,
        subtype="PCM_24",
        format="WAV",
    ) as writer:
        for offset in range(0, total_samples, gen_chunk):
            n = min(gen_chunk, total_samples - offset)
            t = (offset + np.arange(n, dtype=np.float64)) / float(SR)
            # Dynamic level modulation (quiet & loud phrases every 30 s) so LRA > 4 LU.
            phrase_amp = 0.05 + 0.17 * (0.5 + 0.5 * np.sin(2.0 * math.pi * (t / 30.0)))
            sig = phrase_amp * (
                0.70 * np.sin(2.0 * math.pi * 260.0 * t) + 0.30 * np.sin(2.0 * math.pi * 520.0 * t)
            )
            # Inject a 1 kHz chime specifically at [900.0 s, 901.0 s).
            ov0 = max(offset, chime_s0)
            ov1 = min(offset + n, chime_s1)
            if ov1 > ov0:
                sig[ov0 - offset : ov1 - offset] += 0.30 * np.sin(
                    2.0 * math.pi * 1000.0 * t[ov0 - offset : ov1 - offset]
                )
            # Inject a sharp transient at 600 s to exercise the streaming limiter.
            spike_pos = 600 * SR
            if offset <= spike_pos < offset + n:
                sig[spike_pos - offset] = 0.90
            writer.write(sig.astype(np.float32)[:, np.newaxis])

    assert long_wav.stat().st_size > 240 * 1024 * 1024

    # --- Soak A: Streaming Spectral Box Repair at the 15-minute midpoint ---
    repaired_wav = tmp_path / "soak_30min_repaired.wav"
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        stream_spectral_box_repair(
            long_wav,
            repaired_wav,
            start_s=chime_start_s,
            end_s=chime_end_s,
            f_low_hz=850.0,
            f_high_hz=1150.0,
            attenuation_db=24.0,
        )
        _, spectral_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    spectral_elapsed_s = time.perf_counter() - t0
    spectral_peak_mb = spectral_peak_bytes / (1024.0 * 1024.0)

    assert repaired_wav.is_file()
    assert spectral_peak_mb < 80.0, f"Spectral repair peak RAM too high: {spectral_peak_mb:.2f} MiB"

    # Verify exterior bit-identity at both ends (10 s and 1750 s) and chime attenuation at 900 s.
    with (
        sf.SoundFile(str(long_wav), mode="r") as r_orig,
        sf.SoundFile(str(repaired_wav), mode="r") as r_rep,
    ):
        assert r_rep.frames == total_samples
        for check_start in (10 * SR, 1750 * SR):
            r_orig.seek(check_start)
            r_rep.seek(check_start)
            o_blk = r_orig.read(frames=48000, dtype="int32")
            p_blk = r_rep.read(frames=48000, dtype="int32")
            assert np.array_equal(o_blk, p_blk)

        r_orig.seek(chime_s0)
        r_rep.seek(chime_s0)
        o_win = r_orig.read(frames=SR, dtype="float32")
        p_win = r_rep.read(frames=SR, dtype="float32")
        sos_chime = scipy.signal.butter(4, [900.0, 1100.0], btype="bandpass", fs=SR, output="sos")
        c_before = float(np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_chime, o_win) ** 2)))
        c_after = float(np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_chime, p_win) ** 2)))
        assert 20.0 * math.log10(c_before / max(c_after, 1e-9)) > 15.0

    # Delete repaired_wav early to keep tmp_path disk footprint lean.
    repaired_wav.unlink()

    # --- Soak B: Streaming Delivery Loudness (EBU R128) on the full 30-minute WAV ---
    delivered_wav = tmp_path / "soak_30min_ebu_r128.wav"
    gc.collect()
    tracemalloc.start()
    t1 = time.perf_counter()
    try:
        metrics = apply_delivery_loudness_streaming(
            long_wav,
            delivered_wav,
            "ebu_r128",
        )
        _, delivery_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    delivery_elapsed_s = time.perf_counter() - t1
    delivery_peak_mb = delivery_peak_bytes / (1024.0 * 1024.0)

    assert delivered_wav.is_file()
    assert sf.info(str(delivered_wav)).frames == total_samples
    assert delivery_peak_mb < 80.0, (
        f"Delivery loudness peak RAM too high: {delivery_peak_mb:.2f} MiB"
    )
    assert float(metrics["integrated_lufs"]) == pytest.approx(-23.0, abs=0.5)
    assert float(metrics["true_peak_dbtp"]) <= -0.95
    assert float(metrics["loudness_range_lu"]) > 3.0

    # Emit exact soak telemetry for test logs / audit verification.
    print(
        f"[SOAK 30min 48kHz] spectral_repair: peak={spectral_peak_mb:.2f} MiB, "
        f"time={spectral_elapsed_s:.2f}s | delivery_ebu_r128: peak={delivery_peak_mb:.2f} MiB, "
        f"time={delivery_elapsed_s:.2f}s"
    )
