"""Integration tests for Wave 2 Pro HTTP endpoints, live unit re-stitching, and opt-in pro chain."""

from __future__ import annotations

import base64
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hawavoclean.audio.delivery import extract_wav_metadata_chunks, inject_wav_metadata_chunks
from hawavoclean.hashing import hash_file
from hawavoclean.pipeline import run_pipeline
from hawavoclean.server.app import SESSION_PATH, create_app
from hawavoclean.server.contracts import ManualStrategyV1, ProcessingRequestV1
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord, default_command
from hawavoclean.server.unit_restitch import UnitSliceStore

pytestmark = pytest.mark.unit
TOKEN = "pro-test-token"
H = {"X-Hawa-Token": TOKEN}


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate work dir, profiles dir, and LavaSR env for each test."""
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("HAWAVOCLEAN_LAVASR_MODEL_DIR", str(tmp_path / "no-lavasr"))
    return tmp_path


@pytest.fixture
def real_client(work: Path) -> Iterator[TestClient]:
    """FastAPI TestClient backed by a real default-command JobManager."""
    assert work.is_dir()
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    manager.shutdown()


def _speech_like_wav(
    path: Path,
    *,
    seconds: float = 2.0,
    sr: int = 48000,
    noise_level: float = 0.03,
    channels: int = 1,
) -> Path:
    """Create a deterministic speech-like harmonic + noise WAV."""
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(42)
    # Two clear vocalized bursts with a brief pause between them
    env = np.zeros(n, dtype=np.float64)
    env[int(0.15 * sr) : int(0.95 * sr)] = 1.0
    env[int(1.15 * sr) : int(min(n, 1.85 * sr))] = 1.0
    voice = env * (
        0.18 * np.sin(2.0 * np.pi * 165.0 * t)
        + 0.11 * np.sin(2.0 * np.pi * 330.0 * t)
        + 0.06 * np.sin(2.0 * np.pi * 660.0 * t)
        + 0.04 * np.sin(2.0 * np.pi * 1320.0 * t)
    )
    sig = (voice + noise_level * rng.standard_normal(n)).astype(np.float32)
    if channels == 1:
        sf.write(str(path), sig, sr, format="WAV", subtype="PCM_24")
    else:
        data = np.column_stack([sig for _ in range(channels)])
        sf.write(str(path), data, sr, format="WAV", subtype="PCM_24")
    return path


def _wait_done(client: TestClient, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    snap: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snap = client.get(f"/api/jobs/{job_id}", headers=H).json()
        if snap["state"] in TERMINAL_STATES:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached terminal state: {snap}")


def _session_headers(client: TestClient) -> dict[str, str]:
    res = client.post(SESSION_PATH, headers=H)
    assert res.status_code == 200
    token = res.json()["sessionToken"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 1. POST /api/v1/preview & managed /api/audio access
# ---------------------------------------------------------------------------


def test_v1_preview_warm_hit_and_session_audio_access(real_client: TestClient, work: Path) -> None:
    src = _speech_like_wav(work / "preview_src.wav", seconds=2.0)

    # First call (may be cold or warm depending on prior tests), second call MUST be warm hit.
    r1 = real_client.post(
        "/api/v1/preview",
        headers=H,
        json={
            "path": str(src),
            "startS": 0.1,
            "endS": 1.2,
            "profile": "production",
            "strength": 0.85,
            "ambienceFloorDb": -54.0,
        },
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert Path(body1["previewPath"]).is_file()
    assert body1["audioUrl"].startswith("/api/audio?path=")
    assert body1["renderMs"] > 0.0
    assert body1["durationS"] == pytest.approx(1.1, abs=0.05)
    assert "lufs" in body1["metricsBefore"] and "noiseFloorDb" in body1["metricsBefore"]
    assert "lufs" in body1["metricsAfter"] and "noiseFloorDb" in body1["metricsAfter"]

    r2 = real_client.post(
        "/api/v1/preview",
        headers=H,
        json={
            "path": str(src),
            "startS": 0.2,
            "endS": 1.0,
            "profile": "production",
        },
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["warmHit"] is True

    # Session client can stream the managed preview WAV via audioUrl (200 & 206 ranged)
    sess_h = _session_headers(real_client)
    audio_resp = real_client.get(body1["audioUrl"], headers=sess_h)
    assert audio_resp.status_code == 200
    assert audio_resp.content[:4] == b"RIFF"

    ranged = real_client.get(body1["audioUrl"], headers={**sess_h, "Range": "bytes=0-15"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 16

    # Session client is STILL forbidden from reading non-preview files inside previews/ or work/
    decoy = (work / "previews" / "secret.wav").resolve()
    decoy.write_bytes(b"RIFF0000WAVE")
    denied_decoy = real_client.get("/api/audio", params={"path": str(decoy)}, headers=sess_h)
    assert denied_decoy.status_code == 403

    denied_work = real_client.get("/api/audio", params={"path": str(src)}, headers=sess_h)
    assert denied_work.status_code == 403


def test_v1_preview_path_policy_and_validation(real_client: TestClient, work: Path) -> None:
    src = _speech_like_wav(work / "preview_bounds.wav", seconds=1.0)

    # Outside allowed root -> 403
    r_forbidden = real_client.post(
        "/api/v1/preview",
        headers=H,
        json={"path": "/etc/hosts", "startS": 0.0, "endS": 0.5, "profile": "production"},
    )
    assert r_forbidden.status_code == 403
    assert r_forbidden.json()["error"] == "forbidden"

    # Invalid window (endS <= startS) -> 400
    r_bad_window = real_client.post(
        "/api/v1/preview",
        headers=H,
        json={"path": str(src), "startS": 0.8, "endS": 0.4, "profile": "production"},
    )
    assert r_bad_window.status_code == 400

    # Unknown profile -> 400
    r_bad_prof = real_client.post(
        "/api/v1/preview",
        headers=H,
        json={"path": str(src), "startS": 0.0, "endS": 0.5, "profile": "nonexistent_profile"},
    )
    assert r_bad_prof.status_code == 400


# ---------------------------------------------------------------------------
# 2. POST /api/v1/spectrogram
# ---------------------------------------------------------------------------


def test_v1_spectrogram_original_cleaned_delta_and_policy(
    real_client: TestClient, work: Path
) -> None:
    orig = _speech_like_wav(work / "spec_orig.wav", seconds=1.5, noise_level=0.04)
    clean = _speech_like_wav(work / "spec_clean.wav", seconds=1.5, noise_level=0.002)

    r_orig = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={
            "path": str(orig),
            "mode": "original",
            "startS": 0.0,
            "endS": 1.5,
            "widthCols": 64,
            "nBins": 32,
            "nFft": 1024,
            "scale": "log",
        },
    )
    assert r_orig.status_code == 200, r_orig.text
    tile = r_orig.json()
    assert tile["n_bins"] == 32
    assert tile["width_cols"] == 64
    assert tile["scale"] == "log"
    raw = base64.b64decode(tile["data_b64"])
    assert len(raw) == 32 * 64

    r_delta = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={
            "path": str(orig),
            "cleanedPath": str(clean),
            "mode": "delta",
            "startS": 0.0,
            "endS": 1.5,
            "widthCols": 48,
            "nBins": 24,
            "scale": "mel",
        },
    )
    assert r_delta.status_code == 200, r_delta.text
    delta_tile = r_delta.json()
    assert delta_tile["scale"] == "mel"
    delta_grid = np.frombuffer(base64.b64decode(delta_tile["data_b64"]), dtype=np.uint8)
    assert delta_grid.size == 24 * 48
    assert int(delta_grid.max()) > 0

    # Delta without cleanedPath -> 400
    r_missing_clean = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={"path": str(orig), "mode": "delta", "startS": 0.0, "endS": 1.0},
    )
    assert r_missing_clean.status_code == 400

    # Path policy enforced on both path and cleanedPath -> 403
    r_bad_path = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={"path": "/etc/passwd", "mode": "original", "startS": 0.0, "endS": 1.0},
    )
    assert r_bad_path.status_code == 403

    r_bad_clean = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={
            "path": str(orig),
            "cleanedPath": "/etc/passwd",
            "mode": "delta",
            "startS": 0.0,
            "endS": 1.0,
        },
    )
    assert r_bad_clean.status_code == 403


# ---------------------------------------------------------------------------
# 3. POST /api/v1/export
# ---------------------------------------------------------------------------


def test_v1_export_delivery_presets_metadata_and_policy(
    real_client: TestClient, work: Path
) -> None:
    master = _speech_like_wav(work / "master_export.wav", seconds=4.0, noise_level=0.005)
    bext_payload = b"HawaVoClean BWF Test" + b"\x00" * 600
    ixml_payload = b"<BWFXML><PROJECT>ProStudio</PROJECT></BWFXML>"
    assert inject_wav_metadata_chunks(master, {b"bext": bext_payload, b"iXML": ixml_payload})

    dest_ebu = work / "exported_ebu.wav"
    r_ebu = real_client.post(
        "/api/v1/export",
        headers=H,
        json={
            "masterPath": str(master),
            "destPath": str(dest_ebu),
            "format": "wav24",
            "deliveryPreset": "ebu_r128",
            "preserveMetadata": True,
        },
    )
    assert r_ebu.status_code == 200, r_ebu.text
    res_ebu = r_ebu.json()
    assert Path(res_ebu["outputPath"]).resolve() == dest_ebu.resolve()
    assert res_ebu["format"] == "wav24"
    assert res_ebu["deliveryPreset"] == "ebu_r128"
    assert res_ebu["integratedLufs"] == pytest.approx(-23.0, abs=1.5)
    assert res_ebu["truePeakDbtp"] <= -0.95
    assert res_ebu["lraLu"] >= 0.0

    # BWF/iXML chunks preserved through delivery loudness + export
    chunks = extract_wav_metadata_chunks(dest_ebu)
    assert chunks[b"bext"] == bext_payload
    assert chunks[b"iXML"] == ixml_payload

    # FLAC with default destPath (omitted)
    r_flac = real_client.post(
        "/api/v1/export",
        headers=H,
        json={
            "masterPath": str(master),
            "format": "flac",
            "deliveryPreset": "match_input",
        },
    )
    assert r_flac.status_code == 200, r_flac.text
    flac_path = Path(r_flac.json()["outputPath"])
    assert flac_path.is_file() and flac_path.suffix == ".flac"

    # Path policy rejection on destPath outside allowed root -> 403
    r_bad_dest = real_client.post(
        "/api/v1/export",
        headers=H,
        json={
            "masterPath": str(master),
            "destPath": "/etc/hawavoclean_forbidden.wav",
            "format": "wav24",
        },
    )
    assert r_bad_dest.status_code == 403


# ---------------------------------------------------------------------------
# 4. POST /api/v1/tracks/inspect & /api/v1/tracks/extract
# ---------------------------------------------------------------------------


def _make_polywav(path: Path, sr: int = 48000, seconds: float = 1.0) -> Path:
    n = int(round(sr * seconds))
    t = np.arange(n, dtype=np.float64) / float(sr)
    ch0 = (0.2 * np.sin(2.0 * np.pi * 600.0 * t)).astype(np.float32)  # dialogue band
    ch1 = np.zeros(n, dtype=np.float32)  # silent
    ch2 = (0.15 * np.sin(2.0 * np.pi * 900.0 * t)).astype(np.float32)  # dialogue band
    ch3 = (0.001 * np.sin(2.0 * np.pi * 50.0 * t)).astype(np.float32)  # faint hum
    poly = np.column_stack([ch0, ch1, ch2, ch3])
    sf.write(str(path), poly, sr, format="WAV", subtype="PCM_24")
    ixml = (
        b"<BWFXML><TRACK_LIST>"
        b"<TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX><NAME>Boom</NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>2</CHANNEL_INDEX><NAME>Spare</NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>3</CHANNEL_INDEX><NAME>Lav_Host</NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>4</CHANNEL_INDEX><NAME>Ambience</NAME></TRACK>"
        b"</TRACK_LIST></BWFXML>"
    )
    if len(ixml) & 1:
        ixml += b" "
    # Append iXML chunk cleanly using helper
    inject_wav_metadata_chunks(path, {b"iXML": ixml})
    # Sanity verify RIFF size is valid
    with open(path, "rb") as f:
        hdr = f.read(12)
        assert hdr[:4] == b"RIFF" and struct.unpack("<I", hdr[4:8])[0] > 0
    return path


def test_v1_tracks_inspect_extract_and_policy(real_client: TestClient, work: Path) -> None:
    poly = _make_polywav(work / "field_take.wav")

    r_insp = real_client.post("/api/v1/tracks/inspect", headers=H, json={"path": str(poly)})
    assert r_insp.status_code == 200, r_insp.text
    data = r_insp.json()
    assert data["trackCount"] == 4
    names = [t["name"] for t in data["tracks"]]
    assert names == ["Boom", "Spare", "Lav_Host", "Ambience"]
    assert data["tracks"][0]["likely_dialogue"] is True
    assert data["tracks"][1]["likely_dialogue"] is False
    assert data["tracks"][2]["likely_dialogue"] is True

    out_dir = work / "stems"
    r_ext = real_client.post(
        "/api/v1/tracks/extract",
        headers=H,
        json={"path": str(poly), "trackIndices": [0, 2], "outDir": str(out_dir)},
    )
    assert r_ext.status_code == 200, r_ext.text
    ext_paths = [Path(p) for p in r_ext.json()["extractedPaths"]]
    assert len(ext_paths) == 2
    for p in ext_paths:
        assert p.is_file()
        info = sf.info(str(p))
        assert info.channels == 1

    # Path policy rejection on inspect & extract -> 403
    r_forbid_in = real_client.post("/api/v1/tracks/inspect", headers=H, json={"path": "/etc/hosts"})
    assert r_forbid_in.status_code == 403

    r_forbid_out = real_client.post(
        "/api/v1/tracks/extract",
        headers=H,
        json={"path": str(poly), "trackIndices": [0], "outDir": "/etc/forbidden_stems"},
    )
    assert r_forbid_out.status_code == 403


# ---------------------------------------------------------------------------
# 5. Live Unit Re-Stitching via POST /api/v1/jobs/{job_id}/override
# ---------------------------------------------------------------------------


def test_v1_live_unit_restitch_override_end_to_end(real_client: TestClient, work: Path) -> None:
    src = _speech_like_wav(work / "live_override_src.wav", seconds=2.0, noise_level=0.025)
    upload = real_client.post(
        "/api/upload",
        headers=H,
        files={"file": ("live_override.wav", src.read_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    submit = real_client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            "schemaVersion": 1,
            "sourceIds": [source_id],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "recordBundle": False,
            "idempotencyKey": "live-restitch-key-1",
        },
    )
    assert submit.status_code == 202, submit.text
    job_id = submit.json()["jobs"][0]["jobId"]
    done_snap = _wait_done(real_client, job_id)
    assert done_snap["state"] == "done", done_snap

    # Verify JobManager cached unit slices under work/unit_slices/<job_id>/
    slice_store = UnitSliceStore(work)
    saved_slices = slice_store.list_units(job_id)
    assert len(saved_slices) >= 1

    report = done_snap["report"]
    assert isinstance(report, dict)
    units = report["units"]
    speech_unit = next((u for u in units if u.get("is_speech") and u.get("candidate_sha256")), None)
    assert speech_unit is not None, f"Expected a speech unit with candidate in {units}"
    target_uid = int(speech_unit["unit_id"])
    initial_sha = str(report["output"]["sha256"])

    # 1. Override via unitId -> force_original
    ov1 = real_client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unitId": target_uid, "decision": "force_original"},
    )
    assert ov1.status_code == 200, ov1.text
    d1 = ov1.json()
    assert d1["restitched"] is True
    assert d1["restitchMs"] >= 0.0
    assert d1["new_decision"] == "reverted"
    assert len(d1["masterSha256"]) == 64
    sha_orig = d1["masterSha256"]

    # Master artifact endpoint serves the newly re-stitched bytes matching masterSha256
    art1 = real_client.get(f"/api/v1/jobs/{job_id}/artifacts/master", headers=H)
    assert art1.status_code == 200
    assert art1.content[:4] == b"RIFF"
    import hashlib

    assert hashlib.sha256(art1.content).hexdigest() == sha_orig

    # 2. Override via custom_strength = 0.5
    ov2 = real_client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unitId": target_uid, "decision": "custom_strength", "strength": 0.5},
    )
    assert ov2.status_code == 200, ov2.text
    d2 = ov2.json()
    assert d2["restitched"] is True
    assert d2["new_decision"] == "enhanced"
    sha_half = d2["masterSha256"]
    assert sha_half != sha_orig

    # 3. Override via force_enhanced
    ov3 = real_client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unitId": target_uid, "decision": "force_enhanced"},
    )
    assert ov3.status_code == 200, ov3.text
    d3 = ov3.json()
    assert d3["restitched"] is True
    assert d3["new_decision"] == "enhanced"
    assert d3["masterSha256"] != sha_orig

    # 4. Override back to auto
    ov4 = real_client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unitId": target_uid, "decision": "auto"},
    )
    assert ov4.status_code == 200, ov4.text
    d4 = ov4.json()
    assert d4["restitched"] is True
    assert d4["override"] == "auto"

    # Updated v1 job status reflects the latest report hash & cleared manual_override
    v1_status = real_client.get(f"/api/v1/jobs/{job_id}", headers=H).json()
    assert v1_status["report"]["output"]["sha256"] == d4["masterSha256"]
    assert initial_sha != ""


def test_unit_slice_dir_evicted_with_terminal_job_pruning(work: Path) -> None:
    now = [1000.0]
    manager = JobManager(
        command_factory=lambda rec: [
            "python3",
            "-c",
            (
                "import json, sys; "
                f"open({str(rec.report_path)!r}, 'w').write(json.dumps({{'units': []}})); "
                f"open({str(rec.output_path)!r}, 'wb').write(b'RIFF'); "
                "print(json.dumps({'event':'done','progress':1.0}), flush=True)"
            ),
        ],
        terminal_ttl_s=10.0,
        clock=lambda: now[0],
    )
    try:
        src = _speech_like_wav(work / "prune_src.wav", seconds=0.5)
        snap = manager.submit(
            input_path=src,
            output_path=work / "prune_out.wav",
            profile="production",
            overwrite=True,
        )
        job_id = snap["job_id"]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            st = manager.get_status(job_id)
            if st and st["state"] in TERMINAL_STATES:
                break
            time.sleep(0.05)

        store = UnitSliceStore(work)
        store.save_unit(job_id, 0, np.zeros(128, dtype=np.float32), None, {"is_speech": False})
        assert store.job_dir(job_id).is_dir()

        # Advance clock past TTL and trigger prune
        now[0] += 20.0
        assert manager.get_status(job_id) is None
        assert not store.job_dir(job_id).exists()
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# 6. Opt-In Pro Repair, Ambience Floor & Delivery Preset via Contract/CLI/Pipeline
# ---------------------------------------------------------------------------


def test_manual_strategy_opt_in_pro_fields_contract_and_command(work: Path) -> None:
    # Backwards-compatible default omits optional fields
    default_req = ProcessingRequestV1.model_validate(
        {
            "schemaVersion": 1,
            "sourceIds": ["src1"],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "idempotencyKey": "k-default",
        }
    )
    assert isinstance(default_req.strategy, ManualStrategyV1)
    assert default_req.strategy.ambience_floor_db is None
    assert default_req.strategy.delivery_preset is None
    assert default_req.strategy.pro_repair is None

    # Opt-in pro fields validate cleanly (camelCase & mouthDeclick alias)
    pro_req = ProcessingRequestV1.model_validate(
        {
            "schemaVersion": 1,
            "sourceIds": ["src1"],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
                "ambienceFloorDb": -52.0,
                "deliveryPreset": "ebu_r128",
                "proRepair": {"declip": True, "deplosive": True, "mouthDeclick": True},
            },
            "executionPolicy": "offline_only",
            "idempotencyKey": "k-pro",
        }
    )
    assert isinstance(pro_req.strategy, ManualStrategyV1)
    assert pro_req.strategy.ambience_floor_db == -52.0
    assert pro_req.strategy.delivery_preset == "ebu_r128"
    assert pro_req.strategy.pro_repair == {"declip": True, "deplosive": True, "mouthDeclick": True}

    # default_command omits pro flags when None and includes them when set
    rec_default = JobRecord(
        job_id="j_def",
        input_path=work / "in.wav",
        output_path=work / "out.wav",
        report_path=work / "out.hawavoclean.json",
        profile="production",
        overwrite=True,
    )
    cmd_default = default_command(rec_default)
    assert "--ambience-floor-db" not in cmd_default
    assert "--delivery-preset" not in cmd_default
    assert "--pro-repair" not in cmd_default

    rec_pro = JobRecord(
        job_id="j_pro",
        input_path=work / "in.wav",
        output_path=work / "out.wav",
        report_path=work / "out.hawavoclean.json",
        profile="production",
        overwrite=True,
        ambience_floor_db=-50.0,
        delivery_preset="podcast",
        pro_repair={"declip": True, "deplosive": False, "mouthDeclick": True},
    )
    cmd_pro = default_command(rec_pro)
    assert "--ambience-floor-db" in cmd_pro and "-50.0" in cmd_pro
    assert "--delivery-preset" in cmd_pro and "podcast" in cmd_pro
    assert "--pro-repair" in cmd_pro and "declip,mouthDeclick" in cmd_pro


def test_pipeline_default_bit_identity_and_opt_in_pro_chain(
    real_client: TestClient, work: Path
) -> None:
    src = _speech_like_wav(work / "pro_chain_src.wav", seconds=2.0, noise_level=0.02)

    # 1. Default run has zero slice dir side-effects and is bit-identical across runs
    out_a = work / "default_a.wav"
    out_b = work / "default_b.wav"
    run_pipeline(input_path=src, output_path=out_a, profile="production", overwrite=True)
    run_pipeline(input_path=src, output_path=out_b, profile="production", overwrite=True)
    assert hash_file(out_a) == hash_file(out_b)
    assert not (work / "unit_slices").exists()

    # 2. Opt-in pro_repair + ambience_floor_db + delivery_preset via HTTP /api/v1/jobs
    upload = real_client.post(
        "/api/upload",
        headers=H,
        files={"file": ("pro_chain.wav", src.read_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    submit = real_client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            "schemaVersion": 1,
            "sourceIds": [source_id],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
                "ambienceFloorDb": -54.0,
                "deliveryPreset": "ebu_r128",
                "proRepair": {"declip": True, "deplosive": True, "mouthDeclick": True},
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "recordBundle": False,
            "idempotencyKey": "pro-chain-job-1",
        },
    )
    assert submit.status_code == 202, submit.text
    job_id = submit.json()["jobs"][0]["jobId"]
    done = _wait_done(real_client, job_id)
    assert done["state"] == "done", done
    out_meta = done["report"]["output"]
    assert out_meta["integrated_lufs"] == pytest.approx(-23.0, abs=2.0)
    assert out_meta["true_peak_dbtp"] <= -0.95


# ---------------------------------------------------------------------------
# 7. POST /api/v1/spectral-repair & managed spectral_*.wav /api/audio access
# ---------------------------------------------------------------------------


def test_v1_spectral_repair_endpoint_audio_access_and_policy(
    real_client: TestClient, work: Path
) -> None:
    sr = 48000
    n = int(1.5 * sr)
    t = np.arange(n, dtype=np.float64) / float(sr)
    speech_300 = 0.25 * np.sin(2.0 * np.pi * 300.0 * t)
    chime_1000 = np.zeros(n, dtype=np.float64)
    s0, s1 = int(0.5 * sr), int(1.0 * sr)
    chime_1000[s0:s1] = 0.30 * np.sin(2.0 * np.pi * 1000.0 * t[s0:s1])
    sig = (speech_300 + chime_1000).astype(np.float32)

    src = work / "spectral_src.wav"
    sf.write(str(src), sig, sr, format="WAV", subtype="PCM_24")

    resp = real_client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={
            "path": str(src),
            "startS": 0.5,
            "endS": 1.0,
            "fLowHz": 850.0,
            "fHighHz": 1150.0,
            "attenuationDb": 24.0,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    out_path = Path(body["outputPath"])
    assert out_path.is_file()
    assert out_path.parent.resolve() == (work / "previews").resolve()
    assert out_path.name.startswith("spectral_") and out_path.suffix == ".wav"
    assert body["audioUrl"].startswith("/api/audio?path=")
    assert body["startS"] == pytest.approx(0.5)
    assert body["endS"] == pytest.approx(1.0)
    assert body["fLowHz"] == pytest.approx(850.0)
    assert body["fHighHz"] == pytest.approx(1150.0)
    assert body["attenuationDb"] == pytest.approx(24.0)
    assert body["renderMs"] > 0.0

    # Session client can stream managed spectral_*.wav via /api/audio (200 & 206).
    sess_h = _session_headers(real_client)
    audio_resp = real_client.get(body["audioUrl"], headers=sess_h)
    assert audio_resp.status_code == 200
    assert audio_resp.content[:4] == b"RIFF"

    ranged = real_client.get(body["audioUrl"], headers={**sess_h, "Range": "bytes=0-15"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 16

    # Symlink spoofing spectral_*.wav inside previews/ must be rejected with 403.
    sym = (work / "previews" / "spectral_symlink.wav").resolve()
    sym.symlink_to(src)
    denied_sym = real_client.get("/api/audio", params={"path": str(sym)}, headers=sess_h)
    assert denied_sym.status_code == 403

    # Verify acoustic attenuation on the written WAV file.
    import scipy.signal

    repaired_audio, rep_sr = sf.read(str(out_path), dtype="float32")
    assert rep_sr == sr
    sos_chime = scipy.signal.butter(4, [900.0, 1100.0], btype="bandpass", fs=sr, output="sos")
    sos_speech = scipy.signal.butter(4, [220.0, 380.0], btype="bandpass", fs=sr, output="sos")
    c_before = float(np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_chime, sig[s0:s1]) ** 2)))
    c_after = float(
        np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_chime, repaired_audio[s0:s1]) ** 2))
    )
    assert 20.0 * np.log10(c_before / max(c_after, 1e-9)) > 15.0

    v_before = float(np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_speech, sig[s0:s1]) ** 2)))
    v_after = float(
        np.sqrt(np.mean(scipy.signal.sosfiltfilt(sos_speech, repaired_audio[s0:s1]) ** 2))
    )
    assert abs(20.0 * np.log10(v_after / max(v_before, 1e-9))) < 0.5

    # Unauthorized path rejected with 403 (both root outside allowlist and session without cap).
    r_forbidden = real_client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={
            "path": "/etc/hosts",
            "startS": 0.0,
            "endS": 0.5,
            "fLowHz": 500.0,
            "fHighHz": 1500.0,
        },
    )
    assert r_forbidden.status_code == 403
    assert r_forbidden.json()["error"] == "forbidden"

    r_sess_denied = real_client.post(
        "/api/v1/spectral-repair",
        headers=sess_h,
        json={
            "path": str(src),
            "startS": 0.0,
            "endS": 0.5,
            "fLowHz": 500.0,
            "fHighHz": 1500.0,
        },
    )
    assert r_sess_denied.status_code == 403

    # Validation errors -> 400.
    r_bad_time = real_client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={"path": str(src), "startS": 0.8, "endS": 0.4, "fLowHz": 500.0, "fHighHz": 1500.0},
    )
    assert r_bad_time.status_code == 400

    r_bad_freq = real_client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={"path": str(src), "startS": 0.1, "endS": 0.5, "fLowHz": 1500.0, "fHighHz": 500.0},
    )
    assert r_bad_freq.status_code == 400

    r_over_nyq = real_client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={"path": str(src), "startS": 0.1, "endS": 0.5, "fLowHz": 500.0, "fHighHz": 30000.0},
    )
    assert r_over_nyq.status_code == 400

    r_bad_att = real_client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={
            "path": str(src),
            "startS": 0.1,
            "endS": 0.5,
            "fLowHz": 500.0,
            "fHighHz": 1500.0,
            "attenuationDb": 90.0,
        },
    )
    assert r_bad_att.status_code == 400
