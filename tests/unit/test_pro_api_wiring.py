"""Integration tests for Wave 2 Pro HTTP endpoints, live unit re-stitching, and opt-in pro chain."""

from __future__ import annotations

import base64
import struct
import sys
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


def _outside_allowed_path(*parts: str) -> str:
    """Construct a cross-platform absolute path outside home, /Volumes, and work_root."""
    return str(Path(Path.cwd().anchor, *parts))


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
        json={
            "path": _outside_allowed_path("etc", "hosts"),
            "startS": 0.0,
            "endS": 0.5,
            "profile": "production",
        },
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
    outside_passwd = _outside_allowed_path("etc", "passwd")
    r_bad_path = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={"path": outside_passwd, "mode": "original", "startS": 0.0, "endS": 1.0},
    )
    assert r_bad_path.status_code == 403

    r_bad_clean = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={
            "path": str(orig),
            "cleanedPath": outside_passwd,
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
            "destPath": _outside_allowed_path("etc", "hawavoclean_forbidden.wav"),
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
    r_forbid_in = real_client.post(
        "/api/v1/tracks/inspect",
        headers=H,
        json={"path": _outside_allowed_path("etc", "hosts")},
    )
    assert r_forbid_in.status_code == 403

    r_forbid_out = real_client.post(
        "/api/v1/tracks/extract",
        headers=H,
        json={
            "path": str(poly),
            "trackIndices": [0],
            "outDir": _outside_allowed_path("etc", "forbidden_stems"),
        },
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
            sys.executable,
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
            "path": _outside_allowed_path("etc", "hosts"),
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


# ---------------------------------------------------------------------------
# 8. Security & Reliability Audit Regressions (#124 Findings #1 - #5)
# ---------------------------------------------------------------------------


def test_audit_export_rejects_symlink_escape_immutable_gen_dir_and_bad_ext(
    real_client: TestClient, work: Path
) -> None:
    import tempfile

    src = _speech_like_wav(work / "audit_export_src.wav", seconds=1.5, noise_level=0.02)
    sess_h = _session_headers(real_client)

    # Upload & complete a real job so the master has an immutable generation dir.
    upload = real_client.post(
        "/api/upload",
        headers=sess_h,
        files={"file": ("audit_export.wav", src.read_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    submit = real_client.post(
        "/api/v1/jobs",
        headers=sess_h,
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
            "idempotencyKey": "audit-export-symlink-job",
        },
    )
    assert submit.status_code == 202, submit.text
    job_id = submit.json()["jobs"][0]["jobId"]
    done = _wait_done(real_client, job_id)
    assert done["state"] == "done", done
    public_master = Path(done["output_path"])
    from hawavoclean.publication import resolve_committed_publication

    committed = resolve_committed_publication(public_master)
    assert committed is not None
    immutable_master = committed[0]
    assert immutable_master.parent != public_master.parent

    exports_dir = (work / "exports").resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hawa-audit-outside-") as outside_str:
        outside_dir = Path(outside_str).resolve()
        victim = outside_dir / "victim.wav"
        sentinel = b"OUTSIDE_VICTIM_SENTINEL"
        victim.write_bytes(sentinel)

        # 1. Symlink destination inside exports/ pointing outside allowed roots -> 403 for both root & session.
        sym_dest = exports_dir / "escape.wav"
        sym_dest.symlink_to(victim)
        for hdrs in (H, sess_h):
            resp = real_client.post(
                "/api/v1/export",
                headers=hdrs,
                json={
                    "masterPath": str(public_master),
                    "destPath": str(sym_dest),
                    "format": "wav24",
                },
            )
            assert resp.status_code == 403, resp.text
            assert victim.read_bytes() == sentinel

        # 2. Symlinked ancestor dir inside exports/ pointing outside -> 403, nothing created outside.
        sym_dir = exports_dir / "sym_out"
        sym_dir.symlink_to(outside_dir)
        for hdrs in (H, sess_h):
            resp = real_client.post(
                "/api/v1/export",
                headers=hdrs,
                json={
                    "masterPath": str(public_master),
                    "destPath": str(sym_dir / "sneaky.wav"),
                    "format": "wav24",
                },
            )
            assert resp.status_code == 403, resp.text
            assert not (outside_dir / "sneaky.wav").exists()

    # 3. Non-root session must NOT write inside the internal immutable generation dir.
    imm_dest = immutable_master.parent / "tamper.wav"
    r_imm = real_client.post(
        "/api/v1/export",
        headers=sess_h,
        json={
            "masterPath": str(public_master),
            "destPath": str(imm_dest),
            "format": "wav24",
        },
    )
    assert r_imm.status_code == 403, r_imm.text
    assert not imm_dest.exists()

    # 4. Extension mismatch against requested format -> 400.
    r_bad_ext = real_client.post(
        "/api/v1/export",
        headers=sess_h,
        json={
            "masterPath": str(public_master),
            "destPath": str(exports_dir / "wrong_ext.sh"),
            "format": "wav24",
        },
    )
    assert r_bad_ext.status_code == 400, r_bad_ext.text
    assert not (exports_dir / "wrong_ext.sh").exists()

    # 5. Legitimate web-mode export by non-root session is downloadable via /api/audio (200 & 206),
    # while unrelated sibling files and swapped symlinks remain 403.
    r_ok = real_client.post(
        "/api/v1/export",
        headers=sess_h,
        json={
            "masterPath": str(public_master),
            "destPath": str(exports_dir / "delivery_ok.wav"),
            "format": "wav24",
            "deliveryPreset": "match_input",
        },
    )
    assert r_ok.status_code == 200, r_ok.text
    out_path = Path(r_ok.json()["outputPath"])
    assert out_path.is_file()

    dl = real_client.get("/api/audio", params={"path": str(out_path)}, headers=sess_h)
    assert dl.status_code == 200
    assert dl.content[:4] == b"RIFF"

    dl_range = real_client.get(
        "/api/audio",
        params={"path": str(out_path)},
        headers={**sess_h, "Range": "bytes=0-15"},
    )
    assert dl_range.status_code == 206
    assert len(dl_range.content) == 16

    unrelated = exports_dir / "unrelated_secret.wav"
    unrelated.write_bytes(b"RIFF0000WAVE")
    assert (
        real_client.get("/api/audio", params={"path": str(unrelated)}, headers=sess_h).status_code
        == 403
    )

    # Swapping a previously authorized delivery export to a symlink revokes access immediately.
    out_path.unlink()
    out_path.symlink_to(unrelated)
    assert (
        real_client.get("/api/audio", params={"path": str(out_path)}, headers=sess_h).status_code
        == 403
    )


def test_audit_preview_profile_traversal_and_spectrogram_window_cap(
    real_client: TestClient, work: Path
) -> None:
    src = _speech_like_wav(work / "audit_prev_spec.wav", seconds=1.0)

    # Preview rejects traversal and arbitrary profile names.
    for bad_profile in ("../../etc/passwd", "../production", "/etc/hosts", "evil_profile"):
        r_prev = real_client.post(
            "/api/v1/preview",
            headers=H,
            json={"path": str(src), "startS": 0.0, "endS": 0.5, "profile": bad_profile},
        )
        assert r_prev.status_code in (400, 422), (bad_profile, r_prev.status_code, r_prev.text)

    # Spectrogram rejects > 300s window duration cleanly before decoding.
    r_oom = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={"path": str(src), "mode": "original", "startS": 0.0, "endS": 3600.0},
    )
    assert r_oom.status_code in (400, 422), r_oom.text

    r_oom_delta = real_client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={
            "path": str(src),
            "cleanedPath": str(src),
            "mode": "delta",
            "startS": 10.0,
            "endS": 311.0,
        },
    )
    assert r_oom_delta.status_code in (400, 422), r_oom_delta.text


def test_audit_non_root_session_extracted_polywav_audio_analyze_and_jobs(
    real_client: TestClient, work: Path
) -> None:
    poly = _make_polywav(work / "session_poly.wav", seconds=1.2)
    sess_h = _session_headers(real_client)

    # Upload Poly-WAV as non-root session.
    up = real_client.post(
        "/api/upload",
        headers=sess_h,
        files={"file": ("session_poly.wav", poly.read_bytes(), "audio/wav")},
    )
    assert up.status_code == 200, up.text
    uploaded_path = up.json()["path"]

    # Inspect and extract tracks into default managed work_root()/extracted_tracks.
    r_insp = real_client.post(
        "/api/v1/tracks/inspect",
        headers=sess_h,
        json={"path": uploaded_path},
    )
    assert r_insp.status_code == 200, r_insp.text
    assert r_insp.json()["trackCount"] == 4

    r_ext = real_client.post(
        "/api/v1/tracks/extract",
        headers=sess_h,
        json={"path": uploaded_path, "trackIndices": [0, 2]},
    )
    assert r_ext.status_code == 200, r_ext.text
    body = r_ext.json()
    ext_paths = body["extractedPaths"]
    ext_source_ids = body["sourceIds"]
    assert len(ext_paths) == 2
    assert len(ext_source_ids) == 2
    assert all(len(sid) == 32 for sid in ext_source_ids)

    # Non-root session can read via /api/audio (200 & 206) and analyze extracted tracks.
    a0 = real_client.get("/api/audio", params={"path": ext_paths[0]}, headers=sess_h)
    assert a0.status_code == 200
    assert a0.content[:4] == b"RIFF"

    a0_range = real_client.get(
        "/api/audio",
        params={"path": ext_paths[0]},
        headers={**sess_h, "Range": "bytes=0-15"},
    )
    assert a0_range.status_code == 206
    assert len(a0_range.content) == 16

    an = real_client.post("/api/analyze", headers=sess_h, json={"path": ext_paths[0]})
    assert an.status_code == 200, an.text

    # Non-root session can submit /api/v1/jobs using extracted track sourceIds.
    for idx, src_ref in enumerate(ext_source_ids):
        sub = real_client.post(
            "/api/v1/jobs",
            headers=sess_h,
            json={
                "schemaVersion": 1,
                "sourceIds": [src_ref],
                "strategy": {
                    "kind": "manual",
                    "route": "production",
                    "allowGenerativeReconstruction": False,
                },
                "executionPolicy": "offline_only",
                "conflictPolicy": "unique",
                "recordBundle": False,
                "idempotencyKey": f"audit-extracted-track-job-{idx}",
            },
        )
        assert sub.status_code == 202, sub.text
        done = _wait_done(real_client, sub.json()["jobs"][0]["jobId"])
        assert done["state"] == "done", done

    # Symlink placed inside work/extracted_tracks pointing outside must still be rejected with 403.
    sym_track = (work / "extracted_tracks" / "symlink_escape.wav").resolve()
    sym_track.symlink_to(poly)
    assert (
        real_client.get("/api/audio", params={"path": str(sym_track)}, headers=sess_h).status_code
        == 403
    )


def test_audit_override_concurrent_serialization_and_record_bundle_409(
    real_client: TestClient, work: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    src = _speech_like_wav(work / "audit_override_src.wav", seconds=2.0, noise_level=0.025)
    upload = real_client.post(
        "/api/upload",
        headers=H,
        files={"file": ("audit_override.wav", src.read_bytes(), "audio/wav")},
    )
    assert upload.status_code == 200
    source_id = upload.json()["source_id"]

    # 1. Sealed record_bundle=True job MUST refuse override with 409 and keep all artifacts intact.
    sub_bundle = real_client.post(
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
            "recordBundle": True,
            "idempotencyKey": "audit-sealed-bundle-job",
        },
    )
    assert sub_bundle.status_code == 202, sub_bundle.text
    bundle_job_id = sub_bundle.json()["jobs"][0]["jobId"]
    done_bundle = _wait_done(real_client, bundle_job_id)
    assert done_bundle["state"] == "done", done_bundle
    assert done_bundle["record_bundle"] is True

    master_before = real_client.get(f"/api/v1/jobs/{bundle_job_id}/artifacts/master", headers=H)
    record_before = real_client.get(f"/api/v1/jobs/{bundle_job_id}/artifacts/record", headers=H)
    assert master_before.status_code == 200
    assert record_before.status_code == 200

    refused = real_client.post(
        f"/api/v1/jobs/{bundle_job_id}/override",
        headers=H,
        json={"unitIndex": 0, "decision": "force_original"},
    )
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"] == "sealed_record_bundle"

    # Artifacts remain verifiable and byte-identical after refused override.
    master_after = real_client.get(f"/api/v1/jobs/{bundle_job_id}/artifacts/master", headers=H)
    record_after = real_client.get(f"/api/v1/jobs/{bundle_job_id}/artifacts/record", headers=H)
    report_after = real_client.get(f"/api/v1/jobs/{bundle_job_id}/artifacts/report", headers=H)
    summary_after = real_client.get(f"/api/v1/jobs/{bundle_job_id}/artifacts/summary", headers=H)
    assert master_after.status_code == 200 and master_after.content == master_before.content
    assert record_after.status_code == 200 and record_after.content == record_before.content
    assert report_after.status_code == 200
    assert summary_after.status_code == 200

    # 2. Concurrent overrides on a mutable (record_bundle=False) job serialize cleanly without lost updates.
    # Build a 34.0s clip (> hard_max_group_s=30s) with a 1.5s silent gap so segmentation yields >= 2 speech units.
    sr = 48000
    n_two = int(34.0 * sr)
    t_two = np.arange(n_two, dtype=np.float64) / float(sr)
    rng_two = np.random.default_rng(7)
    env_two = np.zeros(n_two, dtype=np.float64)
    env_two[int(0.2 * sr) : int(16.5 * sr)] = 1.0
    env_two[int(18.0 * sr) : int(33.5 * sr)] = 1.0
    voice_two = env_two * (
        0.20 * np.sin(2.0 * np.pi * 165.0 * t_two)
        + 0.12 * np.sin(2.0 * np.pi * 330.0 * t_two)
        + 0.06 * np.sin(2.0 * np.pi * 660.0 * t_two)
    )
    noise_two = env_two * 0.02 * rng_two.standard_normal(n_two)
    two_unit_src = work / "audit_override_two_units.wav"
    sf.write(
        str(two_unit_src),
        (voice_two + noise_two).astype(np.float32),
        sr,
        format="WAV",
        subtype="PCM_24",
    )

    upload2 = real_client.post(
        "/api/upload",
        headers=H,
        files={"file": ("audit_override_live.wav", two_unit_src.read_bytes(), "audio/wav")},
    )
    assert upload2.status_code == 200
    live_source_id = upload2.json()["source_id"]

    sub_live = real_client.post(
        "/api/v1/jobs",
        headers=H,
        json={
            "schemaVersion": 1,
            "sourceIds": [live_source_id],
            "strategy": {
                "kind": "manual",
                "route": "production",
                "allowGenerativeReconstruction": False,
            },
            "executionPolicy": "offline_only",
            "conflictPolicy": "unique",
            "recordBundle": False,
            "idempotencyKey": "audit-concurrent-override-job",
        },
    )
    assert sub_live.status_code == 202, sub_live.text
    live_job_id = sub_live.json()["jobs"][0]["jobId"]
    done_live = _wait_done(real_client, live_job_id)
    assert done_live["state"] == "done", done_live
    units = done_live["report"]["units"]
    speech_units = [u for u in units if u.get("is_speech")]
    assert len(speech_units) >= 2, f"Expected >=2 speech units, got {len(speech_units)}"
    uid_a = int(speech_units[0]["unit_id"])
    uid_b = int(speech_units[1]["unit_id"])

    # Verify JobRecord.snapshot() returns an isolated deep copy of report.
    from typing import cast

    app_state = cast(Any, real_client.app).state
    snap1 = app_state.job_manager.get_status(live_job_id)
    assert snap1 is not None
    snap1["report"]["units"][0]["final_decision"] = "CORRUPTED_LOCAL_COPY"
    snap2 = app_state.job_manager.get_status(live_job_id)
    assert snap2 is not None
    assert snap2["report"]["units"][0]["final_decision"] != "CORRUPTED_LOCAL_COPY"

    def _post_override(uid: int, decision: str) -> int:
        res = real_client.post(
            f"/api/v1/jobs/{live_job_id}/override",
            headers=H,
            json={"unitId": uid, "decision": decision},
        )
        assert res.status_code == 200, res.text
        return int(res.status_code)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_a = pool.submit(_post_override, uid_a, "force_original")
        f_b = pool.submit(_post_override, uid_b, "force_enhanced")
        assert f_a.result() == 200
        assert f_b.result() == 200

    final_status = real_client.get(f"/api/v1/jobs/{live_job_id}", headers=H).json()
    by_id = {int(u["unit_id"]): u for u in final_status["report"]["units"]}
    assert by_id[uid_a]["manual_override"] == "force_original"
    assert by_id[uid_a]["final_decision"] == "reverted"
    assert by_id[uid_b]["manual_override"] == "force_enhanced"
    assert by_id[uid_b]["final_decision"] == "enhanced"

    # Master artifact matches the latest serialized report SHA-256.
    import hashlib

    final_art = real_client.get(f"/api/v1/jobs/{live_job_id}/artifacts/master", headers=H)
    assert final_art.status_code == 200
    assert (
        hashlib.sha256(final_art.content).hexdigest() == final_status["report"]["output"]["sha256"]
    )
