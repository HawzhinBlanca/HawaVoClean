"""Red-team security, DoS pruning, concurrency, and corrupt-cache tests for Pro v1 endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.hashing import hash_file
from hawavoclean.server.app import (
    DEFAULT_MAX_EXTRACTED_TRACK_DIRS,
    DEFAULT_MAX_JSON_BODY_BYTES,
    SESSION_PATH,
    _DeliveryOutputRegistry,
    cleanup_extracted_tracks_dir,
    create_app,
)
from hawavoclean.server.contracts import (
    ManualStrategyV1,
    RegionPreviewRequestV1,
    SeparationStrategyV1,
    SpectralRepairRequestV1,
    SpectrogramRequestV1,
    TrackExtractRequestV1,
    UnitOverrideRequestV1,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager
from hawavoclean.server.unit_restitch import UnitSliceStore, restitch_master_with_overrides
from hawavoclean.server.warm_worker import DEFAULT_MAX_PREVIEW_FILES, WarmCoreCache

pytestmark = pytest.mark.unit
TOKEN = "redteam-secret-token"
H = {"X-Hawa-Token": TOKEN}


def _outside_allowed_path(*parts: str) -> str:
    """Build a cross-platform absolute path guaranteed outside allowed_roots()."""
    return str(Path(Path.cwd().anchor, *parts))


def _speech_wav(
    path: Path,
    *,
    seconds: float = 1.5,
    sr: int = 48000,
    channels: int = 1,
) -> Path:
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(1337)
    env = np.zeros(n, dtype=np.float64)
    env[int(0.15 * sr) : int(min(n, 1.35 * sr))] = 1.0
    voice = env * (
        0.20 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.12 * np.sin(2.0 * np.pi * 360.0 * t)
        + 0.06 * np.sin(2.0 * np.pi * 720.0 * t)
    )
    sig = (voice + 0.02 * rng.standard_normal(n)).astype(np.float32)
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
    raise AssertionError(f"job {job_id} never finished: {snap}")


def _session_headers(client: TestClient) -> dict[str, str]:
    res = client.post(SESSION_PATH, headers=H)
    assert res.status_code == 200
    return {"Authorization": f"Bearer {res.json()['sessionToken']}"}


def _assert_no_leaks(text: str, work: Path) -> None:
    assert "Traceback" not in text
    assert str(work) not in text
    assert str(work.resolve()) not in text


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("HAWAVOCLEAN_LAVASR_MODEL_DIR", str(tmp_path / "no-lavasr"))
    return tmp_path


@pytest.fixture
def client(work: Path) -> Iterator[TestClient]:
    assert work.is_dir()
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c
    manager.shutdown()


# ---------------------------------------------------------------------------
# 1. Auth, Origin, Host & Cross-Site Enforcement Across All 7 Pro Routes + /api/audio
# ---------------------------------------------------------------------------


def test_redteam_auth_origin_host_across_all_pro_endpoints(client: TestClient, work: Path) -> None:
    wav = _speech_wav(work / "auth_probe.wav", seconds=0.6)
    routes: list[tuple[str, str, dict[str, Any] | None]] = [
        ("POST", "/api/v1/preview", {"path": str(wav), "startS": 0.0, "endS": 0.4}),
        ("POST", "/api/v1/spectrogram", {"path": str(wav), "startS": 0.0, "endS": 0.4}),
        ("POST", "/api/v1/export", {"masterPath": str(wav), "format": "wav24"}),
        ("POST", "/api/v1/tracks/inspect", {"path": str(wav)}),
        ("POST", "/api/v1/tracks/extract", {"path": str(wav), "trackIndices": [0]}),
        (
            "POST",
            "/api/v1/spectral-repair",
            {"path": str(wav), "startS": 0.0, "endS": 0.4, "fLowHz": 300.0, "fHighHz": 900.0},
        ),
        (
            "POST",
            "/api/v1/jobs/nonexistent/override",
            {"unitIndex": 0, "decision": "force_original"},
        ),
        ("GET", f"/api/audio?path={wav}", None),
    ]

    for method, url, payload in routes:
        # No credentials -> 401
        r_noauth = client.request(method, url, json=payload)
        assert r_noauth.status_code == 401, (url, r_noauth.status_code)

        # Wrong token -> 401
        r_badtok = client.request(method, url, headers={"X-Hawa-Token": "wrong"}, json=payload)
        assert r_badtok.status_code == 401, (url, r_badtok.status_code)

        # Query string token -> 400 (forbidden even with valid header)
        sep = "&" if "?" in url else "?"
        r_qtok = client.request(method, f"{url}{sep}token={TOKEN}", headers=H, json=payload)
        assert r_qtok.status_code == 400, (url, r_qtok.status_code)
        assert r_qtok.json()["error"] == "query_auth_forbidden"

        # Untrusted Origin -> 403
        r_origin = client.request(
            method, url, headers={**H, "Origin": "https://attacker.example"}, json=payload
        )
        assert r_origin.status_code == 403, (url, r_origin.status_code)

        # Cross-site Sec-Fetch-Site -> 403
        r_cross = client.request(
            method, url, headers={**H, "Sec-Fetch-Site": "cross-site"}, json=payload
        )
        assert r_cross.status_code == 403, (url, r_cross.status_code)

        # Non-loopback Host -> 403
        r_host = client.request(
            method, url, headers={**H, "Host": "evil.example:8000"}, json=payload
        )
        assert r_host.status_code == 403, (url, r_host.status_code)


# ---------------------------------------------------------------------------
# 2. Special-File, Path, Oversized/Malformed JSON & Type-Confusion Attacks
# ---------------------------------------------------------------------------


def test_redteam_path_special_files_and_error_redaction(client: TestClient, work: Path) -> None:
    wav = _speech_wav(work / "valid_audio.wav", seconds=0.6)
    corrupt = work / "not_audio.wav"
    corrupt.write_bytes(b"THIS IS DEFINITELY NOT A WAV FILE")
    a_dir = work / "subdir_trap.wav"
    a_dir.mkdir()
    outside = _outside_allowed_path("hawavoclean_redteam_outside", "secret.wav")
    traversal = str(work / ".." / ".." / ".." / ".." / "etc" / "passwd")

    # 1. NUL bytes, relative paths, and directories rejected cleanly across endpoints
    hostile_inputs: list[tuple[str, int]] = [
        (f"{work}/evil\x00.wav", 400),
        ("relative/path.wav", 400),
        (str(a_dir), 404),
        (outside, 403),
    ]
    for bad_path, expected_status in hostile_inputs:
        for url, body in (
            ("/api/v1/preview", {"path": bad_path, "startS": 0.0, "endS": 0.3}),
            ("/api/v1/spectrogram", {"path": bad_path, "startS": 0.0, "endS": 0.3}),
            ("/api/v1/export", {"masterPath": bad_path, "format": "wav24"}),
            ("/api/v1/tracks/inspect", {"path": bad_path}),
            ("/api/v1/tracks/extract", {"path": bad_path, "trackIndices": [0]}),
            (
                "/api/v1/spectral-repair",
                {
                    "path": bad_path,
                    "startS": 0.0,
                    "endS": 0.3,
                    "fLowHz": 300.0,
                    "fHighHz": 900.0,
                },
            ),
        ):
            r = client.post(url, headers=H, json=body)
            assert r.status_code == expected_status, (url, bad_path, r.status_code, r.text)
            assert "Traceback" not in r.text

        r_audio = client.get("/api/audio", params={"path": bad_path}, headers=H)
        assert r_audio.status_code == expected_status, (bad_path, r_audio.status_code)

    # 2. Traversal resolving outside allowed roots -> 403 (or 404 if tmp is deep under home)
    r_trav = client.post(
        "/api/v1/preview",
        headers=H,
        json={"path": traversal, "startS": 0.0, "endS": 0.3},
    )
    assert r_trav.status_code in (400, 403, 404)

    # 3. Corrupt / non-audio file -> 400 with NO host path or stack trace leakage
    for url, body in (
        ("/api/v1/preview", {"path": str(corrupt), "startS": 0.0, "endS": 0.3}),
        ("/api/v1/spectrogram", {"path": str(corrupt), "startS": 0.0, "endS": 0.3}),
        ("/api/v1/export", {"masterPath": str(corrupt), "format": "wav24"}),
        ("/api/v1/tracks/inspect", {"path": str(corrupt)}),
        ("/api/v1/tracks/extract", {"path": str(corrupt), "trackIndices": [0]}),
        (
            "/api/v1/spectral-repair",
            {
                "path": str(corrupt),
                "startS": 0.0,
                "endS": 0.3,
                "fLowHz": 300.0,
                "fHighHz": 900.0,
            },
        ),
    ):
        r = client.post(url, headers=H, json=body)
        assert r.status_code == 400, (url, r.status_code, r.text)
        _assert_no_leaks(r.text, work)

    # 4. Out-of-range track index must NOT leak host path
    r_bad_trk = client.post(
        "/api/v1/tracks/extract",
        headers=H,
        json={"path": str(wav), "trackIndices": [9]},
    )
    assert r_bad_trk.status_code == 400
    _assert_no_leaks(r_bad_trk.text, work)

    # 5. Export destPath is an existing directory -> 400 (never 500)
    r_dir_dest = client.post(
        "/api/v1/export",
        headers=H,
        json={"masterPath": str(wav), "destPath": str(a_dir), "format": "wav24"},
    )
    assert r_dir_dest.status_code == 400, r_dir_dest.text

    # 6. Extract outDir is an existing regular file -> 400 (never 500)
    r_file_outdir = client.post(
        "/api/v1/tracks/extract",
        headers=H,
        json={"path": str(wav), "trackIndices": [0], "outDir": str(corrupt)},
    )
    assert r_file_outdir.status_code == 400, r_file_outdir.text

    # 7. Extract outDir is a symlink -> 403
    real_dir = work / "real_stems"
    real_dir.mkdir()
    sym_dir = work / "sym_stems"
    try:
        sym_dir.symlink_to(real_dir)
    except OSError:
        pass
    else:
        r_sym_outdir = client.post(
            "/api/v1/tracks/extract",
            headers=H,
            json={"path": str(wav), "trackIndices": [0], "outDir": str(sym_dir)},
        )
        assert r_sym_outdir.status_code == 403, r_sym_outdir.text

    # 8. FIFO / named pipe (POSIX) rejected cleanly without hanging!
    if hasattr(os, "mkfifo"):
        fifo_path = work / "trap_pipe.wav"
        os.mkfifo(fifo_path)
        assert (
            client.get("/api/audio", params={"path": str(fifo_path)}, headers=H).status_code == 404
        )
        assert (
            client.post(
                "/api/v1/preview",
                headers=H,
                json={"path": str(fifo_path), "startS": 0.0, "endS": 0.3},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/v1/export",
                headers=H,
                json={"masterPath": str(wav), "destPath": str(fifo_path), "format": "wav24"},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/tracks/extract",
                headers=H,
                json={"path": str(wav), "trackIndices": [0], "outDir": str(fifo_path)},
            ).status_code
            == 400
        )


def test_redteam_malformed_oversized_nan_inf_bool_and_extra_json(
    client: TestClient, work: Path
) -> None:
    wav = _speech_wav(work / "json_probe.wav", seconds=0.6)

    # 1. Oversized JSON (> 1 MiB) -> 413 payload_too_large on Pro endpoints
    huge_payload = b'{"path":"' + (b"a" * (DEFAULT_MAX_JSON_BODY_BYTES + 128)) + b'"}'
    for url in (
        "/api/v1/preview",
        "/api/v1/spectrogram",
        "/api/v1/export",
        "/api/v1/tracks/inspect",
        "/api/v1/tracks/extract",
        "/api/v1/spectral-repair",
        "/api/v1/jobs/any/override",
    ):
        r_big = client.post(
            url,
            headers={**H, "Content-Type": "application/json"},
            content=huge_payload,
        )
        assert r_big.status_code == 413, (url, r_big.status_code)
        assert r_big.json()["error"] == "payload_too_large"

        # Malformed JSON -> 400
        r_bad = client.post(
            url,
            headers={**H, "Content-Type": "application/json"},
            content=b"{not valid json",
        )
        assert r_bad.status_code == 400, (url, r_bad.status_code)

    # 2. NaN / Infinity literals -> 400
    for url, raw_json in (
        ("/api/v1/preview", f'{{"path":{json.dumps(str(wav))},"startS":NaN,"endS":0.5}}'),
        ("/api/v1/preview", f'{{"path":{json.dumps(str(wav))},"startS":0.0,"endS":Infinity}}'),
        ("/api/v1/spectrogram", f'{{"path":{json.dumps(str(wav))},"startS":0.0,"endS":NaN}}'),
        (
            "/api/v1/spectral-repair",
            f'{{"path":{json.dumps(str(wav))},"startS":0.0,"endS":0.4,"fLowHz":300,"fHighHz":Infinity}}',
        ),
        (
            "/api/v1/jobs/any/override",
            '{"unitIndex":0,"decision":"custom_strength","strength":NaN}',
        ),
    ):
        r_nan = client.post(
            url,
            headers={**H, "Content-Type": "application/json"},
            content=raw_json.encode("utf-8"),
        )
        assert r_nan.status_code == 400, (url, raw_json, r_nan.status_code, r_nan.text)

    # 3. Bool-as-number fields -> 400 across all Pro endpoints
    bool_cases: list[tuple[str, dict[str, Any]]] = [
        ("/api/v1/preview", {"path": str(wav), "startS": False, "endS": True}),
        ("/api/v1/preview", {"path": str(wav), "startS": 0.0, "endS": 0.4, "strength": True}),
        ("/api/v1/spectrogram", {"path": str(wav), "startS": 0.0, "endS": 0.4, "widthCols": True}),
        ("/api/v1/tracks/extract", {"path": str(wav), "trackIndices": [True]}),
        (
            "/api/v1/spectral-repair",
            {
                "path": str(wav),
                "startS": 0.0,
                "endS": 0.4,
                "fLowHz": False,
                "fHighHz": 900.0,
                "attenuationDb": True,
            },
        ),
        (
            "/api/v1/jobs/any/override",
            {"unitIndex": True, "decision": "force_original"},
        ),
        (
            "/api/v1/jobs/any/override",
            {"unitIndex": 0, "decision": "custom_strength", "strength": True},
        ),
    ]
    for url, body in bool_cases:
        r_bool = client.post(url, headers=H, json=body)
        assert r_bool.status_code == 400, (url, body, r_bool.status_code, r_bool.text)

    # 4. Unknown / extra fields -> 422
    for url, body in (
        ("/api/v1/preview", {"path": str(wav), "startS": 0.0, "endS": 0.4, "bogus": 1}),
        ("/api/v1/spectrogram", {"path": str(wav), "startS": 0.0, "endS": 0.4, "bogus": 1}),
        ("/api/v1/export", {"masterPath": str(wav), "bogus": 1}),
        ("/api/v1/tracks/inspect", {"path": str(wav), "bogus": 1}),
        ("/api/v1/tracks/extract", {"path": str(wav), "trackIndices": [0], "bogus": 1}),
        (
            "/api/v1/spectral-repair",
            {
                "path": str(wav),
                "startS": 0.0,
                "endS": 0.4,
                "fLowHz": 300.0,
                "fHighHz": 900.0,
                "bogus": 1,
            },
        ),
        (
            "/api/v1/jobs/any/override",
            {"unitIndex": 0, "decision": "force_original", "bogus": 1},
        ),
    ):
        r_extra = client.post(url, headers=H, json=body)
        assert r_extra.status_code == 422, (url, r_extra.status_code, r_extra.text)


# ---------------------------------------------------------------------------
# 3. Disk-Exhaustion DoS Hardening: Previews, Spectral Repairs & Extracted Tracks
# ---------------------------------------------------------------------------


def test_redteam_disk_dos_pruning_previews_spectral_and_extracted_tracks(
    client: TestClient, work: Path
) -> None:
    wav = _speech_wav(work / "dos_src.wav", seconds=0.8, channels=2)
    preview_dir = (work / "previews").resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Seed 72 managed preview_* and spectral_* files plus 1 expired file and 1 non-managed file.
    base_ts = time.time() - 120.0
    decoy = preview_dir / "user_keep.wav"
    decoy.write_bytes(b"KEEP")
    expired = preview_dir / "spectral_expired.wav"
    expired.write_bytes(b"OLD")
    os.utime(expired, (10.0, 10.0))
    for i in range(36):
        p = preview_dir / f"preview_seed_{i:03d}.wav"
        s = preview_dir / f"spectral_seed_{i:03d}.wav"
        p.write_bytes(b"RIFF")
        s.write_bytes(b"RIFF")
        os.utime(p, (base_ts + i, base_ts + i))
        os.utime(s, (base_ts + i + 0.5, base_ts + i + 0.5))

    # 1. /api/v1/spectral-repair prunes previews/ down to <= DEFAULT_MAX_PREVIEW_FILES (64).
    r_spec = client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={
            "path": str(wav),
            "startS": 0.1,
            "endS": 0.5,
            "fLowHz": 400.0,
            "fHighHz": 1200.0,
        },
    )
    assert r_spec.status_code == 200, r_spec.text
    spec_out = Path(r_spec.json()["outputPath"])
    assert spec_out.is_file()
    assert decoy.is_file()
    assert not expired.exists()
    managed = [
        e
        for e in preview_dir.iterdir()
        if e.is_file() and e.name.startswith(("preview_", "spectral_"))
    ]
    assert len(managed) <= DEFAULT_MAX_PREVIEW_FILES

    # 2. /api/v1/preview also keeps managed count <= DEFAULT_MAX_PREVIEW_FILES and protects its file.
    r_prev = client.post(
        "/api/v1/preview",
        headers=H,
        json={"path": str(wav), "startS": 0.1, "endS": 0.5, "profile": "production"},
    )
    assert r_prev.status_code == 200, r_prev.text
    prev_out = Path(r_prev.json()["previewPath"])
    assert prev_out.is_file()
    managed_after = [
        e
        for e in preview_dir.iterdir()
        if e.is_file() and e.name.startswith(("preview_", "spectral_"))
    ]
    assert len(managed_after) <= DEFAULT_MAX_PREVIEW_FILES

    # 3. work_root()/extracted_tracks bounded pruning on /api/v1/tracks/extract.
    extracted_root = (work / "extracted_tracks").resolve()
    extracted_root.mkdir(parents=True, exist_ok=True)
    expired_dir = extracted_root / "expired_take"
    expired_dir.mkdir()
    os.utime(expired_dir, (10.0, 10.0))
    for i in range(DEFAULT_MAX_EXTRACTED_TRACK_DIRS + 8):
        d = extracted_root / f"seed_take_{i:03d}"
        d.mkdir()
        (d / "track.wav").write_bytes(b"RIFF")
        os.utime(d, (base_ts + i, base_ts + i))

    r_ext = client.post(
        "/api/v1/tracks/extract",
        headers=H,
        json={"path": str(wav), "trackIndices": [0]},
    )
    assert r_ext.status_code == 200, r_ext.text
    created_dir = Path(r_ext.json()["outDir"])
    assert created_dir.is_dir()
    assert not expired_dir.exists()
    remaining_dirs = [e for e in extracted_root.iterdir() if e.is_dir() and not e.is_symlink()]
    assert len(remaining_dirs) <= DEFAULT_MAX_EXTRACTED_TRACK_DIRS
    assert cleanup_extracted_tracks_dir(work / "nonexistent_extracted") == 0


# ---------------------------------------------------------------------------
# 4. Concurrency Torture: 8-Thread WarmCoreCache & 10-Thread Live Override
# ---------------------------------------------------------------------------


class _FastStubEnhancer:
    def __init__(self, core_id: str, sample_rate: int) -> None:
        self._meta = EnhancerMetadata(
            core_id=core_id,
            version="1.0",
            algorithm="stub",
            sample_rate=sample_rate,
            phase_coherent=True,
        )

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        pass

    def enhance(
        self, waveform: np.ndarray[Any, np.dtype[np.float32]], sample_rate: int
    ) -> EnhancementResult:
        return EnhancementResult(
            waveform=waveform.copy(),
            sample_rate=sample_rate,
            model_runtime_ms=0.01,
            input_samples=int(waveform.shape[0]),
            output_samples=int(waveform.shape[0]),
        )


def test_redteam_warm_core_cache_8_thread_concurrency_torture() -> None:
    clock_lock = threading.Lock()
    now = [100.0]

    def tick() -> float:
        with clock_lock:
            now[0] += 0.25
            return now[0]

    def factory(core_id: str, sample_rate: int) -> _FastStubEnhancer:
        # Brief sleep widens the race window across threads.
        time.sleep(0.001)
        return _FastStubEnhancer(core_id, sample_rate)

    cache = WarmCoreCache(
        max_cores=2,
        idle_ttl_s=2.0,
        memory_limit_mb=4096,
        core_factory=factory,
        clock=tick,
    )
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(tid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for step in range(30):
                cid = f"core-{(tid + step) % 4}"
                sr = 48000 if (step % 3) != 0 else 16000
                enh, _ = cache.get_or_load(cid, sample_rate=sr)
                assert enh.metadata.core_id == cid
                assert enh.metadata.sample_rate == sr
                if step % 5 == 0:
                    cache.evict_expired()
                if step % 11 == 0 and tid % 2 == 0:
                    cache.clear()
                st = cache.stats()
                assert st["cached_count"] <= 2
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive(), "WarmCoreCache worker thread deadlocked"

    assert not errors
    assert cache.stats()["cached_count"] <= 2


def test_redteam_10_thread_concurrent_override_and_corrupt_slice_torture(
    client: TestClient, work: Path
) -> None:
    src = _speech_wav(work / "torture_override.wav", seconds=1.8)
    up = client.post(
        "/api/upload",
        headers=H,
        files={"file": ("torture_override.wav", src.read_bytes(), "audio/wav")},
    )
    assert up.status_code == 200
    source_id = up.json()["source_id"]

    sub = client.post(
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
            "idempotencyKey": "redteam-10-thread-override",
        },
    )
    assert sub.status_code == 202, sub.text
    job_id = sub.json()["jobs"][0]["jobId"]
    done = _wait_done(client, job_id)
    assert done["state"] == "done", done

    units = done["report"]["units"]
    speech_unit = next((u for u in units if u.get("is_speech") and u.get("candidate_sha256")), None)
    assert speech_unit is not None
    uid = int(speech_unit["unit_id"])
    ch = int(speech_unit.get("channel", 0))

    # 1. Hammer POST /api/v1/jobs/{id}/override from 10 concurrent threads!
    payloads: list[dict[str, Any]] = [
        {"unitId": uid, "decision": "force_original"},
        {"unitId": uid, "decision": "force_enhanced"},
        {"unitId": uid, "decision": "custom_strength", "strength": 0.25},
        {"unitId": uid, "decision": "custom_strength", "strength": 0.75},
        {"unitId": uid, "decision": "auto"},
        {"unitId": uid, "decision": "force_original"},
        {"unitId": uid, "decision": "custom_strength", "strength": 0.5},
        {"unitId": uid, "decision": "force_enhanced"},
        {"unitId": uid, "decision": "custom_strength", "strength": 0.0},
        {"unitId": uid, "decision": "force_enhanced"},
    ]
    barrier = threading.Barrier(len(payloads))

    def _fire(body: dict[str, Any]) -> int:
        barrier.wait(timeout=10.0)
        resp = client.post(f"/api/v1/jobs/{job_id}/override", headers=H, json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["restitched"] is True
        assert len(data["masterSha256"]) == 64
        return int(resp.status_code)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_fire, payloads))
    assert results == [200] * 10

    # Verify serialized integrity: JSON report is valid and master WAV SHA-256 matches artifact.
    status_after = client.get(f"/api/v1/jobs/{job_id}", headers=H)
    assert status_after.status_code == 200
    status_json = status_after.json()
    report_sha = str(status_json["report"]["output"]["sha256"])
    report_path = Path(status_json["reportPath"])
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk_report["output"]["sha256"] == report_sha

    master_art = client.get(f"/api/v1/jobs/{job_id}/artifacts/master", headers=H)
    assert master_art.status_code == 200
    assert hashlib.sha256(master_art.content).hexdigest() == report_sha
    master_bytes_before_corrupt = master_art.content

    # 2. Corrupt / tamper slice length -> fails closed with 400 and NEVER corrupts master WAV!
    slice_store = UnitSliceStore(work)
    slice_path = slice_store.job_dir(job_id) / f"unit_{uid:06d}_ch{ch}.npz"
    original_slice_bytes = slice_path.read_bytes()

    # 2a. Tamper slice length (rewrite .npz with wrong array length vs unit_meta bounds)
    slice_store.save_unit(
        job_id,
        unit_id=uid,
        original=np.zeros(64, dtype=np.float32),
        candidate=np.zeros(64, dtype=np.float32),
        unit_meta={
            "unit_id": uid,
            "channel": ch,
            "start_sample": int(speech_unit["start_sample"]),
            "end_sample": int(speech_unit["end_sample"]),
        },
        channel=ch,
    )
    r_tamper = client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unitId": uid, "decision": "force_original"},
    )
    assert r_tamper.status_code == 400, r_tamper.text
    assert r_tamper.json()["error"] == "invalid_override"
    _assert_no_leaks(r_tamper.text, work)
    assert (
        client.get(f"/api/v1/jobs/{job_id}/artifacts/master", headers=H).content
        == master_bytes_before_corrupt
    )

    # 2b. Truncate .npz slice to corrupt bytes -> fails closed with 400, master intact
    slice_path.write_bytes(original_slice_bytes[:24])
    r_trunc = client.post(
        f"/api/v1/jobs/{job_id}/override",
        headers=H,
        json={"unitId": uid, "decision": "force_original"},
    )
    assert r_trunc.status_code == 400, r_trunc.text
    _assert_no_leaks(r_trunc.text, work)
    assert (
        client.get(f"/api/v1/jobs/{job_id}/artifacts/master", headers=H).content
        == master_bytes_before_corrupt
    )

    # 2c. Direct restitch_master_with_overrides fails closed on empty / grossly tampered slices
    #     and preserves existing master file on disk.
    direct_master = work / "direct_master.wav"
    direct_master.write_bytes(b"INTACT_MASTER_BYTES")
    tamper_store = UnitSliceStore(work)
    tamper_store.save_unit(
        "tamper_job",
        0,
        np.zeros(0, dtype=np.float32),
        None,
    )
    rec0 = [
        {
            "unit_id": 0,
            "channel": 0,
            "start_sample": 0,
            "end_sample": 24000,
            "is_speech": True,
            "final_decision": "REVERTED",
        }
    ]
    with pytest.raises(ValueError, match="empty original"):
        restitch_master_with_overrides(tamper_store.job_dir("tamper_job"), rec0, {}, direct_master)
    assert direct_master.read_bytes() == b"INTACT_MASTER_BYTES"

    tamper_store.save_unit(
        "tamper_job",
        0,
        np.zeros(24000, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="empty candidate"):
        restitch_master_with_overrides(tamper_store.job_dir("tamper_job"), rec0, {}, direct_master)
    assert direct_master.read_bytes() == b"INTACT_MASTER_BYTES"

    tamper_store.save_unit(
        "tamper_job",
        0,
        np.zeros(100, dtype=np.float32),
        None,
    )
    with pytest.raises(ValueError, match="Tampered cached slice length"):
        restitch_master_with_overrides(tamper_store.job_dir("tamper_job"), rec0, {}, direct_master)
    assert direct_master.read_bytes() == b"INTACT_MASTER_BYTES"

    tamper_store.save_unit(
        "tamper_job",
        0,
        np.zeros(24000, dtype=np.float32),
        None,
        unit_meta={"start_sample": 10, "end_sample": 24010},
    )
    with pytest.raises(ValueError, match="sample bounds mismatch"):
        restitch_master_with_overrides(tamper_store.job_dir("tamper_job"), rec0, {}, direct_master)
    assert direct_master.read_bytes() == b"INTACT_MASTER_BYTES"


# ---------------------------------------------------------------------------
# 5. Pro API & Contract Edge Branches (Coverage Push)
# ---------------------------------------------------------------------------


def test_pro_endpoints_and_helpers_edge_branches(client: TestClient, work: Path) -> None:
    wav = _speech_wav(work / "edge_pro.wav", seconds=1.0, channels=2)
    sess_h = _session_headers(client)

    # Preview with includePerceptualMos=True and startS past file end (400)
    r_mos = client.post(
        "/api/v1/preview",
        headers=H,
        json={
            "path": str(wav),
            "startS": 0.0,
            "endS": 0.6,
            "profile": "production",
            "includePerceptualMos": True,
        },
    )
    assert r_mos.status_code == 200, r_mos.text
    assert "mos" in r_mos.json()["metricsBefore"]
    assert "mos" in r_mos.json()["metricsAfter"]

    r_past = client.post(
        "/api/v1/preview",
        headers=H,
        json={"path": str(wav), "startS": 5.0, "endS": 6.0, "profile": "production"},
    )
    assert r_past.status_code == 400

    # Spectrogram mode="cleaned" + startS past end (400, no host path leak)
    r_clean_spec = client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={
            "path": str(wav),
            "cleanedPath": str(wav),
            "mode": "cleaned",
            "startS": 0.0,
            "endS": 0.5,
            "dbFloor": -80.0,
            "dbCeil": -5.0,
        },
    )
    assert r_clean_spec.status_code == 200, r_clean_spec.text

    r_spec_past = client.post(
        "/api/v1/spectrogram",
        headers=H,
        json={"path": str(wav), "mode": "original", "startS": 5.0, "endS": 6.0},
    )
    assert r_spec_past.status_code == 400
    _assert_no_leaks(r_spec_past.text, work)

    # Spectral repair startS past file end -> 400
    r_sr_past = client.post(
        "/api/v1/spectral-repair",
        headers=H,
        json={
            "path": str(wav),
            "startS": 5.0,
            "endS": 6.0,
            "fLowHz": 300.0,
            "fHighHz": 900.0,
        },
    )
    assert r_sr_past.status_code == 400

    # Export with preserveMetadata=False and default destPath
    r_exp = client.post(
        "/api/v1/export",
        headers=H,
        json={
            "masterPath": str(wav),
            "format": "wav24",
            "deliveryPreset": "podcast",
            "preserveMetadata": False,
        },
    )
    assert r_exp.status_code == 200, r_exp.text
    assert Path(r_exp.json()["outputPath"]).is_file()

    # Non-root session extract to unauthorized outDir -> 403
    up = client.post(
        "/api/upload",
        headers=sess_h,
        files={"file": ("sess_track.wav", wav.read_bytes(), "audio/wav")},
    )
    assert up.status_code == 200
    r_sess_outdir = client.post(
        "/api/v1/tracks/extract",
        headers=sess_h,
        json={
            "path": up.json()["path"],
            "trackIndices": [0],
            "outDir": str(work / "not_allowed_for_session"),
        },
    )
    assert r_sess_outdir.status_code == 403

    # _DeliveryOutputRegistry edge cases: empty, relative, dir, symlink, eviction, missing, replaced
    reg = _DeliveryOutputRegistry(maximum=2)
    assert reg.resolve_authorized_path("") is None
    assert reg.resolve_authorized_path("relative.wav") is None
    d_not_file = work / "dir_export"
    d_not_file.mkdir()
    reg.record(d_not_file)
    assert reg.resolve_authorized_path(str(d_not_file)) is None

    f1 = work / "exp1.wav"
    f2 = work / "exp2.wav"
    f3 = work / "exp3.wav"
    for f in (f1, f2, f3):
        f.write_bytes(b"RIFF")
        reg.record(f)
    # f1 evicted because maximum=2
    assert reg.resolve_authorized_path(str(f1)) is None
    assert reg.resolve_authorized_path(str(f2)) == f2.resolve()
    f2.unlink()
    assert reg.resolve_authorized_path(str(f2)) is None

    try:
        sym_f = work / "sym_exp.wav"
        sym_f.symlink_to(f3)
    except OSError:
        pass
    else:
        reg.record(sym_f)
        assert reg.resolve_authorized_path(str(sym_f)) is None


def test_override_fallback_metadata_only_and_error_states(work: Path) -> None:
    """Exercise metadata-only override fallback (when slices absent) and 404/409/400 branches."""
    src = _speech_wav(work / "mock_ov_src.wav", seconds=0.5)
    out = work / "mock_ov_out.wav"
    rep = work / "mock_ov_out.hawavoclean.json"

    def _fake_cmd(rec: Any) -> list[str]:
        script = (
            "import json, pathlib; "
            f"pathlib.Path({str(rec.output_path)!r}).write_bytes(b'RIFF0000WAVE'); "
            f"pathlib.Path({str(rec.report_path)!r}).write_text(json.dumps({{"
            "'units': ["
            "{'unit_id': 0, 'final_decision': 'reverted', 'guard_a_verdict': 'FAIL', 'is_speech': True}, "
            "{'unit_id': 1, 'final_decision': 'enhanced', 'guard_a_verdict': 'PASS', 'is_speech': True}"
            "], "
            "'summary': {'enhanced': 1, 'reverted': 1}"
            "})); "
            "print(json.dumps({'event': 'done', 'progress': 1.0}), flush=True)"
        )
        return [sys.executable, "-c", script]

    manager = JobManager(command_factory=_fake_cmd)
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None)
    try:
        with TestClient(app, base_url="http://127.0.0.1") as c:
            snap = manager.submit(
                input_path=src, output_path=out, profile="production", overwrite=True
            )
            jid = snap["job_id"]
            done = _wait_done(c, jid)
            assert done["state"] == "done"

            # 404 on unknown job
            assert (
                c.post(
                    "/api/v1/jobs/no-such-job/override",
                    headers=H,
                    json={"unitIndex": 0, "decision": "force_original"},
                ).status_code
                == 404
            )
            # 400 on out-of-range unitIndex and unknown unitId
            assert (
                c.post(
                    f"/api/v1/jobs/{jid}/override",
                    headers=H,
                    json={"unitIndex": 99, "decision": "force_original"},
                ).status_code
                == 400
            )
            assert (
                c.post(
                    f"/api/v1/jobs/{jid}/override",
                    headers=H,
                    json={"unitId": 999, "decision": "force_original"},
                ).status_code
                == 400
            )

            # Metadata-only fallback (no cached slices) for all 4 decisions
            for body, expected_dec in (
                ({"unitIndex": 0, "decision": "force_enhanced"}, "enhanced"),
                ({"unitIndex": 0, "decision": "force_original"}, "reverted"),
                ({"unitIndex": 0, "decision": "custom_strength", "strength": 0.4}, "enhanced"),
                ({"unitIndex": 0, "decision": "custom_strength", "strength": 0.0}, "reverted"),
                ({"unitIndex": 0, "decision": "auto"}, "reverted"),
                ({"unitIndex": 1, "decision": "auto"}, "enhanced"),
            ):
                r = c.post(f"/api/v1/jobs/{jid}/override", headers=H, json=body)
                assert r.status_code == 200, (body, r.text)
                assert r.json()["restitched"] is False
                assert r.json()["new_decision"] == expected_dec
            assert rep.is_file()
    finally:
        manager.shutdown()


def test_pro_contracts_validation_branches() -> None:
    with pytest.raises(ValidationError):
        UnitOverrideRequestV1.model_validate({"decision": "force_original"})
    with pytest.raises(ValidationError):
        UnitOverrideRequestV1.model_validate({"unitIndex": 0, "decision": "custom_strength"})
    with pytest.raises(ValidationError):
        RegionPreviewRequestV1.model_validate(
            {"path": "/a.wav", "startS": 0.5, "endS": 0.2, "profile": "production"}
        )
    with pytest.raises(ValidationError):
        SpectrogramRequestV1.model_validate(
            {"path": "/a.wav", "startS": 1.0, "endS": 0.5, "mode": "original"}
        )
    with pytest.raises(ValidationError):
        SpectrogramRequestV1.model_validate(
            {"path": "/a.wav", "startS": 0.0, "endS": 1.0, "dbFloor": 0.0, "dbCeil": -10.0}
        )
    with pytest.raises(ValidationError):
        SpectralRepairRequestV1.model_validate(
            {"path": "/a.wav", "startS": 0.0, "endS": 1.0, "fLowHz": 1000.0, "fHighHz": 500.0}
        )
    with pytest.raises(ValidationError):
        TrackExtractRequestV1.model_validate({"path": "/a.wav", "trackIndices": [-1]})
    with pytest.raises(ValidationError):
        TrackExtractRequestV1.model_validate({"path": "/a.wav", "trackIndices": [True]})
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "proRepair": "bad"}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "proRepair": {"unknownFlag": True}}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "proRepair": {"declip": "yes"}}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "timeRange": [False, 2.0]}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "timeRange": [2.0, 1.0]}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "timeRange": [0.0, float("inf")]}
        )
    with pytest.raises(ValidationError):
        SeparationStrategyV1.model_validate(
            {
                "kind": "separate",
                "operation": "extract",
                "prompt": "speech",
                "strength": True,
            }
        )
    with pytest.raises(ValidationError):
        SeparationStrategyV1.model_validate(
            {
                "kind": "separate",
                "operation": "extract",
                "prompt": "speech",
                "timeRange": [True, 1.0],
            }
        )
    ok_sep = SeparationStrategyV1.model_validate(
        {
            "kind": "separate",
            "operation": "extract",
            "prompt": "speech",
            "timeRange": [0.5, 2.5],
        }
    )
    assert ok_sep.time_range == (0.5, 2.5)
    assert hash_file(Path(__file__)) != ""
