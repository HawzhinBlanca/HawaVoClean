"""End-to-end tests for live guarded `restore_source`, `studio_pro`, and `smart_safe` wiring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from scipy import signal

import hawavoclean.cli as cli
import hawavoclean.restoration.neural_bwe as neural_bwe
import hawavoclean.server.app as app_module
from hawavoclean.assistant import (
    AcousticContext,
    AudioPlanRequest,
    plan_audio,
)
from hawavoclean.assistant import (
    _schema as assistant_schema,
)
from hawavoclean.errors import ExitCode
from hawavoclean.publication import publication_paths, resolve_committed_publication
from hawavoclean.record_bundle import verify_processing_record
from hawavoclean.restoration.base import RestorationCandidate, RestorationRenderResult
from hawavoclean.restoration.neural_bwe import (
    LAVASR_MODEL_DIR_ENV,
    LAVASR_MODEL_NAME,
    LAVASR_V2_WEIGHTS_SHA256,
    probe_neural_bwe_runtime,
)
from hawavoclean.restoration.protected_band import ProtectedBandVerification
from hawavoclean.server.app import (
    _job_status_v1,
    _restore_source_capability,
    capabilities_v1,
    create_app,
    default_output_path,
)
from hawavoclean.server.contracts import (
    CapabilityStatusV1,
    ManualStrategyV1,
    SeparationStrategyV1,
)
from hawavoclean.server.jobs import TERMINAL_STATES, JobManager, JobRecord, default_command

pytestmark = pytest.mark.unit
TOKEN = "restore-live-token"
HEADERS = {"X-Hawa-Token": TOKEN}


def _make_lowpass_speech_like(
    path: Path,
    *,
    duration_s: float = 1.0,
    sr: int = 48000,
    cutoff_hz: float = 3800.0,
    channels: int = 1,
    subtype: str = "PCM_16",
) -> Path:
    rng = np.random.default_rng(7)
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    # Speech-like harmonic stack + amplitude envelope
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.0 * t)
    sig = (
        0.22 * np.sin(2.0 * np.pi * 160.0 * t)
        + 0.14 * np.sin(2.0 * np.pi * 480.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 1400.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 2800.0 * t)
        + 0.015 * rng.standard_normal(n).astype(np.float32)
    ) * env
    sos = signal.butter(8, cutoff_hz / (sr / 2.0), btype="lowpass", output="sos")
    lp = signal.sosfiltfilt(sos, sig).astype(np.float32)
    data = np.stack([lp, lp], axis=1) if channels == 2 else lp
    sf.write(str(path), data, sr, format="WAV", subtype=subtype)
    return path


def _make_fullband_wav(path: Path, *, duration_s: float = 0.8, sr: int = 48000) -> Path:
    rng = np.random.default_rng(11)
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    sig = (
        0.18 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 16000.0 * t)
        + 0.03 * rng.standard_normal(n).astype(np.float32)
    ).astype(np.float32)
    sf.write(str(path), sig, sr, format="WAV", subtype="PCM_16")
    return path


def _run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["hawavoclean", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def _fake_highband_generator(
    chunk: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Fast deterministic synthetic BWE generator adding safe 8 kHz harmonic shimmer."""
    t = np.arange(len(chunk), dtype=np.float32) / float(sr)
    hf = 0.015 * np.sin(2.0 * np.pi * 8000.0 * t)
    return np.ascontiguousarray(chunk + hf, dtype=np.float32)


class _FastRestorer(neural_bwe.NeuralBWERestorer):
    def __init__(self, _device: str = "auto", **kwargs: Any) -> None:
        kwargs.pop("device", None)
        kwargs.pop("generator", None)
        super().__init__(device="cpu", generator=_fake_highband_generator, **kwargs)


def _install_fast_restorer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch probe + NeuralBWERestorer to run cleanly and deterministically without weights."""
    monkeypatch.setattr(
        neural_bwe,
        "probe_neural_bwe_runtime",
        lambda _model_dir=None: (True, "LavaSR v2 weights verified (Apache-2.0)"),
    )
    monkeypatch.setattr(neural_bwe, "NeuralBWERestorer", _FastRestorer)


def _wait_v1_terminal(client: TestClient, job_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        res = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
        assert res.status_code == 200
        last = res.json()
        if last["state"] in {"completed", "failed", "cancelled", "interrupted"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"v1 job {job_id} did not finish: {last}")


# ---------------------------------------------------------------------------
# Capabilities & redaction
# ---------------------------------------------------------------------------


def test_restore_source_capability_available_and_blocked_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Available path
    monkeypatch.setattr(
        neural_bwe,
        "probe_neural_bwe_runtime",
        lambda _model_dir=None: (True, "LavaSR v2 weights verified (Apache-2.0)"),
    )
    cap_ok = _restore_source_capability()
    assert cap_ok.capability_id == "restore_source"
    assert cap_ok.available is True
    assert cap_ok.maturity == "experimental"
    assert cap_ok.reason is not None and "LavaSR v2" in cap_ok.reason
    assert len(cap_ok.providers) == 1 and cap_ok.providers[0] in {"mps", "cuda", "cpu"}

    # 2. Missing weights / path-redacted failure
    missing_dir = tmp_path / "secret-lavasr-dir"
    monkeypatch.setenv(LAVASR_MODEL_DIR_ENV, str(missing_dir))
    monkeypatch.setattr(
        neural_bwe,
        "probe_neural_bwe_runtime",
        lambda _model_dir=None: (False, f"missing under {missing_dir} and {Path.home()}/priv"),
    )
    cap_blocked = _restore_source_capability()
    assert cap_blocked.available is False
    assert cap_blocked.maturity == "blocked"
    assert cap_blocked.providers == []
    assert cap_blocked.reason is not None
    assert str(missing_dir) not in cap_blocked.reason
    assert str(Path.home()) not in cap_blocked.reason
    assert "<lavasr-model-dir>" in cap_blocked.reason
    assert "<home>" in cap_blocked.reason

    # 2b. Missing weights without LAVASR_MODEL_DIR_ENV set (sanitizes <home> cleanly)
    monkeypatch.delenv(LAVASR_MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(
        neural_bwe,
        "probe_neural_bwe_runtime",
        lambda _model_dir=None: (False, f"missing under {Path.home()}/priv"),
    )
    cap_no_env = _restore_source_capability()
    assert cap_no_env.available is False and cap_no_env.maturity == "blocked"
    assert cap_no_env.reason is not None and "<home>" in cap_no_env.reason
    assert str(Path.home()) not in cap_no_env.reason

    # 3. Probe exception fails closed with redacted message
    monkeypatch.setenv(LAVASR_MODEL_DIR_ENV, str(missing_dir))

    def _boom(_model_dir: Path | None = None) -> tuple[bool, str]:
        raise RuntimeError(f"broken runtime at {missing_dir}")

    monkeypatch.setattr(neural_bwe, "probe_neural_bwe_runtime", _boom)
    cap_err = _restore_source_capability()
    assert cap_err.available is False
    assert cap_err.maturity == "blocked"
    assert cap_err.reason is not None
    assert "Source restoration runtime probe failed" in cap_err.reason
    assert str(missing_dir) not in cap_err.reason
    assert "<lavasr-model-dir>" in cap_err.reason

    # 4. restore_enrolled remains permanently blocked in capabilities_v1()
    monkeypatch.setattr(
        neural_bwe,
        "probe_neural_bwe_runtime",
        lambda _model_dir=None: (True, "LavaSR v2 weights verified (Apache-2.0)"),
    )
    caps = {c.capability_id: c for c in capabilities_v1().capabilities}
    assert caps["restore_source"].available is True
    assert caps["restore_source"].maturity == "experimental"
    assert caps["restore_enrolled"].available is False
    assert caps["restore_enrolled"].maturity == "blocked"


def test_default_output_path_and_job_status_v1_restore_stage() -> None:
    assert default_output_path(Path("/a/clip.m4a"), "restore_source") == Path(
        "/a/clip_restored.wav"
    )
    status = _job_status_v1(
        {
            "job_id": "j_r1",
            "state": "running",
            "stage": "restore",
            "progress": 0.65,
            "message": "Restoring",
            "output_path": "/tmp/out_restored.wav",
            "report_path": "/tmp/out_restored.hawavoclean.json",
            "created_at": "2026-11-19T00:00:00Z",
        }
    )
    assert status.state == "rendering"


# ---------------------------------------------------------------------------
# Job command plumbing (jobs.py)
# ---------------------------------------------------------------------------


def test_default_command_source_restore_vs_enrolled_restore(tmp_path: Path) -> None:
    src_rec = JobRecord(
        job_id="j_src",
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "in_restored.wav",
        report_path=tmp_path / "in_restored.hawavoclean.json",
        profile="studio_pro",
        overwrite=True,
        mode="restore",
        speaker_id=None,
        cutoff_hz=3800.0,
        strength=0.75,
        time_range=(0.25, 1.5),
    )
    cmd = default_command(src_rec)
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "restore"
    assert "--speaker-id" not in cmd
    assert "--profiles-dir" not in cmd
    assert "--cutoff-hz" in cmd and cmd[cmd.index("--cutoff-hz") + 1] == "3800.0"
    assert "--strength" in cmd and cmd[cmd.index("--strength") + 1] == "0.75"
    assert "--time-range" in cmd
    tr_idx = cmd.index("--time-range")
    assert cmd[tr_idx + 1 : tr_idx + 3] == ["0.25", "1.5"]


# ---------------------------------------------------------------------------
# Server v1 API consent, blocked-when-missing, dispatch & profile selection
# ---------------------------------------------------------------------------


def test_v1_restore_source_consent_blocked_and_dispatch_profile_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))

    captured: list[JobRecord] = []

    def _capture_factory(record: JobRecord) -> list[str]:
        captured.append(record)
        report = {"schema_version": 1, "summary": {"units_total": 1, "enhanced": 1}}
        return [
            sys.executable,
            "-u",
            "-c",
            (
                "import json, sys; "
                f"open({str(record.report_path)!r}, 'w').write({json.dumps(json.dumps(report))}); "
                f"open({str(record.output_path)!r}, 'wb').write(b'RIFF'); "
                "sys.stdout.write(json.dumps({'event':'done','progress':1.0}) + '\\n'); sys.stdout.flush()"
            ),
        ]

    manager = JobManager(command_factory=_capture_factory)
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None, min_free_bytes=0)
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            wav_path = _make_lowpass_speech_like(tmp_path / "sample.wav", duration_s=0.4)
            upload = client.post(
                "/api/upload",
                headers=HEADERS,
                files={"file": ("sample.wav", wav_path.read_bytes(), "audio/wav")},
            )
            assert upload.status_code == 200
            source_id = upload.json()["source_id"]

            # 1. Missing reconstruction consent -> 400 bad_request
            no_consent = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "restore_source",
                        "allowGenerativeReconstruction": False,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "restore-no-consent",
                },
            )
            assert no_consent.status_code == 400
            assert "consent" in no_consent.json()["message"].lower()

            # 2. Missing LavaSR weights -> 503 capability_blocked
            monkeypatch.setenv(LAVASR_MODEL_DIR_ENV, str(tmp_path / "no-lavasr"))
            blocked = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "restore_source",
                        "allowGenerativeReconstruction": True,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "restore-blocked",
                },
            )
            assert blocked.status_code == 503
            assert blocked.json()["error"] == "capability_blocked"

            # 3. restore_enrolled stays 503 even when restore_source is unlocked
            monkeypatch.setattr(
                neural_bwe,
                "probe_neural_bwe_runtime",
                lambda _model_dir=None: (True, "LavaSR v2 weights verified (Apache-2.0)"),
            )
            enrolled_blocked = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "restore_enrolled",
                        "speakerProfileId": "character_01",
                        "allowGenerativeReconstruction": True,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "restore-enrolled-blocked",
                },
            )
            assert enrolled_blocked.status_code == 503
            assert enrolled_blocked.json()["error"] == "capability_blocked"

            # 4. Live restore_source dispatch: falls back studio_pro -> studio -> production
            def _only_studio_qualified(profile: str) -> CapabilityStatusV1:
                if profile == "studio":
                    return CapabilityStatusV1(
                        capability_id="studio",
                        available=True,
                        maturity="qualified",
                        providers=["cpu"],
                    )
                return CapabilityStatusV1(
                    capability_id=profile,
                    available=False,
                    maturity="blocked",
                    reason="not installed",
                )

            monkeypatch.setattr(app_module, "_natural_route_capability", _only_studio_qualified)
            ok = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "restore_source",
                        "allowGenerativeReconstruction": True,
                        "expertCutoffHz": 3900.0,
                        "strength": 0.7,
                        "timeRange": [0.05, 0.3],
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "restore-live-dispatch",
                },
            )
            assert ok.status_code == 202, ok.text
            job_item = ok.json()["jobs"][0]
            assert job_item["outputPath"].endswith("_restored.wav")
            final = _wait_v1_terminal(client, job_item["jobId"])
            assert final["state"] == "completed"
            assert len(captured) == 1
            rec = captured[0]
            assert rec.mode == "restore"
            assert rec.speaker_id is None
            assert rec.profile == "studio"
            assert rec.cutoff_hz == 3900.0
            assert rec.strength == 0.7
            assert rec.time_range == (0.05, 0.3)

            # 5. If both studio_pro and studio are blocked, falls back cleanly to production
            upload2 = client.post(
                "/api/upload",
                headers=HEADERS,
                files={"file": ("sample2.wav", wav_path.read_bytes(), "audio/wav")},
            )
            source_id_2 = upload2.json()["source_id"]
            monkeypatch.setattr(
                app_module,
                "_natural_route_capability",
                lambda profile: CapabilityStatusV1(
                    capability_id=profile, available=False, maturity="blocked", reason="absent"
                ),
            )
            ok2 = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id_2],
                    "strategy": {
                        "kind": "manual",
                        "route": "restore_source",
                        "allowGenerativeReconstruction": True,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": False,
                    "idempotencyKey": "restore-fallback-prod",
                },
            )
            assert ok2.status_code == 202
            assert _wait_v1_terminal(client, ok2.json()["jobs"][0]["jobId"])["state"] == "completed"
            assert captured[1].profile == "production"
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# CLI guarded restore_source pipeline unit + branch coverage
# ---------------------------------------------------------------------------


def test_cli_helpers_profile_and_strength_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import hawavoclean.natural_contract as nc

    assert cli._source_restore_strengths(1.0) == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert cli._source_restore_strengths(0.6) == [0.6, 0.5, 0.25, 0.0]
    assert cli._source_restore_strengths(0.2) == [0.2, 0.0]

    def _only_prod(profile: str, **_kwargs: Any) -> object:
        if profile != "production":
            raise RuntimeError("not installed")
        return object()

    monkeypatch.setattr(nc, "load_natural_route_contract", _only_prod)
    assert cli._select_restore_clean_profile(None) == "production"
    assert cli._select_restore_clean_profile("studio_pro") == "production"
    assert cli._select_restore_clean_profile("studio") == "production"

    def _none_ok(_profile: str, **_kwargs: Any) -> object:
        raise RuntimeError("none")

    monkeypatch.setattr(nc, "load_natural_route_contract", _none_ok)
    assert cli._select_restore_clean_profile(None) == "production"


def test_run_source_restoration_direct_guard_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hawavoclean.errors import InvalidUserInputError, PreflightError

    src = _make_lowpass_speech_like(tmp_path / "direct.wav", duration_s=0.4)
    base_args = argparse.Namespace(
        input=str(src),
        output=str(tmp_path / "out.wav"),
        config=None,
        profile="production",
        cutoff="bogus",
        cutoff_hz=None,
        overwrite=True,
        original_input_path=None,
    )
    with pytest.raises(InvalidUserInputError, match="Unknown cutoff mode"):
        cli._run_source_restoration(
            args=base_args,
            requested_profile="production",
            strength=1.0,
            time_range=None,
            clean_only=False,
            on_progress=None,
        )

    base_args.cutoff = "auto"
    monkeypatch.setattr(
        neural_bwe, "probe_neural_bwe_runtime", lambda _model_dir=None: (False, "absent")
    )
    with pytest.raises(PreflightError, match="Source restoration runtime unavailable"):
        cli._run_source_restoration(
            args=base_args,
            requested_profile="production",
            strength=1.0,
            time_range=None,
            clean_only=False,
            on_progress=None,
        )


def test_cli_source_restore_missing_weights_emits_json_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LAVASR_MODEL_DIR_ENV, str(tmp_path / "no-lavasr"))
    events: list[dict[str, Any]] = []

    class _CaptureSink:
        def emit(self, obj: dict[str, Any]) -> None:
            events.append(obj)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "_JsonLineSink", _CaptureSink)
    src = _make_lowpass_speech_like(tmp_path / "in.wav", duration_s=0.4)
    args = argparse.Namespace(
        input=str(src),
        output=str(tmp_path / "out.wav"),
        config=None,
        profile=None,
        mode="restore",
        speaker_id=None,
        cutoff="auto",
        cutoff_hz=None,
        profiles_dir=None,
        strength=None,
        time_range=None,
        overwrite=True,
        record_bundle=None,
        progress_json=True,
        passes=1,
        clean_only=False,
        original_input_path=None,
        allow_research_restore=False,
    )
    rc = cli.cmd_process(args)
    assert rc == int(ExitCode.INVALID_USER_INPUT)
    assert any(
        ev.get("event") == "error" and ev.get("code") == "INVALID_USER_INPUT" for ev in events
    )


def test_cli_source_restore_preflight_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fast_restorer(monkeypatch)
    src = _make_lowpass_speech_like(tmp_path / "in.wav", duration_s=0.5)

    # Bad extension
    rc_ext = _run_cli(
        monkeypatch, "process", str(src), "-o", str(tmp_path / "bad.mp3"), "--mode", "restore"
    )
    assert rc_ext == int(ExitCode.PUBLICATION_FAILURE)

    # Output is directory
    out_dir = tmp_path / "adir.wav"
    out_dir.mkdir()
    rc_dir = _run_cli(monkeypatch, "process", str(src), "-o", str(out_dir), "--mode", "restore")
    assert rc_dir == int(ExitCode.PUBLICATION_FAILURE)

    # Output == input
    rc_same = _run_cli(
        monkeypatch, "process", str(src), "-o", str(src), "--mode", "restore", "--overwrite"
    )
    assert rc_same == int(ExitCode.PUBLICATION_FAILURE)

    # Existing output without --overwrite
    existing = tmp_path / "existing.wav"
    existing.write_bytes(b"RIFF")
    rc_exist = _run_cli(monkeypatch, "process", str(src), "-o", str(existing), "--mode", "restore")
    assert rc_exist == int(ExitCode.PUBLICATION_FAILURE)

    # --cutoff manual without --cutoff-hz
    rc_cut = _run_cli(
        monkeypatch,
        "process",
        str(src),
        "-o",
        str(tmp_path / "cut.wav"),
        "--mode",
        "restore",
        "--cutoff",
        "manual",
    )
    assert rc_cut == int(ExitCode.INVALID_USER_INPUT)

    # --time-range start exceeds duration
    rc_range = _run_cli(
        monkeypatch,
        "process",
        str(src),
        "-o",
        str(tmp_path / "range.wav"),
        "--mode",
        "restore",
        "--time-range",
        "5.0",
        "6.0",
    )
    assert rc_range == int(ExitCode.INVALID_USER_INPUT)

    # --strength rejected with --mode smart_safe
    rc_ss = _run_cli(
        monkeypatch,
        "process",
        str(src),
        "-o",
        str(tmp_path / "ss.wav"),
        "--mode",
        "smart_safe",
        "--strength",
        "0.5",
    )
    assert rc_ss == int(ExitCode.INVALID_USER_INPUT)


def test_cli_source_restore_active_bundle_and_16khz_stereo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fast_restorer(monkeypatch)
    src = _make_lowpass_speech_like(
        tmp_path / "phone_16k.wav", duration_s=0.8, sr=16000, cutoff_hz=3600.0, channels=2
    )
    out = tmp_path / "phone_restored.wav"
    bundle = tmp_path / "phone_restored.hawavoclean.zip"

    events: list[dict[str, Any]] = []

    class _CaptureSink:
        def emit(self, obj: dict[str, Any]) -> None:
            events.append(obj)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "_JsonLineSink", _CaptureSink)
    rc = _run_cli(
        monkeypatch,
        "process",
        str(src),
        "-o",
        str(out),
        "--profile",
        "production",
        "--mode",
        "restore",
        "--strength",
        "0.8",
        "--record-bundle",
        str(bundle),
        "--progress-json",
        "--overwrite",
    )
    assert rc == int(ExitCode.SUCCESS)
    assert any(ev.get("event") == "progress" and ev.get("stage") == "enhance" for ev in events)
    assert any(ev.get("event") == "done" for ev in events)
    assert out.is_file()
    assert bundle.is_file()
    assert resolve_committed_publication(out) is not None

    report_path = publication_paths(out).json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["output"]["sample_rate"] == 48000
    assert report["output"]["channels"] == 2
    assert report["output"]["true_peak_dbtp"] <= -1.0
    rest = report["restoration"]
    assert rest["mode"] == "restore_source"
    assert rest["maturity"] == "experimental"
    assert rest["reconstruction_disclosed"] is True
    assert rest["speaker_id"] is None
    assert rest["restorer"]["name"] == LAVASR_MODEL_NAME
    assert rest["restorer"]["weights_sha256"] == LAVASR_V2_WEIGHTS_SHA256
    assert rest["guard_r"]["verdict"] == "PASS"
    assert rest["guard_r"]["accepted_strength"] == pytest.approx(0.8)
    assert rest["protected_band"]["passes_invariance"] is True
    assert rest["protected_band"]["worst_band_energy_deviation_db"] <= 6.0
    assert rest["post_mastering_verification"]["passes"] is True

    # Full Processing Record ZIP verifies cleanly
    verified = verify_processing_record(bundle)
    assert verified.master_sha256 == report["output"]["sha256"]

    # CLI verify command passes on the published pair
    assert _run_cli(monkeypatch, "verify", str(out), "-r", str(report_path)) == int(
        ExitCode.SUCCESS
    )


def test_cli_source_restore_fullband_bypass_and_surgical_time_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fast_restorer(monkeypatch)

    # 1. Fullband input -> bypassed fail-closed without calling NeuralBWERestorer
    fullband = _make_fullband_wav(tmp_path / "fullband.wav", duration_s=0.8)
    out_fb = tmp_path / "fullband_restored.wav"
    assert (
        _run_cli(
            monkeypatch,
            "process",
            str(fullband),
            "-o",
            str(out_fb),
            "--profile",
            "production",
            "--mode",
            "restore",
            "--overwrite",
        )
        == 0
    )
    rep_fb = json.loads(publication_paths(out_fb).json.read_text(encoding="utf-8"))
    assert rep_fb["restoration"]["guard_r"]["verdict"] == "NO_RESTORE"
    assert rep_fb["restoration"]["guard_r"]["accepted_strength"] == 0.0
    assert rep_fb["restoration"]["segments"]["bypassed"] == 1

    # 2. Surgical --time-range on FLOAT32 WAV -> exact bit-identity outside [start, end)
    src_f32 = _make_lowpass_speech_like(
        tmp_path / "clip_f32.wav", duration_s=1.0, sr=48000, cutoff_hz=3800.0, subtype="FLOAT"
    )
    out_tr = tmp_path / "clip_f32_restored.wav"
    assert (
        _run_cli(
            monkeypatch,
            "process",
            str(src_f32),
            "-o",
            str(out_tr),
            "--profile",
            "production",
            "--mode",
            "restore",
            "--cutoff-hz",
            "3800",
            "--time-range",
            "0.25",
            "0.75",
            "--overwrite",
        )
        == 0
    )
    orig_wave, sr_in = sf.read(str(src_f32), dtype="float32")
    out_wave, sr_out = sf.read(str(out_tr), dtype="float32")
    assert sr_in == sr_out == 48000
    assert len(orig_wave) == len(out_wave)
    start_idx = int(round(0.25 * sr_in))
    end_idx = int(round(0.75 * sr_in))
    assert np.array_equal(out_wave[:start_idx], orig_wave[:start_idx])
    assert np.array_equal(out_wave[end_idx:], orig_wave[end_idx:])
    rep_tr = json.loads(publication_paths(out_tr).json.read_text(encoding="utf-8"))
    assert rep_tr["restoration"]["time_range"] == [0.25, 0.75]
    assert rep_tr["restoration"]["bandwidth"]["cutoff_mode"] == "manual"
    assert rep_tr["restoration"]["guard_r"]["verdict"] == "PASS"


def test_cli_source_restore_fail_closed_reverts_on_invariance_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hawavoclean.restoration.protected_band as pb_mod

    src = _make_lowpass_speech_like(tmp_path / "lp.wav", duration_s=0.6)

    # Case 1: NeuralBWERestorer.render reports invariance_violation -> FAIL revert
    class _FailInvRestorer:
        device = "cpu"

        def __init__(self, **_kwargs: Any) -> None:
            return None

        def render(
            self,
            audio_48k: np.ndarray[Any, Any],
            **_kwargs: Any,
        ) -> RestorationRenderResult:
            return RestorationRenderResult(
                success=False,
                fallback_status="invariance_violation",
                model_name=LAVASR_MODEL_NAME,
                provider="cpu",
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(strength=0.0, audio=audio_48k.copy(), cutoff_hz=3800.0)
                ],
                error_message="strength=1 failed protected-band invariance (worst_band=12.40 dB)",
            )

    monkeypatch.setattr(
        neural_bwe, "probe_neural_bwe_runtime", lambda _model_dir=None: (True, "ok")
    )
    monkeypatch.setattr(neural_bwe, "NeuralBWERestorer", _FailInvRestorer)
    out1 = tmp_path / "revert_inv.wav"
    assert (
        _run_cli(
            monkeypatch,
            "process",
            str(src),
            "-o",
            str(out1),
            "--profile",
            "production",
            "--mode",
            "restore",
            "--cutoff-hz",
            "3800",
            "--overwrite",
        )
        == 0
    )
    rep1 = json.loads(publication_paths(out1).json.read_text(encoding="utf-8"))
    assert rep1["restoration"]["guard_r"]["verdict"] == "FAIL"
    assert rep1["restoration"]["guard_r"]["accepted_strength"] == 0.0
    assert rep1["restoration"]["segments"]["reverted"] == 1
    assert rep1["restoration"]["protected_band"]["passes_invariance"] is False

    # Case 2: NeuralBWERestorer.render reports runtime_error -> ERROR revert
    class _ErrRestorer(_FailInvRestorer):
        def render(
            self,
            audio_48k: np.ndarray[Any, Any],
            **_kwargs: Any,
        ) -> RestorationRenderResult:
            return RestorationRenderResult(
                success=False,
                fallback_status="runtime_error",
                model_name=LAVASR_MODEL_NAME,
                provider="cpu",
                solver="lavasr_v2_protected_band",
                candidates=[
                    RestorationCandidate(strength=0.0, audio=audio_48k.copy(), cutoff_hz=3800.0)
                ],
                error_message="RuntimeError: simulated OOM",
            )

    monkeypatch.setattr(neural_bwe, "NeuralBWERestorer", _ErrRestorer)
    out2 = tmp_path / "revert_err.wav"
    assert (
        _run_cli(
            monkeypatch,
            "process",
            str(src),
            "-o",
            str(out2),
            "--profile",
            "production",
            "--mode",
            "restore",
            "--cutoff-hz",
            "3800",
            "--overwrite",
        )
        == 0
    )
    rep2 = json.loads(publication_paths(out2).json.read_text(encoding="utf-8"))
    assert rep2["restoration"]["guard_r"]["verdict"] == "ERROR"
    assert rep2["restoration"]["segments"]["errors"] == 1

    # Case 3: Independent pre-mastering verify_protected_band_invariance fails -> FAIL revert
    _install_fast_restorer(monkeypatch)
    bad_pb = ProtectedBandVerification(
        max_waveform_abs_error=0.2,
        rms_waveform_error=0.1,
        complex_stft_relative_error=0.3,
        max_phase_deviation_rad=1.0,
        worst_band_energy_deviation_db=9.5,
        worst_band_center_hz=1200.0,
        passes_invariance=False,
    )
    monkeypatch.setattr(pb_mod, "verify_protected_band_invariance", lambda *_a, **_kw: bad_pb)
    out3 = tmp_path / "revert_pre_pb.wav"
    assert (
        _run_cli(
            monkeypatch,
            "process",
            str(src),
            "-o",
            str(out3),
            "--profile",
            "production",
            "--mode",
            "restore",
            "--cutoff-hz",
            "3800",
            "--overwrite",
        )
        == 0
    )
    rep3 = json.loads(publication_paths(out3).json.read_text(encoding="utf-8"))
    assert rep3["restoration"]["guard_r"]["verdict"] == "FAIL"
    assert rep3["restoration"]["segments"]["reverted"] == 1

    # Case 4: Pre-mastering passes, post-mastering verify_protected_band_invariance fails -> revert
    good_pb = ProtectedBandVerification(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    calls = {"n": 0}

    def _pass_then_fail(*_args: Any, **_kwargs: Any) -> ProtectedBandVerification:
        calls["n"] += 1
        return good_pb if calls["n"] == 1 else bad_pb

    monkeypatch.setattr(pb_mod, "verify_protected_band_invariance", _pass_then_fail)
    out4 = tmp_path / "revert_post_pb.wav"
    assert (
        _run_cli(
            monkeypatch,
            "process",
            str(src),
            "-o",
            str(out4),
            "--profile",
            "production",
            "--mode",
            "restore",
            "--cutoff-hz",
            "3800",
            "--overwrite",
        )
        == 0
    )
    rep4 = json.loads(publication_paths(out4).json.read_text(encoding="utf-8"))
    assert rep4["restoration"]["guard_r"]["verdict"] == "FAIL"
    assert rep4["restoration"]["post_mastering_verification"]["passes"] is False
    assert rep4["restoration"]["post_mastering_verification"]["fallback_applied"] is True


# ---------------------------------------------------------------------------
# Live .venv end-to-end API -> JobManager -> CLI -> LavaSR v2 MPS test
# ---------------------------------------------------------------------------


def test_live_v1_api_restore_source_end_to_end_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When LavaSR v2 weights are present in .venv, prove real end-to-end API execution."""
    ok, _reason = probe_neural_bwe_runtime()
    if not ok:
        pytest.skip("LavaSR v2 weights not staged in this environment")

    monkeypatch.setenv("HAWAVOCLEAN_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("HAWAVOCLEAN_PROFILES_DIR", str(tmp_path / "profiles"))

    wav_path = _make_lowpass_speech_like(
        tmp_path / "narrowband_live.wav", duration_s=1.2, sr=48000, cutoff_hz=3800.0
    )
    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None, min_free_bytes=0)
    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            caps = {
                c["capabilityId"]: c
                for c in client.get("/api/v1/capabilities", headers=HEADERS).json()["capabilities"]
            }
            assert caps["restore_source"]["available"] is True
            assert caps["restore_source"]["maturity"] == "experimental"

            up = client.post(
                "/api/upload",
                headers=HEADERS,
                files={"file": ("narrowband_live.wav", wav_path.read_bytes(), "audio/wav")},
            )
            assert up.status_code == 200
            source_id = up.json()["source_id"]

            res = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "restore_source",
                        "allowGenerativeReconstruction": True,
                        "expertCutoffHz": 3800.0,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "recordBundle": True,
                    "idempotencyKey": "live-restore-e2e",
                },
            )
            assert res.status_code == 202, res.text
            job_id = res.json()["jobs"][0]["jobId"]
            final = _wait_v1_terminal(client, job_id, timeout_s=120.0)
            assert final["state"] == "completed", final
            report = final["report"]
            assert report is not None
            assert report["restoration"]["mode"] == "restore_source"
            assert report["restoration"]["maturity"] == "experimental"
            assert report["restoration"]["guard_r"]["verdict"] == "PASS"
            assert report["restoration"]["protected_band"]["passes_invariance"] is True
            assert report["restoration"]["protected_band"]["worst_band_energy_deviation_db"] < 0.05
            assert report["output"]["true_peak_dbtp"] <= -1.0
            assert final["bundle"]["internalHashesVerified"] is True

            master = client.get(f"/api/v1/jobs/{job_id}/artifacts/master", headers=HEADERS)
            assert master.status_code == 200 and master.content[:4] == b"RIFF"
            record_zip = client.get(f"/api/v1/jobs/{job_id}/artifacts/record", headers=HEADERS)
            assert record_zip.status_code == 200 and record_zip.content[:2] == b"PK"

            # Verify JobManager internal terminal state
            snap = manager.get_status(job_id)
            assert snap is not None and snap["state"] in TERMINAL_STATES
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# Mono dereverb (`dereverb` / studio-dereverb-48k-v1) wiring & assistant routing
# ---------------------------------------------------------------------------


def test_dereverb_contract_capabilities_dispatch_and_assistant_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Contracts, CLI choices, and default output path
    assert "dereverb" in cli.PROFILE_CHOICES
    manual = ManualStrategyV1(kind="manual", route="dereverb")
    assert manual.route == "dereverb"
    sep = SeparationStrategyV1(
        kind="separate",
        operation="extract",
        prompt="speech",
        cleanup_profile="dereverb",
    )
    assert sep.cleanup_profile == "dereverb"
    assert default_output_path(Path("/a/clip.m4a"), "dereverb") == Path("/a/clip_dereverb.wav")

    # 2. Assistant schema + rules/telemetry routing for dereverb
    monkeypatch.setenv("HAWAVOCLEAN_CHAT_PROVIDER", "rules")
    schema = assistant_schema()
    by_op = {variant["properties"]["operation"]["const"]: variant for variant in schema["anyOf"]}
    assert "dereverb" in by_op["clean"]["properties"]["cleanup_profile"]["enum"]
    assert "dereverb" in by_op["extract"]["properties"]["cleanup_profile"]["anyOf"][0]["enum"]

    ctx_dereverb = AcousticContext(
        reverberation_risk=0.52,
        noise_risk=0.20,
        available_profiles=["production", "studio", "studio_pro", "dereverb"],
    )
    plan_reverb_only = plan_audio(
        AudioPlanRequest(message="Clean this voice", acoustic_context=ctx_dereverb)
    )
    assert plan_reverb_only.operation == "clean"
    assert plan_reverb_only.cleanup_profile == "dereverb"
    assert "reverberation risk 52%" in plan_reverb_only.explanation

    ctx_reverb_and_noise = AcousticContext(
        reverberation_risk=0.45,
        noise_risk=0.65,
        available_profiles=["production", "studio", "studio_pro", "dereverb"],
    )
    plan_both = plan_audio(
        AudioPlanRequest(message="Recommend cleanup", acoustic_context=ctx_reverb_and_noise)
    )
    assert plan_both.operation == "clean"
    assert plan_both.cleanup_profile == "dereverb"
    assert "noise risk 65%" in plan_both.explanation

    ctx_no_dereverb = AcousticContext(
        reverberation_risk=0.65,
        available_profiles=["production", "studio", "studio_pro"],
    )
    plan_fallback = plan_audio(
        AudioPlanRequest(message="Remove room echo", acoustic_context=ctx_no_dereverb)
    )
    assert plan_fallback.operation == "clean"
    assert plan_fallback.cleanup_profile == "studio_pro"

    for prompt in (
        "Remove room echo",
        "Remove the room echo from this recording",
        "Clean this interview and reduce room echo",
        "Dereverb this voice",
    ):
        plan_prompt = plan_audio(AudioPlanRequest(message=prompt, acoustic_context=ctx_dereverb))
        assert plan_prompt.operation == "clean", prompt
        assert plan_prompt.cleanup_profile == "dereverb", prompt

    # Explicit "dereverb" works even without acoustic context; generic room echo falls back cleanly
    plan_explicit_no_ctx = plan_audio(AudioPlanRequest(message="Please de-reverb this clip"))
    assert plan_explicit_no_ctx.operation == "clean"
    assert plan_explicit_no_ctx.cleanup_profile == "dereverb"

    plan_clean_echo_no_dereverb = plan_audio(
        AudioPlanRequest(
            message="Clean this interview and reduce room echo",
            acoustic_context=ctx_no_dereverb,
        )
    )
    assert plan_clean_echo_no_dereverb.operation == "clean"
    assert plan_clean_echo_no_dereverb.cleanup_profile == "studio_pro"

    plan_extract_reverb = plan_audio(
        AudioPlanRequest(
            message="Keep speech and remove room reverb",
            acoustic_context=ctx_dereverb,
        )
    )
    assert plan_extract_reverb.operation == "extract"
    assert plan_extract_reverb.target == "speech"
    assert plan_extract_reverb.cleanup_profile == "dereverb"

    plan_extract_no_dereverb = plan_audio(
        AudioPlanRequest(
            message="Keep speech and remove room reverb",
            acoustic_context=ctx_no_dereverb,
        )
    )
    assert plan_extract_no_dereverb.operation == "extract"
    assert plan_extract_no_dereverb.cleanup_profile == "studio_pro"

    # 3. Server API capabilities_v1() + POST /api/v1/jobs qualified dispatch & fail-closed block
    wav_path = _make_fullband_wav(tmp_path / "reverb_clip.wav")
    submitted: list[dict[str, Any]] = []
    dereverb_ready = True

    def _fake_nat(profile: str) -> CapabilityStatusV1:
        if profile == "dereverb" and not dereverb_ready:
            return CapabilityStatusV1(
                capability_id="dereverb",
                available=False,
                maturity="blocked",
                reason="dereverb weights missing",
            )
        return CapabilityStatusV1(
            capability_id=profile,
            available=True,
            maturity="qualified",
            reason=f"{profile} verified",
            providers=["mps"] if profile in {"studio_pro", "dereverb"} else ["cpu"],
        )

    monkeypatch.setattr(app_module, "_natural_route_capability", _fake_nat)
    caps = {c.capability_id: c for c in capabilities_v1().capabilities}
    assert "dereverb" in caps
    assert caps["dereverb"].available is True
    assert caps["dereverb"].maturity == "qualified"
    assert caps["dereverb"].providers == ["mps"]

    manager = JobManager()
    app = create_app(TOKEN, None, job_manager=manager, on_shutdown=lambda: None, min_free_bytes=0)

    def _capture_submit(**kwargs: Any) -> dict[str, Any]:
        submitted.append(kwargs)
        out_path = Path(kwargs["output_path"])
        return {
            "job_id": f"j_derev_{len(submitted)}",
            "state": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Queued",
            "output_path": str(out_path),
            "report_path": str(out_path.with_suffix(".hawavoclean.json")),
            "created_at": "2026-11-19T00:00:00Z",
        }

    monkeypatch.setattr(manager, "submit", _capture_submit)

    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            up = client.post(
                "/api/upload",
                headers=HEADERS,
                files={"file": ("reverb_clip.wav", wav_path.read_bytes(), "audio/wav")},
            )
            assert up.status_code == 200
            source_id = up.json()["source_id"]

            ok_res = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "dereverb",
                        "strength": 0.85,
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "idempotencyKey": "dereverb-qualified",
                },
            )
            assert ok_res.status_code == 202, ok_res.text
            assert submitted[-1]["profile"] == "dereverb"
            assert submitted[-1]["mode"] == "natural"
            assert submitted[-1]["strength"] == pytest.approx(0.85)
            assert submitted[-1]["output_path"].name.endswith("_dereverb.wav")

            # Blocked fail-closed when dereverb capability is unavailable
            dereverb_ready = False
            blk_res = client.post(
                "/api/v1/jobs",
                headers=HEADERS,
                json={
                    "schemaVersion": 1,
                    "sourceIds": [source_id],
                    "strategy": {
                        "kind": "manual",
                        "route": "dereverb",
                    },
                    "executionPolicy": "offline_only",
                    "conflictPolicy": "unique",
                    "idempotencyKey": "dereverb-blocked",
                },
            )
            assert blk_res.status_code == 503
            assert blk_res.json()["error"] == "capability_blocked"
    finally:
        manager.shutdown()
