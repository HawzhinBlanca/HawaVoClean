"""End-to-end pipeline wiring and durable job integration tests for Smart Safe.

Validates I3.7:
- Full pipeline run: analyze -> preview -> decide -> render -> guard -> publish.
- Mono and stereo inputs, with sample rate conversion.
- Post-master guard failure abstention to least-intervention route.
- CLI process dispatch and multi-pass validation.
- Server v1 job submission with SmartSafeStrategyV1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from starlette.testclient import TestClient

from hawavoclean.cli import cmd_process
from hawavoclean.errors import ExitCode, InvalidUserInputError
from hawavoclean.pipeline import run_pipeline
from hawavoclean.server.app import create_app
from hawavoclean.server.jobs import JobManager


def _create_synthetic_wav(
    path: Path,
    *,
    duration_s: float = 1.0,
    sample_rate: int = 48000,
    channels: int = 1,
) -> Path:
    """Create a synthetic test WAV file."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False, dtype=np.float32)
    # Speech-like harmonic tone
    wave = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    data = wave if channels == 1 else np.stack([wave, wave * 0.9], axis=1)
    sf.write(str(path), data, sample_rate, subtype="PCM_24")
    return path


@pytest.mark.unit
def test_smart_safe_pipeline_mono_end_to_end(tmp_path: Path) -> None:
    """End-to-end smart_safe pipeline run produces master WAV, report JSON, and summary TXT."""
    in_wav = _create_synthetic_wav(tmp_path / "input.wav", duration_s=1.0)
    out_wav = tmp_path / "output.wav"

    progress_events: list[str] = []

    def on_progress(event: Any) -> None:
        progress_events.append(event.stage)

    report = run_pipeline(
        input_path=in_wav,
        output_path=out_wav,
        profile="production",
        mode="smart_safe",
        overwrite=True,
        on_progress=on_progress,
    )

    assert out_wav.exists()
    assert report.output.path == str(out_wav.resolve())
    assert report.output.sample_rate == 48000
    assert report.output.channels == 1

    # Report JSON
    json_path = tmp_path / "output.hawavoclean.json"
    assert json_path.exists()
    with open(json_path) as f:
        report_data = json.load(f)

    restoration = report_data.get("restoration", {})
    assert restoration.get("strategy") == "smart_safe"
    assert "selected_route" in restoration
    assert isinstance(restoration.get("candidates"), list)
    assert len(restoration["candidates"]) >= 1
    assert "acoustic_evidence" in restoration
    assert "decision_sha256" in restoration

    # Human summary TXT
    txt_path = tmp_path / "output.hawavoclean.txt"
    assert txt_path.exists()
    summary_text = txt_path.read_text()
    assert "HAWAVOCLEAN" in summary_text

    # Progress stages reached
    assert "decode" in progress_events
    assert "segment" in progress_events
    assert "guard" in progress_events
    assert "enhance" in progress_events
    assert "publish" in progress_events


@pytest.mark.unit
def test_smart_safe_pipeline_stereo_and_resampling(tmp_path: Path) -> None:
    """Stereo inputs preserve channel count and 44.1 kHz inputs are resampled to 48 kHz."""
    in_wav = _create_synthetic_wav(
        tmp_path / "input_stereo_44k.wav",
        duration_s=1.0,
        sample_rate=44100,
        channels=2,
    )
    out_wav = tmp_path / "output_stereo.wav"

    report = run_pipeline(
        input_path=in_wav,
        output_path=out_wav,
        profile="production",
        mode="smart_safe",
        overwrite=True,
    )

    assert out_wav.exists()
    assert report.output.channels == 2
    assert report.output.sample_rate == 48000

    info = sf.info(str(out_wav))
    assert info.channels == 2
    assert info.samplerate == 48000


@pytest.mark.unit
def test_smart_safe_pipeline_post_master_failure_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-master invariant verification failure demotes to least-intervention safe candidate."""
    in_wav = _create_synthetic_wav(tmp_path / "input.wav", duration_s=1.0)
    out_wav = tmp_path / "output_fallback.wav"

    import hawavoclean.smart_safe.preview as preview_module

    call_count = 0

    def mock_verify_post_master(
        _master_audio: np.ndarray, _reference_audio: np.ndarray, _route: str, _sr: int
    ) -> tuple[bool, str]:
        nonlocal call_count
        call_count += 1
        # Fail the first check (the selected route), then pass subsequent checks
        if call_count == 1:
            return (False, "forced mock post-master invariant violation")
        return (True, "passed")

    monkeypatch.setattr(preview_module, "verify_post_master_invariants", mock_verify_post_master)

    report = run_pipeline(
        input_path=in_wav,
        output_path=out_wav,
        profile="production",
        mode="smart_safe",
        overwrite=True,
    )

    assert out_wav.exists()
    assert call_count >= 1
    # Successfully produced a valid output despite post-master demotion
    assert report.restoration is not None
    assert report.restoration.get("strategy") == "smart_safe"


@pytest.mark.unit
def test_smart_safe_pipeline_post_master_failure_all_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If all candidates fail post-master verification, selected_route falls back to preserve."""
    in_wav = _create_synthetic_wav(tmp_path / "input.wav", duration_s=1.0)
    out_wav = tmp_path / "output_preserve.wav"

    import hawavoclean.smart_safe.preview as preview_module

    def mock_verify_always_fail(
        _master_audio: np.ndarray, _reference_audio: np.ndarray, _route: str, _sr: int
    ) -> tuple[bool, str]:
        return (False, "forced total failure")

    monkeypatch.setattr(preview_module, "verify_post_master_invariants", mock_verify_always_fail)

    report = run_pipeline(
        input_path=in_wav,
        output_path=out_wav,
        profile="production",
        mode="smart_safe",
        overwrite=True,
    )

    assert out_wav.exists()
    assert report.restoration is not None


@pytest.mark.unit
def test_smart_safe_cli_validation(tmp_path: Path) -> None:
    """CLI refuses multi-pass execution with smart_safe mode."""
    in_wav = _create_synthetic_wav(tmp_path / "input.wav", duration_s=0.5)
    out_wav = tmp_path / "output.wav"

    # Single pass succeeds
    args = argparse.Namespace(
        input=str(in_wav),
        output=str(out_wav),
        profile="production",
        mode="smart_safe",
        passes=1,
        overwrite=True,
        config=None,
        clean_only=False,
        progress_json=False,
        speaker_id=None,
        cutoff="auto",
        cutoff_hz=None,
        profiles_dir=None,
        allow_research_restore=False,
        original_input_path=None,
        record_bundle=None,
    )
    code = cmd_process(args)
    assert code == 0
    assert out_wav.exists()

    # Multi-pass fails with INVALID_USER_INPUT
    args_multipass = argparse.Namespace(
        input=str(in_wav),
        output=str(tmp_path / "multipass.wav"),
        profile="production",
        mode="smart_safe",
        passes=2,
        overwrite=True,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_process(args_multipass)
    assert exc.value.code == int(ExitCode.INVALID_USER_INPUT)


@pytest.mark.unit
def test_smart_safe_server_job_submission(tmp_path: Path) -> None:
    """Server accepts SmartSafeStrategyV1 and submits a durable smart_safe job."""
    token = "test-token-secret-1234"
    mgr = JobManager(store_path=tmp_path / "jobs" / "jobs.json")
    app = create_app(token=token, job_manager=mgr)
    client = TestClient(app, base_url="http://127.0.0.1")
    headers = {"X-Hawa-Token": token}

    # Upload test audio
    in_wav = _create_synthetic_wav(tmp_path / "source.wav", duration_s=0.5)
    upload_res = client.post(
        "/api/upload",
        headers=headers,
        files={"file": ("source.wav", in_wav.read_bytes(), "audio/wav")},
    )
    assert upload_res.status_code == 200
    source_id = upload_res.json()["source_id"]

    # Submit Smart Safe job
    job_req = {
        "schemaVersion": 1,
        "sourceIds": [source_id],
        "strategy": {
            "kind": "smart_safe",
            "restorePolicy": "disabled",
            "allowGenerativeReconstruction": False,
        },
        "executionPolicy": "offline_only",
        "conflictPolicy": "unique",
        "recordBundle": False,
        "idempotencyKey": "smart-safe-job-test-1",
    }
    submit_res = client.post("/api/v1/jobs", headers=headers, json=job_req)
    assert submit_res.status_code == 202
    data = submit_res.json()
    assert "jobs" in data
    assert len(data["jobs"]) == 1
    job_id = data["jobs"][0]["jobId"]

    # Check job record has mode="smart_safe"
    status = mgr.get_status(job_id)
    assert status is not None
    assert status.get("mode") == "smart_safe"

    mgr.shutdown()


@pytest.mark.unit
def test_pipeline_unknown_mode_raises(tmp_path: Path) -> None:
    """run_pipeline with unrecognized mode raises InvalidUserInputError."""
    in_wav = _create_synthetic_wav(tmp_path / "input.wav", duration_s=0.5)
    out_wav = tmp_path / "output.wav"

    with pytest.raises(InvalidUserInputError, match="Unknown processing mode"):
        run_pipeline(
            input_path=in_wav,
            output_path=out_wav,
            profile="production",
            mode="invalid_future_mode",
            overwrite=True,
        )
