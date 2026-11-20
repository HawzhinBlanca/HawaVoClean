"""Regression tests for real-world audio-only MP4, long-clip NISQA v2, SmartSafe post-master
parity, GPU worker concurrency capping, and bilateral mouth de-click tuning.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.delivery import export_delivery_artifact
from hawavoclean.audio.probe import probe_audio
from hawavoclean.finishing.pro_repair import apply_mouth_declick
from hawavoclean.guard.perceptual_mos import PerceptualQualityScore, PerceptualQualityScorer
from hawavoclean.pipeline import _natural_worker_limit, run_pipeline
from hawavoclean.restoration.highband_events import HighBandEventResult
from hawavoclean.smart_safe.preview import verify_post_master_invariants

SR = 48000


def _fake_ffprobe_stdout(streams: list[dict[str, Any]], duration: str = "2.0") -> bytes:
    payload = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": duration},
        "streams": streams,
    }
    return json.dumps(payload).encode("utf-8")


@pytest.mark.unit
def test_probe_audio_has_video_distinguishes_audio_only_cover_art_and_real_video(
    tmp_path: Path,
) -> None:
    dummy = tmp_path / "Flute 09.m4a.mp4"
    dummy.write_bytes(b"\x00" * 64)
    audio_stream = {
        "index": 0,
        "codec_type": "audio",
        "codec_name": "aac",
        "sample_rate": "48000",
        "channels": 1,
        "duration": "2.0",
        "nb_samples": "96000",
    }

    # 1. Audio-only .mp4 voice memo -> has_video is False
    with (
        patch("hawavoclean.audio.probe.ffprobe_bin_path", return_value="/usr/bin/ffprobe"),
        patch(
            "hawavoclean.audio.probe._run_bounded_capture",
            return_value=subprocess.CompletedProcess(
                ["ffprobe"], 0, stdout=_fake_ffprobe_stdout([audio_stream]), stderr=b""
            ),
        ),
    ):
        res_audio_only = probe_audio(dummy)
    assert res_audio_only.has_video is False

    # 2. Audio + embedded album cover art (attached_pic=1) -> has_video is False
    cover_stream = {
        "index": 1,
        "codec_type": "video",
        "codec_name": "mjpeg",
        "disposition": {"attached_pic": 1},
    }
    with (
        patch("hawavoclean.audio.probe.ffprobe_bin_path", return_value="/usr/bin/ffprobe"),
        patch(
            "hawavoclean.audio.probe._run_bounded_capture",
            return_value=subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                stdout=_fake_ffprobe_stdout([audio_stream, cover_stream]),
                stderr=b"",
            ),
        ),
    ):
        res_cover = probe_audio(dummy)
    assert res_cover.has_video is False

    # 3. Real video stream (attached_pic=0) -> has_video is True
    video_stream = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "disposition": {"attached_pic": 0},
    }
    audio_stream_1 = {**audio_stream, "index": 1}
    with (
        patch("hawavoclean.audio.probe.ffprobe_bin_path", return_value="/usr/bin/ffprobe"),
        patch(
            "hawavoclean.audio.probe._run_bounded_capture",
            return_value=subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                stdout=_fake_ffprobe_stdout([video_stream, audio_stream_1]),
                stderr=b"",
            ),
        ),
    ):
        res_video = probe_audio(dummy)
    assert res_video.has_video is True


@pytest.mark.unit
def test_video_mux_rejects_audio_only_mp4_cleanly(tmp_path: Path) -> None:
    master = tmp_path / "master.wav"
    audio_only = tmp_path / "voice.m4a.mp4"
    t = np.linspace(0.0, 0.5, SR // 2, endpoint=False, dtype=np.float32)
    sig = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(master), sig, SR, subtype="PCM_24")
    sf.write(str(audio_only), sig, SR, format="WAV", subtype="PCM_24")

    with (
        patch("hawavoclean.audio.delivery.ffmpeg_bin_path", return_value="/usr/bin/ffmpeg"),
        pytest.raises(ValueError, match="no video stream"),
    ):
        export_delivery_artifact(
            master,
            tmp_path / "out.mp4",
            fmt="video_mux",
            source_media_path=audio_only,
        )


@pytest.mark.unit
def test_perceptual_scorer_windows_long_clips_without_falling_back_to_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clips >18 s window into bounded <=8 active 12 s chunks instead of crashing NISQA v2."""
    scorer = PerceptualQualityScorer(backend="nisqa_v2", allow_fallback=False)
    monkeypatch.setattr("hawavoclean.guard.perceptual_mos._is_nisqa_available", lambda: True)

    seen_lengths: list[int] = []

    def _fake_single(
        _self: PerceptualQualityScorer,
        chunk: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> list[float]:
        seen_lengths.append(len(chunk))
        assert len(chunk) <= int(12.0 * sample_rate)
        return [4.1, 4.0, 4.3, 3.9, 4.2]

    monkeypatch.setattr(PerceptualQualityScorer, "_score_nisqa_single", _fake_single)

    # 120 s clip has 10 raw 12 s segments -> subsampled evenly to 8 max windows
    t = np.arange(int(120.0 * 16000), dtype=np.float32) / 16000.0
    long_audio = (0.15 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
    res = scorer.score(long_audio, 16000)

    assert res.backend == "nisqa_v2"
    assert res.calibrated is True
    assert res.mos == pytest.approx(4.1, abs=1e-3)
    assert len(seen_lengths) == 8


@pytest.mark.unit
def test_post_master_invariants_calibrated_nisqa_parity_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-master artifact guard uses calibrated perceptual parity for impulse-only deltas."""
    t = np.linspace(0.0, 1.0, SR, endpoint=False, dtype=np.float32)
    speech = (
        0.2 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 660.0 * t)
    ).astype(np.float32)
    impulse_only = HighBandEventResult(
        speech_window_leakage=0.0,
        spurious_burst_count=0,
        hf_envelope_divergence=0.0,
        impulse_discontinuity_ratio=35.0,
        passes_event_check=False,
    )
    monkeypatch.setattr(
        "hawavoclean.smart_safe.preview.HighBandEventDetector.evaluate",
        lambda *_a, **_k: impulse_only,
    )

    class _StubScorer(PerceptualQualityScorer):
        def __init__(self, ref_dis: float, cand_dis: float) -> None:
            super().__init__(backend="acoustic_proxy")
            self._scores = [
                PerceptualQualityScore(2.9, 2.5, ref_dis, 3.2, 3.8, 0.92, "nisqa_v2", True),
                PerceptualQualityScore(4.3, 4.2, cand_dis, 4.0, 4.3, 0.92, "nisqa_v2", True),
            ]

        def score(self, _audio: np.ndarray[Any, Any], _sample_rate: int) -> PerceptualQualityScore:
            return self._scores.pop(0)

    # Improved discontinuity passes parity check
    ok_good, _ = verify_post_master_invariants(
        speech,
        speech,
        "studio",
        SR,
        perceptual_scorer=_StubScorer(ref_dis=4.19, cand_dis=4.44),
    )
    assert ok_good is True

    # Discontinuity collapse still fails closed
    ok_bad, reason_bad = verify_post_master_invariants(
        speech,
        speech,
        "studio",
        SR,
        perceptual_scorer=_StubScorer(ref_dis=4.19, cand_dis=2.60),
    )
    assert ok_bad is False
    assert "artifact guard failed" in reason_bad


@pytest.mark.unit
def test_smart_safe_pipeline_records_post_master_abstention_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = np.linspace(0.0, 0.5, SR // 2, endpoint=False, dtype=np.float32)
    sig = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    sf.write(str(in_wav), sig, SR, subtype="PCM_24")

    import hawavoclean.smart_safe.preview as preview_module

    monkeypatch.setattr(
        preview_module,
        "verify_post_master_invariants",
        lambda *_a, **_k: (False, "synthetic post-master failure"),
    )
    report = run_pipeline(in_wav, out_wav, profile="production", mode="smart_safe", overwrite=True)
    assert isinstance(report.restoration, dict)
    assert report.restoration["abstained"] is True
    assert report.restoration["post_master_verified"] is False
    assert report.restoration["post_master_failure_reason"] == "synthetic post-master failure"


@pytest.mark.unit
def test_natural_worker_limit_caps_single_gpu_cores_unless_overridden() -> None:
    assert _natural_worker_limit(12, streaming=False, device="mps", explicit_override=False) == 1
    assert _natural_worker_limit(12, streaming=False, device="cuda", explicit_override=False) == 1
    assert _natural_worker_limit(12, streaming=False, device="mps", explicit_override=True) == 12
    assert _natural_worker_limit(12, streaming=False, device="cpu", explicit_override=False) == 12
    assert _natural_worker_limit(12, streaming=True, device="cpu", explicit_override=False) == 2


@pytest.mark.unit
def test_mouth_declick_preserves_consonant_onset_flanked_by_speech() -> None:
    """Bilateral isolation prevents false triggers at sharp consonant onsets."""
    rng = np.random.default_rng(123)
    n = SR
    sig = np.zeros(n, dtype=np.float32)
    # Silence for 0.3 s, then a 1.5 ms plosive onset transient (AR residual
    # clearly above abs_floor=0.0198) immediately followed by 60 ms of bounded
    # fricative/consonant speech energy whose 95th-percentile flank residual
    # protects the onset ONLY when the bilateral flank check is active.
    onset = int(0.3 * SR)
    tail_len = int(0.060 * SR)
    # Sharp 3-sample plosive burst (AR residual ~0.065 > local threshold, yet
    # < flank_p95 * 5.0 ~ 0.081 so ONLY the bilateral flank check protects it).
    burst = np.array([0.068, -0.062, 0.038], dtype=np.float32)
    sig[onset : onset + len(burst)] = burst
    tail = rng.uniform(-0.017, 0.017, size=tail_len).astype(np.float32)
    sig[onset + len(burst) : onset + len(burst) + tail_len] = tail

    out, count = apply_mouth_declick(sig, SR, sensitivity=0.5)
    assert count == 0
    assert np.allclose(out, sig, atol=1e-6)


@pytest.mark.unit
def test_pipeline_stereo_mode_true_stereo_ms_and_pro_repair_finishing_overrides(
    tmp_path: Path,
) -> None:
    """stereo_mode='true_stereo_ms' and finishing toggles wire cleanly through run_pipeline."""
    t = np.arange(SR, dtype=np.float64) / SR
    f0 = 150.0 + 15.0 * np.sin(2.0 * np.pi * 1.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / SR
    voice = (0.25 * np.sin(phase) + 0.12 * np.sin(2.0 * phase)).astype(np.float32)
    # Distinct L/R imaging (ambiguous stereo if auto-classified, valid under true_stereo_ms)
    side = (0.04 * np.sin(2.0 * np.pi * 510.0 * t)).astype(np.float32)
    stereo = np.stack([voice + side, voice - side], axis=1)
    in_wav = tmp_path / "stereo_ms.wav"
    out_wav = tmp_path / "stereo_ms_out.wav"
    sf.write(str(in_wav), stereo, SR, subtype="PCM_24")

    report = run_pipeline(
        in_wav,
        out_wav,
        profile="production",
        stereo_mode="true_stereo_ms",
        pro_repair={"deess": True, "tonal_restoration": False, "level_rider": False},
        overwrite=True,
    )
    assert out_wav.is_file()
    assert report.output.channels == 2
    assert report.output.samples == SR
    # Stereo imaging is preserved (L and R are not collapsed to identical dual-mono)
    out_data, _ = sf.read(str(out_wav), dtype="float32", always_2d=True)
    assert float(np.max(np.abs(out_data[:, 0] - out_data[:, 1]))) > 1e-3

    # Unknown stereo_mode fails closed with InvalidUserInputError
    from hawavoclean.errors import InvalidUserInputError

    with pytest.raises(InvalidUserInputError, match="Unknown stereo_mode"):
        run_pipeline(
            in_wav,
            tmp_path / "bad.wav",
            profile="production",
            stereo_mode="invalid",
            overwrite=True,
        )
