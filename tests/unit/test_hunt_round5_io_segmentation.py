"""Bugs found by adversarial review of audio I/O and segmentation (round 5)."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.channels import classify_channels
from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.encode import encode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioBuffer, ChannelMode
from hawavoclean.config import SegmentationConfig
from hawavoclean.errors import InvalidUserInputError
from hawavoclean.segmentation.utterances import build_speech_units
from hawavoclean.segmentation.vad import detect_speech_energy

SR = 48000
ffmpeg = shutil.which("ffmpeg") or ""


# 1. ffmpeg must never read the terminal ------------------------------------------
def test_decode_command_isolates_stdin(monkeypatch: Any, tmp_path: Path) -> None:
    """A keypress ('q') during decode aborted ffmpeg with exit 0 and the
    pipeline published a half-length master as success. The decode command
    must pass -nostdin AND stdin=DEVNULL."""
    import subprocess as sp

    wav = tmp_path / "t.wav"
    sf.write(str(wav), np.zeros(SR, np.float32), SR)
    media = probe_audio(wav)
    seen: dict[str, Any] = {}
    real_run = sp.run

    def spy(cmd: list[str], **kw: Any) -> Any:
        seen["cmd"] = cmd
        seen["stdin"] = kw.get("stdin")
        return real_run(cmd, **kw)

    monkeypatch.setattr(sp, "run", spy)
    decode_audio(media, timeout_s=30)
    assert "-nostdin" in seen["cmd"], "ffmpeg invoked without -nostdin"
    assert seen["stdin"] is sp.DEVNULL, "ffmpeg inherits the parent's stdin"


# 2. streamed containers with no duration ----------------------------------------
@pytest.mark.skipif(not ffmpeg, reason="ffmpeg required")
def test_streamed_webm_without_duration_is_accepted(tmp_path: Path) -> None:
    webm = tmp_path / "live.webm"
    with open(webm, "wb") as f:
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=2",
                "-c:a",
                "libopus",
                "-f",
                "webm",
                "-",
            ],
            stdout=f,
            check=True,
        )
    media = probe_audio(webm)
    buf = decode_audio(media, timeout_s=60)
    assert buf.samples > SR  # ~2 s decoded


# 3. absurd sample rates rejected cleanly ---------------------------------------------
@pytest.mark.parametrize("sr", [1, 50, 100, 4000])
def test_unsupported_low_sample_rate_rejected(sr: int, tmp_path: Path) -> None:
    p = tmp_path / f"sr{sr}.wav"
    sf.write(
        str(p),
        np.random.default_rng(0).standard_normal(max(150, sr)).astype(np.float32) * 0.1,
        sr,
        subtype="PCM_16",
    )
    with pytest.raises(InvalidUserInputError, match="[Ss]ample rate"):
        probe_audio(p)


# 4. trailing silence must not be glued onto a speech unit -----------------------------------
def test_long_silence_gap_does_not_inflate_speech_unit() -> None:
    t = np.arange(5 * SR) / SR
    sp = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    w = np.concatenate(
        [sp, (1e-4 * np.random.default_rng(0).standard_normal(600 * SR)).astype(np.float32), sp]
    )
    cfg = SegmentationConfig()
    units = build_speech_units(w, SR, 0, cfg)
    speech = [u for u in units if u.is_speech]
    longest = max(u.core_length_samples for u in speech) / SR
    assert longest <= cfg.hard_max_group_s, (
        f"a {longest:.0f}s 'speech' unit (hard max {cfg.hard_max_group_s}s)"
    )


# 5. declared channel mode must match the file -------------------------------------------------
def test_declared_mode_validated_against_channel_count() -> None:
    stereo = AudioBuffer(
        np.random.default_rng(0).standard_normal((2, SR)).astype(np.float32) * 0.1, SR
    )
    mono = AudioBuffer(np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1, SR)
    with pytest.raises(InvalidUserInputError):
        classify_channels(stereo, declared_mode="mono")
    with pytest.raises(InvalidUserInputError):
        classify_channels(mono, declared_mode="dual_mono_same")
    with pytest.raises(InvalidUserInputError):
        classify_channels(mono, declared_mode="split_speakers")


# 6. DC offset must not turn pauses into speech -------------------------------------------------
def test_dc_offset_does_not_make_pauses_speech() -> None:
    t = np.arange(SR) / SR
    rng = np.random.default_rng(0)
    word = (0.1 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 1.3 * t))).astype(
        np.float32
    )
    pause = (1e-4 * rng.standard_normal(2 * SR)).astype(np.float32)
    clean = np.concatenate([word, pause] * 10)
    dc = clean + np.float32(0.003)
    iv_clean = detect_speech_energy(clean, SR)
    iv_dc = detect_speech_energy(dc, SR)
    speech_clean = sum(i.length_samples for i in iv_clean) / SR
    speech_dc = sum(i.length_samples for i in iv_dc) / SR
    assert abs(speech_dc - speech_clean) < 2.0, (
        f"DC offset changed detected speech from {speech_clean:.1f}s to {speech_dc:.1f}s"
    )


# 7. one transient must not hide quiet speech ---------------------------------------------------------
def test_single_transient_does_not_suppress_quiet_speech() -> None:
    t = np.arange(10 * SR) / SR
    s = (
        (np.sin(2 * np.pi * 3 * t) > 0)
        * np.sin(2 * np.pi * 180 * t)
        * (1 + 0.5 * np.sin(2 * np.pi * 1.3 * t))
    ).astype(np.float32)
    x = np.asarray(s * 10 ** (-40 / 20) * np.sqrt(2), dtype=np.float32)
    baseline = sum(i.length_samples for i in detect_speech_energy(x, SR)) / SR
    x[5 * SR : 5 * SR + 960] = 0.9  # one 20 ms burst
    with_burst = sum(i.length_samples for i in detect_speech_energy(x, SR)) / SR
    assert with_burst > 0.6 * baseline, (
        f"one transient suppressed speech detection: {baseline:.1f}s -> {with_burst:.1f}s"
    )


# 8. dual-mono output stays bit-identical L/R -------------------------------------------------------------
def test_dual_mono_encode_is_bit_identical(tmp_path: Path) -> None:
    s = (np.random.default_rng(1).standard_normal(SR) * 0.1).astype(np.float32)
    buf = AudioBuffer(np.stack([s, s]), SR, channel_mode=ChannelMode.DUAL_MONO_SAME)
    out = encode_audio(buf, tmp_path / "d.wav", "pcm24", dither=True, seed_context="job1")
    y, _ = sf.read(str(out), dtype="int32", always_2d=True)
    assert np.array_equal(y[:, 0], y[:, 1]), "dual-mono L/R differ at the LSB after dither"
