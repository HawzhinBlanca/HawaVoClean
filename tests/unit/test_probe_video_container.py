"""Found by fuzzing: an MP4 whose FIRST stream is video was rejected with
'rate=0, channels=0' because probe_audio read streams[0]. Phone recordings
named *.m4a.mp4 and any video file with a soundtrack hit this."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.errors import InvalidUserInputError

ffmpeg = shutil.which("ffmpeg") or ""


@pytest.mark.skipif(not ffmpeg, reason="ffmpeg required to build the container")
def test_video_first_container_probes_the_audio_stream(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    sr = 44100
    t = np.arange(sr * 3) / sr
    sf.write(str(wav), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=3",
            "-i",
            str(wav),
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(mp4),
        ],
        check=True,
    )
    media = probe_audio(mp4)
    assert media.channels == 1
    assert media.sample_rate == sr
    assert media.samples > sr * 2

    buf = decode_audio(media, timeout_s=60)
    # For lossy codecs ffprobe's duration is an estimate (AAC frame padding);
    # the pipeline re-syncs to the decoded length. Decoded length must be
    # close to, and never shorter than, the probed estimate.
    assert buf.samples >= media.samples
    assert buf.samples - media.samples < sr * 0.1


@pytest.mark.skipif(not ffmpeg, reason="ffmpeg required to build the container")
def test_container_with_no_audio_stream_is_rejected_clearly(tmp_path: Path) -> None:
    mp4 = tmp_path / "silent.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=2",
            "-c:v",
            "libx264",
            str(mp4),
        ],
        check=True,
    )
    with pytest.raises(InvalidUserInputError, match="[Nn]o audio stream"):
        probe_audio(mp4)
