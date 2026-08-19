"""Generate adversarial real-world input shapes for the pipeline."""

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

OUT = Path(os.environ.get("FUZZ_OUT_DIR", Path(__file__).parent / "inputs"))
OUT.mkdir(parents=True, exist_ok=True)
SR = 48000
rng = np.random.default_rng(42)


def speechlike(
    sec: float, sr: int = SR, f0: float = 160, amp: float = 0.3
) -> np.ndarray[Any, np.dtype[np.float32]]:
    t = np.arange(int(sec * sr)) / sr
    x = np.zeros_like(t)
    for h in range(1, 25):
        x += (amp / h) * np.sin(2 * np.pi * f0 * h * t)
    gate = ((t % 1.4) < 0.8).astype(float)
    out = x * gate * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))
    return np.asarray(out, dtype=np.float32)


cases: dict[str, tuple[np.ndarray[Any, Any], int]] = {}
# 1. Degenerate lengths
cases["empty_0samples"] = (np.zeros(0, np.float32), SR)
cases["one_sample"] = (np.array([0.5], np.float32), SR)
cases["ten_ms"] = (speechlike(0.01), SR)
cases["half_second"] = (speechlike(0.5), SR)
cases["exactly_one_unit_30s_continuous"] = (speechlike(30.0, amp=0.3) * np.float32(1.0), SR)
# 2. Extreme levels
cases["digital_silence_10s"] = (np.zeros(SR * 10, np.float32), SR)
cases["near_silence_minus90db"] = ((speechlike(8) * 10 ** (-90 / 20)).astype(np.float32), SR)
cases["full_scale_clipped"] = (np.clip(speechlike(8) * 8, -1, 1).astype(np.float32), SR)
cases["float_over_unity_2x"] = (
    (speechlike(8) * 2.0).astype(np.float32),
    SR,
)  # float32 WAV can exceed 1.0
cases["dc_offset_0p4"] = ((speechlike(8) + 0.4).astype(np.float32), SR)
cases["single_click_in_silence"] = (
    np.concatenate(
        [np.zeros(SR * 4, np.float32), np.array([0.9], np.float32), np.zeros(SR * 4, np.float32)]
    ),
    SR,
)
# 3. Content types
cases["pure_sine_1khz"] = (
    (0.5 * np.sin(2 * np.pi * 1000 * np.arange(SR * 8) / SR)).astype(np.float32),
    SR,
)
cases["white_noise_only"] = ((0.2 * rng.standard_normal(SR * 8)).astype(np.float32), SR)
cases["nyquist_tone_24k"] = ((0.3 * np.cos(np.pi * np.arange(SR * 5))).astype(np.float32), SR)
cases["subsonic_5hz_rumble"] = (
    (0.8 * np.sin(2 * np.pi * 5 * np.arange(SR * 8) / SR) + speechlike(8) * 0.3).astype(np.float32),
    SR,
)
cases["nan_in_file"] = (
    np.where(np.arange(SR * 5) == SR * 2, np.nan, speechlike(5)).astype(np.float32),
    SR,
)
cases["inf_in_file"] = (
    np.where(np.arange(SR * 5) == SR * 2, np.inf, speechlike(5)).astype(np.float32),
    SR,
)
# 4. Sample rates
for sr in (8000, 11025, 16000, 22050, 44100):
    cases[f"rate_{sr}"] = (speechlike(6, sr=sr), sr)
cases["rate_96k_must_reject"] = (speechlike(3, sr=96000), 96000)
# 5. Channel layouts
s = speechlike(8)
cases["stereo_identical"] = (np.stack([s, s], 1), SR)
cases["stereo_one_silent"] = (np.stack([s, np.zeros_like(s)], 1), SR)
cases["stereo_inverted_polarity"] = (np.stack([s, -s], 1), SR)
cases["stereo_split_speakers"] = (
    np.stack([speechlike(8, f0=120), np.roll(speechlike(8, f0=220), SR)], 1),
    SR,
)
cases["stereo_tiny_level_diff"] = (
    np.stack([s, s * 0.985], 1),
    SR,
)  # right at the dual-mono threshold
cases["six_channel"] = (np.stack([s] * 6, 1), SR)
# 6. Long
cases["long_8min"] = (speechlike(480.0), SR)

for name, (data, sr) in cases.items():
    sf.write(str(OUT / f"{name}.wav"), data, sr, subtype="FLOAT")
# 7. Format/container oddities via ffmpeg
base = OUT / "half_second.wav"
src8 = OUT / "rate_16000.wav"
for name, args in {
    "mp3_lossy": ["-c:a", "libmp3lame", "-b:a", "64k"],
    "aac_m4a": ["-c:a", "aac", "-b:a", "96k"],
    "pcm_24bit": ["-c:a", "pcm_s24le"],
    "pcm_8bit_u8": ["-c:a", "pcm_u8"],
    "flac": ["-c:a", "flac"],
    "ogg_vorbis": ["-c:a", "libvorbis"],
}.items():
    ext = {
        "mp3_lossy": "mp3",
        "aac_m4a": "m4a",
        "pcm_24bit": "wav",
        "pcm_8bit_u8": "wav",
        "flac": "flac",
        "ogg_vorbis": "ogg",
    }[name]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(OUT / "rate_44100.wav"),
            *args,
            str(OUT / f"{name}.{ext}"),
        ],
        check=False,
    )
# 8. Broken / hostile files
(OUT / "zero_byte.wav").write_bytes(b"")
(OUT / "truncated_header.wav").write_bytes((OUT / "half_second.wav").read_bytes()[:20])
(OUT / "truncated_data.wav").write_bytes((OUT / "rate_44100.wav").read_bytes()[:-40000])
(OUT / "text_not_audio.wav").write_bytes(b"this is not audio\n" * 100)
(OUT / "name with spaces & quote's.wav").write_bytes((OUT / "half_second.wav").read_bytes())
# video container with audio
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=4",
        "-i",
        str(OUT / "rate_44100.wav"),
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(OUT / "video_with_audio.mp4"),
    ],
    check=False,
)
# video with NO audio stream
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=2",
        "-c:v",
        "libx264",
        str(OUT / "video_no_audio.mp4"),
    ],
    check=False,
)
print(f"{len(list(OUT.iterdir()))} inputs generated")
