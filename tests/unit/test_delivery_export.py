"""Unit tests for broadcast delivery loudness, BWF/iXML metadata, and multi-format export."""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.delivery import (
    DELIVERY_PRESETS,
    DeliveryPreset,
    ExportFormat,
    apply_delivery_loudness,
    compute_loudness_range_lu,
    export_delivery_artifact,
    extract_wav_metadata_chunks,
    inject_wav_metadata_chunks,
)
from hawavoclean.paths import ffmpeg_bin_path, ffprobe_bin_path

SR = 48000


def _make_sine(
    duration_s: float,
    freq_hz: float = 1000.0,
    amp: float = 0.1,
    channels: int = 1,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    n = int(round(duration_s * SR))
    t = np.arange(n, dtype=np.float64) / SR
    mono = (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    if channels == 1:
        return mono
    return np.ascontiguousarray(np.tile(mono[np.newaxis, :], (channels, 1)), dtype=np.float32)


def _sample_bext_payload() -> bytes:
    """Build a valid 602-byte Broadcast Wave Extension (bext v1) chunk payload."""
    buf = bytearray(602)
    desc = b"HawaVoClean Broadcast Master"
    orig = b"HawaVoClean Studio"
    date = b"2026-11-19"
    time_str = b"19:30:00"
    buf[0 : len(desc)] = desc
    buf[256 : 256 + len(orig)] = orig
    buf[320 : 320 + len(date)] = date
    buf[330 : 330 + len(time_str)] = time_str
    # TimeReference = 1 hour at 48 kHz = 172,800,000 samples (uint64 LE at offset 338).
    struct.pack_into("<Q", buf, 338, 172_800_000)
    # Version = 1 (uint16 LE at offset 346).
    struct.pack_into("<H", buf, 346, 1)
    return bytes(buf)


def _sample_ixml_payload() -> bytes:
    """Build an odd-length iXML payload (tests RIFF word-alignment padding)."""
    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b"<BWFXML><IXML_VERSION>1.5</IXML_VERSION>"
        b"<SCENE>42A</SCENE><TAKE>3</TAKE>"
        b"<TRACK_LIST><TRACK_COUNT>1</TRACK_COUNT></TRACK_LIST>"
        b"</BWFXML>"
    )
    if len(xml) % 2 == 0:
        xml += b" "
    assert len(xml) % 2 == 1
    return xml


@pytest.mark.unit
def test_delivery_presets_registry() -> None:
    expected: set[DeliveryPreset] = {
        "match_input",
        "ebu_r128",
        "atsc_a85",
        "podcast",
        "streaming",
    }
    assert set(DELIVERY_PRESETS.keys()) == expected
    assert DELIVERY_PRESETS["match_input"].bypass_limiter is True
    assert DELIVERY_PRESETS["match_input"].target_lufs_stereo is None
    assert DELIVERY_PRESETS["ebu_r128"].target_lufs_stereo == -23.0
    assert DELIVERY_PRESETS["ebu_r128"].true_peak_ceiling_dbtp == -1.0
    assert DELIVERY_PRESETS["atsc_a85"].target_lufs_stereo == -24.0
    assert DELIVERY_PRESETS["atsc_a85"].true_peak_ceiling_dbtp == -2.0
    assert DELIVERY_PRESETS["podcast"].target_lufs_stereo == -16.0
    assert DELIVERY_PRESETS["podcast"].target_lufs_mono == -19.0
    assert DELIVERY_PRESETS["streaming"].target_lufs_stereo == -14.0


@pytest.mark.unit
def test_loudness_range_lu_steady_dynamic_short_and_silence() -> None:
    # Steady 6 s tone -> LRA ~ 0 LU.
    steady = _make_sine(6.0, freq_hz=440.0, amp=0.2)
    assert compute_loudness_range_lu(steady, SR) < 0.5

    # Alternating quiet (-28 dB relative) and loud sections across 12 s -> high LRA.
    seg_quiet = _make_sine(4.0, freq_hz=500.0, amp=0.05)
    seg_loud = _make_sine(4.0, freq_hz=500.0, amp=0.35)
    dynamic = np.concatenate([seg_quiet, seg_loud, seg_quiet])
    lra_dyn = compute_loudness_range_lu(dynamic, SR)
    assert 8.0 <= lra_dyn <= 18.0

    # Stereo (2, N) works identically to mono shape logic.
    stereo_dyn = np.stack([dynamic, dynamic], axis=0)
    assert compute_loudness_range_lu(stereo_dyn, SR) == pytest.approx(lra_dyn, abs=0.2)

    # Short clip (< 3 s) and silence return 0.0 LU.
    assert compute_loudness_range_lu(_make_sine(1.5), SR) == 0.0
    assert compute_loudness_range_lu(np.zeros(6 * SR, dtype=np.float32), SR) == 0.0


@pytest.mark.unit
def test_apply_delivery_loudness_presets_and_match_input() -> None:
    mono = _make_sine(2.0, freq_hz=1000.0, amp=0.1, channels=1)
    stereo = _make_sine(2.0, freq_hz=1000.0, amp=0.1, channels=2)

    # EBU R128 -> -23 LUFS, <= -1.0 dBTP.
    ebu_out, ebu_m = apply_delivery_loudness(stereo, SR, "ebu_r128")
    assert ebu_out.shape == stereo.shape
    assert ebu_m["preset"] == "ebu_r128"
    assert ebu_m["limiter_mode"] == "lookahead"
    assert float(ebu_m["integrated_lufs"]) == pytest.approx(-23.0, abs=0.5)
    assert float(ebu_m["true_peak_dbtp"]) <= -1.0 + 0.05

    # ATSC A/85 -> -24 LUFS, <= -2.0 dBTP.
    atsc_out, atsc_m = apply_delivery_loudness(stereo, SR, "atsc_a85")
    assert atsc_out.shape == stereo.shape
    assert float(atsc_m["integrated_lufs"]) == pytest.approx(-24.0, abs=0.5)
    assert float(atsc_m["true_peak_dbtp"]) <= -2.0 + 0.05

    # Podcast: mono hits -19 LUFS, stereo hits -16 LUFS.
    pod_mono_out, pod_mono_m = apply_delivery_loudness(mono, SR, "podcast")
    pod_st_out, pod_st_m = apply_delivery_loudness(stereo, SR, "podcast")
    assert pod_mono_out.ndim == 1
    assert pod_st_out.ndim == 2
    assert float(pod_mono_m["integrated_lufs"]) == pytest.approx(-19.0, abs=0.5)
    assert float(pod_st_m["integrated_lufs"]) == pytest.approx(-16.0, abs=0.5)

    # Streaming with loud peak spike enforces -1.0 dBTP ceiling via limiter.
    spiky = _make_sine(2.0, freq_hz=800.0, amp=0.08, channels=2)
    spiky[:, SR] = 0.95
    stream_out, stream_m = apply_delivery_loudness(spiky, SR, "streaming")
    assert stream_out.shape == spiky.shape
    assert float(stream_m["true_peak_dbtp"]) <= -1.0 + 0.05
    assert float(stream_m["limiter_max_gr_db"]) > 0.0

    # Match input (no input_lufs): exact bit-identical bypass.
    match_out, match_m = apply_delivery_loudness(mono, SR, "match_input")
    assert match_m["limiter_mode"] == "bypass"
    assert float(match_m["applied_gain_db"]) == 0.0
    assert np.array_equal(match_out, mono)

    # Match input with explicit input_lufs shifts gain linearly without limiting.
    matched_out, matched_m = apply_delivery_loudness(mono, SR, "match_input", input_lufs=-18.0)
    assert matched_m["limiter_mode"] == "bypass"
    assert float(matched_m["limiter_max_gr_db"]) == 0.0
    assert float(matched_m["integrated_lufs"]) == pytest.approx(-18.0, abs=0.5)
    assert matched_out.shape == mono.shape

    # Silence and empty inputs do not explode.
    sil = np.zeros((2, SR), dtype=np.float32)
    sil_out, sil_m = apply_delivery_loudness(sil, SR, "streaming")
    assert float(sil_m["applied_gain_db"]) == 0.0
    assert np.array_equal(sil_out, sil)

    empty = np.empty((1, 0), dtype=np.float32)
    empty_out, _ = apply_delivery_loudness(empty, SR, "ebu_r128")
    assert empty_out.size == 0


@pytest.mark.unit
def test_loudness_and_lra_fail_closed() -> None:
    bad = np.array([0.1, np.nan, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_loudness_range_lu(bad, SR)
    with pytest.raises(ValueError, match="NaN or Inf"):
        apply_delivery_loudness(bad, SR, "ebu_r128")
    with pytest.raises(ValueError, match="Unsupported delivery preset"):
        apply_delivery_loudness(
            np.zeros(100, dtype=np.float32),
            SR,
            cast(DeliveryPreset, "bogus"),
        )
    with pytest.raises(ValueError, match="input_lufs"):
        apply_delivery_loudness(
            np.zeros(100, dtype=np.float32),
            SR,
            "match_input",
            input_lufs=float("inf"),
        )


@pytest.mark.unit
def test_wav_metadata_chunks_round_trip_riff_and_rf64(tmp_path: Path) -> None:
    audio = _make_sine(0.25, freq_hz=440.0, amp=0.3, channels=2).T  # (samples, channels)
    bext = _sample_bext_payload()
    ixml = _sample_ixml_payload()

    for container in ("WAV", "RF64"):
        wav_path = tmp_path / f"master_{container.lower()}.wav"
        sf.write(str(wav_path), audio, SR, subtype="PCM_24", format=container)
        before_samples, before_sr = sf.read(str(wav_path), dtype="int32")

        assert extract_wav_metadata_chunks(wav_path) == {}
        assert inject_wav_metadata_chunks(wav_path, {}) is False

        injected = inject_wav_metadata_chunks(wav_path, {b"bext": bext, b"iXML": ixml})
        assert injected is True

        # Verify lossless metadata recovery.
        extracted = extract_wav_metadata_chunks(wav_path)
        assert extracted[b"bext"] == bext
        assert extracted[b"iXML"] == ixml

        # Verify audio PCM samples are 100% bit-identical.
        after_samples, after_sr = sf.read(str(wav_path), dtype="int32")
        assert after_sr == before_sr
        assert np.array_equal(after_samples, before_samples)

        # Verify RIFF / RF64 ds64 container size headers match actual file size - 8.
        raw = wav_path.read_bytes()
        expected_riff_size = len(raw) - 8
        if container == "WAV":
            assert raw[:4] == b"RIFF"
            assert struct.unpack_from("<I", raw, 4)[0] == expected_riff_size
        else:
            assert raw[:4] == b"RF64"
            assert struct.unpack_from("<I", raw, 4)[0] == 0xFFFFFFFF
            assert raw[12:16] == b"ds64"
            assert struct.unpack_from("<Q", raw, 20)[0] == expected_riff_size

        # Re-inject updated iXML replaces cleanly without duplicating chunks.
        new_ixml = b"<BWFXML><SCENE>99</SCENE></BWFXML>"
        assert inject_wav_metadata_chunks(wav_path, {b"iXML": new_ixml}) is True
        updated = extract_wav_metadata_chunks(wav_path)
        assert updated[b"bext"] == bext
        assert updated[b"iXML"] == new_ixml


@pytest.mark.unit
def test_wav_metadata_validation_and_non_wav(tmp_path: Path) -> None:
    non_wav = tmp_path / "dummy.bin"
    non_wav.write_bytes(b"NOT_A_WAV_HEADER_12345")
    assert extract_wav_metadata_chunks(non_wav) == {}

    truncated = tmp_path / "broken.wav"
    truncated.write_bytes(b"RIFF\x64\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00")
    with pytest.raises(ValueError, match="Truncated WAV chunk"):
        extract_wav_metadata_chunks(truncated)

    wav_path = tmp_path / "valid.wav"
    sf.write(str(wav_path), np.zeros((128, 1), dtype=np.float32), SR, subtype="PCM_24")
    with pytest.raises(ValueError, match="protected"):
        inject_wav_metadata_chunks(wav_path, {b"data": b"nope"})
    with pytest.raises(ValueError, match="exactly 4 bytes"):
        inject_wav_metadata_chunks(wav_path, {b"bad": b"nope"})


@pytest.mark.unit
def test_export_delivery_artifact_audio_formats_and_metadata(tmp_path: Path) -> None:
    audio = _make_sine(0.5, freq_hz=440.0, amp=0.25, channels=2).T
    source_bwf = tmp_path / "source_bwf.wav"
    master_wav = tmp_path / "master.wav"
    sf.write(str(source_bwf), audio, SR, subtype="PCM_24")
    sf.write(str(master_wav), audio, SR, subtype="FLOAT")

    bext = _sample_bext_payload()
    ixml = _sample_ixml_payload()
    inject_wav_metadata_chunks(source_bwf, {b"bext": bext, b"iXML": ixml})

    # wav24 preserves BWF/iXML from source_media_path.
    out_wav24 = export_delivery_artifact(
        master_wav,
        tmp_path / "delivery_24.wav",
        fmt="wav24",
        source_media_path=source_bwf,
        preserve_metadata=True,
    )
    info24 = sf.info(str(out_wav24))
    assert info24.subtype == "PCM_24"
    assert info24.channels == 2
    assert extract_wav_metadata_chunks(out_wav24) == {b"bext": bext, b"iXML": ixml}

    # preserve_metadata=False strips metadata cleanly.
    out_nometa = export_delivery_artifact(
        master_wav,
        tmp_path / "delivery_nometa.wav",
        fmt="wav24",
        source_media_path=source_bwf,
        preserve_metadata=False,
    )
    assert extract_wav_metadata_chunks(out_nometa) == {}

    # wav32f exports 32-bit float with preserved metadata.
    out_wav32 = export_delivery_artifact(
        master_wav,
        tmp_path / "delivery_32f.wav",
        fmt="wav32f",
        source_media_path=source_bwf,
        preserve_metadata=True,
    )
    assert sf.info(str(out_wav32)).subtype == "FLOAT"
    assert extract_wav_metadata_chunks(out_wav32) == {b"bext": bext, b"iXML": ixml}

    # flac exports lossless 24-bit FLAC.
    out_flac = export_delivery_artifact(master_wav, tmp_path / "delivery.flac", fmt="flac")
    info_flac = sf.info(str(out_flac))
    assert info_flac.format == "FLAC"
    assert info_flac.subtype == "PCM_24"
    assert info_flac.frames == len(audio)

    # aac via bundled/system ffmpeg.
    if ffmpeg_bin_path():
        out_aac = export_delivery_artifact(master_wav, tmp_path / "delivery.m4a", fmt="aac")
        assert out_aac.is_file()
        assert out_aac.stat().st_size > 100


@pytest.mark.unit
def test_export_delivery_artifact_video_mux_lossless(tmp_path: Path) -> None:
    ffmpeg = ffmpeg_bin_path()
    ffprobe = ffprobe_bin_path()
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe unavailable")

    # Generate a tiny real 0.5 s H.264 MP4 test clip with 440 Hz audio.
    source_mp4 = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=0.5:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source_mp4),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )

    # Create a distinct cleaned master WAV (880 Hz stereo).
    master_wav = tmp_path / "cleaned_master.wav"
    cleaned_audio = _make_sine(0.5, freq_hz=880.0, amp=0.25, channels=2).T
    sf.write(str(master_wav), cleaned_audio, SR, subtype="PCM_24")

    # Mux back into MP4 and MOV with -c:v copy.
    for ext, expected_audio_codec in ((".mp4", "aac"), (".mov", "pcm_s24le")):
        out_video = export_delivery_artifact(
            master_wav,
            tmp_path / f"delivered{ext}",
            fmt="video_mux",
            source_media_path=source_mp4,
            preserve_metadata=True,
        )
        assert out_video.is_file()
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(out_video),
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        streams = json.loads(probe.stdout.decode("utf-8"))["streams"]
        video_streams = [s for s in streams if s["codec_type"] == "video"]
        audio_streams = [s for s in streams if s["codec_type"] == "audio"]
        assert len(video_streams) == 1
        assert video_streams[0]["codec_name"] == "h264"
        assert video_streams[0]["width"] == 64
        assert video_streams[0]["height"] == 64
        assert len(audio_streams) == 1
        assert audio_streams[0]["codec_name"] == expected_audio_codec


@pytest.mark.unit
def test_export_delivery_artifact_path_validation_and_atomic_cleanup(tmp_path: Path) -> None:
    master_wav = tmp_path / "master.wav"
    sf.write(str(master_wav), np.zeros((256, 1), dtype=np.float32), SR, subtype="PCM_24")

    with pytest.raises(ValueError, match="Unsupported export format"):
        export_delivery_artifact(master_wav, tmp_path / "out.wav", fmt=cast(ExportFormat, "mp3"))

    with pytest.raises(ValueError, match="overwrite master_wav"):
        export_delivery_artifact(master_wav, master_wav, fmt="wav24")

    with pytest.raises(ValueError, match="directory"):
        export_delivery_artifact(master_wav, tmp_path, fmt="wav24")

    with pytest.raises(ValueError, match="Invalid"):
        export_delivery_artifact(master_wav, Path("bad\x00path.wav"), fmt="wav24")

    with pytest.raises(ValueError, match="source_media_path is required"):
        export_delivery_artifact(master_wav, tmp_path / "out.mp4", fmt="video_mux")

    with pytest.raises(FileNotFoundError):
        export_delivery_artifact(tmp_path / "missing.wav", tmp_path / "out.wav", fmt="wav24")

    # No stray temp files left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["master.wav"]


@pytest.mark.unit
def test_export_delivery_artifact_rejects_symlink_escape_and_bad_extension(
    tmp_path: Path,
) -> None:
    master_wav = tmp_path / "master.wav"
    sf.write(str(master_wav), np.zeros((256, 1), dtype=np.float32), SR, subtype="PCM_24")

    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.wav"
    sentinel = b"DO_NOT_OVERWRITE_SENTINEL"
    victim.write_bytes(sentinel)

    exports = tmp_path / "exports"
    exports.mkdir()

    # Final component symlink pointing outside -> rejected, victim never overwritten.
    symlink_dest = exports / "escape.wav"
    symlink_dest.symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        export_delivery_artifact(master_wav, symlink_dest, fmt="wav24")
    assert victim.read_bytes() == sentinel

    # Symlinked ancestor directory pointing outside -> rejected, nothing written outside.
    symlink_dir = exports / "link_dir"
    symlink_dir.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        export_delivery_artifact(master_wav, symlink_dir / "nested.wav", fmt="wav24")
    assert not (outside / "nested.wav").exists()

    # Extension mismatch against format -> rejected.
    with pytest.raises(ValueError, match="extension"):
        export_delivery_artifact(master_wav, exports / "bad.txt", fmt="wav24")
    with pytest.raises(ValueError, match="extension"):
        export_delivery_artifact(master_wav, exports / "bad.wav", fmt="flac")
    with pytest.raises(ValueError, match="extension"):
        export_delivery_artifact(master_wav, exports / "bad.wav", fmt="aac")
    with pytest.raises(ValueError, match="extension"):
        export_delivery_artifact(
            master_wav, exports / "bad.txt", fmt="video_mux", source_media_path=master_wav
        )
