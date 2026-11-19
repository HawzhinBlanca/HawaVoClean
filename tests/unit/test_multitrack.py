"""Unit tests for Mid/Side true stereo enhancement and Poly-WAV/BWF track routing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio.multitrack import (
    decode_mid_side,
    encode_mid_side,
    enhance_true_stereo_ms,
    extract_polywav_tracks,
    inspect_polywav_tracks,
)


def _append_ixml_chunk(wav_path: Path, track_names: list[str]) -> None:
    """Append a standards-compliant BWF iXML chunk to a classic RIFF WAVE file."""
    track_elems = "".join(
        f"<TRACK><CHANNEL_INDEX>{idx + 1}</CHANNEL_INDEX><NAME>{name}</NAME></TRACK>"
        for idx, name in enumerate(track_names)
    )
    xml_bytes = f"<?xml version='1.0' encoding='UTF-8'?><BWFXML><TRACK_LIST>{track_elems}</TRACK_LIST></BWFXML>".encode()
    if len(xml_bytes) % 2 == 1:
        xml_bytes += b"\x00"

    raw = bytearray(wav_path.read_bytes())
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    raw.extend(b"iXML")
    raw.extend(len(xml_bytes).to_bytes(4, "little"))
    raw.extend(xml_bytes)
    new_riff_size = len(raw) - 8
    raw[4:8] = new_riff_size.to_bytes(4, "little")
    wav_path.write_bytes(bytes(raw))


def test_mid_side_exact_float32_roundtrip() -> None:
    rng = np.random.default_rng(123)
    left = rng.uniform(-0.9, 0.9, size=48000).astype(np.float32)
    right = rng.uniform(-0.9, 0.9, size=48000).astype(np.float32)

    mid, side = encode_mid_side(left, right)
    assert mid.dtype == np.float32 and side.dtype == np.float32
    l_rec, r_rec = decode_mid_side(mid, side)

    assert float(np.max(np.abs(l_rec - left))) < 1e-6
    assert float(np.max(np.abs(r_rec - right))) < 1e-6

    # Pure mono -> side is identically 0.
    m_mono, s_mono = encode_mid_side(left, left)
    assert float(np.max(np.abs(s_mono))) == 0.0
    np.testing.assert_allclose(m_mono, left, atol=1e-6)


def test_enhance_true_stereo_ms_preserves_imaging_and_controls_side() -> None:
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / float(sr)
    center_voice = (0.4 * np.sin(2.0 * math.pi * 500.0 * t)).astype(np.float32)
    side_ambience = (0.2 * np.sin(2.0 * math.pi * 1200.0 * t)).astype(np.float32)
    left, right = decode_mid_side(center_voice, side_ambience)
    stereo = np.vstack((left, right))

    # Identity + side_attenuation=0.0 round-trips within 1e-6.
    out_id = enhance_true_stereo_ms(stereo, lambda m: m, side_attenuation=0.0)
    assert out_id.shape == stereo.shape
    assert float(np.max(np.abs(out_id - stereo))) < 1e-6

    # side_attenuation=1.0 collapses to mono enhanced Mid.
    out_mono = enhance_true_stereo_ms(
        stereo, lambda m: (m * 0.5).astype(np.float32), side_attenuation=1.0
    )
    np.testing.assert_allclose(out_mono[0], out_mono[1], atol=1e-6)
    np.testing.assert_allclose(out_mono[0], center_voice * 0.5, atol=1e-5)

    # side_attenuation=0.5 preserves phase-coherent stereo with 50% Side reduction.
    out_half = enhance_true_stereo_ms(stereo, lambda m: m, side_attenuation=0.5)
    m_out, s_out = encode_mid_side(out_half[0], out_half[1])
    np.testing.assert_allclose(m_out, center_voice, atol=1e-5)
    np.testing.assert_allclose(s_out, side_ambience * 0.5, atol=1e-5)

    # Empty stereo array passes cleanly.
    empty = enhance_true_stereo_ms(np.empty((2, 0), dtype=np.float32), lambda m: m)
    assert empty.shape == (2, 0)


def test_mid_side_validation_errors() -> None:
    a = np.zeros(10, dtype=np.float32)
    b = np.zeros(12, dtype=np.float32)
    with pytest.raises(ValueError, match="length mismatch"):
        encode_mid_side(a, b)
    with pytest.raises(ValueError, match="finite"):
        encode_mid_side(
            np.asarray([float("nan")], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
        )
    with pytest.raises(ValueError, match="shape"):
        enhance_true_stereo_ms(np.zeros((3, 10), dtype=np.float32), lambda m: m)
    with pytest.raises(ValueError, match="side_attenuation"):
        enhance_true_stereo_ms(
            np.zeros((2, 10), dtype=np.float32), lambda m: m, side_attenuation=1.2
        )
    with pytest.raises(ValueError, match="1D arrays"):
        encode_mid_side(np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="1D arrays"):
        decode_mid_side(np.zeros((2, 4), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="length mismatch"):
        decode_mid_side(a, b)
    with pytest.raises(ValueError, match="finite"):
        decode_mid_side(
            np.asarray([float("inf")], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
        )
    with pytest.raises(ValueError, match="1D array"):
        enhance_true_stereo_ms(
            np.zeros((2, 10), dtype=np.float32),
            lambda _m: np.zeros((2, 10), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="NaN or Inf"):
        enhance_true_stereo_ms(
            np.zeros((2, 10), dtype=np.float32),
            lambda m: np.full_like(m, float("nan")),
        )

    # Enhancer returning shorter or longer array is fitted cleanly.
    stereo10 = np.ones((2, 10), dtype=np.float32) * 0.25
    short_out = enhance_true_stereo_ms(stereo10, lambda m: m[:6])
    long_out = enhance_true_stereo_ms(stereo10, lambda m: np.pad(m, (0, 5)))
    assert short_out.shape == (2, 10)
    assert long_out.shape == (2, 10)


def test_inspect_and_extract_polywav_tracks_with_bwf_ixml(tmp_path: Path) -> None:
    sr = 48000
    n = sr  # 1.0 s
    t = np.arange(n, dtype=np.float64) / float(sr)

    # 6-track field recorder Poly-WAV:
    # 0: Boom (active speech-band)
    # 1: Lav 1 (active speech-band, quieter)
    # 2: Silent track
    # 3: 50 Hz mains hum only
    # 4: 10 kHz ultrasonic/hiss tone only
    # 5: Lav 2 (intermittent speech-band bursts)
    burst = np.where((t % 0.25) < 0.15, 1.0, 0.0)
    tracks_data: list[np.ndarray[Any, np.dtype[np.float32]]] = [
        (0.35 * np.sin(2.0 * math.pi * 800.0 * t)).astype(np.float32),
        (0.15 * np.sin(2.0 * math.pi * 1200.0 * t)).astype(np.float32),
        np.zeros(n, dtype=np.float32),
        (0.25 * np.sin(2.0 * math.pi * 50.0 * t)).astype(np.float32),
        (0.20 * np.sin(2.0 * math.pi * 10000.0 * t)).astype(np.float32),
        (0.30 * burst * np.sin(2.0 * math.pi * 650.0 * t)).astype(np.float32),
    ]
    poly = np.column_stack(tracks_data)
    poly_path = tmp_path / "scene_04_take_02.wav"
    sf.write(str(poly_path), poly, sr, subtype="PCM_24")
    _append_ixml_chunk(poly_path, ["Boom", "Lav 1", "Spare", "HumBus", "Hiss", "Lav 2"])

    infos = inspect_polywav_tracks(poly_path)
    assert len(infos) == 6
    assert [info.name for info in infos] == ["Boom", "Lav 1", "Spare", "HumBus", "Hiss", "Lav 2"]
    assert infos[0].likely_dialogue is True
    assert infos[1].likely_dialogue is True
    assert infos[2].likely_dialogue is False
    assert infos[2].rms_dbfs <= -119.0
    assert infos[3].likely_dialogue is False  # 50 Hz hum rejected
    assert infos[4].likely_dialogue is False  # 10 kHz hiss rejected
    assert infos[5].likely_dialogue is True
    assert 0.4 <= infos[5].speech_activity_ratio <= 0.85
    assert infos[0].to_dict()["index"] == 0

    out_dir = tmp_path / "extracted"
    extracted = extract_polywav_tracks(poly_path, [0, 5, 1], out_dir)
    assert len(extracted) == 3
    assert "Boom" in extracted[0].name
    assert "Lav_2" in extracted[1].name
    assert "Lav_1" in extracted[2].name

    for out_file, orig_idx in zip(extracted, [0, 5, 1], strict=True):
        dec, dec_sr = sf.read(str(out_file), dtype="float32")
        assert dec_sr == sr
        assert dec.ndim == 1
        np.testing.assert_allclose(dec, tracks_data[orig_idx], atol=1e-5)


def test_polywav_16_channels_and_validation(tmp_path: Path) -> None:
    sr = 48000
    n = 4800
    data16 = np.zeros((n, 16), dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / float(sr)
    data16[:, 15] = (0.3 * np.sin(2.0 * math.pi * 1000.0 * t)).astype(np.float32)
    p16 = tmp_path / "poly16.wav"
    sf.write(str(p16), data16, sr, subtype="FLOAT")

    infos = inspect_polywav_tracks(p16)
    assert len(infos) == 16
    assert infos[15].name == "Track 16"
    assert infos[15].likely_dialogue is True

    extracted = extract_polywav_tracks(p16, [15], tmp_path / "out16")
    assert len(extracted) == 1
    dec, _ = sf.read(str(extracted[0]), dtype="float32")
    np.testing.assert_allclose(dec, data16[:, 15], atol=1e-6)

    # 17 channels exceeds max supported Poly-WAV count.
    p17 = tmp_path / "poly17.wav"
    sf.write(str(p17), np.zeros((480, 17), dtype=np.float32), sr, subtype="PCM_24")
    with pytest.raises(ValueError, match="outside supported range"):
        inspect_polywav_tracks(p17)

    with pytest.raises(ValueError, match="out of range"):
        extract_polywav_tracks(p16, [16], tmp_path / "out_err")
    with pytest.raises(ValueError, match="must not be empty"):
        extract_polywav_tracks(p16, [], tmp_path / "out_err")
    with pytest.raises(FileNotFoundError):
        inspect_polywav_tracks(tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError):
        extract_polywav_tracks(tmp_path / "missing.wav", [0], tmp_path / "out_err")
    with pytest.raises(ValueError, match="outside supported range"):
        extract_polywav_tracks(p17, [0], tmp_path / "out_err")

    # Multi-block streaming (> 48000 samples) and very short (< 20ms) files.
    t_long = np.arange(52000, dtype=np.float64) / float(sr)
    long_poly = np.column_stack(
        [
            (0.3 * np.sin(2.0 * math.pi * 700.0 * t_long)).astype(np.float32),
            np.zeros(52000, dtype=np.float32),
        ]
    )
    p_long = tmp_path / "long_poly.wav"
    sf.write(str(p_long), long_poly, sr, subtype="PCM_24")
    long_infos = inspect_polywav_tracks(p_long)
    assert long_infos[0].likely_dialogue is True
    assert long_infos[1].likely_dialogue is False

    short_wav = tmp_path / "tiny.wav"
    sf.write(
        str(short_wav),
        (0.3 * np.sin(2.0 * math.pi * 800.0 * np.arange(200) / float(sr))).astype(np.float32),
        sr,
        subtype="PCM_24",
    )
    short_infos = inspect_polywav_tracks(short_wav)
    assert len(short_infos) == 1
    assert short_infos[0].likely_dialogue is True
