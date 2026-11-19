"""Adversarial DSP, binary-parser, iXML, and cross-platform hardening tests.

Covers numerical torture under strict warnings-as-errors for ``pro_repair.py``,
hostile RIFF/RF64 binary container & export hardening for ``delivery.py``,
and hostile iXML / Poly-WAV / Mid-Side torture for ``multitrack.py``.
"""

from __future__ import annotations

import io
import math
import os
import struct
import subprocess
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.audio import delivery as delivery_mod
from hawavoclean.audio import multitrack as multitrack_mod
from hawavoclean.audio.delivery import (
    MAX_METADATA_CHUNK_BYTES,
    apply_delivery_loudness,
    compute_loudness_range_lu,
    export_delivery_artifact,
    extract_wav_metadata_chunks,
    inject_wav_metadata_chunks,
)
from hawavoclean.audio.multitrack import (
    MAX_IXML_CHUNK_BYTES,
    decode_mid_side,
    encode_mid_side,
    enhance_true_stereo_ms,
    extract_polywav_tracks,
    inspect_polywav_tracks,
)
from hawavoclean.errors import OutputValidationError, PreflightError
from hawavoclean.finishing import pro_repair as pro_repair_mod
from hawavoclean.finishing.pro_repair import (
    ProRepairOptions,
    apply_declip,
    apply_dynamic_deplosive,
    apply_mouth_declick,
    apply_pro_repair_chain,
    apply_room_tone_floor,
    apply_spectral_box_repair,
)
from hawavoclean.platform_fs import is_reparse_or_symlink

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adversarial_signals(n: int) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    """Generate pathological float32 waveforms of exact length ``n``."""
    dirac = np.zeros(n, dtype=np.float32)
    if n > 0:
        dirac[n // 2] = np.float32(1.0)
    nyquist = np.array([1.0 if (i & 1) == 0 else -1.0 for i in range(n)], dtype=np.float32)
    t = np.linspace(0.0, 20.0, n, dtype=np.float64)
    extreme = (np.sin(t) * 1e6).astype(np.float32)
    step_f32_max = np.where(
        np.arange(n) < (n // 2), np.float32(-3.4e38), np.float32(3.4e38)
    ).astype(np.float32)
    return {
        "zeros": np.zeros(n, dtype=np.float32),
        "dc_pos": np.full(n, np.float32(0.5), dtype=np.float32),
        "dc_neg": np.full(n, np.float32(-1.0), dtype=np.float32),
        "dirac": dirac,
        "nyquist": nyquist,
        "subnormal": np.full(n, np.float32(1e-40), dtype=np.float32),
        "extreme_1e6": extreme,
        "step_f32_max": step_f32_max,
    }


def _pack_riff_wav(chunks: list[tuple[bytes, bytes]], *, pad_last: bool = True) -> bytes:
    """Assemble a classic RIFF WAVE byte stream from raw ``(chunk_id, payload)`` pairs."""
    body = bytearray(b"WAVE")
    for idx, (cid, payload) in enumerate(chunks):
        body.extend(cid)
        body.extend(struct.pack("<I", len(payload)))
        body.extend(payload)
        if (len(payload) & 1) and (pad_last or idx < len(chunks) - 1):
            body.append(0)
    return b"RIFF" + struct.pack("<I", len(body)) + bytes(body)


def _minimal_fmt_payload(channels: int = 1, sr: int = 48000, bits: int = 16) -> bytes:
    block_align = channels * (bits // 8)
    byte_rate = sr * block_align
    return struct.pack("<HHIIHH", 1, channels, sr, byte_rate, block_align, bits)


def _pack_rf64_wav(
    chunks: list[tuple[bytes, bytes, bool]],
    *,
    riff_size_override: int | None = None,
    data_size_override: int | None = None,
    extra_table: list[tuple[bytes, int]] | None = None,
) -> bytes:
    """Assemble an RF64 WAVE stream with ``ds64`` first chunk.

    Each tuple in ``chunks`` is ``(chunk_id, payload, use_sentinel_size)``.
    """
    data_len = sum(len(p) for cid, p, _ in chunks if cid == b"data")
    table = list(extra_table or [])
    ds64_len = 28 + 12 * len(table)

    body = bytearray()
    for cid, payload, use_sentinel in chunks:
        body.extend(cid)
        body.extend(struct.pack("<I", 0xFFFFFFFF if use_sentinel else len(payload)))
        body.extend(payload)
        if len(payload) & 1:
            body.append(0)

    riff_size = 4 + 8 + ds64_len + len(body)
    if riff_size_override is not None:
        riff_size = riff_size_override
    data_size = data_size_override if data_size_override is not None else data_len

    ds64_payload = bytearray(struct.pack("<QQQI", riff_size, data_size, 0, len(table)))
    for cid, csize in table:
        ds64_payload.extend(cid)
        ds64_payload.extend(struct.pack("<Q", csize))

    out = bytearray(b"RF64")
    out.extend(struct.pack("<I", 0xFFFFFFFF))
    out.extend(b"WAVE")
    out.extend(b"ds64")
    out.extend(struct.pack("<I", len(ds64_payload)))
    out.extend(ds64_payload)
    out.extend(body)
    return bytes(out)


def _append_raw_chunk_to_wav(wav_path: Path, chunk_id: bytes, payload: bytes) -> None:
    """Append an arbitrary chunk directly onto a classic RIFF WAV and update RIFF size."""
    raw = bytearray(wav_path.read_bytes())
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    raw.extend(chunk_id)
    raw.extend(struct.pack("<I", len(payload)))
    raw.extend(payload)
    if len(payload) & 1:
        raw.append(0)
    raw[4:8] = struct.pack("<I", len(raw) - 8)
    wav_path.write_bytes(bytes(raw))


# ---------------------------------------------------------------------------
# Part 1: pro_repair.py Numerical & Edge-Case Torture (zero warnings)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
@pytest.mark.parametrize("n", [0, 1, 2, 31, 63, 64, 512])
def test_pro_repair_torture_matrix_zero_warnings(sr: int, n: int) -> None:
    """All 6 pro_repair functions survive pathological signals across lengths & sample rates."""
    signals = _make_adversarial_signals(n)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for sig in signals.values():
            shapes = [
                sig,
                sig[np.newaxis, :],
                np.tile(sig[np.newaxis, :], (2, 1)),
                np.tile(sig[np.newaxis, :], (4, 1)),
            ]
            for arr in shapes:
                out_dp, _ = apply_dynamic_deplosive(arr, sr)
                out_dc, _ = apply_mouth_declick(arr, sr, sensitivity=1.0)
                out_cl, _ = apply_declip(arr, sr)
                out_rt = apply_room_tone_floor(arr, np.zeros_like(arr), sr, target_floor_db=-50.0)
                out_sb = apply_spectral_box_repair(
                    arr,
                    sr,
                    start_s=0.0,
                    end_s=0.001,
                    f_low_hz=0.0,
                    f_high_hz=sr / 2.0,
                )
                out_ch, _ = apply_pro_repair_chain(
                    arr,
                    sr,
                    ProRepairOptions(room_tone_floor=True),
                    original_waveform=arr,
                )
                for out in (out_dp, out_dc, out_cl, out_rt, out_sb, out_ch):
                    assert out.shape == arr.shape
                    assert out.dtype == np.float32
                    assert out.flags.c_contiguous
                    assert np.all(np.isfinite(out))


@pytest.mark.unit
def test_pro_repair_validation_overflow_and_low_sr() -> None:
    """Fail-closed validation on 3D shapes, float64 overflow, bool sample_rate, and bad args."""
    x = np.zeros(128, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="1D or 2D"):
            apply_declip(np.zeros((1, 2, 4), dtype=np.float32), 48000)
        with pytest.raises(ValueError, match="sample_rate"):
            apply_declip(x, True)
        with pytest.raises(ValueError, match="NaN or Inf"):
            apply_declip(np.array([1e100, 0.0], dtype=np.float64), 48000)
        with pytest.raises(ValueError, match="threshold_db"):
            apply_dynamic_deplosive(x, 48000, threshold_db=float("nan"))
        with pytest.raises(ValueError, match="max_reduction_db"):
            apply_dynamic_deplosive(x, 48000, max_reduction_db=-1.0)
        with pytest.raises(ValueError, match="max_click_ms"):
            apply_mouth_declick(x, 48000, max_click_ms=0.0)
        with pytest.raises(ValueError, match="clip_threshold"):
            apply_declip(x, 48000, clip_threshold=0.0)
        with pytest.raises(ValueError, match="target_floor_db"):
            apply_room_tone_floor(x, x, 48000, target_floor_db=-5.0)
        with pytest.raises(ValueError, match="finite"):
            apply_spectral_box_repair(
                x, 48000, start_s=True, end_s=0.5, f_low_hz=100.0, f_high_hz=500.0
            )
        with pytest.raises(ValueError, match="n_fft"):
            apply_spectral_box_repair(
                x, 48000, start_s=0.0, end_s=0.5, f_low_hz=100.0, f_high_hz=500.0, n_fft=65
            )

        # Low sample rate (<= 400) and zero max_reduction_db return untouched cleanly.
        t = np.arange(200, dtype=np.float64) / 400.0
        tone = (0.5 * np.sin(2.0 * np.pi * 60.0 * t)).astype(np.float32)
        out_low_sr, cnt_low = apply_dynamic_deplosive(tone, 400)
        assert cnt_low == 0 and np.array_equal(out_low_sr, tone)
        out_zero_gr, cnt_z = apply_dynamic_deplosive(tone, 48000, max_reduction_db=0.0)
        assert cnt_z == 0 and np.array_equal(out_zero_gr, tone)

        # Extreme threshold_db & max_reduction_db never underflow.
        t48 = np.arange(4800, dtype=np.float64) / 48000.0
        plosive = (0.7 * np.sin(2.0 * np.pi * 60.0 * t48)).astype(np.float32)
        out_huge_gr, cnt_huge = apply_dynamic_deplosive(
            plosive, 48000, threshold_db=-1e5, max_reduction_db=1e5
        )
        assert cnt_huge >= 1 and np.all(np.isfinite(out_huge_gr))

        # Zero-channel 2D arrays return empty with exact shape.
        empty_ch = np.empty((0, 64), dtype=np.float32)
        assert apply_dynamic_deplosive(empty_ch, 48000)[0].shape == (0, 64)
        assert apply_mouth_declick(empty_ch, 48000)[0].shape == (0, 64)
        assert apply_declip(empty_ch, 48000)[0].shape == (0, 64)
        assert apply_room_tone_floor(empty_ch, empty_ch, 48000).shape == (0, 64)
        assert apply_spectral_box_repair(
            empty_ch, 48000, start_s=0.0, end_s=0.1, f_low_hz=100.0, f_high_hz=500.0
        ).shape == (0, 64)


@pytest.mark.unit
def test_pro_repair_declick_ar_and_boundary_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise AR fallbacks, Hermite fallbacks, boundary spans, and extreme max_click_ms."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # _merge_runs empty list
        assert pro_repair_mod._merge_runs([], 4) == []

        # _fit_ar_coefficients edge cases: too short, Toeplitz exception, NaN coeffs
        tiny = np.ones(4, dtype=np.float64)
        assert np.all(pro_repair_mod._fit_ar_coefficients(tiny, 6) == 0.0)
        sig = np.sin(np.linspace(0.0, 10.0, 64))
        with patch("scipy.linalg.solve_toeplitz", side_effect=RuntimeError("singular")):
            assert np.all(pro_repair_mod._fit_ar_coefficients(sig, 4) == 0.0)
        with patch("scipy.linalg.solve_toeplitz", return_value=np.full(4, np.nan)):
            assert np.all(pro_repair_mod._fit_ar_coefficients(sig, 4) == 0.0)

        # _reconstruct_click_span edge cases: invalid bounds, short span (<=4), short context
        arr = np.sin(np.linspace(0.0, 20.0, 200))
        assert np.array_equal(pro_repair_mod._reconstruct_click_span(arr, 0, 5), arr[0:5])
        assert np.array_equal(pro_repair_mod._reconstruct_click_span(arr, 195, 200), arr[195:200])
        assert pro_repair_mod._reconstruct_click_span(arr, 20, 23).shape == (3,)
        assert pro_repair_mod._reconstruct_click_span(arr, 5, 20).shape == (15,)

        # Divergent AR recursion -> step_limit aborts cleanly before float64 overflow.
        base = np.sin(np.linspace(0.0, 40.0, 400))
        monkeypatch.setattr(
            pro_repair_mod,
            "_fit_ar_coefficients",
            lambda _s, order: np.full(order, 2.5, dtype=np.float64),
        )
        out_div = pro_repair_mod._reconstruct_click_span(base, 150, 190)
        assert np.all(np.isfinite(out_div))

        # Forward stays bounded while backward diverges.
        call_idx = {"n": 0}

        def _fwd_ok_bwd_div(_s: Any, order: int) -> np.ndarray[Any, np.dtype[np.float64]]:
            call_idx["n"] += 1
            if call_idx["n"] == 1:
                return np.zeros(order, dtype=np.float64)
            return np.full(order, 2.5, dtype=np.float64)

        monkeypatch.setattr(pro_repair_mod, "_fit_ar_coefficients", _fwd_ok_bwd_div)
        out_bwd_div = pro_repair_mod._reconstruct_click_span(base, 150, 190)
        assert np.all(np.isfinite(out_bwd_div))

        # Moderate overshoot (> 1.25 * ctx_peak, <= 10 * ctx_peak) -> Hermite fallback.
        const_ctx = np.ones(400, dtype=np.float64)
        monkeypatch.setattr(
            pro_repair_mod,
            "_fit_ar_coefficients",
            lambda _s, order: np.concatenate(([1.5], np.zeros(order - 1))),
        )
        out_over = pro_repair_mod._reconstruct_click_span(const_ctx, 150, 156)
        assert np.all(np.isfinite(out_over))

    # Restore monkeypatch for _declick_channel tests.
    monkeypatch.undo()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Quiet signal (< 1e-5) -> early return
        quiet = np.full(256, np.float32(1e-7), dtype=np.float32)
        out_q, cnt_q = apply_mouth_declick(quiet, 48000, sensitivity=1.0)
        assert cnt_q == 0 and np.array_equal(out_q, quiet)

        # Non-multiple of frame_len + huge max_click_ms (no OverflowError!) + boundary click
        n_odd = 4805
        t = np.arange(n_odd, dtype=np.float64) / 48000.0
        clean = (0.2 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        clean[0] = np.float32(0.9)  # boundary click at sample 0 -> skipped after clamping
        clean[2400] += np.float32(0.8)  # interior click
        out_huge, cnt_huge = apply_mouth_declick(clean, 48000, sensitivity=0.9, max_click_ms=1e308)
        assert cnt_huge >= 1 and np.all(np.isfinite(out_huge))

        # Click whose padded span exceeds max_click_samples -> tight fallback + boundary skip
        tight = (0.15 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
        tight[0] += np.float32(0.95)
        tight[1] -= np.float32(0.95)
        tight[2000 : 2000 + 3] += np.array([0.8, -0.8, 0.8], dtype=np.float32)
        out_tight, _ = apply_mouth_declick(tight, 48000, sensitivity=1.0, max_click_ms=0.1)
        assert np.all(np.isfinite(out_tight))

        # Hermite fallback to linear when cubic overshoots bound
        steep = np.zeros(10, dtype=np.float64)
        steep[1] = -10.0
        steep[2] = 0.1
        steep[7] = 0.1
        steep[8] = 10.0
        filled = pro_repair_mod._hermite_fill_span(steep, 3, 7)
        assert np.all(np.isfinite(filled))


@pytest.mark.unit
def test_pro_repair_declip_room_tone_chain_and_spectral_branches() -> None:
    """Cover declip boundaries, room-tone fallbacks, chain combinations, and spectral box edges."""
    sr = 48000
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Declip: n < 8, boundary-touching plateau (start < 2, end > n - 2), negative plateau
        short7 = np.ones(7, dtype=np.float32)
        assert apply_declip(short7, sr)[1] == 0

        boundary_clip = np.zeros(64, dtype=np.float32)
        boundary_clip[0:5] = 1.0  # start < 2 -> skipped
        boundary_clip[59:64] = -1.0  # end > n - 2 -> skipped
        assert apply_declip(boundary_clip, sr)[1] == 0

        # Room-tone: continuous speech with NO pauses -> residual fallback path
        rng = np.random.default_rng(99)
        n = sr
        t = np.arange(n, dtype=np.float64) / sr
        continuous_speech = (0.3 * np.sin(2.0 * np.pi * 350.0 * t)).astype(np.float32)
        noisy_continuous = (continuous_speech + 0.02 * rng.standard_normal(n)).astype(np.float32)
        filled_res = apply_room_tone_floor(
            noisy_continuous, continuous_speech, sr, target_floor_db=-40.0
        )
        assert filled_res.shape == continuous_speech.shape
        assert np.all(np.isfinite(filled_res))

        # Continuous speech where original == cleaned -> residual is zero -> returns unchanged
        unchanged_res = apply_room_tone_floor(
            continuous_speech, continuous_speech, sr, target_floor_db=-40.0
        )
        assert np.array_equal(unchanged_res, continuous_speech)

        # Cleaned already louder than target floor -> skipped (cleaned_floor_rms >= target_rms)
        loud_clean = (0.2 * rng.standard_normal(n)).astype(np.float32)
        unchanged_floor = apply_room_tone_floor(loud_clean, loud_clean, sr, target_floor_db=-60.0)
        assert np.array_equal(unchanged_floor, loud_clean)

        # Extreme negative target_floor_db (-1e10) -> no underflow, skipped
        unchanged_neg = apply_room_tone_floor(
            noisy_continuous, np.zeros(n, dtype=np.float32), sr, target_floor_db=-1e10
        )
        assert np.array_equal(unchanged_neg, np.zeros(n, dtype=np.float32))

        # Quiet cleaned (max_c <= 1e-4) and pause_mask < 2 edge case
        quiet_c = np.full(n, np.float32(1e-5), dtype=np.float32)
        orig_with_floor = (0.01 * rng.standard_normal(n)).astype(np.float32)
        out_qc = apply_room_tone_floor(orig_with_floor, quiet_c, sr, target_floor_db=-50.0)
        assert np.all(np.isfinite(out_qc))

        # _synthesize_room_tone with zero mag_profile -> returns zeros
        zero_bed = pro_repair_mod._synthesize_room_tone(
            np.zeros(65, dtype=np.float64), 256, 0.01, "seed"
        )
        assert np.array_equal(zero_bed, np.zeros(256, dtype=np.float64))

        # _extract_ambient_profile with tiny ambient (< 1e-6) and tiny 2-frame quiet pause
        assert (
            pro_repair_mod._extract_ambient_profile(
                np.full(128, 1e-7, dtype=np.float64), np.zeros(128, dtype=np.float64), sr
            )
            is None
        )
        # Only 1 pause frame (pause_mask count == 1) -> hits else branch at pause_mask < 2
        two_frame_orig = np.concatenate(
            [np.full(64, 0.01, dtype=np.float64), np.full(64, 0.5, dtype=np.float64)]
        )
        two_frame_clean = np.concatenate(
            [np.zeros(64, dtype=np.float64), np.full(64, 0.5, dtype=np.float64)]
        )
        _ = pro_repair_mod._extract_ambient_profile(two_frame_orig, two_frame_clean, 3200)

        # apply_pro_repair_chain combinations:
        # 1) All disabled
        out_off, acts_off = apply_pro_repair_chain(
            continuous_speech,
            sr,
            ProRepairOptions(declip=False, deplosive=False, declick=False, room_tone_floor=False),
        )
        assert acts_off == [] and np.array_equal(out_off, continuous_speech)

        # 2) Enabled on clean silence (0 events each) + room_tone_floor with blend_ratio
        zeros = np.zeros(sr, dtype=np.float32)
        _, acts_clean = apply_pro_repair_chain(
            zeros,
            sr,
            ProRepairOptions(
                declip=True,
                deplosive=True,
                declick=True,
                room_tone_floor=True,
                room_tone_target_floor_db=None,
                room_tone_blend_ratio=0.25,
            ),
            original_waveform=orig_with_floor,
        )
        assert acts_clean == ["room_tone_floor(blend_ratio=0.25)"]

        # 3) room_tone_floor enabled with both target=None and blend=0 (no change)
        _, acts_noop = apply_pro_repair_chain(
            zeros,
            sr,
            ProRepairOptions(
                declip=False,
                deplosive=False,
                declick=False,
                room_tone_floor=True,
                room_tone_target_floor_db=None,
                room_tone_blend_ratio=0.0,
            ),
            original_waveform=orig_with_floor,
        )
        assert acts_noop == []

        # 4) Default options (options=None)
        _, acts_def = apply_pro_repair_chain(zeros, sr)
        assert acts_def == []

        # apply_spectral_box_repair edges:
        sig = (0.3 * np.sin(2.0 * np.pi * 1000.0 * t[:4800])).astype(np.float32)  # 0.1 s
        # Past EOF (normal and 1e308), end_s past EOF (1e308)
        assert np.array_equal(
            apply_spectral_box_repair(
                sig, sr, start_s=1e308, end_s=1.5e308, f_low_hz=100.0, f_high_hz=500.0
            ),
            sig,
        )
        out_end_huge = apply_spectral_box_repair(
            sig, sr, start_s=0.0, end_s=1e308, f_low_hz=0.0, f_high_hz=24000.0
        )
        assert out_end_huge.shape == sig.shape and np.all(np.isfinite(out_end_huge))

        # Sub-sample window (end_idx <= start_idx) -> untouched
        assert np.array_equal(
            apply_spectral_box_repair(
                sig, sr, start_s=0.0, end_s=1e-7, f_low_hz=100.0, f_high_hz=500.0
            ),
            sig,
        )

        # n_fft % 4 == 0 (not % 8): 68; n_fft % 2 == 0 (not % 4): 66
        for nfft in (66, 68):
            out_n = apply_spectral_box_repair(
                sig, sr, start_s=0.02, end_s=0.08, f_low_hz=800.0, f_high_hz=1200.0, n_fft=nfft
            )
            assert out_n.shape == sig.shape and np.all(np.isfinite(out_n))

        # Sub-bin frequency band & sub-frame time box -> nearest bin/frame fallbacks
        out_sub = apply_spectral_box_repair(
            sig, sr, start_s=0.0500, end_s=0.0501, f_low_hz=1000.0, f_high_hz=1000.05, n_fft=2048
        )
        assert out_sub.shape == sig.shape and np.all(np.isfinite(out_sub))

        # _raised_cosine_box_1d direct edge cases (no points inside, feather=0, no taper)
        coords = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        assert np.all(pro_repair_mod._raised_cosine_box_1d(coords, 10.0, 20.0, 1.0) == 0.0)
        assert np.array_equal(
            pro_repair_mod._raised_cosine_box_1d(coords, 1.0, 2.0, 0.0),
            np.array([0.0, 1.0, 1.0, 0.0]),
        )
        no_taper = pro_repair_mod._raised_cosine_box_1d(
            coords, 0.0, 3.0, 0.5, taper_lo=False, taper_hi=False
        )
        assert np.all(no_taper == 1.0)


# ---------------------------------------------------------------------------
# Part 2: delivery.py Hostile RIFF/RF64 Parser, Loudness & Export Hardening
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delivery_riff_rf64_hostile_binary_parser(tmp_path: Path) -> None:
    """Torture RIFF/RF64 parser with truncated, overflow, sentinel, flood, and oversized chunks."""
    # 1. Truncated headers (< 12 bytes) and invalid magic
    for name, content in (
        ("empty.wav", b""),
        ("short4.wav", b"RIFF"),
        ("short11.wav", b"RIFF\x00\x00\x00\x00WAV"),
        ("bad_magic.wav", b"FORM\x00\x00\x00\x10AIFFdata\x00\x00\x00\x00"),
        ("bad_form.wav", b"RIFF\x00\x00\x00\x10AVI data\x00\x00\x00\x00"),
    ):
        p = tmp_path / name
        p.write_bytes(content)
        assert extract_wav_metadata_chunks(p) == {}
        with pytest.raises(ValueError, match="not a valid RIFF/RF64"):
            inject_wav_metadata_chunks(p, {b"iXML": b"<BWFXML/>"})

    # 2. Missing file
    with pytest.raises(FileNotFoundError):
        extract_wav_metadata_chunks(tmp_path / "does_not_exist.wav")
    with pytest.raises(FileNotFoundError):
        inject_wav_metadata_chunks(tmp_path / "does_not_exist.wav", {b"iXML": b"<BWFXML/>"})

    # 3. Truncated chunk header (3 trailing bytes after valid fmt chunk)
    trunc_hdr = tmp_path / "trunc_hdr.wav"
    trunc_hdr.write_bytes(_pack_riff_wav([(b"fmt ", _minimal_fmt_payload())]) + b"JNK")
    with pytest.raises(ValueError, match="Truncated WAV chunk header"):
        extract_wav_metadata_chunks(trunc_hdr)

    # 4. Classic RIFF with chunk_size = 0xFFFFFFFF sentinel
    riff_sentinel = tmp_path / "riff_sentinel.wav"
    riff_sentinel.write_bytes(b"RIFF\x24\x00\x00\x00WAVEdata\xff\xff\xff\xff\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="0xFFFFFFFF"):
        extract_wav_metadata_chunks(riff_sentinel)

    # 5. RF64 with 0xFFFFFFFF sentinel without ds64 entry
    rf64_no_ds64 = tmp_path / "rf64_no_ds64.wav"
    rf64_no_ds64.write_bytes(b"RF64\xff\xff\xff\xffWAVEdata\xff\xff\xff\xff\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="without ds64 entry"):
        extract_wav_metadata_chunks(rf64_no_ds64)

    # 6. RF64 malformed ds64 (< 28 bytes)
    rf64_short_ds64 = tmp_path / "rf64_short_ds64.wav"
    rf64_short_ds64.write_bytes(b"RF64\xff\xff\xff\xffWAVEds64\x10\x00\x00\x00" + b"\x00" * 16)
    with pytest.raises(ValueError, match="expected >=28 bytes"):
        extract_wav_metadata_chunks(rf64_short_ds64)

    # 7. RF64 truncated ds64 size table
    rf64_bad_tbl = tmp_path / "rf64_bad_tbl.wav"
    bad_ds64 = struct.pack("<QQQI", 100, 0, 0, 5)  # claims 5 entries, provides 0
    rf64_bad_tbl.write_bytes(b"RF64\xff\xff\xff\xffWAVEds64" + struct.pack("<I", 28) + bad_ds64)
    with pytest.raises(ValueError, match="Malformed RF64 ds64 size table"):
        extract_wav_metadata_chunks(rf64_bad_tbl)

    # 8. RF64 ds64 size overflow (riff_size, data_size, table entry, and > file_size)
    rf64_over_riff = tmp_path / "rf64_over_riff.wav"
    rf64_over_riff.write_bytes(
        _pack_rf64_wav([(b"fmt ", _minimal_fmt_payload(), False)], riff_size_override=(1 << 64) - 1)
    )
    with pytest.raises(ValueError, match="ds64 chunk size overflow"):
        extract_wav_metadata_chunks(rf64_over_riff)

    rf64_over_data = tmp_path / "rf64_over_data.wav"
    rf64_over_data.write_bytes(
        _pack_rf64_wav([(b"fmt ", _minimal_fmt_payload(), False)], data_size_override=999_999)
    )
    with pytest.raises(ValueError, match="data_size exceeds file size"):
        extract_wav_metadata_chunks(rf64_over_data)

    rf64_over_tbl = tmp_path / "rf64_over_tbl.wav"
    rf64_over_tbl.write_bytes(
        _pack_rf64_wav(
            [(b"fmt ", _minimal_fmt_payload(), False)],
            extra_table=[(b"JUNK", (1 << 64) - 1)],
        )
    )
    with pytest.raises(ValueError, match="size table overflow"):
        extract_wav_metadata_chunks(rf64_over_tbl)

    rf64_tbl_gt_file = tmp_path / "rf64_tbl_gt_file.wav"
    rf64_tbl_gt_file.write_bytes(
        _pack_rf64_wav(
            [(b"fmt ", _minimal_fmt_payload(), False)],
            extra_table=[(b"JUNK", 999_999)],
        )
    )
    with pytest.raises(ValueError, match="exceeds file size"):
        extract_wav_metadata_chunks(rf64_tbl_gt_file)

    # 9. RF64 ds64 NOT first chunk
    rf64_late_ds64 = tmp_path / "rf64_late_ds64.wav"
    fmt_bytes = _minimal_fmt_payload()
    ds64_ok = struct.pack("<QQQI", 100, 0, 0, 0)
    rf64_late_ds64.write_bytes(
        b"RF64\xff\xff\xff\xffWAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_bytes))
        + fmt_bytes
        + b"ds64"
        + struct.pack("<I", len(ds64_ok))
        + ds64_ok
    )
    with pytest.raises(ValueError, match="must be the first chunk"):
        extract_wav_metadata_chunks(rf64_late_ds64)

    # 10. RF64 valid ds64 size table entry using 0xFFFFFFFF sentinel
    rf64_valid_tbl = tmp_path / "rf64_valid_tbl.wav"
    rf64_valid_tbl.write_bytes(
        _pack_rf64_wav(
            [
                (b"fmt ", _minimal_fmt_payload(), False),
                (b"LIST", b"INFO1234", True),
                (b"data", b"\x00\x00\x00\x00", True),
            ],
            extra_table=[(b"LIST", 8)],
        )
    )
    assert extract_wav_metadata_chunks(rf64_valid_tbl) == {b"LIST": b"INFO1234"}

    # 11. RF64 oversized ds64 (> _MAX_DS64_CHUNK_BYTES DoS cap)
    rf64_big_ds64 = tmp_path / "rf64_big_ds64.wav"
    big_ds64 = b"\x00" * (delivery_mod._MAX_DS64_CHUNK_BYTES + 4)
    rf64_big_ds64.write_bytes(
        b"RF64\xff\xff\xff\xffWAVEds64" + struct.pack("<I", len(big_ds64)) + big_ds64
    )
    with pytest.raises(ValueError, match="safety cap"):
        extract_wav_metadata_chunks(rf64_big_ds64)

    # 12. Oversized metadata chunk (> MAX_METADATA_CHUNK_BYTES = 4 MiB DoS cap)
    oversized_wav = tmp_path / "oversized_meta.wav"
    oversized_payload = b"X" * (MAX_METADATA_CHUNK_BYTES + 8)
    oversized_wav.write_bytes(
        _pack_riff_wav(
            [
                (b"fmt ", _minimal_fmt_payload()),
                (b"data", b"\x00\x00\x00\x00"),
                (b"iXML", oversized_payload),
            ]
        )
    )
    with pytest.raises(ValueError, match="safety cap"):
        extract_wav_metadata_chunks(oversized_wav)

    small_wav = tmp_path / "small.wav"
    small_wav.write_bytes(
        _pack_riff_wav([(b"fmt ", _minimal_fmt_payload()), (b"data", b"\x00\x00\x00\x00")])
    )
    with pytest.raises(ValueError, match="safety cap"):
        inject_wav_metadata_chunks(small_wav, {b"bext": oversized_payload})

    # 13. Chunk-flood DoS (> _MAX_WAV_CHUNKS zero-size chunks)
    flood_wav = tmp_path / "flood.wav"
    flood_chunks = [(b"JUNK", b"") for _ in range(delivery_mod._MAX_WAV_CHUNKS + 2)]
    flood_wav.write_bytes(_pack_riff_wav(flood_chunks))
    with pytest.raises(ValueError, match="maximum allowed chunk count"):
        extract_wav_metadata_chunks(flood_wav)

    # 14. Odd-sized chunk padding boundary: odd bext, odd LIST, odd last chunk without EOF pad
    odd_wav = tmp_path / "odd_boundary.wav"
    odd_wav.write_bytes(
        _pack_riff_wav(
            [
                (b"fmt ", _minimal_fmt_payload()),
                (b"data", b"\x00\x00\x00\x00"),
                (b"JUNK", b"123"),  # odd-sized existing non-preserved chunk
                (b"iXML", b"<A/>!"),  # 5-byte odd last chunk without trailing pad byte
            ],
            pad_last=False,
        )
    )
    assert extract_wav_metadata_chunks(odd_wav) == {b"iXML": b"<A/>!"}

    # Inject odd-length bext (5 bytes -> goes after data) + custom LIST chunk + verify round-trip
    assert inject_wav_metadata_chunks(odd_wav, {b"bext": b"12345", b"LIST": b"abc"}) is True
    extracted_odd = extract_wav_metadata_chunks(odd_wav)
    assert extracted_odd[b"bext"] == b"12345"
    assert extracted_odd[b"LIST"] == b"abc"
    assert extracted_odd[b"iXML"] == b"<A/>!"


@pytest.mark.unit
def test_delivery_inject_errors_short_reads_and_windows_handle_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover inject error paths, RF64 odd ds64, short-read guards, and closed-handle os.replace."""
    wav_path = tmp_path / "base.wav"
    wav_path.write_bytes(
        _pack_riff_wav([(b"fmt ", _minimal_fmt_payload()), (b"data", b"\x00\x00\x00\x00")])
    )

    # Non-bytes payload
    with pytest.raises(ValueError, match="must be bytes"):
        inject_wav_metadata_chunks(wav_path, {b"iXML": "not_bytes"})  # type: ignore[dict-item]

    # WAV missing data chunk
    no_data = tmp_path / "no_data.wav"
    no_data.write_bytes(_pack_riff_wav([(b"fmt ", _minimal_fmt_payload())]))
    with pytest.raises(ValueError, match="no audio data chunk"):
        inject_wav_metadata_chunks(no_data, {b"iXML": b"<A/>"})

    # Classic RIFF 4 GiB limit exceeded -> cleans up temp file
    monkeypatch.setattr(delivery_mod, "_WAV_CLASSIC_LIMIT_BYTES", 32)
    with pytest.raises(OutputValidationError, match="classic 4 GiB RIFF limit"):
        inject_wav_metadata_chunks(wav_path, {b"iXML": b"<BWFXML>1234567890</BWFXML>"})
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []
    monkeypatch.undo()

    # RF64 with odd-length ds64 payload (29 bytes) exercises ds64 pad byte writer
    rf64_odd_ds64 = tmp_path / "rf64_odd_ds64.wav"
    fmt_b = _minimal_fmt_payload()
    ds64_29 = struct.pack("<QQQI", 100, 4, 0, 0) + b"\x00"
    rf64_odd_ds64.write_bytes(
        b"RF64\xff\xff\xff\xffWAVE"
        + b"ds64"
        + struct.pack("<I", 29)
        + ds64_29
        + b"\x00"
        + b"fmt "
        + struct.pack("<I", len(fmt_b))
        + fmt_b
        + b"data\x04\x00\x00\x00\x00\x00\x00\x00"
    )
    assert inject_wav_metadata_chunks(rf64_odd_ds64, {b"iXML": b"<OK/>"}) is True
    assert extract_wav_metadata_chunks(rf64_odd_ds64) == {b"iXML": b"<OK/>"}

    # _stream_copy_range unexpected EOF
    with pytest.raises(OutputValidationError, match="Unexpected EOF"):
        delivery_mod._stream_copy_range(io.BytesIO(b"ab"), io.BytesIO(), 0, 10)

    # _scan_wav_chunks short read when file shrinks mid-scan
    with pytest.raises(ValueError, match="Truncated WAV chunk header"):
        delivery_mod._scan_wav_chunks(io.BytesIO(b"RIFF\x20\x00\x00\x00WAVE"), 24)
    with pytest.raises(ValueError, match="Truncated RF64 ds64"):
        delivery_mod._scan_wav_chunks(
            io.BytesIO(b"RF64\xff\xff\xff\xffWAVEds64\x1c\x00\x00\x00\x00\x00"), 48
        )

    # Windows handle safety: verify ALL open file handles are closed BEFORE os.replace!
    open_handles: list[Any] = []
    real_open = open
    real_replace = os.replace

    def _tracking_open(*args: Any, **kwargs: Any) -> Any:
        fh = real_open(*args, **kwargs)
        open_handles.append(fh)
        return fh

    replace_checked = {"ok": False}

    def _checking_replace(src: Any, dst: Any) -> None:
        assert all(fh.closed for fh in open_handles), "File handle still open during os.replace!"
        replace_checked["ok"] = True
        real_replace(src, dst)

    with (
        patch("builtins.open", side_effect=_tracking_open),
        patch("hawavoclean.audio.delivery.os.replace", side_effect=_checking_replace),
    ):
        assert inject_wav_metadata_chunks(wav_path, {b"bext": b"1234"}) is True
    assert replace_checked["ok"] is True


@pytest.mark.unit
def test_delivery_loudness_and_export_edge_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover remaining loudness, LRA, path validation, soundfile, and FFmpeg error branches."""
    sr = 48000
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # _validate_audio errors
        with pytest.raises(ValueError, match="sample_rate"):
            compute_loudness_range_lu(np.zeros(10, dtype=np.float32), 0)
        with pytest.raises(ValueError, match="1D or 2D"):
            compute_loudness_range_lu(np.zeros((1, 2, 3), dtype=np.float32), sr)
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_loudness_range_lu(np.array([1e100, 0.0], dtype=np.float64), sr)

        # LRA edge branches:
        # 1) Exact 3.0 s length -> only 1 block (starts.size < 2) -> 0.0
        assert compute_loudness_range_lu(np.ones(3 * sr, dtype=np.float32) * 0.2, sr) == 0.0

        # 2) 6-channel surround audio (hits ch >= len(LOUDNESS_CHANNEL_GAINS))
        t6 = np.arange(4 * sr, dtype=np.float64) / sr
        tone6 = (0.1 * np.sin(2.0 * np.pi * 500.0 * t6)).astype(np.float32)
        surround = np.tile(tone6[np.newaxis, :], (6, 1))
        assert compute_loudness_range_lu(surround, sr) >= 0.0

        # 3) Single loud burst at very start so only 1 block survives relative gate (keep < 2)
        #    Crafted: 3.1 s where first 0.05 s is loud, rest is near -75 LUFS; or 10 s with 1 spike.
        one_rel = np.full(int(3.2 * sr), np.float32(1e-3), dtype=np.float32)
        one_rel[: int(0.05 * sr)] = (
            0.9 * np.sin(2.0 * np.pi * 1000.0 * t6[: int(0.05 * sr)])
        ).astype(np.float32)
        assert compute_loudness_range_lu(one_rel, sr) >= 0.0

        # 4) match_input when input_lufs equals pre.integrated_lufs (abs(gain_db) < 1e-6)
        tone = (0.1 * np.sin(2.0 * np.pi * 1000.0 * t6[:sr])).astype(np.float32)
        _, m0 = apply_delivery_loudness(tone, sr, "match_input")
        out_eq, m_eq = apply_delivery_loudness(
            tone, sr, "match_input", input_lufs=float(m0["integrated_lufs"])
        )
        assert float(m_eq["applied_gain_db"]) == 0.0
        assert np.array_equal(out_eq, tone)

        # 5) Extreme float32 amplitude with gain does not emit overflow warnings
        huge_tone = (np.sin(2.0 * np.pi * 1000.0 * t6[:sr]) * 1e37).astype(np.float32)
        out_huge, _ = apply_delivery_loudness(huge_tone, sr, "match_input", input_lufs=-5.0)
        assert np.all(np.isfinite(out_huge))

    # Export edge cases & error branches
    master = tmp_path / "master.wav"
    sf.write(str(master), np.zeros((256, 1), dtype=np.float32), sr, subtype="PCM_24")

    # Invalid paths
    with pytest.raises(ValueError, match="Invalid master_wav"):
        export_delivery_artifact(Path("bad\x00master.wav"), tmp_path / "out.wav", fmt="wav24")
    with pytest.raises(ValueError, match="Invalid dest_path"):
        export_delivery_artifact(master, Path("."), fmt="wav24")

    # Overwrite source_media_path in place & missing source_media_path for video_mux
    src_mp4 = tmp_path / "src.mp4"
    src_mp4.write_bytes(b"fake")
    with pytest.raises(ValueError, match="overwrite source_media_path"):
        export_delivery_artifact(master, src_mp4, fmt="video_mux", source_media_path=src_mp4)
    with pytest.raises(FileNotFoundError, match="Source media not found"):
        export_delivery_artifact(
            master, tmp_path / "out.mp4", fmt="video_mux", source_media_path=tmp_path / "nope.mp4"
        )

    # Empty 0-frame WAV master -> OutputValidationError
    empty_wav = tmp_path / "empty_frames.wav"
    sf.write(str(empty_wav), np.empty((0, 1), dtype=np.float32), sr, subtype="PCM_24")
    with pytest.raises(OutputValidationError, match="empty or invalid"):
        export_delivery_artifact(empty_wav, tmp_path / "out_empty.wav", fmt="wav24")

    # NaN samples in FLOAT WAV master -> OutputValidationError
    nan_wav = tmp_path / "nan_master.wav"
    nan_arr = np.array([[0.0], [float("nan")], [0.1]], dtype=np.float32)
    sf.write(str(nan_wav), nan_arr, sr, subtype="FLOAT")
    with pytest.raises(OutputValidationError, match="NaN or Inf"):
        export_delivery_artifact(nan_wav, tmp_path / "out_nan.wav", fmt="wav24")

    # Missing FFmpeg -> PreflightError for aac and video_mux
    with patch("hawavoclean.audio.delivery.ffmpeg_bin_path", return_value=None):
        with pytest.raises(PreflightError, match="AAC export"):
            export_delivery_artifact(master, tmp_path / "out.m4a", fmt="aac")
        with pytest.raises(PreflightError, match="video mux-back"):
            export_delivery_artifact(
                master, tmp_path / "out.mp4", fmt="video_mux", source_media_path=src_mp4
            )

    # _run_ffmpeg_checked timeout and non-zero exit
    with (
        patch(
            "hawavoclean.audio.delivery.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1.0),
        ),
        pytest.raises(OutputValidationError, match="timed out"),
    ):
        delivery_mod._run_ffmpeg_checked(["ffmpeg"], timeout_s=1.0, label="test")

    with (
        patch(
            "hawavoclean.audio.delivery.subprocess.run",
            side_effect=subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], stderr=b"boom"),
        ),
        pytest.raises(OutputValidationError, match="boom"),
    ):
        delivery_mod._run_ffmpeg_checked(["ffmpeg"], timeout_s=1.0, label="test")

    # AAC with .aac extension (no +faststart), video_mux with preserve_metadata=False & .mkv,
    # and empty artifact validation
    with (
        patch("hawavoclean.audio.delivery.ffmpeg_bin_path", return_value="/usr/bin/ffmpeg"),
        patch("hawavoclean.audio.delivery._run_ffmpeg_checked") as mock_ff,
    ):
        # Mock produces empty output -> OutputValidationError
        with pytest.raises(OutputValidationError, match="empty artifact"):
            export_delivery_artifact(master, tmp_path / "out.aac", fmt="aac")
        assert "-movflags" not in mock_ff.call_args[0][0]

        def _write_dummy(cmd: list[str], **_kw: Any) -> None:
            Path(cmd[-1]).write_bytes(b"MUXED_BYTES")

        mock_ff.side_effect = _write_dummy
        out_mkv = export_delivery_artifact(
            master,
            tmp_path / "out.mkv",
            fmt="video_mux",
            source_media_path=src_mp4,
            preserve_metadata=False,
        )
        assert out_mkv.read_bytes() == b"MUXED_BYTES"
        assert "-map_metadata" not in mock_ff.call_args[0][0]

    # TOCTOU symlink check right before os.replace
    real_is_reparse = is_reparse_or_symlink
    dest_race = tmp_path / "race.wav"
    calls = {"n": 0}

    def _toctou_reparse(p: Path) -> bool:
        if p.name == "race.wav":
            calls["n"] += 1
            return calls["n"] >= 3
        return real_is_reparse(p)

    monkeypatch.setattr(delivery_mod, "is_reparse_or_symlink", _toctou_reparse)
    with pytest.raises(ValueError, match="symlink"):
        export_delivery_artifact(master, dest_race, fmt="wav24")
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# Part 3: multitrack.py iXML / Poly-WAV / Mid-Side Torture
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multitrack_ixml_hostile_xml_and_binary_torture(tmp_path: Path) -> None:
    """Reject XML bombs, XXE, >1 MiB iXML, malformed UTF-8, and corrupt chunks gracefully."""
    sr = 48000
    mono = np.zeros((480, 2), dtype=np.float32)

    # 1. Billion Laughs XML bomb -> rejected safely before XML parser, falls back to Track 1/2
    bomb_wav = tmp_path / "bomb.wav"
    sf.write(str(bomb_wav), mono, sr, subtype="PCM_24")
    billion_laughs = (
        b'<?xml version="1.0"?>'
        b"<!DOCTYPE lolz ["
        b'<!ENTITY lol "lol">'
        b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
        b"]><BWFXML><TRACK_LIST><TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX>"
        b"<NAME>&lol3;</NAME></TRACK></TRACK_LIST></BWFXML>"
    )
    _append_raw_chunk_to_wav(bomb_wav, b"iXML", billion_laughs)
    infos_bomb = inspect_polywav_tracks(bomb_wav)
    assert [i.name for i in infos_bomb] == ["Track 1", "Track 2"]

    # 2. XXE external entity attack -> rejected safely
    xxe_wav = tmp_path / "xxe.wav"
    sf.write(str(xxe_wav), mono, sr, subtype="PCM_24")
    xxe = (
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        b"<BWFXML><TRACK_LIST><TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX>"
        b"<NAME>&xxe;</NAME></TRACK></TRACK_LIST></BWFXML>"
    )
    _append_raw_chunk_to_wav(xxe_wav, b"iXML", xxe)
    assert [i.name for i in inspect_polywav_tracks(xxe_wav)] == ["Track 1", "Track 2"]

    # 3. Oversized > 1 MiB iXML chunk -> skipped cleanly without parsing
    big_ixml_wav = tmp_path / "big_ixml.wav"
    sf.write(str(big_ixml_wav), mono, sr, subtype="PCM_24")
    big_xml = (
        b"<BWFXML><TRACK_LIST><TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX><NAME>Boom</NAME></TRACK>"
        b"</TRACK_LIST><PAD>" + b"A" * (MAX_IXML_CHUNK_BYTES + 64) + b"</PAD></BWFXML>"
    )
    _append_raw_chunk_to_wav(big_ixml_wav, b"iXML", big_xml)
    assert [i.name for i in inspect_polywav_tracks(big_ixml_wav)] == ["Track 1", "Track 2"]

    # 4. Malformed UTF-8, XML-illegal control bytes, & truncated XML -> graceful fallback
    bad_utf8_wav = tmp_path / "bad_utf8.wav"
    sf.write(str(bad_utf8_wav), mono, sr, subtype="PCM_24")
    _append_raw_chunk_to_wav(bad_utf8_wav, b"iXML", b"<BWFXML>\xff\xfe<BROKEN")
    assert [i.name for i in inspect_polywav_tracks(bad_utf8_wav)] == ["Track 1", "Track 2"]

    bad_ctrl_wav = tmp_path / "bad_ctrl.wav"
    sf.write(str(bad_ctrl_wav), mono, sr, subtype="PCM_24")
    _append_raw_chunk_to_wav(
        bad_ctrl_wav,
        b"iXML",
        b"<BWFXML><TRACK_LIST><TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX>"
        b"<NAME>Lav\x07\x01</NAME></TRACK></TRACK_LIST></BWFXML>",
    )
    assert [i.name for i in inspect_polywav_tracks(bad_ctrl_wav)] == ["Track 1", "Track 2"]

    # 5. Mixed valid & invalid <TRACK> entries -> valid track still extracted!
    mixed_wav = tmp_path / "mixed_ixml.wav"
    sf.write(str(mixed_wav), mono, sr, subtype="PCM_24")
    mixed_xml = (
        b"<BWFXML><TRACK_LIST>"
        b"<TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX></TRACK>"
        b"<TRACK><CHANNEL_INDEX>not_int</CHANNEL_INDEX><NAME>BadIdx</NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>0</CHANNEL_INDEX><NAME>ZeroIdx</NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>99</CHANNEL_INDEX><NAME>HugeIdx</NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX><NAME>   </NAME></TRACK>"
        b"<TRACK><CHANNEL_INDEX>2</CHANNEL_INDEX><NAME>Lav\t\n_Main</NAME></TRACK>"
        b"</TRACK_LIST></BWFXML>"
    )
    _append_raw_chunk_to_wav(mixed_wav, b"iXML", mixed_xml)
    assert [i.name for i in inspect_polywav_tracks(mixed_wav)] == ["Track 1", "Lav_Main"]

    # 6. RF64 with ds64 + 0xFFFFFFFF data chunk + iXML AFTER data -> properly extracts names!
    rf64_wav = tmp_path / "rf64_ixml.wav"
    sf.write(str(rf64_wav), mono, sr, subtype="PCM_24", format="RF64")
    ixml_ok = (
        b"<BWFXML><TRACK_LIST>"
        b"<TRACK><CHANNEL_INDEX>1</CHANNEL_INDEX><NAME>RF64_Boom</NAME></TRACK>"
        b"</TRACK_LIST></BWFXML>"
    )
    inject_wav_metadata_chunks(rf64_wav, {b"iXML": ixml_ok})
    assert [i.name for i in inspect_polywav_tracks(rf64_wav)] == ["RF64_Boom", "Track 2"]

    # 7. _read_ixml_track_names direct binary edge cases
    tiny = tmp_path / "tiny.bin"
    tiny.write_bytes(b"abc")
    assert multitrack_mod._read_ixml_track_names(tiny) == {}

    non_wav = tmp_path / "not_wav.bin"
    non_wav.write_bytes(b"NOT_A_WAV_FILE_123456")
    assert multitrack_mod._read_ixml_track_names(non_wav) == {}

    riff_ff = tmp_path / "riff_ff.wav"
    riff_ff.write_bytes(b"RIFF\x20\x00\x00\x00WAVEdata\xff\xff\xff\xff\x00\x00")
    assert multitrack_mod._read_ixml_track_names(riff_ff) == {}

    rf64_ff = tmp_path / "rf64_ff.wav"
    rf64_ff.write_bytes(b"RF64\xff\xff\xff\xffWAVEdata\xff\xff\xff\xff\x00\x00")
    assert multitrack_mod._read_ixml_track_names(rf64_ff) == {}

    trunc_chunk = tmp_path / "trunc_chunk.wav"
    trunc_chunk.write_bytes(b"RIFF\x20\x00\x00\x00WAVEfmt \x99\x00\x00\x00\x01\x02")
    assert multitrack_mod._read_ixml_track_names(trunc_chunk) == {}

    # RF64 with ds64 size table entry for JUNK + iXML after it
    rf64_tbl = tmp_path / "rf64_tbl.wav"
    rf64_tbl.write_bytes(
        _pack_rf64_wav(
            [
                (b"JUNK", b"12345678", True),
                (b"iXML", ixml_ok, False),
            ],
            extra_table=[(b"JUNK", 8)],
        )
    )
    assert multitrack_mod._read_ixml_track_names(rf64_tbl) == {0: "RF64_Boom"}

    # Empty iXML payload
    assert multitrack_mod._parse_ixml_xml(b"\x00\x00") == {}


@pytest.mark.unit
def test_multitrack_polywav_and_mid_side_adversarial(tmp_path: Path) -> None:
    """Torture Poly-WAV edge cases (0-ch, 1-sample, 0-sample, low-SR, NaN, dupes) & Mid/Side."""
    sr = 48000
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # 1. 1-sample Poly-WAV (4 channels, 1 sample)
        p1 = tmp_path / "one_sample.wav"
        data1 = np.array([[0.25, -0.5, 0.0, 0.1]], dtype=np.float32)
        sf.write(str(p1), data1, sr, subtype="PCM_24")
        infos1 = inspect_polywav_tracks(p1)
        assert len(infos1) == 4
        assert all(i.samples == 1 for i in infos1)
        # Duplicate track indices -> writes once, returns requested multiplicity
        extracted_dup = extract_polywav_tracks(p1, [0, 0, 2], tmp_path / "dup_out")
        assert len(extracted_dup) == 3
        assert extracted_dup[0] == extracted_dup[1]
        assert extracted_dup[0].is_file() and extracted_dup[2].is_file()

        # 2. 0-sample Poly-WAV
        p0 = tmp_path / "zero_samples.wav"
        sf.write(str(p0), np.empty((0, 2), dtype=np.float32), sr, subtype="PCM_24")
        infos0 = inspect_polywav_tracks(p0)
        assert len(infos0) == 2 and all(i.samples == 0 and i.duration_s == 0.0 for i in infos0)

        # 3. Low sample rate (600 Hz -> nyquist=300 <= 400 -> bypasses speech bandpass)
        p_low = tmp_path / "low_sr.wav"
        t_low = np.arange(120, dtype=np.float64) / 600.0
        low_sig = (0.3 * np.sin(2.0 * math.pi * 100.0 * t_low)).astype(np.float32)
        sf.write(str(p_low), low_sig[:, np.newaxis], 600, subtype="PCM_24")
        infos_low = inspect_polywav_tracks(p_low)
        assert len(infos_low) == 1 and infos_low[0].sample_rate == 600

        # 4. Multi-block remainder crossing POLY_STREAM_BLOCK_FRAMES (48000) -> 48100 samples
        #    Choose 48100 where frame_len = 960 -> block1 leaves 48000 % 960 = 0, so use 48010 at 44100 Hz
        #    (frame_len = 882; 48000 % 882 = 372 remainder carried into block 2!)
        p_rem = tmp_path / "rem_stream.wav"
        n_rem = 49000
        sr_rem = 44100
        t_rem = np.arange(n_rem, dtype=np.float64) / sr_rem
        rem_sig = (0.3 * np.sin(2.0 * math.pi * 800.0 * t_rem)).astype(np.float32)
        sf.write(str(p_rem), rem_sig[:, np.newaxis], sr_rem, subtype="PCM_24")
        infos_rem = inspect_polywav_tracks(p_rem)
        assert infos_rem[0].samples == n_rem and infos_rem[0].likely_dialogue is True

        # 5. NaN in FLOAT Poly-WAV -> inspect & extract both fail closed + clean up temp files!
        p_nan = tmp_path / "nan_poly.wav"
        nan_poly = np.array([[0.1, 0.2], [float("nan"), 0.3]], dtype=np.float32)
        sf.write(str(p_nan), nan_poly, sr, subtype="FLOAT")
        with pytest.raises(ValueError, match="NaN or Inf"):
            inspect_polywav_tracks(p_nan)
        nan_out = tmp_path / "nan_out"
        with pytest.raises(ValueError, match="NaN or Inf"):
            extract_polywav_tracks(p_nan, [0, 1], nan_out)
        assert list(nan_out.iterdir()) == []

        # 6. Truncated WAV -> libsndfile rejects cleanly
        trunc_wav = tmp_path / "trunc.wav"
        trunc_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        with pytest.raises(sf.LibsndfileError):
            inspect_polywav_tracks(trunc_wav)

        # 7. 0-channel WAV (raw header) -> rejected; mocked 0-channel & invalid sr -> ValueError
        zero_ch_wav = tmp_path / "zero_ch.wav"
        zero_ch_wav.write_bytes(
            _pack_riff_wav([(b"fmt ", _minimal_fmt_payload(channels=0)), (b"data", b"")])
        )
        with pytest.raises((sf.LibsndfileError, ValueError)):
            inspect_polywav_tracks(zero_ch_wav)

        fake_reader = MagicMock()
        fake_reader.__enter__.return_value = fake_reader
        fake_reader.channels = 0
        fake_reader.samplerate = 48000
        with (
            patch("hawavoclean.audio.multitrack.sf.SoundFile", return_value=fake_reader),
            pytest.raises(ValueError, match="outside supported range"),
        ):
            inspect_polywav_tracks(p1)

        fake_reader.channels = 2
        fake_reader.samplerate = 0
        with patch("hawavoclean.audio.multitrack.sf.SoundFile", return_value=fake_reader):
            with pytest.raises(ValueError, match="Invalid sample_rate"):
                inspect_polywav_tracks(p1)
            with pytest.raises(ValueError, match="Invalid sample_rate"):
                extract_polywav_tracks(p1, [0], tmp_path / "bad_sr")

        # 8. track_indices out-of-range / negative / bool
        bad_index_cases: list[list[int]] = [[-1], [4], [True]]
        for bad_indices in bad_index_cases:
            with pytest.raises(ValueError, match="out of range"):
                extract_polywav_tracks(p1, bad_indices, tmp_path / "bad_idx")

        # 8b. Windows handle safety: verify reader & all writers are closed BEFORE os.replace
        opened_sf: list[sf.SoundFile] = []
        real_sf_cls = sf.SoundFile
        real_replace = os.replace
        replace_count = {"n": 0}

        def _tracking_sf(*args: Any, **kwargs: Any) -> sf.SoundFile:
            inst = real_sf_cls(*args, **kwargs)
            opened_sf.append(inst)
            return inst

        def _checking_replace(src: Any, dst: Any) -> None:
            assert all(h.closed for h in opened_sf), (
                "SoundFile handle still open during os.replace!"
            )
            replace_count["n"] += 1
            real_replace(src, dst)

        with (
            patch("hawavoclean.audio.multitrack.sf.SoundFile", side_effect=_tracking_sf),
            patch("hawavoclean.audio.multitrack.os.replace", side_effect=_checking_replace),
        ):
            extract_polywav_tracks(p1, [0, 1], tmp_path / "win_safe_out")
        assert replace_count["n"] == 2

        # 9. Mid/Side adversarial: NaN/Inf, float64 overflow, 3e38 extreme, bool side_attenuation
        with pytest.raises(ValueError, match="finite"):
            encode_mid_side(np.array([1e100]), np.array([1e100]))
        with pytest.raises(ValueError, match="finite"):
            decode_mid_side(np.array([1e100]), np.array([0.0]))
        with pytest.raises(ValueError, match="finite"):
            enhance_true_stereo_ms(np.full((2, 8), 1e100, dtype=np.float64), lambda m: m)
        with pytest.raises(ValueError, match="NaN or Inf"):
            enhance_true_stereo_ms(
                np.ones((2, 8), dtype=np.float32),
                lambda m: np.full(m.shape, 1e100, dtype=np.float64),
            )
        with pytest.raises(ValueError, match="side_attenuation"):
            enhance_true_stereo_ms(
                np.zeros((2, 8), dtype=np.float32), lambda m: m, side_attenuation=True
            )
        with pytest.raises(ValueError, match="shape"):
            enhance_true_stereo_ms(np.zeros((1, 8), dtype=np.float32), lambda m: m)
        with pytest.raises(ValueError, match="shape"):
            enhance_true_stereo_ms(np.zeros(8, dtype=np.float32), lambda m: m)

        # Empty 1D encode/decode
        m_e, s_e = encode_mid_side(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        assert m_e.size == 0 and s_e.size == 0
        l_e, r_e = decode_mid_side(m_e, s_e)
        assert l_e.size == 0 and r_e.size == 0

        # Extreme 3e38 decode & enhance stay finite with 0 warnings
        l_hi, r_hi = decode_mid_side(
            np.array([3e38], dtype=np.float32), np.array([3e38], dtype=np.float32)
        )
        assert np.all(np.isfinite(l_hi)) and np.all(np.isfinite(r_hi))
        out_hi = enhance_true_stereo_ms(
            np.array([[3e38, -3e38], [-3e38, 3e38]], dtype=np.float32),
            lambda m: np.full_like(m, np.float32(3e38)),
            side_attenuation=0.0,
        )
        assert out_hi.shape == (2, 2) and np.all(np.isfinite(out_hi))
