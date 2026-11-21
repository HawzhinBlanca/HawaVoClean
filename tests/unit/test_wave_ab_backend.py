"""Comprehensive unit tests proving Waves A + B backend fixes and regressions."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.signal
import soundfile as sf
from pydantic import ValidationError

from hawavoclean.audio.decode import (
    WINDOW_PREROLL_S,
    decode_audio_to_memmap,
    decode_audio_window,
)
from hawavoclean.audio.delivery import (
    DELIVERY_PRESETS,
    apply_delivery_loudness,
    apply_delivery_loudness_streaming,
    export_delivery_artifact,
    extract_wav_metadata_chunks,
    inject_wav_metadata_chunks,
    measure_delivery_loudness_streaming,
)
from hawavoclean.audio.multitrack import denoise_side_channel
from hawavoclean.audio.probe import probe_audio
from hawavoclean.config import FinishingConfig
from hawavoclean.enhancement.dereverb_neural import NeuralDereverberator
from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.finishing.deess import apply_split_band_deesser
from hawavoclean.finishing.detect import detect_defects
from hawavoclean.finishing.dynamics import apply_dialogue_leveler
from hawavoclean.finishing.limiter import (
    apply_lookahead_limiter,
    apply_lookahead_limiter_to_memmap,
)
from hawavoclean.finishing.loudness import measure_loudness_and_peaks
from hawavoclean.finishing.pro_repair import (
    apply_dynamic_deplosive,
    apply_mouth_declick,
)
from hawavoclean.finishing.repair import remove_electrical_hum
from hawavoclean.finishing.safe_finish import apply_finishing_stages
from hawavoclean.guard import nisqa_arch
from hawavoclean.pipeline import run_pipeline
from hawavoclean.runtime import process_current_rss_bytes
from hawavoclean.segmentation.utterances import find_lowest_energy_zero_crossing
from hawavoclean.server.app import _run_delivery_export
from hawavoclean.server.contracts import (
    ManualStrategyV1,
    RegionPreviewRequestV1,
    SmartSafeStrategyV1,
)
from hawavoclean.server.jobs import JobRecord, default_command
from hawavoclean.server.unit_restitch import (
    load_job_restitch_meta,
    restitch_master_with_overrides,
    save_job_restitch_meta,
    save_unit_slice_to_dir,
)
from hawavoclean.server.warm_worker import (
    RegionPreviewRequest,
    WarmCoreCache,
    render_region_preview,
)
from hawavoclean.smart_safe import preview as smart_safe_preview


def _speech_like(
    duration_s: float = 1.6,
    sr: int = 48000,
    amp: float = 0.25,
    seed: int = 7,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    env = 0.5 * (1.0 + np.sin(2.0 * np.pi * 3.0 * t)) * 0.8 + 0.2
    sig = (
        0.55 * np.sin(2.0 * np.pi * 210.0 * t)
        + 0.30 * np.sin(2.0 * np.pi * 620.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 1450.0 * t)
        + 0.03 * rng.standard_normal(n)
    ) * env
    sig = sig / max(float(np.max(np.abs(sig))), 1e-6) * amp
    return np.ascontiguousarray(sig, dtype=np.float32)


def _band_rms(
    x: np.ndarray[Any, np.dtype[np.float32]], sr: int, lo_hz: float, hi_hz: float
) -> float:
    sos = scipy.signal.butter(4, [lo_hz, hi_hz], btype="bandpass", fs=sr, output="sos")
    y = scipy.signal.sosfiltfilt(sos, x.astype(np.float64))
    return float(np.sqrt(np.mean(y * y) + 1e-18))


def _low_rms(x: np.ndarray[Any, np.dtype[np.float32]], sr: int, cutoff_hz: float) -> float:
    sos = scipy.signal.butter(4, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    y = scipy.signal.sosfiltfilt(sos, x.astype(np.float64))
    return float(np.sqrt(np.mean(y * y) + 1e-18))


class _IdentityEnhancer:
    def __init__(self, *, delay_samples: int = 0, gain: float = 1.0) -> None:
        self._delay = delay_samples
        self._gain = gain
        self.metadata = EnhancerMetadata(
            core_id="identity-test-v1",
            version="1.0",
            algorithm="identity",
            sample_rate=48000,
            phase_coherent=True,
            params_hash="0" * 64,
        )

    def warmup(self) -> None:
        return None

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        out = waveform.astype(np.float32, copy=True) * np.float32(self._gain)
        if self._delay > 0 and out.size > self._delay:
            shifted = np.zeros_like(out)
            shifted[self._delay :] = out[: -self._delay]
            out = shifted
        return EnhancementResult(
            waveform=np.ascontiguousarray(out, dtype=np.float32),
            sample_rate=sample_rate,
            model_runtime_ms=1.0,
            input_samples=int(waveform.shape[-1]),
            output_samples=int(out.shape[-1]),
        )


# =============================================================================
# WAVE A.1 — HPF End-to-End Wiring & Default Bit-Identity
# =============================================================================


@pytest.mark.unit
def test_hpf_contracts_validation_and_job_command_forwarding(tmp_path: Path) -> None:
    m = ManualStrategyV1.model_validate(
        {"kind": "manual", "route": "production", "hpfEnabled": True, "hpfCutoffHz": 120.0}
    )
    assert m.hpf_enabled is True
    assert m.hpf_cutoff_hz == 120.0

    s = SmartSafeStrategyV1.model_validate(
        {
            "kind": "smart_safe",
            "hpfEnabled": False,
            "hpfCutoffHz": 85.0,
            "deliveryPreset": "dialogue_gated",
            "ambienceFloorDb": -28.0,
            "stereoMode": "true_stereo_ms",
            "proRepair": {"declip": True},
        }
    )
    assert s.hpf_enabled is False
    assert s.hpf_cutoff_hz == 85.0
    assert s.delivery_preset == "dialogue_gated"
    assert s.stereo_mode == "true_stereo_ms"

    p = RegionPreviewRequestV1.model_validate(
        {
            "path": str(tmp_path / "a.wav"),
            "startS": 0.0,
            "endS": 1.0,
            "hpfEnabled": True,
            "hpfCutoffHz": 140.0,
            "deliveryPreset": "dialogue_gated",
            "proRepair": {"declip": True, "deess": True},
            "stereoMode": "true_stereo_ms",
        }
    )
    assert p.hpf_cutoff_hz == 140.0
    assert p.delivery_preset == "dialogue_gated"

    # Reject out-of-range and bool values
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "hpfCutoffHz": 10.0}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "hpfCutoffHz": 300.0}
        )
    with pytest.raises(ValidationError):
        ManualStrategyV1.model_validate(
            {"kind": "manual", "route": "production", "hpfCutoffHz": True}
        )

    # JobRecord -> default_command forwards HPF flags for both natural & smart_safe
    rec_manual = JobRecord(
        job_id="j1",
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.hawavoclean.json",
        profile="production",
        overwrite=True,
        mode="natural",
        hpf_enabled=True,
        hpf_cutoff_hz=110.0,
    )
    cmd_manual = default_command(rec_manual)
    assert "--hpf-cutoff-hz" in cmd_manual
    assert cmd_manual[cmd_manual.index("--hpf-cutoff-hz") + 1] == "110.0"

    rec_no_hpf = JobRecord(
        job_id="j2",
        input_path=tmp_path / "in.wav",
        output_path=tmp_path / "out.wav",
        report_path=tmp_path / "out.hawavoclean.json",
        profile="production",
        overwrite=True,
        mode="smart_safe",
        hpf_enabled=False,
        delivery_preset="dialogue_gated",
        stereo_mode="true_stereo_ms",
        ambience_floor_db=-30.0,
    )
    cmd_smart = default_command(rec_no_hpf)
    assert "--no-hpf" in cmd_smart
    assert "--delivery-preset" in cmd_smart
    assert "dialogue_gated" in cmd_smart
    assert "--stereo-mode" in cmd_smart
    assert "true_stereo_ms" in cmd_smart
    assert "--ambience-floor-db" in cmd_smart


@pytest.mark.unit
def test_hpf_default_none_bit_identical_and_override_cuts_rumble() -> None:
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    sig = (0.3 * np.sin(2 * np.pi * 60.0 * t) + 0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(
        np.float32
    )
    cfg = FinishingConfig()

    out_default, stages_default = apply_finishing_stages(sig, sr, cfg, "gentle")
    out_none, stages_none = apply_finishing_stages(
        sig, sr, cfg, "gentle", hpf_enabled=None, hpf_cutoff_hz=None
    )
    assert np.array_equal(out_default, out_none)
    assert stages_default == stages_none

    out_160, stages_160 = apply_finishing_stages(
        sig, sr, cfg, "gentle", hpf_enabled=True, hpf_cutoff_hz=160.0
    )
    out_off, _ = apply_finishing_stages(sig, sr, cfg, "gentle", hpf_enabled=False)

    assert any(s.startswith("dc_subsonic_removed(cutoff=160Hz)") for s in stages_160)
    assert _low_rms(out_160, sr, 90.0) < 0.25 * _low_rms(out_off, sr, 90.0)


# =============================================================================
# WAVE A.2 — WYSIWYG Preview (GCC-PHAT, HPF, Pro-Repair) + Current-RSS Cache
# =============================================================================


@pytest.mark.unit
def test_process_current_rss_and_warm_cache_eviction_recovery() -> None:
    rss = process_current_rss_bytes()
    assert isinstance(rss, int) and rss > 0

    # Simulate a spike above budget followed by a drop back below budget:
    # WarmCoreCache should evict on spike, then keep warm entries once RSS recovers.
    rss_values = [500_000_000, 100_000_000, 100_000_000]
    idx = 0

    def fake_rss() -> int:
        nonlocal idx
        val = rss_values[min(idx, len(rss_values) - 1)]
        idx += 1
        return val

    loads = 0

    def core_factory(_core_id: str, _sr: int) -> Any:
        nonlocal loads
        loads += 1
        return _IdentityEnhancer(gain=float(loads))

    cache = WarmCoreCache(
        max_cores=2,
        memory_limit_mb=200,
        core_factory=core_factory,
        rss_bytes_fn=fake_rss,
    )

    first, hit1 = cache.get_or_load("core-a")
    assert not hit1
    # First entry stays because RSS check runs before loading/evicting older entries
    # Now load a second core while RSS is still spiked -> evicts oldest on memory pressure
    cache2 = WarmCoreCache(
        max_cores=2,
        memory_limit_mb=200,
        core_factory=core_factory,
        rss_bytes_fn=lambda: 500_000_000,
    )
    cache2.get_or_load("core-a")
    cache2.get_or_load("core-b")
    assert cache2.stats()["cached_count"] <= 1
    assert cache2.stats()["memory_evictions"] >= 1

    # Once RSS drops below budget (100 MB < 200 MB), warm hits succeed cleanly!
    second, hit2 = cache.get_or_load("core-a")
    assert hit2 and second is first


@pytest.mark.unit
def test_render_region_preview_wysiwyg_gcc_phat_hpf_and_pro_repair(tmp_path: Path) -> None:
    sr = 48000
    t = np.arange(int(1.2 * sr), dtype=np.float64) / sr
    # Speech with 55 Hz rumble + crisp 1 kHz tone (sensitive to comb filtering if misaligned)
    mono = (
        0.25 * np.sin(2.0 * np.pi * 55.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 1000.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 2400.0 * t)
    ).astype(np.float32)
    stereo = np.stack([mono, (mono * 0.7).astype(np.float32)], axis=0)
    src = tmp_path / "preview_src.wav"
    sf.write(str(src), stereo.T, sr, subtype="FLOAT")

    # Enhancer introduces a 24-sample (0.5 ms) delay — an exact 180-degree phase
    # inversion at 1000 Hz! Without GCC-PHAT delay alignment, 50% wet/dry blend cancels 1000 Hz.
    delayed_enhancer = _IdentityEnhancer(delay_samples=24)
    cache = WarmCoreCache(max_cores=2, core_factory=lambda _cid, _sr: delayed_enhancer)
    cache.get_or_load("wiener-dd-48k-v1")

    req = RegionPreviewRequest(
        source_path=src,
        start_s=0.1,
        end_s=1.1,
        profile="production",
        core_id="wiener-dd-48k-v1",
        strength=0.5,
        hpf_enabled=True,
        hpf_cutoff_hz=140.0,
        pro_repair={"declip": True, "deess": True, "level_rider": True},
        stereo_mode="true_stereo_ms",
        delivery_preset="dialogue_gated",
        output_dir=tmp_path / "previews",
    )
    res = render_region_preview(req, cache=cache)
    assert res.warm_hit is True
    assert res.channels == 2
    out_data, out_sr = sf.read(str(res.preview_path), dtype="float32", always_2d=True)
    assert out_sr == sr
    out_ch0 = out_data[:, 0]

    # 1) HPF at 140 Hz strongly attenuated 55 Hz rumble relative to 1 kHz signal
    rumble_rms = _low_rms(out_ch0, sr, 90.0)
    tone_rms = _band_rms(out_ch0, sr, 850.0, 1150.0)
    assert rumble_rms < 0.25 * tone_rms

    # 2) GCC-PHAT alignment prevented 1 kHz comb cancellation on 50% wet/dry blend
    assert tone_rms > 0.005

    # 3) True stereo preserved (L != R)
    assert float(np.max(np.abs(out_data[:, 0] - out_data[:, 1]))) > 1e-3


# =============================================================================
# WAVE A.3 — Pre-Enhancement Declip + Fast `-ss` Before `-i` + Mono os.replace
# =============================================================================


@pytest.mark.unit
def test_pre_enhancement_declip_runs_before_enhancer_in_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 48000
    t = np.arange(int(1.2 * sr), dtype=np.float64) / sr
    clean = (1.45 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    clipped = np.clip(clean, -1.0, 1.0).astype(np.float32)
    src = tmp_path / "clipped.wav"
    sf.write(str(src), clipped, sr, subtype="FLOAT")

    seen_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    class _SpyEnhancer(_IdentityEnhancer):
        def __init__(self, **_kw: Any) -> None:
            super().__init__()

        def enhance(
            self,
            waveform: np.ndarray[Any, np.dtype[np.float32]],
            sample_rate: int,
        ) -> EnhancementResult:
            seen_waveforms.append(waveform.copy())
            return super().enhance(waveform, sample_rate)

    import dataclasses

    import hawavoclean.pipeline as pl
    from hawavoclean.enhancement.factory import resolve_core as real_resolve_core

    def fake_resolve(core_id: str) -> Any:
        reg = real_resolve_core(core_id)
        return dataclasses.replace(reg, enhancer_class=_SpyEnhancer)

    monkeypatch.setattr(pl, "resolve_core", fake_resolve)

    cfg_toml = tmp_path / "inline.toml"
    cfg_toml.write_text("[runtime]\nisolated_worker = false\n", encoding="utf-8")

    out_wav = tmp_path / "out_declip.wav"
    run_pipeline(
        input_path=src,
        output_path=out_wav,
        profile="production",
        config_path=cfg_toml,
        overwrite=True,
        pro_repair={"declip": True},
    )
    assert out_wav.is_file()
    assert seen_waveforms, "Expected enhancer to receive audio"
    # Because declip ran PRE-enhancement, the waveform arriving at the enhancer already
    # has flat-top clipping plateaus repaired and 3rd-harmonic (900 Hz) THD sharply reduced.
    in_h3_ratio = _band_rms(clipped, sr, 850.0, 950.0) / _band_rms(clipped, sr, 250.0, 350.0)
    seen_wave = seen_waveforms[0]
    seen_h3_ratio = _band_rms(seen_wave, sr, 850.0, 950.0) / _band_rms(seen_wave, sr, 250.0, 350.0)
    assert seen_h3_ratio < 0.6 * in_h3_ratio


@pytest.mark.unit
def test_decode_window_fast_input_seek_and_mono_os_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 48000
    sig = _speech_like(duration_s=3.0, sr=sr, amp=0.3)
    src = tmp_path / "long_mono.wav"
    sf.write(str(src), sig, sr, subtype="FLOAT")
    probe = probe_audio(src)

    recorded_cmds: list[list[str]] = []
    real_run = subprocess.run

    def spy_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(cmd, list) and cmd and "ffmpeg" in str(cmd[0]):
            recorded_cmds.append([str(c) for c in cmd])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)
    win = decode_audio_window(probe, start_s=1.5, end_s=2.0)
    assert recorded_cmds, "Expected ffmpeg invocation"
    cmd = recorded_cmds[-1]
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx, f"Expected fast input seek -ss before -i, got {cmd}"
    assert math.isclose(float(cmd[ss_idx + 1]), 1.5 - WINDOW_PREROLL_S, abs_tol=1e-4)

    expected = sig[int(1.5 * sr) : int(2.0 * sr)]
    assert win.data.shape[-1] == expected.shape[0]
    assert float(np.max(np.abs(win.data[0] - expected))) < 1e-4

    # Mono decode_audio_to_memmap uses O(1) os.replace instead of double-copying
    replace_calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src_p: Any, dst_p: Any) -> None:
        replace_calls.append((str(src_p), str(dst_p)))
        real_replace(src_p, dst_p)

    monkeypatch.setattr(os, "replace", spy_replace)
    mm_dest = tmp_path / "mono.f32"
    mm = decode_audio_to_memmap(probe, mm_dest)
    assert mm.data.shape[0] == 1
    assert any(dst == str(mm_dest) for _, dst in replace_calls)


# =============================================================================
# WAVE A.4 — Streaming Delivery/Pro-Repair/M-S, SmartSafe Parity, Time-Range, Restitch
# =============================================================================


@pytest.mark.unit
def test_streaming_natural_uses_streaming_delivery_and_chunked_ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 48000
    left = _speech_like(duration_s=1.4, sr=sr, amp=0.25, seed=1)
    right = _speech_like(duration_s=1.4, sr=sr, amp=0.18, seed=2)
    src = tmp_path / "stereo_stream.wav"
    sf.write(str(src), np.stack([left, right], axis=1), sr, subtype="FLOAT")

    import hawavoclean.audio.delivery as delivery_mod
    import hawavoclean.pipeline as pl

    streaming_called = False
    real_streaming = delivery_mod.apply_delivery_loudness_streaming

    def spy_streaming(*args: Any, **kwargs: Any) -> dict[str, float | str]:
        nonlocal streaming_called
        streaming_called = True
        return real_streaming(*args, **kwargs)

    monkeypatch.setattr(pl, "NATURAL_STREAMING_THRESHOLD_BYTES", 1)
    monkeypatch.setattr(pl, "apply_delivery_loudness_streaming", spy_streaming)

    out_wav = tmp_path / "stream_out.wav"
    report = run_pipeline(
        input_path=src,
        output_path=out_wav,
        profile="production",
        overwrite=True,
        delivery_preset="podcast",
        stereo_mode="true_stereo_ms",
        ambience_floor_db=-32.0,
        pro_repair={"deplosive": True, "mouth_declick": True, "deess": True},
        hpf_enabled=True,
        hpf_cutoff_hz=100.0,
    )
    assert out_wav.is_file()
    assert streaming_called is True
    assert report.output.channels == 2
    out_data, _ = sf.read(str(out_wav), dtype="float32", always_2d=True)
    # True stereo M/S stays stereo (L != R)
    assert float(np.max(np.abs(out_data[:, 0] - out_data[:, 1]))) > 1e-3


@pytest.mark.unit
def test_smart_safe_honors_delivery_hpf_loudness_stats_and_native_44k(
    tmp_path: Path,
) -> None:
    sr = 44100
    sig = _speech_like(duration_s=1.4, sr=sr, amp=0.2)
    t = np.arange(sig.size, dtype=np.float64) / sr
    sig = np.ascontiguousarray(sig + 0.15 * np.sin(2.0 * np.pi * 55.0 * t), dtype=np.float32)
    src = tmp_path / "smart_44k.wav"
    sf.write(str(src), sig, sr, subtype="FLOAT")

    out_wav = tmp_path / "smart_out.wav"
    report = run_pipeline(
        input_path=src,
        output_path=out_wav,
        profile="production",
        mode="smart_safe",
        overwrite=True,
        hpf_enabled=True,
        hpf_cutoff_hz=150.0,
        delivery_preset="podcast",
        pro_repair={"deess": True},
    )
    assert out_wav.is_file()
    assert report.input.integrated_lufs is not None
    assert report.input.true_peak_dbtp is not None
    assert report.output.integrated_lufs is not None
    assert report.output.true_peak_dbtp is not None
    # Delivery preset "podcast" targets -19 LUFS for mono (-16 LUFS stereo)
    assert abs(float(report.output.integrated_lufs) - (-19.0)) < 2.5

    selected_route = (report.restoration or {}).get("selected_route")
    if selected_route == "preserve":
        assert report.output.sample_rate == 44100


@pytest.mark.unit
def test_time_range_blend_no_loudness_jump_and_bit_identical_exterior(
    tmp_path: Path,
) -> None:
    from hawavoclean import cli as cli_mod

    sr = 48000
    # Quiet -30 LUFS voice memo so naive pipeline normalization to -16 LUFS would cause +14 dB jump
    sig = _speech_like(duration_s=2.0, sr=sr, amp=0.05)
    src = tmp_path / "quiet_memo.wav"
    sf.write(str(src), sig, sr, subtype="FLOAT")
    out_wav = tmp_path / "spliced.wav"

    args = argparse.Namespace(
        input=str(src),
        output=str(out_wav),
        profile="production",
        config=None,
        overwrite=True,
        quiet=True,
        verbose=False,
        original_input_path=None,
    )
    cli_mod._run_natural_edit_blend(
        args=args,
        strength=1.0,
        time_range=(0.5, 1.5),
        clean_only=False,
        on_progress=None,
        allow_research_restore=False,
    )
    out_data, out_sr = sf.read(str(out_wav), dtype="float32")
    assert out_sr == sr
    assert out_data.shape[0] == sig.shape[0]

    # Exterior [0.0..0.4s] and [1.6..2.0s] stays bit-identical (within 24-bit WAV quantization)
    assert float(np.max(np.abs(out_data[: int(0.4 * sr)] - sig[: int(0.4 * sr)]))) < 1e-5
    assert float(np.max(np.abs(out_data[int(1.6 * sr) :] - sig[int(1.6 * sr) :]))) < 1e-5

    # Interior [0.6..1.4s] RMS stays matched to exterior (no ±10 dB step jump!)
    ext_rms = float(np.sqrt(np.mean(out_data[: int(0.4 * sr)] ** 2)))
    int_rms = float(np.sqrt(np.mean(out_data[int(0.6 * sr) : int(1.4 * sr)] ** 2)))
    jump_db = abs(20.0 * math.log10((int_rms + 1e-12) / (ext_rms + 1e-12)))
    assert jump_db < 2.5


@pytest.mark.unit
def test_unit_restitch_preserves_true_stereo_ms_finishing_delivery_and_bwf(
    tmp_path: Path,
) -> None:
    sr = 48000
    n = sr  # 1.0 s
    left = _speech_like(duration_s=1.0, sr=sr, amp=0.22, seed=11)
    right = _speech_like(duration_s=1.0, sr=sr, amp=0.14, seed=22)
    mid_orig = 0.5 * (left + right)
    side_orig = 0.5 * (left - right)
    mid_enh = mid_orig * np.float32(0.85)
    side_enh = denoise_side_channel(side_orig, sr)

    out_wav = tmp_path / "restitched.wav"
    sf.write(str(out_wav), np.stack([left, right], axis=1), sr, subtype="PCM_24")
    bwf_chunks = {b"bext": b"BEXT_TEST_PAYLOAD_01", b"iXML": b"<BWFXML/>"}
    inject_wav_metadata_chunks(out_wav, bwf_chunks)

    slice_dir = tmp_path / "slices"
    unit_meta = {
        "unit_id": 0,
        "channel": 0,
        "is_speech": True,
        "start_sample": 0,
        "end_sample": n,
        "guard_decision": "ENHANCED",
        "final_decision": "ENHANCED",
    }
    save_unit_slice_to_dir(
        slice_dir,
        unit_id=0,
        original=mid_orig.astype(np.float32),
        candidate=mid_enh.astype(np.float32),
        unit_meta=unit_meta,
        channel=0,
        side_original=side_orig.astype(np.float32),
        side_candidate=side_enh.astype(np.float32),
    )
    meta = {
        "stereo_mode": "true_stereo_ms",
        "delivery_preset": "podcast",
        "hpf_enabled": True,
        "hpf_cutoff_hz": 100.0,
        "finishing_enabled": True,
    }
    save_job_restitch_meta(slice_dir, meta)
    assert load_job_restitch_meta(slice_dir) == meta

    res = restitch_master_with_overrides(
        job_dir=slice_dir,
        unit_records=[unit_meta],
        overrides={0: ("force_original", None)},
        master_output_path=out_wav,
        sample_rate=sr,
        channels=2,
    )
    assert res["summary"]["reverted"] == 1
    assert out_wav.is_file()
    out_data, _ = sf.read(str(out_wav), dtype="float32", always_2d=True)
    # 1) Never collapsed to mono and Right channel is genuinely reconstructed from Mid/Side (never zeros)
    assert out_data.shape[1] == 2
    assert float(np.max(np.abs(out_data[:, 0] - out_data[:, 1]))) > 1e-3
    assert float(np.sqrt(np.mean(out_data[:, 1] ** 2))) > 1e-3
    reconstructed_side = 0.5 * (out_data[:, 0] - out_data[:, 1])
    side_corr = float(
        np.corrcoef(reconstructed_side.astype(np.float64), side_orig.astype(np.float64))[0, 1]
    )
    assert side_corr > 0.75, (
        f"expected true_stereo_ms Side reconstruction, got corr={side_corr:.3f}"
    )
    # 2) BWF/iXML chunks preserved
    extracted = extract_wav_metadata_chunks(out_wav)
    assert extracted.get(b"bext") == b"BEXT_TEST_PAYLOAD_01"
    assert extracted.get(b"iXML") == b"<BWFXML/>"


# =============================================================================
# WAVE B.5 — Parabolic De-Hum (60 Hz 3rd/4th/6th Harmonics + Clean Identity)
# =============================================================================


@pytest.mark.unit
def test_parabolic_dehum_snaps_60hz_notches_high_harmonics_and_preserves_clean() -> None:
    sr = 48000
    t = np.arange(2 * sr, dtype=np.float64) / sr
    speech = _speech_like(duration_s=2.0, sr=sr, amp=0.18, seed=42)
    # Exact 60.0 Hz hum with strong 3rd (180), 4th (240), and 6th (360) harmonics
    hum = (
        0.04 * np.sin(2.0 * np.pi * 60.0 * t)
        + 0.03 * np.sin(2.0 * np.pi * 180.0 * t)
        + 0.025 * np.sin(2.0 * np.pi * 240.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 360.0 * t)
    )
    noisy = np.ascontiguousarray(speech + hum, dtype=np.float32)

    defects = detect_defects(noisy, sr)
    assert defects.has_hum is True
    assert defects.hum_freq_hz == 60.0

    cleaned = remove_electrical_hum(noisy, sr, hum_freq_hz=defects.hum_freq_hz)
    # Verify >12 dB attenuation at 3rd (180 Hz), 4th (240 Hz), and 6th (360 Hz) harmonics
    for harmonic_hz in (180.0, 240.0, 360.0):
        before = _band_rms(noisy, sr, harmonic_hz - 4.0, harmonic_hz + 4.0)
        after = _band_rms(cleaned, sr, harmonic_hz - 4.0, harmonic_hz + 4.0)
        atten_db = 20.0 * math.log10((before + 1e-12) / (after + 1e-12))
        assert atten_db > 12.0, (
            f"Expected >12 dB attenuation at {harmonic_hz} Hz, got {atten_db:.1f}"
        )

    # Clean speech with no hum stays has_hum=False
    clean_defects = detect_defects(speech, sr)
    assert clean_defects.has_hum is False


# =============================================================================
# WAVE B.6 — Envelope-Smoothed De-Ess & Level-Relative Pro Finishing (-34 LUFS)
# =============================================================================


@pytest.mark.unit
def test_envelope_smoothed_deess_and_quiet_voice_memo_pro_finishing() -> None:
    sr = 48000
    n = sr
    t = np.arange(n, dtype=np.float64) / sr

    # Quiet -34 LUFS voice memo (amp ~ 0.02) with a harsh 7 kHz sibilant burst + low-freq plosive + click
    low_voice = 0.020 * np.sin(2.0 * np.pi * 220.0 * t)
    sib_env = np.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    sibilant = 0.025 * sib_env * np.sin(2.0 * np.pi * 7200.0 * t)
    quiet_memo = np.ascontiguousarray(low_voice + sibilant, dtype=np.float32)

    # 1) De-esser adapts to quiet -34 LUFS memo and applies real 10ms/50ms envelope smoothing
    deessed_fast, _ = apply_split_band_deesser(quiet_memo, sr, attack_ms=0.1, release_ms=0.1)
    deessed_smooth, gr_smooth = apply_split_band_deesser(
        quiet_memo, sr, attack_ms=10.0, release_ms=50.0
    )
    assert gr_smooth > 0.5, f"Quiet -34 LUFS sibilant should trigger de-ess, got {gr_smooth}"
    assert not np.allclose(deessed_fast, deessed_smooth)

    # 2) Dialogue leveler adapts to quiet -34 LUFS speech (does not ignore it)
    uneven_quiet = np.concatenate(
        [
            (0.012 * np.sin(2.0 * np.pi * 240.0 * t[: sr // 2])).astype(np.float32),
            (0.036 * np.sin(2.0 * np.pi * 240.0 * t[sr // 2 :])).astype(np.float32),
        ]
    )
    _, leveler_gr = apply_dialogue_leveler(uneven_quiet, sr)
    assert leveler_gr > 0.5

    # 3) De-plosive catches a quiet -34 LUFS plosive thump
    plosive = 0.045 * np.exp(-0.5 * ((t - 0.35) / 0.015) ** 2) * np.sin(2.0 * np.pi * 70.0 * t)
    quiet_plosive = np.ascontiguousarray(low_voice + plosive, dtype=np.float32)
    deplosed, n_plosives = apply_dynamic_deplosive(quiet_plosive, sr)
    assert n_plosives >= 1
    assert _low_rms(deplosed, sr, 110.0) < _low_rms(quiet_plosive, sr, 110.0)

    # 4) Mouth de-click catches a quiet -34 LUFS saliva click
    quiet_click = low_voice.astype(np.float32).copy()
    click_idx = sr // 2
    quiet_click[click_idx : click_idx + 3] += np.array([0.04, -0.05, 0.03], dtype=np.float32)
    declicked, n_clicks = apply_mouth_declick(quiet_click, sr)
    assert n_clicks >= 1
    assert float(np.max(np.abs(declicked[click_idx - 2 : click_idx + 5]))) < float(
        np.max(np.abs(quiet_click[click_idx - 2 : click_idx + 5]))
    )


# =============================================================================
# WAVE B.7 — 48 kHz Dereverb >8 kHz HF Air Preservation + Real Warmup + M/S Side
# =============================================================================


@pytest.mark.unit
def test_dereverb_48k_preserves_hf_air_and_real_warmup() -> None:
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    # Speech with both core band (400 Hz) and natural consonant air (11 kHz)
    sig = (
        0.25 * np.sin(2.0 * np.pi * 400.0 * t) + 0.08 * np.sin(2.0 * np.pi * 11000.0 * t)
    ).astype(np.float32)

    runner_rms: list[float] = []

    def mock_16k_runner(
        chunk_16k: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        runner_rms.append(float(np.sqrt(np.mean(chunk_16k.astype(np.float64) ** 2))))
        return chunk_16k.copy()

    derev = NeuralDereverberator(enable_bwe=False, chunk_runner=mock_16k_runner)

    # 1) warmup() must pass a non-zero test tone so MPS kernels actually compile
    derev.warmup()
    assert runner_rms and runner_rms[0] > 1e-4

    # 2) 48 kHz dereverb preserves >8 kHz consonant air instead of brick-walling at 8 kHz
    res = derev.dereverb(sig, sample_rate=sr)
    assert res.sample_rate == sr
    in_hf = _band_rms(sig, sr, 9500.0, 12500.0)
    out_hf = _band_rms(res.waveform, sr, 9500.0, 12500.0)
    assert out_hf > 0.5 * in_hf, f"Expected >8 kHz air preserved: in={in_hf:.5f}, out={out_hf:.5f}"


@pytest.mark.unit
def test_denoise_side_channel_reduces_stationary_noise() -> None:
    sr = 48000
    rng = np.random.default_rng(99)
    n = sr
    t = np.arange(n, dtype=np.float64) / sr
    # Brief directional speech burst + stationary hiss on Side channel
    burst = np.where((t >= 0.35) & (t <= 0.65), 0.15 * np.sin(2.0 * np.pi * 500.0 * t), 0.0)
    noise = 0.02 * rng.standard_normal(n)
    side = np.ascontiguousarray(burst + noise, dtype=np.float32)
    cleaned = denoise_side_channel(side, sr)
    noise_before = float(np.sqrt(np.mean(side[: int(0.25 * sr)] ** 2)))
    noise_after = float(np.sqrt(np.mean(cleaned[: int(0.25 * sr)] ** 2)))
    assert noise_after < 0.65 * noise_before


# =============================================================================
# WAVE B.8 — Dialogue-Gated Loudness, True match_input Export & Dithered wav24
# =============================================================================


@pytest.mark.unit
def test_dialogue_gated_loudness_true_match_input_and_dithered_wav24(tmp_path: Path) -> None:
    sr = 48000
    assert "dialogue_gated" in DELIVERY_PRESETS
    assert DELIVERY_PRESETS["dialogue_gated"].target_lufs_mono == -27.0
    assert DELIVERY_PRESETS["dialogue_gated"].target_lufs_stereo == -27.0
    assert DELIVERY_PRESETS["dialogue_gated"].true_peak_ceiling_dbtp == -2.0
    assert DELIVERY_PRESETS["dialogue_gated"].dialogue_gated is True
    assert DELIVERY_PRESETS["atsc_a85"].dialogue_gated is True

    # Construct 3s clip: 1.5s loud 60 Hz rumble pause + 1.5s speech at 800 Hz
    t_half = np.arange(int(1.5 * sr), dtype=np.float64) / sr
    rumble_pause = (0.35 * np.sin(2.0 * np.pi * 60.0 * t_half)).astype(np.float32)
    speech_part = _speech_like(duration_s=1.5, sr=sr, amp=0.12)
    combo = np.concatenate([rumble_pause, speech_part])[None, :]

    m_std = measure_loudness_and_peaks(combo, sr, dialogue_gated=False)
    m_dg = measure_loudness_and_peaks(combo, sr, dialogue_gated=True)
    m_speech_only = measure_loudness_and_peaks(speech_part[None, :], sr, dialogue_gated=True)
    assert abs(m_dg.integrated_lufs - m_speech_only.integrated_lufs) < 1.5
    assert abs(m_dg.integrated_lufs - m_std.integrated_lufs) > 0.3

    # Apply dialogue_gated preset (in-memory & streaming agree)
    _, metrics = apply_delivery_loudness(combo, sr, "dialogue_gated")
    assert abs(float(metrics["integrated_lufs"]) - (-27.0)) < 1.5
    assert float(metrics["true_peak_dbtp"]) <= -1.9

    src_wav = tmp_path / "combo.wav"
    stream_out = tmp_path / "combo_stream.wav"
    sf.write(str(src_wav), combo.T, sr, subtype="FLOAT")
    s_metrics = apply_delivery_loudness_streaming(src_wav, stream_out, "dialogue_gated")
    assert abs(float(s_metrics["integrated_lufs"]) - (-27.0)) < 1.5
    assert float(s_metrics["true_peak_dbtp"]) <= -1.9
    stream_data, _ = sf.read(str(stream_out), dtype="float32", always_2d=True)
    m_stream = measure_loudness_and_peaks(stream_data.T, sr, dialogue_gated=True)
    assert abs(m_stream.integrated_lufs - float(s_metrics["integrated_lufs"])) < 0.2

    # True match_input export: quiet source (-28 LUFS) vs loud master (-14 LUFS) -> exported matches source
    quiet_src = tmp_path / "quiet_source.wav"
    loud_master = tmp_path / "loud_master.wav"
    sf.write(str(quiet_src), _speech_like(duration_s=1.5, sr=sr, amp=0.05), sr, subtype="FLOAT")
    sf.write(str(loud_master), _speech_like(duration_s=1.5, sr=sr, amp=0.35), sr, subtype="FLOAT")

    src_lufs = measure_delivery_loudness_streaming(quiet_src)[0].integrated_lufs
    dest_matched = tmp_path / "matched_export.wav"
    exp_res = _run_delivery_export(
        master_resolved=loud_master,
        dest_resolved=dest_matched,
        fmt="wav24",
        delivery_preset="match_input",
        source_resolved=quiet_src,
        preserve_metadata=False,
    )
    assert dest_matched.is_file()
    assert abs(float(exp_res["integratedLufs"]) - src_lufs) < 1.5

    # wav24 export applies deterministic TPDF dither (low-level DC isn't identically zero-truncated)
    tiny_master = tmp_path / "tiny.wav"
    tiny_sig = np.full(sr // 10, 1e-8, dtype=np.float32)
    sf.write(str(tiny_master), tiny_sig, sr, subtype="FLOAT")
    dithered_wav = tmp_path / "dithered24.wav"
    export_delivery_artifact(tiny_master, dithered_wav, fmt="wav24", preserve_metadata=False)
    raw_i32, _ = sf.read(str(dithered_wav), dtype="int32")
    lsb_codes = raw_i32 >> 8
    assert int(np.ptp(lsb_codes)) >= 1


# =============================================================================
# WAVE B.9 — 3x Speedup & Caching (NISQA Cache, SmartSafe Route Cache, Vectorized Limiter & ZC)
# =============================================================================


@pytest.mark.unit
def test_nisqa_weights_verification_cache_and_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_weights = tmp_path / "nisqa_v2.tar"
    payload = b"mock-nisqa-weights-v1"
    fake_weights.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(nisqa_arch, "NISQA_V2_WEIGHTS_SHA256", digest)
    nisqa_arch._VERIFY_CACHE.clear()

    from hawavoclean.hashing import hash_file as real_hash_file

    hash_calls = 0

    def spy_hash_file(path: Path) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return real_hash_file(path)

    monkeypatch.setattr(nisqa_arch, "hash_file", spy_hash_file)

    ok1, _ = nisqa_arch.verify_nisqa_weights(fake_weights)
    ok2, _ = nisqa_arch.verify_nisqa_weights(fake_weights)
    ok3, _ = nisqa_arch.verify_nisqa_weights(fake_weights)
    assert ok1 and ok2 and ok3
    assert hash_calls == 1, f"Expected 1 SHA-256 read with cache, got {hash_calls}"

    # Modify file content (size changes) -> cache invalidates and rejects mismatch!
    fake_weights.write_bytes(payload + b"-corrupted")
    ok_corrupt, msg_corrupt = nisqa_arch.verify_nisqa_weights(fake_weights)
    assert ok_corrupt is False
    assert "mismatch" in msg_corrupt.lower()
    assert hash_calls == 2


@pytest.mark.unit
def test_smart_safe_preview_caches_enhancer_instances() -> None:
    smart_safe_preview._ROUTE_ENHANCER_CACHE.clear()
    created = 0

    class _DummyA:
        def __init__(self, device: str = "cpu") -> None:
            nonlocal created
            created += 1
            self.device = device

    class _DummyB:
        def __init__(self, device: str = "cpu") -> None:
            nonlocal created
            created += 1
            self.device = device

    a1: object = smart_safe_preview._get_cached_enhancer(_DummyA, device="cpu")
    a2: object = smart_safe_preview._get_cached_enhancer(_DummyA, device="cpu")
    b1: object = smart_safe_preview._get_cached_enhancer(_DummyB, device="cpu")
    assert a1 is a2
    assert b1 is not a1
    assert created == 2


@pytest.mark.unit
def test_vectorized_memmap_limiter_and_zero_crossing_search(tmp_path: Path) -> None:
    sr = 48000
    rng = np.random.default_rng(123)
    wav = (0.3 * rng.standard_normal((2, sr))).astype(np.float32)
    # Inject hot peaks across chunk boundaries
    wav[:, 10000:10200] *= 5.0
    wav[:, 24000:24300] *= 4.0

    in_mem = apply_lookahead_limiter(wav, sr, ceiling_dbtp=-1.0, release_ms=50.0)

    mm_path = tmp_path / "lim.f32"
    apply_lookahead_limiter_to_memmap(
        wav, sr, mm_path, ceiling_dbtp=-1.0, release_ms=50.0, chunk_samples=16384
    )
    mm_out = np.memmap(mm_path, dtype=np.float32, mode="r", shape=wav.shape)
    assert np.allclose(np.asarray(mm_out), in_mem.limited_waveform, atol=1e-5)

    # O(N) find_lowest_energy_zero_crossing locates the quiet dip
    t = np.arange(sr, dtype=np.float64) / sr
    tone = (0.4 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
    tone[20000:22000] *= 0.02  # quiet dip around sample 21000
    zc = find_lowest_energy_zero_crossing(tone, target_sample=21000, search_window_samples=4000)
    assert 20000 <= zc <= 22000
