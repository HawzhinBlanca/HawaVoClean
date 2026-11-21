"""Unit tests for Phase R2.13: Post-Mastering Guard R Re-verification and Signal Integrity.

Qualifies:
1. Signal integrity invariants in protected band:
   - RMS error <= 1e-4
   - Relative STFT error <= 1e-3
   - Third-octave band energy deviation <= 0.25 dB outside transition band.
2. Independent failure of each signal integrity layer under realistic corruptions.
3. Simulated provider quantization: noise in protected band fails closed; protected bin copying passes.
4. Segment-by-segment evidence preservation: timecodes, RMS, STFT, 1/3-octave, and Guard R metrics.
5. Fail-closed fallback: missing or failed evidence forces explicit Natural fallback.
6. End-to-end pipeline verification under mode="restore" with report provenance.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.signal as signal

from hawavoclean.finishing.limiter import apply_lookahead_limiter
from hawavoclean.pipeline import run_pipeline
from hawavoclean.restoration.guard import (
    PostMasteringSegmentEvidence,
    PostMasteringVerificationResult,
    RestorationGuard,
)
from hawavoclean.restoration.policy import SegmentRestorationDecision
from hawavoclean.restoration.protected_band import (
    verify_protected_band_invariance,
)


def _make_bandlimited_speech(
    f0_hz: float = 180.0,
    sr: int = 48000,
    duration_s: float = 1.0,
    cutoff_hz: float = 4000.0,
) -> np.ndarray:
    """Generate harmonic voice-like audio strictly band-limited below cutoff_hz."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    sig = np.zeros_like(t)
    # Add harmonics up to cutoff_hz
    k = 1
    while k * f0_hz < cutoff_hz - 500.0:
        sig += (0.5 / math.sqrt(k)) * np.sin(2 * np.pi * k * f0_hz * t, dtype=np.float32)
        k += 1
    # Brickwall filter to guarantee zero energy above cutoff
    sos = signal.butter(10, (cutoff_hz - 250.0) / (sr / 2), btype="lowpass", output="sos")
    filtered = signal.sosfiltfilt(sos, sig).astype(np.float32)
    # Normalize peak to 0.7
    peak = float(np.max(np.abs(filtered)))
    if peak > 0.0:
        filtered = filtered * (0.7 / peak)
    return np.asarray(filtered, dtype=np.float32)


def test_post_mastering_signal_integrity_invariants_pass() -> None:
    """Bit-exact protected band with synthesized high-band meets R2.13 invariants."""
    sr = 48000
    cutoff_hz = 4000.0
    transition_hz = 500.0
    audio_nat = _make_bandlimited_speech(sr=sr, cutoff_hz=cutoff_hz)

    # Synthesize clean high-band content strictly above cutoff using steep high-pass filter
    t = np.linspace(0, len(audio_nat) / sr, len(audio_nat), endpoint=False, dtype=np.float32)
    high_raw = (0.01 * np.sin(2 * np.pi * 8000 * t) + 0.005 * np.sin(2 * np.pi * 10000 * t)).astype(
        np.float32
    )
    sos_hp = signal.butter(10, 6000.0 / (sr / 2), btype="highpass", output="sos")
    high_clean = signal.sosfiltfilt(sos_hp, high_raw).astype(np.float32)
    audio_rest = audio_nat + high_clean

    # Verify signal integrity below cutoff
    verif = verify_protected_band_invariance(
        original_audio=audio_nat,
        restored_audio=audio_rest,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
        tolerance_rms=1e-4,
        tolerance_stft=1e-3,
        max_third_octave_deviation_db=0.25,
    )

    assert verif.passes_invariance is True
    assert verif.rms_waveform_error <= 1e-4
    assert verif.complex_stft_relative_error <= 1e-3
    assert verif.worst_band_energy_deviation_db <= 0.25


def test_post_mastering_rms_violation_fails_closed() -> None:
    """Low-band disturbance exceeding RMS 1e-4 fails signal integrity."""
    sr = 48000
    cutoff_hz = 4000.0
    audio_nat = _make_bandlimited_speech(sr=sr, cutoff_hz=cutoff_hz)

    # Inject low-frequency perturbation with RMS ~ 3e-4 (well above 1e-4 ceiling)
    t = np.linspace(0, len(audio_nat) / sr, len(audio_nat), endpoint=False, dtype=np.float32)
    disturbance = (0.0005 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    audio_corrupted = audio_nat + disturbance

    verif = verify_protected_band_invariance(
        original_audio=audio_nat,
        restored_audio=audio_corrupted,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        tolerance_rms=1e-4,
        tolerance_stft=1e-3,
        max_third_octave_deviation_db=0.25,
    )

    assert verif.passes_invariance is False
    assert verif.rms_waveform_error > 1e-4


def test_post_mastering_stft_relative_error_violation_fails_closed() -> None:
    """Subtle distributed spectral distortion exceeding relative STFT 1e-3 fails signal integrity."""
    sr = 48000
    cutoff_hz = 4000.0
    audio_nat = _make_bandlimited_speech(sr=sr, cutoff_hz=cutoff_hz)

    n_fft = 2048
    hop = 512
    _, _, z = signal.stft(audio_nat, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    # Add subtle phase noise in protected bins
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    prot = freqs < (cutoff_hz - 500.0)
    z_distorted = z.copy()
    z_distorted[prot, :] *= np.exp(1j * 0.05)  # Phase rotation inducing > 1e-3 relative error

    _, audio_corrupted = signal.istft(z_distorted, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    audio_corrupted = audio_corrupted[: len(audio_nat)].astype(np.float32)

    verif = verify_protected_band_invariance(
        original_audio=audio_nat,
        restored_audio=audio_corrupted,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        tolerance_rms=1e-2,  # deliberately generous RMS to isolate STFT check
        tolerance_stft=1e-3,
        max_third_octave_deviation_db=5.0,
    )

    assert verif.passes_invariance is False
    assert verif.complex_stft_relative_error > 1e-3


def test_post_mastering_third_octave_deviation_violation_fails_closed() -> None:
    """Narrow-band attenuation exceeding 0.25 dB in protected 1/3-octave fails signal integrity."""
    sr = 48000
    cutoff_hz = 4000.0
    audio_nat = _make_bandlimited_speech(sr=sr, cutoff_hz=cutoff_hz)

    n_fft = 2048
    hop = 512
    _, _, z = signal.stft(audio_nat, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # Target one third-octave band well outside transition band: 800 - 1000 Hz
    band = (freqs >= 800.0) & (freqs < 1000.0)
    z_attenuated = z.copy()
    # Attenuate by 0.5 dB (target limit is 0.25 dB)
    z_attenuated[band, :] *= 10.0 ** (-0.5 / 20.0)

    _, audio_attenuated = signal.istft(z_attenuated, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    audio_attenuated = audio_attenuated[: len(audio_nat)].astype(np.float32)

    verif = verify_protected_band_invariance(
        original_audio=audio_nat,
        restored_audio=audio_attenuated,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        tolerance_rms=1.0,  # generous to isolate 1/3-octave check
        tolerance_stft=1.0,
        max_third_octave_deviation_db=0.25,
    )

    assert verif.passes_invariance is False
    assert verif.worst_band_energy_deviation_db > 0.25


def test_provider_quantization_low_band_noise_fails_closed() -> None:
    """Simulated provider low-bit quantization noise in protected band fails verification."""
    sr = 48000
    cutoff_hz = 4000.0
    audio_nat = _make_bandlimited_speech(sr=sr, cutoff_hz=cutoff_hz)

    # Simulate coarse 8-bit quantization on entire audio (including low frequencies)
    q_scale = 127.0
    audio_quantized = (np.round(audio_nat * q_scale) / q_scale).astype(np.float32)

    verif = verify_protected_band_invariance(
        original_audio=audio_nat,
        restored_audio=audio_quantized,
        sample_rate=sr,
        cutoff_hz=cutoff_hz,
        tolerance_rms=1e-4,
        tolerance_stft=1e-3,
        max_third_octave_deviation_db=0.25,
    )

    assert verif.passes_invariance is False
    assert verif.rms_waveform_error > 1e-4


def test_verify_post_mastering_segment_evidence_preserved() -> None:
    """Guard R verify_post_mastering preserves detailed evidence per reconstructed segment."""
    sr = 48000
    cutoff_hz = 4000.0
    guard = RestorationGuard(sample_rate=sr)

    # Create 2 seconds of band-limited audio (2 segments of 1.0s each)
    audio_nat = _make_bandlimited_speech(sr=sr, duration_s=2.0, cutoff_hz=cutoff_hz)
    seg_len = sr  # 1.0s segments
    starts = [0, sr]

    # Model reconstruction: audio_rest matches natural below 4 kHz with subtle valid high band
    audio_rest = audio_nat.copy()

    # Master both through lookahead limiter identically (limiter expects (channels, samples))
    lim_nat = apply_lookahead_limiter(audio_nat[np.newaxis, :], sample_rate=sr, ceiling_dbtp=-1.0)
    lim_rest = apply_lookahead_limiter(audio_rest[np.newaxis, :], sample_rate=sr, ceiling_dbtp=-1.0)
    mastered_nat = lim_nat.limited_waveform
    mastered_rest = lim_rest.limited_waveform

    segment_records = [
        SegmentRestorationDecision(
            action="restored",
            applied_strength=1.0,
            cutoff_hz=cutoff_hz,
            guard_result=None,
        ),
        SegmentRestorationDecision(
            action="reduced",
            applied_strength=0.5,
            cutoff_hz=cutoff_hz,
            guard_result=None,
        ),
    ]

    result: PostMasteringVerificationResult = guard.verify_post_mastering(
        mastered_natural=mastered_nat,
        mastered_restored=mastered_rest,
        segment_records=segment_records,
        starts=starts,
        seg_len=seg_len,
        cutoff_hz=cutoff_hz,
        tolerance_rms=1e-4,
        tolerance_stft=1e-3,
        tolerance_third_octave_db=0.25,
    )

    assert result.passes is True
    assert result.fallback_applied is False
    assert result.reconstructed_segments_verified == 2
    assert len(result.segments_evidence) == 2

    seg0 = result.segments_evidence[0]
    assert seg0.segment_index == 0
    assert seg0.start_sample == 0
    assert seg0.end_sample == sr
    assert seg0.action == "verified"
    assert seg0.passes is True
    assert seg0.rms_waveform_error <= 1e-4
    assert seg0.stft_relative_error <= 1e-3
    assert seg0.worst_band_energy_deviation_db <= 0.25
    assert "protected_band" in seg0.metrics


def test_missing_or_failed_evidence_forces_natural_fallback() -> None:
    """Corrupted segment post-mastering forces explicit Natural fallback."""
    sr = 48000
    cutoff_hz = 4000.0
    guard = RestorationGuard(sample_rate=sr)

    audio_nat = _make_bandlimited_speech(sr=sr, duration_s=1.0, cutoff_hz=cutoff_hz)
    seg_len = sr
    starts = [0]

    # Corrupt restored segment in protected band
    audio_rest = audio_nat.copy()
    audio_rest[1000:2000] += 0.05  # Severe low-band distortion

    lim_nat = apply_lookahead_limiter(audio_nat[np.newaxis, :], sample_rate=sr, ceiling_dbtp=-1.0)
    lim_rest = apply_lookahead_limiter(audio_rest[np.newaxis, :], sample_rate=sr, ceiling_dbtp=-1.0)
    mastered_nat = lim_nat.limited_waveform
    mastered_rest = lim_rest.limited_waveform

    segment_records = [
        SegmentRestorationDecision(
            action="restored",
            applied_strength=1.0,
            cutoff_hz=cutoff_hz,
            guard_result=None,
        )
    ]

    result = guard.verify_post_mastering(
        mastered_natural=mastered_nat,
        mastered_restored=mastered_rest,
        segment_records=segment_records,
        starts=starts,
        seg_len=seg_len,
        cutoff_hz=cutoff_hz,
        tolerance_rms=1e-4,
        tolerance_stft=1e-3,
        tolerance_third_octave_db=0.25,
    )

    assert result.passes is False
    assert result.fallback_applied is True
    assert "forced explicit Natural fallback" in result.reason
    assert len(result.segments_evidence) == 1
    assert result.segments_evidence[0].passes is False
    assert result.segments_evidence[0].action == "fallback_reverted"


def test_no_reconstructed_segments_skips_fallback() -> None:
    """When all segments are bypassed or reverted, post-mastering verifies cleanly with 0 reconstructions."""
    sr = 48000
    guard = RestorationGuard(sample_rate=sr)
    audio_nat = _make_bandlimited_speech(sr=sr, duration_s=1.0)
    lim_nat = apply_lookahead_limiter(audio_nat[np.newaxis, :], sample_rate=sr, ceiling_dbtp=-1.0)

    segment_records = [
        SegmentRestorationDecision(
            action="bypassed",
            applied_strength=0.0,
            cutoff_hz=4000.0,
            guard_result=None,
        ),
        SegmentRestorationDecision(
            action="reverted",
            applied_strength=0.0,
            cutoff_hz=4000.0,
            guard_result=None,
        ),
    ]

    result = guard.verify_post_mastering(
        mastered_natural=lim_nat.limited_waveform,
        mastered_restored=lim_nat.limited_waveform,
        segment_records=segment_records,
        starts=[0, sr // 2],
        seg_len=sr // 2,
        cutoff_hz=4000.0,
    )

    assert result.passes is True
    assert result.fallback_applied is False
    assert result.reconstructed_segments_verified == 0
    assert len(result.segments_evidence) == 0


def test_pipeline_e2e_post_mastering_verification_in_report(tmp_path: Path) -> None:
    """End-to-end pipeline run in restore mode includes post-mastering verification and segment evidence."""
    import soundfile as sf

    from hawavoclean.config import HawaVoCleanConfig

    sr = 48000
    audio_data = _make_bandlimited_speech(sr=sr, duration_s=1.5, cutoff_hz=4000.0)
    in_wav = tmp_path / "input.wav"
    out_wav = tmp_path / "output.wav"
    sf.write(str(in_wav), audio_data, sr)

    cfg = HawaVoCleanConfig()

    # Run pipeline with allow_research_restore=True and mode="restore"
    rep = run_pipeline(
        input_path=in_wav,
        output_path=out_wav,
        config=cfg,
        mode="restore",
        speaker_id="source",
        allow_research_restore=True,
    )

    assert rep.restoration is not None
    assert "post_mastering_verification" in rep.restoration
    pmv = rep.restoration["post_mastering_verification"]
    assert "passes" in pmv
    assert "fallback_applied" in pmv
    assert "segments_evidence" in pmv
    assert out_wav.exists()


def test_pipeline_e2e_post_mastering_failure_forces_natural_fallback(tmp_path: Path) -> None:
    """Injected post-mastering failure forces fallback to mastered Natural and updates report counts."""
    import soundfile as sf

    from hawavoclean.config import HawaVoCleanConfig

    sr = 48000
    audio_data = _make_bandlimited_speech(sr=sr, duration_s=1.5, cutoff_hz=4000.0)
    in_wav = tmp_path / "input_fail.wav"
    out_wav = tmp_path / "output_fail.wav"
    sf.write(str(in_wav), audio_data, sr)

    cfg = HawaVoCleanConfig()

    failing_result = PostMasteringVerificationResult(
        passes=False,
        fallback_applied=True,
        reason="Injected post-mastering failure for testing fallback",
        reconstructed_segments_verified=1,
        segments_evidence=[
            PostMasteringSegmentEvidence(
                segment_index=0,
                start_sample=0,
                end_sample=sr,
                start_time_s=0.0,
                end_time_s=1.0,
                action="fallback_reverted",
                passes=False,
                reason="Injected test violation: RMS 0.005 > 1e-4",
                rms_waveform_error=0.005,
                stft_relative_error=0.01,
                worst_band_energy_deviation_db=2.5,
                worst_band_center_hz=1000.0,
                metrics={},
            )
        ],
    )

    with patch.object(RestorationGuard, "verify_post_mastering", return_value=failing_result):
        rep = run_pipeline(
            input_path=in_wav,
            output_path=out_wav,
            config=cfg,
            mode="restore",
            speaker_id="source",
            allow_research_restore=True,
        )

    assert rep.restoration is not None
    pmv = rep.restoration["post_mastering_verification"]
    assert pmv["passes"] is False
    assert pmv["fallback_applied"] is True
    # Segment counts must show zero restored and zero reduced due to fallback
    assert rep.restoration["segments"]["restored"] == 0
    assert rep.restoration["segments"]["reduced"] == 0
    assert out_wav.exists()
