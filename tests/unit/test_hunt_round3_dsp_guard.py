"""Bugs found by adversarial review of DSP/guard/policy (round 3). Each test
is the hunter's minimal repro, kept as a permanent regression."""

from typing import Any, Literal

import numpy as np
import pytest

from hawavoclean.alignment.delay import estimate_gcc_phat_delay
from hawavoclean.assembly.stitch import assemble_channel_timeline
from hawavoclean.config import GuardConfig, PolicyConfig
from hawavoclean.finishing.detect import detect_defects
from hawavoclean.finishing.loudness import compute_static_master_gain, measure_loudness_and_peaks
from hawavoclean.guard.signal import check_signal_integrity
from hawavoclean.guard.spectral_probe import SpectralSignatureProbe
from hawavoclean.guard.verdict import GuardVerdict, evaluate_guard_pass
from hawavoclean.policy.continuity import resolve_source_continuity
from hawavoclean.policy.decision import UnitPolicyDecision, evaluate_unit_policy
from hawavoclean.segmentation.types import SpeechUnit


def _voiced(sr: int = 16000, sec: float = 1.0) -> np.ndarray[Any, np.dtype[np.float32]]:
    t = np.arange(int(sr * sec)) / sr
    return np.asarray(
        np.sin(2 * np.pi * 200 * t) * np.sin(2 * np.pi * 3 * t) * 0.3, dtype=np.float32
    )


# 1. delay sign ------------------------------------------------------------
def test_alignment_removes_delay_instead_of_doubling_it() -> None:
    sr = 48000
    x = np.random.default_rng(0).standard_normal(sr).astype(np.float32)
    d = 10
    cand = np.concatenate([np.zeros(d, np.float32), x[:-d]])  # candidate lags by 10
    r = estimate_gcc_phat_delay(x, cand, sr)
    al = r.aligned_candidate
    assert abs(r.delay_samples - d) < 0.5, f"measured delay {r.delay_samples}, expected +{d}"
    # After a left shift of d, the last d samples are padding; compare the rest.
    assert np.allclose(al[2 * d : -d], x[2 * d : -d], atol=1e-4), (
        "aligned candidate is not aligned to the original (delay sign inverted: "
        "shift applied the wrong way doubles the lag)"
    )


# 8 + 12. delay degenerate cases ----------------------------------------------
def test_alignment_short_input_and_large_max_lag_do_not_zero_the_candidate() -> None:
    x = np.random.default_rng(0).standard_normal(600).astype(np.float32)
    r = estimate_gcc_phat_delay(x, x.copy(), 48000)
    assert abs(r.delay_samples) < 2 and float(np.abs(r.aligned_candidate).max()) > 0.1
    x = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    r = estimate_gcc_phat_delay(x, x.copy(), 48000, max_delay_ms=250.0)
    assert abs(r.delay_samples) < 2 and float(np.abs(r.aligned_candidate).max()) > 0.1


def test_alignment_flat_correlation_yields_zero_delay() -> None:
    x = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    r = estimate_gcc_phat_delay(x, np.zeros(48000, np.float32), 48000)
    assert r.delay_samples == 0.0, f"flat correlation produced delay {r.delay_samples}"


# 3. hum detection ---------------------------------------------------------------
@pytest.mark.parametrize("sr", [16000, 44100, 48000])
@pytest.mark.parametrize("hum", [50.0, 60.0])
def test_hum_is_detected_at_real_sample_rates(sr: int, hum: float) -> None:
    t = np.arange(sr * 3) / sr
    x = (
        0.9 * np.sin(2 * np.pi * hum * t) + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    ).astype(np.float32)
    d = detect_defects(x, sr)
    assert d.has_hum, f"{hum} Hz hum at {sr} Hz not detected"
    assert abs(d.hum_freq_hz - hum) < 6.0


# 13. sibilance at low rates ---------------------------------------------------
def test_detect_defects_no_nan_at_8khz() -> None:
    d = detect_defects(np.random.default_rng(0).standard_normal(8000).astype(np.float32), 8000)
    assert np.isfinite(d.sibilance_ratio)


# 4. empty candidate -------------------------------------------------------------
def test_empty_candidate_never_passes_the_guard() -> None:
    sr = 16000
    x = _voiced(sr)
    d, _ = evaluate_unit_policy(
        x,
        np.zeros(0, np.float32),
        sr,
        True,
        SpectralSignatureProbe(),
        GuardConfig(mode="integrity"),
        PolicyConfig(),
    )
    assert not d.is_enhanced, "empty candidate was selected (unit would become silence)"
    assert len(d.selected_waveform) == len(x)


# 10. NaN fails open ----------------------------------------------------------------
@pytest.mark.parametrize("mode", ["strict_spectral", "integrity"])
def test_nan_candidate_reverts(mode: Literal["strict_spectral", "integrity"]) -> None:
    sr = 16000
    x = _voiced(sr)
    c = np.where(np.arange(sr) < sr // 2, x, np.nan).astype(np.float32)
    res, _ = evaluate_guard_pass(x, c, sr, True, SpectralSignatureProbe(), GuardConfig(mode=mode))
    assert res.verdict == GuardVerdict.REVERT, f"NaN candidate got {res.verdict} in {mode}"


# 5. clipping must be NEWLY introduced ------------------------------------------------
@pytest.mark.parametrize("mode", ["strict_spectral", "integrity"])
def test_full_scale_input_identical_candidate_is_not_clipping(
    mode: Literal["strict_spectral", "integrity"],
) -> None:
    sr = 16000
    x = _voiced(sr)
    x = np.asarray(x / np.abs(x).max(), dtype=np.float32)  # peak-normalised: touches 1.0
    res, _ = evaluate_guard_pass(
        x, x.copy(), sr, True, SpectralSignatureProbe(), GuardConfig(mode=mode)
    )
    assert not any("clipping" in r.lower() for r in res.reasons), res.reasons


# 6. musical noise must be relative ----------------------------------------------------
def test_identical_candidate_never_scores_musical_noise() -> None:
    sr = 48000
    n = sr * 2
    t = np.arange(n) / sr
    env = np.sin(2 * np.pi * 2.5 * t) > 0
    x = (
        env * (0.3 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 360 * t))
        + 1e-3 * np.random.default_rng(0).standard_normal(n)
    ).astype(np.float32)
    r = check_signal_integrity(x, x.copy(), sr)
    assert r.passed, r.failure_reasons
    assert r.musical_noise_score < 0.05


# 7. short-file loudness continuity --------------------------------------------------------
def test_short_file_loudness_is_continuous_across_400ms() -> None:
    sr = 48000
    t = np.arange(sr) / sr
    s = (
        0.25 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
        + 0.05 * np.random.default_rng(0).standard_normal(sr)
    ).astype(np.float32)
    m399 = measure_loudness_and_peaks(s[None, : int(sr * 0.399)], sr)
    m401 = measure_loudness_and_peaks(s[None, : int(sr * 0.401)], sr)
    g399 = compute_static_master_gain(m399.integrated_lufs, -19, m399.true_peak_dbtp)
    g401 = compute_static_master_gain(m401.integrated_lufs, -19, m401.true_peak_dbtp)
    assert abs(g399 - g401) < 2.0, (
        f"gain jumps {g399:.1f} -> {g401:.1f} dB across the 400 ms boundary"
    )


# 9. continuity cascade ----------------------------------------------------------------------
def test_continuity_cascade_converges() -> None:
    def u(i: int, s: int, e: int, fb: bool) -> SpeechUnit:
        return SpeechUnit(i, 0, s, e, s, e, True, fb)

    def d(enh: bool) -> UnitPolicyDecision:
        return UnitPolicyDecision(np.zeros(10, np.float32), enh, 1.0, GuardVerdict.PASS)

    # Ten-sample units cannot carry a 30 ms fade, so the old all-or-nothing
    # remedy is the only one available and must still converge: unit1 reverts
    # (seam with unit2 across forced cut 1|2), then unit0 reverts too.
    out = resolve_source_continuity(
        [u(0, 0, 10, True), u(1, 10, 20, True), u(2, 20, 30, False)],
        [d(True), d(True), d(False)],
        [np.zeros(10, np.float32)] * 3,
        48000,
    )
    flags = [x.is_enhanced for x in out.decisions]
    assert flags == [False, False, False], f"cascade left a seam: {flags}"

    # The regression that made this matter: on units of a realistic length the
    # cascade must not happen at all. One failing unit used to take the whole
    # file with it (Flute 09: 5 passing units discarded, 7.23 dB of separation).
    sr, n = 48000, 48000
    big = [u(i, i * n, (i + 1) * n, True) for i in range(6)]
    out2 = resolve_source_continuity(
        big,
        [d(True)] * 5 + [d(False)],
        [np.zeros(n, np.float32)] * 6,
        sr,
    )
    flags2 = [x.is_enhanced for x in out2.decisions]
    assert flags2 == [True] * 5 + [False], f"cascade ate passing units: {flags2}"
    assert out2.fade_out_samples[4] > 0, "the one real seam was not faded"


# 11. stitch declick uses the wrong unit's flag -----------------------------------------------
def test_stitch_declick_skips_forced_cut_and_ramps_natural_joint() -> None:
    def u(i: int, s: int, e: int, fb: bool) -> SpeechUnit:
        return SpeechUnit(i, 0, s, e, s, e, True, fb)

    a = np.full(48000, 0.5, np.float32)
    b = np.full(48000, -0.5, np.float32)
    forced = assemble_channel_timeline(
        [u(0, 0, 48000, True), u(1, 48000, 96000, False)], [a, b], 96000, 48000
    )
    natural = assemble_channel_timeline(
        [u(0, 0, 48000, False), u(1, 48000, 96000, True)], [a, b], 96000, 48000
    )
    # At a forced cut (speech split) no declick ramp: content must be untouched.
    assert np.allclose(forced[48000:48003], -0.5), forced[48000:48003]
    # At a natural joint the step is diffused.
    assert not np.allclose(natural[48000:48003], -0.5), natural[48000:48003]
