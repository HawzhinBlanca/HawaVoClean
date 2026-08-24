"""Guard modes: strict_spectral enforces spectral identity; integrity keeps
artifact/timing/collapse protections but not identity."""

from typing import Any

import numpy as np
import pytest
from scipy import signal

from hawavoclean.config import GuardConfig, load_config
from hawavoclean.guard.spectral_probe import SpectralSignatureProbe
from hawavoclean.guard.verdict import GuardVerdict, evaluate_guard_pass
from hawavoclean.paths import profile_config_path

SR = 16000


def _voiced(seed: int = 1) -> np.ndarray[Any, np.dtype[np.float32]]:
    rng = np.random.default_rng(seed)
    n = SR * 3
    t = np.arange(n) / SR
    x = np.zeros(n)
    for h in range(1, 8):
        x += (0.4 / h) * np.sin(2 * np.pi * 200 * h * t)
    env = 0.5 + 0.5 * np.square(np.sin(2 * np.pi * 2.0 * t))
    x = x * env + 0.05 * rng.standard_normal(n)
    return np.asarray(x, dtype=np.float32)


def test_anchor_gating_applies_only_in_strict_mode(monkeypatch: Any) -> None:
    """The mode's exact contract: anchor comparison gates the verdict in
    strict_spectral and is advisory-only in integrity mode.

    (Synthetic audio cannot faithfully reproduce a neural restoration's
    output; the end-to-end behavior — real DFN3 output accepted in
    integrity mode, rejected in strict — was measured on a real recording
    and is recorded in the studio calibration artifact's provenance.)
    """
    import hawavoclean.guard.verdict as verdict_mod
    from hawavoclean.guard.token_anchor import AnchorComparisonResult

    probe = SpectralSignatureProbe()
    orig = _voiced()
    cand = orig.copy()  # identity: every non-anchor check passes trivially

    failing = AnchorComparisonResult(
        passed=False,
        insufficient_anchors=False,
        total_original_tokens=12,
        high_conf_anchors_count=10,
        deleted_anchors_count=3,
        substituted_anchors_count=2,
        inserted_tokens_count=0,
        max_timestamp_drift_ms=200.0,
        mean_confidence_delta=0.0,
        failure_reasons=["High-confidence anchor deleted (injected)"],
    )
    monkeypatch.setattr(verdict_mod, "compare_token_anchors", lambda **_kw: failing)

    strict = GuardConfig(mode="strict_spectral")
    res_strict, _ = evaluate_guard_pass(orig, cand, SR, True, probe, strict)
    assert res_strict.verdict == GuardVerdict.REVERT, (
        f"strict mode must gate on anchor failures, got {res_strict.verdict}"
    )

    integrity = GuardConfig(mode="integrity")
    res_integrity, _ = evaluate_guard_pass(orig, cand, SR, True, probe, integrity)
    assert res_integrity.verdict == GuardVerdict.PASS, (
        f"integrity mode must not gate on anchors alone: "
        f"{res_integrity.verdict} {res_integrity.reasons}"
    )
    # The anchor scores must still be RECORDED for the audit trail.
    assert res_integrity.scores.get("deleted_anchors") == 3


def test_integrity_mode_still_rejects_broken_audio() -> None:
    """Integrity mode is not a bypass: gross damage must still revert."""
    probe = SpectralSignatureProbe()
    orig = _voiced()

    # Candidate with the middle third silenced: envelope correlation collapses.
    broken = orig.copy()
    broken[len(orig) // 3 : 2 * len(orig) // 3] = 0.0

    integrity = GuardConfig(mode="integrity", max_posterior_js_div=0.30, max_peak_js_div=0.90)
    res, _ = evaluate_guard_pass(orig, broken, SR, True, probe, integrity)
    assert res.verdict == GuardVerdict.REVERT, (
        f"integrity mode accepted audio with a silenced span: {res.verdict}"
    )


#: The damage matrix runs at 48 kHz, unlike the 16 kHz fixtures above.
_MATRIX_SR = 48000


def _lowpassed(x: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    sos = signal.butter(8, 2000 / (_MATRIX_SR / 2), btype="lowpass", output="sos")
    return np.asarray(signal.sosfiltfilt(sos, x), dtype=np.float32)


def _stretched(x: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.asarray(signal.resample(x, len(x) * 2)[: len(x)], dtype=np.float32)


def _noise(x: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return (np.random.default_rng(0).standard_normal(len(x)) * 0.05).astype(np.float32)


@pytest.mark.parametrize(
    ("damage", "make"),
    [
        ("silence", np.zeros_like),
        ("white noise", _noise),
        ("band gutted at 2 kHz", _lowpassed),
        ("polarity inverted", lambda x: (-x).astype(np.float32)),
        ("6 dB quieter", lambda x: (x * 0.5).astype(np.float32)),
        ("time stretched", _stretched),
        ("content reordered", lambda x: x[::-1].copy()),
    ],
)
def test_integrity_mode_reverts_each_class_of_damage(damage: str, make: Any) -> None:
    """One case per way a core can wreck a unit, on a signal the probe can read.

    A frequency sweep rather than a speaker fixture: the canonical references
    carry no confident token anchors, so their CTC posteriors are near-uniform
    and reordering them is invisible. That is a property of the fixtures, not
    of the guard -- a swept tone reversed scores 0.647 divergence and is
    refused, while the same sweep against itself passes.
    """
    sr = _MATRIX_SR
    t = np.arange(int(sr * 2.0)) / sr
    original = (0.5 * signal.chirp(t, f0=200, f1=3500, t1=t[-1], method="linear")).astype(
        np.float32
    )
    config = load_config(str(profile_config_path("studio"))).guard
    probe = SpectralSignatureProbe(probe_id=config.probe_id, target_sr=16000)

    unharmed, _ = evaluate_guard_pass(
        original, original.copy(), sr, is_speech=True, probe=probe, config=config
    )
    assert unharmed.verdict == GuardVerdict.PASS, "an untouched candidate must pass"

    res, _ = evaluate_guard_pass(
        original, make(original), sr, is_speech=True, probe=probe, config=config
    )
    assert res.verdict != GuardVerdict.PASS, f"integrity mode accepted {damage}"
