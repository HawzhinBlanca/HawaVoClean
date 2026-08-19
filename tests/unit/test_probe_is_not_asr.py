"""Pin what the spectral probe actually measures — and what it cannot.

The probe responds to spectral SHAPE. Different content with the same
spectral envelope looks SIMILAR to it; the same content with a shifted
spectrum looks DIFFERENT. Both assertions pass, and together they document
that this instrument detects spectral change, not linguistic change — the
reason it must never be described as a speech recognizer.
"""

import numpy as np
import scipy.signal

from hawavoclean.guard.posterior import compare_ctc_posteriors
from hawavoclean.guard.spectral_probe import SpectralSignatureProbe

SR = 16000


def _voiced(pattern: list[float], f0: float, seed: int) -> np.ndarray:
    """Speech-like tone whose 'syllables' follow an amplitude pattern."""
    rng = np.random.default_rng(seed)
    n = SR * 3
    t = np.arange(n) / SR
    x = np.zeros(n)
    for h in range(1, 8):
        x += (0.4 / h) * np.sin(2 * np.pi * f0 * h * t)
    seg = n // len(pattern)
    env = np.concatenate([np.full(seg, a) for a in pattern])[:n]
    x = x * env + 0.01 * rng.standard_normal(n)
    return np.asarray(x, dtype=np.float32)


def test_different_content_same_envelope_looks_the_same() -> None:
    """Two DIFFERENT 'utterances' with the same spectrum: probe cannot tell."""
    probe = SpectralSignatureProbe()
    content_a = _voiced([1.0, 0.4, 0.9, 0.5, 0.8, 0.6], f0=200.0, seed=1)
    content_b = _voiced([0.9, 0.5, 0.8, 0.6, 1.0, 0.4], f0=200.0, seed=2)

    res_a = probe.infer(content_a, SR)
    res_b = probe.infer(content_b, SR)
    div = compare_ctc_posteriors(
        res_a.frame_distributions, res_b.frame_distributions, max_mean_js_div=1.0
    )
    assert div.mean_js_divergence < 0.10, (
        f"probe distinguished same-spectrum content (js={div.mean_js_divergence:.4f}); "
        "if this ever fails, the probe has become content-sensitive and this "
        "module's documentation must be revisited"
    )


def test_same_content_shifted_spectrum_looks_different() -> None:
    """The SAME 'utterance' with its spectrum moved: probe flags it."""
    probe = SpectralSignatureProbe()
    content = _voiced([1.0, 0.4, 0.9, 0.5, 0.8, 0.6], f0=200.0, seed=1)
    # Remove the fundamental region entirely — a gross change to the spectral
    # shape the probe actually reads (measured js ~= 0.50 on this input).
    sos = scipy.signal.butter(4, [150.0, 650.0], btype="bandstop", fs=SR, output="sos")
    shifted = np.ascontiguousarray(scipy.signal.sosfiltfilt(sos, content), dtype=np.float32)

    res_orig = probe.infer(content, SR)
    res_shift = probe.infer(shifted, SR)
    div = compare_ctc_posteriors(
        res_orig.frame_distributions, res_shift.frame_distributions, max_mean_js_div=1.0
    )
    assert div.mean_js_divergence > 0.10, (
        f"probe failed to notice a gross spectral change (js={div.mean_js_divergence:.4f})"
    )
