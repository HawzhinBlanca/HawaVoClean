"""The finishing chain must not re-voice normal speech.

Found by ear on a real recording ("harsh, treble, bass removed") and
confirmed by measurement: the 'mud' detector's +2 dB threshold fired on 100%
of real voices (natural speech carries 20-40 dB more energy at 250-500 Hz
than at 2-5 kHz), and the resulting EQ cut low-mids 5.7 dB and boosted
presence 2.5 dB on every unit. DeepFilterNet3 itself was tonally flat
(±0.4 dB). This test pins tonal transparency of finishing on natural voice.
"""

from typing import Any

import numpy as np
import scipy.signal

from hawavoclean.config import FinishingConfig
from hawavoclean.finishing.detect import detect_defects
from hawavoclean.finishing.safe_finish import apply_finishing_stages

SR = 48000


def _natural_voice(
    seconds: float = 6.0, f0: float = 130.0
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Harmonic series with a realistic -6 dB/octave rolloff above 500 Hz,
    syllabic modulation, and a faint breath-noise bed: a plausible male
    voice spectrum (low-mids ~25 dB above presence)."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    for h in range(1, 60):
        f = f0 * h
        if f > 12000:
            break
        amp = 1.0 / h if f < 500 else (500.0 / f) ** 1.0 / h
        x += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * t) ** 2
    x = x * env
    x = x / np.max(np.abs(x)) * 0.3
    x += 0.002 * rng.standard_normal(len(t))
    return np.asarray(x, dtype=np.float32)


def _band_change_db(
    orig: np.ndarray[Any, Any], proc: np.ndarray[Any, Any], lo: float, hi: float
) -> float:
    f, Po = scipy.signal.welch(orig, SR, nperseg=4096)
    _, Pp = scipy.signal.welch(proc, SR, nperseg=4096)
    core = (f >= 1000) & (f < 3000)
    g = 10 * np.log10(np.trapezoid(Pp[core], f[core]) / np.trapezoid(Po[core], f[core]))
    m = (f >= lo) & (f < hi)
    return float(10 * np.log10(np.trapezoid(Pp[m], f[m]) / np.trapezoid(Po[m], f[m])) - g)


def test_natural_voice_is_not_flagged_as_muddy() -> None:
    d = detect_defects(_natural_voice(), SR)
    assert not d.has_mud, (
        f"a natural voice spectrum was flagged as mud (imbalance {d.mud_imbalance_db:+.1f} dB)"
    )


def test_finishing_is_tonally_transparent_on_natural_voice() -> None:
    x = _natural_voice()
    out, actions = apply_finishing_stages(x, SR, FinishingConfig(), "gentle")
    lowmid = _band_change_db(x, out, 250, 500)
    bass = _band_change_db(x, out, 120, 250)
    presence = _band_change_db(x, out, 3000, 6000)
    assert abs(lowmid) < 1.5, f"low-mids changed {lowmid:+.1f} dB on a natural voice ({actions})"
    assert abs(bass) < 1.5, f"bass changed {bass:+.1f} dB on a natural voice ({actions})"
    assert abs(presence) < 1.5, (
        f"presence changed {presence:+.1f} dB on a natural voice ({actions})"
    )


def test_genuinely_muddy_voice_still_gets_corrected_gently() -> None:
    """A voice with a real +12 dB bump at 250-500 Hz (proximity effect /
    boomy room) should be corrected — by a few dB, not by 6."""
    x = _natural_voice()
    sos = scipy.signal.butter(2, [250, 500], btype="bandpass", fs=SR, output="sos")
    boom = np.asarray(x + 3.0 * scipy.signal.sosfiltfilt(sos, x), dtype=np.float32)
    d = detect_defects(boom, SR)
    assert d.has_mud, f"a +12 dB low-mid boom was not flagged (imbalance {d.mud_imbalance_db:+.1f})"
    out, _ = apply_finishing_stages(boom, SR, FinishingConfig(), "gentle")
    cut = _band_change_db(boom, out, 250, 500)
    assert -4.0 < cut < -0.5, f"mud correction was {cut:+.1f} dB (expected a gentle cut)"


def test_full_studio_pipeline_is_tonally_transparent_on_real_recording(tmp_path: Any) -> None:
    """End-to-end on a bundled real-voice-shaped fixture through the
    production chain: dialogue-frame band levels within 1.5 dB of the input
    (1-3 kHz normalised), excluding the deliberate <75 Hz rumble filter."""
    import soundfile as sf

    from hawavoclean.guard.spectral_probe import FixedProbe
    from hawavoclean.pipeline import run_pipeline

    src = tmp_path / "voice.wav"
    x = _natural_voice(seconds=8.0)
    sf.write(str(src), x, SR)
    out = tmp_path / "out.wav"
    run_pipeline(src, out, profile="production", overwrite=True, probe_override=FixedProbe())
    y, _ = sf.read(str(out), dtype="float32")
    for lo, hi in ((120, 250), (250, 500), (500, 1000), (3000, 6000), (6000, 10000)):
        d = _band_change_db(x, y, lo, hi)
        assert abs(d) < 1.5, f"{lo}-{hi} Hz changed {d:+.1f} dB through the full pipeline"
