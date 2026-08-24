"""Unit tests for continuous bandwidth detector and cutoff estimation."""

import numpy as np
import scipy.signal as signal

from hawavoclean.restoration.bandwidth import BandwidthDetector, BandwidthEstimate


def test_bandwidth_detector_fullband() -> None:
    """Test bandwidth detection on a full-band 48 kHz synthetic signal."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Signal with energy up to 20 kHz
    sig = (
        0.5 * np.sin(2 * np.pi * 1000 * t)
        + 0.3 * np.sin(2 * np.pi * 8000 * t)
        + 0.2 * np.sin(2 * np.pi * 18000 * t)
    ).astype(np.float32)

    detector = BandwidthDetector(sample_rate=sr)
    est = detector.detect(sig)

    assert isinstance(est, BandwidthEstimate)
    assert est.effective_cutoff_hz > 16000.0
    assert est.restore_recommended is False


def _speech_like(sr: int, seconds: float = 1.0, tilt: float = 1.6, seed: int = 11) -> np.ndarray:
    """A dense harmonic stack reaching Nyquist, with a natural spectral tilt.

    The detector distinguishes a filter cliff from spectral tilt, which needs a
    spectrum that HAS a tilt. Three pure tones do not: between them the
    spectrum is numerical floor, so any candidate edge looks like a cliff and
    the measurement says more about the fixture than about the detector.
    """
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    rng = np.random.default_rng(seed)
    out = np.zeros_like(t)
    harmonic = 1
    while 130.0 * harmonic < sr / 2:
        out += (1.0 / (harmonic**tilt)) * np.sin(
            2 * np.pi * 130.0 * harmonic * t + rng.uniform(0, 2 * np.pi)
        )
        harmonic += 1
    return (out / float(np.max(np.abs(out)))).astype(np.float32)


def test_bandwidth_detector_lowpass_cutoff() -> None:
    """Test cutoff frequency detection on speech-like audio low-passed at 8 kHz."""
    sr = 48000
    sos = signal.butter(8, 8000 / 24000, btype="lowpass", output="sos")
    lp_sig = signal.sosfiltfilt(sos, _speech_like(sr)).astype(np.float32)
    lp_sig = (lp_sig / np.max(np.abs(lp_sig))).astype(np.float32)

    detector = BandwidthDetector(sample_rate=sr)
    est = detector.detect(lp_sig)

    assert est.restore_recommended is True
    assert 7000.0 <= est.effective_cutoff_hz <= 9500.0
    # Erring high is the safe direction: a cutoff below the true edge would put
    # the protected band under real content and let the model overwrite it.
    assert est.effective_cutoff_hz >= 7600.0


def test_bandwidth_detector_override() -> None:
    """Test that explicit cutoff override takes precedence over auto-detection."""
    sr = 48000
    sig = np.random.randn(sr).astype(np.float32) * 0.1

    detector = BandwidthDetector(sample_rate=sr)
    est = detector.detect(sig, override_cutoff_hz=6500.0)

    assert est.effective_cutoff_hz == 6500.0
    assert est.restore_recommended is True


def test_bandwidth_detector_multichannel() -> None:
    """Test that stereo input is properly handled across channels."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig_mono = (0.5 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)).astype(
        np.float32
    )
    sos = signal.butter(6, 6000 / 24000, btype="lowpass", output="sos")
    lp_mono = signal.sosfiltfilt(sos, sig_mono).astype(np.float32)

    stereo_sig = np.stack([lp_mono, lp_mono * 0.8], axis=0)

    detector = BandwidthDetector(sample_rate=sr)
    est = detector.detect(stereo_sig)

    assert est.restore_recommended is True
    assert 5000.0 <= est.effective_cutoff_hz <= 7000.0
