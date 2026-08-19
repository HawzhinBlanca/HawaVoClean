"""Acoustic defect and imbalance detectors for deterministic local finishing."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DefectDetectionReport:
    """Detection scores for each prospective finishing stage."""

    has_dc_offset: bool
    dc_level: float
    has_hum: bool
    hum_freq_hz: float
    click_count: int
    has_plosives: bool
    mud_imbalance_db: float
    has_harsh_sibilance: bool
    sibilance_ratio: float


def detect_defects(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> DefectDetectionReport:
    """Analyze unit waveform for DC offset, electrical hum, clicks, plosives, and sibilance."""
    n = len(waveform)
    if n < 512:
        return DefectDetectionReport(
            has_dc_offset=False,
            dc_level=0.0,
            has_hum=False,
            hum_freq_hz=0.0,
            click_count=0,
            has_plosives=False,
            mud_imbalance_db=0.0,
            has_harsh_sibilance=False,
            sibilance_ratio=1.0,
        )

    # 1. DC Offset
    dc_level = float(np.mean(waveform))
    has_dc = abs(dc_level) > 0.005

    # 2. Spectral analysis across frames of the waveform
    n_fft = min(2048, 2 ** int(np.floor(np.log2(n))))
    hop = n_fft // 2
    win = np.hanning(n_fft)
    num_frames = max(1, (n - n_fft) // hop + 1)

    # Sample up to 64 evenly spaced frames for performance and accuracy
    if num_frames > 64:
        frame_indices = np.linspace(0, num_frames - 1, 64, dtype=int)
    else:
        frame_indices = np.arange(num_frames)

    stft_mags = np.zeros((len(frame_indices), n_fft // 2 + 1), dtype=np.float32)
    for idx, f_idx in enumerate(frame_indices):
        chunk = waveform[f_idx * hop : f_idx * hop + n_fft] * win
        stft_mags[idx] = np.abs(np.fft.rfft(chunk, n=n_fft))

    fft_mag = np.mean(stft_mags, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 50Hz and 60Hz hum detection
    hum_50_idx = np.argmin(np.abs(freqs - 50.0))
    hum_60_idx = np.argmin(np.abs(freqs - 60.0))
    mean_low_energy = float(np.mean(fft_mag[(freqs >= 30) & (freqs <= 100)])) + 1e-9

    has_hum = False
    hum_freq = 0.0
    if fft_mag[hum_50_idx] > 4.0 * mean_low_energy:
        has_hum = True
        hum_freq = 50.0
    elif fft_mag[hum_60_idx] > 4.0 * mean_low_energy:
        has_hum = True
        hum_freq = 60.0

    # 3. Click / Transient spike detection
    diff = np.abs(np.diff(waveform))
    threshold_click = float(np.mean(diff) + 6.0 * np.std(diff))
    click_count = int(np.sum(diff > max(0.20, threshold_click)))

    # 4. Low-frequency plosive detection (<120Hz sudden energy burst)
    low_band_bins = freqs <= 120
    low_band_energy = float(np.sum(fft_mag[low_band_bins] ** 2))
    total_energy = float(np.sum(fft_mag**2)) + 1e-9
    has_plosives = (low_band_energy / total_energy) > 0.45

    # 5. Mud / Presence imbalance (250-500Hz vs 2k-5kHz)
    mud_bins = (freqs >= 250) & (freqs <= 500)
    pres_bins = (freqs >= 2000) & (freqs <= 5000)
    e_mud = float(np.mean(fft_mag[mud_bins])) + 1e-9
    e_pres = float(np.mean(fft_mag[pres_bins])) + 1e-9
    mud_imbalance_db = float(20.0 * np.log10(e_mud / e_pres))

    # 6. Sibilance (5kHz - 10kHz vs overall mid band)
    sib_bins = (freqs >= 5000) & (freqs <= 10000)
    mid_bins = (freqs >= 1000) & (freqs <= 4000)
    e_sib = float(np.mean(fft_mag[sib_bins])) + 1e-9
    e_mid = float(np.mean(fft_mag[mid_bins])) + 1e-9
    sib_ratio = float(e_sib / e_mid)
    has_harsh_sibilance = sib_ratio > 1.8

    return DefectDetectionReport(
        has_dc_offset=has_dc,
        dc_level=dc_level,
        has_hum=has_hum,
        hum_freq_hz=hum_freq,
        click_count=click_count,
        has_plosives=has_plosives,
        mud_imbalance_db=mud_imbalance_db,
        has_harsh_sibilance=has_harsh_sibilance,
        sibilance_ratio=sib_ratio,
    )
