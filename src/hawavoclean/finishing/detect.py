"""Acoustic defect and imbalance detectors for deterministic local finishing."""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Low-mid (250-500 Hz) vs presence (2-5 kHz) level ratio of a normal voice,
# and how far above it counts as a defect worth correcting. Measured
# 2026-08-19 on four real recordings: +30.7, +41.2, +10.8 dB and a +43.9 dB
# fixture; the reference sits at the upper-middle so only genuine boom
# (proximity effect, resonant room) exceeds it.
NORMAL_VOICE_MUD_REFERENCE_DB = 36.0
MUD_EXCESS_THRESHOLD_DB = 6.0


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
    has_mud: bool
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
            has_mud=False,
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

    # 50/60 Hz mains hum: a dedicated long FFT on the low band. With the
    # 2048-point analysis above there are only ~3 bins between 30-100 Hz at
    # 48 kHz, so "hum bin > 4x the mean of the band" was mathematically
    # impossible and de-hum never ran on real inputs. Here: 16384-point
    # (~2.9 Hz bins at 48 kHz) and the hum bin is compared against the
    # MEDIAN of the 30-150 Hz band EXCLUDING its own neighbourhood.
    has_hum = False
    hum_freq = 0.0
    n_hum_fft = 16384
    if len(waveform) >= n_hum_fft:
        hum_frames = max(1, min(16, (len(waveform) - n_hum_fft) // (n_hum_fft // 2) + 1))
        hum_win = np.hanning(n_hum_fft)
        hum_mag = np.zeros(n_hum_fft // 2 + 1, dtype=np.float64)
        for i in range(hum_frames):
            start = i * (n_hum_fft // 2)
            hum_mag += np.abs(
                np.fft.rfft(waveform[start : start + n_hum_fft] * hum_win, n=n_hum_fft)
            )
        hum_mag /= hum_frames
        hum_freqs = np.fft.rfftfreq(n_hum_fft, d=1.0 / sample_rate)
        band = (hum_freqs >= 30.0) & (hum_freqs <= 150.0)
        best_ratio = 0.0
        for target in (50.0, 60.0):
            idx = int(np.argmin(np.abs(hum_freqs - target)))
            lo = max(0, idx - 2)
            hi = min(len(hum_mag), idx + 3)
            peak = float(np.max(hum_mag[lo:hi]))
            neighbourhood = np.zeros_like(band)
            neighbourhood[lo:hi] = True
            ref_bins = band & ~neighbourhood
            floor = float(np.median(hum_mag[ref_bins])) + 1e-9 if np.any(ref_bins) else 1e-9
            ratio = peak / floor
            if ratio > 8.0 and ratio > best_ratio:
                best_ratio = ratio
                has_hum = True
                hum_freq = float(hum_freqs[lo + int(np.argmax(hum_mag[lo:hi]))])

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
    # Natural speech carries FAR more energy at 250-500 Hz than at 2-5 kHz —
    # measured on real recordings: +11 to +41 dB, median ~+30 dB. "Mud" is
    # EXCESS over that, not the mere presence of low-mids. The old +2 dB
    # threshold flagged every real voice and thinned it.
    mud_imbalance_db = float(20.0 * np.log10(e_mud / e_pres))
    has_mud = mud_imbalance_db > NORMAL_VOICE_MUD_REFERENCE_DB + MUD_EXCESS_THRESHOLD_DB

    # 6. Sibilance (5kHz - 10kHz vs overall mid band); at low sample rates
    # the sibilance band may lie above Nyquist — then there is no sibilance
    # to measure, not a NaN.
    sib_bins = (freqs >= 5000) & (freqs <= 10000)
    mid_bins = (freqs >= 1000) & (freqs <= 4000)
    if np.any(sib_bins) and np.any(mid_bins):
        e_sib = float(np.mean(fft_mag[sib_bins])) + 1e-9
        e_mid = float(np.mean(fft_mag[mid_bins])) + 1e-9
        sib_ratio = float(e_sib / e_mid)
    else:
        sib_ratio = 0.0
    has_harsh_sibilance = sib_ratio > 1.8

    return DefectDetectionReport(
        has_dc_offset=has_dc,
        dc_level=dc_level,
        has_hum=has_hum,
        hum_freq_hz=hum_freq,
        click_count=click_count,
        has_plosives=has_plosives,
        mud_imbalance_db=mud_imbalance_db,
        has_mud=has_mud,
        has_harsh_sibilance=has_harsh_sibilance,
        sibilance_ratio=sib_ratio,
    )
