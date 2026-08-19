"""Acoustic signal integrity and artifact detectors for Guard A and Guard B."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SignalIntegrityResult:
    """Detailed signal integrity metrics and threshold decisions."""

    passed: bool
    consonant_retention_ratio: float
    spectral_hole_score: float
    musical_noise_score: float
    clipping_samples_count: int
    unexplained_hf_ratio: float
    max_edge_discontinuity: float
    failure_reasons: list[str] = field(default_factory=list)


def check_signal_integrity(
    orig_waveform: np.ndarray[Any, np.dtype[np.float32]],
    cand_waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    spectral_hole_thresh: float = 0.15,
    musical_noise_thresh: float = 0.20,
    min_hf_preservation_ratio: float = 0.60,
    max_allowed_clipping_samples: int = 0,
) -> SignalIntegrityResult:
    """Evaluate acoustic sanity, artifact creation, spectral holes, and clipping."""
    reasons: list[str] = []

    orig_len = len(orig_waveform)
    cand_len = len(cand_waveform)
    n_samples = min(orig_len, cand_len)

    if n_samples < 512:
        return SignalIntegrityResult(
            passed=True,
            consonant_retention_ratio=1.0,
            spectral_hole_score=0.0,
            musical_noise_score=0.0,
            clipping_samples_count=0,
            unexplained_hf_ratio=0.0,
            max_edge_discontinuity=0.0,
        )

    w_orig = orig_waveform[:n_samples]
    w_cand = cand_waveform[:n_samples]

    # 1. NEWLY introduced clipping: samples at/over full scale in the
    # candidate where the original was below it. An already peak-normalised
    # input that the candidate preserves is not a clipping artifact.
    clipping_samples = int(np.sum((np.abs(w_cand) >= 1.0) & (np.abs(w_orig) < 1.0)))
    if clipping_samples > max_allowed_clipping_samples:
        reasons.append(
            f"Newly introduced hard clipping detected ({clipping_samples} samples >= 1.0)"
        )

    # 2. STFT Spectral analysis
    n_fft = 1024
    hop = 256
    win = np.hanning(n_fft)

    num_frames = (n_samples - n_fft) // hop + 1
    if num_frames <= 0:
        return SignalIntegrityResult(
            passed=len(reasons) == 0,
            consonant_retention_ratio=1.0,
            spectral_hole_score=0.0,
            musical_noise_score=0.0,
            clipping_samples_count=clipping_samples,
            unexplained_hf_ratio=0.0,
            max_edge_discontinuity=0.0,
            failure_reasons=reasons,
        )

    stft_orig = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.float32)
    stft_cand = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop
        stft_orig[i] = np.abs(np.fft.rfft(w_orig[start : start + n_fft] * win, n=n_fft))
        stft_cand[i] = np.abs(np.fft.rfft(w_cand[start : start + n_fft] * win, n=n_fft))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 3. Consonant band (2000Hz - 8000Hz) retention
    consonant_bins = (freqs >= 2000) & (freqs <= 8000)
    if np.any(consonant_bins):
        e_orig_cons = float(np.mean(stft_orig[:, consonant_bins] ** 2))
        e_cand_cons = float(np.mean(stft_cand[:, consonant_bins] ** 2))
        cons_ratio = e_cand_cons / (e_orig_cons + 1e-9)
        if e_orig_cons > 1e-4 and cons_ratio < min_hf_preservation_ratio:
            reasons.append(
                f"Consonant presence severely attenuated: ratio {cons_ratio:.3f} < minimum {min_hf_preservation_ratio:.3f}"
            )
    else:
        cons_ratio = 1.0

    # 4. Spectral hole detector: a band wiped out of the SIGNAL.
    # Evaluate only active frames (original frame energy within 20 dB of the
    # unit's loud reference) and only bins that carried real energy in the
    # original (within 40 dB of that frame's peak). A lowered noise floor in
    # the gaps between phrases is denoising, not a hole — counting it was a
    # measured false-reject on clean restoration output. The score is the
    # mean fraction of signal bins wiped per active frame, so a missing band
    # registers proportionally to its share of the signal.
    frame_energy = np.sqrt(np.mean(stft_orig**2, axis=1) + 1e-12)
    loud_ref = float(np.percentile(frame_energy, 90))
    active = frame_energy >= loud_ref * 10 ** (-20 / 20)
    if np.any(active):
        act_orig = stft_orig[active]
        act_cand = stft_cand[active]
        frame_peak = np.max(act_orig, axis=1, keepdims=True) + 1e-9
        signal_bins = act_orig >= frame_peak * 10 ** (-40 / 20)
        bin_ratio = (act_cand + 1e-6) / (act_orig + 1e-6)
        wiped = (bin_ratio < 0.05) & signal_bins
        per_frame = np.sum(wiped, axis=1) / np.maximum(np.sum(signal_bins, axis=1), 1)
        spectral_hole_score = float(np.mean(per_frame))
    else:
        spectral_hole_score = 0.0
    if spectral_hole_score > spectral_hole_thresh:
        reasons.append(
            f"Spectral hole artifact detected: score {spectral_hole_score:.3f} > threshold {spectral_hole_thresh:.3f}"
        )

    # 5. Musical noise detector: NEW isolated-peak variance introduced by the
    # candidate, i.e. flatness variance beyond what the original already had.
    # (Tonal bursts separated by pauses have high flatness variance on their
    # own; an identical candidate must score zero.)
    def _flatness_std(mag: np.ndarray[Any, Any]) -> float:
        flat = np.exp(np.mean(np.log(mag + 1e-9), axis=1)) / (np.mean(mag, axis=1) + 1e-9)
        return float(np.std(flat))

    musical_noise_score = max(0.0, _flatness_std(stft_cand) - _flatness_std(stft_orig))
    if musical_noise_score > musical_noise_thresh:
        reasons.append(
            f"Musical noise / isolated spectral spikes detected: score {musical_noise_score:.3f} > threshold {musical_noise_thresh:.3f}"
        )

    # 6. Unexplained high-frequency synthesis (>10kHz energy where original was silent)
    hf_bins = freqs >= 10000
    if np.any(hf_bins):
        e_orig_hf = float(np.mean(stft_orig[:, hf_bins] ** 2))
        e_cand_hf = float(np.mean(stft_cand[:, hf_bins] ** 2))
        if e_orig_hf < 1e-6 and e_cand_hf > 1e-3:
            unexplained_hf = e_cand_hf / (e_orig_hf + 1e-6)
            reasons.append(
                f"Unexplained high-frequency synthesis detected above 10kHz (energy jump {unexplained_hf:.1f}x)"
            )
        else:
            unexplained_hf = 1.0
    else:
        unexplained_hf = 1.0

    # 7. Edge continuity check
    edge_jump_left = abs(float(w_cand[0] - w_orig[0]))
    edge_jump_right = abs(float(w_cand[-1] - w_orig[-1]))
    max_edge = max(edge_jump_left, edge_jump_right)
    if max_edge > 0.80:
        reasons.append(f"Excessive edge amplitude discontinuity ({max_edge:.2f} > 0.80)")

    passed = len(reasons) == 0

    return SignalIntegrityResult(
        passed=passed,
        consonant_retention_ratio=cons_ratio,
        spectral_hole_score=spectral_hole_score,
        musical_noise_score=musical_noise_score,
        clipping_samples_count=clipping_samples,
        unexplained_hf_ratio=unexplained_hf,
        max_edge_discontinuity=max_edge,
        failure_reasons=reasons,
    )
