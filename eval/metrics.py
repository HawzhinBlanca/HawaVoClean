"""Evaluation metrics for linguistic fidelity and audio enhancement quality."""

from typing import Any

import numpy as np


def compute_edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Compute (substitutions, deletions, insertions, correct) Levenshtein distance."""
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack
    i, j = n, m
    subs, dels, ins, hits = 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return subs, dels, ins, hits


def calculate_wer_cer(ref_text: str, hyp_text: str) -> tuple[float, float]:
    """Calculate Word Error Rate (WER) and Character Error Rate (CER)."""
    ref_words = ref_text.strip().split()
    hyp_words = hyp_text.strip().split()

    if not ref_words:
        wer = 0.0 if not hyp_words else 1.0
    else:
        subs_w, dels_w, ins_w, _ = compute_edit_distance(ref_words, hyp_words)
        wer = (subs_w + dels_w + ins_w) / len(ref_words)

    ref_chars = list(ref_text.strip())
    hyp_chars = list(hyp_text.strip())
    if not ref_chars:
        cer = 0.0 if not hyp_chars else 1.0
    else:
        subs_c, dels_c, ins_c, _ = compute_edit_distance(ref_chars, hyp_chars)
        cer = (subs_c + dels_c + ins_c) / len(ref_chars)

    return wer, cer


def calculate_si_sdr(
    reference: np.ndarray[Any, np.dtype[np.float32]],
    estimate: np.ndarray[Any, np.dtype[np.float32]],
    eps: float = 1e-8,
) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    n = min(len(reference), len(estimate))
    if n == 0:
        return 0.0

    ref = reference[:n] - np.mean(reference[:n])
    est = estimate[:n] - np.mean(estimate[:n])

    ref_energy = np.sum(ref**2) + eps
    # Optimal scaling factor alpha
    alpha = np.dot(est, ref) / ref_energy
    target = alpha * ref
    noise = est - target

    target_pow = np.sum(target**2) + eps
    noise_pow = np.sum(noise**2) + eps

    return float(10.0 * np.log10(target_pow / noise_pow))


def calculate_snr(
    clean: np.ndarray[Any, np.dtype[np.float32]],
    noisy: np.ndarray[Any, np.dtype[np.float32]],
) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    n = min(len(clean), len(noisy))
    if n == 0:
        return 0.0

    s = clean[:n]
    noise = noisy[:n] - s

    p_signal = float(np.mean(s**2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12

    return float(10.0 * np.log10(p_signal / p_noise))
