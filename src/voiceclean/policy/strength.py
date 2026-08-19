"""Residual strength candidate generation for phase-coherent cores."""

from typing import Any

import numpy as np


def generate_strength_candidates(
    orig_waveform: np.ndarray[Any, np.dtype[np.float32]],
    enh_waveform: np.ndarray[Any, np.dtype[np.float32]],
    strength_ladder: list[float],
    phase_coherent: bool = True,
) -> list[tuple[float, np.ndarray[Any, np.dtype[np.float32]]]]:
    """Derive linear residual blends for phase-coherent cores, or single candidate for incoherent cores."""
    orig_len = len(orig_waveform)
    enh_len = len(enh_waveform)
    n = min(orig_len, enh_len)

    w_orig = orig_waveform[:n]
    w_enh = enh_waveform[:n]

    if not phase_coherent:
        # Phase-incoherent / reconstructed core: binary candidate only (s=1.0)
        return [(1.0, w_enh.copy())]

    candidates: list[tuple[float, np.ndarray[Any, np.dtype[np.float32]]]] = []
    residual = w_enh - w_orig

    for s in sorted(strength_ladder, reverse=True):
        cand = w_enh.copy() if abs(s - 1.0) < 1e-4 else w_orig + (float(s) * residual)
        candidates.append((float(s), np.ascontiguousarray(cand, dtype=np.float32)))

    return candidates
