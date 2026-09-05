"""Immediate sanity and integrity validation of raw enhancer outputs."""

from typing import Any

import numpy as np


def validate_enhancer_output(
    orig_waveform: np.ndarray[Any, np.dtype[np.float32]],
    cand_waveform: np.ndarray[Any, np.dtype[np.float32]],
    is_speech: bool,
    max_padding_diff_samples: int = 512,
) -> tuple[bool, str]:
    """Validate immediate output sanity before alignment or guard analysis."""
    if cand_waveform is None or not isinstance(cand_waveform, np.ndarray):
        return False, "Candidate output is not a valid numpy array."

    if cand_waveform.dtype != np.float32:
        return False, f"Expected float32 array, got {cand_waveform.dtype}"

    if not np.all(np.isfinite(cand_waveform)):
        return False, "Enhancer output contains NaN or Infinite sample values."

    orig_len = len(orig_waveform)
    cand_len = len(cand_waveform)

    if abs(cand_len - orig_len) > max_padding_diff_samples:
        return False, f"Output length mismatch ({cand_len} vs expected {orig_len})"

    # Check for complete signal collapse on speech units
    if is_speech and orig_len > 0:
        orig_rms = float(np.sqrt(np.mean(orig_waveform**2)))
        cand_rms = float(np.sqrt(np.mean(cand_waveform**2)))

        if orig_rms > 1e-3 and cand_rms < 1e-5:
            return (
                False,
                f"Speech signal collapsed to near-zero silence (RMS {cand_rms:.6f} vs input {orig_rms:.6f})",
            )

        if orig_rms > 0 and (cand_rms / (orig_rms + 1e-9)) > 10.0:
            return (
                False,
                f"Excessive RMS energy explosion ({cand_rms / orig_rms:.1f}x input energy)",
            )

    # Hard clipping check: if original was unclipped, candidate must not hard clip
    orig_max = float(np.max(np.abs(orig_waveform))) if orig_len > 0 else 0.0
    cand_max = float(np.max(np.abs(cand_waveform))) if cand_len > 0 else 0.0

    if orig_max < 0.99 and cand_max > 1.05:
        return (
            False,
            f"Newly introduced hard clipping detected (peak amplitude {cand_max:.3f} > 1.05)",
        )

    return True, ""
