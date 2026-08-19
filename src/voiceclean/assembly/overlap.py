"""Equal-power crossfading and boundary overlap curves."""

from typing import Any

import numpy as np


def compute_equal_power_crossfade(
    fade_len_samples: int,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """Generate equal-power (sin/cos) fade-out and fade-in curves.

    sin^2(t) + cos^2(t) = 1.0 preserves constant perceived energy across crossfade.
    """
    if fade_len_samples <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    t = np.linspace(0.0, np.pi / 2.0, fade_len_samples, dtype=np.float32)
    fade_out = np.cos(t).astype(np.float32)
    fade_in = np.sin(t).astype(np.float32)

    return fade_out, fade_in


def crossfade_signals(
    sig_a: np.ndarray[Any, np.dtype[np.float32]],
    sig_b: np.ndarray[Any, np.dtype[np.float32]],
    fade_len_samples: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Crossfade the tail of sig_a with the head of sig_b over fade_len_samples."""
    fade_len = min(fade_len_samples, len(sig_a), len(sig_b))
    if fade_len <= 0:
        return np.concatenate([sig_a, sig_b])

    fade_out, fade_in = compute_equal_power_crossfade(fade_len)

    # Crossfade overlap region
    tail_a = sig_a[-fade_len:] * fade_out
    head_b = sig_b[:fade_len] * fade_in
    blend = tail_a + head_b

    return np.concatenate([sig_a[:-fade_len], blend, sig_b[fade_len:]]).astype(np.float32)
