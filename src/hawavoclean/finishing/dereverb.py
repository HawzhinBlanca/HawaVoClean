"""Decay-gated late-reverberation suppression.

Late reverb is modelled per STFT bin as the recent signal power decayed at
the configured RT60 (Lebart / Habets late-reverb model). The estimate is
subtracted ONLY inside decays — frames quieter than their recent past — and
scaled by decay depth; frames at or near a local peak (the voice itself)
are left untouched. This is deliberately conservative: single-channel
dereverberation that acts on the voice itself collapses its level
broadband (measured: -7 to -8 dB in every band at WPE taps >= 40).

It cannot remove sustained content between phrases (music beds, held
notes): those are not reverb and are not modelled.
"""

from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal


def suppress_late_reverb(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    rt60_s: float = 0.5,
    floor_db: float = -15.0,
    delay_frames: int = 2,
    onset_protect_db: float = 4.0,
    n_fft: int = 1024,
    hop: int = 256,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return the waveform with decay-gated late-reverb suppression applied."""
    x = np.asarray(waveform, dtype=np.float32)
    if len(x) < n_fft * 2:
        return x.copy()

    _, _, z = scipy.signal.stft(x, sample_rate, nperseg=n_fft, noverlap=n_fft - hop, padded=True)
    power = np.abs(z) ** 2

    # Energy decay per hop for the given RT60 (60 dB = ln(1e6) ≈ 13.8 in power)
    alpha = float(np.exp(-13.816 * hop / (rt60_s * sample_rate)))
    late = np.zeros_like(power)
    acc = np.zeros(power.shape[0])
    for i in range(power.shape[1]):
        acc = alpha * acc + (1.0 - alpha) * power[:, i]
        j = i + delay_frames
        if j < power.shape[1]:
            late[:, j] = acc * (alpha**delay_frames)

    # Decay depth: how far (in dB) the frame sits below the running max of
    # the last ~150 ms. 0 at a local peak (the voice); ramps to 1 once the
    # frame is `onset_protect_db` down. Expressed in dB so the gate engages
    # at the same point regardless of how steep the room's decay is (a
    # linear-ratio gate engaged on fast decays and never on smooth ones).
    frame_power = power.sum(axis=0)
    win = max(1, int(0.15 * sample_rate / hop))
    # LOOK-BACK window [i-win+1, i]: origin=+(win//2) (the limiter uses the
    # opposite sign for its look-AHEAD window).
    recent_max = scipy.ndimage.maximum_filter1d(
        frame_power, size=win, origin=(win - 1) // 2, mode="nearest"
    )
    drop_db = 10.0 * np.log10((recent_max + 1e-12) / (frame_power + 1e-12))
    # Within 6 dB of the recent peak is the voice itself (syllabic modulation
    # inside a phrase) — untouched. Swept on synthetic dry/wet speech: 6 dB
    # protect + 3 dB onset leaves dry voice at -0.27 dB while tightening a
    # wet decay by ~6 dB; narrower bands (1.5 dB) dimmed dry speech 1.7 dB.
    protect_db = 6.0
    full_db = protect_db + max(0.5, onset_protect_db)  # fully engaged beyond this
    depth = np.clip((drop_db - protect_db) / (full_db - protect_db), 0.0, 1.0)

    floor = 10.0 ** (floor_db / 10.0)
    gain = np.sqrt(np.maximum(1.0 - depth[None, :] * late / (power + 1e-12), floor))
    gain = scipy.ndimage.uniform_filter(gain, size=(3, 2))  # anti-musical-noise smoothing

    _, y = scipy.signal.istft(z * gain, sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
    y32 = np.asarray(y[: len(x)], dtype=np.float32)
    if len(y32) < len(x):
        y32 = np.pad(y32, (0, len(x) - len(y32)))
    return y32
