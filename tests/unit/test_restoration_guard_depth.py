"""How much of Guard R is actually live on the shipped checkpoint.

Three of the four high-band layers -- non-speech leakage, spurious bursts and
HF envelope divergence -- are gated on an absolute audibility floor (0.002 RMS,
-54 dBFS) so they never correlate numerical residue. The committed checkpoint
never saw real speech (RISKS R-14) and adds a high band at roughly -73 dBFS,
about 19 dB under that floor, so those three layers cannot fire and only the
impulse-discontinuity layer judges anything.

This is a tripwire, not a guarantee. It is expected to fail the day a trained
checkpoint is dropped in -- and that failure is the point: it marks the moment
three dormant layers become load-bearing and need validating against a real
generated high band for the first time.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.highband_events import HighBandEventDetector
from hawavoclean.restoration.profiles import load_speaker_profile

pytestmark = pytest.mark.unit

SR = 48000
AUDIBILITY_FLOOR_RMS = 0.002
REPO = Path(__file__).resolve().parents[2]


def _band_limited(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    harmonic = 1
    while 150.0 * harmonic < SR / 2:
        x += (1.0 / harmonic**1.5) * np.sin(
            2 * np.pi * 150.0 * harmonic * t + rng.uniform(0, 2 * np.pi)
        )
        harmonic += 1
    env = np.clip(0.5 + 0.8 * np.sin(2 * np.pi * 2.3 * t), 0, None)
    x = (x / np.max(np.abs(x)) * 0.6 * env).astype(np.float32)
    sos = signal.butter(10, 3800 / (SR / 2), btype="lowpass", output="sos")
    return np.asarray(signal.sosfiltfilt(sos, x), dtype=np.float32)


def test_the_shipped_checkpoint_adds_less_than_the_guard_can_judge() -> None:
    """Record what restore mode actually produces, and what that leaves live."""
    profile = load_speaker_profile("character_01", profiles_root=REPO / "profiles")
    embedding = np.asarray(profile.embedding_vector, dtype=np.float32)
    natural = _band_limited(6.0, seed=31)
    cutoff = 4171.875

    restored = next(
        c
        for c in HawaRestoreKD(sample_rate=SR).restore(
            natural,
            sample_rate=SR,
            effective_cutoff_hz=cutoff,
            speaker_id="character_01",
            speaker_embedding=embedding,
            strengths=[1.0, 0.0],
        )
        if c.strength == 1.0
    ).audio

    high = signal.sosfiltfilt(
        signal.butter(8, cutoff / (SR / 2), btype="highpass", output="sos"), restored - natural
    )
    added_rms = float(np.sqrt(np.mean(np.asarray(high, dtype=np.float64) ** 2)))
    assert added_rms < AUDIBILITY_FLOOR_RMS, (
        f"the checkpoint now adds {added_rms:.6f} RMS of high band, at or above the "
        f"{AUDIBILITY_FLOOR_RMS} audibility floor. Three Guard R layers that have never "
        "judged an audible signal just became live: validate leakage, spurious bursts and "
        "envelope divergence against real output before trusting a PASS."
    )

    result = HighBandEventDetector(sample_rate=SR).evaluate(natural, restored, cutoff_hz=cutoff)
    assert result.speech_window_leakage == 0.0
    assert result.spurious_burst_count == 0
    assert result.hf_envelope_divergence == 0.0
    assert result.impulse_discontinuity_ratio > 0.0, "the one live layer must still measure"


def test_the_fixture_profiles_are_not_distinct_enough_for_the_speaker_layer() -> None:
    """The speaker layer cannot separate the speakers currently shipped.

    Guard R rejects a candidate whose speaker similarity falls under 0.75. The
    ten profiles are generated fixtures (RISKS R-14), and 14 of their 45
    pairings sit at or above that very threshold -- the closest, character_03
    against character_09, at 0.96. Measured end to end, character_05's own
    canonical audio passes when judged against character_07's embedding
    (their profiles are 0.91 apart), which is the misattribution the layer
    exists to prevent.

    ``validate_all_profiles`` would never notice: its "embedding distinctness"
    compares SHA-256 hashes, so any two vectors that are not byte-identical
    pass however close they point.

    A tripwire, like the audibility one above. It is expected to fail when real
    consented speakers replace the fixtures (U3) -- and that failure is the
    point: it marks the moment the speaker layer can finally do its job, and
    the moment this number becomes worth trusting.
    """
    ids = [f"character_{i:02d}" for i in range(1, 11)]
    embeddings = {
        s: np.asarray(
            load_speaker_profile(s, profiles_root=REPO / "profiles").embedding_vector,
            dtype=np.float64,
        )
        for s in ids
    }

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    pairs = [
        (cosine(embeddings[a], embeddings[b]), a, b)
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
    ]
    confusable = [p for p in pairs if p[0] >= 0.75]

    assert confusable, (
        "the fixture profiles now separate at the guard's own threshold. The "
        "speaker-identity layer has become able to discriminate: validate it "
        "against real audio, and revisit RISKS R-14."
    )
    assert len(pairs) == 45
