"""Unit tests for HawaRestore-KD backbone model."""

from pathlib import Path

import numpy as np
import pytest
import scipy.signal as signal

from hawavoclean.errors import ModelProvenanceError
from hawavoclean.hashing import hash_file
from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD
from hawavoclean.restoration.profiles import load_speaker_profile


def test_hawarestore_kd_generates_all_candidate_strengths() -> None:
    """Verify that HawaRestore-KD returns candidates for ladder [1.0, 0.75, 0.5, 0.25, 0.0]."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(
        np.float32
    )
    sos = signal.butter(6, 4000 / 24000, btype="lowpass", output="sos")
    lp_sig = signal.sosfiltfilt(sos, sig).astype(np.float32)

    profile = load_speaker_profile("character_01")
    restorer = HawaRestoreKD(sample_rate=sr)

    cands = restorer.restore(
        lp_sig,
        sample_rate=sr,
        effective_cutoff_hz=4000.0,
        speaker_id=profile.speaker_id,
        speaker_embedding=profile.embedding_vector,
        seed=123,
    )

    assert len(cands) == 5
    strengths = [c.strength for c in cands]
    assert strengths == [1.0, 0.75, 0.5, 0.25, 0.0]

    # Strength 0.0 must be the unchanged Natural input
    np.testing.assert_allclose(cands[-1].audio, lp_sig, atol=1e-6)


def test_hawarestore_kd_deterministic_reproducibility() -> None:
    """Verify that identical seed and inputs produce bit-identical restoration candidates."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(
        np.float32
    )

    restorer = HawaRestoreKD(sample_rate=sr)
    cands_a = restorer.restore(sig, sample_rate=sr, effective_cutoff_hz=4000.0, seed=42)
    cands_b = restorer.restore(sig, sample_rate=sr, effective_cutoff_hz=4000.0, seed=42)

    for ca, cb in zip(cands_a, cands_b, strict=True):
        np.testing.assert_array_equal(ca.audio, cb.audio)


def test_hawarestore_kd_chunks_long_input() -> None:
    """Audio longer than one block must be processed in overlapping blocks.

    The whole-file spectrogram cannot go through the network: measured end to
    end it cost roughly 1 GB of RAM per second of audio, so a few minutes of
    speech exhausted the machine before this was chunked.
    """
    sr = 48000
    restorer = HawaRestoreKD(sample_rate=sr, chunk_seconds=0.5)
    n_samples = int(sr * 3.0)

    positions = restorer._block_positions(n_samples)

    assert len(positions) > 1, "a 3 s signal must not be restored as a single block"
    assert positions[0] == 0
    assert positions[-1] + restorer._block_len() >= n_samples, "blocks must reach the end"
    # Consecutive blocks overlap, so the cross-fade always has material to work with.
    for earlier, later in zip(positions, positions[1:], strict=False):
        assert later > earlier
        assert later < earlier + restorer._block_len()


def test_hawarestore_kd_multi_block_output_is_continuous() -> None:
    """A chunked restoration must not leave a click at a block seam."""
    sr = 48000
    t = np.arange(int(sr * 2.5), dtype=np.float32) / sr
    sig = (0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1300 * t)).astype(
        np.float32
    )
    sos = signal.butter(6, 4000 / 24000, btype="lowpass", output="sos")
    lp_sig = signal.sosfiltfilt(sos, sig).astype(np.float32)

    restorer = HawaRestoreKD(sample_rate=sr, chunk_seconds=0.5)
    cands = restorer.restore(
        lp_sig, sample_rate=sr, effective_cutoff_hz=4000.0, strengths=[1.0, 0.0], seed=7
    )

    restored = next(c.audio for c in cands if c.strength == 1.0)
    assert restored.shape == lp_sig.shape

    # A seam discontinuity shows up as a sample-to-sample jump far larger than
    # anything the source signal contains.
    src_jump = float(np.max(np.abs(np.diff(lp_sig))))
    out_jump = float(np.max(np.abs(np.diff(restored))))
    assert out_jump < src_jump * 10.0, f"seam click: {out_jump:.4f} vs source {src_jump:.4f}"


def test_hawarestore_kd_refuses_missing_checkpoint(tmp_path: Path) -> None:
    """Restore mode must fail closed rather than run on untrained weights.

    Silently keeping the random initialisation would publish synthesised audio
    while the report still attests a checkpoint hash.
    """
    with pytest.raises(ModelProvenanceError, match="checkpoint not found"):
        HawaRestoreKD(sample_rate=48000, checkpoint_path=tmp_path / "absent.pt")


def test_hawarestore_kd_refuses_corrupt_checkpoint(tmp_path: Path) -> None:
    """A checkpoint that cannot be loaded must raise, not fall back to random weights."""
    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"not a torch checkpoint")

    with pytest.raises(ModelProvenanceError, match="Failed to load"):
        HawaRestoreKD(sample_rate=48000, checkpoint_path=bad)


def test_hawarestore_kd_reports_hash_of_loaded_weights() -> None:
    """The restorer's attested hash must be the file it actually loaded."""
    restorer = HawaRestoreKD(sample_rate=48000)
    assert restorer.weights_sha256 == hash_file(restorer.checkpoint_path)
    assert restorer.device == "cpu", "the default device must be reproducible across machines"
