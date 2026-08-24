"""Deterministic builder for the 10 synthetic Kurdish speaker character fixtures.

Every voice produced here is a synthetic formant-filtered pulse train — no human
speaker exists behind any profile. Generates distinct, hash-verified
192-dimensional acoustic prototype embeddings, F0 envelope profiles, canonical
audio manifests, and fixture-gating consent records. The consent records emitted
here mark the fixtures as synthetic: `consent_granted=true` gates pipeline use of
the fixture and is not a claim of human consent. Real speaker enrollment with
genuine consent records is required before any consented-speaker claim
(user checkpoint U3).
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from hawavoclean.hashing import hash_file
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor

SPEAKER_DEFINITIONS = [
    {
        "id": "character_01",
        "name": "Kurdish Speaker 01 (Male Deep)",
        "f0_median": 115.0,
        "f0_p05": 85.0,
        "f0_p95": 155.0,
        "formants": [500, 1500, 2500, 3500],
    },
    {
        "id": "character_02",
        "name": "Kurdish Speaker 02 (Female Clear)",
        "f0_median": 215.0,
        "f0_p05": 175.0,
        "f0_p95": 275.0,
        "formants": [650, 1850, 2900, 4100],
    },
    {
        "id": "character_03",
        "name": "Kurdish Speaker 03 (Male Baritone)",
        "f0_median": 130.0,
        "f0_p05": 95.0,
        "f0_p95": 175.0,
        "formants": [550, 1600, 2600, 3600],
    },
    {
        "id": "character_04",
        "name": "Kurdish Speaker 04 (Female Warm)",
        "f0_median": 195.0,
        "f0_p05": 155.0,
        "f0_p95": 255.0,
        "formants": [600, 1750, 2800, 3950],
    },
    {
        "id": "character_05",
        "name": "Kurdish Speaker 05 (Male Tenor)",
        "f0_median": 155.0,
        "f0_p05": 115.0,
        "f0_p95": 205.0,
        "formants": [580, 1680, 2750, 3750],
    },
    {
        "id": "character_06",
        "name": "Kurdish Speaker 06 (Female Mezzo)",
        "f0_median": 225.0,
        "f0_p05": 185.0,
        "f0_p95": 285.0,
        "formants": [680, 1900, 3050, 4200],
    },
    {
        "id": "character_07",
        "name": "Kurdish Speaker 07 (Male Crisp)",
        "f0_median": 140.0,
        "f0_p05": 105.0,
        "f0_p95": 190.0,
        "formants": [560, 1620, 2680, 3680],
    },
    {
        "id": "character_08",
        "name": "Kurdish Speaker 08 (Female Bright)",
        "f0_median": 240.0,
        "f0_p05": 195.0,
        "f0_p95": 305.0,
        "formants": [720, 2050, 3200, 4400],
    },
    {
        "id": "character_09",
        "name": "Kurdish Speaker 09 (Male Resonant)",
        "f0_median": 125.0,
        "f0_p05": 90.0,
        "f0_p95": 165.0,
        "formants": [520, 1550, 2550, 3550],
    },
    {
        "id": "character_10",
        "name": "Kurdish Speaker 10 (Female Dynamic)",
        "f0_median": 205.0,
        "f0_p05": 165.0,
        "f0_p95": 265.0,
        "formants": [640, 1800, 2850, 4000],
    },
]


def synthesize_canonical_speech(
    spk_def: dict, duration_s: float = 3.0, sr: int = 48000
) -> np.ndarray:
    """Synthesize distinct acoustic voice audio with formant filtering."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    f0 = spk_def["f0_median"]

    # Harmonic glottal pulse train
    sig = np.zeros_like(t)
    for h in range(1, 40):
        freq = h * f0
        if freq >= sr / 2:
            break
        amp = (1.0 / (h**0.8)) * np.exp(-freq / 4000.0)
        sig += amp * np.sin(2 * np.pi * freq * t + 0.1 * h)

    # Apply formant resonances
    for f_center in spk_def["formants"]:
        from scipy import signal

        sos = signal.butter(
            2,
            [max(100.0, f_center - 150.0), min(sr / 2 - 500.0, f_center + 150.0)],
            btype="bandpass",
            fs=sr,
            output="sos",
        )
        sig += 0.3 * signal.sosfiltfilt(sos, sig)

    sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.7
    return sig.astype(np.float32)


def build_all_profiles(profiles_root: Path) -> None:
    """Build and validate all 10 speaker character profiles."""
    profiles_root.mkdir(parents=True, exist_ok=True)
    extractor = SpeakerEmbeddingExtractor(sample_rate=48000)

    for spk_def in SPEAKER_DEFINITIONS:
        spk_id = spk_def["id"]
        spk_dir = profiles_root / spk_id
        spk_dir.mkdir(parents=True, exist_ok=True)

        canonical_dir = spk_dir / "canonical"
        consent_dir = spk_dir / "consent"
        embedding_dir = spk_dir / "embedding"
        canonical_dir.mkdir(exist_ok=True)
        consent_dir.mkdir(exist_ok=True)
        embedding_dir.mkdir(exist_ok=True)

        # 1. Generate canonical audio files
        audio = synthesize_canonical_speech(spk_def, duration_s=3.0, sr=48000)
        audio_path = canonical_dir / f"{spk_id}_ref.wav"
        sf.write(audio_path, audio, 48000, subtype="PCM_24")
        audio_hash = hash_file(audio_path)

        # 2. Write canonical manifest
        manifest_path = canonical_dir / "canonical.jsonl"
        manifest_entry = {
            "utt_id": f"{spk_id}_ref",
            "audio_path": f"canonical/{spk_id}_ref.wav",
            "duration_s": 3.0,
            "sample_rate": 48000,
            "sha256": audio_hash,
            "snr_db": 45.0,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        # 3. Extract real 192-dim acoustic prototype embedding
        embedding = extractor.extract(audio)
        emb_path = embedding_dir / "profile.npy"
        np.save(emb_path, embedding)
        emb_hash = hash_file(emb_path)

        # 4. Write consent record (synthetic fixture gate, not a human consent claim)
        consent_path = consent_dir / "consent.json"
        consent_data = {
            "speaker_id": spk_id,
            "display_name": spk_def["name"],
            "consent_granted": True,
            "consent_type": "synthetic_development_fixture",
            "synthetic": True,
            "date": "2026-08-22",
            "generated_by": "research/restoration/profiles_builder.py",
            "note": (
                "This voice is a synthetic formant-filtered pulse train generated by "
                "research/restoration/profiles_builder.py; no human speaker exists and "
                "no human consent was collected. consent_granted=true gates pipeline "
                "use of this fixture only; it is not a claim of human consent. Real "
                "speaker enrollment with genuine consent records is required before "
                "any consented-speaker claim (user checkpoint U3)."
            ),
        }
        with open(consent_path, "w", encoding="utf-8") as f:
            json.dump(consent_data, f, indent=2)

        # 5. Write master profile.json
        profile_data = {
            "schema_version": "1.0",
            "speaker_id": spk_id,
            "display_name": spk_def["name"],
            "consent_record": "consent/consent.json",
            "canonical_audio_manifest": "canonical/canonical.jsonl",
            "canonical_audio_sha256": [audio_hash],
            "profile_embedding_path": "embedding/profile.npy",
            "profile_embedding_sha256": emb_hash,
            "f0_statistics": {
                "median_hz": spk_def["f0_median"],
                "p05_hz": spk_def["f0_p05"],
                "p95_hz": spk_def["f0_p95"],
            },
            "training_split_id": "restore-corpus-v1",
            "adapter": None,
            "created_by_commit": "2b93176",
            "notes": (
                "Synthetic development fixture: formant-filtered pulse train generated "
                "by research/restoration/profiles_builder.py; no human speaker exists. "
                "The embedding is the extracted acoustic prototype vector of the "
                "synthetic audio."
            ),
        }
        with open(spk_dir / "profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        print(f"[OK] Generated profile {spk_id}: embedding SHA-256 = {emb_hash[:12]}...")


if __name__ == "__main__":
    build_all_profiles(Path("profiles"))
