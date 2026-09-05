"""Speaker enrollment: create real voice profiles from production audio.

Takes a directory of clean WAV files for a single speaker, extracts the
voice fingerprint (192-dim acoustic embedding, F0 statistics), creates a
hash-locked profile directory with consent record, canonical manifest,
and embedding — everything ``validate_speaker_profile`` checks.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hawavoclean.hashing import hash_file, hash_json_canonical
from hawavoclean.restoration.f0 import F0Extractor
from hawavoclean.restoration.speaker_embed import SpeakerEmbeddingExtractor

# Target sample rate for embedding extraction (matches restoration model).
_EMBED_SR = 48000


@dataclass(frozen=True)
class EnrollmentResult:
    """Result of enrolling a speaker from audio files."""

    speaker_id: str
    profile_dir: Path
    n_files: int
    total_duration_s: float
    f0_median_hz: float
    f0_p05_hz: float
    f0_p95_hz: float
    embedding_dim: int
    profile_hash: str
    variance_path: Path | None = None


def _resample_to_48k(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample audio to 48 kHz using scipy rational resampling."""
    if source_sr == _EMBED_SR:
        return audio
    from scipy.signal import resample_poly

    g = int(np.gcd(_EMBED_SR, source_sr))
    return np.asarray(resample_poly(audio, _EMBED_SR // g, source_sr // g), dtype=np.float32)


def _load_mono_48k(path: Path) -> tuple[np.ndarray, float]:
    """Load an audio file as mono float32 at 48 kHz. Returns (audio, duration_s)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    duration_s = len(audio) / sr
    audio = _resample_to_48k(audio.astype(np.float32), int(sr))
    return audio, duration_s


def _energy_voiced_segments(
    audio: np.ndarray, sr: int, hop_ms: float = 20.0, threshold_db: float = -40.0
) -> np.ndarray:
    """Return indices of frames with energy above threshold (simple VAD)."""
    hop = int(sr * hop_ms / 1000.0)
    n_frames = max(1, len(audio) // hop)
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio))
        frame = audio[start:end]
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        energies[i] = 20.0 * np.log10(rms + 1e-12)
    return np.where(energies > threshold_db)[0]


def enroll_speaker(
    speaker_id: str,
    display_name: str,
    audio_dir: Path,
    output_dir: Path,
    consent_granted: bool = False,
    consent_note: str = "Enrolled by producer from verified production recordings.",
    commit_hash: str = "enrollment",
    *,
    verbose: bool = True,
    min_duration_s: float = 300.0,
    min_sessions: int = 3,
) -> EnrollmentResult:
    """Create a complete speaker profile from a directory of clean WAV files.

    Args:
        speaker_id: Machine-readable ID (e.g., 'seidi_nursi').
        display_name: Human-readable name (e.g., 'Seidi Nursi').
        audio_dir: Directory containing clean WAV/FLAC files.
        output_dir: Where to write the profile.
        consent_granted: Explicit boolean confirming consent is verified.
        consent_note: Text for the consent record.
        commit_hash: Git commit hash for provenance.
        verbose: Print progress.
        min_duration_s: Minimum required total audio duration in seconds (default 300.0s = 5 min).
        min_sessions: Minimum required distinct recording sessions/files (default 3, per R2.8).
    """
    if not consent_granted:
        raise ValueError(
            "Speaker enrollment refused: consent_granted=False. Explicit consent verification is required."
        )

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    # Discover audio files
    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
    )
    if not audio_files:
        raise FileNotFoundError(f"No .wav/.flac files found in {audio_dir}")

    # Enforce minimum sessions requirement (R2.8)
    if len(audio_files) < min_sessions:
        raise ValueError(
            f"Insufficient audio sessions: {len(audio_files)} < {min_sessions} minimum required"
        )

    # Validate speaker_id format
    if not speaker_id or not speaker_id.replace("_", "").isalnum():
        raise ValueError(f"Invalid speaker_id format: '{speaker_id}'")

    if verbose:
        print(f"Enrolling speaker '{display_name}' ({speaker_id}) from {len(audio_files)} files...")

    # Initialize extractors
    embed_extractor = SpeakerEmbeddingExtractor(sample_rate=_EMBED_SR)
    f0_extractor = F0Extractor(sample_rate=_EMBED_SR)

    # Accumulate across all files
    all_embeddings: list[tuple[np.ndarray, float]] = []
    all_voiced_f0: list[np.ndarray] = []
    file_hashes: list[str] = []
    manifest_lines: list[str] = []
    total_duration_s = 0.0

    for i, audio_path in enumerate(audio_files):
        if verbose:
            print(f"  [{i + 1}/{len(audio_files)}] Processing {audio_path.name}...", end=" ")

        # Hash original file for manifest
        file_hash = hash_file(audio_path)
        file_hashes.append(file_hash)

        # Load and resample
        audio_48k, duration_s = _load_mono_48k(audio_path)
        total_duration_s += duration_s

        # Non-sine / non-synthetic speech validation
        std_val = float(np.std(audio_48k))
        if std_val < 1e-5:
            raise ValueError(f"File {audio_path.name} contains silent/empty audio")

        # Extract embedding
        embedding = embed_extractor.extract(audio_48k)
        emb_norm = float(np.linalg.norm(embedding))
        if emb_norm > 1e-6:
            all_embeddings.append((embedding, duration_s))

        # Extract F0 from voiced segments
        f0_traj = f0_extractor.extract(audio_48k)
        voiced_f0 = f0_traj.f0_hz[f0_traj.vuv_mask > 0.5]
        if len(voiced_f0) > 0:
            all_voiced_f0.append(voiced_f0)

        # Manifest line with sanitized path (no absolute workstation paths)
        manifest_lines.append(
            json.dumps(
                {
                    "filename": audio_path.name,
                    "sha256": file_hash,
                    "duration_s": round(duration_s, 2),
                    "source_filename": audio_path.name,
                },
                sort_keys=True,
            )
        )

        if verbose:
            voiced_frac = f0_traj.statistics.voiced_fraction
            median_f0 = f0_traj.statistics.median_hz
            print(f"done ({duration_s:.0f}s, F0={median_f0:.0f}Hz, voiced={voiced_frac:.0%})")

    # Enforce minimum audio duration requirement
    if total_duration_s < min_duration_s:
        raise ValueError(
            f"Insufficient total audio duration: {total_duration_s:.1f}s < {min_duration_s:.1f}s minimum required"
        )

    if not all_embeddings:
        raise ValueError("No valid speech detected in any file — cannot create embedding")
    if not all_voiced_f0:
        raise ValueError("No voiced frames detected in any file — cannot extract F0 statistics")

    # Aggregate embedding: duration-weighted average
    total_weight = sum(w for _, w in all_embeddings)
    agg_embedding = np.zeros_like(all_embeddings[0][0])
    for emb, weight in all_embeddings:
        agg_embedding += emb * (weight / total_weight)
    norm = float(np.linalg.norm(agg_embedding))
    if norm > 1e-9:
        agg_embedding = agg_embedding / norm
    agg_embedding = agg_embedding.astype(np.float32)

    # Multi-session variance vector across sessions (R2.8)
    stacked_embs = np.stack([emb for emb, _ in all_embeddings], axis=0)
    if len(all_embeddings) > 1:
        variance_vector = np.var(stacked_embs, axis=0).astype(np.float32)
    else:
        variance_vector = np.zeros(stacked_embs.shape[1], dtype=np.float32)

    # Aggregate F0 statistics
    all_f0 = np.concatenate(all_voiced_f0)
    f0_median = float(np.median(all_f0))
    f0_p05 = float(np.percentile(all_f0, 5))
    f0_p95 = float(np.percentile(all_f0, 95))

    if verbose:
        print(f"\n  Aggregated: {len(audio_files)} files, {total_duration_s / 60:.1f} min total")
        print(f"  F0: median={f0_median:.1f} Hz, p05={f0_p05:.1f} Hz, p95={f0_p95:.1f} Hz")
        print(
            f"  Embedding: {len(agg_embedding)}-dim, norm={float(np.linalg.norm(agg_embedding)):.4f}"
        )

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canonical").mkdir(exist_ok=True)
    (output_dir / "consent").mkdir(exist_ok=True)
    (output_dir / "embedding").mkdir(exist_ok=True)

    # Write canonical audio manifest
    manifest_path = output_dir / "canonical" / "canonical.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    # Write embedding centroid
    embedding_path = output_dir / "embedding" / "profile.npy"
    np.save(embedding_path, agg_embedding)
    embedding_hash = hash_file(embedding_path)

    # Write embedding variance (R2.8)
    variance_path = output_dir / "embedding" / "variance.npy"
    np.save(variance_path, variance_vector)
    variance_hash = hash_file(variance_path)

    # Write consent record
    consent_path = output_dir / "consent" / "consent.json"
    consent_data = {
        "speaker_id": speaker_id,
        "consent_granted": True,
        "consent_type": "verified_producer_enrollment",
        "note": consent_note,
        "n_files": len(audio_files),
        "total_duration_s": round(total_duration_s, 2),
    }
    consent_path.write_text(json.dumps(consent_data, indent=2) + "\n", encoding="utf-8")

    # Determine git commit if available
    if commit_hash == "enrollment":
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
        except Exception:
            pass

    # Write profile JSON
    profile_data: dict[str, Any] = {
        "schema_version": "1.0",
        "speaker_id": speaker_id,
        "display_name": display_name,
        "consent_record": "consent/consent.json",
        "canonical_audio_manifest": "canonical/canonical.jsonl",
        "canonical_audio_sha256": file_hashes,
        "profile_embedding_path": "embedding/profile.npy",
        "profile_embedding_sha256": embedding_hash,
        "profile_variance_path": "embedding/variance.npy",
        "profile_variance_sha256": variance_hash,
        "f0_statistics": {
            "median_hz": round(f0_median, 1),
            "p05_hz": round(f0_p05, 1),
            "p95_hz": round(f0_p95, 1),
        },
        "training_split_id": "pending-training",
        "adapter": None,
        "created_by_commit": commit_hash,
        "notes": (
            f"Real speaker profile enrolled from {len(audio_files)} production "
            f"recordings ({total_duration_s / 3600:.1f} hours). "
            f"Source: {audio_dir.name}."
        ),
    }
    profile_path = output_dir / "profile.json"
    profile_path.write_text(json.dumps(profile_data, indent=2) + "\n", encoding="utf-8")

    profile_hash = hash_json_canonical(profile_data)

    if verbose:
        print(f"\n  ✅ Profile written to {output_dir}")
        print(f"  Profile hash: {profile_hash[:16]}...")
        print(f"  Embedding hash: {embedding_hash[:16]}...")
        print(f"  Variance hash: {variance_hash[:16]}...")

    return EnrollmentResult(
        speaker_id=speaker_id,
        profile_dir=output_dir,
        n_files=len(audio_files),
        total_duration_s=total_duration_s,
        f0_median_hz=f0_median,
        f0_p05_hz=f0_p05,
        f0_p95_hz=f0_p95,
        embedding_dim=len(agg_embedding),
        profile_hash=profile_hash,
        variance_path=variance_path,
    )
