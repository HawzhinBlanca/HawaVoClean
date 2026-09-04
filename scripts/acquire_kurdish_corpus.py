#!/usr/bin/env python3
"""Acquire and prepare Kurdish multi-speaker speech datasets for HawaVoClean.

Fetches curated audio clips from open Kurdish speech datasets on Hugging Face
(such as aranemini/central-kurdish-tts4all or google/fleurs), standardizes
the audio to HawaVoClean's native 48 kHz / 24-bit mono PCM WAV, records
cryptographic SHA-256 manifests, and optionally enrolls validated speaker
profiles into the profiles directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

from hawavoclean.enrollment import _resample_to_48k, enroll_speaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("acquire_kurdish_corpus")

TTS4ALL_REPO = "aranemini/central-kurdish-tts4all"
DEFAULT_SPEAKERS = ["fatih", "shahen", "giganet"]


@dataclass(frozen=True)
class AcquiredClip:
    """Metadata for a standardized speech clip."""

    speaker_id: str
    original_filename: str
    local_path: str
    duration_s: float
    sample_rate: int
    channels: int
    sha256: str


@dataclass(frozen=True)
class AcquisitionSummary:
    """Summary of the corpus acquisition run."""

    source_repo: str
    speakers: list[str]
    total_clips: int
    total_duration_s: float
    output_dir: str
    manifest_path: str


def compute_file_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def convert_to_canonical_wav(
    source_path: Path,
    dest_path: Path,
    target_sr: int = 48000,
) -> tuple[float, str]:
    """Convert input audio to 48 kHz 24-bit mono PCM WAV and return (duration_s, sha256)."""
    audio, sr = sf.read(str(source_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio_48k = _resample_to_48k(audio.astype(np.float32), int(sr))
    duration_s = float(len(audio_48k) / target_sr)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(dest_path),
        audio_48k,
        target_sr,
        subtype="PCM_24",
        format="WAV",
    )
    digest = compute_file_sha256(dest_path)
    return duration_s, digest


def fetch_tts4all_speaker_clips(
    speaker: str,
    max_clips: int,
    output_speaker_dir: Path,
) -> list[AcquiredClip]:
    """Fetch and standardize clips for a specific speaker from central-kurdish-tts4all."""
    api = HfApi()
    info = api.dataset_info(TTS4ALL_REPO)
    prefix = f"wavs/{speaker}/"
    siblings = info.siblings or []
    available_files = sorted(
        s.rfilename
        for s in siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".wav")
    )

    if not available_files:
        raise ValueError(f"No WAV files found for speaker '{speaker}' in {TTS4ALL_REPO}")

    target_files = available_files[:max_clips]
    logger.info(
        f"Downloading {len(target_files)} clips for speaker '{speaker}' from {TTS4ALL_REPO}..."
    )

    clips: list[AcquiredClip] = []
    output_speaker_dir.mkdir(parents=True, exist_ok=True)

    for rfilename in target_files:
        basename = Path(rfilename).name
        dest_wav = output_speaker_dir / basename

        # Download raw file from Hugging Face
        cached_path = Path(
            hf_hub_download(
                repo_id=TTS4ALL_REPO,
                filename=rfilename,
                repo_type="dataset",
            )
        )

        # Standardize to 48 kHz / 24-bit mono WAV
        dur_s, digest = convert_to_canonical_wav(cached_path, dest_wav, target_sr=48000)
        clips.append(
            AcquiredClip(
                speaker_id=speaker,
                original_filename=rfilename,
                local_path=str(dest_wav),
                duration_s=round(dur_s, 3),
                sample_rate=48000,
                channels=1,
                sha256=digest,
            )
        )

    return clips


def acquire_corpus(
    speakers: list[str],
    max_clips_per_speaker: int,
    output_dir: Path,
    enroll: bool = False,
    profiles_dir: Path | None = None,
    min_enroll_duration_s: float = 10.0,
    min_enroll_sessions: int = 3,
) -> AcquisitionSummary:
    """Download, convert, and optionally enroll Kurdish speakers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_clips: list[AcquiredClip] = []

    for spk in speakers:
        spk_dir = output_dir / spk
        spk_clips = fetch_tts4all_speaker_clips(spk, max_clips_per_speaker, spk_dir)
        all_clips.extend(spk_clips)
        spk_duration = sum(c.duration_s for c in spk_clips)
        logger.info(f"Speaker '{spk}': {len(spk_clips)} clips, total {spk_duration:.2f}s acquired.")

        if enroll and profiles_dir is not None:
            target_profile = profiles_dir / f"kurdish_{spk}"
            logger.info(f"Enrolling speaker profile for '{spk}' at {target_profile}...")
            enroll_speaker(
                speaker_id=f"kurdish_{spk}",
                display_name=f"Kurdish {spk.capitalize()} (Central)",
                audio_dir=spk_dir,
                output_dir=target_profile,
                consent_granted=True,
                consent_note="Acquired from open Kurdish multi-speaker corpus (aranemini/central-kurdish-tts4all).",
                min_duration_s=min_enroll_duration_s,
                min_sessions=min_enroll_sessions,
                verbose=False,
            )
            logger.info(f"Successfully enrolled profile: {target_profile}")

    manifest_path = output_dir / "acquisition_manifest.json"
    total_dur = float(round(sum(c.duration_s for c in all_clips), 3))
    manifest_data = {
        "source_repo": TTS4ALL_REPO,
        "speakers": speakers,
        "total_clips": len(all_clips),
        "total_duration_s": total_dur,
        "clips": [asdict(c) for c in all_clips],
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Acquisition manifest saved to: {manifest_path}")

    return AcquisitionSummary(
        source_repo=TTS4ALL_REPO,
        speakers=speakers,
        total_clips=len(all_clips),
        total_duration_s=total_dur,
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Acquire and prepare Kurdish speech datasets from Hugging Face for HawaVoClean."
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=DEFAULT_SPEAKERS,
        help=f"Speakers to acquire (default: {' '.join(DEFAULT_SPEAKERS)}).",
    )
    parser.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=5,
        help="Maximum clips to download per speaker (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kurdish_speakers"),
        help="Directory to store standardized 48kHz WAV files (default: data/kurdish_speakers).",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Automatically enroll downloaded speakers into profiles/ directory.",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("profiles"),
        help="Profiles directory for voice enrollment (default: profiles).",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=10.0,
        help="Minimum required duration in seconds for enrollment (default: 10.0).",
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=3,
        help="Minimum required distinct audio sessions for enrollment (default: 3).",
    )

    args = parser.parse_args(argv)

    try:
        summary = acquire_corpus(
            speakers=args.speakers,
            max_clips_per_speaker=args.max_clips_per_speaker,
            output_dir=args.output_dir,
            enroll=args.enroll,
            profiles_dir=args.profiles_dir if args.enroll else None,
            min_enroll_duration_s=args.min_duration_s,
            min_enroll_sessions=args.min_sessions,
        )
        print("\n=== Acquisition Complete ===")
        print(f"Source: {summary.source_repo}")
        print(f"Speakers: {', '.join(summary.speakers)}")
        print(f"Total clips: {summary.total_clips}")
        print(f"Total duration: {summary.total_duration_s:.2f}s")
        print(f"Manifest: {summary.manifest_path}")
        return 0
    except Exception as e:
        logger.error(f"Acquisition failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
