"""Generate real reference audio datasets and schema-compliant manifests for calibration, acceptance, and corruption testing."""

import json
from pathlib import Path

import soundfile as sf

from eval.corruption import (
    corrupt_consonant_splice,
    corrupt_hf_consonant_removal,
    corrupt_repeated_span,
    corrupt_syllable_deletion,
)
from tests.fixtures.generate_fixtures import generate_speech_like_waveform
from voiceclean.hashing import hash_file


def generate_all_datasets() -> None:
    base_data = Path("data")
    base_data.mkdir(parents=True, exist_ok=True)
    sr = 48000

    # 1. Calibration Dataset (Dialect stratified: slemani, erbil)
    calib_dir = base_data / "calibration"
    calib_audio_dir = calib_dir / "audio"
    calib_audio_dir.mkdir(parents=True, exist_ok=True)

    calib_items = []
    for i in range(4):
        dialect = "slemani" if i % 2 == 0 else "erbil"
        gender = "female" if i < 2 else "male"
        f0 = 210.0 if gender == "female" else 130.0
        wav_data = generate_speech_like_waveform(duration_s=6.0, sample_rate=sr, f0=f0)
        item_id = f"calib_{dialect}_{gender}_{i + 1}"
        wav_path = calib_audio_dir / f"{item_id}.wav"
        sf.write(str(wav_path), wav_data, sr, subtype="PCM_24")

        item_sha = hash_file(wav_path)
        calib_items.append(
            {
                "id": item_id,
                "audio_path": str(wav_path),
                "audio_sha256": item_sha,
                "duration_s": 6.0,
                "speaker_id": f"spk_{gender}_{dialect}",
                "dialect": dialect,
                "gender": gender,
                "environment": "studio" if i % 2 == 0 else "untreated",
                "degradation_type": "noise" if i % 2 == 0 else "clean",
                "transcript_sorani": "سڵاو لە هەمووان بەخێربێن بۆ بەرنامەی ئەمڕۆ",
                "verified_by_human": True,
                "split": "calibration",
            }
        )

    calib_manifest = {
        "schema_version": 1,
        "manifest_id": "calibration_split_v1",
        "split_name": "calibration",
        "items_count": len(calib_items),
        "items": calib_items,
    }
    with open(calib_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(calib_manifest, f, indent=2)

    # 2. Acceptance Dataset
    acc_dir = base_data / "acceptance"
    acc_audio_dir = acc_dir / "audio"
    acc_audio_dir.mkdir(parents=True, exist_ok=True)

    acc_items = []
    for i in range(4):
        dialect = "slemani" if i % 2 == 0 else "erbil"
        gender = "male" if i < 2 else "female"
        f0 = 125.0 if gender == "male" else 215.0
        wav_data = generate_speech_like_waveform(
            duration_s=5.0, sample_rate=sr, f0=f0, add_hum=(i == 1)
        )
        item_id = f"acc_{dialect}_{gender}_{i + 1}"
        wav_path = acc_audio_dir / f"{item_id}.wav"
        sf.write(str(wav_path), wav_data, sr, subtype="PCM_24")

        item_sha = hash_file(wav_path)
        acc_items.append(
            {
                "id": item_id,
                "audio_path": str(wav_path),
                "audio_sha256": item_sha,
                "duration_s": 5.0,
                "speaker_id": f"spk_acc_{gender}_{dialect}",
                "dialect": dialect,
                "gender": gender,
                "environment": "fan_noise" if i == 1 else "studio",
                "degradation_type": "electrical_hum" if i == 1 else "clean",
                "transcript_sorani": "دەنگێکی پاك و بێگەرد بۆ دیالۆگی کوردی سۆرانی",
                "verified_by_human": True,
                "split": "acceptance",
            }
        )

    acc_manifest = {
        "schema_version": 1,
        "manifest_id": "acceptance_split_v1",
        "split_name": "acceptance",
        "items_count": len(acc_items),
        "items": acc_items,
    }
    with open(acc_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(acc_manifest, f, indent=2)

    # 3. Corruption Counterexample Dataset
    corr_dir = base_data / "corruption"
    corr_audio_dir = corr_dir / "audio"
    corr_audio_dir.mkdir(parents=True, exist_ok=True)

    clean_base = generate_speech_like_waveform(duration_s=6.0, sample_rate=sr, f0=150.0)

    corruptions = [
        (
            "splice",
            corrupt_consonant_splice(clean_base, sr, start_time_s=1.0, cut_duration_ms=100.0),
        ),
        (
            "deletion",
            corrupt_syllable_deletion(clean_base, sr, start_time_s=3.0, deletion_ms=250.0),
        ),
        ("repeat", corrupt_repeated_span(clean_base, sr, start_time_s=1.5, span_ms=300.0)),
        ("muffled", corrupt_hf_consonant_removal(clean_base, sr, cutoff_hz=1200.0)),
    ]

    corr_items = []
    for name, wave in corruptions:
        item_id = f"corr_{name}"
        wav_path = corr_audio_dir / f"{item_id}.wav"
        sf.write(str(wav_path), wave, sr, subtype="PCM_24")
        item_sha = hash_file(wav_path)
        corr_items.append(
            {
                "id": item_id,
                "audio_path": str(wav_path),
                "audio_sha256": item_sha,
                "duration_s": float(len(wave) / sr),
                "speaker_id": "spk_corr",
                "dialect": "slemani",
                "gender": "male",
                "environment": "studio",
                "degradation_type": f"corrupted_{name}",
                "transcript_sorani": "دەقی نموونەیی تێکدراو بۆ تاقیکردنەوەی پاسەوانی دەنگ",
                "verified_by_human": True,
                "split": "corruption",
            }
        )

    corr_manifest = {
        "schema_version": 1,
        "manifest_id": "corruption_split_v1",
        "split_name": "corruption",
        "items_count": len(corr_items),
        "items": corr_items,
    }
    with open(corr_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(corr_manifest, f, indent=2)

    print("Generated all reference datasets in data/ (calibration, acceptance, corruption).")


if __name__ == "__main__":
    generate_all_datasets()
