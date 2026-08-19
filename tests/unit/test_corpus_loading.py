"""Corpus manifest loading formats and audio digest verification."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voiceclean.errors import InvalidUserInputError
from voiceclean.eval.corpus import load_corpus_manifest, verify_corpus_audio_files
from voiceclean.hashing import hash_file


def _item_dict(item_id: str, path: str, sha: str = "") -> dict[str, object]:
    return {
        "id": item_id,
        "audio_path": path,
        "audio_sha256": sha,
        "duration_s": 1.0,
        "speaker_id": "s",
        "dialect": "synthetic",
        "gender": "unknown",
        "environment": "synthetic",
        "degradation_type": "clean",
        "transcript_sorani": "",
        "verified_by_human": False,
        "split": "calibration",
    }


def test_load_jsonl_manifest(tmp_path: Path) -> None:
    m = tmp_path / "items.jsonl"
    lines = [json.dumps(_item_dict(f"b_{i}", f"x{i}.wav")) for i in (2, 1)]
    m.write_text("\n".join(lines) + "\n")
    manifest = load_corpus_manifest(m)
    assert manifest.items_count == 2
    assert [i.id for i in manifest.items] == ["b_1", "b_2"]  # deterministic sort
    assert manifest.manifest_sha256


def test_load_single_item_json(tmp_path: Path) -> None:
    m = tmp_path / "one.json"
    m.write_text(json.dumps(_item_dict("solo", "a.wav")))
    manifest = load_corpus_manifest(m)
    assert manifest.items_count == 1


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidUserInputError):
        load_corpus_manifest(tmp_path / "absent.json")


def test_verify_corpus_audio_digests(tmp_path: Path) -> None:
    wav = tmp_path / "t.wav"
    sf.write(str(wav), np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_24")
    good_sha = hash_file(wav)

    m = tmp_path / "m.json"
    m.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "m",
                "split_name": "calibration",
                "items_count": 1,
                "items": [_item_dict("ok", str(wav), good_sha)],
            }
        )
    )
    verify_corpus_audio_files(load_corpus_manifest(m))  # must not raise

    m.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "m",
                "split_name": "calibration",
                "items_count": 1,
                "items": [_item_dict("bad", str(wav), "0" * 64)],
            }
        )
    )
    with pytest.raises(InvalidUserInputError):
        verify_corpus_audio_files(load_corpus_manifest(m))

    m.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "m",
                "split_name": "calibration",
                "items_count": 1,
                "items": [_item_dict("gone", str(tmp_path / "missing.wav"))],
            }
        )
    )
    with pytest.raises(InvalidUserInputError):
        verify_corpus_audio_files(load_corpus_manifest(m))
