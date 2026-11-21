"""Unit tests for research benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.research.benchmark import run_benchmark


def test_run_benchmark_workflow(tmp_path: Path) -> None:
    # 1. Create a minimal valid corpus manifest JSON
    manifest_path = tmp_path / "manifest.json"
    audio_path = tmp_path / "sample.wav"
    sr = 48000
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False, dtype=np.float32)
    sig = (0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    sf.write(str(audio_path), sig, sr, format="WAV", subtype="PCM_16")

    manifest_data = {
        "schema_version": 1,
        "manifest_id": "test_manifest",
        "split_name": "development",
        "items_count": 1,
        "manifest_sha256": "0" * 64,
        "items": [
            {
                "id": "item_01",
                "audio_path": str(audio_path),
                "audio_sha256": "0" * 64,
                "duration_s": 0.2,
                "speaker_id": "spk1",
                "dialect": "central",
                "gender": "unknown",
                "environment": "studio",
                "degradation_type": "clean",
                "transcript_sorani": "تێست",
                "split": "development",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_report = tmp_path / "results.json"
    out_audio = tmp_path / "bench_audio"

    results = run_benchmark(
        manifest_path=manifest_path,
        output_report_path=out_report,
        output_audio_dir=out_audio,
        compute_quality_metrics=True,
    )

    assert results["items_evaluated"] == 1
    assert out_report.is_file()
    assert (out_audio / "item_01_bench.wav").is_file()


def test_run_benchmark_with_quality_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys

    manifest_path = tmp_path / "manifest_clean.json"
    audio_path = tmp_path / "sample.wav"
    clean_path = tmp_path / "clean.wav"
    sr = 48000
    t = np.linspace(0, 0.2, int(0.2 * sr), endpoint=False, dtype=np.float32)
    sig = (0.2 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    sf.write(str(audio_path), sig, sr, format="WAV", subtype="PCM_16")
    sf.write(str(clean_path), sig, sr, format="WAV", subtype="PCM_16")

    manifest_data = {
        "schema_version": 1,
        "manifest_id": "test_manifest_clean",
        "split_name": "development",
        "items_count": 1,
        "manifest_sha256": "0" * 64,
        "items": [
            {
                "id": "item_01",
                "audio_path": str(audio_path),
                "audio_sha256": "0" * 64,
                "duration_s": 0.2,
                "speaker_id": "spk1",
                "dialect": "central",
                "gender": "unknown",
                "environment": "studio",
                "degradation_type": "clean",
                "transcript_sorani": "تێست",
                "split": "development",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    fake_metric = SimpleNamespace(
        pesq_wb=3.5,
        estoi=0.9,
        si_snr_db=15.0,
        lsd_db=1.2,
        separation_db=10.0,
    )
    fake_metrics_module = SimpleNamespace(compute_metrics=lambda _c, _e: fake_metric)
    monkeypatch.setitem(sys.modules, "hawavoclean.eval.metrics", fake_metrics_module)

    import hawavoclean.research.benchmark as bench_mod
    from hawavoclean.eval.corpus import load_corpus_manifest as real_load

    def mock_load(path: Path | str) -> Any:
        m = real_load(path)
        item_dict = m.items[0].model_dump()
        item_dict["clean_path"] = str(clean_path)
        fake_item = SimpleNamespace(**item_dict)
        return SimpleNamespace(
            manifest_id=m.manifest_id,
            manifest_sha256=m.manifest_sha256,
            items_count=1,
            items=[fake_item],
        )

    monkeypatch.setattr(bench_mod, "load_corpus_manifest", mock_load)

    out_report = tmp_path / "results2.json"
    results = run_benchmark(
        manifest_path=manifest_path,
        output_report_path=out_report,
        compute_quality_metrics=True,
    )
    assert results["quality_metrics"] is not None
    assert results["quality_metrics"]["pesq_wb"]["mean"] == 3.5
