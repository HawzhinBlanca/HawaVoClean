"""Calibration metrics must be measured, not typed in.

A hardcoded rate cannot respond to its inputs: two calibration runs over
corruption suites of different severity must report different rates.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from hawavoclean.eval.calibrate import run_calibration

SR = 48000


def _make_corpus(tmp_path: Path, name: str) -> Path:
    audio_dir = tmp_path / name / "audio"
    audio_dir.mkdir(parents=True)
    items = []
    rng = np.random.default_rng(7)
    for i in range(2):
        t = np.arange(SR * 4) / SR
        x = (
            0.3 * np.sin(2 * np.pi * (140 + 60 * i) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            + 0.02 * rng.standard_normal(SR * 4)
        ).astype(np.float32)
        p = audio_dir / f"{name}_{i}.wav"
        sf.write(str(p), x, SR, subtype="PCM_24")
        items.append(
            {
                "id": f"{name}_{i}",
                "audio_path": str(p),
                "audio_sha256": "",
                "duration_s": 4.0,
                "speaker_id": "synthetic",
                "dialect": "synthetic",
                "gender": "unknown",
                "environment": "synthetic",
                "degradation_type": "clean",
                "transcript_sorani": "-",
                "verified_by_human": False,
                "split": "calibration",
            }
        )
    manifest = tmp_path / f"{name}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": name,
                "split_name": "calibration",
                "items_count": len(items),
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_reported_rates_respond_to_corruption_severity(tmp_path: Path) -> None:
    m = _make_corpus(tmp_path, "corpus")

    severe = run_calibration(m, tmp_path / "severe.json", corruption_profile="severe")
    mild = run_calibration(m, tmp_path / "mild.json", corruption_profile="mild")

    r_severe = severe["metrics"]["calibration_false_accept_rate"]
    r_mild = mild["metrics"]["calibration_false_accept_rate"]

    assert r_mild > r_severe, (
        f"mild corruptions must be accepted more often than severe ones "
        f"(mild={r_mild}, severe={r_severe}); identical rates mean the metric "
        f"is not derived from the audio at all"
    )
    assert severe["measured"]["item_count"] > 0, "measurement provenance missing"
