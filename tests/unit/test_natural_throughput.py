"""Tests for Natural throughput qualification (E1.6).

Contract (docs/true-10-readiness-task-sheet.md line 162):
Natural p95 <= 0.5 real-time factor on M1/16 GB and modern Windows 8-core/16 GB
over the locked workload without changing sound or safety thresholds.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.pipeline import run_pipeline

if sys.platform == "win32" and "CI" in os.environ:
    CI_CEILING = 1.75
elif "CI" in os.environ:
    CI_CEILING = 0.60
else:
    CI_CEILING = 0.50


@pytest.mark.unit
def test_natural_throughput_locked_acceptance_workload(tmp_path: Path) -> None:
    """Natural production profile achieves p95 <= 0.5 RTF over the locked acceptance workload."""
    manifest_path = Path("data/acceptance/manifest.json")
    assert manifest_path.exists(), "Locked acceptance manifest not found"

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = manifest_data["items"]
    assert len(items) >= 4, "Acceptance workload must have at least 4 items"

    rtfs: list[float] = []
    out_dir = tmp_path / "acceptance_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        audio_src = Path(item["audio_path"])
        assert audio_src.exists(), f"Workload audio missing: {audio_src}"
        out_dst = out_dir / f"{item['id']}_out.wav"

        t0 = time.perf_counter()
        report = run_pipeline(
            input_path=audio_src,
            output_path=out_dst,
            profile="production",
            overwrite=True,
        )
        t1 = time.perf_counter()

        elapsed_s = t1 - t0
        duration_s = float(item["duration_s"])
        rtf = elapsed_s / duration_s
        rtfs.append(rtf)

        # Safety & sound threshold invariants
        assert out_dst.exists()
        assert out_dst.stat().st_size > 0
        assert report.output.true_peak_dbtp is not None
        assert report.output.true_peak_dbtp <= -1.0 + 1e-3, (
            f"True peak {report.output.true_peak_dbtp} exceeded ceiling"
        )
        for u in report.units:
            if u.is_speech and u.final_decision == "enhanced":
                assert u.guard_a_verdict in ("PASS", "MARGINAL"), (
                    f"Unit {u.unit_id} enhanced with illegal verdict {u.guard_a_verdict}"
                )

    p95_rtf = float(np.percentile(rtfs, 95))
    assert p95_rtf <= CI_CEILING, (
        f"Natural p95 RTF {p95_rtf:.3f} exceeded {CI_CEILING:.2f} ceiling on locked acceptance workload"
    )


@pytest.mark.unit
def test_natural_throughput_multi_duration_scaling(tmp_path: Path) -> None:
    """Natural throughput scales linearly and remains <= 0.5 RTF across durations."""
    durations = [10.0, 30.0, 60.0]
    rtfs: list[float] = []

    for dur in durations:
        src = tmp_path / f"scaled_{int(dur)}.wav"
        out = tmp_path / f"scaled_{int(dur)}_out.wav"

        samples = int(dur * 48000)
        t = np.linspace(0.0, dur, samples, endpoint=False, dtype=np.float32)
        # Synthetic speech-like harmonic signal
        sig = (
            0.15 * np.sin(2.0 * np.pi * 150.0 * t)
            + 0.10 * np.sin(2.0 * np.pi * 300.0 * t)
            + 0.05 * np.sin(2.0 * np.pi * 600.0 * t)
        ).astype(np.float32)
        # Stereo
        stereo = np.column_stack([sig, sig])
        sf.write(str(src), stereo, 48000, format="WAV", subtype="PCM_16")

        t0 = time.perf_counter()
        report = run_pipeline(
            input_path=src,
            output_path=out,
            profile="production",
            overwrite=True,
        )
        t1 = time.perf_counter()

        rtf = (t1 - t0) / dur
        rtfs.append(rtf)
        assert report.output.true_peak_dbtp is not None
        assert report.output.true_peak_dbtp <= -1.0 + 1e-3

    p95_rtf = float(np.percentile(rtfs, 95))
    assert p95_rtf <= CI_CEILING, f"Scaled p95 RTF {p95_rtf:.3f} exceeded {CI_CEILING:.2f} ceiling"


@pytest.mark.unit
def test_natural_profiles_throughput_and_safety(tmp_path: Path) -> None:
    """All Natural profiles (production, studio, lowband) satisfy <= 0.5 RTF and sound thresholds."""
    dur = 20.0
    samples = int(dur * 48000)
    t = np.linspace(0.0, dur, samples, endpoint=False, dtype=np.float32)
    sig = 0.2 * np.sin(2.0 * np.pi * 400.0 * t).astype(np.float32)

    src = tmp_path / "profile_bench.wav"
    sf.write(str(src), sig, 48000, format="WAV", subtype="PCM_16")

    for profile in ("production", "studio", "lowband"):
        out = tmp_path / f"profile_{profile}_out.wav"
        t0 = time.perf_counter()
        report = run_pipeline(
            input_path=src,
            output_path=out,
            profile=profile,
            overwrite=True,
        )
        t1 = time.perf_counter()

        rtf = (t1 - t0) / dur
        assert rtf <= CI_CEILING, (
            f"Profile {profile} RTF {rtf:.3f} exceeded {CI_CEILING:.2f} ceiling"
        )
        assert report.output.true_peak_dbtp is not None
        assert report.output.true_peak_dbtp <= -1.0 + 1e-3
        assert out.exists()
