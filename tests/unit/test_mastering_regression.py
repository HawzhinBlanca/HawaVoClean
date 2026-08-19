"""Mastering regressions: loudness target, encoded file structure, ceiling.

These assertions read the PUBLISHED FILE, not the in-memory report, so an
encoder that drops samples or a mis-set loudness target cannot hide behind
report-side bookkeeping.
"""

from pathlib import Path

import pytest
import soundfile as sf

from hawavoclean.finishing.loudness import measure_loudness_and_peaks
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "sample_sorani_podcast.wav"
MONO_TARGET_LUFS = -19.0


@pytest.mark.integration
def test_published_file_hits_loudness_target_and_structure(tmp_path: Path) -> None:
    out = tmp_path / "mastered.wav"
    report = run_pipeline(
        input_path=FIXTURE,
        output_path=out,
        profile="production",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    # 1. The encoded file itself has exactly the input's sample count.
    info = sf.info(str(out))
    assert info.frames == report.input.samples, (
        f"published file holds {info.frames} frames, input had {report.input.samples}"
    )
    assert info.samplerate == report.input.sample_rate

    # 2. Independently measured loudness lands on the mono target.
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    measured = measure_loudness_and_peaks(data.T, sr)
    assert abs(measured.integrated_lufs - MONO_TARGET_LUFS) <= 1.0, (
        f"integrated loudness {measured.integrated_lufs:.2f} LUFS is not at the "
        f"{MONO_TARGET_LUFS} LUFS mono target"
    )

    # 3. True peak respects the -1.0 dBTP ceiling with no tolerance.
    assert measured.true_peak_dbtp <= -1.0, (
        f"true peak {measured.true_peak_dbtp:.3f} dBTP exceeds the -1.0 ceiling"
    )

    # 4. The report's numbers agree with the file's.
    assert report.output.integrated_lufs is not None
    assert abs(report.output.integrated_lufs - measured.integrated_lufs) <= 0.2
