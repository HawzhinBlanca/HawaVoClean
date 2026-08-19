"""Unit tests for pipeline configuration, development profile, and timecode review generation."""

import tempfile
from pathlib import Path

import pytest

from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.pipeline import run_pipeline


@pytest.mark.unit
def test_pipeline_development_profile() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = Path("tests/fixtures/sample_sorani_podcast.wav")
        out_path = tmp / "master.wav"

        rep = run_pipeline(
            input_path=in_path,
            output_path=out_path,
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

        assert rep.summary.units_total >= 1
        assert rep.output.samples == rep.input.samples
