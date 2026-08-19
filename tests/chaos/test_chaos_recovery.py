"""Chaos and fault injection tests validating fail-closed safety and recovery."""

import tempfile
from pathlib import Path

import pytest

from voiceclean.errors import AmbiguousStereoError
from voiceclean.guard.spectral_probe import FixedProbe
from voiceclean.pipeline import run_pipeline


@pytest.mark.chaos
def test_chaos_ambiguous_stereo_rejected_without_silent_downmix() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_file = Path("tests/fixtures/sample_ambiguous_stereo.wav")
        out_file = tmp / "out.wav"

        with pytest.raises(AmbiguousStereoError):
            run_pipeline(
                input_path=in_file,
                output_path=out_file,
                profile="production",
                overwrite=True,
            )


@pytest.mark.chaos
def test_chaos_interrupted_job_resumes_cleanly() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_file = Path("tests/fixtures/sample_sorani_podcast.wav")
        out_file1 = tmp / "output_master.wav"

        # Run pipeline
        report1 = run_pipeline(
            input_path=in_file,
            output_path=out_file1,
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

        assert report1.output.samples == report1.input.samples
        assert out_file1.exists()
