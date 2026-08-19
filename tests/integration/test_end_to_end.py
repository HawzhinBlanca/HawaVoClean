"""Integration tests running full VoiceClean pipeline on reference fixtures."""

import tempfile
from pathlib import Path

import pytest

from voiceclean.guard.spectral_probe import FixedProbe
from voiceclean.pipeline import run_pipeline


@pytest.mark.integration
def test_e2e_mono_podcast_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = Path("tests/fixtures/sample_sorani_podcast.wav")
        out_path = tmp / "podcast_mastered.wav"

        report = run_pipeline(
            input_path=in_path,
            output_path=out_path,
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

        assert out_path.exists()
        assert (tmp / "podcast_mastered.voiceclean.json").exists()
        assert (tmp / "podcast_mastered.voiceclean.txt").exists()

        assert report.output.samples == report.input.samples
        assert report.output.channels == report.input.channels
        assert report.output.sample_rate == report.input.sample_rate
        assert report.output.true_peak_dbtp is not None
        assert report.output.true_peak_dbtp <= -0.9


@pytest.mark.integration
def test_e2e_dual_mono_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = Path("tests/fixtures/sample_dual_mono.wav")
        out_path = tmp / "dual_mono_mastered.wav"

        report = run_pipeline(
            input_path=in_path,
            output_path=out_path,
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

        assert out_path.exists()
        assert report.output.channels == 2
        assert report.output.samples == report.input.samples


@pytest.mark.integration
def test_e2e_split_speakers_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = Path("tests/fixtures/sample_split_speakers.wav")
        out_path = tmp / "split_mastered.wav"

        report = run_pipeline(
            input_path=in_path,
            output_path=out_path,
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

        assert out_path.exists()
        assert report.output.channels == 2
        assert report.output.samples == report.input.samples
