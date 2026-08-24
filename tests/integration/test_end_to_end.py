"""Integration tests running full HawaVoClean pipeline on reference fixtures."""

import tempfile
from pathlib import Path

import pytest

from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.finishing.loudness import measure_loudness_and_peaks
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.pipeline import run_pipeline


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
        assert (tmp / "podcast_mastered.hawavoclean.json").exists()
        assert (tmp / "podcast_mastered.hawavoclean.txt").exists()

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


@pytest.mark.integration
def test_reported_input_loudness_describes_the_input() -> None:
    """``report.input`` must describe the file the user handed us.

    Its loudness used to be measured on the pre-master buffer -- the audio
    after three enhancement cores and reassembly -- while every other field in
    the same block (path, sha256, sample_rate, samples) described the source.
    Anyone comparing input against output LUFS to see what mastering did was
    reading enhancement into the baseline.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path("tests/fixtures/sample_sorani_podcast.wav")
        report = run_pipeline(
            input_path=in_path,
            output_path=Path(tmpdir) / "out.wav",
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )

    media = probe_audio(in_path, max_sample_rate=192000)
    truth = measure_loudness_and_peaks(decode_audio(media).data, media.sample_rate)

    assert report.input.integrated_lufs is not None
    assert report.input.integrated_lufs == pytest.approx(truth.integrated_lufs, abs=0.05)
    assert report.input.true_peak_dbtp == pytest.approx(truth.true_peak_dbtp, abs=0.05)
