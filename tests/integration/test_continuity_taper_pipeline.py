"""The pipeline must apply the fade the continuity policy plans — and say so.

The unit tests cover what :mod:`hawavoclean.policy.continuity` *decides*. This
one covers the wiring: that the decision reaches the published audio, lands
exactly where it was planned and nowhere else, and appears in the audit trail.
Delete the pipeline's call to :func:`apply_continuity_taper` and only this test
notices.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.pipeline as pipeline_module
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.pipeline import run_pipeline
from hawavoclean.policy.continuity import (
    CONTINUITY_TAPER_ACTION,
    CONTINUITY_TAPER_MS,
    ContinuityResolution,
)
from hawavoclean.policy.decision import UnitPolicyDecision
from hawavoclean.report.schema import HawaVoCleanReport
from hawavoclean.segmentation.types import SpeechUnit

FIXTURE = Path("tests/fixtures/sample_sorani_podcast.wav")
SR = 48000
TAPER_N = int(round(SR * CONTINUITY_TAPER_MS / 1000.0))


def _run(out: Path, force_taper_on_first_enhanced: bool) -> HawaVoCleanReport:
    """Run the pipeline, optionally planting a fade the policy would not have
    planned, so the wiring can be observed on a fixture with no forced cuts."""
    real = pipeline_module.resolve_source_continuity  # type: ignore[attr-defined]

    def planted(
        units: list[SpeechUnit],
        decisions: list[UnitPolicyDecision],
        orig_core_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]],
        sample_rate: int,
    ) -> ContinuityResolution:
        res = real(units, decisions, orig_core_waveforms, sample_rate)
        fade_out = list(res.fade_out_samples)
        for i, d in enumerate(res.decisions):
            if d.is_enhanced and len(orig_core_waveforms[i]) > 4 * TAPER_N:
                fade_out[i] = TAPER_N
                break
        return ContinuityResolution(res.decisions, res.reverted_ids, res.fade_in_samples, fade_out)

    if force_taper_on_first_enhanced:
        pipeline_module.resolve_source_continuity = planted  # type: ignore[attr-defined]
    try:
        return run_pipeline(
            input_path=FIXTURE,
            output_path=out,
            profile="development",
            overwrite=True,
            probe_override=FixedProbe(),
        )
    finally:
        pipeline_module.resolve_source_continuity = real  # type: ignore[attr-defined]


@pytest.mark.integration
def test_a_planned_fade_reaches_the_master_and_the_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base_rep = _run(tmp / "base.wav", force_taper_on_first_enhanced=False)
        taper_rep = _run(tmp / "taper.wav", force_taper_on_first_enhanced=True)

        assert base_rep.summary.enhanced >= 1, "fixture no longer enhances anything"
        assert base_rep.summary.continuity_crossfaded == 0
        assert taper_rep.summary.continuity_crossfaded == 1, (
            "the pipeline did not apply — or did not record — the planned fade"
        )

        faded = [
            u
            for u in taper_rep.units
            if any(a.startswith(CONTINUITY_TAPER_ACTION) for a in u.finish_actions)
        ]
        assert len(faded) == 1
        action = next(a for a in faded[0].finish_actions if a.startswith(CONTINUITY_TAPER_ACTION))
        assert f"out={TAPER_N}" in action, action
        # A fade changes the audio, so it must change the unit's output hash.
        base_unit = next(u for u in base_rep.units if u.unit_id == faded[0].unit_id)
        assert faded[0].output_sha256 != base_unit.output_sha256

        base = sf.read(tmp / "base.wav", dtype="float32", always_2d=True)[0].mean(axis=1)
        tapered = sf.read(tmp / "taper.wav", dtype="float32", always_2d=True)[0].mean(axis=1)
        assert len(base) == len(tapered)

        end = faded[0].end_sample
        window = slice(max(0, end - TAPER_N), end)
        diff = np.abs(tapered.astype(np.float64) - base.astype(np.float64))
        inside = float(np.sqrt(np.mean(diff[window] ** 2)))
        outside_mask = np.ones(len(diff), dtype=bool)
        outside_mask[window] = False
        outside = float(np.sqrt(np.mean(diff[outside_mask] ** 2)))

        assert inside > 0.0, "the planned fade changed nothing"
        # Outside the window the two masters differ only by the master gain
        # responding to a 30 ms change in an 8 s file — orders of magnitude
        # smaller than the fade itself.
        assert inside > 100.0 * outside, (
            f"the fade leaked outside its window: inside RMS {inside:.3e}, "
            f"outside RMS {outside:.3e}"
        )
