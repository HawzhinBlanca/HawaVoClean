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
from hawavoclean.assembly.stitch import assemble_channel_timeline
from hawavoclean.audio.decode import decode_audio
from hawavoclean.config import load_config
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.guard.verdict import GuardVerdict
from hawavoclean.paths import profile_config_path
from hawavoclean.pipeline import run_pipeline
from hawavoclean.policy.continuity import (
    CONTINUITY_TAPER_ACTION,
    CONTINUITY_TAPER_MS,
    ContinuityResolution,
    resolve_source_continuity,
)
from hawavoclean.policy.decision import UnitPolicyDecision, evaluate_unit_policy
from hawavoclean.report.schema import HawaVoCleanReport
from hawavoclean.segmentation.types import SpeechUnit

FIXTURE = Path("tests/fixtures/sample_sorani_podcast.wav")
SR = 48000
TAPER_N = int(round(SR * CONTINUITY_TAPER_MS / 1000.0))


def _run(out: Path, monkeypatch: pytest.MonkeyPatch, plant: str = "nothing") -> HawaVoCleanReport:
    """Run the pipeline, optionally planting an outcome the policy would not
    have reached on this fixture, so the wiring downstream of the policy can be
    observed on a file that has no forced cuts of its own.

    ``plant="fade"`` puts a fade on the first enhanced unit; ``plant="revert"``
    marks it continuity-reverted. Both are decisions the policy is entitled to
    make; what is under test is what the pipeline does with them."""
    real = resolve_source_continuity

    def planted(
        units: list[SpeechUnit],
        decisions: list[UnitPolicyDecision],
        orig_core_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]],
        sample_rate: int,
    ) -> ContinuityResolution:
        res = real(units, decisions, orig_core_waveforms, sample_rate)
        fade_out = list(res.fade_out_samples)
        adjusted = list(res.decisions)
        reverted = set(res.reverted_ids)
        for i, d in enumerate(res.decisions):
            if not d.is_enhanced or len(orig_core_waveforms[i]) <= 4 * TAPER_N:
                continue
            if plant == "fade":
                fade_out[i] = TAPER_N
            else:
                adjusted[i] = UnitPolicyDecision(
                    selected_waveform=orig_core_waveforms[i].copy(),
                    is_enhanced=False,
                    chosen_strength=0.0,
                    guard_verdict=d.guard_verdict,
                    guard_scores=d.guard_scores,
                    decision_reason="planted continuity revert",
                )
                reverted.add(units[i].unit_id)
            break
        return ContinuityResolution(adjusted, reverted, res.fade_in_samples, fade_out)

    if plant != "nothing":
        monkeypatch.setattr(pipeline_module, "resolve_source_continuity", planted)
    return run_pipeline(
        input_path=FIXTURE,
        output_path=out,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )


@pytest.mark.integration
def test_a_planned_fade_reaches_the_master_and_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base_rep = _run(tmp / "base.wav", monkeypatch)
        taper_rep = _run(tmp / "taper.wav", monkeypatch, plant="fade")

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


@pytest.mark.integration
def test_a_continuity_revert_is_recorded_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit the continuity rule reverted must not be filed under the guard's
    own REVERT. They are different events — one is the guard rejecting audio,
    the other is this rule spending good audio to protect a seam — and only the
    second is a cost the rule itself is accountable for."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = _run(Path(tmpdir) / "reverted.wav", monkeypatch, plant="revert")

    assert report.summary.continuity_reverted == 1, (
        f"the continuity revert was not counted as one; summary={report.summary}"
    )
    assert report.summary.reverted == 0, "it was filed under the guard's own REVERT instead"
    reverted = [u for u in report.units if u.final_decision == "original_continuity"]
    assert len(reverted) == 1, [u.final_decision for u in report.units]
    assert report.summary.continuity_crossfaded == 0


# --------------------------------------------------------------------------
# The real thing: a forced cut the SEGMENTER made, a fade the POLICY planned,
# and the joint sample checked in the assembled timeline.
#
# The test above plants a fade on a fixture that has no forced cuts, which
# proves the wiring but not the invariant. Every shipped fixture is 8 s while
# `hard_max_group_s` has a schema floor of 10.0, so no fixture on its own can
# produce a forced boundary. Tiling one gives the VAD an unbroken speech
# interval long enough that the segmenter must cut inside it — real speech,
# real segmentation, real policy. The only thing planted is the guard REVERT
# the seam needs in order to exist.
# --------------------------------------------------------------------------

#: 3 x 8 s of real speech. With the segmentation override below the segmenter
#: makes four units of ~5.2 s, every boundary a forced mid-speech cut.
FIXTURE_REPEATS = 3
EXPECTED_UNITS = 4


@pytest.mark.integration
def test_the_joint_of_a_real_forced_cut_is_the_original_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    assert sr == SR
    src = tmp_path / "continuous.wav"
    sf.write(
        src, np.tile(data.mean(axis=1), FIXTURE_REPEATS).astype(np.float32), SR, subtype="PCM_24"
    )

    config = load_config(profile_config_path("development"), is_production=False)
    config = config.model_copy(
        update={
            "segmentation": config.segmentation.model_copy(
                update={"target_speech_group_s": 5.0, "hard_max_group_s": 10.0}
            )
        }
    )

    real_policy = evaluate_unit_policy
    real_assemble = assemble_channel_timeline
    real_decode = decode_audio
    seen = {"speech_units": 0}
    captured: dict[str, Any] = {}

    def revert_the_last_unit(**kwargs: Any) -> tuple[UnitPolicyDecision, Any]:
        """Stand in for a guard rejection on the final unit — the shape that
        used to cascade through the whole file. Nothing else is planted."""
        decision, dbg = real_policy(**kwargs)
        seen["speech_units"] += 1
        if seen["speech_units"] == EXPECTED_UNITS and decision.is_enhanced:
            decision = UnitPolicyDecision(
                selected_waveform=kwargs["orig_core_waveform"].copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=GuardVerdict.REVERT,
                guard_scores=decision.guard_scores,
                decision_reason="planted guard rejection",
            )
        return decision, dbg

    def capture_timeline(**kwargs: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
        timeline = real_assemble(**kwargs)
        captured.setdefault("timelines", []).append(timeline.copy())
        return timeline

    def capture_decode(*args: Any, **kwargs: Any) -> Any:
        buf = real_decode(*args, **kwargs)
        captured["decoded"] = buf.data.copy()
        return buf

    monkeypatch.setattr(pipeline_module, "evaluate_unit_policy", revert_the_last_unit)
    monkeypatch.setattr(pipeline_module, "assemble_channel_timeline", capture_timeline)
    monkeypatch.setattr(pipeline_module, "decode_audio", capture_decode)
    report = run_pipeline(
        input_path=src,
        output_path=tmp_path / "out.wav",
        config=config,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    assert report.summary.units_total == EXPECTED_UNITS, (
        f"segmentation changed: {report.summary.units_total} units, not {EXPECTED_UNITS}. "
        "Re-derive EXPECTED_UNITS or this test plants its revert on the wrong unit."
    )
    assert report.summary.error_passthrough == 0, "the core failed; this run tests nothing"
    # The regression itself: one rejected unit must not take the others with it.
    assert report.summary.enhanced == EXPECTED_UNITS - 1, (
        f"the cascade is back: {report.summary.enhanced} of {EXPECTED_UNITS} units enhanced, "
        f"continuity_reverted={report.summary.continuity_reverted}"
    )
    assert report.summary.continuity_crossfaded == 1, (
        f"expected one faded unit, got {report.summary.continuity_crossfaded}; "
        "if this is 0 the boundary was not forced and the test proves nothing"
    )

    faded = [
        u
        for u in report.units
        if any(a.startswith(CONTINUITY_TAPER_ACTION) for a in u.finish_actions)
    ]
    assert len(faded) == 1
    action = next(a for a in faded[0].finish_actions if a.startswith(CONTINUITY_TAPER_ACTION))
    # The ABSOLUTE length, so passing a wrong sample rate at the call site —
    # which ships a 10 ms fade and changes nothing else observable — fails here.
    assert action == f"{CONTINUITY_TAPER_ACTION}(in=0,out={TAPER_N})", action

    joint = faded[0].end_sample
    timeline = captured["timelines"][faded[0].channel]
    original = captured["decoded"][faded[0].channel]

    # THE invariant. The last sample of the faded unit, in the assembled
    # timeline before mastering, is the original recording — not the finished
    # enhanced audio, and not a fade toward some other array. Fading toward
    # `dec.selected_waveform`, or applying the fade before finishing instead of
    # after, both leave a step here and pass every other assertion in this file.
    assert timeline[joint - 1] == original[joint - 1], (
        f"the joint sample is {timeline[joint - 1]!r}, the original is "
        f"{original[joint - 1]!r}; the fade did not land on the original recording"
    )
    assert timeline[joint] == original[joint], "the reverted neighbour is not original audio"

    # ...and the unit really was enhanced everywhere else, so the assertion
    # above is not passing because nothing happened.
    body = slice(faded[0].start_sample + TAPER_N, joint - TAPER_N)
    assert not np.allclose(timeline[body], original[body], atol=1e-7), (
        "the faded unit is identical to the original throughout; it was reverted, not faded"
    )

    # The fade resolves toward the original as it approaches the joint.
    deviation = np.abs(timeline[joint - TAPER_N : joint] - original[joint - TAPER_N : joint])
    thirds = deviation.reshape(3, -1).mean(axis=1)
    assert thirds[0] > thirds[1] > thirds[2], f"the fade does not resolve: {thirds}"
