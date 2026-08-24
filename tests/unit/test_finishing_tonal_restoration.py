"""The finishing chain must restore a muffled voice — and only a muffled voice.

Reported by a user on a real 24 s recording, and confirmed by measurement:
every band of the output moved by exactly the same +15.7 dB. The chain applied
a flat loudness gain and shipped the file as muffled as it arrived.

The fix is a measured tonal restoration, and the danger it carries is the 3.1.1
regression it must not become ("harsh and treble sounding, dialogs bass removed
lot"). So the gates below are two-sided by design, and the load-bearing pair is
`approved_recording_profile` (measured from a recording whose sound the user
signed off on — must receive exactly 0.0 dB) against `presence_starved_profile`
(the same signal with the deficit of the reported file — must be corrected).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.signal

from hawavoclean.config import FinishingConfig, GuardConfig
from hawavoclean.finishing.detect import (
    TILT_MAX_BRILLIANCE_LIFT_DB,
    TILT_MAX_LOW_CUT_DB,
    TILT_MAX_LOW_LIFT_DB,
    TILT_MAX_PRESENCE_LIFT_DB,
    aggregate_speech_tilt,
    measure_speech_tilt,
)
from hawavoclean.finishing.eq import (
    achieved_band_gains_db,
    apply_tonal_restoration,
    solve_tonal_gains,
    tonal_filter_response,
)
from hawavoclean.finishing.safe_finish import apply_finishing_stages
from hawavoclean.guard.signal import check_signal_integrity
from tests.support.tonal_corpus import (
    SR,
    approved_recording_profile,
    boomy,
    brickwall_lowpassed,
    natural_voice,
    near_silent,
    presence_starved_profile,
    softly_muffled,
    speech_like,
    thin_harsh,
)

# The other finishing stages (subsonic filter, mud trim, de-esser, leveler) each
# move the spectrum for their own reasons. These tests are about the tonal
# stage, so everything else is switched off and tested elsewhere.
TONAL_ONLY = FinishingConfig(
    dc_subsonic_cutoff_hz=20.0,
    dehum_50_60hz=False,
    declick=False,
    dynamic_eq=False,
    deess_band=False,
    level_rider=False,
)

BANDS: tuple[tuple[str, float, float], ...] = (
    ("30-80", 30.0, 80.0),
    ("80-200", 80.0, 200.0),
    ("200-500", 200.0, 500.0),
    ("500-1k", 500.0, 1000.0),
    ("1-2k", 1000.0, 2000.0),
    ("2-4k", 2000.0, 4000.0),
    ("4-8k", 4000.0, 8000.0),
    ("8-16k", 8000.0, 16000.0),
)


def band_deltas_db(before: np.ndarray[Any, Any], after: np.ndarray[Any, Any]) -> dict[str, float]:
    """Per-band change in dB, normalised so the 200-500 Hz body reads 0.0.

    Level is not tonality: a stage that lifts everything equally has changed
    nothing tonal, and this gate must not fire on it.
    """
    f, p_before = scipy.signal.welch(before, SR, nperseg=4096)
    _, p_after = scipy.signal.welch(after, SR, nperseg=4096)

    def band(power: np.ndarray[Any, Any], lo: float, hi: float) -> float:
        mask = (f >= lo) & (f < hi)
        return float(np.trapezoid(power[mask], f[mask]))

    body = 10.0 * np.log10(band(p_after, 200.0, 500.0) / band(p_before, 200.0, 500.0))
    return {
        name: 10.0 * np.log10(band(p_after, lo, hi) / band(p_before, lo, hi)) - body
        for name, lo, hi in BANDS
    }


def tonal_only(x: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], list[str]]:
    return apply_finishing_stages(x, SR, TONAL_ONLY, "gentle")


# --------------------------------------------------------------------------
# The pair the whole design turns on
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_approved_recording_profile_is_left_alone() -> None:
    """A recording the user approved must receive exactly zero correction.

    It measures presence-shy against any textbook speech spectrum (-27.8 dB at
    1.5-3 kHz, -33.3 dB at 3-6 kHz relative to its body) and it still sounds
    right to the person who made it. A target curve that "improves" this is the
    3.1.1 regression wearing new clothes.
    """
    tilt = measure_speech_tilt(approved_recording_profile(), SR)
    assert tilt.measured
    assert tilt.low_shelf_db == 0.0, tilt.summary()
    assert tilt.presence_db == 0.0, tilt.summary()
    assert tilt.brilliance_db == 0.0, tilt.summary()


@pytest.mark.unit
def test_approved_recording_profile_survives_the_whole_chain_untouched() -> None:
    out, actions = tonal_only(approved_recording_profile())
    assert not any(a.startswith("tonal_restore") for a in actions), actions
    for name, delta in band_deltas_db(approved_recording_profile(), out).items():
        assert abs(delta) < 0.25, f"{name} moved {delta:+.2f} dB on an approved recording"


@pytest.mark.unit
def test_presence_starved_profile_is_restored() -> None:
    """The same signal, 11 dB short above 3 kHz — the measured gap between the
    approved recording and the one the user called embarrassingly bad."""
    source = presence_starved_profile()
    tilt = measure_speech_tilt(source, SR)
    assert tilt.brilliance_db > 5.0, tilt.summary()
    out, actions = tonal_only(source)
    assert any(a.startswith("tonal_restore") for a in actions), actions
    deltas = band_deltas_db(source, out)
    assert deltas["2-4k"] > 2.0, deltas
    assert deltas["4-8k"] > 3.0, deltas
    # And the half of the spectrum the two recordings agree on stays put.
    for name in ("80-200", "500-1k"):
        assert abs(deltas[name]) < 1.0, f"{name} moved {deltas[name]:+.2f} dB: {deltas}"


@pytest.mark.unit
def test_the_pair_is_only_separable_above_3k() -> None:
    """Pins the fact the design rests on. If these two ever diverge below
    3 kHz, a cheaper correction would work and this one is over-built; if they
    stop diverging above it, this correction cannot tell them apart at all."""
    approved = {b.name: b for b in measure_speech_tilt(approved_recording_profile(), SR).bands}
    starved = {b.name: b for b in measure_speech_tilt(presence_starved_profile(), SR).bands}
    for name in ("low", "presence"):
        gap = abs(approved[name].level_rel_body_db - starved[name].level_rel_body_db)
        assert gap < 2.5, f"{name} differs by {gap:.1f} dB — the pair is separable there now"
    gap = approved["brilliance"].level_rel_body_db - starved["brilliance"].level_rel_body_db
    assert gap > 8.0, f"brilliance gap collapsed to {gap:.1f} dB"


# --------------------------------------------------------------------------
# Controls that must not move
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_pinned_natural_voice_fixture_gets_nothing() -> None:
    """The 3.1.1 transparency fixture. Two independent reasons hold it at zero:
    it is inside the target band, and it has no pauses so no band shows the
    dynamics gate's minimum headroom."""
    tilt = measure_speech_tilt(natural_voice(), SR)
    assert (tilt.low_shelf_db, tilt.presence_db, tilt.brilliance_db) == (0.0, 0.0, 0.0)
    out, actions = tonal_only(natural_voice())
    assert not any(a.startswith("tonal_restore") for a in actions)
    for name, delta in band_deltas_db(natural_voice(), out).items():
        assert abs(delta) < 0.25, f"{name} moved {delta:+.2f} dB"


@pytest.mark.unit
def test_ordinary_speech_gets_nothing() -> None:
    x = speech_like()
    tilt = measure_speech_tilt(x, SR)
    assert (tilt.low_shelf_db, tilt.presence_db, tilt.brilliance_db) == (0.0, 0.0, 0.0)
    for name, delta in band_deltas_db(x, tonal_only(x)[0]).items():
        assert abs(delta) < 0.25, f"{name} moved {delta:+.2f} dB on ordinary speech"


@pytest.mark.unit
def test_near_silence_is_not_measured() -> None:
    tilt = measure_speech_tilt(near_silent(), SR)
    assert not tilt.measured
    assert not tilt.is_correction


# --------------------------------------------------------------------------
# Gate 1: never lift a band that has nothing in it
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_brickwall_lowpass_is_refused_not_amplified() -> None:
    """Above the cut there is only dither. Gain would raise the hiss and
    nothing else, so the answer is no — stated in the report, not silently."""
    x = brickwall_lowpassed()
    tilt = measure_speech_tilt(x, SR)
    assert tilt.presence_db == 0.0 and tilt.brilliance_db == 0.0, tilt.summary()
    by_name = {b.name: b for b in tilt.bands}
    for name in ("presence", "brilliance"):
        assert by_name[name].level_rel_body_db < -45.0, tilt.summary()
        assert by_name[name].gate_reason in (":no-dynamics", ":not-captured"), tilt.summary()
    assert not any(a.startswith("tonal_restore") for a in tonal_only(x)[1])


@pytest.mark.unit
def test_the_gate_distinguishes_empty_from_merely_quiet() -> None:
    """Both controls are muffled. Only the one that still has speech dynamics
    above the cut may be lifted — that difference IS the gate."""
    empty = measure_speech_tilt(brickwall_lowpassed(), SR)
    quiet = measure_speech_tilt(softly_muffled(), SR)
    empty_presence = {b.name: b for b in empty.bands}["presence"]
    quiet_presence = {b.name: b for b in quiet.bands}["presence"]
    assert empty_presence.headroom_db < 6.0, empty.summary()
    assert quiet_presence.headroom_db > 20.0, quiet.summary()
    assert empty.presence_db == 0.0
    assert quiet.presence_db > 5.0


@pytest.mark.unit
def test_softly_muffled_is_lifted_where_content_survives() -> None:
    x = softly_muffled()
    out, actions = tonal_only(x)
    assert any(a.startswith("tonal_restore") for a in actions), actions
    deltas = band_deltas_db(x, out)
    assert deltas["2-4k"] > 4.0, deltas
    # The 3-6 kHz band of this control is past the reach backstop: refused.
    assert {b.name: b for b in measure_speech_tilt(x, SR).bands}[
        "brilliance"
    ].gate_reason == ":not-captured"


# --------------------------------------------------------------------------
# Low end: cut boom, never thin a thin voice
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_boomy_recording_gets_a_bounded_cut() -> None:
    x = boomy()
    tilt = measure_speech_tilt(x, SR)
    assert -TILT_MAX_LOW_CUT_DB <= tilt.low_shelf_db < -1.0, tilt.summary()
    deltas = band_deltas_db(x, tonal_only(x)[0])
    assert deltas["80-200"] < -0.5, deltas


@pytest.mark.unit
def test_thin_voice_is_never_thinned_further() -> None:
    """The 3.1.1 failure as an input. Presence must not be cut (the stage is
    lift-only there), and the missing bottom may be given back, bounded."""
    x = thin_harsh()
    tilt = measure_speech_tilt(x, SR)
    assert tilt.presence_db == 0.0 and tilt.brilliance_db == 0.0, tilt.summary()
    assert 0.0 < tilt.low_shelf_db <= TILT_MAX_LOW_LIFT_DB, tilt.summary()
    deltas = band_deltas_db(x, tonal_only(x)[0])
    assert deltas["80-200"] > 0.5, deltas
    assert deltas["2-4k"] < 0.1, deltas


@pytest.mark.unit
def test_a_boomy_recording_can_never_earn_a_bass_lift() -> None:
    """The low LIFT is gated on presence being in genuine surplus. Without that
    gate a dull, bass-light recording would ask for more bass."""
    for maker in (boomy, softly_muffled, presence_starved_profile, approved_recording_profile):
        tilt = measure_speech_tilt(maker(), SR)
        if tilt.low_shelf_db > 0.0:
            presence = {b.name: b for b in tilt.bands}["presence"].level_rel_body_db
            assert presence > -18.0, f"{maker.__name__} got a bass lift without surplus"


# --------------------------------------------------------------------------
# Bounds are measured, not assumed
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100, 48000])
@pytest.mark.parametrize(
    ("low", "presence", "brilliance"),
    [(-6.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 12.0), (4.0, 0.0, 0.0)],
)
def test_one_section_delivers_exactly_the_gain_it_is_asked_for(
    sample_rate: int, low: float, presence: float, brilliance: float
) -> None:
    """`filtfilt` squares the response: `apply_speech_eq` has always delivered
    twice the dB it was asked for (-6.0 requested measures -11.92). This bank
    must not, and the proof is its own measured response rather than a comment.
    """
    _, response = tonal_filter_response(sample_rate, low, presence, brilliance)
    want = max(abs(low), presence, brilliance)
    assert float(np.max(np.abs(response))) == pytest.approx(want, abs=0.15)


@pytest.mark.unit
@pytest.mark.parametrize("sample_rate", [16000, 22050, 44100, 48000])
def test_overlapping_sections_never_exceed_their_combined_budget(sample_rate: int) -> None:
    """Two bells whose skirts overlap add where they meet, so a cascade peaks
    ABOVE the largest single section — +6.7 dB for a +3/+6 pair. That is why
    `apply_tonal_restoration` bounds the finished cascade rather than each
    section, and why the caller's cap is the total, not the biggest term."""
    low, presence, brilliance = -2.0, 3.0, 6.0
    _, response = tonal_filter_response(sample_rate, low, presence, brilliance)
    peak = float(np.max(np.abs(response)))
    assert peak >= max(abs(low), presence, brilliance) - 0.15
    assert peak <= abs(low) + presence + brilliance


@pytest.mark.unit
def test_solver_delivers_the_requested_band_move() -> None:
    """Three overlapping sections cannot each act alone; the solver decouples
    them, so what lands in a band is what the measurement asked for."""
    for want in ((-4.0, 0.0, 0.0), (0.0, 6.0, 0.0), (0.0, 0.0, 8.0), (-2.0, 3.0, 6.0)):
        gains = solve_tonal_gains(
            SR,
            *want,
            max_low_cut_db=6.0,
            max_low_lift_db=4.0,
            max_presence_db=10.0,
            max_brilliance_db=12.0,
        )
        achieved = achieved_band_gains_db(SR, *gains)
        for index, (got, asked) in enumerate(zip(achieved, want, strict=True)):
            if asked != 0.0:
                assert got == pytest.approx(asked, abs=0.3), f"{want} -> {achieved}"
                continue
            # A band nobody asked about still moves, and the bank deliberately
            # does not cancel it (see the comment on TONAL bells in eq.py: a
            # solver free to cut the bells turns a bass cut into a bass-and-
            # treble cut). Two invariants make the leftover safe: it is never a
            # CUT, and it never exceeds half of what the band that earned the
            # move received — so the correction is always dominated by the
            # measurement that justified it.
            assert got > -0.05, f"{want} -> {achieved}: band {index} was cut"
            budget = max(abs(v) for v in want) / 2.0
            assert got <= budget, f"{want} -> {achieved}: band {index} over half the move"


@pytest.mark.unit
def test_caps_bind_and_under_correct_rather_than_running_away() -> None:
    gains = solve_tonal_gains(
        SR,
        -40.0,
        40.0,
        40.0,
        max_low_cut_db=TILT_MAX_LOW_CUT_DB,
        max_low_lift_db=TILT_MAX_LOW_LIFT_DB,
        max_presence_db=TILT_MAX_PRESENCE_LIFT_DB,
        max_brilliance_db=TILT_MAX_BRILLIANCE_LIFT_DB,
    )
    assert gains[0] >= -TILT_MAX_LOW_CUT_DB
    assert 0.0 <= gains[1] <= TILT_MAX_PRESENCE_LIFT_DB
    assert 0.0 <= gains[2] <= TILT_MAX_BRILLIANCE_LIFT_DB


@pytest.mark.unit
def test_a_cascade_that_exceeds_its_own_bound_is_refused() -> None:
    x = speech_like()
    out = apply_tonal_restoration(x, SR, 0.0, 0.0, 12.0, max_abs_gain_db=4.0)
    assert np.array_equal(out, x), "a cascade over its declared bound must not be applied"


@pytest.mark.unit
def test_correction_never_introduces_clipping() -> None:
    hot = np.clip(speech_like() * 2.9, -0.999, 0.999).astype(np.float32)
    out = apply_tonal_restoration(hot, SR, 4.0, 10.0, 12.0)
    assert float(np.max(np.abs(out))) <= max(float(np.max(np.abs(hot))), 0.999) + 1e-6


@pytest.mark.unit
def test_measured_correction_stays_inside_the_deadband_edge() -> None:
    """Over-brightening is meant to be impossible by construction: the
    correction stops at the low edge of "acceptable" and never crosses it."""
    x = presence_starved_profile()
    before = measure_speech_tilt(x, SR)
    out, _ = tonal_only(x)
    after = measure_speech_tilt(out, SR)
    approved = {b.name: b for b in measure_speech_tilt(approved_recording_profile(), SR).bands}
    for band in after.bands:
        if band.name == "brilliance":
            was = {b.name: b for b in before.bands}[band.name].level_rel_body_db
            assert band.level_rel_body_db > was + 3.0, "no restoration happened"
            assert band.level_rel_body_db <= approved[band.name].level_rel_body_db + 0.5, (
                "corrected past the recording the user already approved"
            )


# --------------------------------------------------------------------------
# Guard B, determinism, and the shapes the pipeline actually hands over
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_guard_b_signal_integrity_passes_on_a_real_restoration() -> None:
    x = presence_starved_profile()
    out, _ = tonal_only(x)
    cfg = GuardConfig()
    result = check_signal_integrity(
        x,
        out,
        SR,
        spectral_hole_thresh=cfg.spectral_hole_thresh,
        musical_noise_thresh=cfg.musical_noise_thresh,
        min_hf_preservation_ratio=cfg.min_hf_preservation_ratio,
    )
    assert result.passed, result.failure_reasons
    assert result.consonant_retention_ratio > 1.0
    assert result.spectral_hole_score < cfg.spectral_hole_thresh
    assert result.clipping_samples_count == 0


@pytest.mark.unit
def test_guard_b_still_rejects_damage_dressed_as_a_tonal_move() -> None:
    """The guard has to remain able to say no. A cascade that guts the
    consonant band must fail signal integrity even though it is 'just EQ'."""
    x = speech_like()
    damaged = apply_tonal_restoration(x, SR, 0.0, 0.0, 0.0)
    damaged = np.asarray(damaged, dtype=np.float64)
    sos = scipy.signal.butter(6, 1500, btype="lowpass", fs=SR, output="sos")
    damaged = np.ascontiguousarray(scipy.signal.sosfiltfilt(sos, damaged), dtype=np.float32)
    cfg = GuardConfig()
    result = check_signal_integrity(
        x,
        damaged,
        SR,
        spectral_hole_thresh=cfg.spectral_hole_thresh,
        musical_noise_thresh=cfg.musical_noise_thresh,
        min_hf_preservation_ratio=cfg.min_hf_preservation_ratio,
    )
    assert not result.passed


@pytest.mark.unit
def test_minimal_intensity_does_less_not_something_else() -> None:
    x = presence_starved_profile()
    tilt = measure_speech_tilt(x, SR)
    gentle, _ = apply_finishing_stages(x, SR, TONAL_ONLY, "gentle", tilt=tilt)
    minimal, _ = apply_finishing_stages(x, SR, TONAL_ONLY, "minimal", tilt=tilt)
    g = band_deltas_db(x, gentle)["4-8k"]
    m = band_deltas_db(x, minimal)["4-8k"]
    assert 0.0 < m < g, f"minimal={m:+.2f} gentle={g:+.2f}"


@pytest.mark.unit
def test_one_file_level_answer_for_every_unit() -> None:
    """Units are finished independently. Measured per unit, adjacent blocks of
    one recording landed on opposite sides of a gate and swapped 10 dB of EQ
    mid-file. The aggregate is what the pipeline applies to all of them."""
    source = presence_starved_profile()
    units = [source[i : i + SR * 2] for i in range(0, len(source) - SR, SR * 2)]
    per_unit = [measure_speech_tilt(u, SR) for u in units]
    combined = aggregate_speech_tilt(per_unit)
    assert combined.measured
    assert combined.brilliance_db > 3.0, combined.summary()
    outs = [apply_finishing_stages(u, SR, TONAL_ONLY, "gentle", tilt=combined)[0] for u in units]
    moves = [band_deltas_db(u, o)["4-8k"] for u, o in zip(units, outs, strict=True)]
    assert max(moves) - min(moves) < 1.0, f"tone stepped between units: {moves}"


@pytest.mark.unit
def test_aggregate_ignores_unmeasurable_units_and_survives_all_of_them() -> None:
    good = measure_speech_tilt(presence_starved_profile(), SR)
    empty = measure_speech_tilt(near_silent(), SR)
    assert aggregate_speech_tilt([empty, empty]).measured is False
    assert aggregate_speech_tilt([empty, good, empty]).brilliance_db == pytest.approx(
        aggregate_speech_tilt([good]).brilliance_db, abs=1e-9
    )
    assert aggregate_speech_tilt([]).measured is False


@pytest.mark.unit
def test_measurement_and_correction_are_deterministic() -> None:
    x = presence_starved_profile()
    first, first_actions = tonal_only(x)
    second, second_actions = tonal_only(x)
    assert first_actions == second_actions
    assert np.array_equal(first, second)
    a = measure_speech_tilt(x, SR)
    b = measure_speech_tilt(x, SR)
    assert (a.low_shelf_db, a.presence_db, a.brilliance_db) == (
        b.low_shelf_db,
        b.presence_db,
        b.brilliance_db,
    )


@pytest.mark.unit
@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
def test_bands_above_nyquist_are_skipped_not_crashed(sample_rate: int) -> None:
    rng = np.random.default_rng(3)
    x = np.asarray(rng.standard_normal(sample_rate * 3) * 0.1, dtype=np.float32)
    tilt = measure_speech_tilt(x, sample_rate)
    if tilt.measured:
        for band in tilt.bands:
            if band.low_hz >= sample_rate / 2:
                assert band.correction_db == 0.0
                assert band.gate_reason == ":above-nyquist"
    out = apply_tonal_restoration(x, sample_rate, -3.0, 5.0, 7.0)
    assert len(out) == len(x)
    assert np.all(np.isfinite(out))


@pytest.mark.unit
@pytest.mark.parametrize("length", [0, 1, 127, 128, 511, 8191, 8192])
def test_short_and_degenerate_input_is_safe(length: int) -> None:
    x = np.zeros(length, dtype=np.float32)
    tilt = measure_speech_tilt(x, SR)
    assert not tilt.is_correction
    out = apply_tonal_restoration(x, SR, -3.0, 5.0, 7.0)
    assert len(out) == length
    assert np.all(np.isfinite(out))


@pytest.mark.unit
def test_non_finite_input_is_refused_rather_than_propagated() -> None:
    x = np.asarray(speech_like(), dtype=np.float32).copy()
    x[1000] = np.nan
    assert not measure_speech_tilt(x, SR).measured


@pytest.mark.unit
def test_disabling_the_stage_restores_the_old_behaviour() -> None:
    x = presence_starved_profile()
    off = FinishingConfig(
        dc_subsonic_cutoff_hz=20.0,
        dehum_50_60hz=False,
        declick=False,
        dynamic_eq=False,
        deess_band=False,
        level_rider=False,
        tonal_restoration=False,
    )
    out, actions = apply_finishing_stages(x, SR, off, "gentle")
    assert not any(a.startswith("tonal_restore") for a in actions)
    assert np.array_equal(out, x)


@pytest.mark.unit
def test_a_refused_band_is_not_lifted_by_a_neighbours_spill() -> None:
    """The bells overlap, so a presence lift reaches a little way into the band
    above it. When that band was REFUSED — nothing in it to restore — the spill
    must stay small enough not to become an audible hiss lift in its own right.
    """
    x = softly_muffled()
    tilt = measure_speech_tilt(x, SR)
    brilliance = {b.name: b for b in tilt.bands}["brilliance"]
    assert brilliance.correction_db == 0.0, tilt.summary()
    assert tilt.presence_db > 5.0, tilt.summary()
    gains = solve_tonal_gains(
        SR,
        tilt.low_shelf_db,
        tilt.presence_db,
        tilt.brilliance_db,
        max_low_cut_db=TILT_MAX_LOW_CUT_DB,
        max_low_lift_db=TILT_MAX_LOW_LIFT_DB,
        max_presence_db=TILT_MAX_PRESENCE_LIFT_DB,
        max_brilliance_db=TILT_MAX_BRILLIANCE_LIFT_DB,
    )
    spill = achieved_band_gains_db(SR, *gains)[2]
    assert spill < 2.0, f"a refused band received {spill:+.2f} dB from its neighbour"


# --------------------------------------------------------------------------
# The real recordings, when they are present
# --------------------------------------------------------------------------
#
# `test_output/` is gitignored, so these two cannot be the permanent gate — the
# synthetic profiles above carry that job. When the real audio IS on the
# machine, these run and pin the actual numbers the design was calibrated on.

REPO_ROOT = Path(__file__).resolve().parents[2]
APPROVED_AUDIO = REPO_ROOT / "test_output" / "ui-smoke" / "Flute 09.m4a.mp4"
REPORTED_AUDIO = REPO_ROOT / "test_output" / "tonal" / "Teat1vo.mp3"


def _decode_mono(path: Path) -> tuple[np.ndarray[Any, Any], int]:
    from hawavoclean.audio.decode import decode_audio
    from hawavoclean.audio.probe import probe_audio

    buffer = decode_audio(probe_audio(path))
    return (
        np.ascontiguousarray(np.mean(buffer.data, axis=0), dtype=np.float32),
        buffer.sample_rate,
    )


@pytest.mark.unit
@pytest.mark.skipif(not APPROVED_AUDIO.exists(), reason="user audio not on this machine")
def test_real_approved_recording_receives_nothing() -> None:
    x, sample_rate = _decode_mono(APPROVED_AUDIO)
    tilt = measure_speech_tilt(x, sample_rate)
    assert tilt.measured
    assert (tilt.low_shelf_db, tilt.presence_db, tilt.brilliance_db) == (0.0, 0.0, 0.0), (
        tilt.summary()
    )


@pytest.mark.unit
@pytest.mark.skipif(not REPORTED_AUDIO.exists(), reason="user audio not on this machine")
def test_real_reported_recording_is_restored_above_3k_only() -> None:
    x, sample_rate = _decode_mono(REPORTED_AUDIO)
    tilt = measure_speech_tilt(x, sample_rate)
    assert tilt.measured
    assert tilt.low_shelf_db == 0.0, tilt.summary()
    assert tilt.presence_db == 0.0, tilt.summary()
    assert 5.0 < tilt.brilliance_db <= TILT_MAX_BRILLIANCE_LIFT_DB, tilt.summary()


def test_tonal_gain_caps_are_magnitudes_and_say_so() -> None:
    """A cut cap passed with the obvious negative sign must be refused, loudly.

    ``solve_tonal_gains`` negates ``max_low_cut_db`` to build the lower bound,
    so the caller passes a magnitude: 6.0 means "cut by at most 6 dB". Pass
    -6.0 -- the obvious reading of "max cut" -- and the bound became +6.0
    while the upper bound was already +6.0. ``np.clip`` against a collapsed
    range returns that single value without complaint, so the low shelf pinned
    to full LIFT on every input, including audio measured as needing a cut,
    and nothing said a word. Found by making the mistake in a probe.
    """
    with pytest.raises(ValueError, match="max_low_cut_db"):
        solve_tonal_gains(
            48000,
            -3.0,
            2.0,
            1.5,
            max_low_cut_db=-TILT_MAX_LOW_CUT_DB,
            max_low_lift_db=6.0,
            max_presence_db=4.0,
            max_brilliance_db=3.0,
        )


def test_a_low_band_measured_as_hot_is_actually_cut() -> None:
    """The bound this protects: a negative request must produce a real cut."""
    low, _, _ = solve_tonal_gains(
        48000,
        -3.0,
        2.0,
        1.5,
        max_low_cut_db=TILT_MAX_LOW_CUT_DB,
        max_low_lift_db=6.0,
        max_presence_db=4.0,
        max_brilliance_db=3.0,
    )
    assert low < 0.0, f"a low band asked to come down was moved by {low:+.2f} dB"
    assert low >= -TILT_MAX_LOW_CUT_DB
