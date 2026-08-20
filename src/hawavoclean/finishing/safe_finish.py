"""Safe finishing orchestration and Guard B validation ladder."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from hawavoclean.config import FinishingConfig, GuardConfig
from hawavoclean.finishing.deess import apply_split_band_deesser
from hawavoclean.finishing.detect import (
    TILT_MAX_BRILLIANCE_LIFT_DB,
    TILT_MAX_LOW_CUT_DB,
    TILT_MAX_LOW_LIFT_DB,
    TILT_MAX_PRESENCE_LIFT_DB,
    SpeechTiltReport,
    detect_defects,
    measure_speech_tilt,
)
from hawavoclean.finishing.dynamics import apply_dialogue_leveler
from hawavoclean.finishing.eq import apply_speech_eq, apply_tonal_restoration, solve_tonal_gains
from hawavoclean.finishing.repair import (
    remove_dc_subsonic,
    remove_electrical_hum,
    repair_transient_clicks,
)
from hawavoclean.guard.protocol import ProbeResult, SpectralProbe
from hawavoclean.guard.verdict import (
    GuardVerdict,
    evaluate_guard_pass,
)


@dataclass
class SafeFinishResult:
    """Outcome of local finishing and Guard B verification."""

    finished_waveform: np.ndarray[Any, np.dtype[np.float32]]
    preset_applied: Literal["gentle", "minimal", "bypass"]
    guard_b_verdict: GuardVerdict
    guard_b_scores: dict[str, float | str | bool] = field(default_factory=dict)
    actions_taken: list[str] = field(default_factory=list)


def apply_finishing_stages(
    waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    config: FinishingConfig,
    intensity: Literal["gentle", "minimal"],
    tilt: SpeechTiltReport | None = None,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], list[str]]:
    """Apply deterministic finishing chain conditioned on defect detection.

    `tilt` is the FILE-level tonal measurement when the caller has one. Units
    are finished independently, so measuring the tilt per unit lets the tone
    step between adjacent blocks — 2.8 dB of 3-6 kHz across two 12 s units of
    the recording that prompted this, which is an audible pump at every unit
    boundary. The pipeline measures once for the whole file and passes it here;
    a direct caller that does not gets a per-unit measurement instead.
    """
    current = waveform.copy()
    actions: list[str] = []

    defects = detect_defects(current, sample_rate)

    # 1. DC & Subsonic Rumble removal
    if defects.has_dc_offset or config.dc_subsonic_cutoff_hz > 30.0:
        current = remove_dc_subsonic(current, sample_rate, cutoff_hz=config.dc_subsonic_cutoff_hz)
        actions.append(f"dc_subsonic_removed(cutoff={config.dc_subsonic_cutoff_hz}Hz)")

    # 2. Narrow de-hum
    if config.dehum_50_60hz and defects.has_hum:
        current = remove_electrical_hum(current, sample_rate, hum_freq_hz=defects.hum_freq_hz)
        actions.append(f"dehum_notch({defects.hum_freq_hz}Hz)")

    # 3. Click repair
    if config.declick and defects.click_count > 0:
        current, repaired = repair_transient_clicks(current)
        if repaired > 0:
            actions.append(f"clicks_repaired(count={repaired})")

    # 4. EQ adjustment — ONLY when low-mids are in genuine excess of a normal
    # voice, and scaled to the excess (capped). No blanket presence/air boost:
    # the earlier unconditional +2.5 dB at 3-6 kHz plus a -3 dB low-mid cut
    # re-voiced every recording thin and bright.
    if config.dynamic_eq and defects.has_mud:
        from hawavoclean.finishing.detect import (
            NORMAL_VOICE_MUD_REFERENCE_DB,
        )

        excess = defects.mud_imbalance_db - NORMAL_VOICE_MUD_REFERENCE_DB
        # Take back a third of the excess, gently. The filter's band gain
        # lands ~1.5x its nominal setting in the 250-500 Hz measure, so the
        # nominal cap of 2 dB keeps the audible cut under ~3 dB.
        cut = -min(2.0, max(0.5, excess / 3.0))
        if intensity == "minimal":
            cut *= 0.5
        current = apply_speech_eq(
            current, sample_rate, mud_cut_db=cut, presence_boost_db=0.0, air_shelf_db=0.0
        )
        actions.append(f"low_mid_trim({cut:+.1f}dB, excess={excess:+.1f}dB)")

    # 4b. Measured tonal restoration. Distinct from the mud trim above and
    # deliberately kept separate from it: the mud trim answers "are the
    # low-mids in excess", this answers "does this voice reach the
    # intelligibility target, and by how much is it short". A recording can
    # need one, the other, both or neither, and both are bounded, so the worst
    # case is the sum of two bounded moves.
    sibilance_may_have_changed = False
    if config.tonal_restoration:
        report = tilt if tilt is not None else measure_speech_tilt(current, sample_rate)
        # The minimal rung of the Guard B ladder halves everything, as the rest
        # of this chain does: if gentle was rejected, do less, not something else.
        scale = 1.0 if intensity == "gentle" else 0.5
        want_low = report.low_shelf_db * scale
        want_presence = report.presence_db * scale
        want_brilliance = report.brilliance_db * scale
        if report.measured and report.is_correction:
            cap = float(config.max_tonal_gain_db)
            low_gain, presence_gain, brilliance_gain = solve_tonal_gains(
                sample_rate,
                want_low,
                want_presence,
                want_brilliance,
                max_low_cut_db=min(TILT_MAX_LOW_CUT_DB, cap),
                max_low_lift_db=min(TILT_MAX_LOW_LIFT_DB, cap),
                max_presence_db=min(TILT_MAX_PRESENCE_LIFT_DB, cap),
                max_brilliance_db=min(TILT_MAX_BRILLIANCE_LIFT_DB, cap),
            )
            before = current
            current = apply_tonal_restoration(
                current,
                sample_rate,
                low_gain,
                presence_gain,
                brilliance_gain,
                max_abs_gain_db=cap,
            )
            if current is not before:
                sibilance_may_have_changed = brilliance_gain >= 0.5
                actions.append(
                    f"tonal_restore(low={low_gain:+.1f}dB,presence={presence_gain:+.1f}dB,"
                    f"brilliance={brilliance_gain:+.1f}dB; {report.summary()})"
                )

    # 5. De-essing. A brilliance lift can raise sibilance that was not harsh
    # before it, and the detection above ran on the unlifted signal — so when
    # the tonal stage moved 3-6 kHz, ask again rather than shipping the
    # sibilance the lift just created.
    if config.deess_band and sibilance_may_have_changed and not defects.has_harsh_sibilance:
        defects = detect_defects(current, sample_rate)
    if config.deess_band and defects.has_harsh_sibilance:
        max_gr = config.max_deess_gr_db if intensity == "gentle" else config.max_deess_gr_db * 0.5
        current, gr = apply_split_band_deesser(current, sample_rate, max_reduction_db=max_gr)
        if gr > 0.1:
            actions.append(f"deesser(gr_max={gr:.1f}dB)")

    # 6. Dialogue leveler
    if config.level_rider and intensity == "gentle":
        current, gr_comp = apply_dialogue_leveler(
            current, sample_rate, max_gain_reduction_db=config.max_compression_gr_db
        )
        if gr_comp > 0.1:
            actions.append(f"dialogue_leveler(gr_max={gr_comp:.1f}dB)")

    return np.ascontiguousarray(current, dtype=np.float32), actions


def safe_finish_speech_unit(
    pre_finish_waveform: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
    is_speech: bool,
    probe: SpectralProbe,
    finishing_config: FinishingConfig,
    guard_config: GuardConfig,
    cached_pre_finish_probe: ProbeResult | None = None,
    tilt: SpeechTiltReport | None = None,
) -> tuple[SafeFinishResult, ProbeResult]:
    """Execute safe finishing ladder guarded by Guard B.

    `tilt` is the file-level tonal measurement, so every unit of one recording
    receives the identical filter; see `apply_finishing_stages`.
    """
    if not finishing_config.enabled or not is_speech or finishing_config.preset == "bypass":
        return (
            SafeFinishResult(
                finished_waveform=pre_finish_waveform.copy(),
                preset_applied="bypass",
                guard_b_verdict=GuardVerdict.NO_SPEECH if not is_speech else GuardVerdict.PASS,
                actions_taken=[],
            ),
            cached_pre_finish_probe or probe.infer(pre_finish_waveform, sample_rate),
        )

    ref_probe = cached_pre_finish_probe or probe.infer(pre_finish_waveform, sample_rate)

    # Step 1: Try Gentle preset
    gentle_wave, gentle_actions = apply_finishing_stages(
        pre_finish_waveform, sample_rate, finishing_config, intensity="gentle", tilt=tilt
    )
    guard_b_gentle, _ = evaluate_guard_pass(
        orig_waveform=pre_finish_waveform,
        cand_waveform=gentle_wave,
        sample_rate=sample_rate,
        is_speech=True,
        probe=probe,
        config=guard_config,
        cached_orig_probe=ref_probe,
        is_finishing_pass=True,
    )

    if guard_b_gentle.verdict == GuardVerdict.PASS:
        return (
            SafeFinishResult(
                finished_waveform=gentle_wave,
                preset_applied="gentle",
                guard_b_verdict=GuardVerdict.PASS,
                guard_b_scores=guard_b_gentle.scores,
                actions_taken=gentle_actions,
            ),
            ref_probe,
        )

    # Step 2: Try Minimal preset
    min_wave, min_actions = apply_finishing_stages(
        pre_finish_waveform, sample_rate, finishing_config, intensity="minimal", tilt=tilt
    )
    guard_b_min, _ = evaluate_guard_pass(
        orig_waveform=pre_finish_waveform,
        cand_waveform=min_wave,
        sample_rate=sample_rate,
        is_speech=True,
        probe=probe,
        config=guard_config,
        cached_orig_probe=ref_probe,
        is_finishing_pass=True,
    )

    if guard_b_min.verdict == GuardVerdict.PASS:
        return (
            SafeFinishResult(
                finished_waveform=min_wave,
                preset_applied="minimal",
                guard_b_verdict=GuardVerdict.PASS,
                guard_b_scores=guard_b_min.scores,
                actions_taken=min_actions,
            ),
            ref_probe,
        )

    # Step 3: Ladder exhausted -> Bypass finishing entirely
    return (
        SafeFinishResult(
            finished_waveform=pre_finish_waveform.copy(),
            preset_applied="bypass",
            guard_b_verdict=GuardVerdict.REVERT,
            guard_b_scores=guard_b_min.scores,
            actions_taken=["bypassed_after_guard_b_rejection"],
        ),
        ref_probe,
    )
