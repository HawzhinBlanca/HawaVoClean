#!/usr/bin/env python3
"""Mutation gate: prove the test suite can detect real regressions — and prove
it was the *right* test that detected each one.

Applies behaviour-breaking mutations one at a time and requires, for every
mutation, that at least one of its **declared owning tests** fails. A mutation
that leaves its owners green is an untested behaviour: the gate fails and the
missing test must be written.

Why it is built this way
------------------------
The first version of this gate ran ``pytest -x`` and treated *any* non-zero
exit as "caught". That is vacuous in two separate ways, and an audit caught it
doing both at once:

1. ``-x`` stops at the first failure, so the reported failure is very often not
   the mutated behaviour at all — it is simply whatever ran first. ``tests/``
   collects ``chaos`` before ``unit``, so a single unstable chaos test
   short-circuited the run **before the mutated module was ever imported**. The
   audit's run printed "12/12 caught" with 7 of the 12 credited to
   ``tests/chaos/test_interrupt_cleanup.py`` — a test that touches none of the
   mutated code.
2. Any pre-existing failure on the tree — one broken test, one flake — makes
   every mutation "caught" forever. The gate reports a perfect score precisely
   when the suite is at its least trustworthy.

So this version:

* **Establishes a green baseline first.** The whole suite must pass before a
  single mutation is applied. A red baseline aborts loudly; it is not a gate
  result, it is a broken tree, and any score computed on top of it is a lie.
* **Credits a mutation only to its owners.** Each mutation declares the tests
  that exercise the behaviour it breaks. Credit requires one of *those* to
  fail. No unrelated failure — flake, environment, or collateral damage — can
  ever be mistaken for detection.
* **Records which tests failed**, and asserts that set is non-empty and
  disjoint from the baseline's failures and from the documented
  :data:`QUARANTINE` list.
* **Verifies every owner exists** before running anything, so a renamed or
  deleted test is an abort rather than a silent hole.
* **Restores each file from an in-memory copy**, not ``git checkout --``, so a
  concurrent edit elsewhere in the tree can never be destroyed by the gate.

Run from the repo root::

    python scripts/mutation_gate.py           # owner-scoped: fast, exact credit
    python scripts/mutation_gate.py --full    # whole suite per mutation, also
                                              # reports collateral damage
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Tests that must never be allowed to credit a mutation, whatever they do.
#: Each entry is a node-id prefix plus the reason it is here. This list is not
#: an excuse to leave a test broken — it is a statement that a *pass* from this
#: test is not evidence about the code under mutation.
QUARANTINE: dict[str, str] = {
    "tests/chaos/": (
        "Chaos tests drive real subprocesses, signals and the process table. "
        "Their verdict depends on machine state, and none of them exercises a "
        "mutated module's arithmetic — so a failure here is never evidence "
        "that a mutation was detected."
    ),
    "tests/fuzz/": (
        "The fuzz gate is deselected by default (`-m 'not fuzz'`) and runs the "
        "CLI per case; it is a separate gate with its own runtime budget."
    ),
}


@dataclass(frozen=True)
class Mutation:
    """One behaviour-breaking edit, and the tests that must notice it."""

    code: str
    name: str
    rel: str
    old: str
    new: str
    #: pytest node ids (or `file::test` prefixes) that exercise the behaviour
    #: this mutation breaks. At least one of them MUST fail.
    owners: tuple[str, ...] = field(default=())

    @property
    def title(self) -> str:
        return f"{self.code}: {self.name}"


MUTATIONS: list[Mutation] = [
    Mutation(
        "M1",
        "limiter ceiling raised 0.5 dB",
        "src/hawavoclean/finishing/limiter.py",
        "ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))",
        "ceiling_linear = float(10.0 ** ((ceiling_dbtp + 0.5) / 20.0))",
        # Owners: the ceiling is the limiter's one hard promise. Discovered by
        # running the whole suite against this mutation: 58 tests broke, and
        # these are the three that are ABOUT the ceiling.
        owners=(
            "tests/unit/test_limiter_loudness.py::test_limiter_enforces_ceiling",
            "tests/property/test_limiter_ceiling.py::test_limiter_enforces_ceiling_without_flat_tops",
            "tests/property/test_properties.py::test_property_limiter_strictly_enforces_ceiling",
        ),
    ),
    Mutation(
        "M2",
        "lookahead anticipation deleted (shift semantics)",
        "src/hawavoclean/finishing/limiter.py",
        """    if lookahead_samples > 0 and samples > 1:
        size = min(lookahead_samples + 1, samples)
        # Look-AHEAD window [i, i+size-1]: origin = -(size//2) is always within
        # scipy's valid range (-(size//2) .. (size-1)//2) for any size parity.
        windowed_min = scipy.ndimage.minimum_filter1d(
            inst_gain, size=size, origin=-(size // 2), mode="nearest"
        )
    else:
        windowed_min = inst_gain
    del peak_envelope, inst_gain  # consumed; free before the next full-size array
    anticipated = _slope_limited_min_envelope(windowed_min, lookahead_samples)""",
        """    windowed_min = inst_gain
    del peak_envelope, inst_gain
    anticipated = windowed_min""",
        # Owner: the ONLY test in the suite that distinguishes a look-ahead
        # envelope from a plain one. If it is ever deleted, this mutation
        # becomes invisible — which is exactly what the gate must refuse.
        owners=(
            "tests/property/test_limiter_ceiling.py::test_gain_reduction_anticipates_the_peak",
        ),
    ),
    Mutation(
        "M3",
        "guard verdict always PASS",
        "src/hawavoclean/guard/verdict.py",
        "verdict = GuardVerdict.PASS if len(reasons) == 0 else GuardVerdict.REVERT",
        "verdict = GuardVerdict.PASS",
        # Owners: the guard is only worth having if it can say no.
        owners=(
            "tests/unit/test_guard_modes.py::test_integrity_mode_still_rejects_broken_audio",
            "tests/unit/test_guard_modes.py::test_anchor_gating_applies_only_in_strict_mode",
        ),
    ),
    Mutation(
        "M4",
        "policy returns enhanced audio on guard REVERT",
        "src/hawavoclean/policy/decision.py",
        """    return (
        UnitPolicyDecision(
            selected_waveform=orig_core_waveform.copy(),
            is_enhanced=False,
            chosen_strength=0.0,
            guard_verdict=final_verdict,
            guard_scores=final_scores,
            decision_reason=f"Reverted to original audio: {failure_reasons}",
        ),
        orig_probe,
    )""",
        """    return (
        UnitPolicyDecision(
            selected_waveform=candidates[0][1],
            is_enhanced=True,
            chosen_strength=candidates[0][0],
            guard_verdict=final_verdict,
            guard_scores=final_scores,
            decision_reason=f"Reverted to original audio: {failure_reasons}",
        ),
        orig_probe,
    )""",
        # Owner: a REVERT verdict must hand back the ORIGINAL waveform.
        owners=(
            "tests/unit/test_hunt_round3_dsp_guard.py::test_empty_candidate_never_passes_the_guard",
        ),
    ),
    Mutation(
        "M5",
        "unit reported enhanced when the enhancer failed",
        "src/hawavoclean/policy/decision.py",
        """                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=False,
                chosen_strength=0.0,
                guard_verdict=GuardVerdict.ERROR,""",
        """                selected_waveform=orig_core_waveform.copy(),
                is_enhanced=True,
                chosen_strength=1.0,
                guard_verdict=GuardVerdict.ERROR,""",
        # Owner: an enhancer crash must fail closed, not be reported enhanced.
        owners=("tests/unit/test_policy_verdict.py::test_policy_enhancer_error_fails_closed",),
    ),
    Mutation(
        "M6",
        "stitch writes every unit one sample late",
        "src/hawavoclean/assembly/stitch.py",
        "        timeline[start:end] = wave",
        "        timeline[start + 1 : min(end + 1, total_samples)] = wave[: min(end + 1, total_samples) - (start + 1)]",
        # Owners: sample-exact placement on the timeline.
        owners=(
            "tests/unit/test_stitch_no_duplication.py::test_two_unit_butt_joint_duplicates_nothing",
            "tests/unit/test_hunt_round3_dsp_guard.py::test_stitch_declick_skips_forced_cut_and_ramps_natural_joint",
        ),
    ),
    Mutation(
        "M7",
        "assembly sample-count invariant disabled",
        "src/hawavoclean/assembly/validate.py",
        """    if samples != expected_samples:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )""",
        """    if False:
        raise OutputValidationError(
            f"Assembled output samples {samples} != expected input samples {expected_samples}"
        )""",
        # Owner: the invariant has exactly one test, and it is this one.
        owners=("tests/unit/test_assembly.py::test_validate_rejects_sample_count_mismatch",),
    ),
    Mutation(
        "M8",
        "ambiguous stereo silently classified dual-mono",
        "src/hawavoclean/audio/channels.py",
        "raise AmbiguousStereoError(",
        "return ChannelMode.DUAL_MONO_SAME\n        raise AmbiguousStereoError(",
        # Owner: ambiguous stereo must raise, never be silently downmixed.
        owners=("tests/unit/test_channels.py::test_channel_mode_ambiguous_stereo_rejected",),
    ),
    Mutation(
        "M9",
        "overwrite commit pointer keeps the previous generation",
        "src/hawavoclean/publication.py",
        """            _checkpoint("before_pointer_commit")
            _replace_current(paths, generation_id)
            _replace_json(""",
        """            _checkpoint("before_pointer_commit")
            _replace_current(paths, prior or generation_id)
            _replace_json(""",
        # Owner: an overwrite is complete only when the single authoritative
        # pointer names the new immutable generation. This catches a publisher
        # that prepares perfect artifacts yet silently keeps serving the old set.
        owners=(
            "tests/unit/test_publication_transaction.py::test_overwrite_retains_prior_immutable_generation",
        ),
    ),
    Mutation(
        "M10",
        "Wiener gain floor removed",
        "src/hawavoclean/enhancement/production.py",
        '"gain_floor": 0.05,',
        '"gain_floor": 0.0,',
        # Owners: the gain floor is part of the core's locked parameter set, so
        # the lockfile invariant is its first and sharpest detector; the
        # mastering regression is the one that hears the difference.
        # (44 tests broke under this mutation — nearly all of them the
        # lockfile check cascading through every CLI entry point.)
        owners=(
            "tests/unit/test_studio_core.py::test_every_registered_core_has_a_consistent_lockfile",
            "tests/unit/test_mastering_regression.py::test_published_file_hits_loudness_target_and_structure",
        ),
    ),
    Mutation(
        "M11",
        "encoder drops the final sample",
        "src/hawavoclean/audio/encode.py",
        "        sf.write(\n            str(dest_path),\n            interleaved,",
        "        sf.write(\n            str(dest_path),\n            interleaved[:-1],",
        # Owners: encode/probe round-trips that count samples.
        owners=(
            "tests/unit/test_audio_io.py::test_encode_and_probe_float32",
            "tests/unit/test_edge_paths.py::test_encode_float32_subtype",
        ),
    ),
    Mutation(
        "M12",
        "mono loudness target off by 6 dB",
        "src/hawavoclean/resources/configs/production.toml",
        "target_lufs_mono = -19.0",
        "target_lufs_mono = -13.0",
        # Owner: the only test that measures the published file's loudness.
        owners=(
            "tests/unit/test_mastering_regression.py::test_published_file_hits_loudness_target_and_structure",
        ),
    ),
    Mutation(
        "M13",
        "auto multipass ships the pass it just judged regressive",
        "src/hawavoclean/multipass.py",
        """            if auto and k > 1:
                keep, reason = auto_pass_verdict(records[-1], record)
                if not keep:
                    logger.info(f"Auto mode discards pass {k}: {reason}")
                    records.append(
                        record.model_copy(update={"discarded": True, "discard_reason": reason})
                    )
                    break
""",
        "",
        # Owner: the discard is auto mode's whole promise — a pass that fails
        # the criteria must be recorded AND not shipped. Deleting the discard
        # block makes auto always run to the cap and ship the last pass.
        owners=(
            "tests/unit/test_multipass.py::test_auto_discards_regressing_pass_and_ships_previous",
        ),
    ),
    Mutation(
        "M14",
        "continuity fade lands on the wrong edge",
        "src/hawavoclean/policy/continuity.py",
        """    fade_in = max(0, min(fade_in_samples, n))""",
        """    fade_in_samples, fade_out_samples = fade_out_samples, fade_in_samples
    fade_in = max(0, min(fade_in_samples, n))""",
        # Owner: a fade on the wrong edge satisfies every "something faded"
        # assertion while leaving the seam exactly where it was. Only a test
        # that checks WHICH edge went back to the original catches it.
        owners=(
            "tests/unit/test_continuity_taper.py::test_the_fade_removes_the_step_a_hard_cut_leaves",
        ),
    ),
    Mutation(
        "M15",
        "continuity ignores a seam on a unit's left edge",
        "src/hawavoclean/policy/continuity.py",
        """        left = i > 0 and forced_cut_between(i - 1, i) and not adjusted[i - 1].is_enhanced""",
        """        left = False""",
        # Owner: half the seams are on the left. A unit between two originals
        # would fade only its right edge and ship a hard step on its left.
        owners=(
            "tests/unit/test_continuity_taper.py::test_a_unit_between_two_originals_fades_on_both_sides",
            "tests/unit/test_policy_verdict.py::test_enforce_continuity_fades_cut_speech",
        ),
    ),
    Mutation(
        "M16",
        "a unit too short to fade is faded anyway instead of reverting",
        "src/hawavoclean/policy/continuity.py",
        """    def can_afford_fade(i: int) -> bool:
        return taper <= len(orig_core_waveforms[i]) * MAX_TAPER_FRACTION""",
        """    def can_afford_fade(i: int) -> bool:
        return True""",
        # Owner: this is the fail-closed path. Without it a 20 ms unit gets a
        # 30 ms fade — the whole unit, and then some.
        owners=(
            "tests/unit/test_continuity_taper.py::test_a_unit_too_short_to_fade_still_reverts",
            "tests/unit/test_continuity_taper.py::test_a_cascade_of_short_units_terminates",
        ),
    ),
    Mutation(
        "M17",
        "the pipeline plans the continuity fade and never applies it",
        "src/hawavoclean/pipeline.py",
        """            if taper_in[idx] > 0 or taper_out[idx] > 0:
                final_wave = apply_continuity_taper(
                    final_wave, orig_core_waveforms[idx], taper_in[idx], taper_out[idx]
                )
                finish_actions = [
                    *finish_actions,
                    f"{CONTINUITY_TAPER_ACTION}(in={taper_in[idx]},out={taper_out[idx]})",
                ]
""",
        "",
        # Owner: every unit test covers what the policy DECIDES. Only an
        # end-to-end run notices when the decision never reaches the audio.
        owners=(
            "tests/integration/test_continuity_taper_pipeline.py::test_a_planned_fade_reaches_the_master_and_the_report",
        ),
    ),
    Mutation(
        "M18",
        "the continuity fade's ramp becomes a hard step",
        "src/hawavoclean/policy/continuity.py",
        "    w = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)",
        "    w = (t >= 0.5).astype(np.float32)",
        # Owner: a step is monotone, bounded in [0,1], and still lands the
        # original at the joint — so every obvious assertion passes. It is
        # WORSE than the seam it replaces: it moves the whole discontinuity
        # 15 ms upstream, off the low-energy zero crossing the segmenter chose.
        # Only a bound on the ramp's slope tells them apart.
        owners=(
            "tests/unit/test_continuity_taper.py::test_the_fade_weight_is_monotone_across_the_window",
        ),
    ),
    Mutation(
        "M19",
        "the continuity fade resolves to the candidate, not the original",
        "src/hawavoclean/pipeline.py",
        "final_wave, orig_core_waveforms[idx], taper_in[idx], taper_out[idx]",
        "final_wave, dec.selected_waveform, taper_in[idx], taper_out[idx]",
        # Owner: the whole promise is that the joint sample IS the original
        # recording. Fading toward the guard-approved candidate still fades,
        # still changes the hash, still concentrates the change in the window —
        # and leaves a step at the joint bigger than applying no fade at all.
        owners=(
            "tests/integration/test_continuity_taper_pipeline.py::test_the_joint_of_a_real_forced_cut_is_the_original_recording",
        ),
    ),
    Mutation(
        "M20",
        "the continuity fade is planned at the wrong sample rate",
        "src/hawavoclean/pipeline.py",
        "all_units, decisions, orig_core_waveforms, audio_buf.sample_rate",
        "all_units, decisions, orig_core_waveforms, 16000",
        # Owner: ships a 10 ms fade instead of 30 ms. Every self-consistent
        # assertion still holds — a fade happened, in the right place, on the
        # right unit — so only the ABSOLUTE length in the audit trail catches it.
        owners=(
            "tests/integration/test_continuity_taper_pipeline.py::test_the_joint_of_a_real_forced_cut_is_the_original_recording",
        ),
    ),
    Mutation(
        "M21",
        "a continuity revert is filed under the guard's own REVERT",
        "src/hawavoclean/pipeline.py",
        "continuity_reverted_ids = resolution.reverted_ids",
        "continuity_reverted_ids = set()",
        # Owner: the guard rejecting audio and this rule spending good audio to
        # protect a seam are different events, and only the second is a cost
        # the rule is accountable for. Conflating them hides the rule's price
        # in a number that means something else — which is how the cascade
        # stayed invisible for as long as it did.
        owners=(
            "tests/integration/test_continuity_taper_pipeline.py::test_a_continuity_revert_is_recorded_as_one",
        ),
    ),
    Mutation(
        "M22",
        "band-split crossover moved to 2500 Hz",
        "src/hawavoclean/enhancement/studio_lowband.py",
        '"crossover_hz": 1000.0,',
        '"crossover_hz": 2500.0,',
        # Owners: the crossover is this core's one locked decision — at 2500 Hz
        # the measured spectral-hole score is 0.187 against a 0.100 threshold,
        # i.e. every unit reverted. Moving it must be a relock, so the lockfile
        # invariant and the params-hash test are the detectors.
        owners=(
            "tests/unit/test_studio_core.py::test_every_registered_core_has_a_consistent_lockfile",
            "tests/unit/test_studio_lowband_core.py::test_crossover_frequency_and_order_are_inside_the_params_hash",
            "tests/unit/test_studio_lowband_core.py::test_lock_declares_the_shipped_crossover",
        ),
    ),
    Mutation(
        "M23",
        "crossover high band taken with a second filter, not the complement",
        "src/hawavoclean/enhancement/studio_lowband.py",
        """    low = lowpass_zero_phase(enhanced, sample_rate, crossover_hz, order)
    orig_low = lowpass_zero_phase(original, sample_rate, crossover_hz, order)
    return (low + (original - orig_low)).astype(np.float32)""",
        """    import scipy.signal

    low = lowpass_zero_phase(enhanced, sample_rate, crossover_hz, order)
    sos = scipy.signal.butter(order, crossover_hz, btype="high", fs=sample_rate, output="sos")
    orig32 = np.asarray(original, dtype=np.float32)
    padlen = min(3 * (2 * len(sos) + 1), max(0, len(orig32) - 1))
    high = np.asarray(
        scipy.signal.sosfiltfilt(sos, orig32.astype(np.float64), padlen=padlen), dtype=np.float32
    )
    return (low + high).astype(np.float32)""",
        # Owner: the exact reason the split is one filter and a subtraction. Two
        # independently designed Butterworths sum to a ripple and a phase step
        # at the crossover — precisely where the first formant is — and only
        # the reconstruction property notices.
        owners=(
            "tests/unit/test_studio_lowband_core.py::test_crossover_reconstructs_the_input_when_the_model_changes_nothing",
        ),
    ),
    Mutation(
        "M24",
        "guard protected-band rejection disabled",
        "src/hawavoclean/restoration/guard.py",
        "        if not prot_verif.passes_invariance:",
        "        if False and not prot_verif.passes_invariance:",
        # Owner: the revert test's broken candidate is rejected by exactly this
        # layer (its FAIL verdict carries the protected-band metrics). With the
        # layer disabled the candidate sails through every other check and the
        # guard ships altered protected-band audio as a PASS.
        owners=(
            "tests/unit/test_restoration_guard.py::test_guard_r_revert_is_reported_as_fail_not_pass",
        ),
    ),
    Mutation(
        "M25",
        "guard admits the zero-strength Natural candidate to evaluation",
        "src/hawavoclean/restoration/guard.py",
        "            (c for c in candidates if c.strength > 0.0), key=lambda c: c.strength, reverse=True",
        "            (c for c in candidates), key=lambda c: c.strength, reverse=True",
        # Owner: the original shipped bug. The 0.0 candidate IS the Natural
        # audio, so scored against itself it passes trivially and a total
        # revert is published as a passing verdict.
        owners=(
            "tests/unit/test_restoration_guard.py::test_guard_r_revert_is_reported_as_fail_not_pass",
        ),
    ),
    Mutation(
        "M26",
        "guard clipping rejection threshold effectively removed",
        "src/hawavoclean/restoration/guard.py",
        "        if peak_amp > 1.05:",
        "        if peak_amp > 1e9:",
        owners=(
            "tests/unit/test_restoration_dsp_branches.py::test_guard_rejects_clipping_above_headroom",
        ),
    ),
    Mutation(
        "M27",
        "protected-band transition mask generates everywhere",
        "src/hawavoclean/restoration/protected_band.py",
        "    mask[freqs >= f_high] = 1.0",
        "    mask[:] = 1.0",
        # Owners: the mask IS the protected band. All-ones means the trusted
        # observed spectrum below the cutoff is replaced by generated content.
        owners=(
            "tests/unit/test_restoration_protected_band.py::test_transition_mask_shape_and_values",
            "tests/unit/test_restoration_protected_band.py::test_merge_protected_spectrum_invariance",
        ),
    ),
    Mutation(
        "M28",
        "policy healthy-bandwidth bypass deleted",
        "src/hawavoclean/restoration/policy.py",
        """        if (
            not bandwidth_est.restore_recommended
            or bandwidth_est.confidence < self.config.cutoff_confidence_min
        ):""",
        "        if False:",
        # Owners: healthy or uncertain audio must never enter the generative
        # path at all — bypassing is the No-False-Restoration promise.
        owners=(
            "tests/unit/test_restoration_validation_branches.py::test_bypass_when_bandwidth_healthy",
            "tests/unit/test_restoration_validation_branches.py::test_bypass_on_low_confidence",
        ),
    ),
    Mutation(
        "M29",
        "restorer attests a fabricated weights hash",
        "src/hawavoclean/restoration/hawarestore_kd.py",
        "        self.weights_sha256 = hash_file(ckpt_path)",
        '        self.weights_sha256 = "0" * 64',
        # Owner: the report's weights_sha256 must be computed from the file the
        # network actually loaded — a fabricated digest is provenance fraud.
        owners=(
            "tests/unit/test_restoration_hawarestore.py::test_hawarestore_kd_reports_hash_of_loaded_weights",
        ),
    ),
    Mutation(
        "M30",
        "bandwidth detector recommends restoring full-band audio",
        "src/hawavoclean/restoration/bandwidth.py",
        """                confidence=0.95,
                shape="fullband",
                restore_recommended=False,""",
        """                confidence=0.95,
                shape="fullband",
                restore_recommended=True,""",
        # Owner: this early return is where full-band audio is decided now
        # that the detector requires a cliff rather than a slope. The gate
        # caught the drift: this mutation used to sit on the 16 kHz rule
        # below, which no signal reaches, so it went green while the
        # behaviour it names was in another branch entirely.
        owners=("tests/unit/test_restoration_bandwidth.py::test_bandwidth_detector_fullband",),
    ),
    Mutation(
        "M31",
        "strengths ladder no longer requires the 0.0 fallback",
        "src/hawavoclean/restoration/config.py",
        """        if 0.0 not in v:
            raise ValueError("Strengths ladder must include 0.0 as safe fallback candidate.")""",
        """        if False:
            raise ValueError("Strengths ladder must include 0.0 as safe fallback candidate.")""",
        owners=(
            "tests/unit/test_restoration_validation_branches.py::test_strengths_without_zero_fallback_rejected",
        ),
    ),
    Mutation(
        "M32",
        "restore child command loses its absolute profiles dir",
        "src/hawavoclean/server/jobs.py",
        """        cmd += ["--mode", "restore", "--speaker-id", str(record.speaker_id)]
        cmd += ["--profiles-dir", str(profiles_root())]""",
        """        cmd += ["--mode", "restore", "--speaker-id", str(record.speaker_id)]""",
        # Owner: without the explicit absolute dir the child resolves profiles
        # against its own working directory — the CWD trap this repo has
        # already been bitten by twice.
        owners=("tests/unit/test_server_jobs.py::test_default_command_restore_mode_flags",),
    ),
    Mutation(
        "M33",
        "server stops validating speaker_id before it reaches a child argv",
        "src/hawavoclean/server/app.py",
        "        if req.speaker_id is not None and not SPEAKER_ID_PATTERN.fullmatch(req.speaker_id):",
        "        if False:",
        owners=("tests/unit/test_server_api.py::test_restore_job_request_validation",),
    ),
    Mutation(
        "M35",
        "job identity ignores which speaker rebuilt the audio",
        "src/hawavoclean/hashing.py",
        '''    if restore_context:
        composite = f"{composite}:{restore_context}"''',
        '''    if False:
        composite = f"{composite}:{restore_context}"''',
        # Owner: the report, the provenance record and the dither seed are
        # all keyed on job_id. Drop the restore inputs and a natural master,
        # a reconstruction of it, and a second speaker's reconstruction all
        # claim one identity -- in a system whose stated top risk is exactly
        # that confusion.
        owners=(
            "tests/unit/test_hashing_journal.py::test_job_id_separates_a_restoration_from_the_master_it_replaces",
        ),
    ),
    Mutation(
        "M36",
        "a profile may be loaded under another speaker's name",
        "src/hawavoclean/restoration/profiles.py",
        "    if profile.speaker_id != speaker_id:",
        "    if False:",
        # Owner: consent is validated against the id the profile DECLARES,
        # while the report attributes the run to the name it was looked up
        # by. Let those differ and a reconstruction is credited to someone
        # who never agreed to it.
        owners=(
            "tests/unit/test_restoration_profiles.py::test_a_profile_may_not_be_looked_up_under_another_speakers_name",
        ),
    ),
    Mutation(
        "M37",
        "high-band layer stops seeing a click as a local outlier",
        "src/hawavoclean/restoration/highband_events.py",
        '            local = ndimage.uniform_filter1d(steps, size=min(window, steps.size), mode="reflect")',
        '            local = ndimage.uniform_filter1d(steps, size=min(window, steps.size), mode="nearest")',
        # Owner: "nearest" extends the edge value outward, so a click on
        # the first or last sample becomes its own baseline and hides --
        # exactly the seam a segmented restoration stitches on.
        owners=(
            "tests/unit/test_restoration_dsp_branches.py::test_highband_click_is_rejected_wherever_it_lands",
        ),
    ),
    Mutation(
        "M38",
        "high-band layer vetoes every candidate that changed anything",
        "src/hawavoclean/restoration/highband_events.py",
        "            impulse_ratio = float(np.max(steps / (local + 1e-12)))",
        "            impulse_ratio = float(np.max(steps) / 1e-12)",
        # Owner: strip the local normalisation and the metric measures
        # raw activity again, so every restoration that adds a high band
        # trips it -- which is how the old endpoint-offset metric came to
        # admit only the candidate that changes nothing.
        owners=(
            "tests/unit/test_restoration_dsp_branches.py::test_highband_accepts_a_restoration_that_only_added_a_high_band",
        ),
    ),
    Mutation(
        "M34",
        "terminal cleanup callbacks run inside the job lock again",
        "src/hawavoclean/server/jobs.py",
        """        # Queue the callbacks; ``_locked`` runs them once the lock is released.
        self._pending_terminal.append(record)""",
        """        for callback in self._terminal_callbacks:
            callback(record)""",
        # Owner: the callbacks the manager invites callers to register have to
        # be able to ask it a question -- retention checks active_input_paths()
        # before deleting an upload. Under the lock that is a self-deadlock on
        # a non-reentrant Lock, which is why the owning test asserts from a
        # watcher thread: once wedged, every in-line assertion blocks too.
        owners=(
            "tests/unit/test_server_retention.py::test_a_terminal_callback_may_ask_the_manager_a_question",
        ),
    ),
]

_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")
PYTEST_BASE = [sys.executable, "-m", "pytest", "-q", "--tb=no", "-rf", "-p", "no:cacheprovider"]


def run_pytest(selection: list[str] | None = None, timeout: int = 2400) -> tuple[int, set[str]]:
    """Run pytest and return ``(returncode, failing node ids)``.

    Deliberately NOT ``-x``: the whole point is to know *which* tests failed,
    and a run that stops at the first failure cannot answer that.
    """
    proc = subprocess.run(
        PYTEST_BASE + (selection or []),
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=timeout,
    )
    out = proc.stdout + proc.stderr
    failed = {m.group(1) for line in out.splitlines() if (m := _FAILED_LINE.match(line))}
    return proc.returncode, failed


def collect_ids(selection: list[str]) -> set[str]:
    """Node ids pytest can actually resolve for ``selection`` (empty on error)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            # `-q -q`: the project's ini adds `-v`, and only a NEGATIVE net
            # verbosity makes --collect-only print one node id per line rather
            # than a tree. A single -q cancels the -v and prints the tree.
            "-q",
            "-q",
            "-p",
            "no:cacheprovider",
            *selection,
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    if proc.returncode != 0:
        return set()
    ids: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "-", "no tests ran")):
            ids.add(line)
    return ids


def quarantined(node_id: str) -> str | None:
    for prefix, reason in QUARANTINE.items():
        if node_id.startswith(prefix):
            return reason
    return None


def owner_matches(node_id: str, owner: str) -> bool:
    """Whether ``node_id`` is ``owner``, a parametrisation of it
    (``file::test[param]``), or a test inside it when ``owner`` names a file or
    a class."""
    return node_id == owner or node_id.startswith(f"{owner}[") or node_id.startswith(f"{owner}::")


def owned_by(node_id: str, owners: tuple[str, ...]) -> bool:
    return any(owner_matches(node_id, o) for o in owners)


def dirty_paths(paths: list[str]) -> list[str]:
    """Which of ``paths`` git reports as modified (staged or unstaged)."""
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return [line[3:].strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full",
        action="store_true",
        help="run the WHOLE suite per mutation instead of only its owners; "
        "credit still requires an owner failure, and everything else that "
        "broke is reported as collateral damage.",
    )
    args = ap.parse_args()

    print("=" * 78)
    print("  MUTATION GATE — credit is only ever given to a mutation's own tests")
    print("=" * 78)
    for prefix, reason in QUARANTINE.items():
        print(f"  quarantined (never credits a mutation): {prefix}")
        print(f"      {reason}")

    targets = sorted({m.rel for m in MUTATIONS})

    # ---------------------------------------------------------------- preflight
    dirty = dirty_paths(targets)
    if dirty:
        print("\nABORT: a file this gate mutates has uncommitted changes:")
        for d in dirty:
            print(f"  {d}")
        print("Commit or stash them — the gate rewrites these files in place.")
        return 2

    missing: list[str] = []
    for m in MUTATIONS:
        if not m.owners:
            missing.append(f"{m.code}: no owning tests declared")
            continue
        resolved = collect_ids(list(m.owners))
        for owner in m.owners:
            if not any(owner_matches(r, owner) for r in resolved):
                missing.append(f"{m.code}: owner does not exist — {owner}")
            elif quarantined(owner):
                missing.append(f"{m.code}: owner is quarantined — {owner}")
    if missing:
        print("\nABORT: the mutation-to-test pinning is broken:")
        for line in missing:
            print(f"  {line}")
        print("Every mutation must name tests that exist and are creditable.")
        return 2

    # ---------------------------------------------------------------- baseline
    print("\n[baseline] running the whole suite on the clean tree…")
    rc, baseline_failed = run_pytest()
    if rc != 0 or baseline_failed:
        print(f"\nABORT: the suite is NOT green on the clean tree (pytest rc={rc}).")
        for f in sorted(baseline_failed):
            print(f"  FAILED {f}")
        if not baseline_failed:
            print("  (pytest failed without naming a test — collection or plugin error)")
        print(
            "\nA mutation gate run on a red suite is meaningless: every mutation\n"
            "would be 'caught' by the failure that was already there. Fix the\n"
            "suite, then run the gate."
        )
        return 2
    print("[baseline] green — 0 failures. Mutations may proceed.\n")

    # --------------------------------------------------------------- mutations
    rows: list[tuple[str, str, str]] = []
    caught = 0
    for m in MUTATIONS:
        path = REPO / m.rel
        original = path.read_text()
        if m.old not in original:
            print(f"ABORT: mutation target not found in {m.rel} for {m.code}:\n{m.old[:160]}")
            return 2
        path.write_text(original.replace(m.old, m.new, 1))
        try:
            selection = None if args.full else list(m.owners)
            _, failed = run_pytest(selection)
        finally:
            # Restored from the copy taken above, never `git checkout --`: the
            # gate must not be able to destroy a concurrent edit it did not make.
            path.write_text(original)
        if path.read_text() != original:  # pragma: no cover - filesystem fault
            print(f"ABORT: failed to restore {m.rel}!")
            return 2

        regressions = failed - baseline_failed
        owner_fails = sorted(f for f in regressions if owned_by(f, m.owners))
        collateral = sorted(
            f for f in regressions if not owned_by(f, m.owners) and not quarantined(f)
        )
        credited_flakes = sorted(f for f in regressions if quarantined(f))

        if owner_fails:
            caught += 1
            detail = owner_fails[0].split("::", 1)[-1]
            if len(owner_fails) > 1:
                detail += (
                    f" (+{len(owner_fails) - 1} more owner{'s' if len(owner_fails) > 2 else ''})"
                )
            rows.append(("caught", m.title, detail))
        elif regressions:
            rows.append(
                (
                    "MISSED",
                    m.title,
                    "no owning test failed; only non-owners broke: "
                    + ", ".join((collateral + credited_flakes)[:3]),
                )
            )
        else:
            rows.append(("MISSED", m.title, "suite stayed GREEN — this behaviour is untested"))

        print(f"[{rows[-1][0]:>6}] {m.title}")
        for f in owner_fails:
            print(f"           owner failed: {f}")
        if args.full:
            for f in collateral:
                print(f"           collateral:   {f}")
            for f in credited_flakes:
                print(f"           quarantined:  {f}  (never credited)")

    # ----------------------------------------------------------------- report
    print()
    print("-" * 78)
    for status, title, detail in rows:
        print(f"[{status:>6}] {title} — {detail}")
    print("-" * 78)
    mode = "whole suite" if args.full else "owning tests only"
    print(f"mutation gate ({mode}): {caught}/{len(MUTATIONS)} caught by their own tests")

    still_dirty = dirty_paths(targets)
    if still_dirty:
        print(f"ABORT: tree left dirty after the gate: {still_dirty}")
        return 2
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
