"""Master processing pipeline: decode, segment, enhance, guard, finish, master, publish.

Every run recomputes every unit — there is no resume cache, so the audit
report always describes the run that produced it. The scratch workspace is
removed on success and survives only a genuine crash, for forensics.
"""

import platform
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import soundfile

from voiceclean import __version__
from voiceclean.alignment.delay import estimate_gcc_phat_delay
from voiceclean.assembly.stitch import assemble_channel_timeline
from voiceclean.assembly.validate import validate_assembled_timeline
from voiceclean.audio.channels import classify_channels, handle_channel_layout
from voiceclean.audio.decode import decode_audio
from voiceclean.audio.encode import encode_audio
from voiceclean.audio.probe import probe_audio
from voiceclean.audio.types import AudioBuffer, AudioProbeResult
from voiceclean.config import VoiceCleanConfig, load_config
from voiceclean.enhancement.production import WienerSpectralEnhancer, wiener_params_hash
from voiceclean.enhancement.validate import validate_enhancer_output
from voiceclean.enhancement.worker import IsolatedEnhancementWorker
from voiceclean.errors import PreflightError
from voiceclean.finishing.limiter import apply_lookahead_limiter
from voiceclean.finishing.loudness import compute_static_master_gain, measure_loudness_and_peaks
from voiceclean.finishing.safe_finish import safe_finish_speech_unit
from voiceclean.guard.calibration import apply_calibrated_thresholds, load_calibration_artifact
from voiceclean.guard.protocol import SpectralProbe
from voiceclean.guard.spectral_probe import SpectralSignatureProbe
from voiceclean.guard.verdict import GuardVerdict
from voiceclean.hashing import hash_bytes, hash_file, hash_json_canonical
from voiceclean.job import JobWorkspace
from voiceclean.journal import JournalEvent
from voiceclean.logging import get_logger
from voiceclean.paths import models_dir, profile_config_path, resolve_calibration_file
from voiceclean.policy.continuity import enforce_source_continuity
from voiceclean.policy.decision import UnitPolicyDecision, evaluate_unit_policy
from voiceclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    MediaStats,
    ReviewTimecode,
    UnitDecisionRecord,
    UnitSummary,
    VoiceCleanReport,
)
from voiceclean.report.summary import generate_human_summary
from voiceclean.report.writer import serialize_json_report
from voiceclean.segmentation.types import SpeechUnit
from voiceclean.segmentation.utterances import build_speech_units

logger = get_logger("pipeline")

FINAL_DECISION_BY_VERDICT: dict[GuardVerdict, str] = {
    GuardVerdict.UNVERIFIED: "original_unverified",
    GuardVerdict.ERROR: "original_error",
    GuardVerdict.REVERT: "original_reverted",
}


def _load_core_lock(core_id: str) -> dict[str, Any]:
    """Load and verify the production core lockfile. Missing or mismatched
    provenance is a hard failure, never a silent degradation."""
    lock_path = models_dir() / "production-core.lock.toml"
    if not lock_path.exists():
        raise PreflightError(
            f"Production core lockfile missing: {lock_path}. Refusing to run "
            "without verifiable core provenance."
        )
    with open(lock_path, "rb") as f:
        lock = tomllib.load(f)

    if lock.get("core_id") != core_id:
        raise PreflightError(
            f"Configured core_id {core_id!r} does not match lockfile core {lock.get('core_id')!r}"
        )
    actual_params_hash = wiener_params_hash()
    if lock.get("params_hash") != actual_params_hash:
        raise PreflightError(
            "Core parameter drift: lockfile params_hash "
            f"{str(lock.get('params_hash'))[:16]}... does not match the "
            f"implemented core {actual_params_hash[:16]}..."
        )
    if hash_json_canonical(dict(lock.get("params", {}))) != actual_params_hash:
        raise PreflightError(
            "Core lockfile [params] table does not recompute to params_hash; "
            "the lockfile has been hand-edited."
        )
    return lock


def run_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    config: VoiceCleanConfig | None = None,
    config_path: Path | str | None = None,
    profile: str = "production",
    overwrite: bool = False,
    probe_override: SpectralProbe | None = None,
) -> VoiceCleanReport:
    """Execute the complete end-to-end VoiceClean pipeline."""
    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()

    logger.info(f"Starting VoiceClean pipeline on {in_path} -> {out_path} [profile={profile}]")

    # 1. Configuration, calibration, and core provenance preflight
    is_prod = profile == "production"
    if config is None:
        cfg_file = Path(config_path) if config_path is not None else profile_config_path(profile)
        config = load_config(cfg_file, is_production=is_prod)

    calib_path = resolve_calibration_file(config.guard.calibration_file)
    calib_data = load_calibration_artifact(calib_path)
    active_guard_cfg = apply_calibrated_thresholds(config.guard, calib_data)

    core_lock = _load_core_lock(config.enhancement.core_id)

    # 2. Probe media
    media = probe_audio(in_path, max_sample_rate=config.input.max_sample_rate)
    logger.info(
        f"Probed media: {media.sample_rate}Hz, {media.channels}ch, "
        f"{media.samples:,} samples ({media.duration_s:.2f}s)"
    )

    # 3. Workspace and journal
    workspace = JobWorkspace(
        input_path=in_path,
        input_sha256=media.sha256,
        config=config,
        core_id=config.enhancement.core_id,
        guard_id=active_guard_cfg.guard_id,
        tool_version=__version__,
    )
    workspace.journal.append(
        JournalEvent.JOB_STARTED, {"input": str(in_path), "job_id": workspace.job_id}
    )
    workspace.check_disk_space(media.samples * media.channels * 12, destination=out_path.parent)
    workspace.journal.append(JournalEvent.PREFLIGHT_PASSED)

    # 4. Decode and classify channels
    audio_buf = decode_audio(media, timeout_s=config.runtime.worker_timeout_s)
    if media.samples != audio_buf.samples:
        # Sync with the exact decoded stream length if the container estimate differed
        media = AudioProbeResult(
            path=media.path,
            format_name=media.format_name,
            codec_name=media.codec_name,
            sample_rate=media.sample_rate,
            channels=media.channels,
            duration_s=audio_buf.duration_s,
            samples=audio_buf.samples,
            bit_depth=media.bit_depth,
            sha256=media.sha256,
        )
    channel_mode = classify_channels(audio_buf, declared_mode=config.input.channel_mode)
    audio_buf.channel_mode = channel_mode
    logger.info(f"Channel classification: {channel_mode}")
    workspace.journal.append(JournalEvent.AUDIO_DECODED, {"channel_mode": str(channel_mode)})

    channels_to_process, duplicate_to_stereo = handle_channel_layout(audio_buf, channel_mode)

    # 5. Segmentation
    all_units: list[SpeechUnit] = []
    unit_id_offset = 0
    for ch_idx, ch_wave in enumerate(channels_to_process):
        ch_units = build_speech_units(
            channel_waveform=ch_wave,
            sample_rate=audio_buf.sample_rate,
            channel_id=ch_idx,
            config=config.segmentation,
            start_unit_id=unit_id_offset,
        )
        all_units.extend(ch_units)
        unit_id_offset += len(ch_units)

    logger.info(
        f"Generated {len(all_units)} speech units across "
        f"{len(channels_to_process)} processing channel(s)."
    )
    workspace.journal.append(JournalEvent.SEGMENTATION_COMPLETE, {"units_count": len(all_units)})

    # 6. Models: isolated worker (or in-process core) and the fidelity probe
    if config.runtime.isolated_worker:
        worker: Any = IsolatedEnhancementWorker(
            core_id=config.enhancement.core_id,
            sample_rate=config.enhancement.model_sample_rate,
            timeout_s=config.runtime.worker_timeout_s,
        )
    else:
        worker = WienerSpectralEnhancer(
            core_id=config.enhancement.core_id,
            sample_rate=config.enhancement.model_sample_rate,
            phase_coherent=config.enhancement.phase_coherent,
        )

    probe: SpectralProbe = probe_override or SpectralSignatureProbe(
        probe_id=active_guard_cfg.probe_id,
        target_sr=16000,
    )

    # 7. Guard-A decisions for every unit (finishing comes after continuity)
    decisions: list[UnitPolicyDecision] = []
    cand_hashes: list[str | None] = []
    unit_runtimes: list[float] = []
    orig_core_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    try:
        for u in all_units:
            ch_wave = channels_to_process[u.channel_id]
            core_orig = ch_wave[u.start_sample : u.end_sample]
            orig_core_waveforms.append(core_orig)

            t_unit_start = time.perf_counter()

            if not u.is_speech:
                decisions.append(
                    UnitPolicyDecision(
                        selected_waveform=core_orig.copy(),
                        is_enhanced=False,
                        chosen_strength=0.0,
                        guard_verdict=GuardVerdict.NO_SPEECH,
                        decision_reason="Non-speech unit passthrough.",
                    )
                )
                cand_hashes.append(None)
                unit_runtimes.append((time.perf_counter() - t_unit_start) * 1000.0)
                continue

            context_wave = ch_wave[u.context_start_sample : u.context_end_sample]

            enh_core: np.ndarray[Any, np.dtype[np.float32]] | None = None
            cand_sha256: str | None = None
            try:
                enh_res = worker.enhance(context_wave, audio_buf.sample_rate)
                left_ctx = u.left_context_samples
                core_len = u.core_length_samples
                enh_trimmed = enh_res.waveform[left_ctx : left_ctx + core_len]

                valid, reason = validate_enhancer_output(core_orig, enh_trimmed, is_speech=True)
                if valid:
                    delay_res = estimate_gcc_phat_delay(
                        core_orig,
                        enh_trimmed,
                        audio_buf.sample_rate,
                        max_delay_ms=config.alignment.max_delay_ms,
                    )
                    enh_core = delay_res.aligned_candidate
                    cand_sha256 = hash_bytes(enh_core.tobytes())
                else:
                    logger.warning(f"Unit {u.unit_id} output validation failed: {reason}")
            except Exception as e:
                logger.warning(f"Enhancement worker failed for unit {u.unit_id}: {e}")
                enh_core = None

            pol_dec, _ = evaluate_unit_policy(
                orig_core_waveform=core_orig,
                enh_core_waveform=enh_core,
                sample_rate=audio_buf.sample_rate,
                is_speech=True,
                probe=probe,
                guard_config=active_guard_cfg,
                policy_config=config.policy,
                phase_coherent=config.enhancement.phase_coherent,
            )
            decisions.append(pol_dec)
            cand_hashes.append(cand_sha256)
            unit_runtimes.append((time.perf_counter() - t_unit_start) * 1000.0)

        # 8. Source continuity — BEFORE records are built and units finished,
        # so a continuity revert is what gets finished, recorded, and stitched.
        continuity_reverted_ids: set[int] = set()
        if config.policy.enforce_continuity:
            adjusted = enforce_source_continuity(all_units, decisions, orig_core_waveforms)
            for u, before, after in zip(all_units, decisions, adjusted, strict=True):
                if before.is_enhanced and not after.is_enhanced:
                    continuity_reverted_ids.add(u.unit_id)
            decisions = adjusted

        # 9. Finishing (Guard B) on surviving enhanced units, then records
        unit_decision_records: list[UnitDecisionRecord] = []
        final_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]] = []

        for idx, u in enumerate(all_units):
            dec = decisions[idx]
            t_finish_start = time.perf_counter()

            final_wave = dec.selected_waveform
            finish_preset = "bypass"
            finish_actions: list[str] = []
            guard_b_verdict: GuardVerdict | None = None
            guard_b_scores: dict[str, Any] = {}

            if u.is_speech and dec.is_enhanced and config.finishing.enabled:
                finish_res, _ = safe_finish_speech_unit(
                    pre_finish_waveform=dec.selected_waveform,
                    sample_rate=audio_buf.sample_rate,
                    is_speech=True,
                    probe=probe,
                    finishing_config=config.finishing,
                    guard_config=active_guard_cfg,
                )
                final_wave = finish_res.finished_waveform
                finish_preset = finish_res.preset_applied
                finish_actions = finish_res.actions_taken
                guard_b_verdict = finish_res.guard_b_verdict
                guard_b_scores = finish_res.guard_b_scores

            final_waveforms.append(final_wave)
            workspace.journal.append(
                JournalEvent.UNIT_COMMITTED,
                {"unit_id": u.unit_id, "enhanced": dec.is_enhanced},
            )

            if not u.is_speech:
                final_cat = "original_no_speech"
            elif u.unit_id in continuity_reverted_ids:
                final_cat = "original_continuity"
            elif dec.is_enhanced:
                final_cat = "enhanced"
            else:
                final_cat = FINAL_DECISION_BY_VERDICT.get(dec.guard_verdict, "original_reverted")

            runtime_ms = unit_runtimes[idx] + (time.perf_counter() - t_finish_start) * 1000.0

            unit_decision_records.append(
                UnitDecisionRecord(
                    unit_id=u.unit_id,
                    channel=u.channel_id,
                    start_sample=u.start_sample,
                    end_sample=u.end_sample,
                    start_time_s=float(u.start_sample / audio_buf.sample_rate),
                    end_time_s=float(u.end_sample / audio_buf.sample_rate),
                    is_speech=u.is_speech,
                    input_sha256=u.input_sha256 or hash_bytes(orig_core_waveforms[idx].tobytes()),
                    candidate_sha256=cand_hashes[idx],
                    output_sha256=hash_bytes(final_wave.astype(np.float32).tobytes()),
                    guard_a_verdict=dec.guard_verdict,
                    guard_a_scores=dec.guard_scores,
                    guard_b_verdict=guard_b_verdict,
                    guard_b_scores=guard_b_scores,
                    chosen_strength=dec.chosen_strength,
                    finish_preset_applied=finish_preset,
                    finish_actions=finish_actions,
                    final_decision=final_cat,
                    decision_reason=dec.decision_reason,
                    runtime_ms=runtime_ms,
                )
            )

    finally:
        if hasattr(worker, "close"):
            worker.close()

    # 10. Assembly
    assembled_channels: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    for ch_idx in range(len(channels_to_process)):
        ch_pairs = [
            (u, final_waveforms[i]) for i, u in enumerate(all_units) if u.channel_id == ch_idx
        ]
        ch_timeline = assemble_channel_timeline(
            units=[p[0] for p in ch_pairs],
            unit_waveforms=[p[1] for p in ch_pairs],
            total_samples=media.samples,
            sample_rate=audio_buf.sample_rate,
        )
        assembled_channels.append(ch_timeline)

    if duplicate_to_stereo and len(assembled_channels) == 1:
        assembled_channels.append(assembled_channels[0].copy())

    assembled_data = np.stack(assembled_channels, axis=0)
    assembled_buffer = AudioBuffer(
        data=assembled_data,
        sample_rate=audio_buf.sample_rate,
        channel_mode=channel_mode,
    )

    validate_assembled_timeline(
        assembled_buffer=assembled_buffer,
        expected_channels=media.channels,
        expected_samples=media.samples,
        expected_sample_rate=media.sample_rate,
        units=all_units,
    )
    workspace.journal.append(JournalEvent.ASSEMBLY_COMPLETE)

    # 11. Loudness normalization and true-peak limiting
    initial_loudness = measure_loudness_and_peaks(
        assembled_buffer.data, assembled_buffer.sample_rate
    )
    target_lufs = (
        config.loudness.target_lufs_stereo
        if media.channels > 1
        else config.loudness.target_lufs_mono
    )

    static_gain_db = compute_static_master_gain(
        measured_lufs=initial_loudness.integrated_lufs,
        target_lufs=target_lufs,
        current_true_peak_dbtp=initial_loudness.true_peak_dbtp,
        true_peak_ceiling_dbtp=config.loudness.true_peak_ceiling_dbtp,
        max_limiter_reduction_db=config.loudness.max_limiter_reduction_db,
    )

    gain_linear = 10.0 ** (static_gain_db / 20.0)
    gained_data = assembled_buffer.data * gain_linear

    limited_res = apply_lookahead_limiter(
        waveform=gained_data,
        sample_rate=assembled_buffer.sample_rate,
        ceiling_dbtp=config.loudness.true_peak_ceiling_dbtp,
    )
    if limited_res.max_gain_reduction_db > config.loudness.max_limiter_reduction_db:
        logger.warning(
            f"Limiter reduced peaks by {limited_res.max_gain_reduction_db:.2f} dB, "
            f"beyond the static-gain headroom budget of "
            f"{config.loudness.max_limiter_reduction_db:.2f} dB — transient-heavy material."
        )
    mastered_buffer = AudioBuffer(
        data=limited_res.limited_waveform,
        sample_rate=assembled_buffer.sample_rate,
        channel_mode=channel_mode,
    )

    final_loudness = measure_loudness_and_peaks(mastered_buffer.data, mastered_buffer.sample_rate)
    workspace.journal.append(
        JournalEvent.FINAL_VALIDATION_PASSED,
        {"lufs": final_loudness.integrated_lufs, "dbtp": final_loudness.true_peak_dbtp},
    )

    # 12. Encode master into the workspace
    tmp_out = workspace.root / "candidate-output.wav.tmp"
    encode_audio(
        buffer=mastered_buffer,
        output_path=tmp_out,
        output_bit_depth=config.input.output_bit_depth,
        dither=config.loudness.dither,
        seed_context=workspace.job_id,
    )

    # 13. Audit report
    enhanced_cnt = sum(1 for r in unit_decision_records if r.final_decision == "enhanced")
    continuity_cnt = sum(
        1 for r in unit_decision_records if r.final_decision == "original_continuity"
    )
    reverted_cnt = sum(1 for r in unit_decision_records if r.final_decision == "original_reverted")
    unverified_cnt = sum(
        1 for r in unit_decision_records if r.final_decision == "original_unverified"
    )
    error_cnt = sum(1 for r in unit_decision_records if r.final_decision == "original_error")
    no_speech_cnt = sum(
        1 for r in unit_decision_records if r.final_decision == "original_no_speech"
    )
    finish_app_cnt = sum(
        1 for r in unit_decision_records if r.finish_preset_applied in ("gentle", "minimal")
    )
    finish_byp_cnt = sum(
        1 for r in unit_decision_records if r.finish_preset_applied == "bypass" and r.is_speech
    )

    review_timecodes: list[ReviewTimecode] = []
    for r in unit_decision_records:
        if r.guard_a_verdict in (GuardVerdict.REVERT, GuardVerdict.UNVERIFIED, GuardVerdict.ERROR):
            review_timecodes.append(
                ReviewTimecode(
                    unit_id=r.unit_id,
                    start_time_s=r.start_time_s,
                    end_time_s=r.end_time_s,
                    channel=r.channel,
                    verdict=r.guard_a_verdict,
                    reason=r.decision_reason,
                )
            )

    out_sha256 = hash_file(tmp_out)

    report = VoiceCleanReport(
        schema_version=1,
        job_id=workspace.job_id,
        config_hash=workspace.config_hash,
        input=MediaStats(
            path=str(in_path),
            sha256=media.sha256,
            sample_rate=media.sample_rate,
            channels=media.channels,
            samples=media.samples,
            duration_s=media.duration_s,
            integrated_lufs=initial_loudness.integrated_lufs,
            true_peak_dbtp=initial_loudness.true_peak_dbtp,
        ),
        output=MediaStats(
            path=str(out_path),
            sha256=out_sha256,
            sample_rate=mastered_buffer.sample_rate,
            channels=mastered_buffer.channels,
            samples=mastered_buffer.samples,
            duration_s=mastered_buffer.duration_s,
            integrated_lufs=final_loudness.integrated_lufs,
            true_peak_dbtp=final_loudness.true_peak_dbtp,
        ),
        core=CoreMetadata(
            id=config.enhancement.core_id,
            algorithm=str(core_lock["algorithm"]),
            params_hash=str(core_lock["params_hash"]),
            phase_coherent=config.enhancement.phase_coherent,
        ),
        guard=GuardMetadata(
            id=active_guard_cfg.guard_id,
            probe_hash=probe.probe_hash,
            calibration_id=str(calib_data["calibration_id"]),
        ),
        environment=EnvironmentMetadata(
            platform=platform.platform(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            soundfile_version=soundfile.__version__,
            cpu_model=platform.processor(),
        ),
        summary=UnitSummary(
            units_total=len(all_units),
            enhanced=enhanced_cnt,
            reverted=reverted_cnt,
            unverified=unverified_cnt,
            error_passthrough=error_cnt,
            continuity_reverted=continuity_cnt,
            no_speech=no_speech_cnt,
            finish_applied=finish_app_cnt,
            finish_bypassed=finish_byp_cnt,
        ),
        review_timecodes=review_timecodes,
        units=unit_decision_records,
    )

    json_str = serialize_json_report(report)
    txt_str = generate_human_summary(report)

    # 14. Atomic publication, then workspace cleanup
    dest_audio, dest_json, dest_txt = workspace.publish_atomically(
        temp_audio_path=tmp_out,
        destination_audio_path=out_path,
        json_report_str=json_str,
        txt_summary_str=txt_str,
        overwrite=overwrite,
    )

    workspace.journal.append(
        JournalEvent.OUTPUT_PUBLISHED,
        {"audio": str(dest_audio), "json": str(dest_json), "txt": str(dest_txt)},
    )
    workspace.journal.append(JournalEvent.JOB_COMPLETE)
    workspace.cleanup()

    logger.info(f"Pipeline finished successfully! Published master to {dest_audio}")
    return report
