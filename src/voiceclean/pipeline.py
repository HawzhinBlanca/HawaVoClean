"""Master processing pipeline orchestrating decoding, segmentation, isolated worker enhancement, guards, finishing, loudness, and atomic publication."""

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from voiceclean.enhancement.production import ProductionEnhancerCore
from voiceclean.enhancement.validate import validate_enhancer_output
from voiceclean.enhancement.worker import IsolatedEnhancementWorker
from voiceclean.finishing.limiter import apply_lookahead_limiter
from voiceclean.finishing.loudness import compute_static_master_gain, measure_loudness_and_peaks
from voiceclean.finishing.safe_finish import safe_finish_speech_unit
from voiceclean.guard.calibration import apply_calibrated_thresholds, load_calibration_artifact
from voiceclean.guard.hawzhin_ctc import HawzhinSoraniASR
from voiceclean.guard.protocol import SoraniASR
from voiceclean.guard.verdict import GuardVerdict
from voiceclean.hashing import hash_bytes, hash_file
from voiceclean.job import JobWorkspace
from voiceclean.journal import JournalEvent
from voiceclean.logging import get_logger
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


def run_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    config: VoiceCleanConfig | None = None,
    config_path: Path | str | None = None,
    profile: str = "production",
    overwrite: bool = False,
    asr_override: SoraniASR | None = None,
) -> VoiceCleanReport:
    """Execute complete end-to-end Hawzhin VoiceClean pipeline."""
    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()

    logger.info(f"Starting VoiceClean pipeline on {in_path} -> {out_path} [profile={profile}]")

    # 1. Configuration & Calibration Preflight
    is_prod = profile == "production"
    if config is None:
        cfg_file = config_path or (
            "configs/production.toml" if is_prod else "configs/development.toml"
        )
        config = load_config(cfg_file, is_production=is_prod)

    # Load and lock calibration thresholds
    calib_data = load_calibration_artifact(config.guard.calibration_file)
    active_guard_cfg = apply_calibrated_thresholds(config.guard, calib_data)

    # Load production core lock for provenance metadata
    import tomllib
    lock_path = Path("models/production-core.lock.toml")
    if lock_path.exists():
        with open(lock_path, "rb") as f:
            core_lock_data = tomllib.load(f)
    else:
        core_lock_data = {}

    # 2. Probe Media
    probe = probe_audio(in_path, max_sample_rate=config.input.max_sample_rate)
    logger.info(
        f"Probed media: {probe.sample_rate}Hz, {probe.channels}ch, {probe.samples:,} samples ({probe.duration_s:.2f}s)"
    )

    # 3. Initialize Workspace & Journal
    workspace = JobWorkspace(
        input_path=in_path,
        input_sha256=probe.sha256,
        config=config,
        core_id=config.enhancement.core_id,
        guard_id=active_guard_cfg.guard_id,
        tool_version=__version__,
    )
    workspace.journal.append(
        JournalEvent.JOB_STARTED, {"input": str(in_path), "job_id": workspace.job_id}
    )

    # Check disk space (estimate 4 bytes * samples * channels * 3)
    workspace.check_disk_space(probe.samples * probe.channels * 12)
    workspace.journal.append(JournalEvent.PREFLIGHT_PASSED)

    # 4. Safe Decode & Channel Classification
    audio_buf = decode_audio(probe, timeout_s=config.runtime.worker_timeout_s)
    # Sync probe with exact decoded stream sample count if container estimate differed
    if probe.samples != audio_buf.samples:
        probe = AudioProbeResult(
            path=probe.path,
            format_name=probe.format_name,
            codec_name=probe.codec_name,
            sample_rate=probe.sample_rate,
            channels=probe.channels,
            duration_s=audio_buf.duration_s,
            samples=audio_buf.samples,
            bit_depth=probe.bit_depth,
            sha256=probe.sha256,
        )
    channel_mode = classify_channels(audio_buf, declared_mode=config.input.channel_mode)
    audio_buf.channel_mode = channel_mode
    logger.info(f"Channel classification: {channel_mode}")
    workspace.journal.append(JournalEvent.AUDIO_DECODED, {"channel_mode": str(channel_mode)})

    # Determine processing channels
    channels_to_process, duplicate_to_stereo = handle_channel_layout(audio_buf, channel_mode)

    # 5. Segmentation into SpeechUnits
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
        f"Generated {len(all_units)} speech units across {len(channels_to_process)} processing channel(s)."
    )
    workspace.journal.append(JournalEvent.SEGMENTATION_COMPLETE, {"units_count": len(all_units)})

    # 6. Instantiate Models (Worker & ASR Guard)
    if config.runtime.isolated_worker:
        worker: Any = IsolatedEnhancementWorker(
            core_id=config.enhancement.core_id,
            sample_rate=config.enhancement.model_sample_rate,
            timeout_s=config.runtime.worker_timeout_s,
        )
    else:
        worker = ProductionEnhancerCore(
            core_id=config.enhancement.core_id,
            sample_rate=config.enhancement.model_sample_rate,
            phase_coherent=config.enhancement.phase_coherent,
        )

    asr_engine: SoraniASR = asr_override or HawzhinSoraniASR(
        model_id=active_guard_cfg.asr_model_id,
        target_sr=16000,
    )

    # 7. Unit Processing Loop
    committed_units = workspace.journal.get_committed_units()
    unit_decisions: dict[int, UnitPolicyDecision] = {}
    unit_decision_records: list[UnitDecisionRecord] = []
    orig_core_waveforms: dict[int, np.ndarray] = {}

    try:
        for u in all_units:
            ch_wave = channels_to_process[u.channel_id]
            core_orig = ch_wave[u.start_sample : u.end_sample]
            orig_core_waveforms[u.unit_id] = core_orig

            t_unit_start = time.perf_counter()

            # Check if unit already committed in journal
            if u.unit_id in committed_units:
                cached_wave = workspace.load_unit_result(u.unit_id, u.channel_id)
                if cached_wave is not None:
                    unit_decisions[u.unit_id] = UnitPolicyDecision(
                        selected_waveform=cached_wave,
                        is_enhanced=True,
                        chosen_strength=1.0,
                        guard_verdict=GuardVerdict.PASS,
                        decision_reason="Loaded from committed resume cache.",
                    )
                    unit_decision_records.append(
                        UnitDecisionRecord(
                            unit_id=u.unit_id,
                            channel=u.channel_id,
                            start_sample=u.start_sample,
                            end_sample=u.end_sample,
                            start_time_s=float(u.start_sample / audio_buf.sample_rate),
                            end_time_s=float(u.end_sample / audio_buf.sample_rate),
                            is_speech=u.is_speech,
                            input_sha256=u.input_sha256,
                            guard_a_verdict=GuardVerdict.PASS,
                            final_decision="enhanced",
                            decision_reason="Resumed from workspace cache.",
                        )
                    )
                    continue

            if not u.is_speech:
                # Non-speech unit: passthrough original audio
                dec = UnitPolicyDecision(
                    selected_waveform=core_orig.copy(),
                    is_enhanced=False,
                    chosen_strength=0.0,
                    guard_verdict=GuardVerdict.NO_SPEECH,
                    decision_reason="Non-speech unit passthrough.",
                )
                unit_decisions[u.unit_id] = dec
                workspace.save_unit_result(u.unit_id, u.channel_id, dec.selected_waveform)
                workspace.journal.append(
                    JournalEvent.UNIT_COMMITTED, {"unit_id": u.unit_id, "is_speech": False}
                )

                unit_decision_records.append(
                    UnitDecisionRecord(
                        unit_id=u.unit_id,
                        channel=u.channel_id,
                        start_sample=u.start_sample,
                        end_sample=u.end_sample,
                        start_time_s=float(u.start_sample / audio_buf.sample_rate),
                        end_time_s=float(u.end_sample / audio_buf.sample_rate),
                        is_speech=False,
                        input_sha256=u.input_sha256,
                        output_sha256=u.input_sha256,
                        guard_a_verdict=GuardVerdict.NO_SPEECH,
                        final_decision="original_no_speech",
                        decision_reason="Non-speech unit.",
                    )
                )
                continue

            # Speech Unit Processing: Extract context audio
            context_wave = ch_wave[u.context_start_sample : u.context_end_sample]

            # Neural Enhancement Inference
            enh_core: np.ndarray | None = None
            cand_sha256: str | None = None
            try:
                enh_res = worker.enhance(context_wave, audio_buf.sample_rate)
                # Trim context back to core length
                left_ctx = u.left_context_samples
                core_len = u.core_length_samples
                enh_trimmed = enh_res.waveform[left_ctx : left_ctx + core_len]

                # Immediate output validation
                valid, reason = validate_enhancer_output(core_orig, enh_trimmed, is_speech=True)
                if valid:
                    # Delay Alignment
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
                    enh_core = None
            except Exception as e:
                logger.warning(f"Enhancement worker failed for unit {u.unit_id}: {e}")
                enh_core = None

            # Policy Decision (Guard A)
            pol_dec, orig_asr = evaluate_unit_policy(
                orig_core_waveform=core_orig,
                enh_core_waveform=enh_core,
                sample_rate=audio_buf.sample_rate,
                is_speech=True,
                asr_engine=asr_engine,
                guard_config=active_guard_cfg,
                policy_config=config.policy,
                phase_coherent=config.enhancement.phase_coherent,
            )

            final_wave = pol_dec.selected_waveform
            finish_preset = "bypass"
            finish_actions: list[str] = []
            guard_b_verdict: GuardVerdict | None = None
            guard_b_scores: dict[str, Any] = {}

            # If accepted by Guard A, run Safe Finishing (Guard B)
            if pol_dec.is_enhanced and config.finishing.enabled:
                finish_res, _ = safe_finish_speech_unit(
                    pre_finish_waveform=pol_dec.selected_waveform,
                    sample_rate=audio_buf.sample_rate,
                    is_speech=True,
                    asr_engine=asr_engine,
                    finishing_config=config.finishing,
                    guard_config=active_guard_cfg,
                )
                final_wave = finish_res.finished_waveform
                finish_preset = finish_res.preset_applied
                finish_actions = finish_res.actions_taken
                guard_b_verdict = finish_res.guard_b_verdict
                guard_b_scores = finish_res.guard_b_scores

            # Commit unit
            unit_decisions[u.unit_id] = UnitPolicyDecision(
                selected_waveform=final_wave,
                is_enhanced=pol_dec.is_enhanced,
                chosen_strength=pol_dec.chosen_strength,
                guard_verdict=pol_dec.guard_verdict,
                guard_scores=pol_dec.guard_scores,
                decision_reason=pol_dec.decision_reason,
            )

            workspace.save_unit_result(u.unit_id, u.channel_id, final_wave)
            workspace.journal.append(
                JournalEvent.UNIT_COMMITTED, {"unit_id": u.unit_id, "enhanced": pol_dec.is_enhanced}
            )

            t_elapsed_unit = (time.perf_counter() - t_unit_start) * 1000.0

            final_cat = (
                "enhanced"
                if pol_dec.is_enhanced
                else (
                    "original_unverified"
                    if pol_dec.guard_verdict == GuardVerdict.UNVERIFIED
                    else (
                        "original_error"
                        if pol_dec.guard_verdict == GuardVerdict.ERROR
                        else "original_reverted"
                    )
                )
            )

            unit_decision_records.append(
                UnitDecisionRecord(
                    unit_id=u.unit_id,
                    channel=u.channel_id,
                    start_sample=u.start_sample,
                    end_sample=u.end_sample,
                    start_time_s=float(u.start_sample / audio_buf.sample_rate),
                    end_time_s=float(u.end_sample / audio_buf.sample_rate),
                    is_speech=True,
                    input_sha256=u.input_sha256,
                    candidate_sha256=cand_sha256,
                    output_sha256="",
                    guard_a_verdict=pol_dec.guard_verdict,
                    guard_a_scores=pol_dec.guard_scores,
                    guard_b_verdict=guard_b_verdict,
                    guard_b_scores=guard_b_scores,
                    chosen_strength=pol_dec.chosen_strength,
                    finish_preset_applied=finish_preset,
                    finish_actions=finish_actions,
                    final_decision=final_cat,
                    decision_reason=pol_dec.decision_reason,
                    runtime_ms=t_elapsed_unit,
                )
            )

    finally:
        if hasattr(worker, "close"):
            worker.close()

    # 8. Enforce Source Continuity
    if config.policy.enforce_continuity:
        decisions_list = [unit_decisions[u.unit_id] for u in all_units]
        orig_waves_list = [orig_core_waveforms[u.unit_id] for u in all_units]
        adjusted = enforce_source_continuity(all_units, decisions_list, orig_waves_list)
        for u, dec in zip(all_units, adjusted):
            unit_decisions[u.unit_id] = dec

    # 9. Assembly
    assembled_channels: list[np.ndarray] = []
    for ch_idx in range(len(channels_to_process)):
        ch_units = [u for u in all_units if u.channel_id == ch_idx]
        ch_dec_waves = [unit_decisions[uid].selected_waveform for uid in (u.unit_id for u in ch_units)]
        ch_timeline = assemble_channel_timeline(
            units=ch_units,
            unit_waveforms=ch_dec_waves,
            total_samples=probe.samples,
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

    # Validate postconditions
    validate_assembled_timeline(
        assembled_buffer=assembled_buffer,
        expected_channels=probe.channels,
        expected_samples=probe.samples,
        expected_sample_rate=probe.sample_rate,
        units=all_units,
    )
    workspace.journal.append(JournalEvent.ASSEMBLY_COMPLETE)

    # 10. Global Loudness Normalization & True-Peak Limiting
    initial_loudness = measure_loudness_and_peaks(
        assembled_buffer.data, assembled_buffer.sample_rate
    )
    target_lufs = (
        config.loudness.target_lufs_stereo
        if probe.channels > 1
        else config.loudness.target_lufs_mono
    )

    static_gain_db = compute_static_master_gain(
        measured_lufs=initial_loudness.integrated_lufs,
        target_lufs=target_lufs,
        current_true_peak_dbtp=initial_loudness.true_peak_dbtp,
        true_peak_ceiling_dbtp=config.loudness.true_peak_ceiling_dbtp,
        max_limiter_reduction_db=config.loudness.max_limiter_reduction_db,
    )

    # Apply static gain
    gain_linear = 10.0 ** (static_gain_db / 20.0)
    gained_data = assembled_buffer.data * gain_linear

    # Apply lookahead true-peak limiter
    limited_res = apply_lookahead_limiter(
        waveform=gained_data,
        sample_rate=assembled_buffer.sample_rate,
        ceiling_dbtp=config.loudness.true_peak_ceiling_dbtp,
    )
    mastered_buffer = AudioBuffer(
        data=limited_res.limited_waveform,
        sample_rate=assembled_buffer.sample_rate,
        channel_mode=channel_mode,
    )

    # Final measurement
    final_loudness = measure_loudness_and_peaks(mastered_buffer.data, mastered_buffer.sample_rate)
    workspace.journal.append(
        JournalEvent.FINAL_VALIDATION_PASSED,
        {"lufs": final_loudness.integrated_lufs, "dbtp": final_loudness.true_peak_dbtp},
    )

    # 11. Write Master WAV to Workspace Temp File
    tmp_out = workspace.root / "candidate-output.wav.tmp"
    encode_audio(
        buffer=mastered_buffer,
        output_path=tmp_out,
        output_bit_depth=config.input.output_bit_depth,
        dither=config.loudness.dither,
        seed_context=workspace.job_id,
    )

    # 12. Build Audit Report & Summaries
    all_decisions = unit_decisions.values()
    enhanced_cnt = sum(1 for d in all_decisions if d.is_enhanced)
    reverted_cnt = sum(1 for d in all_decisions if d.guard_verdict == GuardVerdict.REVERT)
    unverified_cnt = sum(1 for d in all_decisions if d.guard_verdict == GuardVerdict.UNVERIFIED)
    error_cnt = sum(1 for d in all_decisions if d.guard_verdict == GuardVerdict.ERROR)
    no_speech_cnt = sum(1 for d in all_decisions if d.guard_verdict == GuardVerdict.NO_SPEECH)
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
            sha256=probe.sha256,
            sample_rate=probe.sample_rate,
            channels=probe.channels,
            samples=probe.samples,
            duration_s=probe.duration_s,
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
            commit=str(core_lock_data.get("commit", "unknown")),
            weight_sha256={str(k): str(v) for k, v in core_lock_data.get("weight_sha256", {}).items()},
            phase_coherent=config.enhancement.phase_coherent,
        ),
        guard=GuardMetadata(
            id=active_guard_cfg.guard_id,
            model_sha256="hawzhin_guard_sha256",
            calibration_id=str(calib_data.get("calibration_id", "calib_001")),
        ),
        environment=EnvironmentMetadata(
            platform=platform.platform(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            cpu_model=platform.processor(),
        ),
        summary=UnitSummary(
            units_total=len(all_units),
            enhanced=enhanced_cnt,
            reverted=reverted_cnt,
            unverified=unverified_cnt,
            error_passthrough=error_cnt,
            no_speech=no_speech_cnt,
            finish_applied=finish_app_cnt,
            finish_bypassed=finish_byp_cnt,
        ),
        review_timecodes=review_timecodes,
        units=unit_decision_records,
    )

    json_str = serialize_json_report(report)
    txt_str = generate_human_summary(report)

    # 13. Atomic Publication
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

    logger.info(f"Pipeline finished successfully! Published master to {dest_audio}")
    return report
