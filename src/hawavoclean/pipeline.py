"""Master processing pipeline: decode, segment, enhance, guard, finish, master, publish.

Every run recomputes every unit — there is no resume cache, so the audit
report always describes the run that produced it. The scratch workspace is
removed on success and survives only a genuine crash, for forensics.
"""

import hashlib
import os
import platform
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import soundfile

from hawavoclean import __version__
from hawavoclean.alignment.delay import estimate_gcc_phat_delay
from hawavoclean.assembly.stitch import assemble_channel_timeline
from hawavoclean.assembly.validate import validate_assembled_timeline
from hawavoclean.audio.channels import classify_channels, handle_channel_layout
from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.encode import encode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.audio.types import AudioBuffer, AudioProbeResult
from hawavoclean.config import HawaVoCleanConfig, load_config
from hawavoclean.enhancement.factory import resolve_core
from hawavoclean.enhancement.validate import validate_enhancer_output
from hawavoclean.enhancement.worker import (
    EnhancementRun,
    EnhancementWorkerPool,
    IsolatedEnhancementWorker,
    UnitEnhancement,
    WorkerSpec,
    acquire_pool,
    configured_worker_hint,
    release_pool,
)
from hawavoclean.errors import (
    CalibrationError,
    ConfigError,
    HawaVoCleanError,
    InvalidUserInputError,
    PreflightError,
    PublicationError,
    WorkerCrashError,
)
from hawavoclean.finishing.detect import (
    SpeechTiltReport,
    aggregate_speech_tilt,
    measure_speech_tilt,
)
from hawavoclean.finishing.limiter import apply_lookahead_limiter
from hawavoclean.finishing.loudness import compute_static_master_gain, measure_loudness_and_peaks
from hawavoclean.finishing.safe_finish import safe_finish_speech_unit
from hawavoclean.guard.calibration import apply_calibrated_thresholds, load_calibration_artifact
from hawavoclean.guard.protocol import SpectralProbe
from hawavoclean.guard.spectral_probe import SpectralSignatureProbe
from hawavoclean.guard.verdict import GuardVerdict
from hawavoclean.hashing import hash_bytes, hash_file, hash_json_canonical
from hawavoclean.job import JobWorkspace
from hawavoclean.journal import JournalEvent
from hawavoclean.logging import get_logger
from hawavoclean.paths import models_dir, profile_config_path, resolve_calibration_file
from hawavoclean.policy.continuity import (
    CONTINUITY_TAPER_ACTION,
    apply_continuity_taper,
    resolve_source_continuity,
)
from hawavoclean.policy.decision import UnitPolicyDecision, evaluate_unit_policy
from hawavoclean.progress import (
    PROGRESS_DECODE,
    PROGRESS_FINISH_END,
    PROGRESS_FINISH_START,
    PROGRESS_PREFLIGHT,
    PROGRESS_PUBLISH,
    PROGRESS_SEGMENT,
    ProgressCallback,
    ProgressEvent,
    emit_progress,
    unit_progress,
)
from hawavoclean.provenance import deterministic_settings, runtime_versions
from hawavoclean.publication import public_output_path, publication_exists, publication_paths
from hawavoclean.release import REPORT_SCHEMA_VERSION
from hawavoclean.report.schema import (
    CoreMetadata,
    EnvironmentMetadata,
    GuardMetadata,
    HawaVoCleanReport,
    MediaStats,
    ReviewTimecode,
    UnitDecisionRecord,
    UnitSummary,
    current_build_metadata,
    current_release_metadata,
)
from hawavoclean.report.summary import generate_human_summary
from hawavoclean.report.writer import serialize_json_report
from hawavoclean.restoration import (
    BandwidthDetector,
    ProfileValidationError,
    RestorationConfig,
    RestorationGuard,
    RestorationPolicyManager,
    RestorationReport,
    RestorationSegmentCounts,
    SegmentRestorationDecision,
    load_speaker_profile,
)
from hawavoclean.segmentation.types import SpeechUnit
from hawavoclean.segmentation.utterances import build_speech_units

logger = get_logger("pipeline")

FINAL_DECISION_BY_VERDICT: dict[GuardVerdict, str] = {
    GuardVerdict.UNVERIFIED: "original_unverified",
    GuardVerdict.ERROR: "original_error",
    GuardVerdict.REVERT: "original_reverted",
}


def _preflight_destination(in_path: Path, out_path: Path, overwrite: bool) -> None:
    """Refuse destinations that would destroy the source or cannot be written,
    BEFORE decoding a single sample.

    - output == input (or a report sidecar == input) would overwrite the
      source: the one thing this tool promises never to do.
    - an existing destination without --overwrite is refused here, not after
      minutes of processing.
    - an unwritable destination directory is refused here, cleanly.
    """
    sidecars = publication_paths(out_path).public
    for candidate in sidecars:
        if candidate == in_path:
            raise PublicationError(
                f"Refusing to write output over the input: {in_path}. "
                "Choose a different output path."
            )
    if not overwrite and publication_exists(out_path):
        raise PublicationError(
            f"Destination output file already exists and overwrite=False: {out_path} "
            "(pass --overwrite to replace it)"
        )
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        probe_file = out_path.parent / f".hawavoclean-writable-{os.getpid()}"
        probe_file.write_bytes(b"")
        probe_file.unlink()
    except OSError as e:
        raise PublicationError(
            f"Destination directory is not writable: {out_path.parent} ({e})"
        ) from e


def _load_core_lock(core_id: str) -> tuple[dict[str, Any], str]:
    """Load and verify the configured core's lockfile. Missing or mismatched
    provenance is a hard failure, never a silent degradation."""
    registration = resolve_core(core_id)
    import importlib.util

    missing = [m for m in registration.requires_modules if importlib.util.find_spec(m) is None]
    if missing:
        raise PreflightError(
            f"Core {core_id!r} needs optional dependencies that are not "
            f"installed ({', '.join(missing)}). Install them with: "
            "uv sync --extra studio"
        )
    lock_path = models_dir() / registration.lock_filename
    if not lock_path.exists():
        raise PreflightError(
            f"Core lockfile missing: {lock_path}. Refusing to run without "
            "verifiable core provenance."
        )
    raw_lock = lock_path.read_bytes()
    lock = tomllib.loads(raw_lock.decode("utf-8"))

    if lock.get("core_id") != core_id:
        raise PreflightError(
            f"Configured core_id {core_id!r} does not match lockfile core {lock.get('core_id')!r}"
        )
    actual_params_hash = registration.implementation_params_hash()
    if lock.get("params_hash") != actual_params_hash:
        raise PreflightError(
            "Core parameter drift: lockfile params_hash "
            f"{str(lock.get('params_hash'))[:16]}... does not match the "
            f"implemented core {actual_params_hash[:16]}..."
        )
    # The lock's own tables must reconstruct params_hash (weights digests are
    # part of the implementation payload when the core has weights).
    payload: dict[str, Any] = dict(lock.get("params", {}))
    weight_table = {str(k): str(v) for k, v in dict(lock.get("weight_sha256", {})).items()}
    if weight_table:
        payload["weights_sha256"] = weight_table
    if hash_json_canonical(payload) != actual_params_hash:
        raise PreflightError(
            "Core lockfile tables do not recompute to params_hash; "
            "the lockfile has been hand-edited."
        )
    # Weights on disk must match their locked digests.
    for rel, digest in weight_table.items():
        weight_path = models_dir() / rel
        if not weight_path.exists():
            raise PreflightError(f"Locked weights file missing: {weight_path}")
        if hash_file(weight_path) != digest:
            raise PreflightError(f"Weights digest mismatch for {rel}")
    return lock, hash_bytes(raw_lock)


def run_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    config: HawaVoCleanConfig | None = None,
    config_path: Path | str | None = None,
    profile: str = "production",
    overwrite: bool = False,
    probe_override: SpectralProbe | None = None,
    on_progress: ProgressCallback | None = None,
    mode: str = "natural",
    speaker_id: str | None = None,
    cutoff: str = "auto",
    cutoff_hz: float | None = None,
    profiles_dir: str | Path = "profiles",
) -> HawaVoCleanReport:
    """Execute the complete end-to-end HawaVoClean pipeline.

    ``on_progress`` (optional) receives a :class:`ProgressEvent` at every
    stage boundary; exceptions it raises are logged and ignored.
    """
    in_path = Path(input_path).resolve()
    out_path = public_output_path(output_path)

    if mode not in ("natural", "restore"):
        raise InvalidUserInputError(f"Unknown processing mode: '{mode}' (expected natural|restore)")
    if mode == "restore" and not speaker_id:
        raise InvalidUserInputError("Restore mode requires an explicit --speaker-id <ID>")
    if mode == "restore" and speaker_id:
        # Fail on a bad speaker id NOW, not at step 10.5. The profile used to
        # be looked up only when restoration ran, which is after decode,
        # segmentation and the enhancement of every unit -- so a typo in
        # --speaker-id cost the user the entire enhancement pass before
        # admitting the id was never going to resolve.
        try:
            load_speaker_profile(speaker_id, profiles_root=profiles_dir)
        except ProfileValidationError as exc:
            raise InvalidUserInputError(str(exc)) from exc
    if cutoff not in ("auto", "manual"):
        raise InvalidUserInputError(f"Unknown cutoff mode: '{cutoff}' (expected auto|manual)")
    if cutoff == "manual" and cutoff_hz is None:
        raise InvalidUserInputError("--cutoff manual requires an explicit --cutoff-hz <Hz>")
    # An explicit frequency *is* manual selection. Deriving the mode here keeps
    # the report's cutoff_mode truthful whichever way the caller spelled it,
    # instead of recording "auto" over an operator-asserted boundary.
    cutoff = "manual" if cutoff_hz is not None else "auto"

    logger.info(
        f"Starting HawaVoClean pipeline on {in_path} -> {out_path} [profile={profile}, mode={mode}]"
    )
    _preflight_destination(in_path, out_path, overwrite)

    # 1. Configuration, calibration, and core provenance preflight
    is_prod = profile == "production"
    if config is None:
        cfg_file = Path(config_path) if config_path is not None else profile_config_path(profile)
        config = load_config(cfg_file, is_production=is_prod)

    calib_path = resolve_calibration_file(config.guard.calibration_file)
    calib_data = load_calibration_artifact(calib_path)
    if hash_json_canonical(calib_data["thresholds"]) != calib_data.get("calibration_id"):
        raise CalibrationError(
            f"Guard calibration artifact {calib_path} has been edited: calibration_id "
            "does not recompute from its thresholds. Refusing to run with a tampered guard."
        )
    active_guard_cfg = apply_calibrated_thresholds(config.guard, calib_data)

    core_lock, core_lock_sha256 = _load_core_lock(config.enhancement.core_id)
    if bool(core_lock.get("phase_coherent", True)) != config.enhancement.phase_coherent:
        raise ConfigError(
            f"enhancement.phase_coherent = {config.enhancement.phase_coherent} but core "
            f"{config.enhancement.core_id!r} is "
            f"{'phase-coherent' if core_lock.get('phase_coherent', True) else 'NOT phase-coherent'}; "
            "the report would misstate the core and the policy would blend residuals incorrectly."
        )
    expected_rates = [int(r) for r in core_lock.get("expected_sample_rates", [])]
    if expected_rates and config.enhancement.model_sample_rate not in expected_rates:
        raise ConfigError(
            f"enhancement.model_sample_rate = {config.enhancement.model_sample_rate} but core "
            f"{config.enhancement.core_id!r} runs at {expected_rates}"
        )

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
        # Restore-only inputs belong in the job identity: without them a
        # natural master and a reconstruction of the same file shared an id,
        # as did two reconstructions from two different speaker profiles.
        restore_context=(
            f"{mode}:{speaker_id}:{cutoff}:{cutoff_hz}" if mode == "restore" else None
        ),
    )
    workspace.journal.append(
        JournalEvent.JOB_STARTED, {"input": str(in_path), "job_id": workspace.job_id}
    )
    workspace.check_disk_space(media.samples * media.channels * 12, destination=out_path.parent)
    workspace.journal.append(JournalEvent.PREFLIGHT_PASSED)
    emit_progress(
        on_progress,
        ProgressEvent("preflight", PROGRESS_PREFLIGHT, "Preflight checks passed"),
    )

    try:
        return _run_after_preflight(
            config,
            active_guard_cfg,
            calib_data,
            hash_file(calib_path),
            core_lock,
            core_lock_sha256,
            media,
            workspace,
            in_path,
            out_path,
            overwrite,
            probe_override,
            on_progress,
            mode=mode,
            speaker_id=speaker_id,
            cutoff=cutoff,
            cutoff_hz=cutoff_hz,
            profiles_dir=profiles_dir,
        )
    except HawaVoCleanError:
        # Known, reported failures (bad input, refused destination, ...) must
        # not leak a scratch workspace; a genuine crash keeps it for forensics.
        workspace.cleanup()
        raise


#: Restoration segment length. Long enough that Guard R's F0, harmonic and
#: linguistic layers see real speech context; short enough that one bad moment
#: costs only its own segment and that per-layer analysis stays bounded however
#: long the recording is.
_RESTORE_SEGMENT_S = 10.0
#: Overlap cross-faded between neighbouring segments, so two segments that
#: reach different verdicts meet without a click.
_RESTORE_SEGMENT_OVERLAP_S = 0.25
_RESTORE_SEGMENT_HOP_S = _RESTORE_SEGMENT_S - _RESTORE_SEGMENT_OVERLAP_S


def _restore_in_segments(
    *,
    natural_audio: "np.ndarray[Any, np.dtype[np.float32]]",
    sample_rate: int,
    bandwidth_est: Any,
    speaker_profile: Any,
    policy: RestorationPolicyManager,
    base_seed: int,
) -> tuple["np.ndarray[Any, np.dtype[np.float32]]", list[SegmentRestorationDecision]]:
    """Run the restoration policy over overlapping segments and stitch them.

    The bandwidth estimate is deliberately shared: a band limit is a property of
    the recording, not of a moment inside it, so re-detecting per segment would
    let a quiet passage invent a different cutoff and hand the model licence to
    overwrite content the rest of the file proves is real.

    Each segment is seeded from ``(base_seed, index)``, so the result depends on
    the job and the position in the file and not on how the work was divided.
    """
    n_samples = natural_audio.shape[-1]
    seg_len = max(int(sample_rate * _RESTORE_SEGMENT_S), 1)
    overlap = min(int(sample_rate * _RESTORE_SEGMENT_OVERLAP_S), seg_len // 2)
    hop = max(seg_len - overlap, 1)

    starts: list[int] = []
    pos = 0
    while True:
        starts.append(pos)
        if pos + seg_len >= n_samples:
            break
        pos += hop

    out = np.zeros_like(natural_audio)
    records: list[SegmentRestorationDecision] = []
    covered = 0
    for index, start in enumerate(starts):
        stop = min(start + seg_len, n_samples)
        segment = natural_audio[..., start:stop]
        restored, decision = policy.process_segment(
            natural_audio=segment,
            sample_rate=sample_rate,
            bandwidth_est=bandwidth_est,
            speaker_profile=speaker_profile,
            segment_seed=(base_seed + 7919 * index) % (2**63 - 1),
        )
        records.append(decision)
        restored = np.asarray(restored, dtype=np.float32)

        fade = max(0, min(covered - start, stop - start)) if index > 0 else 0
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            out[..., start : start + fade] *= 1.0 - ramp
            out[..., start : start + fade] += restored[..., :fade] * ramp
            out[..., start + fade : stop] = restored[..., fade:]
        else:
            out[..., start:stop] = restored
        covered = stop

    return out, records


def _run_after_preflight(
    config: HawaVoCleanConfig,
    active_guard_cfg: Any,
    calib_data: dict[str, Any],
    calibration_sha256: str,
    core_lock: dict[str, Any],
    core_lock_sha256: str,
    media: AudioProbeResult,
    workspace: JobWorkspace,
    in_path: Path,
    out_path: Path,
    overwrite: bool,
    probe_override: SpectralProbe | None,
    on_progress: ProgressCallback | None = None,
    mode: str = "natural",
    speaker_id: str | None = None,
    cutoff: str = "auto",
    cutoff_hz: float | None = None,
    profiles_dir: str | Path = "profiles",
) -> HawaVoCleanReport:
    """Everything after preflight; split out so the caller can scope cleanup."""
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
    # The report's ``input`` block describes the file the user handed us, so
    # its loudness has to be measured here, on the decoded source. It used to
    # be taken from the pre-master buffer, which is the audio AFTER three
    # enhancement cores and reassembly -- so a reader comparing input against
    # output LUFS to see what mastering did was reading enhancement into the
    # baseline, and every other field beside it (path, sha256, sample_rate)
    # genuinely described the source.
    source_loudness = measure_loudness_and_peaks(audio_buf.data, audio_buf.sample_rate)
    channel_mode = classify_channels(audio_buf, declared_mode=config.input.channel_mode)
    audio_buf.channel_mode = channel_mode
    logger.info(f"Channel classification: {channel_mode}")
    workspace.journal.append(JournalEvent.AUDIO_DECODED, {"channel_mode": str(channel_mode)})
    emit_progress(
        on_progress,
        ProgressEvent(
            "decode",
            PROGRESS_DECODE,
            f"Decoded {audio_buf.duration_s:.1f} s @ {audio_buf.sample_rate / 1000.0:g} kHz, "
            f"{audio_buf.channels} ch",
        ),
    )

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
    emit_progress(
        on_progress,
        ProgressEvent(
            "segment",
            PROGRESS_SEGMENT,
            f"{len(all_units)} unit{'' if len(all_units) == 1 else 's'}",
        ),
    )
    units_total = len(all_units)

    # Dispatch every speech unit's context window at once. These are numpy
    # views into the decoded channel, so building the list copies nothing; the
    # copy happens per request, in the worker's own send, exactly as before.
    speech_slot: dict[int, int] = {}
    context_items: list[tuple[np.ndarray[Any, np.dtype[np.float32]], int]] = []
    for idx, u in enumerate(all_units):
        if not u.is_speech:
            continue
        speech_slot[idx] = len(context_items)
        ctx = channels_to_process[u.channel_id][u.context_start_sample : u.context_end_sample]
        context_items.append((ctx, audio_buf.sample_rate))

    probe: SpectralProbe = probe_override or SpectralSignatureProbe(
        probe_id=active_guard_cfg.probe_id,
        target_sr=16000,
    )

    # 6. Models: isolated worker pool (or in-process core) and the fidelity probe
    #
    # Speech units are independent by construction, so the pool enhances
    # several at once. Nothing downstream can tell: candidates come back
    # indexed by unit, the strength ladder is a local blend of the candidate
    # (policy/strength.py) rather than another core call, and both shipped
    # cores are stateless across calls — verified by running the same unit
    # first, second and alone and hashing the output (tests/unit/
    # test_enhancement_pool.py::test_core_is_stateless_across_calls).
    core_registration = resolve_core(config.enhancement.core_id)
    pool: EnhancementWorkerPool | None = None
    inline_enhancer: Any = None
    if config.runtime.isolated_worker:
        pool = acquire_pool(
            WorkerSpec(
                core_id=config.enhancement.core_id,
                sample_rate=config.enhancement.model_sample_rate,
                timeout_s=config.runtime.worker_timeout_s,
                enhancer_class=core_registration.enhancer_class,
                phase_coherent=config.enhancement.phase_coherent,
            ),
            max_size=configured_worker_hint(config.runtime.num_threads),
            prewarm=len(context_items),
            worker_factory=IsolatedEnhancementWorker,
        )
    else:
        inline_enhancer = core_registration.enhancer_class(
            core_id=config.enhancement.core_id,
            sample_rate=config.enhancement.model_sample_rate,
            phase_coherent=config.enhancement.phase_coherent,
        )

    # 7. Guard-A decisions for every unit (finishing comes after continuity)
    decisions: list[UnitPolicyDecision] = []
    cand_hashes: list[str | None] = []
    unit_runtimes: list[float] = []
    orig_core_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    enh_run: EnhancementRun | None = None
    try:
        if pool is not None:
            enh_run = pool.begin(context_items)
    except BaseException:
        release_pool(pool)
        raise

    def enhancement_for(slot: int) -> UnitEnhancement:
        """Unit ``slot``'s candidate: from the pool, or computed here when the
        run is configured for an in-process core."""
        if enh_run is not None:
            return enh_run.result(slot)
        wave, rate = context_items[slot]
        t_enh = time.perf_counter()
        try:
            res = inline_enhancer.enhance(wave, rate)
            return UnitEnhancement(res, None, (time.perf_counter() - t_enh) * 1000.0)
        except Exception as e:
            return UnitEnhancement(None, e, (time.perf_counter() - t_enh) * 1000.0)

    try:
        for idx, u in enumerate(all_units):
            unit_no = idx + 1
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
                emit_progress(
                    on_progress,
                    ProgressEvent(
                        "guard",
                        unit_progress(unit_no, units_total, done=True),
                        f"Unit {unit_no}/{units_total}: NO_SPEECH",
                        unit_index=unit_no,
                        unit_total=units_total,
                    ),
                )
                continue

            emit_progress(
                on_progress,
                ProgressEvent(
                    "enhance",
                    unit_progress(unit_no, units_total, done=False),
                    f"Enhancing unit {unit_no}/{units_total}",
                    unit_index=unit_no,
                    unit_total=units_total,
                ),
            )
            # Blocks only if this unit is not enhanced yet; the pool carries
            # on with the later units while the guard below runs.
            enh = enhancement_for(speech_slot[idx])
            t_guard_start = time.perf_counter()

            enh_core: np.ndarray[Any, np.dtype[np.float32]] | None = None
            cand_sha256: str | None = None
            try:
                if enh.error is not None or enh.result is None:
                    raise enh.error or WorkerCrashError("no enhancement result")
                enh_res = enh.result
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
            # Per-unit WORK, not per-unit wall clock: under a pool the two
            # stop being the same number, and the report is about the unit.
            unit_runtimes.append(enh.elapsed_ms + (time.perf_counter() - t_guard_start) * 1000.0)
            verdict_label = "ENHANCED" if pol_dec.is_enhanced else pol_dec.guard_verdict.name
            emit_progress(
                on_progress,
                ProgressEvent(
                    "guard",
                    unit_progress(unit_no, units_total, done=True),
                    f"Unit {unit_no}/{units_total}: {verdict_label}",
                    unit_index=unit_no,
                    unit_total=units_total,
                ),
            )

        # 8. Source continuity — BEFORE records are built and units finished,
        # so a continuity revert is what gets finished, recorded, and stitched.
        # The fades it plans are applied AFTER finishing instead: the seam a
        # listener hears is between the *finished* enhanced audio and the
        # original, so fading any earlier would leave the finishing EQ's own
        # step sitting at the joint.
        continuity_reverted_ids: set[int] = set()
        taper_in = [0] * len(all_units)
        taper_out = [0] * len(all_units)
        if config.policy.enforce_continuity:
            resolution = resolve_source_continuity(
                all_units, decisions, orig_core_waveforms, audio_buf.sample_rate
            )
            decisions = resolution.decisions
            continuity_reverted_ids = resolution.reverted_ids
            taper_in = resolution.fade_in_samples
            taper_out = resolution.fade_out_samples

        # 9. Finishing (Guard B) on surviving enhanced units, then records
        emit_progress(
            on_progress,
            ProgressEvent("finish", PROGRESS_FINISH_START, "Finishing: EQ/limiter/loudness"),
        )
        unit_decision_records: list[UnitDecisionRecord] = []
        final_waveforms: list[np.ndarray[Any, np.dtype[np.float32]]] = []

        # Tonal balance is a property of the RECORDING, not of a 20 s block of
        # it. Measure every unit that will actually be finished, combine by
        # median, and hand the one answer to all of them — otherwise adjacent
        # units get different EQ and the tone pumps at every boundary (measured
        # at 2.8 dB of 3-6 kHz between two units of one file). Nothing is held
        # here but three floats per unit; the audio is not copied.
        file_tilt: SpeechTiltReport | None = None
        if config.finishing.enabled and config.finishing.tonal_restoration:
            per_unit_tilts = [
                measure_speech_tilt(decisions[i].selected_waveform, audio_buf.sample_rate)
                for i, unit in enumerate(all_units)
                if unit.is_speech and decisions[i].is_enhanced
            ]
            if per_unit_tilts:
                file_tilt = aggregate_speech_tilt(per_unit_tilts)

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
                    tilt=file_tilt,
                )
                final_wave = finish_res.finished_waveform
                finish_preset = finish_res.preset_applied
                finish_actions = finish_res.actions_taken
                guard_b_verdict = finish_res.guard_b_verdict
                guard_b_scores = finish_res.guard_b_scores

            if taper_in[idx] > 0 or taper_out[idx] > 0:
                final_wave = apply_continuity_taper(
                    final_wave, orig_core_waveforms[idx], taper_in[idx], taper_out[idx]
                )
                finish_actions = [
                    *finish_actions,
                    f"{CONTINUITY_TAPER_ACTION}(in={taper_in[idx]},out={taper_out[idx]})",
                ]

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
        if enh_run is not None:
            enh_run.join()
        # A cached pool (batch) stays warm; anything else is stopped here.
        release_pool(pool)
        if inline_enhancer is not None and hasattr(inline_enhancer, "close"):
            inline_enhancer.close()

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

    # 10.5. Spectral Restoration Subsystem (HawaRestore-KD)
    restoration_report: dict[str, Any] | None = None
    if mode == "restore":
        # Imported here, not at module scope: HawaRestoreKD is a torch model,
        # torch is an optional extra, and a natural-mode-only install is
        # supported. At module scope this made every `import hawavoclean.cli`
        # require torch, so the published wheel could not print its own
        # version without the restore extra.
        from hawavoclean.restoration.hawarestore_kd import HawaRestoreKD

        assert speaker_id is not None
        speaker_profile = load_speaker_profile(speaker_id, profiles_root=profiles_dir)

        # Target sample rate for restored output is 48000 Hz
        target_sr = 48000
        current_data = assembled_buffer.data
        if assembled_buffer.sample_rate != target_sr:
            gcd = np.gcd(assembled_buffer.sample_rate, target_sr)
            down = assembled_buffer.sample_rate // gcd
            up = target_sr // gcd
            current_data = scipy.signal.resample_poly(current_data, up, down, axis=-1).astype(
                np.float32
            )

        # Detect bandwidth and effective cutoff
        bw_detector = BandwidthDetector(sample_rate=target_sr)
        bw_est = bw_detector.detect(current_data, override_cutoff_hz=cutoff_hz)

        # Run HawaRestore-KD and Guard R
        restorer = HawaRestoreKD(sample_rate=target_sr)
        guard_r = RestorationGuard(sample_rate=target_sr)
        rest_cfg = RestorationConfig(mode="explicit", enabled=True)
        policy_mgr = RestorationPolicyManager(config=rest_cfg, restorer=restorer, guard=guard_r)

        seed_val = int(hashlib.sha256(workspace.job_id.encode("utf-8")).hexdigest()[:8], 16)

        # Restore in segments, not as one file-length block. Handing the whole
        # recording to Guard R made every decision all-or-nothing: measured, a
        # single defective 50 ms window — 0.25% of a 20 s file — failed the
        # guard and discarded restoration for 100% of it. It also made
        # ``segments`` and ``review_timecodes`` dead fields that could only ever
        # read 1-and-zeros and empty, and it let each guard layer's analysis
        # grow with file length instead of staying bounded.
        #
        # Segments overlap and are cross-faded, so neighbours that reach
        # different verdicts meet without a click at the seam.
        restored_data, segment_records = _restore_in_segments(
            natural_audio=current_data,
            sample_rate=target_sr,
            bandwidth_est=bw_est,
            speaker_profile=speaker_profile,
            policy=policy_mgr,
            base_seed=seed_val,
        )
        counts = Counter(record.action for record in segment_records)
        # The audit should carry the evidence of a refusal when one happened, so
        # prefer a segment the guard turned away over one it waved through.
        rest_dec = next(
            (r for r in segment_records if r.action in ("reverted", "error")),
            segment_records[0],
        )
        review_segments = [
            {
                "segment_index": index,
                "start_time_s": round(index * _RESTORE_SEGMENT_HOP_S, 3),
                "action": record.action,
                "verdict": record.guard_result.verdict if record.guard_result else "n/a",
                "reason": record.guard_result.reason if record.guard_result else "",
            }
            for index, record in enumerate(segment_records)
            if record.action in ("reverted", "error")
        ]

        assembled_buffer = AudioBuffer(
            data=restored_data,
            sample_rate=target_sr,
            channel_mode=channel_mode,
        )

        rest_rep = RestorationReport(
            mode="restore",
            speaker_id=speaker_id,
            profile_hash=speaker_profile.compute_hash(),
            natural_output_hash=hash_bytes(assembled_data.tobytes()),
            # ``cutoff_mode`` records whether the protected-band boundary was
            # measured or asserted by the operator — the reader cannot tell them
            # apart from the frequency alone.
            bandwidth={**bw_est.to_dict(), "cutoff_mode": cutoff},
            restorer={
                "name": "hawarestore-kd",
                "commit": "26dc21c44e11f9f19e823f02b0d4641dd5ea5af2",
                # Reported by the restorer itself, so the hash can only describe
                # weights that were actually loaded into the network.
                "weights_sha256": restorer.weights_sha256,
                "checkpoint_path": str(restorer.checkpoint_path),
                "device": restorer.device,
                "seed_policy": "deterministic_job_id",
                "solver": "midpoint",
                "steps": 4,
                "guidance_scale": 0.0,
            },
            segments=RestorationSegmentCounts(
                restored=counts.get("restored", 0),
                reduced=counts.get("reduced", 0),
                reverted=counts.get("reverted", 0),
                bypassed=counts.get("bypassed", 0),
                errors=counts.get("error", 0),
            ),
            guard_r=rest_dec.guard_result.to_dict() if rest_dec.guard_result else {},
            review_timecodes=review_segments,
        )
        restoration_report = rest_rep.to_dict()

    # 11. Loudness normalization and true-peak limiting
    premaster_loudness = measure_loudness_and_peaks(
        assembled_buffer.data, assembled_buffer.sample_rate
    )
    target_lufs = (
        config.loudness.target_lufs_stereo
        if media.channels > 1
        else config.loudness.target_lufs_mono
    )

    static_gain_db = compute_static_master_gain(
        measured_lufs=premaster_loudness.integrated_lufs,
        target_lufs=target_lufs,
        current_true_peak_dbtp=premaster_loudness.true_peak_dbtp,
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

    emit_progress(
        on_progress,
        ProgressEvent(
            "finish",
            PROGRESS_FINISH_END,
            f"Mastered to {final_loudness.integrated_lufs:.1f} LUFS, "
            f"{final_loudness.true_peak_dbtp:.1f} dBTP",
        ),
    )

    # 13. Audit report
    enhanced_cnt = sum(1 for r in unit_decision_records if r.final_decision == "enhanced")
    continuity_cnt = sum(
        1 for r in unit_decision_records if r.final_decision == "original_continuity"
    )
    crossfaded_cnt = sum(
        1
        for r in unit_decision_records
        if any(a.startswith(CONTINUITY_TAPER_ACTION) for a in r.finish_actions)
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

    report = HawaVoCleanReport(
        schema_version=REPORT_SCHEMA_VERSION,
        release=current_release_metadata(),
        build=current_build_metadata(),
        job_id=workspace.job_id,
        config_hash=workspace.config_hash,
        input=MediaStats(
            path=str(in_path),
            sha256=media.sha256,
            sample_rate=media.sample_rate,
            channels=media.channels,
            samples=media.samples,
            duration_s=media.duration_s,
            integrated_lufs=source_loudness.integrated_lufs,
            true_peak_dbtp=source_loudness.true_peak_dbtp,
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
            lock_sha256=core_lock_sha256,
            weight_sha256={
                str(key): str(value)
                for key, value in dict(core_lock.get("weight_sha256", {})).items()
            },
        ),
        guard=GuardMetadata(
            id=active_guard_cfg.guard_id,
            probe_hash=probe.probe_hash,
            calibration_id=str(calib_data["calibration_id"]),
            calibration_sha256=calibration_sha256,
        ),
        environment=EnvironmentMetadata(
            platform=platform.platform(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            soundfile_version=soundfile.__version__,
            cpu_model=platform.processor(),
            runtime_versions=runtime_versions(),
            deterministic_settings=deterministic_settings(config),
        ),
        summary=UnitSummary(
            units_total=len(all_units),
            enhanced=enhanced_cnt,
            reverted=reverted_cnt,
            unverified=unverified_cnt,
            error_passthrough=error_cnt,
            continuity_reverted=continuity_cnt,
            continuity_crossfaded=crossfaded_cnt,
            no_speech=no_speech_cnt,
            finish_applied=finish_app_cnt,
            finish_bypassed=finish_byp_cnt,
        ),
        review_timecodes=review_timecodes,
        units=unit_decision_records,
        restoration=restoration_report,
    )

    json_str = serialize_json_report(report)
    txt_str = generate_human_summary(report)

    # 14. Atomic publication, then workspace cleanup
    emit_progress(on_progress, ProgressEvent("publish", PROGRESS_PUBLISH, "Publishing master"))
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
