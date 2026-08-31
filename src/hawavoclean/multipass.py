"""Multi-pass enhancement orchestration: run the ordinary pipeline N times,
feeding each pass the previous pass's published output.

Why intermediate passes keep FULL finishing, mastering included
---------------------------------------------------------------
This is the measured recipe, not an accident. On the muffled, noisy teat1vo
lab source (``test_output/teat1vo-lab/src.mp3``, production profile), pass 1
clears the guard only at strength 0.50 — and it is pass 1's finishing (the
bounded tonal restoration) that lifts the presence band the guard needs to
hear. That restored output lets pass 2 run at strength 1.00. Speech/floor
separation (this module's metric, measured 2026-08-20 on this tree): source
14.93 dB -> pass 1 19.65 dB -> pass 2 23.64 dB, with zero musical-noise
inflation (guard ``musical_noise`` 0.000; the lab evidence also recorded
pause spectral-flatness sigma improving versus the source). Stripping
mastering from intermediate passes would hand pass 2 the un-restored
spectrum and lose exactly the headroom that unlocks full strength. A third
pass converges: separation dips to 23.18 dB (gain -0.46 dB, measured) while
the lab's pause-peakiness reading kept rising across passes (33.6 -> 36.9 ->
38.8 dB) — which is why auto mode requires each new pass to EARN its place
(guard verdicts must not regress, separation must improve by at least
0.5 dB) and ships the previous pass whenever a new one fails.

Fail-closed properties
----------------------
- Every pass is a full :func:`hawavoclean.pipeline.run_pipeline` run —
  guard-protected end to end. Multipass adds no path that ships un-guarded
  audio.
- If ANY pass raises, the whole run fails and nothing is published; auto
  mode's discard applies only to a pass that completed but failed the
  improvement criteria, never to an error.
- Intermediate outputs and their reports live in a private ``multipass-*``
  directory under the work root, removed on every exit path — success,
  error, and interrupt (SIGINT/SIGTERM unwind through the same ``finally``).
- The destination is preflighted before pass 1 decodes a sample, and the
  final master + amended report + summary are published atomically through
  the same staging discipline as a single-pass run.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hawavoclean.audio.decode import decode_audio
from hawavoclean.audio.probe import probe_audio
from hawavoclean.errors import InvalidUserInputError
from hawavoclean.job import JobWorkspace
from hawavoclean.logging import get_logger
from hawavoclean.paths import work_root
from hawavoclean.pipeline import _preflight_destination, run_pipeline
from hawavoclean.progress import ProgressCallback, ProgressEvent, emit_progress
from hawavoclean.publication import public_output_path, resolve_committed_publication
from hawavoclean.report.schema import HawaVoCleanReport, MediaStats, PassRecord
from hawavoclean.report.summary import generate_human_summary
from hawavoclean.report.writer import serialize_json_report

logger = get_logger("multipass")

#: Hard cap on passes, explicit or auto. Beyond this the lab evidence shows
#: convergence (separation flat or falling, pause peakiness still rising).
MAX_PASSES = 4

#: Auto mode keeps a pass only if it deepens speech/floor separation by at
#: least this much over the previous pass.
MIN_SEPARATION_GAIN_DB = 0.5


def speech_floor_separation_db(
    mono: np.ndarray[Any, np.dtype[np.floating[Any]]],
    frame_samples: int = 2048,
    hop_samples: int = 1024,
) -> float:
    """Speech/floor separation of a mono waveform, in dB.

    Frame RMS (2048-sample frames, 1024 hop), then the spread between the
    90th and 10th percentile of the per-frame RMS in dB. A recording whose
    speech stands far out of its noise floor scores high; a uniformly loud
    (or uniformly silent) signal scores ~0.

    Implemented over a cumulative energy sum, so memory stays linear in the
    signal — numerically identical to the naive framed computation (unit
    tested against it).
    """
    x = np.asarray(mono, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    frame = min(int(frame_samples), x.size)
    hop = max(1, int(hop_samples))
    cumulative = np.concatenate(([0.0], np.cumsum(x * x)))
    starts = np.arange(0, x.size - frame + 1, hop)
    energy = cumulative[starts + frame] - cumulative[starts]
    # max(., 1e-20) on mean-square == an RMS floor of 1e-10 (-200 dB).
    rms_db = 10.0 * np.log10(np.maximum(energy / frame, 1e-20))
    return float(np.percentile(rms_db, 90.0) - np.percentile(rms_db, 10.0))


#: B2 · Cumulative spectral drift ceiling. Auto mode ships the previous pass
#: if the current pass's log-spectral distance from the ORIGINAL source exceeds
#: this. The value is calibrated against the lab corpus: a single production
#: pass on teat1vo-lab produces LSD ≈ 0.6 dB, two passes ≈ 0.9 dB, three
#: passes ≈ 1.1 dB. Beyond 1.5 dB the lab reviewers reported audible artifacts.
MAX_CUMULATIVE_DRIFT_DB = 1.5


def cumulative_spectral_drift(
    original_mono: np.ndarray[Any, np.dtype[np.floating[Any]]],
    candidate_mono: np.ndarray[Any, np.dtype[np.floating[Any]]],
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """Log-Spectral Distance between original and candidate (cumulative drift).

    Measures how far the candidate has drifted from the original recording
    in spectral space, regardless of how many passes produced it.  Used by
    the multipass auto-mode to halt before artifacts compound.
    """
    win = np.hanning(n_fft)

    def _stft_power(
        x: np.ndarray[Any, np.dtype[np.floating[Any]]],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        num_frames = max(1, (len(x) - n_fft) // hop + 1)
        power = np.zeros((num_frames, n_fft // 2 + 1))
        for i in range(num_frames):
            chunk = x[i * hop : i * hop + n_fft] * win
            spec = np.fft.rfft(chunk, n=n_fft)
            power[i] = np.abs(spec) ** 2
        return power

    min_len = min(len(original_mono), len(candidate_mono))
    orig = original_mono[:min_len].astype(np.float64)
    cand = candidate_mono[:min_len].astype(np.float64)

    ref_power = _stft_power(orig)
    cand_power = _stft_power(cand)

    ref_log = np.log10(np.maximum(ref_power, 1e-10))
    cand_log = np.log10(np.maximum(cand_power, 1e-10))

    frame_lsd = np.sqrt(np.mean((ref_log - cand_log) ** 2, axis=1))
    return float(np.mean(frame_lsd))


def measure_separation_db(audio_path: Path) -> float:
    """Separation of a written audio file's mono mix, via the ordinary
    probe/decode path (the pipeline does not keep its decoded output)."""
    probe = probe_audio(audio_path)
    buf = decode_audio(probe)
    mono = buf.data.mean(axis=0)
    return speech_floor_separation_db(mono)


def auto_pass_verdict(prev: PassRecord, new: PassRecord) -> tuple[bool, str | None]:
    """Whether auto mode keeps ``new`` after ``prev``.

    Keep only while BOTH hold: (a) guard verdicts do not regress (the new
    pass's enhanced-unit count is >= the previous pass's), and (b) measured
    separation improves by >= :data:`MIN_SEPARATION_GAIN_DB`. Returns
    ``(keep, reason)`` where ``reason`` names every failed criterion.
    """
    reasons: list[str] = []
    if new.enhanced < prev.enhanced:
        reasons.append(
            f"guard regressed: pass {new.pass_index} enhanced {new.enhanced}/"
            f"{new.units_total} units vs pass {prev.pass_index}'s {prev.enhanced}"
        )
    gain = new.separation_db - prev.separation_db
    if gain < MIN_SEPARATION_GAIN_DB:
        reasons.append(
            f"separation gain {gain:+.2f} dB below the "
            f"+{MIN_SEPARATION_GAIN_DB:.2f} dB floor "
            f"(pass {prev.pass_index}: {prev.separation_db:.2f} dB, "
            f"pass {new.pass_index}: {new.separation_db:.2f} dB)"
        )
    if reasons:
        return False, "; ".join(reasons)
    return True, None


def rescale_event(event: ProgressEvent, pass_index: int, pass_total: int | None) -> ProgressEvent:
    """Map one pass's [0, 1] progress into its share of the whole run.

    Explicit N: pass k owns [(k-1)/N, k/N]. Auto (``pass_total is None``):
    the current pass is treated as the last until another starts, so pass k
    owns [(k-1)/k, 1].
    """
    denominator = pass_total if pass_total is not None else pass_index
    low = (pass_index - 1) / denominator
    return ProgressEvent(
        stage=event.stage,
        progress=low + event.progress / denominator,
        message=event.message,
        unit_index=event.unit_index,
        unit_total=event.unit_total,
        pass_index=pass_index,
        pass_total=pass_total,
    )


def _pass_progress(
    on_progress: ProgressCallback | None, pass_index: int, pass_total: int | None
) -> ProgressCallback | None:
    if on_progress is None:
        return None

    def forward(event: ProgressEvent) -> None:
        emit_progress(on_progress, rescale_event(event, pass_index, pass_total))

    return forward


def _pass_record(pass_index: int, report: HawaVoCleanReport, separation_db: float) -> PassRecord:
    strengths = sorted({u.chosen_strength for u in report.units if u.final_decision == "enhanced"})
    return PassRecord(
        pass_index=pass_index,
        input_sha256=report.input.sha256,
        output_sha256=report.output.sha256,
        units_total=report.summary.units_total,
        enhanced=report.summary.enhanced,
        reverted=report.summary.reverted,
        chosen_strengths=strengths,
        separation_db=separation_db,
        integrated_lufs=report.output.integrated_lufs,
    )


def run_multipass(
    input_path: Path | str,
    output_path: Path | str,
    passes: int | Literal["auto"] | str,
    config_path: Path | str | None = None,
    profile: str = "production",
    overwrite: bool = False,
    on_progress: ProgressCallback | None = None,
) -> HawaVoCleanReport:
    """Run the pipeline ``passes`` times (or ``"auto"``) and publish the
    final master with a per-pass audit trail in its report.

    ``passes=1`` is delegated straight to :func:`run_pipeline` — the ordinary
    single-pass run, byte-identical output, empty ``passes`` list.
    """
    auto = passes == "auto"
    if not auto:
        try:
            pass_count = int(passes)
        except (TypeError, ValueError):
            raise InvalidUserInputError(
                f"--passes must be an integer 1..{MAX_PASSES} or 'auto', got {passes!r}"
            ) from None
        if not 1 <= pass_count <= MAX_PASSES:
            raise InvalidUserInputError(
                f"--passes must be an integer 1..{MAX_PASSES} or 'auto', got {passes!r}"
            )
        if pass_count == 1:
            return run_pipeline(
                input_path=input_path,
                output_path=output_path,
                config_path=config_path,
                profile=profile,
                overwrite=overwrite,
                on_progress=on_progress,
            )
    target = MAX_PASSES if auto else pass_count

    in_path = Path(input_path).resolve()
    out_path = public_output_path(output_path)
    # Refuse a bad destination BEFORE pass 1 decodes a sample — the per-pass
    # preflight only ever sees the private temp destinations.
    _preflight_destination(in_path, out_path, overwrite)

    base = work_root()
    base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="multipass-", dir=base))
    logger.info(
        f"Multipass run: {in_path} -> {out_path} "
        f"[passes={'auto' if auto else target}, profile={profile}]"
    )

    try:
        records: list[PassRecord] = []
        original_input: MediaStats | None = None
        shipped_report: HawaVoCleanReport | None = None
        shipped_audio: Path | None = None
        current_input = in_path
        checkpoint_path = tmp_root / "checkpoint.json"

        def _write_checkpoint(pass_index: int, record: PassRecord) -> None:
            """B3: Persist multipass state after each pass for crash recovery."""
            import json as _json

            checkpoint = {
                "pass_index": pass_index,
                "input_path": str(in_path),
                "output_path": str(out_path),
                "profile": profile,
                "auto": auto,
                "target": target,
                "records": [r.model_dump(mode="json") for r in records],
                "current_record": record.model_dump(mode="json"),
            }
            checkpoint_path.write_text(_json.dumps(checkpoint, indent=2) + "\n")

        # B2: Decode original source ONCE for cumulative drift comparison.
        # This stays in memory for the (short) loop; only the mono mix is kept.
        original_probe = probe_audio(in_path)
        original_buf = decode_audio(original_probe)
        original_mono = original_buf.data.mean(axis=0).astype(np.float64)

        for k in range(1, target + 1):
            pass_out = tmp_root / f"pass{k}.wav"
            report = run_pipeline(
                input_path=current_input,
                output_path=pass_out,
                config_path=config_path,
                profile=profile,
                overwrite=False,
                on_progress=_pass_progress(on_progress, k, None if auto else target),
            )
            committed_pass = resolve_committed_publication(pass_out)
            pass_audio = committed_pass[0] if committed_pass is not None else pass_out
            record = _pass_record(k, report, measure_separation_db(pass_audio))

            # B2: Measure cumulative drift from original for passes > 1.
            drift: float | None = None
            if k > 1:
                pass_probe = probe_audio(pass_audio)
                pass_buf = decode_audio(pass_probe)
                pass_mono = pass_buf.data.mean(axis=0).astype(np.float64)
                drift = cumulative_spectral_drift(original_mono, pass_mono)
                record = record.model_copy(update={"cumulative_drift_db": drift})
                logger.info(
                    f"Pass {k} cumulative drift from source: {drift:.3f} dB "
                    f"(ceiling {MAX_CUMULATIVE_DRIFT_DB:.1f} dB)"
                )

            if k == 1:
                # Only pass 1's report holds the ORIGINAL source's MediaStats;
                # every later pass's "input" is a temp file about to vanish.
                original_input = report.input

                if auto and record.separation_db >= 50.0:
                    logger.info(
                        f"Pass 1 separation {record.separation_db:.2f} dB achieves pristine clarity "
                        "(>= 50 dB); auto mode ships single neural pass without cascading."
                    )
                    records.append(record)
                    shipped_report = report
                    shipped_audio = pass_audio
                    break

            if auto and k > 1:
                # B2: Check cumulative drift BEFORE the existing auto verdict
                if drift is not None and drift > MAX_CUMULATIVE_DRIFT_DB:
                    drift_reason = (
                        f"cumulative spectral drift {drift:.3f} dB exceeds the "
                        f"{MAX_CUMULATIVE_DRIFT_DB:.1f} dB ceiling"
                    )
                    logger.info(f"Auto mode halts pass {k}: {drift_reason}")
                    records.append(
                        record.model_copy(
                            update={"discarded": True, "discard_reason": drift_reason}
                        )
                    )
                    break
                keep, pass_reason = auto_pass_verdict(records[-1], record)
                if not keep:
                    logger.info(f"Auto mode discards pass {k}: {pass_reason}")
                    records.append(
                        record.model_copy(update={"discarded": True, "discard_reason": pass_reason})
                    )
                    break

            records.append(record)
            shipped_report = report
            shipped_audio = pass_audio
            current_input = pass_audio
            _write_checkpoint(k, record)  # B3: crash recovery
            logger.info(
                f"Pass {k}: {record.enhanced}/{record.units_total} enhanced, "
                f"separation {record.separation_db:.2f} dB"
            )

        # Pass 1 either completed (populating all three) or raised.
        assert shipped_report is not None and shipped_audio is not None
        assert original_input is not None

        # The final report is the SHIPPED pass's report — its unit records
        # are the master's forensic trail — re-rooted at the user's paths:
        # input is the original source, output path the real destination
        # (the audio bytes are the shipped pass's, so every hash still
        # verifies), and passes[] the journey, discarded pass included.
        final_report = shipped_report.model_copy(
            update={
                "input": original_input,
                "output": shipped_report.output.model_copy(update={"path": str(out_path)}),
                "passes": records,
            }
        )

        JobWorkspace.publish_atomically(
            temp_audio_path=shipped_audio,
            destination_audio_path=out_path,
            json_report_str=serialize_json_report(final_report),
            txt_summary_str=generate_human_summary(final_report),
            overwrite=overwrite,
        )
        shipped_index = next(r.pass_index for r in reversed(records) if not r.discarded)
        logger.info(
            f"Multipass finished: shipped pass {shipped_index} "
            f"({len(records)} recorded) to {out_path}"
        )
        return final_report
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
