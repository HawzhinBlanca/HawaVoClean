"""Per-unit slice cache and sub-100ms non-destructive master re-stitching.

Stores original and enhanced candidate waveforms per speech unit as compact
uncompressed ``.npz`` files under ``<work_dir>/unit_slices/<job_id>/`` with
atomic writes and bounded retention cleanup. Re-stitching applies manual
per-unit overrides (``auto``, ``force_original``, ``force_enhanced``, or
``custom_strength``) without re-running neural inference, blends seams with
raised-cosine continuity tapers, stitches via
:func:`~hawavoclean.assembly.stitch.assemble_channel_timeline`, and encodes
the updated master WAV atomically.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from hawavoclean.assembly.stitch import assemble_channel_timeline
from hawavoclean.audio.encode import encode_audio
from hawavoclean.audio.types import AudioBuffer
from hawavoclean.hashing import hash_file
from hawavoclean.paths import work_root
from hawavoclean.policy.continuity import CONTINUITY_TAPER_MS, apply_continuity_taper
from hawavoclean.segmentation.types import SpeechUnit

UnitOverrideAction = Literal[
    "auto",
    "force_original",
    "force_enhanced",
    "custom_strength",
]

_VALID_OVERRIDE_ACTIONS: frozenset[str] = frozenset(
    {"auto", "force_original", "force_enhanced", "custom_strength"}
)
_SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SLICE_FILE_RE = re.compile(r"^unit_(\d+)_ch(\d+)\.npz$")

DEFAULT_MAX_SLICE_JOBS = 32
DEFAULT_MAX_SLICE_AGE_S = 7.0 * 86400.0


def _validate_job_id(job_id: str) -> str:
    if not _SAFE_JOB_ID_RE.match(job_id) or ".." in job_id:
        raise ValueError(f"Invalid job_id for UnitSliceStore: {job_id!r}")
    return job_id


def _slice_filename(unit_id: int, channel: int) -> str:
    if isinstance(unit_id, bool) or unit_id < 0:
        raise ValueError(f"unit_id must be >= 0, got {unit_id!r}")
    if isinstance(channel, bool) or channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel!r}")
    return f"unit_{unit_id:06d}_ch{channel}.npz"


@dataclass(frozen=True)
class UnitSlice:
    """Cached per-unit original/candidate waveforms and metadata."""

    unit_id: int
    channel: int
    original: np.ndarray[Any, np.dtype[np.float32]]
    candidate: np.ndarray[Any, np.dtype[np.float32]] | None
    unit_meta: dict[str, Any]


def save_unit_slice_to_dir(
    job_dir: Path,
    unit_id: int,
    original: np.ndarray[Any, Any],
    candidate: np.ndarray[Any, Any] | None,
    unit_meta: dict[str, Any] | None = None,
    *,
    channel: int = 0,
) -> Path:
    """Atomically write one unit's ``(original, candidate, unit_meta)`` slice into ``job_dir``."""
    orig_arr = np.ascontiguousarray(original, dtype=np.float32)
    if orig_arr.ndim != 1:
        raise ValueError(f"original slice must be 1D, got shape {orig_arr.shape}")
    if not np.all(np.isfinite(orig_arr)):
        raise ValueError(f"original slice for unit {unit_id} contains NaN or Inf")

    has_candidate = candidate is not None
    if candidate is not None:
        cand_arr = np.ascontiguousarray(candidate, dtype=np.float32)
        if cand_arr.ndim != 1:
            raise ValueError(f"candidate slice must be 1D, got shape {cand_arr.shape}")
        if not np.all(np.isfinite(cand_arr)):
            raise ValueError(f"candidate slice for unit {unit_id} contains NaN or Inf")
    else:
        cand_arr = np.empty(0, dtype=np.float32)

    meta_payload = json.dumps(
        unit_meta or {},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    target_dir = Path(job_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / _slice_filename(unit_id, channel)
    tmp_path = target_dir / (f".{final_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")

    try:
        with open(tmp_path, "wb") as handle:
            np.savez(
                handle,
                original=orig_arr,
                candidate=cand_arr,
                has_candidate=np.asarray(has_candidate, dtype=np.bool_),
                meta_json=np.frombuffer(meta_payload, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path


def load_unit_slice(
    job_dir: Path,
    unit_id: int,
    *,
    channel: int = 0,
) -> UnitSlice:
    """Load one cached unit slice from ``job_dir``."""
    target_path = Path(job_dir).resolve() / _slice_filename(unit_id, channel)
    if not target_path.is_file():
        raise FileNotFoundError(
            f"Cached slice not found for unit {unit_id} channel {channel}: {target_path}"
        )
    with np.load(target_path, allow_pickle=False) as archive:
        orig_arr = np.ascontiguousarray(archive["original"], dtype=np.float32)
        has_candidate = bool(archive["has_candidate"].item())
        cand_arr: np.ndarray[Any, np.dtype[np.float32]] | None = (
            np.ascontiguousarray(archive["candidate"], dtype=np.float32) if has_candidate else None
        )
        raw_meta = archive["meta_json"].tobytes().decode("utf-8")
        meta = json.loads(raw_meta)
        if not isinstance(meta, dict):
            raise ValueError(f"Corrupt unit metadata in {target_path}")
    return UnitSlice(
        unit_id=unit_id,
        channel=channel,
        original=orig_arr,
        candidate=cand_arr,
        unit_meta=meta,
    )


class UnitSliceStore:
    """Managed per-job slice store under ``<work_dir>/unit_slices/<job_id>/``."""

    def __init__(
        self,
        work_dir: Path | None = None,
        *,
        max_jobs: int = DEFAULT_MAX_SLICE_JOBS,
        max_age_s: float = DEFAULT_MAX_SLICE_AGE_S,
    ) -> None:
        if isinstance(max_jobs, bool) or max_jobs < 1:
            raise ValueError(f"max_jobs must be >= 1, got {max_jobs!r}")
        if not math.isfinite(max_age_s) or max_age_s <= 0.0:
            raise ValueError(f"max_age_s must be positive finite, got {max_age_s!r}")

        self._work_dir = (Path(work_dir) if work_dir is not None else work_root()).resolve()
        self._base_dir = self._work_dir / "unit_slices"
        self._max_jobs = int(max_jobs)
        self._max_age_s = float(max_age_s)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def job_dir(self, job_id: str) -> Path:
        safe = _validate_job_id(job_id)
        return self._base_dir / safe

    def save_unit(
        self,
        job_id: str,
        unit_id: int,
        original: np.ndarray[Any, Any],
        candidate: np.ndarray[Any, Any] | None,
        unit_meta: dict[str, Any] | None = None,
        *,
        channel: int = 0,
    ) -> Path:
        """Atomically save a unit slice under ``unit_slices/<job_id>/``."""
        return save_unit_slice_to_dir(
            self.job_dir(job_id),
            unit_id=unit_id,
            original=original,
            candidate=candidate,
            unit_meta=unit_meta,
            channel=channel,
        )

    def load_unit(
        self,
        job_id: str,
        unit_id: int,
        *,
        channel: int = 0,
    ) -> UnitSlice:
        """Load a saved unit slice for ``job_id``."""
        return load_unit_slice(self.job_dir(job_id), unit_id=unit_id, channel=channel)

    def list_units(self, job_id: str) -> list[tuple[int, int]]:
        """List saved ``(channel, unit_id)`` pairs for ``job_id`` sorted by channel then id."""
        directory = self.job_dir(job_id)
        if not directory.is_dir():
            return []
        found: list[tuple[int, int]] = []
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            match = _SLICE_FILE_RE.match(entry.name)
            if match is not None:
                unit_id = int(match.group(1))
                ch = int(match.group(2))
                found.append((ch, unit_id))
        found.sort()
        return found

    def delete_job(self, job_id: str) -> bool:
        """Delete all cached slices for ``job_id``."""
        directory = self.job_dir(job_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        return True

    def cleanup(self) -> list[str]:
        """Remove expired or excess job slice directories under ``unit_slices/``."""
        if not self._base_dir.is_dir():
            return []

        now = time.time()
        entries: list[tuple[float, str, Path]] = []
        for child in self._base_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            entries.append((mtime, child.name, child))

        removed: list[str] = []
        survivors: list[tuple[float, str, Path]] = []
        for mtime, name, child in entries:
            if (now - mtime) > self._max_age_s:
                shutil.rmtree(child, ignore_errors=True)
                removed.append(name)
            else:
                survivors.append((mtime, name, child))

        survivors.sort(key=lambda item: item[0], reverse=True)
        for _mtime, name, child in survivors[self._max_jobs :]:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(name)
        return removed


def _fit_waveform_length(
    wave: np.ndarray[Any, np.dtype[np.float32]],
    target_len: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    n = int(wave.shape[0])
    if n == target_len:
        return np.ascontiguousarray(wave, dtype=np.float32)
    if n > target_len:
        return np.ascontiguousarray(wave[:target_len], dtype=np.float32)
    return np.ascontiguousarray(
        np.pad(wave, (0, target_len - n), mode="constant"), dtype=np.float32
    )


def _blend_strength(
    original: np.ndarray[Any, np.dtype[np.float32]],
    candidate: np.ndarray[Any, np.dtype[np.float32]],
    strength: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if strength <= 0.0:
        return original.copy()
    if abs(strength - 1.0) < 1e-6:
        return candidate.copy()
    return np.ascontiguousarray(
        original + float(strength) * (candidate - original), dtype=np.float32
    )


@dataclass
class _ResolvedUnit:
    unit_id: int
    channel: int
    start_sample: int
    end_sample: int
    is_speech: bool
    forced_boundary: bool
    original: np.ndarray[Any, np.dtype[np.float32]]
    selected: np.ndarray[Any, np.dtype[np.float32]]
    is_enhanced: bool
    chosen_strength: float
    final_decision: str
    manual_override: str | None
    record: dict[str, Any]


def restitch_master_with_overrides(
    job_dir: Path,
    unit_records: list[dict[str, Any]],
    overrides: dict[int, tuple[UnitOverrideAction, float | None]],
    master_output_path: Path,
    sample_rate: int = 48000,
    channels: int = 1,
    *,
    crossfade_ms: float = 20.0,
    continuity_taper_ms: float = CONTINUITY_TAPER_MS,
    output_bit_depth: Literal["pcm24", "float32"] = "pcm24",
    dither: bool = True,
    seed_context: str = "hawavoclean-restitch",
) -> dict[str, Any]:
    """Re-stitch a job master from cached unit slices with per-unit manual overrides.

    Loads cached ``(original, candidate)`` slices from ``job_dir``, applies
    each unit's override (``auto``, ``force_original``, ``force_enhanced``, or
    ``custom_strength``), smooths transitions across enhanced/original
    boundaries using raised-cosine continuity tapers, stitches via
    :func:`assemble_channel_timeline`, and writes ``master_output_path``
    atomically via ``encode_audio`` + ``os.replace``.
    """
    t0 = time.perf_counter()

    if not unit_records:
        raise ValueError("unit_records must not be empty")
    if isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if isinstance(channels, bool) or channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels!r}")

    resolved_job_dir = Path(job_dir).resolve()
    by_channel: dict[int, list[_ResolvedUnit]] = {ch: [] for ch in range(channels)}
    total_samples = 0

    for rec in unit_records:
        unit_id = int(rec["unit_id"])
        ch = int(rec.get("channel", 0))
        if not 0 <= ch < channels:
            raise ValueError(f"Unit {unit_id} channel {ch} out of range 0..{channels - 1}")
        start_sample = int(rec["start_sample"])
        end_sample = int(rec["end_sample"])
        if start_sample < 0 or end_sample <= start_sample:
            raise ValueError(
                f"Unit {unit_id} has invalid sample bounds [{start_sample}, {end_sample})"
            )
        core_len = end_sample - start_sample
        total_samples = max(total_samples, end_sample)

        unit_slice = load_unit_slice(resolved_job_dir, unit_id=unit_id, channel=ch)
        orig = _fit_waveform_length(unit_slice.original, core_len)
        cand = (
            _fit_waveform_length(unit_slice.candidate, core_len)
            if unit_slice.candidate is not None
            else None
        )

        is_speech = bool(rec.get("is_speech", True))
        forced_boundary = bool(
            rec.get("forced_boundary", unit_slice.unit_meta.get("forced_boundary", False))
        )

        action, custom_strength = overrides.get(unit_id, ("auto", None))
        if action not in _VALID_OVERRIDE_ACTIONS:
            raise ValueError(f"Unsupported override action {action!r} for unit {unit_id}")

        if action == "force_original":
            selected = orig.copy()
            is_enhanced = False
            chosen_strength = 0.0
            final_decision = "NO_SPEECH" if not is_speech else "REVERTED"
            manual_override: str | None = "force_original"
        elif action == "force_enhanced":
            if cand is None:
                raise ValueError(
                    f"Unit {unit_id} cannot use 'force_enhanced': no cached candidate slice"
                )
            selected = cand.copy()
            is_enhanced = True
            chosen_strength = 1.0
            final_decision = "ENHANCED"
            manual_override = "force_enhanced"
        elif action == "custom_strength":
            if (
                custom_strength is None
                or isinstance(custom_strength, bool)
                or not math.isfinite(custom_strength)
                or not 0.0 <= float(custom_strength) <= 1.0
            ):
                raise ValueError(
                    f"Unit {unit_id} 'custom_strength' requires finite float in [0.0, 1.0], "
                    f"got {custom_strength!r}"
                )
            s = float(custom_strength)
            if s == 0.0:
                selected = orig.copy()
                is_enhanced = False
                chosen_strength = 0.0
                final_decision = "NO_SPEECH" if not is_speech else "REVERTED"
            else:
                if cand is None:
                    raise ValueError(
                        f"Unit {unit_id} cannot use 'custom_strength' ({s}): no cached candidate"
                    )
                selected = _blend_strength(orig, cand, s)
                is_enhanced = True
                chosen_strength = s
                final_decision = "ENHANCED"
            manual_override = f"custom_strength:{s:.3f}"
        else:
            # "auto": preserve original job decision.
            prev_decision = str(
                rec.get("final_decision", "NO_SPEECH" if not is_speech else "REVERTED")
            )
            prev_strength = float(rec.get("chosen_strength", 0.0))
            if (
                prev_decision.upper() in ("ENHANCED", "CONTINUITY_CROSSFADED")
                and cand is not None
                and prev_strength > 0.0
            ):
                clamped_s = float(min(1.0, max(0.0, prev_strength)))
                selected = _blend_strength(orig, cand, clamped_s)
                is_enhanced = True
                chosen_strength = clamped_s
                final_decision = prev_decision
            else:
                selected = orig.copy()
                is_enhanced = False
                chosen_strength = 0.0
                final_decision = prev_decision
            manual_override = None

        by_channel[ch].append(
            _ResolvedUnit(
                unit_id=unit_id,
                channel=ch,
                start_sample=start_sample,
                end_sample=end_sample,
                is_speech=is_speech,
                forced_boundary=forced_boundary,
                original=orig,
                selected=selected,
                is_enhanced=is_enhanced,
                chosen_strength=chosen_strength,
                final_decision=final_decision,
                manual_override=manual_override,
                record=dict(rec),
            )
        )

    taper_samples = max(0, int(round(sample_rate * (continuity_taper_ms / 1000.0))))
    channel_timelines: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    updated_records: list[dict[str, Any]] = []

    units_total = 0
    enhanced_count = 0
    reverted_count = 0
    no_speech_count = 0
    continuity_crossfaded_count = 0
    manual_overrides_count = 0

    for ch in range(channels):
        ch_units = sorted(by_channel[ch], key=lambda u: (u.start_sample, u.unit_id))
        if not ch_units:
            channel_timelines.append(np.zeros(total_samples, dtype=np.float32))
            continue

        speech_units: list[SpeechUnit] = []
        waveforms: list[np.ndarray[Any, np.dtype[np.float32]]] = []

        for idx, item in enumerate(ch_units):
            core_len = item.end_sample - item.start_sample
            max_taper = core_len // 4
            fade_in = 0
            fade_out = 0

            if item.is_enhanced and taper_samples > 0 and max_taper > 0:
                if idx > 0:
                    prev_item = ch_units[idx - 1]
                    if prev_item.end_sample == item.start_sample and not prev_item.is_enhanced:
                        fade_in = min(taper_samples, max_taper)
                if idx + 1 < len(ch_units):
                    next_item = ch_units[idx + 1]
                    if item.end_sample == next_item.start_sample and not next_item.is_enhanced:
                        fade_out = min(taper_samples, max_taper)

            if fade_in > 0 or fade_out > 0:
                wave = apply_continuity_taper(item.selected, item.original, fade_in, fade_out)
                continuity_crossfaded_count += 1
            else:
                wave = item.selected

            speech_units.append(
                SpeechUnit(
                    unit_id=item.unit_id,
                    channel_id=ch,
                    start_sample=item.start_sample,
                    end_sample=item.end_sample,
                    context_start_sample=item.start_sample,
                    context_end_sample=item.end_sample,
                    is_speech=item.is_speech,
                    forced_boundary=item.forced_boundary,
                )
            )
            waveforms.append(wave)

            units_total += 1
            if item.manual_override is not None:
                manual_overrides_count += 1
            if not item.is_speech and not item.is_enhanced:
                no_speech_count += 1
            elif item.is_enhanced:
                enhanced_count += 1
            else:
                reverted_count += 1

            updated = dict(item.record)
            updated["chosen_strength"] = round(item.chosen_strength, 4)
            updated["final_decision"] = item.final_decision
            updated["manual_override"] = item.manual_override
            updated_records.append(updated)

        timeline = assemble_channel_timeline(
            speech_units,
            waveforms,
            total_samples=total_samples,
            sample_rate=sample_rate,
            crossfade_ms=crossfade_ms,
        )
        channel_timelines.append(timeline)

    master_data = np.ascontiguousarray(np.vstack(channel_timelines), dtype=np.float32)
    master_buf = AudioBuffer(data=master_data, sample_rate=sample_rate)

    dest_path = Path(master_output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(
        f".{dest_path.name}.{os.getpid()}.{threading.get_ident()}.tmp.wav"
    )
    try:
        encode_audio(
            master_buf,
            tmp_path,
            output_bit_depth=output_bit_depth,
            dither=dither,
            seed_context=seed_context,
        )
        os.replace(tmp_path, dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    master_sha256 = hash_file(dest_path)
    restitch_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "master_path": str(dest_path),
        "master_sha256": master_sha256,
        "restitch_ms": restitch_ms,
        "sample_rate": sample_rate,
        "channels": channels,
        "total_samples": total_samples,
        "summary": {
            "units_total": units_total,
            "enhanced": enhanced_count,
            "reverted": reverted_count,
            "no_speech": no_speech_count,
            "continuity_crossfaded": continuity_crossfaded_count,
            "manual_overrides": manual_overrides_count,
        },
        "units": updated_records,
    }
