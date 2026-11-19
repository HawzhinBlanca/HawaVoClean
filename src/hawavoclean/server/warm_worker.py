"""Persistent warm enhancement worker cache and sub-second region preview renderer.

Keeps recently used :class:`~hawavoclean.enhancement.protocol.Enhancer` cores
resident in memory across preview requests with thread-safe LRU eviction,
idle TTL expiration, and RSS memory-cap enforcement against
``runtime.worker_memory_limit_mb`` (via :mod:`hawavoclean.runtime`).

Region previews decode *only* the requested ``[start_s, end_s]`` window
(clamped to :data:`MAX_PREVIEW_DURATION_S`), run the warm core with optional
wet/dry strength blending, compute before/after LUFS, noise floor, and optional
perceptual MOS deltas, and write the preview WAV atomically inside a managed
preview directory.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hawavoclean.audio.decode import decode_audio_window
from hawavoclean.audio.encode import encode_audio
from hawavoclean.audio.resample import resample_audio
from hawavoclean.audio.types import AudioBuffer
from hawavoclean.config import load_config
from hawavoclean.enhancement.factory import CORE_REGISTRY, resolve_core
from hawavoclean.enhancement.protocol import Enhancer
from hawavoclean.finishing.loudness import measure_loudness_and_peaks
from hawavoclean.guard.perceptual_mos import PerceptualQualityScorer
from hawavoclean.paths import profile_config_path, work_root
from hawavoclean.runtime import memory_budget_mb, process_peak_rss_bytes
from hawavoclean.server.analysis import FLOOR_DB, cached_probe

MAX_PREVIEW_DURATION_S = 15.0
DEFAULT_PREVIEW_SAMPLE_RATE = 48000
DEFAULT_MAX_PREVIEW_FILES = 64
DEFAULT_PREVIEW_MAX_AGE_S = 3600.0
_BYTES_PER_MIB = 1024 * 1024


@dataclass
class _CachedCore:
    core_id: str
    sample_rate: int
    enhancer: Enhancer
    loaded_at: float
    last_used_at: float
    hits: int = 0


def _default_core_factory(core_id: str, sample_rate: int) -> Enhancer:
    """Instantiate and warm up a registered enhancement core."""
    registration = resolve_core(core_id)
    enhancer = registration.enhancer_class(core_id=core_id, sample_rate=sample_rate)
    enhancer.warmup()
    return enhancer


class WarmCoreCache:
    """Thread-safe LRU cache holding warmed :class:`Enhancer` cores keyed by ``core_id``."""

    def __init__(
        self,
        *,
        max_cores: int = 2,
        idle_ttl_s: float = 300.0,
        memory_limit_mb: int | None = None,
        core_factory: Callable[[str, int], Enhancer] | None = None,
        clock: Callable[[], float] | None = None,
        rss_bytes_fn: Callable[[], int] | None = None,
    ) -> None:
        if isinstance(max_cores, bool) or max_cores < 1:
            raise ValueError(f"max_cores must be >= 1, got {max_cores!r}")
        if not math.isfinite(idle_ttl_s) or idle_ttl_s <= 0.0:
            raise ValueError(f"idle_ttl_s must be a positive finite number, got {idle_ttl_s!r}")
        if memory_limit_mb is not None and (
            isinstance(memory_limit_mb, bool) or memory_limit_mb < 1
        ):
            raise ValueError(f"memory_limit_mb must be >= 1, got {memory_limit_mb!r}")

        self._max_cores = int(max_cores)
        self._idle_ttl_s = float(idle_ttl_s)
        self._memory_limit_mb = int(memory_limit_mb) if memory_limit_mb is not None else None
        self._core_factory = core_factory or _default_core_factory
        self._clock = clock or time.monotonic
        self._rss_bytes_fn = rss_bytes_fn or process_peak_rss_bytes

        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _CachedCore] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._ttl_evictions = 0
        self._memory_evictions = 0

    def _effective_memory_limit_mb(self) -> int | None:
        if self._memory_limit_mb is not None:
            return self._memory_limit_mb
        return memory_budget_mb()

    def _current_rss_mb(self) -> float:
        return float(self._rss_bytes_fn()) / float(_BYTES_PER_MIB)

    def _evict_expired_locked(self, now: float) -> list[str]:
        evicted: list[str] = []
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if (now - entry.last_used_at) >= self._idle_ttl_s
        ]
        for key in expired_keys:
            self._entries.pop(key, None)
            self._evictions += 1
            self._ttl_evictions += 1
            evicted.append(key)
        return evicted

    def _evict_for_memory_cap_locked(self, *, keep_key: str | None = None) -> list[str]:
        limit_mb = self._effective_memory_limit_mb()
        if limit_mb is None:
            return []

        evicted: list[str] = []
        while self._entries and self._current_rss_mb() > float(limit_mb):
            victim = next((k for k in self._entries if k != keep_key), None)
            if victim is None:
                break
            self._entries.pop(victim, None)
            self._evictions += 1
            self._memory_evictions += 1
            evicted.append(victim)
        return evicted

    def get_or_load(
        self,
        core_id: str,
        sample_rate: int = DEFAULT_PREVIEW_SAMPLE_RATE,
    ) -> tuple[Enhancer, bool]:
        """Return ``(enhancer, warm_hit)`` for ``core_id`` at ``sample_rate``.

        Evicts expired entries and enforces LRU and RSS memory caps before
        loading a cold core. Model construction runs outside the cache lock so
        unrelated cache operations stay responsive.
        """
        if not core_id:
            raise ValueError("core_id must be a non-empty string")
        if isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

        evicted_before_load = False
        with self._lock:
            now = self._clock()
            if self._evict_expired_locked(now):
                evicted_before_load = True

            existing = self._entries.get(core_id)
            if existing is not None and existing.sample_rate == sample_rate:
                existing.last_used_at = now
                existing.hits += 1
                self._entries.move_to_end(core_id)
                self._hits += 1
                return existing.enhancer, True

            if existing is not None:
                # Sample rate changed for the same core_id; drop stale instance.
                self._entries.pop(core_id, None)
                self._evictions += 1
                evicted_before_load = True

            if self._evict_for_memory_cap_locked():
                evicted_before_load = True

            while len(self._entries) >= self._max_cores:
                self._entries.popitem(last=False)
                self._evictions += 1
                evicted_before_load = True

        if evicted_before_load:
            gc.collect()

        # Construct and warm up outside the lock.
        enhancer = self._core_factory(core_id, sample_rate)

        evicted_after_load = False
        with self._lock:
            now = self._clock()
            if self._evict_expired_locked(now):
                evicted_after_load = True

            raced = self._entries.get(core_id)
            if raced is not None and raced.sample_rate == sample_rate:
                raced.last_used_at = now
                raced.hits += 1
                self._entries.move_to_end(core_id)
                self._hits += 1
                return raced.enhancer, True

            while len(self._entries) >= self._max_cores:
                self._entries.popitem(last=False)
                self._evictions += 1
                evicted_after_load = True

            self._entries[core_id] = _CachedCore(
                core_id=core_id,
                sample_rate=sample_rate,
                enhancer=enhancer,
                loaded_at=now,
                last_used_at=now,
            )
            self._entries.move_to_end(core_id)
            if self._evict_for_memory_cap_locked(keep_key=core_id):
                evicted_after_load = True
            self._misses += 1

        if evicted_after_load:
            gc.collect()
        return enhancer, False

    def evict_expired(self) -> list[str]:
        """Evict all cores past ``idle_ttl_s`` or over the configured RSS cap."""
        with self._lock:
            now = self._clock()
            evicted = self._evict_expired_locked(now)
            evicted.extend(self._evict_for_memory_cap_locked())
        if evicted:
            gc.collect()
        return evicted

    def clear(self) -> int:
        """Remove all cached cores and trigger garbage collection."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._evictions += count
        if count:
            gc.collect()
        return count

    def stats(self) -> dict[str, Any]:
        """Snapshot of cache occupancy, hit/miss counters, and memory state."""
        with self._lock:
            now = self._clock()
            cores = [
                {
                    "core_id": entry.core_id,
                    "sample_rate": entry.sample_rate,
                    "hits": entry.hits,
                    "idle_s": round(max(0.0, now - entry.last_used_at), 3),
                }
                for entry in self._entries.values()
            ]
            return {
                "cached_count": len(self._entries),
                "max_cores": self._max_cores,
                "idle_ttl_s": self._idle_ttl_s,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "ttl_evictions": self._ttl_evictions,
                "memory_evictions": self._memory_evictions,
                "rss_mb": round(self._current_rss_mb(), 2),
                "memory_limit_mb": self._effective_memory_limit_mb(),
                "cores": cores,
            }


_default_cache_lock = threading.Lock()
_default_cache: WarmCoreCache | None = None


def get_default_warm_cache() -> WarmCoreCache:
    """Return the process-wide default :class:`WarmCoreCache` singleton."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = WarmCoreCache()
        return _default_cache


@dataclass(frozen=True)
class RegionPreviewRequest:
    """Parameters for a sub-second windowed enhancement preview."""

    source_path: Path
    start_s: float
    end_s: float
    profile: str = "production"
    core_id: str | None = None
    strength: float = 1.0
    include_perceptual_mos: bool = False
    output_dir: Path | None = None


@dataclass(frozen=True)
class RegionPreviewResult:
    """Rendered preview audio location, timing, and before/after quality deltas."""

    preview_path: Path
    core_id: str
    profile: str
    start_s: float
    end_s: float
    duration_s: float
    sample_rate: int
    channels: int
    strength: float
    warm_hit: bool
    render_ms: float
    before_lufs: float
    after_lufs: float
    lufs_delta: float
    before_noise_floor_db: float
    after_noise_floor_db: float
    noise_floor_delta_db: float
    before_mos: float | None = None
    after_mos: float | None = None
    mos_delta: float | None = None
    mos_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "preview_path": str(self.preview_path),
            "core_id": self.core_id,
            "profile": self.profile,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "strength": self.strength,
            "warm_hit": self.warm_hit,
            "render_ms": self.render_ms,
            "before_lufs": self.before_lufs,
            "after_lufs": self.after_lufs,
            "lufs_delta": self.lufs_delta,
            "before_noise_floor_db": self.before_noise_floor_db,
            "after_noise_floor_db": self.after_noise_floor_db,
            "noise_floor_delta_db": self.noise_floor_delta_db,
            "before_mos": self.before_mos,
            "after_mos": self.after_mos,
            "mos_delta": self.mos_delta,
            "mos_backend": self.mos_backend,
        }


def resolve_preview_core_id(profile: str, core_id: str | None = None) -> tuple[str, int]:
    """Resolve ``(core_id, model_sample_rate)`` without mutating process runtime env."""
    if core_id is not None:
        resolve_core(core_id)
        return core_id, DEFAULT_PREVIEW_SAMPLE_RATE
    if profile in CORE_REGISTRY:
        return profile, DEFAULT_PREVIEW_SAMPLE_RATE

    cfg_path = profile_config_path(profile)
    if not cfg_path.is_file():
        raise ValueError(
            f"Unknown preview profile {profile!r}; expected a built-in profile or registered core_id"
        )
    cfg = load_config(cfg_path, is_production=False, activate=False)
    return cfg.enhancement.core_id, int(cfg.enhancement.model_sample_rate)


def _estimate_noise_floor_db(
    mono: np.ndarray[Any, np.dtype[np.float32]],
    sample_rate: int,
) -> float:
    """Estimate noise floor in dBFS via 10th-percentile 20 ms frame RMS."""
    n = int(mono.shape[0])
    if n == 0:
        return FLOOR_DB
    frame_len = max(16, int(round(sample_rate * 0.020)))
    hop = max(8, frame_len // 2)
    if n < frame_len:
        rms_sq = float(np.mean(np.square(mono.astype(np.float64))))
        if rms_sq <= 0.0:
            return FLOOR_DB
        return max(FLOOR_DB, float(10.0 * math.log10(rms_sq)))

    frames = np.lib.stride_tricks.sliding_window_view(mono, frame_len)[::hop]
    mean_sq = np.mean(np.square(frames.astype(np.float64)), axis=1)
    with np.errstate(divide="ignore"):
        frame_db = np.where(mean_sq > 0.0, 10.0 * np.log10(np.maximum(mean_sq, 1e-24)), FLOOR_DB)
    frame_db = np.maximum(frame_db, FLOOR_DB)
    live = frame_db[frame_db > FLOOR_DB]
    if live.size == 0:
        return FLOOR_DB
    return float(np.percentile(live, 10))


def _fit_length(
    arr: np.ndarray[Any, np.dtype[np.float32]],
    target_len: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    n = int(arr.shape[0])
    if n == target_len:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if n > target_len:
        return np.ascontiguousarray(arr[:target_len], dtype=np.float32)
    return np.ascontiguousarray(np.pad(arr, (0, target_len - n), mode="constant"), dtype=np.float32)


def cleanup_preview_dir(
    preview_dir: Path,
    *,
    max_files: int = DEFAULT_MAX_PREVIEW_FILES,
    max_age_s: float = DEFAULT_PREVIEW_MAX_AGE_S,
) -> int:
    """Prune old or excess managed preview WAV files safely inside ``preview_dir``."""
    if not preview_dir.is_dir():
        return 0
    now = time.time()
    candidates: list[tuple[float, Path]] = []
    for entry in preview_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".wav" or not entry.name.startswith("preview_"):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, entry))

    removed = 0
    survivors: list[tuple[float, Path]] = []
    for mtime, entry in candidates:
        if max_age_s > 0.0 and (now - mtime) > max_age_s:
            try:
                entry.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
        else:
            survivors.append((mtime, entry))

    survivors.sort(key=lambda item: item[0], reverse=True)
    for _mtime, entry in survivors[max(0, max_files) :]:
        try:
            entry.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass
    return removed


def render_region_preview(
    request: RegionPreviewRequest,
    *,
    cache: WarmCoreCache | None = None,
    scorer: PerceptualQualityScorer | None = None,
) -> RegionPreviewResult:
    """Decode only ``[start_s, end_s]`` (clamped to 15 s), enhance via warm cache, and save WAV."""
    t0 = time.perf_counter()

    if not (math.isfinite(request.start_s) and math.isfinite(request.end_s)):
        raise ValueError(
            f"start_s and end_s must be finite numbers, got ({request.start_s}, {request.end_s})"
        )
    if request.start_s < 0.0:
        raise ValueError(f"start_s must be >= 0, got {request.start_s}")
    if request.end_s <= request.start_s:
        raise ValueError(
            f"end_s must be greater than start_s, got ({request.start_s}, {request.end_s})"
        )
    if not math.isfinite(request.strength) or not 0.0 <= request.strength <= 1.0:
        raise ValueError(f"strength must be finite and in [0.0, 1.0], got {request.strength}")

    source_path = Path(request.source_path).resolve()
    probe = cached_probe(source_path)
    file_duration_s = probe.samples / float(probe.sample_rate)
    if request.start_s >= file_duration_s:
        raise ValueError(
            f"start_s {request.start_s:.3f}s is at or past file end ({file_duration_s:.3f}s)"
        )

    clamped_end_s = min(
        request.end_s,
        request.start_s + MAX_PREVIEW_DURATION_S,
        file_duration_s,
    )
    if clamped_end_s <= request.start_s:
        raise ValueError("Clamped preview window is empty")

    window_buf = decode_audio_window(probe, request.start_s, clamped_end_s)
    core_id, target_sr = resolve_preview_core_id(request.profile, request.core_id)

    orig_data = (
        resample_audio(window_buf.data, window_buf.sample_rate, target_sr)
        if window_buf.sample_rate != target_sr
        else np.ascontiguousarray(window_buf.data, dtype=np.float32)
    )
    channels, n_samples = orig_data.shape

    active_cache = cache if cache is not None else get_default_warm_cache()
    enhancer, warm_hit = active_cache.get_or_load(core_id, sample_rate=target_sr)

    strength = float(request.strength)
    out_channels: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    if strength == 0.0:
        for ch_idx in range(channels):
            out_channels.append(orig_data[ch_idx].copy())
    else:
        for ch_idx in range(channels):
            if ch_idx > 0 and np.array_equal(orig_data[ch_idx], orig_data[0]):
                out_channels.append(out_channels[0].copy())
                continue
            ch_orig = np.ascontiguousarray(orig_data[ch_idx], dtype=np.float32)
            enh_res = enhancer.enhance(ch_orig, target_sr)
            ch_enh = _fit_length(enh_res.waveform, n_samples)
            if abs(strength - 1.0) < 1e-6:
                blended = ch_enh
            else:
                blended = np.ascontiguousarray(
                    ch_orig + strength * (ch_enh - ch_orig), dtype=np.float32
                )
            out_channels.append(blended)

    after_data = np.ascontiguousarray(np.vstack(out_channels), dtype=np.float32)

    before_lufs_m = measure_loudness_and_peaks(orig_data, target_sr)
    after_lufs_m = measure_loudness_and_peaks(after_data, target_sr)
    before_lufs = round(float(before_lufs_m.integrated_lufs), 2)
    after_lufs = round(float(after_lufs_m.integrated_lufs), 2)

    before_mono = (
        orig_data[0]
        if channels == 1
        else np.ascontiguousarray(np.mean(orig_data, axis=0), dtype=np.float32)
    )
    after_mono = (
        after_data[0]
        if channels == 1
        else np.ascontiguousarray(np.mean(after_data, axis=0), dtype=np.float32)
    )
    before_nf = round(_estimate_noise_floor_db(before_mono, target_sr), 2)
    after_nf = round(_estimate_noise_floor_db(after_mono, target_sr), 2)

    before_mos: float | None = None
    after_mos: float | None = None
    mos_delta: float | None = None
    mos_backend: str | None = None
    if request.include_perceptual_mos:
        active_scorer = scorer if scorer is not None else PerceptualQualityScorer()
        before_score = active_scorer.score(before_mono, target_sr)
        after_score = active_scorer.score(after_mono, target_sr)
        before_mos = before_score.mos
        after_mos = after_score.mos
        mos_delta = round(after_score.mos - before_score.mos, 3)
        mos_backend = after_score.backend

    preview_dir = (
        Path(request.output_dir).resolve()
        if request.output_dir is not None
        else (work_root() / "previews").resolve()
    )
    preview_dir.mkdir(parents=True, exist_ok=True)
    cleanup_preview_dir(preview_dir)

    key_material = (
        f"{source_path}:{request.start_s:.6f}:{clamped_end_s:.6f}:"
        f"{core_id}:{strength:.4f}:{time.time_ns()}:{os.getpid()}:{threading.get_ident()}"
    )
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    final_path = preview_dir / f"preview_{digest}.wav"
    tmp_path = preview_dir / f".preview_{digest}_{os.getpid()}_{threading.get_ident()}.tmp.wav"

    out_buf = AudioBuffer(data=after_data, sample_rate=target_sr)
    try:
        encode_audio(
            out_buf,
            tmp_path,
            output_bit_depth="pcm24",
            dither=True,
            seed_context=f"preview:{digest}",
        )
        os.replace(tmp_path, final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    actual_duration_s = round(n_samples / float(target_sr), 4)
    render_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return RegionPreviewResult(
        preview_path=final_path,
        core_id=core_id,
        profile=request.profile,
        start_s=round(request.start_s, 6),
        end_s=round(request.start_s + actual_duration_s, 6),
        duration_s=actual_duration_s,
        sample_rate=target_sr,
        channels=channels,
        strength=strength,
        warm_hit=warm_hit,
        render_ms=render_ms,
        before_lufs=before_lufs,
        after_lufs=after_lufs,
        lufs_delta=round(after_lufs - before_lufs, 2),
        before_noise_floor_db=before_nf,
        after_noise_floor_db=after_nf,
        noise_floor_delta_db=round(after_nf - before_nf, 2),
        before_mos=before_mos,
        after_mos=after_mos,
        mos_delta=mos_delta,
        mos_backend=mos_backend,
    )
