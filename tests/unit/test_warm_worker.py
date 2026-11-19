"""Unit tests for WarmCoreCache and sub-second windowed region previews."""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from hawavoclean.enhancement.protocol import EnhancementResult, EnhancerMetadata
from hawavoclean.guard.perceptual_mos import PerceptualQualityScorer
from hawavoclean.runtime import MEMORY_LIMIT_ENV_VAR
from hawavoclean.server.warm_worker import (
    MAX_PREVIEW_DURATION_S,
    RegionPreviewRequest,
    WarmCoreCache,
    cleanup_preview_dir,
    render_region_preview,
)


class _DummyEnhancer:
    """Fast deterministic enhancer stub implementing the Enhancer protocol."""

    def __init__(
        self,
        core_id: str = "dummy-core",
        sample_rate: int = 48000,
        phase_coherent: bool = True,
        *,
        gain: float = 0.5,
    ) -> None:
        self._meta = EnhancerMetadata(
            core_id=core_id,
            version="1.0",
            algorithm="dummy",
            sample_rate=sample_rate,
            phase_coherent=phase_coherent,
        )
        self._gain = gain
        self.warmup_calls = 0
        self.enhance_calls = 0

    @property
    def metadata(self) -> EnhancerMetadata:
        return self._meta

    def warmup(self) -> None:
        self.warmup_calls += 1

    def enhance(
        self,
        waveform: np.ndarray[Any, np.dtype[np.float32]],
        sample_rate: int,
    ) -> EnhancementResult:
        self.enhance_calls += 1
        out = np.ascontiguousarray(waveform * np.float32(self._gain), dtype=np.float32)
        return EnhancementResult(
            waveform=out,
            sample_rate=sample_rate,
            model_runtime_ms=0.1,
            input_samples=int(waveform.shape[0]),
            output_samples=int(out.shape[0]),
        )


def _write_noisy_speech_wav(
    path: Path,
    *,
    duration_s: float = 2.0,
    sample_rate: int = 48000,
    channels: int = 1,
) -> Path:
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    rng = np.random.default_rng(42)
    # Speech-like modulated tones + steady background noise.
    env = np.where((t % 0.6) < 0.35, 1.0, 0.0)
    speech = (
        0.25 * env * (np.sin(2.0 * math.pi * 220.0 * t) + 0.5 * np.sin(2.0 * math.pi * 660.0 * t))
    )
    noise = 0.02 * rng.standard_normal(n)
    sig = np.clip(speech + noise, -0.95, 0.95).astype(np.float32)
    if channels == 1:
        sf.write(str(path), sig, sample_rate, subtype="PCM_24")
    else:
        data = np.column_stack([sig for _ in range(channels)])
        sf.write(str(path), data, sample_rate, subtype="PCM_24")
    return path


def test_warm_core_cache_lru_hits_and_eviction() -> None:
    created: list[str] = []

    def factory(core_id: str, sr: int) -> _DummyEnhancer:
        created.append(core_id)
        e = _DummyEnhancer(core_id=core_id, sample_rate=sr)
        e.warmup()
        return e

    cache = WarmCoreCache(max_cores=2, idle_ttl_s=300.0, core_factory=factory)

    core_a1, hit_a1 = cache.get_or_load("core-a")
    core_a2, hit_a2 = cache.get_or_load("core-a")
    assert hit_a1 is False
    assert hit_a2 is True
    assert core_a1 is core_a2
    assert created == ["core-a"]

    _core_b, hit_b = cache.get_or_load("core-b")
    assert hit_b is False
    # Touch core-a so core-b becomes LRU.
    _, hit_a3 = cache.get_or_load("core-a")
    assert hit_a3 is True

    # Loading core-c evicts core-b, keeping core-a warm.
    _core_c, hit_c = cache.get_or_load("core-c")
    assert hit_c is False
    _, hit_a4 = cache.get_or_load("core-a")
    assert hit_a4 is True

    stats = cache.stats()
    assert stats["cached_count"] == 2
    assert stats["hits"] == 3
    assert stats["misses"] == 3
    assert stats["evictions"] >= 1
    assert {c["core_id"] for c in stats["cores"]} == {"core-a", "core-c"}

    cleared = cache.clear()
    assert cleared == 2
    assert cache.stats()["cached_count"] == 0


def test_warm_core_cache_idle_ttl_eviction() -> None:
    now = [100.0]

    def factory(core_id: str, sr: int) -> _DummyEnhancer:
        return _DummyEnhancer(core_id=core_id, sample_rate=sr)

    cache = WarmCoreCache(
        max_cores=2,
        idle_ttl_s=60.0,
        core_factory=factory,
        clock=lambda: now[0],
    )

    _, hit1 = cache.get_or_load("core-a")
    assert hit1 is False

    now[0] = 159.0
    assert cache.evict_expired() == []
    _, hit2 = cache.get_or_load("core-a")
    assert hit2 is True

    now[0] = 220.0  # 61 s idle -> expired
    evicted = cache.evict_expired()
    assert evicted == ["core-a"]
    assert cache.stats()["ttl_evictions"] == 1
    assert cache.stats()["cached_count"] == 0


def test_warm_core_cache_memory_cap_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    rss_bytes = [500 * 1024 * 1024]  # 500 MiB

    def factory(core_id: str, sr: int) -> _DummyEnhancer:
        return _DummyEnhancer(core_id=core_id, sample_rate=sr)

    monkeypatch.setenv(MEMORY_LIMIT_ENV_VAR, "1024")
    cache = WarmCoreCache(
        max_cores=3,
        idle_ttl_s=300.0,
        core_factory=factory,
        rss_bytes_fn=lambda: rss_bytes[0],
    )
    assert cache.stats()["memory_limit_mb"] == 1024

    cache.get_or_load("core-a")
    cache.get_or_load("core-b")
    assert cache.stats()["cached_count"] == 2

    # Simulate RSS exceeding worker_memory_limit_mb.
    rss_bytes[0] = 1500 * 1024 * 1024
    evicted = cache.evict_expired()
    assert set(evicted) == {"core-a", "core-b"}
    assert cache.stats()["memory_evictions"] == 2


def test_warm_core_cache_thread_safety() -> None:
    def factory(core_id: str, sr: int) -> _DummyEnhancer:
        return _DummyEnhancer(core_id=core_id, sample_rate=sr)

    cache = WarmCoreCache(max_cores=2, idle_ttl_s=300.0, core_factory=factory)
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for step in range(25):
                cid = f"core-{(idx + step) % 3}"
                enh, _ = cache.get_or_load(cid)
                assert enh.metadata.core_id == cid
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert cache.stats()["cached_count"] <= 2


def test_warm_core_cache_validation() -> None:
    with pytest.raises(ValueError, match="max_cores"):
        WarmCoreCache(max_cores=0)
    with pytest.raises(ValueError, match="idle_ttl_s"):
        WarmCoreCache(idle_ttl_s=0.0)
    with pytest.raises(ValueError, match="memory_limit_mb"):
        WarmCoreCache(memory_limit_mb=0)

    cache = WarmCoreCache()
    with pytest.raises(ValueError, match="core_id"):
        cache.get_or_load("")
    with pytest.raises(ValueError, match="sample_rate"):
        cache.get_or_load("wiener-dd-48k-v1", sample_rate=0)


def test_render_region_preview_warm_hit_metrics_and_strength(tmp_path: Path) -> None:
    src = _write_noisy_speech_wav(tmp_path / "noisy.wav", duration_s=3.0, sample_rate=48000)
    preview_dir = tmp_path / "previews"
    cache = WarmCoreCache(max_cores=2)

    req1 = RegionPreviewRequest(
        source_path=src,
        start_s=0.5,
        end_s=2.0,
        profile="production",
        strength=1.0,
        include_perceptual_mos=True,
        output_dir=preview_dir,
    )
    res1 = render_region_preview(
        req1,
        cache=cache,
        scorer=PerceptualQualityScorer(backend="acoustic_proxy"),
    )
    assert res1.warm_hit is False
    assert res1.core_id == "wiener-dd-48k-v1"
    assert res1.preview_path.is_file()
    assert res1.preview_path.parent == preview_dir.resolve()
    assert 1.45 <= res1.duration_s <= 1.55
    assert res1.sample_rate == 48000
    assert res1.channels == 1
    assert res1.render_ms >= 0.0
    assert math.isfinite(res1.before_lufs) and math.isfinite(res1.after_lufs)
    assert math.isfinite(res1.before_noise_floor_db) and math.isfinite(res1.after_noise_floor_db)
    assert res1.before_mos is not None and res1.after_mos is not None
    assert res1.mos_delta == round(res1.after_mos - res1.before_mos, 3)
    assert res1.mos_backend == "acoustic_proxy"

    # Second preview hits the warm cache.
    req2 = RegionPreviewRequest(
        source_path=src,
        start_s=0.5,
        end_s=2.0,
        profile="production",
        strength=0.5,
        include_perceptual_mos=False,
        output_dir=preview_dir,
    )
    res2 = render_region_preview(req2, cache=cache)
    assert res2.warm_hit is True
    assert res2.before_mos is None
    assert res2.to_dict()["warm_hit"] is True

    # Strength=0.0 preserves original signal cleanly.
    req0 = RegionPreviewRequest(
        source_path=src,
        start_s=0.5,
        end_s=1.5,
        profile="production",
        strength=0.0,
        output_dir=preview_dir,
    )
    res0 = render_region_preview(req0, cache=cache)
    assert res0.lufs_delta == 0.0
    assert res0.noise_floor_delta_db == 0.0


def test_render_region_preview_clamps_window_to_15s(tmp_path: Path) -> None:
    src = _write_noisy_speech_wav(tmp_path / "long.wav", duration_s=18.0, sample_rate=16000)
    cache = WarmCoreCache(
        core_factory=lambda cid, sr: _DummyEnhancer(core_id=cid, sample_rate=sr, gain=0.8)
    )
    req = RegionPreviewRequest(
        source_path=src,
        start_s=0.5,
        end_s=17.5,  # 17s requested -> must clamp to 15.0s
        core_id="wiener-dd-48k-v1",
        output_dir=tmp_path / "previews",
    )
    res = render_region_preview(req, cache=cache)
    assert res.duration_s <= MAX_PREVIEW_DURATION_S + 0.01
    assert res.duration_s >= MAX_PREVIEW_DURATION_S - 0.05


def test_render_region_preview_validation_and_cleanup(tmp_path: Path) -> None:
    src = _write_noisy_speech_wav(tmp_path / "short.wav", duration_s=1.0)
    with pytest.raises(ValueError, match="start_s"):
        render_region_preview(RegionPreviewRequest(source_path=src, start_s=-0.5, end_s=0.5))
    with pytest.raises(ValueError, match="end_s"):
        render_region_preview(RegionPreviewRequest(source_path=src, start_s=0.5, end_s=0.5))
    with pytest.raises(ValueError, match="strength"):
        render_region_preview(
            RegionPreviewRequest(source_path=src, start_s=0.0, end_s=0.5, strength=1.5)
        )
    with pytest.raises(ValueError, match="past file end"):
        render_region_preview(RegionPreviewRequest(source_path=src, start_s=2.0, end_s=3.0))
    with pytest.raises(ValueError, match="Unknown preview profile"):
        render_region_preview(
            RegionPreviewRequest(source_path=src, start_s=0.0, end_s=0.5, profile="no_such_profile")
        )

    # Cleanup prunes excess preview WAV files only.
    pdir = tmp_path / "managed_previews"
    pdir.mkdir()
    keep_non_preview = pdir / "other.wav"
    keep_non_preview.write_bytes(b"keep")
    import time

    base_ts = time.time() - 10.0
    for idx in range(5):
        p = pdir / f"preview_{idx:04d}.wav"
        p.write_bytes(b"wav")
        os.utime(p, (base_ts + idx, base_ts + idx))
    removed = cleanup_preview_dir(pdir, max_files=2, max_age_s=0.0)
    assert removed == 3
    assert keep_non_preview.is_file()
    remaining = sorted(p.name for p in pdir.glob("preview_*.wav"))
    assert remaining == ["preview_0003.wav", "preview_0004.wav"]
    assert cleanup_preview_dir(tmp_path / "does_not_exist") == 0

    # Expired preview file pruned by max_age_s.
    old_p = pdir / "preview_old.wav"
    old_p.write_bytes(b"old")
    os.utime(old_p, (10.0, 10.0))
    assert cleanup_preview_dir(pdir, max_files=10, max_age_s=60.0) == 1
    assert not old_p.exists()


def test_warm_worker_edge_branches(tmp_path: Path) -> None:
    from hawavoclean.server.warm_worker import (
        _estimate_noise_floor_db,
        _fit_length,
        get_default_warm_cache,
        resolve_preview_core_id,
    )

    assert get_default_warm_cache() is get_default_warm_cache()
    cid, sr = resolve_preview_core_id("wiener-dd-48k-v1")
    assert cid == "wiener-dd-48k-v1" and sr == 48000

    assert _estimate_noise_floor_db(np.empty(0, dtype=np.float32), 48000) == -120.0
    assert _estimate_noise_floor_db(np.zeros(100, dtype=np.float32), 48000) == -120.0
    assert _estimate_noise_floor_db(np.full(100, 0.1, dtype=np.float32), 48000) > -30.0
    assert _estimate_noise_floor_db(np.zeros(4800, dtype=np.float32), 48000) == -120.0

    assert _fit_length(np.ones(5, dtype=np.float32), 8).shape == (8,)
    assert _fit_length(np.ones(10, dtype=np.float32), 8).shape == (8,)

    # Explicit memory_limit_mb and sample_rate change replace stale entry.
    rss = [2000 * 1024 * 1024]
    cache = WarmCoreCache(
        max_cores=2,
        memory_limit_mb=1024,
        core_factory=lambda c, r: _DummyEnhancer(core_id=c, sample_rate=r),
        rss_bytes_fn=lambda: rss[0],
    )
    e1, h1 = cache.get_or_load("core-x", sample_rate=48000)
    assert h1 is False
    e2, h2 = cache.get_or_load("core-x", sample_rate=16000)
    assert h2 is False and e2 is not e1
    # Adding second core while over memory cap evicts older core.
    cache.get_or_load("core-y", sample_rate=48000)
    assert cache.stats()["memory_evictions"] >= 1

    # Stereo dual-mono preview exercises channel deduplication + non-finite check.
    stereo_src = _write_noisy_speech_wav(
        tmp_path / "stereo_dup.wav", duration_s=1.0, sample_rate=48000, channels=2
    )
    res_st = render_region_preview(
        RegionPreviewRequest(
            source_path=stereo_src,
            start_s=0.0,
            end_s=0.6,
            core_id="wiener-dd-48k-v1",
            output_dir=tmp_path / "prev_st",
        ),
        cache=WarmCoreCache(core_factory=lambda c, r: _DummyEnhancer(core_id=c, sample_rate=r)),
    )
    assert res_st.channels == 2
    with pytest.raises(ValueError, match="finite"):
        render_region_preview(
            RegionPreviewRequest(source_path=stereo_src, start_s=float("nan"), end_s=0.5)
        )


def test_preview_profile_path_traversal_and_arbitrary_toml_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import ValidationError

    from hawavoclean.server.contracts import RegionPreviewRequestV1
    from hawavoclean.server.warm_worker import resolve_preview_core_id

    # Create a real TOML outside config_dir to prove traversal cannot load it.
    outside_toml = tmp_path / "evil.toml"
    outside_toml.write_text("[enhancement]\ncore_id = 'wiener-dd-48k-v1'\n", encoding="utf-8")
    fake_cfg_dir = tmp_path / "configs"
    fake_cfg_dir.mkdir()
    monkeypatch.setenv("HAWAVOCLEAN_CONFIG_DIR", str(fake_cfg_dir))

    for hostile_profile in (
        "../../etc/passwd",
        "../evil",
        str(outside_toml.with_suffix("")),
        "arbitrary_profile",
    ):
        with pytest.raises(ValidationError):
            RegionPreviewRequestV1.model_validate(
                {"path": "/tmp/a.wav", "startS": 0.0, "endS": 1.0, "profile": hostile_profile}
            )
        with pytest.raises(ValueError, match="Unknown preview profile"):
            resolve_preview_core_id(hostile_profile)

    # Even for a built-in profile name, a symlink escaping config_dir is rejected.
    (fake_cfg_dir / "production.toml").symlink_to(outside_toml)
    with pytest.raises(ValueError, match="Unknown preview profile"):
        resolve_preview_core_id("production")


def test_warm_worker_additional_branches_and_cleanup_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from hawavoclean.server import warm_worker

    # 1. TTL eviction inside get_or_load (both pre-load and post-load) + post-load memory cap + empty clear.
    now = [100.0]
    rss = [100 * 1024 * 1024]

    def adv_factory(core_id: str, sr: int) -> _DummyEnhancer:
        if core_id == "core-slow":
            now[0] += 100.0  # expire previously cached core while loading
            rss[0] = 2000 * 1024 * 1024
        elif core_id == "core-mem":
            rss[0] = 2000 * 1024 * 1024
        return _DummyEnhancer(core_id=core_id, sample_rate=sr)

    cache = WarmCoreCache(
        max_cores=3,
        idle_ttl_s=50.0,
        memory_limit_mb=512,
        core_factory=adv_factory,
        clock=lambda: now[0],
        rss_bytes_fn=lambda: rss[0],
    )
    assert cache.clear() == 0

    cache.get_or_load("core-a", sample_rate=48000)
    now[0] += 60.0  # triggers pre-load TTL eviction inside get_or_load
    cache.get_or_load("core-b", sample_rate=48000)
    assert cache.stats()["ttl_evictions"] == 1

    # Post-load TTL eviction while constructing core-slow
    cache.get_or_load("core-slow", sample_rate=48000)
    assert cache.stats()["ttl_evictions"] == 2

    # Post-load memory cap eviction while keeping the newly loaded core
    rss[0] = 100 * 1024 * 1024
    cache.clear()
    cache.get_or_load("core-keep", sample_rate=48000)
    cache.get_or_load("core-mem", sample_rate=48000)
    assert cache.stats()["memory_evictions"] >= 1
    assert [c["core_id"] for c in cache.stats()["cores"]] == ["core-mem"]

    # 2. cleanup_preview_dir prunes both preview_* and spectral_* WAVs, honors protect, skips symlinks.
    pdir = tmp_path / "mixed_previews"
    pdir.mkdir()
    base_ts = time.time() - 10.0
    protected = pdir / "spectral_keepme.wav"
    protected.write_bytes(b"RIFF")
    os.utime(protected, (base_ts - 5.0, base_ts - 5.0))
    for i in range(4):
        p = pdir / f"preview_{i}.wav"
        s = pdir / f"spectral_{i}.wav"
        p.write_bytes(b"RIFF")
        s.write_bytes(b"RIFF")
        os.utime(p, (base_ts + i, base_ts + i))
        os.utime(s, (base_ts + i + 0.5, base_ts + i + 0.5))

    sym = pdir / "preview_symlink.wav"
    try:
        sym.symlink_to(protected)
    except OSError:
        sym = None  # type: ignore[assignment]

    removed = cleanup_preview_dir(pdir, max_files=3, max_age_s=0.0, protect=protected)
    assert removed == 6
    assert protected.is_file()
    remaining_regular = [
        e.name
        for e in pdir.iterdir()
        if e.is_file() and not e.is_symlink() and e.name.startswith(("preview_", "spectral_"))
    ]
    assert len(remaining_regular) == 3
    assert "spectral_keepme.wav" in remaining_regular

    # 3. Failed encode_audio during render_region_preview cleans up temp file.
    src = _write_noisy_speech_wav(tmp_path / "fail_enc.wav", duration_s=1.0)
    fail_dir = tmp_path / "fail_previews"

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(warm_worker, "encode_audio", _boom)
    with pytest.raises(OSError, match="disk full"):
        render_region_preview(
            RegionPreviewRequest(
                source_path=src,
                start_s=0.0,
                end_s=0.5,
                core_id="wiener-dd-48k-v1",
                output_dir=fail_dir,
            ),
            cache=WarmCoreCache(core_factory=lambda c, r: _DummyEnhancer(core_id=c, sample_rate=r)),
        )
    assert not list(fail_dir.glob("*.tmp.wav"))
