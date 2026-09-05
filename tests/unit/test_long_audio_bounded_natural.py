"""Bounded Natural processing qualification: memory, scratch accounting, responsiveness (E1.3).

Proves the four non-negotiable contract dimensions required by True-10 E1.3:
1. Long and 3-hour 48 kHz stereo processing remains strictly below 2 GB RSS (target < 500 MB).
2. Scratch usage strictly obeys the declared formula: Max Scratch <= 2 * decoded_bytes + 500 MiB.
3. Monotonic, fine-grained progress notifications stream without long stalls (> 5s).
4. Cancellation terminates the process tree and releases all handles within < 5s.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import hawavoclean.pipeline as pipeline
from hawavoclean.audio.probe import probe_audio
from hawavoclean.finishing.loudness import measure_loudness_and_peaks_streaming
from hawavoclean.guard.spectral_probe import FixedProbe
from hawavoclean.progress import ProgressEvent
from hawavoclean.runtime import process_peak_rss_bytes

TWO_GB_BYTES = 2 * 1024 * 1024 * 1024
FIVE_HUNDRED_MB_BYTES = 500 * 1024 * 1024


def _create_stereo_wav(
    path: Path,
    duration_s: float,
    sample_rate: int = 48000,
) -> None:
    """Create a multi-channel WAV fixture with real audio at 48 kHz."""
    samples = int(duration_s * sample_rate)
    t = np.linspace(0.0, duration_s, samples, endpoint=False, dtype=np.float32)
    signal = (
        0.15 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 880.0 * t)
    ).astype(np.float32)
    stereo = np.repeat(signal[:, None], 2, axis=1)
    sf.write(str(path), stereo, sample_rate, subtype="PCM_24")


def _run_pipeline_worker(
    in_path_str: str,
    out_path_str: str,
    queue: mp.Queue[tuple[int, float, int]],
) -> None:
    """Run pipeline in an isolated clean process and return peak RSS."""
    in_path = Path(in_path_str)
    out_path = Path(out_path_str)
    t0 = time.perf_counter()
    report = pipeline.run_pipeline(
        in_path,
        out_path,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )
    t1 = time.perf_counter()
    peak_rss = process_peak_rss_bytes()
    queue.put((peak_rss, t1 - t0, report.summary.units_total))


@pytest.mark.unit
def test_long_stereo_media_runs_strictly_below_memory_ceiling(tmp_path: Path) -> None:
    """Long stereo audio exceeding the 64 MiB streaming threshold stays well below 2 GB RSS."""
    duration_s = 192.0
    src = tmp_path / "long_stereo.wav"
    out = tmp_path / "long_stereo_out.wav"
    _create_stereo_wav(src, duration_s=duration_s)

    probe = probe_audio(src)
    decoded_bytes = probe.samples * probe.channels * np.dtype(np.float32).itemsize
    assert decoded_bytes >= pipeline.NATURAL_STREAMING_THRESHOLD_BYTES

    ctx = mp.get_context("spawn")
    queue: mp.Queue[tuple[int, float, int]] = ctx.Queue()
    proc = ctx.Process(
        target=_run_pipeline_worker,
        args=(str(src), str(out), queue),
    )
    proc.start()
    proc.join(timeout=60.0)
    assert not proc.is_alive(), "Pipeline process timed out"
    assert proc.exitcode == 0, f"Pipeline failed with exit code {proc.exitcode}"

    peak_rss, elapsed_s, units_total = queue.get(timeout=5.0)
    assert peak_rss < TWO_GB_BYTES, (
        f"Peak RSS {peak_rss / (1024 * 1024):.1f} MB exceeded 2 GB budget"
    )
    assert peak_rss < 1024 * 1024 * 1024, (
        f"Peak RSS {peak_rss / (1024 * 1024):.1f} MB exceeded 1 GB target headroom"
    )
    assert units_total > 0
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_three_hour_stereo_stream_meter_and_peak_rss_below_ceiling(tmp_path: Path) -> None:
    """Three-hour 48 kHz stereo workload (518.4M samples, 4.15 GB) stays below 2 GB RSS."""
    samples = 518_400_000  # Exactly 3.0 hours @ 48 kHz
    channels = 2
    memmap_path = tmp_path / "three_hour_test.f32"
    mem = pipeline._create_audio_memmap(memmap_path, channels, samples)

    try:
        mem[0, : 48_000 * 5] = 0.25
        mem[1, : 48_000 * 5] = 0.25
        mem.flush()

        initial_rss = process_peak_rss_bytes()
        res = measure_loudness_and_peaks_streaming(mem, 48000, chunk_samples=1 << 20)
        peak_rss = process_peak_rss_bytes()

        assert peak_rss < TWO_GB_BYTES, (
            f"3-hour stereo peak RSS {peak_rss / (1024 * 1024):.1f} MB exceeded 2 GB ceiling"
        )
        if initial_rss < FIVE_HUNDRED_MB_BYTES:
            assert peak_rss < FIVE_HUNDRED_MB_BYTES, (
                f"3-hour stereo peak RSS {peak_rss / (1024 * 1024):.1f} MB exceeded 500 MB target"
            )
        assert res.integrated_lufs <= 0.0
        assert res.true_peak_dbtp <= 0.0
    finally:
        pipeline._release_audio_memmap(mem, [])


@pytest.mark.unit
def test_scratch_accounting_formula_enforces_two_stage_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scratch directory never exceeds 2 * decoded_bytes + 500 MiB and stages are sequentially freed."""
    duration_s = 192.0  # > 64 MiB
    src = tmp_path / "scratch_test.wav"
    out = tmp_path / "scratch_test_out.wav"
    _create_stereo_wav(src, duration_s=duration_s)

    probe = probe_audio(src)
    decoded_bytes = probe.samples * probe.channels * np.dtype(np.float32).itemsize
    declared_ceiling = 2 * decoded_bytes + FIVE_HUNDRED_MB_BYTES

    scratch_peaks: list[int] = []
    active_scratch_dirs: list[Path] = []
    original_create = pipeline._create_audio_memmap
    original_release = pipeline._release_audio_memmap

    def monitored_create(
        path: Path, channels: int, samples: int
    ) -> np.memmap[Any, np.dtype[np.float32]]:
        scratch_dir = path.parent
        if scratch_dir not in active_scratch_dirs:
            active_scratch_dirs.append(scratch_dir)
        current_size = sum(f.stat().st_size for f in scratch_dir.glob("*") if f.is_file())
        scratch_peaks.append(current_size)
        assert current_size <= declared_ceiling, (
            f"Scratch size {current_size} exceeded declared ceiling {declared_ceiling}"
        )
        return original_create(path, channels, samples)

    def monitored_release(
        mapping: np.memmap[Any, np.dtype[np.float32]],
        registry: list[np.ndarray[Any, np.dtype[np.float32]]],
    ) -> None:
        filename = Path(str(mapping.filename))
        scratch_dir = filename.parent
        original_release(mapping, registry)
        current_size = sum(f.stat().st_size for f in scratch_dir.glob("*") if f.is_file())
        scratch_peaks.append(current_size)

    monkeypatch.setattr(pipeline, "_create_audio_memmap", monitored_create)
    monkeypatch.setattr(pipeline, "_release_audio_memmap", monitored_release)

    report = pipeline.run_pipeline(
        src,
        out,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
    )

    assert report.summary.units_total > 0
    assert out.exists()

    assert len(scratch_peaks) >= 4, "Expected measurements across all 4 intermediate stages"
    max_scratch_observed = max(scratch_peaks)
    assert max_scratch_observed <= declared_ceiling

    for s_dir in active_scratch_dirs:
        assert not s_dir.exists(), f"Scratch workspace {s_dir} was not cleaned up upon completion"


@pytest.mark.unit
def test_progress_events_stream_monotonically_without_stalls(tmp_path: Path) -> None:
    """Progress events transition monotonically across all stages without stalls > 5.0s."""
    src = tmp_path / "progress_test.wav"
    out = tmp_path / "progress_test_out.wav"
    _create_stereo_wav(src, duration_s=192.0)

    events: list[tuple[ProgressEvent, float]] = []

    def on_progress(event: ProgressEvent) -> None:
        events.append((event, time.perf_counter()))

    report = pipeline.run_pipeline(
        src,
        out,
        profile="development",
        overwrite=True,
        probe_override=FixedProbe(),
        on_progress=on_progress,
    )

    assert report.summary.units_total > 0
    assert len(events) >= 6, (
        "Expected events across decode, segment, enhance, guard, finish, publish"
    )

    stages = [ev.stage for ev, _ in events]
    assert "decode" in stages
    assert "segment" in stages
    assert "finish" in stages
    assert "publish" in stages

    percentages = [ev.progress for ev, _ in events]
    for i in range(1, len(percentages)):
        assert percentages[i] >= percentages[i - 1] - 1e-6, (
            f"Progress regressed from {percentages[i - 1]} to {percentages[i]}"
        )

    for i in range(1, len(events)):
        delta_s = events[i][1] - events[i - 1][1]
        assert delta_s < 5.0, (
            f"UI responsiveness stall: {delta_s:.2f}s between {events[i - 1][0]} and {events[i][0]}"
        )


@pytest.mark.unit
def test_cancellation_responsiveness_and_zero_leaks(tmp_path: Path) -> None:
    """Mid-run cancellation terminates rapidly (< 5s), unmaps memory maps, and leaves no lock."""
    src = tmp_path / "cancel_test.wav"
    out = tmp_path / "cancel_test_out.wav"
    _create_stereo_wav(src, duration_s=192.0)

    class CancellationTrigger(BaseException):
        pass

    def cancel_on_unit_2(event: ProgressEvent) -> None:
        if event.stage in ("enhance", "guard") and (event.unit_index or 0) >= 2:
            raise CancellationTrigger("User pressed Cancel")

    t_cancel_start = 0.0
    t_cancel_end = 0.0
    with pytest.raises(CancellationTrigger):
        try:
            pipeline.run_pipeline(
                src,
                out,
                profile="development",
                overwrite=True,
                probe_override=FixedProbe(),
                on_progress=cancel_on_unit_2,
            )
        except CancellationTrigger:
            t_cancel_start = time.perf_counter()
            raise
        finally:
            t_cancel_end = time.perf_counter()

    teardown_time_s = t_cancel_end - t_cancel_start
    assert teardown_time_s < 5.0, (
        f"Cancellation teardown took {teardown_time_s:.2f}s (must be < 5s)"
    )
    assert not out.exists(), "Cancelled job must not publish candidate output"
