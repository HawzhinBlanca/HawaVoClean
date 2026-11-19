"""Hybrid separation tests: RoFormer routing, device auto, chunking, time_range & wet/dry blend."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import platform
import sys
import types
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

import hawavoclean.audio_edit as edits
import hawavoclean.separation.roformer as roformer
import hawavoclean.separation.runtime as runtime
import hawavoclean.separation.worker as worker
from hawavoclean.audio_edit import EditRequest, run_audio_edit
from hawavoclean.hashing import hash_file
from hawavoclean.separation import (
    MAX_DURATION_S,
    MAX_FILE_SIZE_BYTES,
    ROFORMER_CONFIG_FILENAME,
    ROFORMER_ENGINE_ID,
    ROFORMER_MODEL_FILENAME,
    RuntimeStatus,
    SeparationCancelledError,
    SeparationError,
    classify_vocal_music_prompt,
    probe_any_separation_runtime,
    probe_roformer_runtime,
    probe_runtime,
    probe_separation_capabilities,
    resolve_separation_device,
    separate_audio,
    separate_roformer,
)
from hawavoclean.server.contracts import SeparationStrategyV1


@pytest.fixture
def isolated_sep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_LOCAL_ROOT", tmp_path / "local-ai")
    monkeypatch.setattr(roformer, "_LOCAL_ROOT", tmp_path / "local-ai")
    monkeypatch.delenv(runtime.PYTHON_ENV, raising=False)
    monkeypatch.delenv(runtime.ROOT_ENV, raising=False)
    monkeypatch.delenv(roformer.ROFORMER_PYTHON_ENV, raising=False)
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    monkeypatch.setenv(roformer.ROFORMER_MODEL_DIR_ENV, str(tmp_path / "no-models"))


def _make_wave(
    path: Path,
    *,
    duration_s: float = 2.0,
    sr: int = 8000,
    channels: int = 2,
    subtype: str = "FLOAT",
) -> Path:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    if channels == 1:
        data = (0.25 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 660 * t)).reshape(
            -1, 1
        )
    else:
        ch0 = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 660 * t)
        ch1 = 0.20 * np.sin(2 * np.pi * 330 * t) + 0.08 * np.sin(2 * np.pi * 880 * t)
        data = np.stack([ch0, ch1], axis=-1)
    sf.write(path, data.astype(np.float32), sr, subtype=subtype)
    return path


# ---------------------------------------------------------------------------
# 1. Prompt router (classify_vocal_music_prompt)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "operation", "expected"),
    [
        ("speech", "extract", "vocals"),
        ("human voice", "extract", "vocals"),
        ("Vocals", "extract", "vocals"),
        ("lead vocals", "extract", "vocals"),
        ("clean dialogue", "extract", "vocals"),
        ("Kurdish speech", "extract", "vocals"),
        ("spoken voiceover", "extract", "vocals"),
        ("speech from background music", "extract", "vocals"),
        ("vocals from music", "extract", "vocals"),
        ("vocals", "remove", "instrumental"),
        ("speech", "remove", "instrumental"),
        ("dialogue from music", "remove", "instrumental"),
        ("background music", "remove", "vocals"),
        ("music", "remove", "vocals"),
        ("BGM", "remove", "vocals"),
        ("instrumental", "remove", "vocals"),
        ("backing track", "remove", "vocals"),
        ("music bed", "remove", "vocals"),
        ("music bleed", "remove", "vocals"),
        ("non-vocal", "remove", "vocals"),
        ("music from dialogue", "remove", "vocals"),
        ("music", "extract", "instrumental"),
        ("background music", "extract", "instrumental"),
        ("accompaniment", "extract", "instrumental"),
        ("soundtrack", "extract", "instrumental"),
        ("backing track", "extract", "instrumental"),
        ("music from vocals", "extract", "instrumental"),
        # Open-vocabulary SFX & ambiguous phrases MUST route to AudioGhost (None).
        ("dog barking", "remove", None),
        ("siren", "remove", None),
        ("flute", "extract", None),
        ("piano", "remove", None),
        ("guitar", "extract", None),
        ("drums", "remove", None),
        ("applause", "remove", None),
        ("bird singing", "remove", None),
        ("music box", "remove", None),
        ("background noise", "remove", None),
        ("crowd talking", "remove", None),
        ("speech and siren", "extract", None),
        ("speech from dog barking", "extract", None),
        ("music from traffic", "remove", None),
        ("from", "extract", None),
        ("non", "remove", None),
        ("vocal music", "extract", None),
        ("   ", "extract", None),
        ("", "remove", None),
    ],
)
def test_classify_vocal_music_prompt(
    prompt: str,
    operation: Literal["extract", "remove"],
    expected: Literal["vocals", "instrumental"] | None,
) -> None:
    assert classify_vocal_music_prompt(prompt, operation) == expected


def test_classify_vocal_music_prompt_rejects_bad_inputs() -> None:
    assert classify_vocal_music_prompt("speech", "invalid") is None  # type: ignore[arg-type]
    assert classify_vocal_music_prompt(123, "extract") is None  # type: ignore[arg-type]
    assert roformer._is_vocal_tokens([]) is False
    assert roformer._is_music_tokens([]) is False


# ---------------------------------------------------------------------------
# 2. Device resolution & hybrid runtime probing
# ---------------------------------------------------------------------------


def test_resolve_separation_device_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_separation_device("cpu") == "cpu"
    assert resolve_separation_device("mps") == "mps"
    assert resolve_separation_device("cuda") == "cuda"
    with pytest.raises(SeparationError, match="supported local separation provider"):
        resolve_separation_device("tpu")

    fake_mps = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_mps)
    assert resolve_separation_device("auto") == "mps"

    fake_cuda = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_cuda)
    assert resolve_separation_device("auto") == "cuda"

    fake_cpu = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_cpu)
    assert resolve_separation_device("auto") == "cpu"

    def no_torch(_name: str) -> Any:
        raise ImportError("no torch")

    monkeypatch.setattr(importlib, "import_module", no_torch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert resolve_separation_device("auto") == "mps"

    monkeypatch.setattr(sys, "platform", "linux")
    assert resolve_separation_device("auto") == "cpu"


def test_hybrid_probe_combinations(isolated_sep: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del isolated_sep
    rf_ok = RuntimeStatus(True, "roformer ready", ("mlx",))
    rf_no = RuntimeStatus(False, "roformer missing")
    ag_ok = RuntimeStatus(True, "audioghost ready", ("mps",))
    ag_no = RuntimeStatus(False, "audioghost missing")

    # Both available -> combined providers & hybrid message
    monkeypatch.setattr(roformer, "probe_roformer_runtime", lambda: rf_ok)
    monkeypatch.setattr(runtime, "probe_audioghost_runtime", lambda *_a, **_k: ag_ok)
    both = probe_runtime(device="auto")
    assert both.available is True
    assert both.providers == ("mlx", "mps")
    assert "Hybrid separation ready" in both.reason

    # Only RoFormer available -> truthful vocal/music scope
    monkeypatch.setattr(runtime, "probe_audioghost_runtime", lambda *_a, **_k: ag_no)
    rf_only = probe_runtime(device="auto")
    assert rf_only.available is True
    assert rf_only.providers == ("mlx",)
    assert "MLX RoFormer" in rf_only.reason
    assert "SAM-Audio" in rf_only.reason

    # Prompt-scoped probe: vocal/music passes, SFX fails closed truthfully
    assert probe_runtime(prompt="background music", operation="remove").available is True
    sfx = probe_runtime(prompt="dog barking", operation="remove")
    assert sfx.available is False
    assert "Open-prompt SFX" in sfx.reason

    # Explicit CUDA skips RoFormer and falls back to AudioGhost
    assert probe_runtime(device="cuda").available is False
    assert probe_separation_capabilities(device="cuda").available is False
    assert probe_separation_capabilities(device="bad").available is False

    # Prompt-scoped probe when AudioGhost IS available handles SFX
    monkeypatch.setattr(runtime, "probe_audioghost_runtime", lambda *_a, **_k: ag_ok)
    assert probe_runtime(prompt="dog barking", operation="remove").available is True

    # Neither available -> returns AudioGhost reason
    monkeypatch.setattr(roformer, "probe_roformer_runtime", lambda: rf_no)
    monkeypatch.setattr(runtime, "probe_audioghost_runtime", lambda *_a, **_k: ag_no)
    assert probe_runtime().available is False
    assert probe_runtime(prompt="dog barking", operation="remove").available is False
    assert probe_runtime(device="invalid").available is False

    # probe_separation_capabilities scans large -> base -> small
    seen_sizes: list[str] = []

    def ag_small_only(model_size: str = "small", device: str = "auto") -> RuntimeStatus:
        del device
        seen_sizes.append(model_size)
        return ag_ok if model_size == "small" else ag_no

    monkeypatch.setattr(runtime, "probe_audioghost_runtime", ag_small_only)
    cap = probe_any_separation_runtime()
    assert cap.available is True
    assert seen_sizes == ["large", "base", "small"]


# ---------------------------------------------------------------------------
# 3. RoFormer discovery, probe, offline guard & separation routing
# ---------------------------------------------------------------------------


def test_roformer_discovery_env_and_probe_branches(
    tmp_path: Path, isolated_sep: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del isolated_sep
    # Model directory discovery & relative path rejection
    monkeypatch.setenv(roformer.ROFORMER_MODEL_DIR_ENV, "relative/models")
    with pytest.raises(SeparationError, match="absolute path"):
        roformer._roformer_model_dir()

    monkeypatch.delenv(roformer.ROFORMER_MODEL_DIR_ENV)
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path / "alt"))
    assert roformer._roformer_model_dir() == tmp_path / "alt"

    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR")
    assert (
        roformer._roformer_model_dir() == Path.home() / ".cache" / "mlx-audio-separator" / "models"
    )

    # Python discovery branches
    monkeypatch.setenv(roformer.ROFORMER_PYTHON_ENV, str(tmp_path / "missing-py"))
    with pytest.raises(SeparationError, match="Python runtime is invalid"):
        roformer._roformer_python()

    fake_py = tmp_path / "roformer-py"
    fake_py.write_text("#!/bin/sh\n")
    fake_py.chmod(0o755)
    monkeypatch.setenv(roformer.ROFORMER_PYTHON_ENV, str(fake_py))
    assert roformer._roformer_python() == fake_py

    monkeypatch.delenv(roformer.ROFORMER_PYTHON_ENV)
    bench_py = roformer._LOCAL_ROOT / "bench-env" / "bin" / "python"
    bench_py.parent.mkdir(parents=True)
    bench_py.write_text("#!/bin/sh\n")
    bench_py.chmod(0o755)
    assert roformer._roformer_python() == bench_py
    bench_py.unlink()

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    assert roformer._roformer_python() == Path(sys.executable)

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(SeparationError, match="not installed"):
        roformer._roformer_python()

    # Cached file checks (missing & empty)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    with pytest.raises(SeparationError, match="not cached locally"):
        roformer._cached_roformer_files(model_dir)
    (model_dir / ROFORMER_MODEL_FILENAME).write_bytes(b"")
    (model_dir / ROFORMER_CONFIG_FILENAME).write_bytes(b"cfg")
    with pytest.raises(SeparationError, match="not cached locally"):
        roformer._cached_roformer_files(model_dir)
    (model_dir / ROFORMER_MODEL_FILENAME).write_bytes(b"ckpt")
    assert roformer._cached_roformer_files(model_dir) == (
        model_dir / ROFORMER_MODEL_FILENAME,
        model_dir / ROFORMER_CONFIG_FILENAME,
    )

    # Environment strips HF tokens and enforces offline mode
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    env = roformer._roformer_environment()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE"] == "1"
    assert "HF_TOKEN" not in env

    # probe_roformer_runtime ready, not ready, and OSError paths
    monkeypatch.setattr(
        roformer, "_run_roformer_worker", lambda *_a, **_k: {"ready": True, "provider": "mlx"}
    )
    status = probe_roformer_runtime()
    assert status.available is True and status.providers == ("mlx",)

    monkeypatch.setattr(roformer, "_run_roformer_worker", lambda *_a, **_k: {"ready": False})
    assert probe_roformer_runtime().available is False

    def boom_os(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise OSError("cannot spawn")

    monkeypatch.setattr(roformer, "_run_roformer_worker", boom_os)
    assert probe_roformer_runtime().available is False


def test_separate_roformer_and_hybrid_routing(
    tmp_path: Path, isolated_sep: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del isolated_sep
    src = _make_wave(tmp_path / "mix.wav", duration_s=0.5, sr=16000, channels=2)

    # Validation guards in separate_roformer
    with pytest.raises(SeparationError, match="outside vocal/music"):
        separate_roformer(src, tmp_path / "x.wav", prompt="dog barking", operation="remove")
    with pytest.raises(SeparationError, match="quality"):
        separate_roformer(
            src, tmp_path / "x.wav", prompt="speech", operation="extract", quality="ultra"
        )
    with pytest.raises(SeparationError, match="seed"):
        separate_roformer(src, tmp_path / "x.wav", prompt="speech", operation="extract", seed=-1)
    existing = tmp_path / "exists.wav"
    existing.write_bytes(b"keep")
    with pytest.raises(SeparationError, match="never replaces"):
        separate_roformer(src, existing, prompt="speech", operation="extract")

    # Happy path via separate_audio routing vocal/music -> RoFormer
    roformer_calls: list[dict[str, Any]] = []

    def fake_rf_worker(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        roformer_calls.append(request)
        data, rate = sf.read(request["input"], dtype="float32", always_2d=True)
        sf.write(request["output"], data * 0.5, rate, subtype="FLOAT")
        return {
            "provider": "mlx",
            "selected_stem": request["target_stem"],
            "model_sha256": "b" * 64,
        }

    monkeypatch.setattr(roformer, "_run_roformer_worker", fake_rf_worker)
    monkeypatch.setattr(
        roformer, "probe_roformer_runtime", lambda: RuntimeStatus(True, "ok", ("mlx",))
    )

    out_vocals = tmp_path / "vocals.wav"
    meta = separate_audio(
        src, out_vocals, prompt="background music", operation="remove", device="auto"
    )
    assert meta["engine"] == ROFORMER_ENGINE_ID
    assert meta["license"] == "MIT"
    assert meta["target_stem"] == "vocals"
    assert meta["provider"] == "mlx"
    assert len(roformer_calls) == 1

    # Open-prompt SFX MUST route to AudioGhost even when RoFormer is available
    ag_calls: list[dict[str, Any]] = []

    def fake_ag_worker(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        ag_calls.append(request)
        data, rate = sf.read(request["input"], dtype="float32", always_2d=True)
        sf.write(request["output"], data * 0.25, rate, subtype="FLOAT")
        return {"device": "mps", "selected_stem": "residual"}

    monkeypatch.setattr(runtime, "_run_worker", fake_ag_worker)
    out_sfx = tmp_path / "sfx.wav"
    sfx_meta = separate_audio(src, out_sfx, prompt="dog barking", operation="remove", device="auto")
    assert sfx_meta["engine"] == "audioghost-sam-audio-lite"
    assert sfx_meta["provider"] == "mps"
    assert len(ag_calls) == 1

    # Post-worker cancel & link race in separate_roformer
    with pytest.raises(SeparationCancelledError):
        separate_roformer(
            src,
            tmp_path / "cancelled.wav",
            prompt="speech",
            operation="extract",
            should_cancel=lambda: True,
        )

    def link_exists(_a: Any, _b: Any) -> None:
        raise FileExistsError("race")

    monkeypatch.setattr(os, "link", link_exists)
    with pytest.raises(SeparationError, match="already exists"):
        separate_roformer(src, tmp_path / "race.wav", prompt="speech", operation="extract")


# ---------------------------------------------------------------------------
# 4. Worker RoFormer & torch auto-device branches
# ---------------------------------------------------------------------------


def test_worker_roformer_and_torch_device_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _resolve_torch_device branches
    t_mps = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    t_cuda = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    t_cpu = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    assert worker._resolve_torch_device("auto", t_mps) == "mps"
    assert worker._resolve_torch_device("auto", t_cuda) == "cuda"
    assert worker._resolve_torch_device("auto", t_cpu) == "cpu"
    assert worker._resolve_torch_device("cpu", t_cpu) == "cpu"
    with pytest.raises(RuntimeError, match="provider_unavailable"):
        worker._resolve_torch_device("mps", t_cpu)
    with pytest.raises(RuntimeError, match="provider_unavailable"):
        worker._resolve_torch_device("cuda", t_cpu)
    with pytest.raises(RuntimeError, match="provider_unavailable"):
        worker._resolve_torch_device("bogus", t_cpu)

    # Stub mlx & mlx_audio_separator modules for deterministic run_roformer testing
    metal_flag = [True]
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.metal = types.SimpleNamespace(is_available=lambda: metal_flag[0])  # type: ignore[attr-defined]
    fake_sep_core = types.ModuleType("mlx_audio_separator.core")
    fail_mode = ["none"]
    offline_checks: list[str] = []

    class _FakeSeparator:
        def __init__(self, *, output_dir: str, output_single_stem: str, **_kwargs: Any) -> None:
            self.output_dir = output_dir
            self.output_single_stem = output_single_stem
            self.failed_files: list[tuple[str, str]] = []

        def download_file_if_not_exists(self, _url: str, _path: str) -> None:
            raise AssertionError("original download method must be overridden")

        def download_model_files(self, _fn: str) -> Any:
            raise AssertionError("original download_model_files must be overridden")

        def load_model(self, model_filename: str) -> None:
            # Exercise the overridden offline-only checks explicitly!
            self.download_model_files(model_filename)
            with pytest.raises(FileNotFoundError, match="missing_assets"):
                self.download_file_if_not_exists("https://blocked", str(tmp_path / "missing.bin"))
            with pytest.raises(FileNotFoundError, match="missing_assets"):
                self.download_model_files("other-model.ckpt")
            offline_checks.append(model_filename)

        def separate(self, input_wav: str) -> list[str]:
            if fail_mode[0] == "failed_files":
                self.failed_files = [(input_wav, "boom")]
                return []
            if fail_mode[0] == "missing_file":
                return ["does_not_exist.wav"]
            data, rate = sf.read(input_wav, dtype="float32", always_2d=True)
            out_name = f"stem_{self.output_single_stem}.wav"
            out_path = Path(self.output_dir) / out_name
            if fail_mode[0] == "bad_channels":
                sf.write(out_path, np.zeros((data.shape[0], 3), dtype=np.float32), rate)
            else:
                sf.write(out_path, data * 0.5, rate, subtype="FLOAT")
            return [out_name]

    fake_sep_core.Separator = _FakeSeparator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_audio_separator", types.ModuleType("mlx_audio_separator"))
    monkeypatch.setitem(sys.modules, "mlx_audio_separator.core", fake_sep_core)
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio_separator.separator.architectures.mdxc_separator",
        types.ModuleType("mlx_audio_separator.separator.architectures.mdxc_separator"),
    )

    model_dir = tmp_path / "rf-models"
    model_dir.mkdir()
    base_req = {
        "engine": "roformer",
        "model_dir": str(model_dir),
        "model_filename": ROFORMER_MODEL_FILENAME,
        "config_filename": ROFORMER_CONFIG_FILENAME,
    }
    with pytest.raises(FileNotFoundError, match="missing_assets"):
        worker.run_roformer({**base_req, "action": "probe"})

    (model_dir / ROFORMER_MODEL_FILENAME).write_bytes(b"ckpt-bytes")
    (model_dir / ROFORMER_CONFIG_FILENAME).write_bytes(b"yaml-bytes")

    buf = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", buf)
    probe_res = worker.run_roformer({**base_req, "action": "probe"})
    assert probe_res["ready"] is True and probe_res["provider"] == "mlx"

    metal_flag[0] = False
    assert worker.run_roformer({**base_req, "action": "probe"})["provider"] == "cpu"
    metal_flag[0] = True

    # Short mono 48 kHz hot-peak clip (tests mono->stereo, 48k->44.1k, peak>1 scaling, STFT pad)
    in_mono = tmp_path / "hot_mono.wav"
    sf.write(in_mono, np.full((600, 1), 2.0, dtype=np.float32), 48000, subtype="FLOAT")
    out_mono = tmp_path / "hot_mono_out.wav"
    sep_res = worker.run_roformer(
        {
            **base_req,
            "action": "separate",
            "input": str(in_mono),
            "output": str(out_mono),
            "target_stem": "vocals",
            "quality": "balanced",
        }
    )
    assert sep_res["selected_stem"] == "vocals"
    assert offline_checks == [ROFORMER_MODEL_FILENAME]
    y_out, sr_out = sf.read(out_mono, dtype="float32", always_2d=True)
    assert (sr_out, y_out.shape) == (48000, (600, 1))
    assert float(np.median(y_out)) == pytest.approx(1.0, abs=0.05)

    # Instrumental stem + error branches
    with pytest.raises(RuntimeError, match="invalid_audio"):
        worker.run_roformer({**base_req, "action": "separate", "target_stem": "drums"})

    fail_mode[0] = "failed_files"
    with pytest.raises(RuntimeError, match="inference_failed"):
        worker.run_roformer(
            {
                **base_req,
                "action": "separate",
                "input": str(in_mono),
                "output": str(tmp_path / "e1.wav"),
                "target_stem": "instrumental",
            }
        )

    fail_mode[0] = "missing_file"
    with pytest.raises(RuntimeError, match="inference_failed"):
        worker.run_roformer(
            {
                **base_req,
                "action": "separate",
                "input": str(in_mono),
                "output": str(tmp_path / "e2.wav"),
                "target_stem": "instrumental",
            }
        )

    fail_mode[0] = "bad_channels"
    with pytest.raises(RuntimeError, match="invalid_audio"):
        worker.run_roformer(
            {
                **base_req,
                "action": "separate",
                "input": str(in_mono),
                "output": str(tmp_path / "e3.wav"),
                "target_stem": "vocals",
            }
        )
    fail_mode[0] = "none"

    # Stereo 44.1 kHz normal-peak clip >= 4096 frames + absolute emitted path
    in_stereo = tmp_path / "stereo44.wav"
    sf.write(in_stereo, np.full((5000, 2), 0.2, dtype=np.float32), 44100, subtype="FLOAT")
    out_stereo = tmp_path / "stereo44_out.wav"
    orig_sep = _FakeSeparator.separate

    def abs_sep(self: _FakeSeparator, input_wav: str) -> list[str]:
        rel = orig_sep(self, input_wav)
        if not rel:
            return rel
        return [str(Path(self.output_dir) / rel[0])]

    monkeypatch.setattr(_FakeSeparator, "separate", abs_sep)
    stereo_res = worker.run_roformer(
        {
            **base_req,
            "action": "separate",
            "input": str(in_stereo),
            "output": str(out_stereo),
            "target_stem": "vocals",
            "quality": "fast",
        }
    )
    assert stereo_res["overlap"] == 2
    assert sf.info(out_stereo).frames == 5000

    # Invalid input WAV (3ch & NaN) and invalid NaN output rejected
    in_3ch = tmp_path / "three.wav"
    sf.write(in_3ch, np.zeros((100, 3), dtype=np.float32), 44100, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="invalid_audio"):
        worker.run_roformer(
            {
                **base_req,
                "action": "separate",
                "input": str(in_3ch),
                "output": str(tmp_path / "e4.wav"),
                "target_stem": "vocals",
            }
        )
    in_nan = tmp_path / "nan.wav"
    sf.write(in_nan, np.full((100, 2), np.nan, dtype=np.float32), 44100, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="invalid_audio"):
        worker.run_roformer(
            {
                **base_req,
                "action": "separate",
                "input": str(in_nan),
                "output": str(tmp_path / "e5.wav"),
                "target_stem": "vocals",
            }
        )

    def nan_sep(self: _FakeSeparator, input_wav: str) -> list[str]:
        data, rate = sf.read(input_wav, dtype="float32", always_2d=True)
        out_path = Path(self.output_dir) / "nan_stem.wav"
        sf.write(out_path, np.full_like(data, np.nan), rate, subtype="FLOAT")
        return [str(out_path)]

    monkeypatch.setattr(_FakeSeparator, "separate", nan_sep)
    with pytest.raises(RuntimeError, match="invalid_audio"):
        worker.run_roformer(
            {
                **base_req,
                "action": "separate",
                "input": str(in_stereo),
                "output": str(tmp_path / "e6.wav"),
                "target_stem": "vocals",
            }
        )

    # main() dispatches engine="roformer" to run_roformer
    monkeypatch.setenv("HAWAVOCLEAN_SEPARATION_PARENT_PID", str(os.getppid()))
    monkeypatch.setattr(worker, "run_roformer", lambda _req: {"ready": True, "engine": "roformer"})
    monkeypatch.setattr(
        sys,
        "stdin",
        types.SimpleNamespace(buffer=io.BytesIO(b'{"engine":"roformer","action":"probe"}\n')),
    )
    assert worker.main() == 0


# ---------------------------------------------------------------------------
# 5. Chunked disk-backed streaming (> 90 s / up to 60 min) in audio_edit.py
# ---------------------------------------------------------------------------


def test_chunked_streaming_overlap_add_is_seamless_and_supports_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MAX_DURATION_S == 3600.0
    assert MAX_FILE_SIZE_BYTES == 4 * 1024**3

    src = _make_wave(tmp_path / "long_clip.wav", duration_s=4.0, sr=8000, channels=2)
    monkeypatch.setattr(edits, "CHUNK_THRESHOLD_S", 2.0)
    monkeypatch.setattr(edits, "CHUNK_DURATION_S", 1.5)
    monkeypatch.setattr(edits, "CHUNK_OVERLAP_S", 0.25)

    chunk_lengths: list[int] = []

    def identity_separate(in_wav: Path, out_wav: Path, **kwargs: Any) -> dict[str, Any]:
        progress = kwargs.get("progress")
        if callable(progress):
            progress(0.5, "chunk")
        data, rate = sf.read(in_wav, dtype="float32", always_2d=True)
        chunk_lengths.append(data.shape[0])
        sf.write(out_wav, data, rate, subtype="FLOAT")
        return {"engine": ROFORMER_ENGINE_ID, "provider": "mlx", "model_sha256": "c" * 64}

    monkeypatch.setattr(edits, "separate_audio", identity_separate)

    dst = tmp_path / "chunked_out.wav"
    report = run_audio_edit(
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(dst),
            options=SeparationStrategyV1(
                kind="separate", operation="remove", prompt="background music"
            ),
        )
    )
    assert len(chunk_lengths) >= 3
    assert report.separation is not None
    assert report.separation["chunked"] is True
    assert report.separation["chunk_count"] == len(chunk_lengths)
    assert report.core.id == ROFORMER_ENGINE_ID
    assert report.environment.compute_device == "mlx"

    orig, _ = sf.read(src, dtype="float32", always_2d=True)
    out, _ = sf.read(dst, dtype="float32", always_2d=True)
    assert out.shape == orig.shape
    # Raised-cosine overlap-add must reconstruct identity signal seamlessly across seams!
    assert float(np.max(np.abs(out - orig))) < 1e-4

    # Also verify chunked streaming + cleanup_profile="production"
    dst_clean = tmp_path / "chunked_clean.wav"
    clean_report = run_audio_edit(
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(dst_clean),
            options=SeparationStrategyV1(
                kind="separate",
                operation="remove",
                prompt="background music",
                cleanup_profile="production",
            ),
        )
    )
    assert clean_report.output.samples == orig.shape[0]
    assert clean_report.separation is not None
    assert "cleanup_output_sha256" in clean_report.separation


# ---------------------------------------------------------------------------
# 6. Surgical time_range splicing — 100% bit-identical outside [start_s, end_s]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subtype", ["FLOAT", "PCM_24"])
def test_surgical_time_range_is_bit_identical_outside_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subtype: str
) -> None:
    sr = 16000
    src = _make_wave(
        tmp_path / f"src_{subtype}.wav", duration_s=5.0, sr=sr, channels=2, subtype=subtype
    )
    orig, _ = sf.read(src, dtype="float32", always_2d=True)

    seen_frames: list[int] = []

    def scale_separate(in_wav: Path, out_wav: Path, **_kwargs: Any) -> dict[str, Any]:
        data, rate = sf.read(in_wav, dtype="float32", always_2d=True)
        seen_frames.append(data.shape[0])
        # Distinctly attenuate processed region so we can verify interior changed
        sf.write(out_wav, data * 0.25, rate, subtype="FLOAT")
        return {"engine": ROFORMER_ENGINE_ID, "provider": "mlx"}

    monkeypatch.setattr(edits, "separate_audio", scale_separate)

    start_s, end_s = 2.0, 3.0
    dst = tmp_path / f"surgical_{subtype}.wav"
    report = run_audio_edit(
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(dst),
            time_range=(start_s, end_s),
            options=SeparationStrategyV1(
                kind="separate", operation="remove", prompt="background music"
            ),
        )
    )

    # Padded extraction window is [start - 1.5, end + 1.5] = [0.5, 4.5] -> 4.0 s, NOT full 5.0 s!
    assert seen_frames == [int(round(4.0 * sr))]

    out, out_sr = sf.read(dst, dtype="float32", always_2d=True)
    assert out_sr == sr and out.shape == orig.shape

    start_idx = int(round(start_s * sr))
    end_idx = int(round(end_s * sr))
    # 100% BIT-IDENTICAL outside [start_s, end_s]!
    assert np.array_equal(out[:start_idx], orig[:start_idx])
    assert np.array_equal(out[end_idx:], orig[end_idx:])

    # Interior past the 30 ms fade reflects the separated 0.25x signal
    fade = int(round(0.030 * sr))
    interior_out = out[start_idx + fade + 16 : end_idx - fade - 16]
    interior_orig = orig[start_idx + fade + 16 : end_idx - fade - 16]
    assert float(np.max(np.abs(interior_out - interior_orig * 0.25))) < 1e-4

    assert report.separation is not None
    assert report.separation["time_range"] == [start_s, end_s]
    assert report.separation["time_range_samples"] == [start_idx, end_idx]


def test_surgical_time_range_edges_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 8000
    src = _make_wave(tmp_path / "edge.wav", duration_s=2.0, sr=sr, channels=1, subtype="FLOAT")
    orig, _ = sf.read(src, dtype="float32", always_2d=True)

    def zero_separate(in_wav: Path, out_wav: Path, **_kwargs: Any) -> dict[str, Any]:
        data, rate = sf.read(in_wav, dtype="float32", always_2d=True)
        sf.write(out_wav, np.zeros_like(data), rate, subtype="FLOAT")
        return {"engine": ROFORMER_ENGINE_ID}

    monkeypatch.setattr(edits, "separate_audio", zero_separate)

    # Window starting at 0.0 and ending before end -> trailing region bit-identical
    dst_head = tmp_path / "head.wav"
    run_audio_edit(
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(dst_head),
            time_range=(0.0, 0.5),
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )
    )
    out_head, _ = sf.read(dst_head, dtype="float32", always_2d=True)
    assert np.array_equal(out_head[4000:], orig[4000:])

    # Window extending past duration clamps safely to EOF -> leading region bit-identical
    dst_tail = tmp_path / "tail.wav"
    run_audio_edit(
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(dst_tail),
            time_range=(1.5, 9.0),
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )
    )
    out_tail, _ = sf.read(dst_tail, dtype="float32", always_2d=True)
    assert np.array_equal(out_tail[:12000], orig[:12000])

    # Validation errors on EditRequest
    with pytest.raises(ValidationError):
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(tmp_path / "bad1.wav"),
            time_range=(2.0, 1.0),
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )
    with pytest.raises(ValidationError):
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(tmp_path / "bad2.wav"),
            time_range=(-0.5, 1.0),
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )
    with pytest.raises(ValidationError):
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(tmp_path / "bad3.wav"),
            time_range=(0.0, float("inf")),
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )
    with pytest.raises(ValidationError):
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(tmp_path / "bad4.wav"),
            strength=0.0,
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )

    # time_range starting past EOF fails cleanly
    with pytest.raises(ValueError, match="at or past end"):
        run_audio_edit(
            EditRequest(
                input_path=str(src),
                original_input_path=str(src),
                output_path=str(tmp_path / "past_eof.wav"),
                time_range=(5.0, 6.0),
                options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
            )
        )


# ---------------------------------------------------------------------------
# 7. Delay-aligned wet/dry parallel blend (strength < 1.0) + bundle + CLI main
# ---------------------------------------------------------------------------


def test_wet_dry_parallel_blend_delay_alignment_and_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sr = 16000
    rng = np.random.default_rng(7)
    n = sr  # 1 second broadband speech-like signal for crisp GCC-PHAT peak
    signal = rng.normal(0.0, 0.12, size=(n, 2)).astype(np.float32)
    src = tmp_path / "broadband.wav"
    sf.write(src, signal, sr, subtype="FLOAT")

    delay = 12  # Simulate 12-sample neural model latency

    def delayed_separate(in_wav: Path, out_wav: Path, **_kwargs: Any) -> dict[str, Any]:
        data, rate = sf.read(in_wav, dtype="float32", always_2d=True)
        lagged = np.pad(data[:-delay], ((delay, 0), (0, 0)), mode="constant") * 0.4
        sf.write(out_wav, lagged.astype(np.float32), rate, subtype="FLOAT")
        return {"engine": ROFORMER_ENGINE_ID, "provider": "mlx"}

    monkeypatch.setattr(edits, "separate_audio", delayed_separate)

    dst = tmp_path / "blended.wav"
    bundle = tmp_path / "record.zip"
    report = run_audio_edit(
        EditRequest(
            input_path=str(src),
            original_input_path=str(src),
            output_path=str(dst),
            strength=0.5,
            bundle_path=str(bundle),
            options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
        )
    )
    assert bundle.is_file()
    assert report.separation is not None
    assert report.separation["strength"] == 0.5
    assert report.separation["blend_shift_samples"] == delay

    out, _ = sf.read(dst, dtype="float32", always_2d=True)
    # Aligned 50/50 blend of 0.4*x and 1.0*x == 0.7*x across interior!
    expected = signal[delay:-delay] * 0.7
    actual = out[delay:-delay]
    assert float(np.max(np.abs(actual - expected))) < 5e-4

    # Also test reading time_range & strength fallback from options when not overridden
    opts = SeparationStrategyV1(kind="separate", operation="remove", prompt="music")
    object.__setattr__(
        opts, "__dict__", {**opts.__dict__, "strength": 0.6, "time_range": (0.1, 0.5)}
    )
    req_from_opts = EditRequest(
        input_path=str(src),
        original_input_path=str(src),
        output_path=str(tmp_path / "from_opts.wav"),
        options=opts,
    )
    resolved_range, resolved_strength = edits._resolve_edit_controls(req_from_opts)
    assert resolved_range == (0.1, 0.5)
    assert resolved_strength == 0.6


def test_audio_edit_main_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hawavoclean.cli as cli
    import hawavoclean.watchdog as wd

    monkeypatch.setattr(cli, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(wd, "install_parent_death_watchdog", lambda: None)

    src = _make_wave(tmp_path / "cli_in.wav", duration_s=0.3, sr=8000, channels=1)
    dst = tmp_path / "cli_out.wav"

    def passthrough(in_wav: Path, out_wav: Path, **_kwargs: Any) -> dict[str, Any]:
        data, rate = sf.read(in_wav, dtype="float32", always_2d=True)
        sf.write(out_wav, data, rate, subtype="FLOAT")
        return {"engine": ROFORMER_ENGINE_ID}

    monkeypatch.setattr(edits, "separate_audio", passthrough)

    payload = EditRequest(
        input_path=str(src),
        original_input_path=str(src),
        output_path=str(dst),
        options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
    ).model_dump_json()
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload + "\n"))
    edits.main()
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[-1]["event"] == "done"

    # Oversized stdin fails cleanly with error event
    monkeypatch.setattr(sys, "stdin", io.StringIO("x" * 16385))
    with pytest.raises(SystemExit) as exc:
        edits.main()
    assert exc.value.code == 1
    err = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert err["event"] == "error"


def test_audio_edit_and_runtime_edge_branches(
    tmp_path: Path, isolated_sep: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    del isolated_sep
    src = _make_wave(tmp_path / "edge_src.wav", duration_s=0.5, sr=8000, channels=2)

    # 1. Source == destination rejected
    with pytest.raises(ValueError, match="cannot be replaced"):
        run_audio_edit(
            EditRequest(
                input_path=str(src),
                original_input_path=str(src),
                output_path=str(src),
                options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
            )
        )

    # 2. Bad options-carried time_range / strength rejected by _resolve_edit_controls
    opts_bad_range = SeparationStrategyV1(kind="separate", operation="remove", prompt="music")
    object.__setattr__(opts_bad_range, "__dict__", {**opts_bad_range.__dict__, "time_range": "bad"})
    req_bad = EditRequest(
        input_path=str(src),
        original_input_path=str(src),
        output_path=str(tmp_path / "o1.wav"),
        options=opts_bad_range,
    )
    with pytest.raises(ValueError, match="pair"):
        edits._resolve_edit_controls(req_bad)

    opts_bad_str = SeparationStrategyV1(kind="separate", operation="remove", prompt="music")
    object.__setattr__(opts_bad_str, "__dict__", {**opts_bad_str.__dict__, "strength": 2.0})
    req_bad_s = EditRequest(
        input_path=str(src),
        original_input_path=str(src),
        output_path=str(tmp_path / "o2.wav"),
        options=opts_bad_str,
    )
    with pytest.raises(ValueError, match="strength"):
        edits._resolve_edit_controls(req_bad_s)

    # 3. Empty decoded stream & mismatched channel block rejected
    monkeypatch.setattr(edits, "iter_decode_audio", lambda *_a, **_k: iter(()))
    with pytest.raises(ValueError, match="no audio samples"):
        run_audio_edit(
            EditRequest(
                input_path=str(src),
                original_input_path=str(src),
                output_path=str(tmp_path / "empty.wav"),
                options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
            )
        )

    from hawavoclean.audio.types import AudioBuffer

    monkeypatch.setattr(
        edits,
        "iter_decode_audio",
        lambda *_a, **_k: iter([AudioBuffer(np.zeros((1, 100), dtype=np.float32), 8000)]),
    )
    with pytest.raises(ValueError, match="sound-separation limit"):
        run_audio_edit(
            EditRequest(
                input_path=str(src),
                original_input_path=str(src),
                output_path=str(tmp_path / "bad_ch.wav"),
                options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
            )
        )
    monkeypatch.undo()

    # 4. _chunk_spans sliver merge, _stream_wav_into_memmap resample/errors, _extract_wav_slice EOF
    spans = edits._chunk_spans(
        total_frames=2850, chunk_frames=1500, overlap_frames=250, max_worker_frames=3000
    )
    assert spans[-1][1] == 2850 and len(spans) == 2

    mm = edits._create_memmap(tmp_path / "t.f32", 2, 100)
    try:
        wrong_ch = tmp_path / "mono.wav"
        sf.write(wrong_ch, np.zeros((100, 1), dtype=np.float32), 8000, subtype="FLOAT")
        with pytest.raises(ValueError, match="valid recording timeline"):
            edits._stream_wav_into_memmap(wrong_ch, mm, 8000, 100)

        nan_wav = tmp_path / "nan_blk.wav"
        sf.write(nan_wav, np.full((100, 2), np.nan, dtype=np.float32), 8000, subtype="FLOAT")
        with pytest.raises(ValueError, match="valid recording timeline"):
            edits._stream_wav_into_memmap(nan_wav, mm, 8000, 100)

        diff_sr = tmp_path / "sr16k.wav"
        sf.write(diff_sr, np.full((200, 2), 0.1, dtype=np.float32), 16000, subtype="FLOAT")
        edits._stream_wav_into_memmap(diff_sr, mm, 8000, 100)
        assert float(np.mean(mm)) == pytest.approx(0.1, abs=0.02)

        with pytest.raises(ValueError, match="unexpectedly"):
            edits._extract_wav_slice(diff_sr, tmp_path / "slice.wav", 190, 50)
    finally:
        edits._close_memmap(mm)

    # 5. Sub-sample time_range & ultra-short / multi-window / silent _estimate_blend_delay
    with pytest.raises(ValueError, match="at least one audio sample"):
        run_audio_edit(
            EditRequest(
                input_path=str(src),
                original_input_path=str(src),
                output_path=str(tmp_path / "tiny_range.wav"),
                time_range=(0.0, 1e-9),
                options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
            )
        )

    short_mm = edits._create_memmap(tmp_path / "short_wet.f32", 2, 200)
    try:
        assert edits._estimate_blend_delay(src, short_mm, 0, 200, 8000) == (0, 0.0)
    finally:
        edits._close_memmap(short_mm)

    long_src = _make_wave(tmp_path / "scan_src.wav", duration_s=2.0, sr=8000, channels=2)
    silent_mm = edits._create_memmap(tmp_path / "silent_wet.f32", 2, 16000)
    try:
        monkeypatch.setattr(edits, "_DELAY_ANALYSIS_MAX_S", 0.5)
        assert edits._estimate_blend_delay(long_src, silent_mm, 0, 16000, 8000) == (0, 0.0)
    finally:
        edits._close_memmap(silent_mm)

    # 6. Chunked NaN failure fails closed & cleans up memmap
    monkeypatch.setattr(edits, "CHUNK_THRESHOLD_S", 0.5)
    monkeypatch.setattr(edits, "CHUNK_DURATION_S", 0.5)
    monkeypatch.setattr(edits, "CHUNK_OVERLAP_S", 0.1)

    def nan_chunk(in_wav: Path, out_wav: Path, **_k: Any) -> dict[str, Any]:
        data, rate = sf.read(in_wav, dtype="float32", always_2d=True)
        sf.write(out_wav, np.full_like(data, np.nan), rate, subtype="FLOAT")
        return {"engine": ROFORMER_ENGINE_ID}

    monkeypatch.setattr(edits, "separate_audio", nan_chunk)
    with pytest.raises(ValueError, match="valid recording timeline"):
        run_audio_edit(
            EditRequest(
                input_path=str(long_src),
                original_input_path=str(long_src),
                output_path=str(tmp_path / "nan_chunk_out.wav"),
                options=SeparationStrategyV1(kind="separate", operation="remove", prompt="music"),
            )
        )

    # 7. _run_supervised_worker protocol security edge cases (multi-result, missing metadata, non-dict)
    from hawavoclean.process_supervisor import ProcessSupervisor

    for mode, expected_err in (
        ("multi", "multiple final responses"),
        ("no_meta", "omitted its result metadata"),
        ("non_dict", "invalid or excessive"),
        ("fail_code", "out of memory|Not enough memory"),
    ):
        script = tmp_path / f"w_{mode}.py"
        if mode == "multi":
            body = (
                'print("HAWAVOCLEAN_SEPARATION_V1={\\"kind\\":\\"result\\",\\"ok\\":true,\\"metadata\\":{}}", flush=True)\n'
                'print("HAWAVOCLEAN_SEPARATION_V1={\\"kind\\":\\"result\\",\\"ok\\":true,\\"metadata\\":{}}", flush=True)\n'
            )
        elif mode == "no_meta":
            body = 'print("HAWAVOCLEAN_SEPARATION_V1={\\"kind\\":\\"result\\",\\"ok\\":true}", flush=True)\n'
        elif mode == "non_dict":
            body = 'print("HAWAVOCLEAN_SEPARATION_V1=[1,2,3]", flush=True)\n'
        else:
            body = 'import sys; print("HAWAVOCLEAN_SEPARATION_V1={\\"kind\\":\\"result\\",\\"ok\\":false,\\"code\\":\\"out_of_memory\\"}", flush=True); sys.exit(2)\n'
        script.write_text("import sys; sys.stdin.readline()\n" + body)
        orig_spawn = ProcessSupervisor.spawn

        def _spawn(
            cmd: list[str], *, _script: Path = script, _orig: Any = orig_spawn, **kw: Any
        ) -> Any:
            return _orig([*cmd[:-1], str(_script)], **kw)

        monkeypatch.setattr(ProcessSupervisor, "spawn", _spawn)
        with pytest.raises(SeparationError, match=expected_err):
            runtime._run_supervised_worker(
                Path(sys.executable),
                {"action": "probe"},
                env=roformer._roformer_environment(),
                timeout_s=5.0,
            )
        monkeypatch.setattr(ProcessSupervisor, "spawn", orig_spawn)


# ---------------------------------------------------------------------------
# 8. Live MLX Mel-Band RoFormer end-to-end smoke (runs when cached locally)
# ---------------------------------------------------------------------------


def test_live_mlx_roformer_surgical_kurdish_clip(tmp_path: Path) -> None:
    sample = Path("data/kurdish_speakers/fatih/sherko_01_10.wav")
    if not sample.is_file() or not probe_roformer_runtime().available:
        pytest.skip("Local MLX RoFormer checkpoint or Kurdish sample not present")

    orig, sr = sf.read(sample, dtype="float32", always_2d=True)
    orig_sha = hash_file(sample)
    dst = tmp_path / "kurdish_surgical.wav"
    report = run_audio_edit(
        EditRequest(
            input_path=str(sample),
            original_input_path=str(sample),
            output_path=str(dst),
            time_range=(1.0, 2.5),
            strength=0.8,
            options=SeparationStrategyV1(
                kind="separate",
                operation="remove",
                prompt="background music",
                quality="fast",
                device="cpu",
            ),
        )
    )
    assert hash_file(sample) == orig_sha
    assert report.core.id == ROFORMER_ENGINE_ID
    assert report.separation is not None
    assert report.separation["license"] == "MIT"
    assert report.separation["provider"] == "mlx"
    out, out_sr = sf.read(dst, dtype="float32", always_2d=True)
    assert out_sr == sr and out.shape == orig.shape
    assert np.array_equal(out[: int(round(1.0 * sr))], orig[: int(round(1.0 * sr))])
    assert np.array_equal(out[int(round(2.5 * sr)) :], orig[int(round(2.5 * sr)) :])
