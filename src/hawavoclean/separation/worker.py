"""Standalone bridge executed by an isolated separation Python interpreter.

Keep this file compatible with Python 3.11+ and independent of hawavoclean's
installed modules. No model downloads are allowed in this process.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SENTINEL = "HAWAVOCLEAN_SEPARATION_V1="
SAM_AUDIO_COMMIT = "bb4c6999d2677c7402360e426afc01ddfad6dce0"
WORKER_MAX_DURATION_S = 120.0
ROFORMER_SAMPLE_RATE = 44100
_ROFORMER_OVERLAP = {"fast": 2, "balanced": 4, "high": 8}


def emit(value: dict[str, Any]) -> None:
    print(SENTINEL + json.dumps(value, allow_nan=False), file=sys.__stdout__, flush=True)


def cached_assets(repo_id: str) -> tuple[Path, Path, Path]:
    from huggingface_hub import try_to_load_from_cache

    def cached(repo: str, filename: str) -> Path | None:
        path = try_to_load_from_cache(repo, filename)
        return Path(path) if isinstance(path, str) and Path(path).is_file() else None

    config, checkpoint = cached(repo_id, "config.json"), cached(repo_id, "checkpoint.pt")
    if not config or not checkpoint:
        raise FileNotFoundError("missing_assets")
    raw = json.loads(config.read_text())
    text_name = (raw.get("text_encoder") or {}).get("name", "t5-base")
    text_weights = cached(text_name, "model.safetensors") or cached(text_name, "pytorch_model.bin")
    if not text_weights or not cached(text_name, "config.json"):
        raise FileNotFoundError("missing_assets")
    # Tokenizer construction checks the actual cached tokenizer files, not
    # merely a cache directory's existence. It does not load model weights.
    AutoTokenizer = importlib.import_module("transformers").AutoTokenizer
    AutoTokenizer.from_pretrained(text_name, local_files_only=True)
    return config, checkpoint, text_weights


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_torch_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("provider_unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("provider_unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise RuntimeError("provider_unavailable")
    return requested


def run(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    if (
        importlib.metadata.version("torch").split("+")[0] != "2.9.0"
        or importlib.metadata.version("torchaudio").split("+")[0] != "2.9.0"
        or int(np.__version__.split(".")[0]) >= 2
    ):
        raise RuntimeError("runtime_incompatible")
    sam_distribution = importlib.metadata.distribution("sam-audio")
    direct_url = json.loads(sam_distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != SAM_AUDIO_COMMIT:
        raise RuntimeError("runtime_incompatible")
    sys.path.insert(0, str(Path(request["root"]) / "backend"))
    sam_lite = importlib.import_module("workers.sam_lite")
    # Verify the upstream runtime's used imports in this interpreter.
    importlib.import_module("sam_audio")

    device = _resolve_torch_device(str(request["device"]), torch)
    repo_id = sam_lite.model_repo_id(request["model_size"])
    config, checkpoint, text_weights = cached_assets(repo_id)
    if request["action"] == "probe":
        return {"ready": True, "model": repo_id, "device": device}

    with sf.SoundFile(request["input"]) as source:
        if (
            source.channels not in (1, 2)
            or source.frames < 1
            or source.frames / source.samplerate > WORKER_MAX_DURATION_S
            or not 8000 <= source.samplerate <= 192000
        ):
            raise RuntimeError("invalid_audio")
        source_sr, source_frames = source.samplerate, source.frames
        source_channels = source.channels
        audio = source.read(dtype="float32", always_2d=True).T
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 10:
        raise RuntimeError("invalid_audio")
    started = time.monotonic()
    emit(
        {"kind": "progress", "fraction": 0.0, "message": "Loading the local sound-separation model"}
    )
    lite = sam_lite.load_lite_model(
        repo_id, device=device, precision="fp32" if device in ("cpu", "mps") else "fp16"
    )
    if source_sr != lite.sample_rate:
        divisor = math.gcd(source_sr, lite.sample_rate)
        audio = resample_poly(audio, lite.sample_rate // divisor, source_sr // divisor, axis=-1)
    expected_model_frames = round(source_frames * lite.sample_rate / source_sr)
    audio = np.asarray(audio[..., :expected_model_frames], dtype=np.float32)
    if audio.shape[-1] < expected_model_frames:
        audio = np.pad(audio, ((0, 0), (0, expected_model_frames - audio.shape[-1])))
    last_progress = [0.0]

    def progress(fraction: float, _message: str) -> None:
        now = time.monotonic()
        if now - last_progress[0] >= 0.25 or fraction >= 1:
            emit(
                {
                    "kind": "progress",
                    "fraction": float(fraction),
                    "message": "Separating the requested sound",
                }
            )
            last_progress[0] = now

    extracting = request["operation"] == "extract"
    result = sam_lite.separate(
        lite,
        torch.from_numpy(audio),
        request["prompt"],
        chunk_seconds=25.0,
        steps=sam_lite.QUALITY_STEPS[request["quality"]],
        seed=request["seed"],
        progress=progress,
        decode_target=extracting,
        decode_residual=not extracting,
    )
    tensor = result.target if extracting else result.residual
    if tensor is None:
        raise RuntimeError("invalid_audio")
    selected = tensor.detach().float().cpu().numpy()
    if selected.ndim != 2 or selected.shape != (source_channels, expected_model_frames):
        raise RuntimeError("invalid_audio")
    if lite.sample_rate != source_sr:
        divisor = math.gcd(source_sr, lite.sample_rate)
        selected = resample_poly(
            selected, source_sr // divisor, lite.sample_rate // divisor, axis=-1
        )
    selected = selected[..., :source_frames]
    if selected.shape[-1] < source_frames:
        selected = np.pad(selected, ((0, 0), (0, source_frames - selected.shape[-1])))
    if not np.isfinite(selected).all() or np.max(np.abs(selected)) > 10:
        raise RuntimeError("invalid_audio")
    sf.write(request["output"], selected.T, source_sr, subtype="FLOAT")
    return {
        "model": repo_id,
        "device": device,
        "sam_audio_commit": SAM_AUDIO_COMMIT,
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "torchaudio",
                "numpy",
                "scipy",
                "transformers",
                "huggingface_hub",
            )
        },
        "model_sample_rate": lite.sample_rate,
        "model_sha256": sha256(checkpoint),
        "model_config_sha256": sha256(config),
        "text_encoder_sha256": sha256(text_weights),
        "selected_stem": "target" if extracting else "residual",
        "precision": lite.precision,
        "runtime_ms": round((time.monotonic() - started) * 1000, 3),
        "steps": result.steps,
        "chunks": result.chunks,
    }


def run_roformer(request: dict[str, Any]) -> dict[str, Any]:
    """Run offline-only MIT Mel-Band RoFormer separation via mlx-audio-separator."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    mx = importlib.import_module("mlx.core")
    sep_core = importlib.import_module("mlx_audio_separator.core")
    importlib.import_module("mlx_audio_separator.separator.architectures.mdxc_separator")

    model_dir = Path(str(request["model_dir"]))
    model_filename = str(request["model_filename"])
    config_filename = str(request["config_filename"])
    model_path = model_dir / model_filename
    yaml_path = model_dir / config_filename
    if not model_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError("missing_assets")

    metal = getattr(mx, "metal", None)
    provider = "mlx" if (metal is not None and bool(metal.is_available())) else "cpu"
    if request["action"] == "probe":
        return {
            "ready": True,
            "model": model_filename,
            "device": provider,
            "provider": provider,
            "engine": "mlx-mel-band-roformer",
        }

    target_stem = str(request["target_stem"])
    if target_stem not in {"vocals", "instrumental"}:
        raise RuntimeError("invalid_audio")
    stem_label = "Vocals" if target_stem == "vocals" else "Instrumental"
    overlap = _ROFORMER_OVERLAP.get(str(request.get("quality", "high")), 4)

    with sf.SoundFile(request["input"]) as source:
        if (
            source.channels not in (1, 2)
            or source.frames < 1
            or source.frames / source.samplerate > WORKER_MAX_DURATION_S
            or not 8000 <= source.samplerate <= 192000
        ):
            raise RuntimeError("invalid_audio")
        source_sr, source_frames = source.samplerate, source.frames
        source_channels = source.channels
        audio = source.read(dtype="float32", always_2d=True).T
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 10:
        raise RuntimeError("invalid_audio")

    started = time.monotonic()
    emit(
        {
            "kind": "progress",
            "fraction": 0.05,
            "message": "Loading local MLX Mel-Band RoFormer model",
        }
    )

    # Preserve levels for hot float signals (> 1.0) without clipping in separator.
    input_peak = float(np.max(np.abs(audio)))
    headroom_scale = input_peak if input_peak > 1.0 else 1.0
    model_audio = audio / headroom_scale if headroom_scale > 1.0 else audio
    if source_channels == 1:
        model_audio = np.repeat(model_audio, 2, axis=0)
    if source_sr != ROFORMER_SAMPLE_RATE:
        divisor = math.gcd(source_sr, ROFORMER_SAMPLE_RATE)
        model_audio = resample_poly(
            model_audio, ROFORMER_SAMPLE_RATE // divisor, source_sr // divisor, axis=-1
        )
    expected_model_frames = max(1, round(source_frames * ROFORMER_SAMPLE_RATE / source_sr))
    model_audio = np.asarray(model_audio[..., :expected_model_frames], dtype=np.float32)
    if model_audio.shape[-1] < expected_model_frames:
        model_audio = np.pad(
            model_audio, ((0, 0), (0, expected_model_frames - model_audio.shape[-1]))
        )
    # RoFormer STFT uses reflect padding of n_fft // 2 = 1024 samples; pad
    # ultra-short clips cleanly and trim back to exact source length afterward.
    min_stft_frames = 4096
    if model_audio.shape[-1] < min_stft_frames:
        model_audio = np.pad(model_audio, ((0, 0), (0, min_stft_frames - model_audio.shape[-1])))

    out_parent = Path(str(request["output"])).parent
    with tempfile.TemporaryDirectory(prefix=".roformer-work-", dir=out_parent) as temp_dir:
        work_dir = Path(temp_dir)
        work_in = work_dir / "input.wav"
        work_out = work_dir / "out"
        work_out.mkdir()
        sf.write(work_in, model_audio.T, ROFORMER_SAMPLE_RATE, subtype="FLOAT")

        sep = sep_core.Separator(
            model_file_dir=str(model_dir),
            output_dir=str(work_out),
            output_format="WAV",
            normalization_threshold=1.0,
            amplification_threshold=0.0,
            output_single_stem=stem_label,
            sample_rate=ROFORMER_SAMPLE_RATE,
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 1,
                "overlap": overlap,
                "pitch_shift": 0,
            },
        )

        # Strictly block any network downloads inside the offline worker.
        def _offline_only_check(_url: str, output_path: str) -> None:
            if not os.path.isfile(output_path):
                raise FileNotFoundError("missing_assets")

        def _offline_model_files(filename: str) -> tuple[str, str, str, str, str]:
            if filename != model_filename or not model_path.is_file() or not yaml_path.is_file():
                raise FileNotFoundError("missing_assets")
            return (
                model_filename,
                "MDXC",
                "Mel-Band RoFormer (MIT)",
                str(model_path),
                str(yaml_path),
            )

        sep.download_file_if_not_exists = _offline_only_check
        sep.download_model_files = _offline_model_files
        sep.load_model(model_filename)
        emit(
            {
                "kind": "progress",
                "fraction": 0.25,
                "message": "Separating vocals and music with MLX RoFormer",
            }
        )
        emitted_files = sep.separate(str(work_in))
        if getattr(sep, "failed_files", None) or not emitted_files:
            raise RuntimeError("inference_failed")

        stem_file = Path(str(emitted_files[0]))
        if not stem_file.is_absolute():
            stem_file = work_out / stem_file
        if not stem_file.is_file():
            raise RuntimeError("inference_failed")

        separated, out_sr = sf.read(stem_file, dtype="float32", always_2d=True)

    selected = np.asarray(separated.T, dtype=np.float32)
    selected = selected[..., :expected_model_frames]
    if source_channels == 1 and selected.shape[0] == 2:
        selected = np.mean(selected, axis=0, keepdims=True, dtype=np.float32)
    elif selected.shape[0] != source_channels:
        raise RuntimeError("invalid_audio")

    if headroom_scale > 1.0:
        selected = selected * np.float32(headroom_scale)
    if out_sr != source_sr:
        divisor = math.gcd(source_sr, out_sr)
        selected = resample_poly(selected, source_sr // divisor, out_sr // divisor, axis=-1)
    selected = np.asarray(selected[..., :source_frames], dtype=np.float32)
    if selected.shape[-1] < source_frames:
        selected = np.pad(selected, ((0, 0), (0, source_frames - selected.shape[-1])))
    if not np.isfinite(selected).all() or np.max(np.abs(selected)) > 10:
        raise RuntimeError("invalid_audio")

    sf.write(request["output"], selected.T, source_sr, subtype="FLOAT")
    emit({"kind": "progress", "fraction": 1.0, "message": "Finished vocal/music separation"})

    versions: dict[str, str] = {}
    for package in ("mlx", "mlx-audio-separator", "numpy", "scipy", "soundfile"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return {
        "model": model_filename,
        "license": "MIT",
        "device": provider,
        "provider": provider,
        "runtime_versions": versions,
        "model_sample_rate": ROFORMER_SAMPLE_RATE,
        "model_sha256": sha256(model_path),
        "model_config_sha256": sha256(yaml_path),
        "selected_stem": target_stem,
        "runtime_ms": round((time.monotonic() - started) * 1000, 3),
        "overlap": overlap,
    }


def main() -> int:
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE="1",
    )
    # A killed parent cannot run its cleanup. Windows additionally has the
    # supervisor's kill-on-close Job Object boundary.
    stop_watchdog = threading.Event()
    watchdog_thread: threading.Thread | None = None
    if os.name == "posix":
        declared_parent = int(os.environ.get("HAWAVOCLEAN_SEPARATION_PARENT_PID", "0"))
        if declared_parent <= 1 or os.getppid() != declared_parent:
            return 130

        def watch_parent() -> None:
            while not stop_watchdog.wait(0.25):
                if os.getppid() != declared_parent:
                    os._exit(130)

        watchdog_thread = threading.Thread(
            target=watch_parent, name="separation-parent-watchdog", daemon=True
        )
        watchdog_thread.start()
    try:
        raw = sys.stdin.buffer.readline(16385)
        if len(raw) > 16384:
            raise ValueError("invalid_request")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("invalid_request")
        # Third-party print output cannot impersonate structured responses.
        with contextlib.redirect_stdout(sys.stderr):
            metadata = (
                run_roformer(request) if request.get("engine") == "roformer" else run(request)
            )
        emit({"kind": "result", "ok": True, "metadata": metadata})
        return 0
    except Exception as error:
        code = str(error)
        if isinstance(error, (ImportError, importlib.metadata.PackageNotFoundError)):
            code = "runtime_incompatible"
        elif (
            isinstance(error, FileNotFoundError)
            or "offline" in code.lower()
            or "cached" in code.lower()
        ):
            code = "missing_assets"
        elif "out of memory" in code.lower():
            code = "out_of_memory"
        allowed = {
            "runtime_incompatible",
            "provider_unavailable",
            "missing_assets",
            "invalid_audio",
            "out_of_memory",
        }
        emit(
            {"kind": "result", "ok": False, "code": code if code in allowed else "inference_failed"}
        )
        return 2
    finally:
        stop_watchdog.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
