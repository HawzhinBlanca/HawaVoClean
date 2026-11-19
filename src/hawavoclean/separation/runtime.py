"""Supervised, offline-only hybrid sound separation runtime.

Routes vocal/music prompts to local MIT Mel-Band RoFormer (Apple Silicon MLX)
when cached locally, and routes open-vocabulary sound-effect prompts (or vocal
fallback when RoFormer is absent) to the isolated AudioGhost SAM-Audio runtime.

Large neural models never import into the HTTP broker. Requests travel over
stdin, not process arguments. Neither probing nor inference touches the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import soundfile as sf

from hawavoclean.process_supervisor import ProcessSupervisor

AUDIOGHOST_COMMIT = "79653d1a789f4fc73893d10733e2324ed088cecf"
MAX_DURATION_S = 3600.0
MAX_FILE_SIZE_BYTES = 4 * 1024**3
WORKER_MAX_DURATION_S = 120.0
MAX_CHANNELS = 2
PYTHON_ENV = "HAWAVOCLEAN_AUDIOGHOST_PYTHON"
ROOT_ENV = "HAWAVOCLEAN_AUDIOGHOST_ROOT"
_ENGINE_BLOBS = {
    "backend/workers/__init__.py": "e7e331b43ce901e5f2e7a5e3638930cfa8eefa8a",
    "backend/workers/compat.py": "98b2a5f82d303a45a9a5b0e2eac9ac076565f279",
    "backend/workers/sam_lite.py": "bea1e4a1e18a6a1735fb7f29eb100392b4d3549d",
}
_SENTINEL = "HAWAVOCLEAN_SEPARATION_V1="
_WORKER_OUTPUT_LIMIT = 256 * 1024
_LOCAL_ROOT = Path(__file__).resolve().parents[3] / ".local-ai"
_DEFAULT_WORKER_ERRORS = {
    "missing_assets": "AudioGhost model or text-encoder files are not installed for offline use.",
    "provider_unavailable": "The requested local separation provider is unavailable.",
    "runtime_incompatible": "AudioGhost dependencies do not match the supported runtime.",
    "invalid_audio": "Sound separation requires valid mono/stereo audio up to two minutes per pass.",
    "out_of_memory": "Not enough memory for this separation model; try the Small model.",
}


class SeparationError(RuntimeError):
    """An unavailable, failed or invalid separation; never a successful bypass."""


class SeparationCancelledError(SeparationError):
    """The caller cancelled the supervised operation."""


@dataclass(frozen=True)
class RuntimeStatus:
    available: bool
    reason: str
    providers: tuple[str, ...] = ()
    upstream_commit: str = AUDIOGHOST_COMMIT


def resolve_separation_device(device: str = "auto") -> Literal["cpu", "cuda", "mps"]:
    """Resolve ``"auto"`` to the best local accelerator (``mps`` -> ``cuda`` -> ``cpu``)."""
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise SeparationError("Choose a supported local separation provider.")
    if device != "auto":
        return cast(Literal["cpu", "cuda", "mps"], device)
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        if sys.platform == "darwin" and platform.machine() == "arm64":
            return "mps"
        return "cpu"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return "mps"
    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is not None and bool(cuda_mod.is_available()):
        return "cuda"
    return "cpu"


def _validate_options(
    prompt: str, operation: str, model_size: str, quality: str, seed: int, device: str
) -> None:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 512:
        raise SeparationError("Describe one sound using 1–512 characters.")
    if any(ord(char) < 32 for char in prompt):
        raise SeparationError("Sound descriptions cannot contain control characters.")
    if operation not in {"extract", "remove"}:
        raise SeparationError("Choose Extract or Remove.")
    if model_size not in {"small", "base", "large"} or quality not in {"fast", "balanced", "high"}:
        raise SeparationError("Unsupported separation model or quality.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise SeparationError("The separation seed must be an integer from 0 to 2147483647.")
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise SeparationError("Choose a supported local separation provider.")


def _runtime_paths() -> tuple[Path, Path]:
    has_python, has_root = PYTHON_ENV in os.environ, ROOT_ENV in os.environ
    if has_python != has_root:
        raise SeparationError("Configure both AudioGhost runtime variables together, or omit both.")
    if has_python:
        python = Path(os.environ[PYTHON_ENV])
        root = Path(os.environ[ROOT_ENV])
    else:
        python, root = _manifest_paths()
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise SeparationError("AudioGhost is not installed: configure its separate Python runtime.")
    if not root.is_absolute() or not root.is_dir():
        raise SeparationError(
            "AudioGhost is not installed: configure its reviewed engine checkout."
        )
    for relative, expected in _ENGINE_BLOBS.items():
        try:
            payload = (root / relative).read_bytes()
        except OSError as exc:
            raise SeparationError("The AudioGhost engine installation is incomplete.") from exc
        digest = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if digest != expected:
            raise SeparationError("AudioGhost engine files do not match the reviewed version.")
    return python, root


def _manifest_paths() -> tuple[Path, Path]:
    """Discover only this source checkout's installer-owned local manifest."""
    value = _manifest_values()
    return Path(value[PYTHON_ENV]), Path(value[ROOT_ENV])


def _manifest_values() -> dict[str, Any]:
    manifest = _LOCAL_ROOT / "audioghost-runtime.json"
    if not manifest.exists():
        raise SeparationError("AudioGhost is not installed: configure its separate Python runtime.")
    try:
        with manifest.open("rb") as file:
            raw = file.read(4097)
        if len(raw) > 4096:
            raise ValueError("manifest too large")
        value = json.loads(raw)
        expected_keys = {PYTHON_ENV, ROOT_ENV, "commit", "models_downloaded"}
        if (
            not isinstance(value, dict)
            or not expected_keys.issubset(value)
            or set(value).difference(expected_keys | {"ffmpeg_library_path"})
            or value["commit"] != AUDIOGHOST_COMMIT
            or not isinstance(value["models_downloaded"], bool)
            or not isinstance(value[PYTHON_ENV], str)
            or not isinstance(value[ROOT_ENV], str)
        ):
            raise ValueError("invalid manifest")
        python, root = Path(value[PYTHON_ENV]), Path(value[ROOT_ENV])
        if (
            python != _LOCAL_ROOT / "audioghost-env/bin/python"
            or root != _LOCAL_ROOT / "audioghost-ai"
        ):
            raise ValueError("unexpected runtime location")
        if "ffmpeg_library_path" in value:
            library = value["ffmpeg_library_path"]
            if (
                not isinstance(library, str)
                or not Path(library).is_absolute()
                or not Path(library).is_dir()
            ):
                raise ValueError("invalid shared-library directory")
    except (OSError, ValueError, TypeError) as exc:
        raise SeparationError("The local AudioGhost installation manifest is invalid.") from exc
    return value


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    # Neither a processing request nor a readiness probe downloads anything.
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        HAWAVOCLEAN_SEPARATION_PARENT_PID=str(os.getpid()),
    )
    # Environment-based auth is not needed for cached, offline inference.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        environment.pop(key, None)
    library = environment.get("HAWAVOCLEAN_AUDIOGHOST_FFMPEG_LIB")
    if (
        library is None
        and PYTHON_ENV not in environment
        and ROOT_ENV not in environment
        and (_LOCAL_ROOT / "audioghost-runtime.json").exists()
    ):
        library = _manifest_values().get("ffmpeg_library_path")
    if library:
        if not Path(library).is_absolute() or not Path(library).is_dir():
            raise SeparationError("The configured AudioGhost FFmpeg library directory is invalid.")
        key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        if os.name == "nt":
            key = "PATH"
        environment[key] = library + (os.pathsep + environment[key] if environment.get(key) else "")
    return environment


def _run_supervised_worker(
    python: Path,
    request: dict[str, Any],
    *,
    env: dict[str, str],
    timeout_s: float,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    error_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute ``worker.py`` under ``ProcessSupervisor`` with bounded pipes."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise SeparationError("The separation deadline must be finite and positive.")
    if should_cancel and should_cancel():
        raise SeparationCancelledError("Sound separation was cancelled.")
    worker = Path(__file__).with_name("worker.py")
    supervisor = ProcessSupervisor.spawn(
        [str(python), "-I", "-B", str(worker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    process = cast(subprocess.Popen[bytes], supervisor.process)
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
    overflow = threading.Event()
    result: dict[str, Any] | None = None

    def drain(stream: Any, parse: bool) -> None:
        total = 0
        try:
            while True:
                line = stream.readline(8193)
                if not line:
                    return
                total += len(line)
                if len(line) > 8192 or total > _WORKER_OUTPUT_LIMIT:
                    overflow.set()
                    return
                if parse and line.startswith(_SENTINEL.encode()):
                    event = json.loads(line[len(_SENTINEL) :])
                    if not isinstance(event, dict):
                        overflow.set()
                        return
                    try:
                        events.put(event, timeout=0.5)
                    except queue.Full:
                        # Progress may be coalesced; terminal results must be delivered.
                        if event.get("kind") != "progress":
                            overflow.set()
                            return
        except (OSError, ValueError, json.JSONDecodeError):
            overflow.set()

    readers = [
        threading.Thread(target=drain, args=(process.stdout, True), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, False), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout_s
    reasons = _DEFAULT_WORKER_ERRORS if error_reasons is None else error_reasons
    try:
        process.stdin.write(json.dumps(request).encode() + b"\n")
        process.stdin.close()
        while process.poll() is None or not events.empty() or any(t.is_alive() for t in readers):
            if should_cancel and should_cancel():
                raise SeparationCancelledError("Sound separation was cancelled.")
            if time.monotonic() >= deadline:
                raise SeparationError("Sound separation exceeded its deadline.")
            if overflow.is_set():
                raise SeparationError(
                    "The separation worker returned an invalid or excessive response."
                )
            try:
                event = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("kind") == "progress" and progress:
                value = event.get("fraction")
                message = event.get("message")
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    progress(min(1.0, max(0.0, value)), str(message)[:160])
            elif event.get("kind") == "result":
                if result is not None:
                    raise SeparationError(
                        "The separation worker returned multiple final responses."
                    )
                result = event
        if overflow.is_set():
            raise SeparationError(
                "The separation worker returned an invalid or excessive response."
            )
        if process.returncode != 0 or not result or result.get("ok") is not True:
            code = result.get("code") if result else None
            raise SeparationError(
                reasons.get(str(code), "Sound separation failed; no result was published.")
            )
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise SeparationError("The separation worker omitted its result metadata.")
        return metadata
    finally:
        supervisor.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(OSError):
                stream.close()
        for reader in readers:
            reader.join(timeout=1)


def _run_worker(
    request: dict[str, Any],
    *,
    timeout_s: float,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    python, root = _runtime_paths()
    return _run_supervised_worker(
        python,
        {**request, "root": str(root)},
        env=_worker_environment(),
        timeout_s=timeout_s,
        progress=progress,
        should_cancel=should_cancel,
    )


def probe_audioghost_runtime(model_size: str = "small", device: str = "auto") -> RuntimeStatus:
    """Inspect pinned AudioGhost imports and cached SAM-Audio assets without loading weights."""
    try:
        _validate_options("speech", "extract", model_size, "high", 0, device)
        _runtime_paths()
        if not _cached_model_present(model_size):
            raise SeparationError("AudioGhost model files are not installed for offline use.")
        metadata = _run_worker(
            {"action": "probe", "model_size": model_size, "device": device}, timeout_s=45.0
        )
        if metadata.get("ready") is not True:
            raise SeparationError("AudioGhost model assets are not installed for offline use.")
        resolved = str(metadata.get("device") or resolve_separation_device(device))
        return RuntimeStatus(
            True,
            "Experimental local separation runtime and cached model assets are installed.",
            (resolved,),
        )
    except SeparationError as exc:
        return RuntimeStatus(False, str(exc))
    except OSError:
        return RuntimeStatus(False, "The separate AudioGhost runtime could not be started.")


def _combine_runtime_statuses(
    roformer_status: RuntimeStatus, audioghost_status: RuntimeStatus
) -> RuntimeStatus:
    if roformer_status.available and audioghost_status.available:
        providers = tuple(dict.fromkeys((*roformer_status.providers, *audioghost_status.providers)))
        return RuntimeStatus(
            True,
            "Hybrid separation ready: MLX Mel-Band RoFormer (vocal/music) and "
            "AudioGhost SAM-Audio (open-prompt SFX).",
            providers,
        )
    if roformer_status.available:
        return RuntimeStatus(
            True,
            "Vocal/music separation ready via MLX RoFormer; "
            "install SAM-Audio weights for open-prompt SFX.",
            roformer_status.providers,
        )
    return audioghost_status


def probe_runtime(
    model_size: str = "small",
    device: str = "auto",
    *,
    prompt: str | None = None,
    operation: Literal["extract", "remove"] | None = None,
) -> RuntimeStatus:
    """Inspect hybrid separation readiness truthfully without loading model weights.

    When ``prompt`` and ``operation`` are provided, validates that a runtime
    capable of handling that prompt scope is ready (RoFormer or AudioGhost for
    vocal/music prompts; AudioGhost for open-vocabulary SFX).
    """
    from hawavoclean.separation import roformer

    try:
        _validate_options(
            prompt if prompt is not None else "speech",
            operation if operation is not None else "extract",
            model_size,
            "high",
            0,
            device,
        )
    except SeparationError as exc:
        return RuntimeStatus(False, str(exc))

    roformer_status = (
        roformer.probe_roformer_runtime()
        if device != "cuda"
        else RuntimeStatus(False, "MLX Mel-Band RoFormer does not use CUDA.")
    )

    if prompt is not None and operation is not None:
        target_stem = roformer.classify_vocal_music_prompt(prompt, operation)
        if target_stem is not None and roformer_status.available:
            return roformer_status
        ag_status = probe_audioghost_runtime(model_size, device)
        if ag_status.available:
            return ag_status
        if target_stem is None and roformer_status.available:
            return RuntimeStatus(
                False,
                "Open-prompt SFX separation requires AudioGhost SAM-Audio weights "
                "(MLX RoFormer is installed for vocal/music prompts only).",
            )
        return ag_status

    ag_status = probe_audioghost_runtime(model_size, device)
    return _combine_runtime_statuses(roformer_status, ag_status)


def probe_separation_capabilities(device: str = "auto") -> RuntimeStatus:
    """Probe both MLX RoFormer and AudioGhost across all SAM-Audio model sizes."""
    from hawavoclean.separation import roformer

    try:
        resolve_separation_device(device)
    except SeparationError as exc:
        return RuntimeStatus(False, str(exc))

    roformer_status = (
        roformer.probe_roformer_runtime()
        if device != "cuda"
        else RuntimeStatus(False, "MLX Mel-Band RoFormer does not use CUDA.")
    )
    ag_status = probe_audioghost_runtime(model_size="large", device=device)
    if not ag_status.available:
        for size in ("base", "small"):
            candidate = probe_audioghost_runtime(model_size=size, device=device)
            if candidate.available:
                ag_status = candidate
                break
    return _combine_runtime_statuses(roformer_status, ag_status)


probe_any_separation_runtime = probe_separation_capabilities


def _cached_model_present(model_size: str) -> bool:
    """Fast absence-only check; a positive result is still verified by the child."""
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        cache = Path(configured)
    else:
        home = os.environ.get("HF_HOME")
        if home:
            cache = Path(home) / "hub"
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            cache = base / "huggingface/hub"
    snapshots = cache / f"models--facebook--sam-audio-{model_size}" / "snapshots"
    return any(
        (config.parent / "checkpoint.pt").is_file() for config in snapshots.glob("*/config.json")
    )


def validate_audio_file(path: Path | str, *, max_duration_s: float | None = None) -> sf.SoundFile:
    """Validate a bounded WAV header before allocating decoded samples."""
    limit = MAX_DURATION_S if max_duration_s is None else max_duration_s
    try:
        audio = sf.SoundFile(str(path))
    except (OSError, RuntimeError) as exc:
        raise SeparationError("Sound separation requires a decoded WAV file.") from exc
    if (
        audio.format not in {"WAV", "WAVEX", "RF64"}
        or not 1 <= audio.channels <= MAX_CHANNELS
        or not 8000 <= audio.samplerate <= 192000
        or audio.frames < 1
        or audio.frames / audio.samplerate > limit
    ):
        audio.close()
        limit_label = (
            f"{int(limit // 60)} minutes"
            if limit >= 60.0 and limit % 60.0 == 0
            else f"{limit:g} seconds"
        )
        raise SeparationError(
            f"Sound separation supports mono/stereo WAV files up to {limit_label}."
        )
    return audio


def _validate_output_stem(
    candidate: Path, expected: tuple[int, int, int], *, max_duration_s: float
) -> None:
    """Independently verify exact timeline, channel count, finiteness, and peak <= 10."""
    with validate_audio_file(candidate, max_duration_s=max_duration_s) as audio:
        if (audio.samplerate, audio.channels, audio.frames) != expected:
            raise SeparationError(
                "The separated stem does not preserve the source timeline and channels."
            )
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all() or float(np.max(np.abs(block))) > 10:
                raise SeparationError("The separated stem contains invalid sample values.")


def separate_audio(
    input_path: Path | str,
    output_path: Path | str,
    *,
    prompt: str,
    operation: Literal["extract", "remove"],
    model_size: str = "small",
    quality: str = "high",
    seed: int = 0,
    device: str = "auto",
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Write one validated selected stem; refuse overwrite and silent fallback.

    Routes clear vocal/music prompts to local MIT Mel-Band RoFormer (MLX) when
    available, and routes open-vocabulary SFX (or vocal fallback when RoFormer
    is not installed) to AudioGhost SAM-Audio. Input is decoded WAV up to
    ``WORKER_MAX_DURATION_S`` per pass (longer clips stream via ``audio_edit``).
    """
    from hawavoclean.separation import roformer

    _validate_options(prompt, operation, model_size, quality, seed, device)
    source, target = Path(input_path).resolve(), Path(output_path).absolute()
    if source == target.resolve() or target.exists():
        raise SeparationError(
            "Choose a new output file; sound separation never replaces an existing file."
        )
    with validate_audio_file(source, max_duration_s=WORKER_MAX_DURATION_S) as audio:
        expected = (audio.samplerate, audio.channels, audio.frames)

    target_stem = roformer.classify_vocal_music_prompt(prompt, operation)
    if target_stem is not None and device != "cuda" and roformer.probe_roformer_runtime().available:
        return roformer.separate_roformer(
            source,
            target,
            prompt=prompt,
            operation=operation,
            quality=quality,
            seed=seed,
            progress=progress,
            should_cancel=should_cancel,
            timeout_s=timeout_s,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hawavoclean-separation-", dir=target.parent
    ) as temporary:
        candidate = Path(temporary) / "selected.wav"
        metadata = _run_worker(
            {
                "action": "separate",
                "input": str(source),
                "output": str(candidate),
                "prompt": prompt.strip(),
                "operation": operation,
                "model_size": model_size,
                "quality": quality,
                "seed": seed,
                "device": device,
            },
            timeout_s=timeout_s,
            progress=progress,
            should_cancel=should_cancel,
        )
        _validate_output_stem(candidate, expected, max_duration_s=WORKER_MAX_DURATION_S)
        if should_cancel and should_cancel():
            raise SeparationCancelledError("Sound separation was cancelled.")
        # Exclusive linking prevents a concurrent writer from being overwritten.
        try:
            os.link(candidate, target)
        except FileExistsError as exc:
            raise SeparationError("The separation output already exists.") from exc
    resolved_provider = str(metadata.get("device") or resolve_separation_device(device))
    return {
        **metadata,
        "engine": "audioghost-sam-audio-lite",
        "upstream_commit": AUDIOGHOST_COMMIT,
        "operation": operation,
        "prompt": prompt.strip(),
        "model_size": model_size,
        "quality": quality,
        "seed": seed,
        "provider": resolved_provider,
        "sample_rate": expected[0],
        "channels": expected[1],
        "samples": expected[2],
        "qualification": "experimental_unqualified",
    }
