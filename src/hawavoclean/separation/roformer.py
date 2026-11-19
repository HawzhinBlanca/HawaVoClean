"""Offline-only MIT Mel-Band RoFormer vocal/music separation via Apple Silicon MLX.

Only handles unambiguous vocal/speech vs. music/instrumental prompts. Any
open-vocabulary sound effect (e.g. ``"dog barking"``, ``"siren"``, ``"flute"``)
returns ``None`` from :func:`classify_vocal_music_prompt` so the unified
runtime routes it to AudioGhost SAM-Audio instead.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from hawavoclean.separation.runtime import (
    WORKER_MAX_DURATION_S,
    RuntimeStatus,
    SeparationCancelledError,
    SeparationError,
    _run_supervised_worker,
    _validate_output_stem,
    validate_audio_file,
)

ROFORMER_MODEL_FILENAME = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
ROFORMER_CONFIG_FILENAME = "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"
ROFORMER_PYTHON_ENV = "HAWAVOCLEAN_ROFORMER_PYTHON"
ROFORMER_MODEL_DIR_ENV = "HAWAVOCLEAN_ROFORMER_MODEL_DIR"
ROFORMER_ENGINE_ID = "mlx-mel-band-roformer"
ROFORMER_LICENSE = "MIT"

_LOCAL_ROOT = Path(__file__).resolve().parents[3] / ".local-ai"
_ROFORMER_ERRORS = {
    "missing_assets": "MIT Mel-Band RoFormer checkpoint is not cached locally for offline use.",
    "provider_unavailable": "Apple Silicon MLX provider is unavailable for RoFormer separation.",
    "runtime_incompatible": "mlx-audio-separator dependencies are missing or incompatible.",
    "invalid_audio": "RoFormer separation requires valid mono/stereo WAV audio up to two minutes per pass.",
    "out_of_memory": "Not enough unified memory for Mel-Band RoFormer separation.",
}

_VOCAL_CORE = frozenset(
    {
        "speech",
        "voice",
        "voices",
        "vocal",
        "vocals",
        "dialogue",
        "dialog",
        "speaker",
        "speakers",
        "spoken",
        "talking",
        "narration",
        "narrator",
        "singing",
        "singer",
        "singers",
        "voiceover",
    }
)
_VOCAL_MODIFIERS = frozenset(
    {
        "the",
        "a",
        "an",
        "only",
        "just",
        "main",
        "lead",
        "primary",
        "human",
        "clean",
        "clear",
        "all",
        "my",
        "his",
        "her",
        "their",
        "foreground",
        "over",
        "word",
        "words",
        "track",
        "stem",
        "kurdish",
        "sorani",
        "english",
        "arabic",
        "persian",
        "male",
        "female",
        "interview",
        "podcast",
        "host",
        "guest",
    }
)
_MUSIC_CORE = frozenset(
    {
        "music",
        "bgm",
        "instrumental",
        "instrumentals",
        "accompaniment",
        "soundtrack",
        "song",
        "songs",
        "score",
        "melody",
        "musical",
    }
)
_MUSIC_MODIFIERS = frozenset(
    {
        "the",
        "a",
        "an",
        "only",
        "just",
        "all",
        "background",
        "bg",
        "loud",
        "soft",
        "ambient",
        "backing",
        "recorded",
        "bleed",
        "track",
        "bed",
        "stem",
        "non",
        "vocal",
    }
)
_MUSIC_MULTIWORD_PHRASES = frozenset({"backing track", "background track", "music bed"})


def _is_vocal_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False
    token_set = set(tokens)
    return bool(token_set & _VOCAL_CORE) and token_set <= (_VOCAL_CORE | _VOCAL_MODIFIERS)


def _is_music_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False
    phrase = " ".join(tokens)
    token_set = set(tokens)
    if phrase in _MUSIC_MULTIWORD_PHRASES:
        return True
    # "vocal" is only permitted in music tokens as part of "non vocal".
    if "vocal" in token_set and "non" not in token_set:
        return False
    if "non" in token_set and "vocal" not in token_set:
        return False
    has_core = bool(token_set & _MUSIC_CORE) or ("non" in token_set and "vocal" in token_set)
    return has_core and token_set <= (_MUSIC_CORE | _MUSIC_MODIFIERS)


def classify_vocal_music_prompt(
    prompt: str, operation: Literal["extract", "remove"]
) -> Literal["vocals", "instrumental"] | None:
    """Map unambiguous vocal/music prompts to a RoFormer stem, or ``None`` for SFX.

    Examples:
        - ``("speech", "extract")`` -> ``"vocals"``
        - ``("background music", "remove")`` -> ``"vocals"``
        - ``("music", "extract")`` -> ``"instrumental"``
        - ``("vocals", "remove")`` -> ``"instrumental"``
        - ``("dog barking", "remove")`` -> ``None`` (routes to AudioGhost SAM-Audio)
    """
    if not isinstance(prompt, str) or operation not in {"extract", "remove"}:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()
    if not normalized:
        return None
    tokens = normalized.split()

    category: Literal["vocals", "music"] | None = None
    if tokens.count("from") == 1:
        split_idx = tokens.index("from")
        left, right = tokens[:split_idx], tokens[split_idx + 1 :]
        if _is_vocal_tokens(left) and _is_music_tokens(right):
            category = "vocals"
        elif _is_music_tokens(left) and _is_vocal_tokens(right):
            category = "music"
    elif _is_vocal_tokens(tokens):
        category = "vocals"
    elif _is_music_tokens(tokens):
        category = "music"

    if category is None:
        return None
    if category == "vocals":
        return "vocals" if operation == "extract" else "instrumental"
    return "instrumental" if operation == "extract" else "vocals"


def _roformer_model_dir() -> Path:
    configured = os.environ.get(ROFORMER_MODEL_DIR_ENV) or os.environ.get(
        "AUDIO_SEPARATOR_MODEL_DIR"
    )
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise SeparationError("Configured RoFormer model directory must be an absolute path.")
        return path
    return Path.home() / ".cache" / "mlx-audio-separator" / "models"


def _roformer_python() -> Path:
    configured = os.environ.get(ROFORMER_PYTHON_ENV)
    if configured is not None:
        python = Path(configured)
        if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
            raise SeparationError("Configured RoFormer Python runtime is invalid.")
        return python
    bench_python = _LOCAL_ROOT / "bench-env" / "bin" / "python"
    if bench_python.is_file() and os.access(bench_python, os.X_OK):
        return bench_python
    if importlib.util.find_spec("mlx_audio_separator") is not None:
        return Path(sys.executable)
    raise SeparationError("MLX Mel-Band RoFormer runtime is not installed.")


def _cached_roformer_files(model_dir: Path) -> tuple[Path, Path]:
    ckpt = model_dir / ROFORMER_MODEL_FILENAME
    cfg = model_dir / ROFORMER_CONFIG_FILENAME
    if (
        not ckpt.is_file()
        or not cfg.is_file()
        or ckpt.stat().st_size == 0
        or cfg.stat().st_size == 0
    ):
        raise SeparationError(
            "MIT Mel-Band RoFormer checkpoint is not cached locally for offline use."
        )
    return ckpt, cfg


def _roformer_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        MLX_AUDIO_SEPARATOR_NO_LEGACY_CACHE="1",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HAWAVOCLEAN_SEPARATION_PARENT_PID=str(os.getpid()),
    )
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        environment.pop(key, None)
    return environment


def _run_roformer_worker(
    request: dict[str, Any],
    *,
    timeout_s: float,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    python = _roformer_python()
    model_dir = _roformer_model_dir()
    _cached_roformer_files(model_dir)
    return _run_supervised_worker(
        python,
        {
            **request,
            "engine": "roformer",
            "model_dir": str(model_dir),
            "model_filename": ROFORMER_MODEL_FILENAME,
            "config_filename": ROFORMER_CONFIG_FILENAME,
        },
        env=_roformer_environment(),
        timeout_s=timeout_s,
        progress=progress,
        should_cancel=should_cancel,
        error_reasons=_ROFORMER_ERRORS,
    )


def probe_roformer_runtime() -> RuntimeStatus:
    """Check RoFormer Python runtime and cached MIT checkpoint without loading weights."""
    try:
        metadata = _run_roformer_worker({"action": "probe"}, timeout_s=20.0)
        if metadata.get("ready") is not True:
            raise SeparationError(
                "MIT Mel-Band RoFormer checkpoint is not ready for offline inference."
            )
        provider = str(metadata.get("provider") or metadata.get("device") or "mlx")
        return RuntimeStatus(
            True,
            "Vocal/music separation ready via MLX RoFormer; "
            "install SAM-Audio weights for open-prompt SFX.",
            (provider,),
        )
    except SeparationError as exc:
        return RuntimeStatus(False, str(exc))
    except OSError:
        return RuntimeStatus(False, "The MLX RoFormer separation runtime could not be started.")


def separate_roformer(
    input_path: Path | str,
    output_path: Path | str,
    *,
    prompt: str,
    operation: Literal["extract", "remove"],
    quality: str = "high",
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run offline MLX Mel-Band RoFormer separation and validate output contract."""
    target_stem = classify_vocal_music_prompt(prompt, operation)
    if target_stem is None:
        raise SeparationError("Prompt is outside vocal/music Mel-Band RoFormer scope.")
    if quality not in {"fast", "balanced", "high"}:
        raise SeparationError("Unsupported separation quality.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise SeparationError("The separation seed must be an integer from 0 to 2147483647.")

    source, target = Path(input_path).resolve(), Path(output_path).absolute()
    if source == target.resolve() or target.exists():
        raise SeparationError(
            "Choose a new output file; sound separation never replaces an existing file."
        )
    with validate_audio_file(source, max_duration_s=WORKER_MAX_DURATION_S) as audio:
        expected = (audio.samplerate, audio.channels, audio.frames)

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hawavoclean-roformer-", dir=target.parent
    ) as temporary:
        candidate = Path(temporary) / "selected.wav"
        metadata = _run_roformer_worker(
            {
                "action": "separate",
                "input": str(source),
                "output": str(candidate),
                "target_stem": target_stem,
                "quality": quality,
                "seed": seed,
            },
            timeout_s=timeout_s,
            progress=progress,
            should_cancel=should_cancel,
        )
        _validate_output_stem(candidate, expected, max_duration_s=WORKER_MAX_DURATION_S)
        if should_cancel and should_cancel():
            raise SeparationCancelledError("Sound separation was cancelled.")
        try:
            os.link(candidate, target)
        except FileExistsError as exc:
            raise SeparationError("The separation output already exists.") from exc

    return {
        **metadata,
        "engine": ROFORMER_ENGINE_ID,
        "license": ROFORMER_LICENSE,
        "operation": operation,
        "prompt": prompt.strip(),
        "target_stem": target_stem,
        "quality": quality,
        "seed": seed,
        "provider": str(metadata.get("provider") or metadata.get("device") or "mlx"),
        "sample_rate": expected[0],
        "channels": expected[1],
        "samples": expected[2],
        "qualification": "experimental_unqualified",
    }
