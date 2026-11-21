#!/usr/bin/env python3
"""Install only the isolated AudioGhost code/runtime, never gated model weights.

Run with --install after reviewing the commands printed by a dry run. Existing
HawaVoClean packages, system packages and services are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

COMMIT = "79653d1a789f4fc73893d10733e2324ed088cecf"
SAM_COMMIT = "bb4c6999d2677c7402360e426afc01ddfad6dce0"
PERCEPTION_COMMIT = "e72b6810b1133e1c879f2cc965d276eb73803f1f"
HELPERS = {
    "backend/workers/__init__.py": "e7e331b43ce901e5f2e7a5e3638930cfa8eefa8a",
    "backend/workers/compat.py": "98b2a5f82d303a45a9a5b0e2eac9ac076565f279",
    "backend/workers/sam_lite.py": "bea1e4a1e18a6a1735fb7f29eb100392b4d3549d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--python", default="python3.11", help="Existing Python 3.11 executable")
    parser.add_argument(
        "--ffmpeg-lib", help="Existing compatible FFmpeg 4–8 shared-library directory"
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    local = repository / ".local-ai"
    root, environment = local / "audioghost-ai", local / "audioghost-env"
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv must be installed first")
    cache = local / "uv-cache"
    library_candidates = (
        [Path(arguments.ffmpeg_lib)]
        if arguments.ffmpeg_lib
        else [
            Path("/opt/homebrew/opt/ffmpeg@7/lib"),
            Path("/opt/homebrew/opt/ffmpeg@8/lib"),
            Path.home() / "miniconda3/lib",
            Path("/opt/anaconda3/lib"),
            Path("/opt/homebrew/opt/ffmpeg/lib"),
        ]
    )
    ffmpeg_library = next(
        (
            path
            for path in library_candidates
            if path.is_absolute()
            and any(
                (path / f"libavutil.{version}.dylib").is_file() for version in (56, 57, 58, 59, 60)
            )
        ),
        None,
    )
    if arguments.ffmpeg_lib and ffmpeg_library is None:
        raise SystemExit("Specified directory does not contain supported FFmpeg shared libraries")
    base = [
        uv,
        "--cache-dir",
        str(cache),
        "pip",
        "install",
        "--python",
        str(environment / "bin/python"),
    ]
    commands = [
        [uv, "--cache-dir", str(cache), "venv", "--python", arguments.python, str(environment)],
        [
            *base,
            "-c",
            str(root / "backend/constraints.txt"),
            "torch==2.9.0",
            "torchvision==0.24.0",
            "torchaudio==2.9.0",
        ],
        [
            *base,
            "-c",
            str(root / "backend/constraints.txt"),
            "-r",
            str(root / "backend/requirements-lean.txt"),
            "soundfile>=0.12,<0.14",
        ],
        [
            *base,
            "--no-deps",
            f"git+https://github.com/facebookresearch/perception_models@{PERCEPTION_COMMIT}",
        ],
        [*base, "--no-deps", f"git+https://github.com/facebookresearch/sam-audio.git@{SAM_COMMIT}"],
    ]
    print(
        json.dumps(
            {
                "engine_commit": COMMIT,
                "commands": commands,
                "model_download": False,
                "license_acceptance": False,
                "ffmpeg_library_path": str(ffmpeg_library) if ffmpeg_library else None,
            },
            indent=2,
        ),
        flush=True,
    )
    if not arguments.install:
        return
    if environment.exists():
        commands.pop(0)
    for relative in [
        *HELPERS,
        "LICENSE",
        "backend/constraints.txt",
        "backend/requirements-lean.txt",
    ]:
        destination = root / relative
        url = f"https://raw.githubusercontent.com/0x0funky/audioghost-ai/{COMMIT}/{relative}"
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read(1024 * 1024)
        if relative in HELPERS:
            digest = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            if digest != HELPERS[relative]:
                raise SystemExit("Downloaded helper does not match its reviewed identity")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise SystemExit(f"Refusing to overwrite modified runtime file: {relative}")
        destination.write_bytes(content)
    for command in commands:
        subprocess.run(command, check=True)
    (local / "audioghost-runtime.json").write_text(
        json.dumps(
            {
                "HAWAVOCLEAN_AUDIOGHOST_PYTHON": str(environment / "bin/python"),
                "HAWAVOCLEAN_AUDIOGHOST_ROOT": str(root),
                "commit": COMMIT,
                "models_downloaded": False,
                **({"ffmpeg_library_path": str(ffmpeg_library)} if ffmpeg_library else {}),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Isolated runtime installed. Model access and weights are a separate explicit setup step."
    )


if __name__ == "__main__":
    main()
